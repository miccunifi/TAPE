import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from einops import rearrange

from utils.utils_models import (compute_mask_2D, window_partition_2D, window_reverse_2D, get_window_size, DropPath, Mlp,
                                trunc_normal_)


class AttentionPooling1d(nn.Module):
    """
    Inspired by https://amaarora.github.io/posts/2023-03-11_Understanding_CLIP_part_2.html and
    https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L58

    Args:
        dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        sequence_length (int): Length of the sequence of transformer tokens.
    """
    def __init__(self, dim: int, num_heads: int, sequence_length: int):
        super().__init__()
        self.sequence_length = sequence_length
        self.pos_embedding = nn.Parameter(torch.randn(sequence_length, dim) / dim ** 0.5)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B*T, M, N, C)

        Returns:
            x (torch.Tensor): (B*T, N, C)
        """
        avg = x.mean(dim=1, keepdim=True)  # (B*T, 1, N, C)
        x = torch.cat([avg, x], dim=1)  # (B*T, M+1, N, C)
        x = x + self.pos_embedding[None, None, :, :]  # (B*T, M+1, N, C)
        x = rearrange(x, 'b m n c -> (m n) b c')  # ((M+1)*N, B*T, C)

        x, _ = F.multi_head_attention_forward(
            query=x[:self.sequence_length], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        x = rearrange(x, 'n b c -> b n c')  # (B*T, N, C)
        return x


class MultiReferenceWindowAttention(nn.Module):
    """ Multi-Reference-(Shifted)Window-Multi-head Cross Attention (MR-(S)W-MCA) module with relative position bias.
    It supports both shifted and non-shifted window. The query is the restored features, while the key and values
    are the reference features.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim: int,
                 window_size: Tuple[int],
                 num_heads: int,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.act = nn.GELU()

        self.dim_reduction = AttentionPooling1d(dim=dim, num_heads=num_heads, sequence_length=window_size[0] * window_size[1])

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, x_kv: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, T, N, C)
            x_kv: input features with shape of (num_windows*B, M, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x_kv = x if x_kv is None else x_kv
        B_, T, N, C = x.shape
        _, M, _, _ = x_kv.shape

        q = self.q(x).reshape(B_, T, N, 1, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        kv = self.kv(x_kv).reshape(B_, M, N, 2, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = q[0], kv[0], kv[1]  # B_, T (M), nH, N, C/nH

        q = q.unsqueeze(2)      # B_, T, 1, nH, N, C/nH
        k = k.unsqueeze(1)      # B_, 1, M, nH, N, C/nH
        v = v.unsqueeze(1)      # B_, 1, M, nH, N, C/nH

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))    # B_, T, M, nH, N, N

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias[None, None, None, ...]

        if mask is not None:
            nW = mask.shape[0]
            attn = rearrange(attn, '(b nW) t m nH n1 n2 -> b t m nW nH n1 n2', nW=nW)
            mask = mask.unsqueeze(1)[None, None, None, ...]
            attn += mask
            attn = rearrange(attn, 'b t m nW nH n1 n2 -> (b nW) t m nH n1 n2')
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, T, M, N, C)

        x = rearrange(x, 'b t m n c -> (b t) m n c')
        x = self.dim_reduction(x)
        x = rearrange(x, '(b t) n c -> b t n c', t=T)

        x = self.act(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MRSFFBlock(nn.Module):
    """ A Multi-Reference Spatial Feature Fusion (MRSFF) block presented in the paper https://arxiv.org/abs/2310.14926.
    It combines the restored and reference features. Based on the Swin Transformer 2D block implementation.
    
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: Tuple[int] = (7, 7),
                 shift_size: Tuple[int] = (0, 0),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.attn = MultiReferenceWindowAttention(
                        dim,
                        window_size=self.window_size,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, kv: torch.Tensor, mask_matrix: torch.Tensor) -> torch.Tensor:
        """ Forward function.

        Args:
            x (torch.Tensor): Input feature, tensor size (B, T, H, W, C).
            kv (torch.Tensor): Reference feature, tensor size (B, M, H, W, C).
            mask_matrix (torch.Tensor): Attention mask for cyclic shift.
        """
        shortcut = x
        x = self.forward_part1(x, kv, mask_matrix)
        x = shortcut + self.drop_path(x)
        x = x + self.forward_part2(x)
        return x

    def forward_part1(self, x: torch.Tensor, kv: torch.Tensor, mask_matrix: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x = rearrange(x, 'b t h w c -> (b t) h w c', b=B, t=T)

        _, M, _, _, _ = kv.shape
        kv = rearrange(kv, 'b m h w c -> (b m) h w c', b=B, m=M)

        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        x = self.norm_q(x)
        kv = self.norm_kv(kv)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        kv = F.pad(kv, (0, 0, pad_l, pad_r, pad_t, pad_b))

        _, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            shifted_kv = torch.roll(kv, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            shifted_kv = kv
            attn_mask = None

        # partition windows
        x_windows = window_partition_2D(shifted_x, window_size)  # B*T*nW, Wh*Ww, C
        kv_windows = window_partition_2D(shifted_kv, window_size)  # B*M*nW, Wh*Ww, C

        _, N, C = x_windows.shape
        x_windows = x_windows.reshape(-1, T, N, C)
        kv_windows = kv_windows.reshape(-1, M, N, C)

        # MR-W-MCA/MR-SW-MCA
        attn_windows = self.attn(x_windows, kv_windows, mask=attn_mask)  # B*T*nW, Wd*Wh*Ww, C

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse_2D(attn_windows, window_size, B * T, Hp, Wp)  # B*T H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        x = rearrange(x, '(b t) h w c -> b t h w c', b=B, t=T)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x: torch.Tensor) -> torch.Tensor:
        # FFN
        return self.drop_path(self.mlp(self.norm2(x)))


class MRSFFLayer(nn.Module):
    """ A Multi-Reference Spatial Feature Fusion (MRSFF) layer.

        Args:
            dim (int): Number of input channels.
            depth (int): Number of blocks.
            num_heads (int): Number of attention heads.
            window_size (tuple[int]): Local window size.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        """

    def __init__(self,
                 dim: int,
                 depth: int,
                 num_heads: int,
                 window_size: Tuple[int] = (7, 7),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm):

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            MRSFFBlock(dim=dim,
                       num_heads=num_heads,
                       window_size=window_size,
                       shift_size=(0, 0) if (i % 2 == 0) else self.shift_size,
                       mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias,
                       qk_scale=qk_scale,
                       drop=drop,
                       attn_drop=attn_drop,
                       drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                       norm_layer=norm_layer)
            for i in range(depth)])

        self.last_conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """ Forward function.
            Args:
                x (torch.Tensor): Input feature, tensor size (B, C, T, H, W).
                kv (torch.Tensor): Reference feature, tensor size (B, C, M, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, T, H, W = x.shape
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        x = rearrange(x, 'b c t h w -> b t h w c')
        kv = rearrange(kv, 'b c m h w -> b m h w c')
        residual = x.clone()

        Hp = int(np.ceil(H / window_size[0])) * window_size[0]
        Wp = int(np.ceil(W / window_size[1])) * window_size[1]
        attn_mask = compute_mask_2D(Hp, Wp, window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, kv, attn_mask)

        x = rearrange(x, 'b t h w c -> b t c h w').reshape(B * T, C, H, W)
        x = self.last_conv(x)
        x = rearrange(x.reshape(B, T, C, H, W), 'b t c h w -> b t h w c')
        x = x + residual
        x = rearrange(x, 'b t h w c -> b c t h w')
        return x
