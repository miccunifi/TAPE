import torch
import torch.nn as nn
from typing import Tuple, List
from einops import rearrange

from utils.utils_models import trunc_normal_
from models.swin_feature_extractor import SwinFeatureExtractor
from models.swin_transformer_3d import SwinTransformer3DLayer
from models.mrsff import MRSFFLayer


class SwinUNet(nn.Module):
    """
    Swin-UNet network for analog video restoration presented in the paper https://arxiv.org/abs/2310.14926.
    The network is composed of a Swin Transformer encoder and a Swin Transformer decoder with MRSFF blocks.
    The network takes as input a window of T input frames and a window of D reference frames. The output is the restored
    window of input frames.

    Args:
        in_chans (int): Number of input channels. Default: 3
        embed_dim (int): Dimension of the token embeddings. Default: 96
        depths (List[int]): Depths of the Swin Transformer layers. Default: None. If None, use [2, 2, 6, 2].
        num_heads (List[int]): Number of attention heads for each layer. Default: None. If None, use [8, 8, 8, 8].
        window_size (Tuple[int]): Window size for each layer. Default: (2, 8, 8).
        mlp_ratio (float): Ratio of the mlp hidden dimension to the embedding dimension. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        use_checkpoint (bool): If True, use gradient checkpointing to save memory. Default: False.
    """
    def __init__(self,
                 in_chans: int = 3,
                 embed_dim: int = 96,
                 depths: List[int] = None,
                 num_heads: List[int] = None,
                 window_size: Tuple[int] = (2, 8, 8),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.2,
                 norm_layer: nn.Module = nn.LayerNorm,
                 use_checkpoint: bool = False):

        super(SwinUNet, self).__init__()
        if num_heads is None:
            num_heads = [8, 8, 8, 8]
        if depths is None:
            depths = [2, 2, 6, 2]
        self.embed_dim = embed_dim

        self.conv_input = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1)
        self.conv_output = nn.Conv2d(embed_dim // 2, in_chans, kernel_size=3, stride=1, padding=1)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.num_layers = len(depths)

        self.encoding_layers = nn.ModuleList()
        for i_layer in range(0, self.num_layers - 1):
            layer = SwinTransformer3DLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                sampling_operation="downsample",
                use_checkpoint=use_checkpoint)
            self.encoding_layers.append(layer)

        self.decoding_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SwinTransformer3DLayer(
                dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                depth=depths[self.num_layers - 1 - i_layer],
                num_heads=num_heads[self.num_layers - 1 - i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:self.num_layers - 1 - i_layer]):sum(depths[:self.num_layers - 1 - i_layer + 1])],
                norm_layer=norm_layer,
                sampling_operation="upsample",
                use_checkpoint=use_checkpoint)
            self.decoding_layers.append(layer)

        self.mrsff_layers = nn.ModuleList()
        for i_layer in range(0, self.num_layers - 1):
            layer = MRSFFLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[self.num_layers - 1 - i_layer],
                num_heads=num_heads[self.num_layers - 1 - i_layer],
                window_size=window_size[1:],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:self.num_layers - 1 - i_layer]):sum(depths[:self.num_layers - 1 - i_layer + 1])],
                norm_layer=norm_layer)
            self.mrsff_layers.append(layer)

        ref_feature_extractor_layers = ["1", "3", "5"]
        self.ref_feature_extractor = SwinFeatureExtractor(layer_name_list=ref_feature_extractor_layers,
                                                          use_input_norm=True, use_range_norm=False, requires_grad=False)
        self.ref_feature_extractor_conv = nn.ModuleList()
        for i, layer in enumerate(ref_feature_extractor_layers):
            self.ref_feature_extractor_conv.append(nn.Sequential(nn.Conv2d(embed_dim * 2 ** i, embed_dim * 2 ** i * 4, 3, 1, 1),
                                                  nn.PixelShuffle(2)))
        self.apply(self._init_weights)

    def forward_encoding(self, imgs_lq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C, H, W = imgs_lq.shape
        restored = rearrange(imgs_lq, 'b t c h w -> (b t) c h w')
        restored = self.conv_input(restored)
        restored = rearrange(restored, '(b t) c h w -> b t c h w', t=T)
        restored = rearrange(restored, 'b t c h w -> b c t h w')

        # UNet encoder
        residual = [restored]
        for layer in self.encoding_layers:
            restored = layer(restored.contiguous())
            residual.append(restored)

        return restored, residual

    def forward_decoding(self, restored: torch.Tensor, imgs_ref: torch.Tensor, residual: List[torch.Tensor]) -> torch.Tensor:
        # Extract features from reference frames
        _, M, _, _, _ = imgs_ref.shape
        imgs_ref = rearrange(imgs_ref, 'b m c h w -> (b m) c h w')
        with torch.no_grad():
            feat_ref = list(self.ref_feature_extractor(imgs_ref).values())
        for i in range(len(feat_ref)):
            feat_ref[i] = self.ref_feature_extractor_conv[i](feat_ref[i])
            feat_ref[i] = rearrange(feat_ref[i], '(b m) c h w -> b m c h w', m=M)
            feat_ref[i] = rearrange(feat_ref[i], 'b m c h w -> b c m h w')

        # UNet decoder
        B, _, T, _, _ = restored.shape
        for i, layer in enumerate(self.decoding_layers):
            if i == 0:
                restored = layer(restored)  # Bottleneck layer
            else:
                restored += residual[-1 - i]    # Encoder-decoder skip connection
                restored_ref = self.mrsff_layers[-i](restored, feat_ref[-i])    # Combine restored and reference features
                restored += restored_ref    # MRSFF skip connection
                restored = layer(restored)  # Decoder layer

        restored = rearrange(restored, 'b c t h w -> b t c h w')
        B, T, C, H, W = restored.shape
        restored = self.conv_output(restored.reshape(B * T, C, H, W))
        restored = restored.reshape(B, T, -1, H, W)
        return restored

    def forward(self, imgs_lq: torch.Tensor, imgs_ref: torch.Tensor) -> torch.Tensor:
        """
        Forward function.

        Args:
            imgs_lq (Tensor): Input frames with shape (b, t, c, h, w).
            imgs_ref (Tensor): Reference frames with shape (b, d, c, h, w).
        """
        out = imgs_lq.clone()
        restored, residual = self.forward_encoding(imgs_lq)
        restored = self.forward_decoding(restored, imgs_ref, residual)
        return out + restored

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
