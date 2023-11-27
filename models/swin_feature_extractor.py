import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights
from einops import rearrange
from typing import List


class SwinFeatureExtractor(nn.Module):

    def __init__(self, layer_name_list: List[str] = None, use_input_norm: bool = True, use_range_norm: bool = False,
                 requires_grad: bool = False):
        """Swin Transformer network for feature extraction.

        Args:
            layer_name_list (List[str]): Forward function returns the corresponding
                features according to the layer_name_list.
            use_input_norm (bool): If True, x: [0, 1] --> (x - mean) / std. Default: True
            use_range_norm (bool): If True, norm images with range [-1, 1] to [0, 1]. Default: False.
            requires_grad (bool): If true, the parameters of the feature extractor network will be
                optimized. Default: False.
        """
        super(SwinFeatureExtractor, self).__init__()
        if not layer_name_list:
            self.layer_name_list = ["1", "3", "5"]
        else:
            self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = use_range_norm

        self.swin_net = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).features

        max_idx = 0
        for i, layer in enumerate(self.swin_net._modules.keys()):
            if layer in self.layer_name_list:
                max_idx = i
        self.swin_net = self.swin_net[:max_idx + 1]

        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        if not requires_grad:
            self.swin_net.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict:
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            dict[str, Tensor]: Output features.
        """
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}
        for key, layer in self.swin_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = rearrange(x.clone(), 'b h w c -> b c h w')

        return output