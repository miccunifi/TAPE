import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
from typing import List

LAYER_NAMES = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
]


class VGGFeatureExtractor(nn.Module):

    def __init__(self, layer_name_list: List[str] = None, use_input_norm: bool = True, use_range_norm: bool = False,
                 requires_grad: bool = False):
        """VGG-19 network for feature extraction for perceptual loss.

        Args:
            layer_name_list Llist[str]): Forward function returns the corresponding features according to the
             layer_name_list. Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
            use_input_norm (bool): If True, x: [0, 1] --> (x - mean) / std. Default: True
            use_range_norm (bool): If True, norm images with range [-1, 1] to [0, 1]. Default: False.
            requires_grad (bool): If true, the parameters of VGG network will be optimized. Default: False.
        """
        super(VGGFeatureExtractor, self).__init__()
        if layer_name_list is None:
            self.layer_name_list = ['conv5_4', 'relu4_4', 'relu3_4', 'relu2_2']
        else:
            self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = use_range_norm

        # only borrow layers that will be used to avoid unused params
        max_idx = 0
        for v in layer_name_list:
            idx = LAYER_NAMES.index(v)
            if idx > max_idx:
                max_idx = idx

        vgg_net = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        features = vgg_net.features[:max_idx + 1]
        net = OrderedDict(zip(LAYER_NAMES, features))
        self.vgg_net = nn.Sequential(net)

        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict:
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()

        return output
