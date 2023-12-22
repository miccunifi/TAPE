import torch
import torch.nn as nn

from models.vgg_feature_extractor import VGGFeatureExtractor


def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target) ** 2 + eps).mean()


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (one variant of Robust L1Loss, a differentiable variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution".

    Args:
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
    """

    def __init__(self, eps: float = 1e-12, loss_weight: float = 1.0):
        super(CharbonnierLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        return self.loss_weight * charbonnier_loss(pred, target, eps=self.eps)


class PerceptualLoss(nn.Module):
    """
    VGG 19 Perceptual loss

    Args:
        layer_weights (dict): Layer weights for perceptual loss.
        use_input_norm (bool): If True, x: [0, 1] --> (x - mean) / std. Default: True
        use_range_norm (bool): If True, norm images with range [-1, 1] to [0, 1]. Default: False.
        criterion (str): Criterion type. Default: 'l2'.
        loss_weight (float): Loss weight for perceptual loss. Default: 1.0.
    """

    def __init__(self, layer_weights: dict, use_input_norm: bool = True, use_range_norm: bool = False,
                 criterion: str = 'l2', loss_weight: float = 1.0):
        super(PerceptualLoss, self).__init__()
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(layer_name_list=list(layer_weights.keys()),
                                       use_input_norm=use_input_norm,
                                       use_range_norm=use_range_norm)
        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError(f'{criterion} criterion is not supported.')
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward function.
        Args:
            pred (Tensor): Input tensor with shape (n, c, h, w).
            target (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        pred_feat = self.vgg(pred)
        target_feat = self.vgg(target.detach())

        loss = 0.0
        for i in pred_feat.keys():
            loss += self.criterion(pred_feat[i], target_feat[i]) * self.layer_weights[i]
        loss *= self.loss_weight
        return loss
