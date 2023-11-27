import torch
from typing import Union, List


def preprocess(imgs: Union[List[torch.Tensor], torch.Tensor], mode: str = "crop", patch_size: int = 768)\
        -> Union[List[torch.Tensor], torch.Tensor]:
    """Preprocesses a tensor of images or list of tensors of images.

    Args:
        imgs (Union[List[torch.Tensor], torch.Tensor]): List of tensors of images or a single tensor of images.
        mode (str, optional): Preprocess mode. Values can be in ["crop", "resize"].
        patch_size (int, optional): Maximum patch size

    Returns:
        Union[List[torch.Tensor], torch.Tensor]: Preprocessed images.
    """
    if isinstance(imgs, list):
        return [preprocess(img, mode=mode, patch_size=patch_size) for img in imgs]
    elif isinstance(imgs, torch.Tensor):
        if mode == "crop":
            return crop(imgs, patch_size=patch_size)
        elif mode == "resize":
            return resize(imgs, patch_size=patch_size)
        else:
            raise ValueError(f"Unknown preprocess mode: {mode}")
    else:
        raise TypeError(f"Unknown type for imgs: {type(imgs)}")


def crop(img: torch.Tensor, patch_size: int = 768) -> torch.Tensor:
    """Center crops a tensor of images to patch_size.

    Args:
        img (torch.Tensor): Tensor of images.
        patch_size (int, optional): Maximum patch size

    Returns:
        torch.Tensor: Cropped images.
    """
    _, _, h, w = img.shape
    if h > patch_size or w > patch_size:
        h_start = max((h - patch_size) // 2, 0)
        w_start = max((w - patch_size) // 2, 0)
        return img[:, :, h_start:h_start + patch_size, w_start:w_start + patch_size]
    else:
        return img


def resize(img: torch.Tensor, patch_size: int = 768) -> torch.Tensor:
    """Resizes a tensor of images so that the biggest dimension is equal to patch_size while keeping the aspect ratio.

    Args:
        img (torch.Tensor): Tensor of images.
        patch_size (int, optional): Maximum patch size

    Returns:
        torch.Tensor: Resized images.
    """
    _, _, h, w = img.shape
    if h > patch_size or w > patch_size:
        if h > w:
            new_h = patch_size
            new_w = int(w * patch_size / h)
        else:
            new_w = patch_size
            new_h = int(h * patch_size / w)
        return torch.nn.functional.interpolate(img, size=(new_h, new_w), mode="bilinear")
    else:
        return img
