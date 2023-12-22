import torch
from typing import Union, List
from pathlib import Path
import cv2
import numpy as np
from pytorch_lightning.loggers import CometLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()


def preprocess(imgs: Union[List[torch.Tensor], torch.Tensor], mode: str = "crop", patch_size: int = 768, **kwargs)\
        -> Union[List[torch.Tensor], torch.Tensor]:
    """Preprocesses a tensor of images or list of tensors of images.

    Args:
        imgs (Union[List[torch.Tensor], torch.Tensor]): List of tensors of images or a single tensor of images.
        mode (str, optional): Preprocess mode. Values can be in ["crop", "resize"].
        patch_size (int, optional): Maximum patch size
        **kwargs: Additional arguments for the preprocess mode.

    Returns:
        Union[List[torch.Tensor], torch.Tensor]: Preprocessed images.
    """
    if isinstance(imgs, list):
        return [preprocess(img, mode=mode, patch_size=patch_size, **kwargs) for img in imgs]
    elif isinstance(imgs, torch.Tensor):
        if mode == "crop":
            return crop(imgs, patch_size=patch_size, crop_mode=kwargs.get("crop_mode", "center"))
        elif mode == "resize":
            return resize(imgs, patch_size=patch_size)
        else:
            raise ValueError(f"Unknown preprocess mode: {mode}")
    else:
        raise TypeError(f"Unknown type for imgs: {type(imgs)}")


def crop(img: torch.Tensor, patch_size: int = 768, crop_mode: str = "center") -> torch.Tensor:
    """Center crops a tensor of images to patch_size.

    Args:
        img (torch.Tensor): Tensor of images.
        patch_size (int, optional): Maximum patch size
        crop_mode (str, optional): Crop mode. Values can be in ["center", "random"].

    Returns:
        torch.Tensor: Cropped images.
    """
    _, _, h, w = img.shape
    if h > patch_size or w > patch_size:
        if crop_mode == "center":
            h_start = max((h - patch_size) // 2, 0)
            w_start = max((w - patch_size) // 2, 0)
            return img[:, :, h_start:h_start + patch_size, w_start:w_start + patch_size]
        elif crop_mode == "random":
            h_start = np.random.randint(0, h - patch_size)
            w_start = np.random.randint(0, w - patch_size)
            return img[:, :, h_start:h_start + patch_size, w_start:w_start + patch_size]
        else:
            raise ValueError(f"Unknown crop mode: {crop_mode}")
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


def imfrombytes(content: bytes, flag: str = 'color', float32: bool = False) -> np.ndarray:
    """Read an image from bytes.
    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.
    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def init_logger(experiment_name: str, api_key: str = None, project_name: str = None, online: bool = True) -> CometLogger:
    """
    Initializes a Comet-ML logger.

    Args:
        experiment_name (str): Experiment name.
        api_key (str, optional): Comet ML API key. Defaults to None.
        project_name (str, optional): Comet ML project name. Defaults to None.
        online (bool, optional): If True, the logger will be online. Defaults to True.
    """
    if online:
        comet_logger = CometLogger(api_key=api_key,
                                   project_name=project_name,
                                   experiment_name=experiment_name)
    else:
        comet_logger = CometLogger(save_dir="comet",
                                   project_name=project_name,
                                   experiment_name=experiment_name)
    return comet_logger
