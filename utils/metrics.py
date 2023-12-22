import torch
from torch import nn
import cv2
import numpy as np
import lpips
from pathlib import Path
import os
import subprocess
import json

from utils.utils import PROJECT_ROOT


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Args:
        pred (ndarray): Images with range [0, 255].
        gt (ndarray): Images with range [0, 255].

    Returns:
        float: psnr result.
    """

    assert pred.shape == gt.shape, f'Image shapes are different: {pred.shape}, {gt.shape}.'

    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def compute_ssim(pred: np.ndarray, gt: np.ndarray, crop_border: int = 0) -> float:
    """Calculate SSIM (structural similarity).

    Ref: Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        pred (ndarray): Images with range [0, 255].
        gt (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation. Default: 0.

    Returns:
        float: ssim result.
    """
    assert pred.shape == gt.shape, f'Image shapes are different: {pred.shape}, {gt.shape}.'
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)

    if crop_border != 0:
        pred = pred[crop_border:-crop_border, crop_border:-crop_border, ...]
        gt = gt[crop_border:-crop_border, crop_border:-crop_border, ...]

    ssims = []
    for i in range(pred.shape[2]):
        ssims.append(_ssim(pred[..., i], gt[..., i]))
    return np.array(ssims).mean()


def _ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.

    Args:
        pred (ndarray): Images with range [0, 255] with order 'HWC'.
        gt (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: ssim result.
    """

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(pred, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(gt, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(pred ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(gt ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(pred * gt, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def compute_lpips(pred: np.ndarray, gt: np.ndarray, lpips_net: nn.Module, device: torch.device) -> float:
    """Compute LPIPS (Learned Perceptual Image Patch Similarity).
    Ref: https://richzhang.github.io/PerceptualSimilarity/

    Args:
        pred (ndarray): Images with range [0, 255].
        gt (ndarray): Images with range [0, 255].
        lpips_net (nn.Module): LPIPS network.
        device (torch.device): Device to use.
    Returns:
        float: LPIPS result.
    """

    assert pred.shape == gt.shape, f'Image shapes are different: {pred.shape}, {gt.shape}.'

    # Convert to tensors and normalize to [-1, 1]
    pred = lpips.im2tensor(pred)
    gt = lpips.im2tensor(gt)

    lpips_val = lpips_net(pred.to(device), gt.to(device))

    return lpips_val.squeeze().cpu().detach().numpy()


def compute_vmaf(restored_video_path: Path, gt_video_path: Path, width: int, height: int) -> float:
    """
    Compute VMAF (Video Multi-Method Assessment Fusion). Generates YUV files from the input videos and then delete them
    after the computation.

    Args:
        restored_video_path (Path): Path to the restored video.
        gt_video_path (Path): Path to the ground truth video.
        width (int): Width of the restored video.
        height (int): Height of the restored video.
    """
    vmaf_base_path = PROJECT_ROOT / "utils" / "vmaf"
    vmaf_model = vmaf_base_path / "model" / "vmaf_v0.6.1.json"

    # Get GT video info for cropping
    gt_video = cv2.VideoCapture(str(gt_video_path))
    gt_width = int(gt_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    gt_height = int(gt_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    left = (gt_width - width) // 2
    top = (gt_height - height) // 2

    # Convert videos to YUV
    yuv_restored_video_path = restored_video_path.parent / f"{restored_video_path.stem}.yuv"
    yuv_gt_video_path = gt_video_path.parent / f"{gt_video_path.stem}.yuv"
    os.system(f'ffmpeg -y -loglevel error -i {restored_video_path} {yuv_restored_video_path}')
    os.system(f'ffmpeg -y -loglevel error -i {gt_video_path} -vf "crop={width}:{height}:{left}:{top}"'
              f' {yuv_gt_video_path}')

    # Compute VMAF
    output = subprocess.check_output(f"PYTHONPATH=python ./python/vmaf/script/run_vmaf.py yuv420p {width} {height}"
                                     f" {yuv_gt_video_path} {yuv_restored_video_path} --out-fmt json"
                                     f" --model {str(vmaf_model)}", cwd=str(vmaf_base_path), shell=True)
    output = output.decode("utf-8").strip("\n")
    output = "\n".join(output.split("\n"))
    output_json = json.loads(output)
    vmaf_score = output_json["aggregate"]["VMAF_score"]

    # Delete YUV files
    os.remove(yuv_restored_video_path)
    os.remove(yuv_gt_video_path)

    return vmaf_score
