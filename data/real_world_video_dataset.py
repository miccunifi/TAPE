import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import json
import cv2

from utils.utils import preprocess


class RealWorldVideoDataset(Dataset):
    """
    Dataset for real world videos (i.e. no ground truth). Each item is given by a window of num_input_frames input
    frames (to be restored) and a window of num_reference_frames reference frames.

    Args:
        input_folder (Path): Path to the folder containing the input frames
        num_input_frames (int): Number of input frames T of the input window
        num_reference_frames (int): Number of reference frames D
        references_file_path (Path): Path to the file containing the references for each frame
        preprocess_mode (str): Preprocessing mode for when the size of the input frames is greater than the patch size.
                               Supported modes: ["crop", "resize"]
        patch_size (int): Maximum patch size
        frame_format (str): Format of the input frames
    Returns:
        dict with keys:
            "imgs_lq" (torch.Tensor): Input frames
            "imgs_ref" (torch.Tensor): Reference frames
            "img_name" (str): Name of the center input frame
    """

    def __init__(self,
                 input_folder: Path,
                 num_input_frames: int = 5,
                 num_reference_frames: int = 5,
                 references_file_path: Path = "references.json",
                 preprocess_mode: str = "crop",
                 patch_size: int = 768,
                 frame_format: str = "jpg"):
        self.input_folder = input_folder
        self.num_input_frames = num_input_frames
        self.num_reference_frames = num_reference_frames
        self.preprocess_mode = preprocess_mode
        self.patch_size = patch_size

        self.img_paths = sorted(list(input_folder.glob(f"*.{frame_format}")))

        # Load references
        with open(references_file_path, 'r') as f:
            self.references = json.load(f)

    def __getitem__(self, idx):
        img_name = self.img_paths[idx].name

        half_input_window_size = self.num_input_frames // 2
        idxs_imgs_lq = np.arange(idx - half_input_window_size, idx + half_input_window_size + 1)
        idxs_imgs_lq = list(idxs_imgs_lq[(idxs_imgs_lq >= 0) & (idxs_imgs_lq <= len(self.img_paths) - 1)])
        imgs_lq = []
        for img_idx in idxs_imgs_lq:
            img = cv2.imread(str(self.img_paths[img_idx]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.
            img_t = ToTensor()(img)
            imgs_lq.append(img_t)

        # Pad with black frames if the window is not complete
        if len(imgs_lq) < self.num_input_frames:
            black_frame = torch.zeros_like(imgs_lq[0])
            missing_frames_left = half_input_window_size - (idx - 0)
            for _ in range(missing_frames_left):
                imgs_lq.insert(0, black_frame)
            missing_frames_right = half_input_window_size - (len(self.img_paths) - 1 - idx)
            for _ in range(missing_frames_right):
                imgs_lq.append(black_frame)
        imgs_lq = torch.stack(imgs_lq)

        imgs_ref = []
        for ref_name in self.references[img_name]:
            img = cv2.imread(str(self.input_folder / ref_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.
            img_t = ToTensor()(img)
            imgs_ref.append(img_t)
        imgs_ref = torch.stack(imgs_ref)

        if self.preprocess_mode != "none":
            imgs_lq, imgs_ref = preprocess([imgs_lq, imgs_ref], mode=self.preprocess_mode, patch_size=self.patch_size)

        return {"imgs_lq": imgs_lq,
                "imgs_ref": imgs_ref,
                "img_name": img_name}

    def __len__(self):
        return len(self.img_paths)
