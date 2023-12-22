import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from torchvision.transforms import ToTensor
import lmdb
from itertools import groupby
import clip
from skimage.filters import threshold_otsu
import json
from PIL import Image

from utils.utils import preprocess, imfrombytes, device
from utils.prompts import prompts


class SyntheticVideoDataset(Dataset):
    """
    Dataset for synthetic videos in LMDB format. Each item is given by a window of num_input_frames input (to be
    restored) and ground-truth frames and a window of num_reference_frames reference frames.

    Args:
        data_base_path (Path): Data base path of the synthetic dataset
        num_input_frames (int): Number of input frames T of the input window
        num_reference_frames (int): Number of reference frames D
        crop_mode (str): Crop mode. Must be in: ["center", "random"]
        patch_size (int): Size of the crops

    Returns:
        dict with keys:
            "imgs_lq" (torch.Tensor): Input frames
            "imgs_gt" (torch.Tensor): Ground-truth frames
            "imgs_ref" (torch.Tensor): Reference frames
            "img_name" (str): Name of the center input frame
    """

    def __init__(self,
                 data_base_path: Path,
                 num_input_frames: int = 5,
                 num_reference_frames: int = 5,
                 crop_mode: str = "center",
                 patch_size: int = 128):
        self.data_base_path = data_base_path
        self.num_input_frames = num_input_frames
        self.num_reference_frames = num_reference_frames
        self.crop_mode = crop_mode
        self.patch_size = patch_size

        self.lq_base_path = data_base_path / "input"
        self.gt_base_path = data_base_path / "gt"

        self.database_lq = lmdb.open(str(self.lq_base_path / "input.lmdb"), readonly=True, lock=False, readahead=False)
        self.database_gt = lmdb.open(str(self.gt_base_path / "gt.lmdb"), readonly=True, lock=False, readahead=False)

        with open(str(self.lq_base_path / "input.lmdb" / "meta_info.txt"), 'r') as f:
            lines = f.readlines()
        # Split each line on space and take the first item
        self.img_names = [line.split()[0] for line in lines]

        # Get start and end index of each clip
        self.clip_intervals = {}
        for key, group in groupby(enumerate([str(Path(x).parent) for x in self.img_names]), key=lambda x: x[1]):
            group = list(group)
            self.clip_intervals[key] = (group[0][0], group[-1][0])

        # Load references
        if not (self.lq_base_path / f"{self.num_reference_frames}_references.json").exists():
            self.generate_references()
        with open(str(self.lq_base_path / f"{self.num_reference_frames}_references.json"), 'r') as f:
            self.references = json.load(f)

    def __getitem__(self, idx):
        center_frame_lq = self.img_names[idx]
        img_name = str(Path(center_frame_lq).name)
        clip_name = str(Path(center_frame_lq).parent)
        clip_interval_start, clip_interval_end = self.clip_intervals[clip_name]
        half_input_window_size = self.num_input_frames // 2

        # Get idxs of the frames in the window
        idxs_imgs = np.arange(idx - half_input_window_size, idx + half_input_window_size + 1)
        idxs_imgs = list(idxs_imgs[(idxs_imgs >= clip_interval_start) & (idxs_imgs <= clip_interval_end)])
        imgs_lq = []
        imgs_gt = []

        # Get the frames from the LMDB database
        with self.database_lq.begin(write=False) as txn_lq:
            for img_idx in idxs_imgs:
                img_bytes = txn_lq.get(self.img_names[img_idx][:-4].encode('ascii'))
                img_t = ToTensor()(imfrombytes(img_bytes, float32=True))
                imgs_lq.append(img_t)

        with self.database_gt.begin(write=False) as txn_gt:
            for img_idx in idxs_imgs:
                img_bytes = txn_gt.get(self.img_names[img_idx][:-4].encode('ascii'))
                img_t = ToTensor()(imfrombytes(img_bytes, float32=True))
                imgs_gt.append(img_t)

        # Pad with black frames if the window is not complete (the center frame is too close to the start or the end of the clip)
        if len(imgs_lq) < self.num_input_frames:
            black_frame = torch.zeros_like(imgs_lq[0])
            missing_frames_left = half_input_window_size - (idx - 0)
            for _ in range(missing_frames_left):
                imgs_lq.insert(0, black_frame)
                imgs_gt.insert(0, black_frame)
            missing_frames_right = half_input_window_size - (len(self.img_names) - 1 - idx)
            for _ in range(missing_frames_right):
                imgs_lq.append(black_frame)
                imgs_gt.append(black_frame)
        imgs_lq = torch.stack(imgs_lq)
        imgs_gt = torch.stack(imgs_gt)

        # Get the reference frames
        imgs_ref = []
        with self.database_lq.begin(write=False) as txn_lq:
            for ref_name in self.references[clip_name][center_frame_lq]:
                img_bytes = txn_lq.get(ref_name[:-4].encode('ascii'))
                img_t = ToTensor()(imfrombytes(img_bytes, float32=True))
                imgs_ref.append(img_t)
        imgs_ref = torch.stack(imgs_ref)

        imgs_lq, imgs_gt, imgs_ref = preprocess([imgs_lq, imgs_gt, imgs_ref], mode="crop",
                                                patch_size=self.patch_size, crop_mode=self.crop_mode)

        return {"imgs_lq": imgs_lq,
                "imgs_gt": imgs_gt,
                "imgs_ref": imgs_ref,
                "img_name": f"{clip_name}/{img_name}"}

    def __len__(self):
        return len(self.img_names)

    def generate_references(self):
        """
        Generate the file with the references for each frame of the dataset.
        """
        print("Generating references...")
        clip_model, clip_preprocess = clip.load("RN50x4", device=device, jit=True)

        # Extract text features using prompt ensembling
        with torch.no_grad(), torch.cuda.amp.autocast():
            tokenized_prompts = clip.tokenize(prompts).to(device)
            text_features = F.normalize(clip_model.encode_text(tokenized_prompts), dim=-1)
            text_features = F.normalize(text_features.mean(dim=0), dim=-1).unsqueeze(0)  # Prompt ensembling

        # Extract image features and compute similarity scores
        output = {k: {} for k in self.clip_intervals.keys()}
        clip_img_names = {k: [] for k in self.clip_intervals.keys()}
        clip_img_features = {k: [] for k in self.clip_intervals.keys()}
        clip_similarity_scores = {k: [] for k in self.clip_intervals.keys()}
        with self.database_lq.begin(write=False) as txn_lq:
            for img_name in self.img_names:
                img_bytes = txn_lq.get(img_name[:-4].encode('ascii'))
                img = Image.fromarray(imfrombytes(img_bytes, float32=False))
                clip_name = str(Path(img_name).parent)
                clip_img_names[clip_name].append(img_name)
                preprocessed_img = clip_preprocess(img).to(device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    img_feat = F.normalize(clip_model.encode_image(preprocessed_img.unsqueeze(0)), dim=-1)
                sim_score = img_feat @ text_features.T
                clip_img_features[clip_name].append(img_feat.cpu())
                clip_similarity_scores[clip_name].append(sim_score.cpu().item())

        for key in clip_img_names.keys():
            clip_img_names[key] = np.array(clip_img_names[key])
            clip_img_features[key] = torch.cat(clip_img_features[key], dim=0)
            clip_similarity_scores[key] = np.array(clip_similarity_scores[key])

            # Classify frames
            sorted_similarity_scores = np.sort(clip_similarity_scores[key])
            threshold = threshold_otsu(sorted_similarity_scores)
            threshold_index = sorted_similarity_scores.searchsorted(threshold)
            indexes = np.argsort(clip_similarity_scores[key])[:threshold_index]  # Indexes of clean frames

            # Select references
            for i, img_feat in enumerate(clip_img_features[key]):
                similarity = F.cosine_similarity(img_feat.unsqueeze(0), clip_img_features[key][indexes], dim=-1)
                similarity_indexes = torch.argsort(similarity, descending=True)
                similarity_indexes = similarity_indexes[:self.num_reference_frames].numpy()
                similar_imgs = clip_img_names[key][similarity_indexes].tolist()
                # Pad with the first image if there are not enough similar images
                while len(similar_imgs) < self.num_reference_frames:
                    similar_imgs.append(similar_imgs[0])
                output[key][clip_img_names[key][i]] = similar_imgs

        # Save references
        with open(str(self.lq_base_path / f"{self.num_reference_frames}_references.json"), 'w') as f:
            json.dump(output, f)
