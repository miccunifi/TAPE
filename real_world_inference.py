import torch
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
import json
from pathlib import Path
import cv2
from tqdm import tqdm
import clip
from PIL import Image
from skimage.filters import threshold_otsu
import torchvision
import shutil

from utils.prompts import prompts
from data.real_world_video_dataset import RealWorldVideoDataset
from models.swin_unet import SwinUNet


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def real_world_test(args):
    """
    Restore a real-world video (i.e. without ground truth) using the pretrained model.
    """

    input_video_name = args.input_path.stem
    output_folder = args.output_path / input_video_name
    output_folder.mkdir(parents=True, exist_ok=False)
    output_folder.mkdir(parents=True, exist_ok=True)
    input_frames_folder = output_folder / "input_frames"
    input_frames_folder.mkdir(parents=True, exist_ok=True)
    restored_frames_folder = output_folder / "restored_frames"
    restored_frames_folder.mkdir(parents=True, exist_ok=True)
    references_file_path = output_folder / "references.json"

    ### 1) Frames extraction
    print("Extracting frames from the video...")
    input_video = cv2.VideoCapture(str(args.input_path))
    fps = input_video.get(cv2.CAP_PROP_FPS)
    frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(frame_count)):
        success, frame = input_video.read()
        if not success:
            raise Exception("Failed to read frame from video")
        padded_i = str(i).zfill(len(str(frame_count)))      # Pad to a number of digits large enough to contain the total number of frames
        cv2.imwrite(str(input_frames_folder / f"{padded_i}.{args.frame_format}"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    input_video.release()

    ### 2) Frame classification and references selection
    print("Classifying frames and selecting references...")
    clip_model, clip_preprocess = clip.load("RN50x4", device=device, jit=True)
    output = {}

    # Extract text features using prompt ensembling
    with torch.no_grad(), torch.cuda.amp.autocast():
        tokenized_prompts = clip.tokenize(prompts).to(device)
        text_features = F.normalize(clip_model.encode_text(tokenized_prompts), dim=-1)
        text_features = F.normalize(text_features.mean(dim=0), dim=-1).unsqueeze(0)  # Prompt ensembling

    # Extract image features and compute similarity scores
    img_features = []
    img_names = []
    similarity_scores = []
    for img_path in tqdm(sorted(list(input_frames_folder.glob("*"))), desc="Extracting CLIP image features"):
        img_names.append(img_path.name)
        img = Image.open(img_path)
        preprocessed_img = clip_preprocess(img).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            img_feat = F.normalize(clip_model.encode_image(preprocessed_img.unsqueeze(0)), dim=-1)
        sim_score = img_feat @ text_features.T
        img_features.append(img_feat.cpu())
        similarity_scores.append(sim_score.cpu().item())

    img_names = np.array(img_names)
    img_features = torch.cat(img_features, dim=0)

    # Classify frames
    similarity_scores = np.array(similarity_scores)
    sorted_similarity_scores = np.sort(similarity_scores)
    threshold = threshold_otsu(sorted_similarity_scores)
    threshold_index = sorted_similarity_scores.searchsorted(threshold)
    indexes = np.argsort(similarity_scores)[:threshold_index]   # Indexes of clean frames

    # Select references
    for i, img_feat in enumerate(tqdm(img_features, desc="Selecting references")):
        similarity = F.cosine_similarity(img_feat.unsqueeze(0), img_features[indexes], dim=-1)
        similarity_indexes = torch.argsort(similarity, descending=True)
        similarity_indexes = similarity_indexes[:args.num_reference_frames].numpy()
        similar_imgs = img_names[similarity_indexes].tolist()
        while len(similar_imgs) < args.num_reference_frames:    # Pad with the first image if there are not enough similar images
            similar_imgs.append(similar_imgs[0])
        output[img_names[i]] = similar_imgs

    # Save references
    with open(references_file_path, 'w') as f:
        json.dump(output, f)

    # Free memory
    del clip_model
    del text_features
    del img_feat
    torch.cuda.empty_cache()

    ### 3) Video restoration
    print("Restoring the video...")
    dataset = RealWorldVideoDataset(input_frames_folder, num_input_frames=args.num_input_frames,
                                    num_reference_frames=args.num_reference_frames,
                                    references_file_path=references_file_path, preprocess_mode=args.preprocess_mode,
                                    patch_size=args.patch_size, frame_format=args.frame_format)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=False, pin_memory=True, drop_last=False)

    if args.preprocess_mode != "none" and (frame_width > args.patch_size or frame_height > args.patch_size):
        if args.preprocess_mode == "crop":
            new_frame_width = min(frame_width, args.patch_size)
            new_frame_height = min(frame_height, args.patch_size)
        elif args.preprocess_mode == "resize":
            if frame_height > frame_height:
                new_frame_height = args.patch_size
                new_frame_width = int(frame_width * args.patch_size / frame_height)
            else:
                new_frame_width = args.patch_size
                new_frame_height = int(frame_height * args.patch_size / frame_width)
        else:
            raise ValueError(f"Unknown preprocess mode: {args.preprocess_mode}")
    else:
        new_frame_width = frame_width
        new_frame_height = frame_height

    output_video = cv2.VideoWriter(str(output_folder / f"restored_{input_video_name}.mp4"),
                                   cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_frame_width, new_frame_height))
    if args.generate_combined_video:
        combined_output_video = cv2.VideoWriter(str(output_folder / f"combined_{input_video_name}.mp4"),
                                                cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_frame_width * 2, new_frame_height))
    else:
        combined_output_video = None

    # Load model
    model = SwinUNet()
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.eval().to(device)

    for batch in tqdm(dataloader, desc="Restoring frames"):
        imgs_lq = batch["imgs_lq"]
        imgs_ref = batch["imgs_ref"]
        img_names = batch["img_name"]

        # Image size must be divisible by 16 (due to the 4 downsampling operations)
        h, w = imgs_lq.shape[-2:]
        pad_width = (16 - (w % 16)) % 16
        pad_height = (16 - (h % 16)) % 16
        pad = (0, pad_width, 0, pad_height)
        imgs_lq = F.pad(imgs_lq, pad=pad, mode="constant", value=0).to(device)
        imgs_ref = F.pad(imgs_ref, pad=pad, mode="constant", value=0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            output = model(imgs_lq, imgs_ref)
            output = torch.clamp(output, min=0, max=1)

        for i, img_name in enumerate(img_names):
            img_num = int(img_name[:-4])
            restored_frame = output[i, args.num_input_frames // 2]
            restored_frame = torchvision.transforms.functional.crop(restored_frame, top=0, left=0, height=h, width=w)
            restored_frame = restored_frame.cpu().numpy().transpose(1, 2, 0) * 255
            restored_frame = cv2.cvtColor(restored_frame, cv2.COLOR_RGB2BGR).astype(np.uint8)
            cv2.imwrite(str(restored_frames_folder / f"{img_num}.{args.frame_format}"), restored_frame)

            # Reconstruct the video
            output_video.write(restored_frame)
            if args.generate_combined_video:
                input_frame = imgs_lq[i, args.num_input_frames // 2]
                input_frame = torchvision.transforms.functional.crop(input_frame, top=0, left=0, height=h, width=w)
                input_frame = input_frame.cpu().numpy().transpose(1, 2, 0) * 255
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR).astype(np.uint8)
                combined_frame = np.concatenate((input_frame, restored_frame), axis=1)
                combined_output_video.write(combined_frame)

    output_video.release()
    if args.generate_combined_video:
        combined_output_video.release()

    # Free memory
    del model
    del imgs_lq
    del imgs_ref
    torch.cuda.empty_cache()

    if args.no_intermediate_products:
        print("Deleting intermediate products...")
        (output_folder / f"restored_{input_video_name}.mp4").rename(Path(args.output_path) / f"restored_{input_video_name}.mp4")
        if args.generate_combined_video:
            (output_folder / f"combined_{input_video_name}.mp4").rename(Path(args.output_path) / f"combined_{input_video_name}.mp4")
        shutil.rmtree(output_folder)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Path to the video to restore")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--checkpoint-path", type=str, default="experiments/pretrained_model/checkpoint.pth",
                        help="Path to the pretrained model checkpoint")
    parser.add_argument("--num-input-frames", type=int, default=5,
                        help="Number of input frames T for each input window")
    parser.add_argument("--num-reference-frames", type=int, default=5,
                        help="Number of reference frames D for each input window")
    parser.add_argument("--preprocess-mode", type=str, default="crop", choices=["crop", "resize", "none"],
                        help="Preprocessing mode, options: ['crop', 'resize', 'none']. 'crop' extracts the --patch-size"
                             " center crop, 'resize' resizes the longest side to --patch-size while keeping the aspect"
                             " ratio, 'none' applies no preprocessing")
    parser.add_argument("--patch-size", type=int, default=512,
                        help="Maximum patch size for --preprocess-mode ['crop', 'resize']")
    parser.add_argument("--frame-format", type=str, default="jpg",
                        help="Frame format of the extracted and restored frames")
    parser.add_argument("--generate-combined-video", action="store_true",
                        help="Whether to generate the combined video (i.e. input and restored videos side by side)")
    parser.add_argument("--no-intermediate-products", action="store_true",
                        help="Whether to delete intermediate products (i.e. input frames, restored frames, references)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=20, help="Number of workers of the data loader")

    args = parser.parse_args()

    args.input_path = Path(args.input_path)
    args.output_path = Path(args.output_path)
    real_world_test(args)
