import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import cv2
from lpips import LPIPS
import pandas as pd
from tqdm import tqdm
from pytorch_lightning.loggers import CometLogger
from dotmap import DotMap

from utils.metrics import compute_psnr, compute_ssim, compute_lpips, compute_vmaf
from data.synthetic_video_dataset import SyntheticVideoDataset
from utils.utils import PROJECT_ROOT, device


def test(args: DotMap, net: nn.Module, logger: CometLogger):
    """
    Test the model on the synthetic dataset test split.

    Args:
        args: Arguments
        net: Network to be tested
        logger: Comet ML logger
    """
    data_base_path = Path(args.data_base_path)
    test_path = data_base_path / "test"
    gt_videos_path = test_path / "gt" / "videos"
    results_path = PROJECT_ROOT / "experiments" / args.experiment_name / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    videos_path = results_path / Path("videos")
    restored_videos_path = videos_path / "restored"
    restored_videos_path.mkdir(parents=True, exist_ok=True)
    combined_videos_path = videos_path / "combined"
    combined_videos_path.mkdir(parents=True, exist_ok=True)

    test_dataset = SyntheticVideoDataset(test_path, num_input_frames=args.num_input_frames,
                                         num_reference_frames=args.num_reference_frames, crop_mode="center",
                                         patch_size=args.test_patch_size)

    net.eval()
    net.to(device)

    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    lpips_net = LPIPS(pretrained=True, net='alex').to(device)
    count_videos = 0
    count_frames = 0
    if not args.no_vmaf:
        column_names = ["Name", "PSNR", "SSIM", "LPIPS", "VMAF"]
        metrics_list = ["PSNR", "SSIM", "LPIPS", "VMAF"]
    else:
        column_names = ["Name", "PSNR", "SSIM", "LPIPS"]
        metrics_list = ["PSNR", "SSIM", "LPIPS"]
    single_metrics_results = {}
    total_metrics_results = {metric: 0 for metric in metrics_list}
    output_csv = [[]]
    output_csv_path = results_path / "metrics_results.csv"

    last_clip = ""
    restored_video_writer = cv2.VideoWriter()
    combined_video_writer = cv2.VideoWriter()

    for batch in tqdm(dataloader):
        img_name = batch["img_name"]
        imgs_lq = batch["imgs_lq"].to(device, non_blocking=True)
        imgs_ref = batch["imgs_ref"].to(device, non_blocking=True)
        imgs_gt = batch["imgs_gt"]

        with torch.no_grad(), torch.cuda.amp.autocast():
            output = torch.clamp(net(imgs_lq, imgs_ref), 0, 1)
            output = output[:, args.num_input_frames // 2].permute(0, 2, 3, 1).cpu().numpy()

        for i in range(output.shape[0]):
            video_clip = Path(img_name[i]).parent
            (results_path / Path(video_clip)).mkdir(parents=True, exist_ok=True)

            restored = (output[i] * 255).astype(np.uint8)
            restored = restored[..., ::-1]    # RGB -> BGR

            input = imgs_lq[i, args.num_input_frames // 2].permute(1, 2, 0).cpu().numpy()
            input = (input * 255).astype(np.uint8)
            input = input[..., ::-1]  # RGB -> BGR

            gt = imgs_gt[i, args.num_input_frames // 2].permute(1, 2, 0).cpu().numpy()
            gt = (gt * 255).astype(np.uint8)
            gt = gt[..., ::-1]  # RGB -> BGR

            if video_clip != last_clip:
                restored_video_writer.release()
                combined_video_writer.release()
                if last_clip != "":
                    gt_video_path = gt_videos_path / f"{last_clip}.mp4"
                    if not args.no_vmaf:
                        single_metrics_results["VMAF"] = compute_vmaf(restored_video_path, gt_video_path,
                                                                    width=restored.shape[0], height=restored.shape[1])

                    for metric in single_metrics_results.keys():
                        if metric != "VMAF":
                            single_metrics_results[metric] /= count_frames
                        total_metrics_results[metric] += single_metrics_results[metric]
                    output_csv_row = list(single_metrics_results.values())
                    output_csv_row.insert(0, last_clip)
                    output_csv.append(output_csv_row)

                last_clip = video_clip
                restored_video_path = restored_videos_path / f"{last_clip}.mp4"
                combined_video_path = combined_videos_path / f"{last_clip}.mp4"
                restored_video_writer = cv2.VideoWriter(str(restored_video_path),
                                                        cv2.VideoWriter_fourcc(*'mp4v'), 60,
                                                        restored.shape[0:2])

                combined_shape = (restored.shape[0] * 3, restored.shape[1])
                combined_video_writer = cv2.VideoWriter(str(combined_video_path),
                                                        cv2.VideoWriter_fourcc(*'mp4v'), 60,
                                                        combined_shape)

                single_metrics_results = {metric: 0 for metric in metrics_list}
                count_videos += 1
                count_frames = 0

            restored_video_writer.write(restored)
            combined = np.hstack((input, restored, gt))
            combined_video_writer.write(combined)

            single_metrics_results["PSNR"] += compute_psnr(restored, gt)
            single_metrics_results["SSIM"] += compute_ssim(restored, gt)
            single_metrics_results["LPIPS"] += compute_lpips(restored, gt, lpips_net, device)
            count_frames += 1

            cv2.imwrite(f"{results_path}/{img_name[i]}.jpg", restored)

    restored_video_writer.release()
    combined_video_writer.release()

    # Compute metrics of last video
    gt_video_path = gt_videos_path / f"{last_clip}.mp4"
    if not args.no_vmaf:
        single_metrics_results["VMAF"] = compute_vmaf(restored_video_path, gt_video_path,
                                                    width=restored.shape[0], height=restored.shape[1])
    for metric in single_metrics_results.keys():
        if metric != "VMAF":
            single_metrics_results[metric] /= count_frames
        total_metrics_results[metric] += single_metrics_results[metric]
    output_csv_row = list(single_metrics_results.values())
    output_csv_row.insert(0, last_clip)
    output_csv.append(output_csv_row)

    for metric in total_metrics_results.keys():
        total_metrics_results[metric] /= count_videos
        logger.experiment.log_metric(metric, total_metrics_results[metric])
    output_csv_row = list(total_metrics_results.values())
    output_csv_row.insert(0, "Total")
    output_csv.append(output_csv_row)

    df = pd.DataFrame(output_csv)
    df.columns = column_names
    df.to_csv(output_csv_path, index=False)
