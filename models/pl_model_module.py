import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics.image
import torchmetrics
from torchvision.transforms.functional import to_pil_image
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
import os.path as osp
from einops import rearrange

from models.losses import CharbonnierLoss, PerceptualLoss


class ModelModule(pl.LightningModule):
    """
    Pytorch Lightning Module for model training.

    Args:
        net (nn.Module): Model to train
        num_input_frames (int): Number of input frames in the input window
        pixel_loss_weight (float): Weight of the pixel loss
        perceptual_loss_weight (float): Weight of the perceptual loss
        lr (float): Learning rate
    """

    def __init__(self, net: nn.Module, num_input_frames: int = 5, pixel_loss_weight: float = 200,
                 perceptual_loss_weight: float = 1, lr: float = 2e-5):
        super(ModelModule, self).__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.num_input_frames = num_input_frames
        self.pixel_loss_weight = pixel_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.lr = lr

        self.pixel_criterion = CharbonnierLoss(loss_weight=self.pixel_loss_weight)

        vgg_layer_weights = {'conv5_4': 1, 'relu4_4': 1, 'relu3_4': 1, 'relu2_2': 1}
        self.perceptual_criterion = PerceptualLoss(layer_weights=vgg_layer_weights, loss_weight=self.perceptual_loss_weight)

        self.psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type="alex")

    def forward(self, *x):
        return self.net(*x)

    def training_step(self, batch, batch_idx):
        imgs_lq = batch["imgs_lq"]
        imgs_gt = batch["imgs_gt"]
        imgs_ref = batch["imgs_ref"]
        outputs = self.net(imgs_lq, imgs_ref)

        B, T, C, H, W = outputs.shape
        outputs = rearrange(outputs, 'b t c h w -> (b t) c h w', b=B, t=T)
        imgs_gt = rearrange(imgs_gt, 'b t c h w -> (b t) c h w', b=B, t=T)

        pixel_loss = self.pixel_criterion(outputs, imgs_gt)
        perceptual_loss = self.perceptual_criterion(outputs, imgs_gt)

        total_loss = pixel_loss + perceptual_loss

        log_loss = {"total_loss": total_loss,
                    "pixel_loss": pixel_loss,
                    "perceptual_loss": perceptual_loss}
        self.log_dict(log_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True, sync_dist=True, batch_size=imgs_gt.shape[0])
        return total_loss

    def validation_step(self, batch, batch_idx):
        imgs_lq = batch["imgs_lq"]
        imgs_gt = batch["imgs_gt"]
        imgs_ref = batch["imgs_ref"]
        outputs = self.net(imgs_lq, imgs_ref).to(torch.float32)

        psnr, ssim, lpips = 0., 0., 0.
        for i, output in enumerate(outputs):
            output = torch.clamp(output, 0, 1)
            img_gt = imgs_gt[i]
            psnr += self.psnr(output, img_gt)
            ssim += self.ssim(output, img_gt, data_range=1.)
            with torch.no_grad():
                lpips += self.lpips(output * 2 - 1, img_gt * 2 - 1)    # Input must be in [-1, 1] range
        psnr /= len(outputs)
        ssim /= len(outputs)
        lpips /= len(outputs)

        log_metrics = {"psnr": psnr,
                       "ssim": ssim,
                       "lpips": lpips}

        self.log_dict(log_metrics, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=imgs_gt.shape[0])

        imgs_name = batch["img_name"]
        for i, img_name in enumerate(imgs_name):
            img_num = int(osp.basename(img_name)[:-4])
            if img_num % 100 == 0 or batch_idx == 0:
                single_img_lq = imgs_lq[0, self.num_input_frames // 2]
                single_img_gt = imgs_gt[0, self.num_input_frames // 2]
                single_img_output = torch.clamp(outputs[0, self.num_input_frames // 2], 0., 1.)
                concatenated_img = torch.cat((single_img_lq, single_img_output, single_img_gt), -1)
                self.logger.experiment.log_image(to_pil_image(concatenated_img.cpu()), str(img_num), step=self.current_epoch)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        self.psnr.reset()
        self.lpips.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.99))
        return optimizer
