import torch
from argparse import ArgumentParser
from dotmap import DotMap

from models.swin_unet import SwinUNet
from utils.utils import PROJECT_ROOT, init_logger
from train import train
from test import test


def main(args):
    # Initialize logger
    logger = init_logger(experiment_name=args.experiment_name, api_key=args.comet_api_key,
                         project_name=args.comet_project_name, online=not args.comet_offline)
    logger.experiment.log_parameters(args.toDict())

    if not args.test_only and args.eval_type == "pretrained":
        print("WARNING: you can only test the pretrained model, not train it. Setting test_only=True")
        args.test_only = True

    # Initialize network
    net = SwinUNet(in_chans=3,
                   embed_dim=args.embed_dim,
                   depths=args.depths,
                   num_heads=args.num_heads,
                   window_size=args.window_size,
                   mlp_ratio=args.mlp_ratio,
                   qkv_bias=not args.no_qkv_bias,
                   qk_scale=args.qk_scale,
                   drop_rate=args.drop_rate,
                   attn_drop_rate=args.attn_drop_rate,
                   drop_path_rate=args.drop_path_rate,
                   norm_layer=getattr(torch.nn, args.norm_layer),
                   use_checkpoint=not args.no_checkpoint)

    if not args.test_only:
        train(args, net, logger)

    if args.eval_type == "scratch":
        # Load the best checkpoint
        checkpoints_path = PROJECT_ROOT / "experiments" / args.experiment_name / "checkpoints"
        checkpoint_file = [ckpt_path for ckpt_path in checkpoints_path.glob("*.pth") if "lpips" in ckpt_path.name][0]
        checkpoint = checkpoints_path / checkpoint_file
        state_dict = torch.load(checkpoint, map_location="cpu")
        # Check if the checkpoint is a PyTorch Lightning checkpoint
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
            state_dict = {k.replace("net.", "", 1): v for k, v in state_dict.items() if k.startswith("net.")}
    elif args.eval_type == "pretrained":
        # Load the pretrained checkpoint
        net = SwinUNet()
        state_dict = torch.load(PROJECT_ROOT / "experiments" / "pretrained_model" / "checkpoint.pth", map_location="cpu")

    net.load_state_dict(state_dict, strict=True)

    test(args, net, logger)


if __name__ == '__main__':
    parser = ArgumentParser()

    # General
    parser.add_argument("--experiment-name", type=str, required=True, help="Experiment name")
    parser.add_argument("--data-base-path", type=str, required=True, help="Base path of the dataset")
    parser.add_argument("--devices", type=int, nargs="+", default=[0], help="GPU device ids")

    # Training
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--num-input-frames", type=int, default=5, help="Number of input frames")
    parser.add_argument("--num-reference-frames", type=int, default=5, help="Number of reference frames")
    parser.add_argument("--train-patch-size", type=int, default=128, help="Patch size for training")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=20, help="Number of workers for the dataloaders")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--pixel-loss-weight", type=float, default=200)
    parser.add_argument("--perceptual-loss-weight", type=float, default=1)

    # Test
    parser.add_argument("--test-patch-size", type=int, default=512, help="Patch size for testing")
    parser.add_argument("--test-only", default=False, action="store_true", help="Skip training and test only")
    parser.add_argument("--eval-type", default="scratch", choices=["scratch", "pretrained"],
                        help="Whether to test a model trained from scratch or the one pretrained by the authors of the"
                             "paper. Must be in ['scratch', 'pretrained']")
    parser.add_argument("--no-vmaf", default=False, action="store_true", help="Skip VMAF computation")

    # Model
    parser.add_argument("--embed-dim", type=int, default=96, help="Dimension of the token embeddings")
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 2, 6, 2], help="Depths of the Swin Transformer layers")
    parser.add_argument("--num-heads", type=int, nargs="+", default=[8, 8, 8, 8], help="Number of attention heads for each layer")
    parser.add_argument("--window-size", type=int, nargs="+", default=[2, 8, 8], help="Window size for each layer")
    parser.add_argument("--mlp-ratio", type=float, default=4., help="Ratio of the mlp hidden dimension to the embedding dimension")
    parser.add_argument("--no-qkv-bias", default=False, action="store_true", help="If True, add a learnable bias to query, key, value")
    parser.add_argument("--qk-scale", type=float, default=None, help="Override default qk scale of head_dim ** -0.5 if set")
    parser.add_argument("--drop-rate", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--attn-drop-rate", type=float, default=0.0, help="Attention dropout rate")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="Stochastic depth rate")
    parser.add_argument("--norm-layer", type=str, default="LayerNorm", help="Normalization layer from torch.nn")
    parser.add_argument("--no-checkpoint", default=False, action="store_true")

    # Logging
    parser.add_argument("--comet-api-key", type=str, default="", help="Comet ML API key")
    parser.add_argument("--comet-project-name", type=str, default="TAPE", help="Comet ML project name")
    parser.add_argument("--comet-offline", default=False, action="store_true", help="Enable offline logging only")

    args = parser.parse_args()

    training_params = {
        "benchmark": True,
        "precision": 16,
        "log_every_n_steps": 50,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": args.devices,
        "max_epochs": args.num_epochs,
    }

    args.training_params = training_params
    args = DotMap(vars(args), _dynamic=False)

    main(args)
