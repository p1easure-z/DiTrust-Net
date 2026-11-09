import argparse
import datetime
import os
import random
import sys
import warnings

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
for path in (CURRENT_DIR, PROJECT_ROOT):
    if path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(1, PROJECT_ROOT)

from kits.logger import setup_logger
from networks.Net import DiTrustNet, default_backbone_weight_path
from trainer import Trainer

warnings.filterwarnings("ignore")


DATASET_ROOTS = {
    "BraTS2020": os.environ.get("BRATS2020_ROOT"),
    "BraTS2021": os.environ.get("BRATS2021_ROOT"),
}
MODEL_NAME = "DiTrust-Net"


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_training_args():
    parser = argparse.ArgumentParser(description="Train DiTrust-Net for Medical Image Segmentation")

    parser.add_argument("--dataset_name",type=str,
                        default="BraTS2020",
                        choices=["BraTS2020", "BraTS2021"],
                        help="Dataset name. Controls the actual data_root and result subdirectories.",)

    parser.add_argument("--epochs", type=int, default=100,
                        help="Total training epochs.")
    parser.add_argument("--batch_size", type=int, default=6,
                        help="Batch size per GPU.")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Epochs for warmup.")
    parser.add_argument("--lr", type=float, default=1e-4, metavar="LR", help="Initial learning rate (AdamW).")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for Cosine Scheduler.")
    parser.add_argument("--lr-scheduler",type=str,default="cos",choices=["poly", "step", "cos"],help="Scheduler.")
    parser.add_argument("--weight_decay", type=float, default=5e-2, metavar="W", help="Weight decay.")

    parser.add_argument("--freeze_backbone", type=str2bool, default=True, help="Freeze both selected backbones.")

    parser.add_argument("--save_dir",type=str,default=os.path.join(CURRENT_DIR, "results"),help="DiTrust-Net results root directory.")

    args = parser.parse_args()
    args.backbone = "vmamba"
    args.model = MODEL_NAME
    args.start_epoch = 0
    return args


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_data_root(dataset_name):
    if dataset_name not in DATASET_ROOTS:
        raise ValueError(f"Unsupported dataset_name '{dataset_name}'. Available: {list(DATASET_ROOTS)}")
    data_root = DATASET_ROOTS[dataset_name]
    if not data_root:
        raise ValueError(f"Set {dataset_name.upper()}_ROOT before training.")
    return os.path.abspath(os.path.expanduser(data_root))


def resolve_run_id():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")


def resolve_result_paths(save_dir, model_name, dataset_name, run_id):
    loss_dir = os.path.join(save_dir, "losses", model_name, dataset_name, run_id)
    return {
        "log_file": os.path.join(save_dir, "logs", model_name, dataset_name, f"{run_id}.txt"),
        "weights_dir": os.path.join(save_dir, "weights", model_name, dataset_name, run_id),
        "loss_dir": loss_dir,
        "loss_file": os.path.join(loss_dir, "loss_detail.txt"),
        "loss_curve_file": os.path.join(loss_dir, "loss_curve.png"),
    }


def prepare_run_context(args):
    args.run_id = resolve_run_id()
    for key, value in resolve_result_paths(args.save_dir, args.model, args.dataset_name, args.run_id).items():
        setattr(args, key, value)
    args.data_root = resolve_data_root(args.dataset_name)
    return args


def configure_runtime(seed=42):
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    seed_everything(seed)


def initialize_logging(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.log_file = setup_logger(args.save_dir, args)
    print(f"[Run] model={args.model}, dataset={args.dataset_name}")
    print()


def set_backbone_trainable(model, trainable):
    for backbone in model.backbones:
        for parameter in backbone.parameters():
            parameter.requires_grad = trainable


def build_training_model(args):
    model = DiTrustNet(backbone_name=args.backbone).cuda()

    backbone_path = default_backbone_weight_path(CURRENT_DIR, args.backbone)
    if os.path.exists(backbone_path):
        print(f"[Weight] backbone={args.backbone}, path={backbone_path}")
        model.load_backbone_weights(backbone_path)
    else:
        print(f"[Weight] backbone={args.backbone}, missing={backbone_path}")

    set_backbone_trainable(model, trainable=not args.freeze_backbone)
    print(f"[Freeze] backbone={args.freeze_backbone}")
    print()
    return model


def run_training(args, model):
    trainer = Trainer(args, model)

    try:
        for epoch in range(args.start_epoch, args.epochs):
            trainer.training(epoch)
            trainer.validation(epoch)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    finally:
        if hasattr(trainer, "best_epoch_details") and trainer.best_epoch_details:
            print(f"\n[Best Model Record (Epoch {trainer.best_dice_epoch})]")
            print(trainer.best_epoch_details)

        torch.cuda.empty_cache()


def main():
    args = prepare_run_context(parse_training_args())
    configure_runtime(seed=42)
    initialize_logging(args)
    model = build_training_model(args)
    run_training(args, model)


if __name__ == "__main__":
    main()
