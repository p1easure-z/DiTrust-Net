#!/usr/bin/env python3
"""Evaluate a trained DiTrust-Net checkpoint on a BraTS test split.

The default metric path intentionally matches ``Trainer.validation``:
``BraTSDatasetLoader(is_train=False)``, final model logits, sigmoid, a 0.5
threshold, global confusion-matrix Dice/IoU/Sensitivity, and the project's
``kits.metrics.hausdorff_95`` implementation averaged over slices.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import get_segmentation_dataset  # noqa: E402
from kits.metrics import SegMetrics, hausdorff_95  # noqa: E402
from networks.Net import DiTrustNet  # noqa: E402


DATASET_ROOTS = {
    "BraTS2020": os.environ.get("BRATS2020_ROOT"),
    "BraTS2021": os.environ.get("BRATS2021_ROOT"),
}
REGIONS = ("WT", "TC", "ET")
METRICS = ("Dice", "IoU", "Sensitivity", "Hausdorff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a trained DiTrust-Net checkpoint with the training metric protocol."
    )
    parser.add_argument(
        "--ckpt",
        default=str(PROJECT_ROOT / "results" / "DiTrust-Net-2021.pth"),
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--dataset-name",
        default="BraTS2021",
        choices=tuple(DATASET_ROOTS),
        help="Dataset associated with the checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset root containing testImage and testGt; defaults to BRATS2020_ROOT or BRATS2021_ROOT.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Report directory; defaults to results/validation/<dataset>/<checkpoint-name>.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Inference device.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Debug only: evaluate the first N slices; 0 evaluates the complete test split.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")


def configure_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    checkpoint = Path(args.ckpt).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    configured_data_root = args.data_root or DATASET_ROOTS[args.dataset_name]
    if not configured_data_root:
        raise ValueError(
            f"Pass --data-root or set {args.dataset_name.upper()}_ROOT."
        )
    data_root = Path(configured_data_root).expanduser().resolve()
    for directory in (data_root / "testImage", data_root / "testGt"):
        if not directory.is_dir():
            raise FileNotFoundError(f"Validation directory not found: {directory}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (
            PROJECT_ROOT
            / "results"
            / "validation"
            / args.dataset_name
            / checkpoint.stem
        ).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint, data_root, output_dir


def paired_validation_paths(data_root: Path, max_samples: int) -> Tuple[Sequence[str], Sequence[str]]:
    image_dir = data_root / "testImage"
    target_dir = data_root / "testGt"
    images = {path.name: path for path in image_dir.iterdir() if path.is_file()}
    targets = {path.name: path for path in target_dir.iterdir() if path.is_file()}
    missing_targets = sorted(set(images) - set(targets))
    missing_images = sorted(set(targets) - set(images))
    if missing_targets or missing_images:
        raise RuntimeError(
            "testImage/testGt pairing mismatch: "
            f"missing_targets={len(missing_targets)}, missing_images={len(missing_images)}"
        )
    names = sorted(images)
    if not names:
        raise RuntimeError(f"No validation samples found under {data_root}")
    if max_samples:
        names = names[:max_samples]
    return [str(images[name]) for name in names], [str(targets[name]) for name in names]


def torch_load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(payload: Any) -> Tuple[Dict[str, torch.Tensor], str, bool]:
    state_dict = payload
    container = "raw_state_dict"
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                state_dict = value
                container = key
                break
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise RuntimeError("Checkpoint does not contain a non-empty state dictionary")
    if not all(torch.is_tensor(value) for value in state_dict.values()):
        raise RuntimeError("Resolved checkpoint state dictionary contains non-tensor values")

    cleaned: Dict[str, torch.Tensor] = {}
    module_prefix_removed = False
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module.") :]
            module_prefix_removed = True
        if key in cleaned:
            raise RuntimeError(f"Duplicate checkpoint key after prefix cleanup: {key}")
        cleaned[key] = value
    return cleaned, container, module_prefix_removed


def build_model(checkpoint: Path, device: torch.device) -> Tuple[DiTrustNet, Dict[str, Any]]:
    model = DiTrustNet(backbone_name="vmamba")
    state_dict, container, prefix_removed = extract_state_dict(torch_load_checkpoint(checkpoint))
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_state))
    shape_mismatches = [
        {
            "key": key,
            "model": list(model_state[key].shape),
            "checkpoint": list(state_dict[key].shape),
        }
        for key in sorted(set(model_state) & set(state_dict))
        if tuple(model_state[key].shape) != tuple(state_dict[key].shape)
    ]
    audit = {
        "container": container,
        "module_prefix_removed": prefix_removed,
        "model_key_count": len(model_state),
        "checkpoint_key_count": len(state_dict),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
    }
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "Strict checkpoint load failed: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"shape_mismatches={len(shape_mismatches)}"
        )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, audit


def safe_metric(value: Any) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def metric_average(values: Mapping[str, float]) -> float:
    array = np.asarray([values[region] for region in REGIONS], dtype=np.float64)
    return float(array.mean()) if np.isfinite(array).all() else float("nan")


def evaluate(
    model: DiTrustNet,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    evaluators = {region: SegMetrics(2) for region in REGIONS}
    hausdorff_sums = {region: 0.0 for region in REGIONS}
    sample_count = 0
    hierarchy_violations = {"TC_outside_WT": 0, "ET_outside_TC": 0}
    nonfinite_logit_count = 0

    progress = tqdm(loader, desc="Validation", ncols=120, unit="batch")
    with torch.no_grad():
        for images, targets, _names in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            if not isinstance(outputs, Mapping) or "logits" not in outputs:
                raise RuntimeError("Model forward did not return outputs['logits']")
            logits = outputs["logits"]
            if logits.shape != targets.shape:
                raise RuntimeError(
                    f"Prediction/target shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}"
                )
            nonfinite_logit_count += int((~torch.isfinite(logits)).sum().item())
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            prediction = (torch.sigmoid(logits) > threshold).to(torch.uint8)

            prediction_array = prediction.cpu().numpy()
            target_array = targets.detach().cpu().numpy()
            batch_count = prediction_array.shape[0]
            sample_count += batch_count

            hierarchy_violations["TC_outside_WT"] += int(
                np.logical_and(prediction_array[:, 1] > 0, prediction_array[:, 0] == 0).sum()
            )
            hierarchy_violations["ET_outside_TC"] += int(
                np.logical_and(prediction_array[:, 2] > 0, prediction_array[:, 1] == 0).sum()
            )

            for channel, region in enumerate(REGIONS):
                evaluators[region].update(
                    prediction_array[:, channel, None],
                    target_array[:, channel, None],
                )
                for sample_index in range(batch_count):
                    hausdorff_sums[region] += hausdorff_95(
                        prediction_array[sample_index : sample_index + 1, channel],
                        target_array[sample_index : sample_index + 1, channel],
                    )
            progress.set_postfix(samples=sample_count)
    progress.close()

    if sample_count == 0:
        raise RuntimeError("Validation loader produced no samples")

    with np.errstate(divide="ignore", invalid="ignore"):
        metrics = {
            "Dice": {region: safe_metric(evaluators[region].dice()) for region in REGIONS},
            "IoU": {region: safe_metric(evaluators[region].IoU()) for region in REGIONS},
            "Sensitivity": {
                region: safe_metric(evaluators[region].sensitivity()) for region in REGIONS
            },
            "Hausdorff": {
                region: safe_metric(hausdorff_sums[region] / sample_count) for region in REGIONS
            },
        }
    for values in metrics.values():
        values["Avg"] = metric_average(values)

    diagnostics = {
        "sample_count": sample_count,
        "nonfinite_logit_count_before_sanitization": nonfinite_logit_count,
        "binary_hierarchy_violation_voxels": hierarchy_violations,
        "confusion_matrices": {
            region: evaluators[region].confusion_matrix.astype(np.int64).tolist()
            for region in REGIONS
        },
    }
    return metrics, diagnostics


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_reports(output_dir: Path, report: Mapping[str, Any]) -> None:
    json_path = output_dir / "validation_metrics.json"
    csv_path = output_dir / "validation_metrics.csv"
    json_path.write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", *REGIONS, "Avg"))
        writer.writeheader()
        for metric in METRICS:
            writer.writerow({"metric": metric, **report["metrics"][metric]})


def format_value(value: float) -> str:
    return f"{value:.4f}" if np.isfinite(value) else "nan"


def print_metrics(metrics: Mapping[str, Mapping[str, float]]) -> None:
    print()
    for metric in METRICS:
        values = metrics[metric]
        print(
            f"{metric}: "
            + ", ".join(f"{name}: {format_value(values[name])}" for name in (*REGIONS, "Avg"))
            + "."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_determinism(args.seed)
    checkpoint, data_root, output_dir = resolve_paths(args)
    device = resolve_device(args.device)
    image_paths, target_paths = paired_validation_paths(data_root, args.max_samples)

    dataset = get_segmentation_dataset(
        "brats",
        img_paths=image_paths,
        mask_paths=target_paths,
        is_train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    print(f"Dataset       : {args.dataset_name}")
    print(f"Data root     : {data_root}")
    print(f"Checkpoint    : {checkpoint}")
    print(f"Device        : {device}")
    print(f"Samples       : {len(dataset)}{' (partial debug run)' if args.max_samples else ' (complete split)'}")
    print(f"Threshold     : {args.threshold}")

    started = time.time()
    model, checkpoint_audit = build_model(checkpoint, device)
    metrics, diagnostics = evaluate(model, loader, device, args.threshold)
    elapsed_seconds = time.time() - started
    print_metrics(metrics)

    report = {
        "protocol": {
            "dataset_loader": "BraTSDatasetLoader(is_train=False)",
            "split": "testImage/testGt",
            "model_output": "outputs['logits']",
            "probability": "sigmoid(logits)",
            "threshold": args.threshold,
            "hard_hierarchy_postprocessing": False,
            "note": "The final logits already include the model's learned soft hierarchy.",
            "confusion_metrics": "global over all evaluated pixels, matching Trainer.validation",
            "hausdorff": (
                "kits.metrics.hausdorff_95 averaged equally over evaluated slices, "
                "matching Trainer.validation at batch_size=1"
            ),
        },
        "dataset": args.dataset_name,
        "data_root": str(data_root),
        "checkpoint": str(checkpoint),
        "checkpoint_audit": checkpoint_audit,
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "complete_split": args.max_samples == 0,
        "evaluated_samples": len(dataset),
        "elapsed_seconds": elapsed_seconds,
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
    write_reports(output_dir, report)
    print()
    print(f"Reports       : {output_dir}")
    print(f"Elapsed       : {elapsed_seconds / 60.0:.2f} min")
    if args.max_samples:
        print("WARNING       : --max-samples was set; these are not complete validation metrics.")
    if args.threshold != 0.5:
        print("WARNING       : threshold differs from Trainer.validation (0.5).")


if __name__ == "__main__":
    main()
