import glob
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm

from data import compute_case_sampling_weights, get_segmentation_dataset
from kits.losses import configure_loss
from kits.metrics import SegMetrics, hausdorff_95
from kits.scheduler import LR_Scheduler


def seed_dataloader_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


class LossHistoryRecorder(object):
    """Writes per-epoch loss details and refreshes a loss curve for the current run."""
    def __init__(self, args):
        self.args = args
        self.directory = getattr(
            args,
            "loss_dir",
            os.path.join(args.save_dir, "losses", args.model, args.dataset_name, args.run_id),
        )
        self.loss_file = getattr(args, "loss_file", os.path.join(self.directory, "loss_detail.txt"))
        self.curve_file = getattr(args, "loss_curve_file", os.path.join(self.directory, "loss_curve.png"))
        self.records = {}
        self.columns = []
        self._plot_warning_shown = False

        os.makedirs(self.directory, exist_ok=True)
        self._load_existing()

    @staticmethod
    def _to_float(value):
        if value is None:
            return np.nan
        try:
            value = float(value)
        except (TypeError, ValueError):
            return np.nan
        return value if np.isfinite(value) else np.nan

    def _register_columns(self, columns):
        for column in columns:
            if column not in self.columns:
                self.columns.append(column)

    def _load_existing(self):
        if not os.path.isfile(self.loss_file):
            return

        header = None
        with open(self.loss_file, "r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if header is None:
                    header = parts
                    if header and header[0] == "epoch":
                        self._register_columns(header[1:])
                    continue
                if not header or len(parts) != len(header):
                    continue

                row = {}
                try:
                    epoch = int(float(parts[0]))
                except ValueError:
                    continue
                row["epoch"] = epoch
                for key, value in zip(header[1:], parts[1:]):
                    row[key] = self._to_float(value)
                self.records[epoch] = row

    def record_epoch(self, epoch, train_losses, test_losses):
        record = {"epoch": int(epoch)}

        train_losses = train_losses or {}
        test_losses = test_losses or {}
        for key, value in train_losses.items():
            record[f"train/{key}"] = self._to_float(value)
        for key, value in test_losses.items():
            record[f"test/{key}"] = self._to_float(value)

        self._register_columns([key for key in record.keys() if key != "epoch"])
        self.records[int(epoch)] = record
        self._write_text()
        self._write_curve()

    def _write_text(self):
        with open(self.loss_file, "w", encoding="utf-8") as handle:
            handle.write("# Per-epoch loss details for DiTrust-Net.\n")
            handle.write("# Columns are tab-separated; train/* and test/* use epoch averages.\n")
            handle.write("\t".join(["epoch"] + self.columns) + "\n")
            for epoch in sorted(self.records):
                row = self.records[epoch]
                values = [str(epoch)]
                for column in self.columns:
                    value = row.get(column, np.nan)
                    values.append(f"{value:.6f}" if np.isfinite(value) else "nan")
                handle.write("\t".join(values) + "\n")

    def _write_curve(self):
        if not self.records:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            if not self._plot_warning_shown:
                message = f"[Loss Record] Skip loss curve generation: {exc}"
                print(message)
                logging.warning(message)
                self._plot_warning_shown = True
            return

        try:
            epochs = sorted(self.records)
            train_columns = [column for column in self.columns if column.startswith("train/")]
            test_columns = [column for column in self.columns if column.startswith("test/")]

            fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
            self._plot_group(axes[0], epochs, train_columns, "Train Loss")
            self._plot_group(axes[1], epochs, test_columns, "Test Loss")
            axes[1].set_xlabel("Epoch")
            fig.tight_layout()
            fig.savefig(self.curve_file, dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            if not self._plot_warning_shown:
                message = f"[Loss Record] Failed to write loss curve: {exc}"
                print(message)
                logging.warning(message)
                self._plot_warning_shown = True

    def _plot_group(self, axis, epochs, columns, title):
        axis.set_title(title)
        axis.set_ylabel("Loss")
        axis.grid(True, linestyle="--", alpha=0.3)

        if not columns:
            axis.text(0.5, 0.5, "No loss records", transform=axis.transAxes, ha="center", va="center")
            return

        for column in columns:
            values = np.asarray([self.records[epoch].get(column, np.nan) for epoch in epochs], dtype=np.float64)
            if not np.isfinite(values).any():
                continue
            axis.plot(epochs, values, linewidth=1.4, label=column.split("/", 1)[1])

        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)


class DiTrustWeightSaver(object):
    def __init__(self, args):
        self.directory = args.weights_dir
        self.model_name = args.model
        os.makedirs(self.directory, exist_ok=True)

    def save_epoch_weights(self, state_dict, epoch):
        weight_name = f"{self.model_name}-epoch{epoch:03d}.pth"
        weight_path = os.path.join(self.directory, weight_name)
        torch.save(state_dict, weight_path)
        return weight_path


class Trainer(object):
    CDMOE_ROUTING_WARMUP_EPOCHS = 5
    CDMOE_TEMPERATURE_START = 2.0
    CDMOE_TEMPERATURE_END = 0.35

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.device = torch.device("cuda")
        self.weight_saver = DiTrustWeightSaver(args)
        self.loss_recorder = LossHistoryRecorder(args)
        self.grad_clip = getattr(args, "grad_clip", 1.0)
        self.class_names = ("WT", "TC", "ET")
        self.weight_decay = float(getattr(args, "weight_decay", 5e-2))
        self.lr_scheduler_mode = getattr(args, "lr_scheduler", "cos")
        self.warmup_epochs = int(getattr(args, "warmup_epochs", 10))
        self.cdmoe_routing_warmup_epochs = self.CDMOE_ROUTING_WARMUP_EPOCHS
        self.cdmoe_temperature_start = self.CDMOE_TEMPERATURE_START
        self.cdmoe_temperature_end = self.CDMOE_TEMPERATURE_END
        self.min_lr = float(getattr(args, "min_lr", 1e-6))
        self._init_dataloader()

        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=self.weight_decay,
        )

        self.criterion_dice = configure_loss("LogCoshDiceLoss", from_logits=True).to(self.device)
        self.criterion_bce = configure_loss("BinaryCrossEntropy", from_logits=True).to(self.device)
        self.uncertainty_weight = 0.05
        dcte_target_weights = np.asarray([
            float(getattr(args, "dcte_error_target_weight", 0.60)),
            float(getattr(args, "dcte_boundary_target_weight", 0.25)),
            float(getattr(args, "dcte_hierarchy_target_weight", 0.15)),
        ], dtype=np.float64)
        if not np.isfinite(dcte_target_weights).all() or np.any(dcte_target_weights < 0.0):
            raise ValueError("DCTE target weights must be finite and non-negative.")
        dcte_target_weight_sum = float(dcte_target_weights.sum())
        if dcte_target_weight_sum <= 0.0:
            raise ValueError("At least one DCTE target weight must be positive.")
        dcte_target_weights = dcte_target_weights / dcte_target_weight_sum
        self.dcte_error_target_weight = float(dcte_target_weights[0])
        self.dcte_boundary_target_weight = float(dcte_target_weights[1])
        self.dcte_hierarchy_target_weight = float(dcte_target_weights[2])
        self.coarse_loss_weight = 0.10
        self.aux_loss_weight = 0.40
        self.fn_penalty_weight = 0.04

        self.scheduler = LR_Scheduler(
            mode=self.lr_scheduler_mode,
            base_lr=args.lr,
            num_epochs=args.epochs,
            iters_per_epoch=len(self.train_loader),
            warmup_epochs=self.warmup_epochs,
            min_lr=self.min_lr,
        )

        self.evaluator_wt = SegMetrics(2)
        self.evaluator_tc = SegMetrics(2)
        self.evaluator_et = SegMetrics(2)

        self._init_records()

    @staticmethod
    def _format_batch_names(batch_names):
        if batch_names is None:
            return "unknown"
        if isinstance(batch_names, (list, tuple)):
            return ", ".join(map(str, batch_names))
        return str(batch_names)

    @staticmethod
    def _sanitize_tensor(tensor, posinf=1e4, neginf=-1e4):
        if tensor is None:
            return None
        return torch.nan_to_num(tensor, nan=0.0, posinf=posinf, neginf=neginf)

    @staticmethod
    def _tensor_to_numpy(tensor, dtype=None):
        cpu_tensor = tensor.detach().cpu().contiguous()
        if cpu_tensor.dtype == torch.bool:
            cpu_tensor = cpu_tensor.to(torch.uint8)
        array = np.from_dlpack(cpu_tensor)
        if dtype is not None:
            return array.astype(dtype, copy=False)
        return array

    def _forward_model(self, image):
        outputs = self.model(image)
        aux_logits = outputs.get("aux_logits", {})
        meta = {
            key: self._sanitize_tensor(value)
            for key, value in outputs.get("meta", {}).items()
            if torch.is_tensor(value)
        }

        zero_loss = outputs["logits"].new_zeros(())
        losses = {}
        for key, value in outputs.get("losses", {}).items():
            if torch.is_tensor(value):
                losses[key] = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
        for key in (
            "cdmoe",
            "cdmoe_sparsity",
            "cdmoe_capacity",
            "cdmoe_selector_capacity",
            "cdmoe_balance",
        ):
            losses.setdefault(key, zero_loss)

        return {
            "logits": self._sanitize_tensor(outputs["logits"]),
            "aux_logits": {
                "x16": self._sanitize_tensor(aux_logits.get("x16")),
                "x8": self._sanitize_tensor(aux_logits.get("x8")),
                "coarse": self._sanitize_tensor(aux_logits.get("coarse")),
            },
            "losses": losses,
            "meta": meta,
        }

    @staticmethod
    def _init_train_loss_meter():
        return {
            "total": 0.0,
            "main_total": 0.0,
            "main_dice": 0.0,
            "main_bce": 0.0,
            "main_fn": 0.0,
            "uncertainty": 0.0,
            "coarse_total": 0.0,
            "coarse_dice": 0.0,
            "coarse_bce": 0.0,
            "coarse_weighted": 0.0,
            "aux_x16_total": 0.0,
            "aux_x16_dice": 0.0,
            "aux_x16_bce": 0.0,
            "aux_x16_weighted": 0.0,
            "aux_x8_total": 0.0,
            "aux_x8_dice": 0.0,
            "aux_x8_bce": 0.0,
            "aux_x8_weighted": 0.0,
            "cdmoe": 0.0,
            "cdmoe_sparsity": 0.0,
            "cdmoe_capacity": 0.0,
            "cdmoe_selector_capacity": 0.0,
            "cdmoe_balance": 0.0,
        }

    @staticmethod
    def _init_test_loss_meter():
        return {
            "total": 0.0,
            "dice": 0.0,
            "bce": 0.0,
            "fn": 0.0,
            "uncertainty": 0.0,
            "coarse_total": 0.0,
            "coarse_dice": 0.0,
            "coarse_bce": 0.0,
            "coarse_weighted": 0.0,
            "cdmoe": 0.0,
            "cdmoe_sparsity": 0.0,
            "cdmoe_capacity": 0.0,
            "cdmoe_selector_capacity": 0.0,
            "cdmoe_balance": 0.0,
        }

    @staticmethod
    def _average_loss_meter(loss_meter, denom):
        denom = max(denom, 1)
        return {key: value / denom for key, value in loss_meter.items()}

    def _configure_cdmoe_routing(self, epoch):
        model = self.model.module if hasattr(self.model, "module") else self.model
        if not hasattr(model, "set_cdmoe_routing_schedule"):
            return
        model.set_cdmoe_routing_schedule(
            epoch=epoch,
            total_epochs=self.args.epochs,
            warmup_epochs=self.cdmoe_routing_warmup_epochs,
            temperature_start=self.cdmoe_temperature_start,
            temperature_end=self.cdmoe_temperature_end,
        )

    def training(self, epoch):
        running_train_loss = 0.0
        train_loss_totals = self._init_train_loss_meter()
        self.model.train()
        self._configure_cdmoe_routing(epoch)
        start_time = time.time()

        self.scheduler(self.optimizer, 0, epoch)

        current_lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch + 1}/{self.args.epochs}] "
            f"lr={current_lr:.8f}, best_dice={self.best_dice_pre:.4f}@{self.best_dice_epoch}, "
            f"best_iou={self.best_iou_pre:.4f}@{self.best_iou_epoch}"
        )

        time.sleep(0.5)
        progress_bar = tqdm(self.train_loader, ncols=120)

        for iteration, batch in enumerate(progress_bar):
            images = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            case_names = batch[2] if len(batch) > 2 else None
            case_description = self._format_batch_names(case_names)

            if not torch.isfinite(images).all() or not torch.isfinite(targets).all():
                logging.warning(
                    f"Non-finite batch tensors detected at epoch={epoch + 1}, "
                    f"iter={iteration + 1}, files={case_description}"
                )
                images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
                targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

            self.scheduler(self.optimizer, iteration, epoch)
            self.optimizer.zero_grad()

            model_outputs = self._forward_model(
                image=images,
            )
            aux_logits_x16 = model_outputs["aux_logits"]["x16"]
            aux_logits_x8 = model_outputs["aux_logits"]["x8"]
            coarse_logits = model_outputs["aux_logits"]["coarse"]
            main_logits = model_outputs["logits"]
            regularization_terms = model_outputs["losses"]
            cdmoe_loss = regularization_terms["cdmoe"]
            cdmoe_sparsity_loss = regularization_terms["cdmoe_sparsity"]
            cdmoe_capacity_loss = regularization_terms["cdmoe_capacity"]
            cdmoe_selector_capacity_loss = regularization_terms["cdmoe_selector_capacity"]
            cdmoe_balance_loss = regularization_terms["cdmoe_balance"]
            model_meta = model_outputs["meta"]

            main_loss, main_breakdown = self._compute_loss(
                main_logits,
                targets,
                extra_meta=model_meta,
                enable_fn_penalty=True,
                return_breakdown=True,
            )
            total_loss = main_loss

            if coarse_logits is not None:
                coarse_loss, coarse_breakdown = self._compute_loss(
                    coarse_logits,
                    targets,
                    return_breakdown=True,
                )
                coarse_weighted = self.coarse_loss_weight * coarse_loss
                total_loss += coarse_weighted
            else:
                coarse_breakdown = {"total": 0.0, "dice": 0.0, "bce": 0.0}
                coarse_weighted = main_logits.new_zeros(())

            if aux_logits_x16 is not None and aux_logits_x8 is not None:
                aux_x16_loss, aux_x16_breakdown = self._compute_loss(
                    aux_logits_x16,
                    targets,
                    return_breakdown=True,
                )
                aux_x8_loss, aux_x8_breakdown = self._compute_loss(
                    aux_logits_x8,
                    targets,
                    return_breakdown=True,
                )
                aux_x16_weighted = self.aux_loss_weight * aux_x16_loss
                aux_x8_weighted = self.aux_loss_weight * aux_x8_loss
                total_loss += aux_x16_weighted + aux_x8_weighted
            else:
                aux_x16_breakdown = {"total": 0.0, "dice": 0.0, "bce": 0.0}
                aux_x8_breakdown = {"total": 0.0, "dice": 0.0, "bce": 0.0}
                aux_x16_weighted = main_logits.new_zeros(())
                aux_x8_weighted = main_logits.new_zeros(())

            cdmoe_term = cdmoe_loss.mean()
            total_loss += cdmoe_term

            if not torch.isfinite(total_loss):
                logging.warning(
                    f"Skip non-finite loss at epoch={epoch + 1}, iter={iteration + 1}, files={case_description}"
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                max_norm=self.grad_clip,
            )
            if not torch.isfinite(torch.as_tensor(gradient_norm)):
                logging.warning(
                    f"Skip optimizer step due to non-finite grad norm at epoch={epoch + 1}, "
                    f"iter={iteration + 1}, files={case_description}"
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue
            self.optimizer.step()

            running_train_loss += total_loss.item()
            train_loss_totals["total"] += total_loss.item()
            train_loss_totals["main_total"] += main_breakdown["total"]
            train_loss_totals["main_dice"] += main_breakdown["dice"]
            train_loss_totals["main_bce"] += main_breakdown["bce"]
            train_loss_totals["main_fn"] += main_breakdown["fn"]
            train_loss_totals["uncertainty"] += main_breakdown["uncertainty"]
            train_loss_totals["coarse_total"] += coarse_breakdown["total"]
            train_loss_totals["coarse_dice"] += coarse_breakdown["dice"]
            train_loss_totals["coarse_bce"] += coarse_breakdown["bce"]
            train_loss_totals["coarse_weighted"] += coarse_weighted.item()
            train_loss_totals["aux_x16_total"] += aux_x16_breakdown["total"]
            train_loss_totals["aux_x16_dice"] += aux_x16_breakdown["dice"]
            train_loss_totals["aux_x16_bce"] += aux_x16_breakdown["bce"]
            train_loss_totals["aux_x16_weighted"] += aux_x16_weighted.item()
            train_loss_totals["aux_x8_total"] += aux_x8_breakdown["total"]
            train_loss_totals["aux_x8_dice"] += aux_x8_breakdown["dice"]
            train_loss_totals["aux_x8_bce"] += aux_x8_breakdown["bce"]
            train_loss_totals["aux_x8_weighted"] += aux_x8_weighted.item()
            train_loss_totals["cdmoe"] += cdmoe_term.item()
            train_loss_totals["cdmoe_sparsity"] += cdmoe_sparsity_loss.mean().detach().item()
            train_loss_totals["cdmoe_capacity"] += cdmoe_capacity_loss.mean().detach().item()
            train_loss_totals["cdmoe_selector_capacity"] += (
                cdmoe_selector_capacity_loss.mean().detach().item()
            )
            train_loss_totals["cdmoe_balance"] += cdmoe_balance_loss.mean().detach().item()

            progress_bar.set_description(f"Train Loss: {running_train_loss / (iteration + 1):.4f}")

        progress_bar.close()

        epoch_seconds = int(time.time() - start_time)
        train_m, train_s = divmod(epoch_seconds, 60)
        self.last_train_loss = running_train_loss / len(self.train_loader)
        self.last_train_time_str = f"{train_m}m{train_s:02d}s"
        self.last_train_loss_breakdown = self._average_loss_meter(train_loss_totals, len(self.train_loader))

    def _init_dataloader(self):
        data_root = self.args.data_root
        train_image_paths = sorted(glob.glob(os.path.join(data_root, "trainImage", "*")))
        train_mask_paths = sorted(glob.glob(os.path.join(data_root, "trainGt", "*")))
        val_image_paths = sorted(glob.glob(os.path.join(data_root, "testImage", "*")))
        val_mask_paths = sorted(glob.glob(os.path.join(data_root, "testGt", "*")))

        if not train_image_paths:
            raise RuntimeError(f"Data path error: {os.path.join(data_root, 'trainImage')}")

        train_dataset = get_segmentation_dataset(
            "brats",
            img_paths=train_image_paths,
            mask_paths=train_mask_paths,
            is_train=True,
        )
        val_dataset = get_segmentation_dataset(
            "brats",
            img_paths=val_image_paths,
            mask_paths=val_mask_paths,
            is_train=False,
        )

        train_sampler = None
        if train_mask_paths:
            sample_weights = compute_case_sampling_weights(train_mask_paths)
            train_sampler = data.WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).double(),
                num_samples=len(sample_weights),
                replacement=True,
            )

        self.train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            num_workers=4,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_dataloader_worker,
            persistent_workers=True,
            prefetch_factor=4,
        )
        self.val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
        )

    def _init_records(self):
        self.best_dice_pre = 0.0
        self.best_iou_pre = 0.0
        self.best_dice_epoch = 0
        self.best_iou_epoch = 0
        self.best_epoch_details = ""
        self.last_train_loss = 0.0
        self.last_train_time_str = ""
        self.last_train_loss_breakdown = None
        self.last_test_loss_breakdown = None

    def _main_model_state_dict(self):
        return self.model.state_dict()

    def _save_epoch_weights(self, epoch, dice_avg, sensitivity_avg):
        weight_path = self.weight_saver.save_epoch_weights(
            state_dict=self._main_model_state_dict(),
            epoch=epoch + 1,
        )
        message = (
            f"[Save] {weight_path} "
            f"(Dice Avg={dice_avg:.4f}, Sensitivity Avg={sensitivity_avg:.4f})"
        )
        print(message)
        logging.info(message)

    def validation(self, epoch):
        running_val_loss = 0.0
        val_loss_totals = self._init_test_loss_meter()
        self.model.eval()
        start_time = time.time()

        self.evaluator_wt.reset()
        self.evaluator_tc.reset()
        self.evaluator_et.reset()
        hausdorff_sum = {"WT": 0.0, "TC": 0.0, "ET": 0.0}

        progress_bar = tqdm(self.val_loader, ncols=120)

        for iteration, batch in enumerate(progress_bar):
            images = batch[0].to(self.device)
            targets = batch[1].to(self.device)

            with torch.no_grad():
                model_outputs = self._forward_model(images)
                main_logits = model_outputs["logits"]
                regularization_terms = model_outputs["losses"]
                model_meta = model_outputs["meta"]
                coarse_logits = model_outputs["aux_logits"]["coarse"]
                prediction_prob = torch.sigmoid(main_logits)

                total_loss, loss_breakdown = self._compute_loss(
                    main_logits,
                    targets,
                    extra_meta=model_meta,
                    enable_fn_penalty=True,
                    return_breakdown=True,
                )
                if coarse_logits is not None:
                    coarse_loss, coarse_breakdown = self._compute_loss(
                        coarse_logits,
                        targets,
                        return_breakdown=True,
                    )
                    coarse_weighted = self.coarse_loss_weight * coarse_loss
                    total_loss = total_loss + coarse_weighted
                    loss_breakdown["total"] += coarse_weighted.detach().item()
                    loss_breakdown["coarse_total"] = coarse_breakdown["total"]
                    loss_breakdown["coarse_dice"] = coarse_breakdown["dice"]
                    loss_breakdown["coarse_bce"] = coarse_breakdown["bce"]
                    loss_breakdown["coarse_weighted"] = coarse_weighted.detach().item()
                else:
                    loss_breakdown["coarse_total"] = 0.0
                    loss_breakdown["coarse_dice"] = 0.0
                    loss_breakdown["coarse_bce"] = 0.0
                    loss_breakdown["coarse_weighted"] = 0.0

                cdmoe_term = regularization_terms["cdmoe"].mean()
                cdmoe_sparsity_term = regularization_terms["cdmoe_sparsity"].mean()
                cdmoe_capacity_term = regularization_terms["cdmoe_capacity"].mean()
                cdmoe_selector_capacity_term = regularization_terms[
                    "cdmoe_selector_capacity"
                ].mean()
                cdmoe_balance_term = regularization_terms["cdmoe_balance"].mean()
                total_loss = total_loss + cdmoe_term
                loss_breakdown["total"] += cdmoe_term.detach().item()
                loss_breakdown["cdmoe"] = cdmoe_term.detach().item()
                loss_breakdown["cdmoe_sparsity"] = cdmoe_sparsity_term.detach().item()
                loss_breakdown["cdmoe_capacity"] = cdmoe_capacity_term.detach().item()
                loss_breakdown["cdmoe_selector_capacity"] = (
                    cdmoe_selector_capacity_term.detach().item()
                )
                loss_breakdown["cdmoe_balance"] = cdmoe_balance_term.detach().item()

                running_val_loss += total_loss.item()
                for key in val_loss_totals.keys():
                    val_loss_totals[key] += loss_breakdown[key]
                progress_bar.set_description(f"Test Loss: {running_val_loss / (iteration + 1):.4f}")

                target_array = self._tensor_to_numpy(targets)
                prediction_array = self._tensor_to_numpy((prediction_prob > 0.5).byte())

                hausdorff_sum["WT"] += hausdorff_95(prediction_array[:, 0], target_array[:, 0])
                hausdorff_sum["TC"] += hausdorff_95(prediction_array[:, 1], target_array[:, 1])
                hausdorff_sum["ET"] += hausdorff_95(prediction_array[:, 2], target_array[:, 2])

                self.evaluator_wt.update(prediction_array[:, 0, None], target_array[:, 0, None])
                self.evaluator_tc.update(prediction_array[:, 1, None], target_array[:, 1, None])
                self.evaluator_et.update(prediction_array[:, 2, None], target_array[:, 2, None])

        progress_bar.close()

        self.last_test_loss_breakdown = self._average_loss_meter(val_loss_totals, len(self.val_loader))
        self._log_metrics(epoch, start_time, running_val_loss, hausdorff_sum)

    @staticmethod
    def _boundary_target(target):
        target = target.float()
        max_pool = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
        min_pool = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
        return (max_pool - min_pool).clamp(0.0, 1.0)

    @staticmethod
    def _foreground_false_negative_penalty(prob, target):
        foreground = target.detach().float()
        missed_foreground = (foreground * (1.0 - prob).clamp(0.0, 1.0)).sum(dim=(0, 2, 3))
        foreground_count = foreground.sum(dim=(0, 2, 3))
        valid_classes = foreground_count > 0
        if not torch.any(valid_classes):
            return prob.new_zeros(())
        class_penalty = missed_foreground[valid_classes] / foreground_count[valid_classes].clamp_min(1.0)
        return class_penalty.mean()

    @staticmethod
    def _hierarchy_risk_map(prob):
        et_prob = prob[:, 2:3]
        tc_prob = prob[:, 1:2]
        wt_prob = prob[:, 0:1]
        tc_over_wt = F.relu(tc_prob - wt_prob)
        et_over_tc = F.relu(et_prob - tc_prob)
        return torch.cat([
            tc_over_wt,
            torch.maximum(tc_over_wt, et_over_tc),
            et_over_tc,
        ], dim=1).clamp(0.0, 1.0)

    def _prepare_meta_map(self, tensor, target_size):
        if tensor is None:
            return None
        tensor = self._sanitize_tensor(tensor)
        if tensor.shape[-2:] != target_size:
            tensor = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)
        return tensor

    def _compute_loss(
        self,
        logits,
        target,
        extra_meta=None,
        enable_fn_penalty=False,
        return_breakdown=False,
    ):
        logits = self._sanitize_tensor(logits)
        loss_dice = self.criterion_dice(logits, target)
        loss_bce = self.criterion_bce(logits, target)
        loss = loss_dice + loss_bce
        breakdown = {
            "total": 0.0,
            "dice": loss_dice.detach().item(),
            "bce": loss_bce.detach().item(),
            "fn": 0.0,
            "uncertainty": 0.0,
        }

        prob = torch.sigmoid(logits)
        if enable_fn_penalty and self.fn_penalty_weight > 0:
            fn_penalty = self._foreground_false_negative_penalty(prob, target)
            fn_term = self.fn_penalty_weight * fn_penalty
            loss = loss + fn_term
            breakdown["fn"] = fn_term.detach().item()

        hierarchy_confusion = None
        if extra_meta:
            hierarchy_confusion = self._prepare_meta_map(
                extra_meta.get("hierarchy_confusion"),
                target_size=prob.shape[-2:],
            )

        if extra_meta:
            uncertainty_map = self._prepare_meta_map(
                extra_meta.get("dce_uncertainty"),
                target_size=prob.shape[-2:],
            )
            if uncertainty_map is not None:
                coarse_probability = self._prepare_meta_map(
                    extra_meta.get("coarse_probability"),
                    target_size=target.shape[-2:],
                )
                calibrated_disagreement = self._prepare_meta_map(
                    extra_meta.get("calibrated_disagreement"),
                    target_size=target.shape[-2:],
                )
                if coarse_probability is None:
                    coarse_probability = prob.detach()
                if calibrated_disagreement is None:
                    calibrated_disagreement = torch.ones_like(coarse_probability)

                coarse_probability = coarse_probability.detach().clamp(0.0, 1.0)
                calibrated_disagreement = calibrated_disagreement.detach().clamp(0.0, 1.0)
                coarse_error = torch.abs(coarse_probability - target).clamp(0.0, 1.0)
                boundary_target = self._boundary_target(target).detach()
                disagreement_boundary = boundary_target * calibrated_disagreement
                if hierarchy_confusion is None:
                    hierarchy_confusion = self._hierarchy_risk_map(coarse_probability)
                hierarchy_risk = hierarchy_confusion.detach().clamp(0.0, 1.0)
                uncertainty_target = (
                    self.dcte_error_target_weight * coarse_error +
                    self.dcte_boundary_target_weight * disagreement_boundary +
                    self.dcte_hierarchy_target_weight * hierarchy_risk
                ).clamp(0.0, 1.0)
                uncertainty_weight = 1.0 + target.detach() + disagreement_boundary + hierarchy_risk
                uncertainty_loss_map = F.binary_cross_entropy(
                    uncertainty_map.clamp(1e-6, 1.0 - 1e-6),
                    uncertainty_target,
                    reduction="none",
                )
                uncertainty_term = self.uncertainty_weight * (uncertainty_loss_map * uncertainty_weight).mean()
                loss = loss + uncertainty_term
                breakdown["uncertainty"] = uncertainty_term.detach().item()

        breakdown["total"] = loss.detach().item()
        if return_breakdown:
            return loss, breakdown
        return loss

    def _log_metrics(self, epoch, start_time, test_loss, hausdorff_sum):
        epoch_seconds = int(time.time() - start_time)
        test_m, test_s = divmod(epoch_seconds, 60)
        test_time_str = f"{test_m}m{test_s:02d}s"
        avg_test_loss = test_loss / len(self.val_loader)

        summary = (
            f"[Train Loss: {self.last_train_loss:.4f}, Time: {self.last_train_time_str}]   "
            f"[Test Loss: {avg_test_loss:.4f}, Time: {test_time_str}]"
        )
        print(summary)
        logging.info(summary)

        d_wt, d_tc, d_et = self.evaluator_wt.dice(), self.evaluator_tc.dice(), self.evaluator_et.dice()
        d_avg = (d_wt + d_tc + d_et) / 3
        i_wt, i_tc, i_et = self.evaluator_wt.IoU(), self.evaluator_tc.IoU(), self.evaluator_et.IoU()
        i_avg = (i_wt + i_tc + i_et) / 3
        s_wt, s_tc, s_et = (
            self.evaluator_wt.sensitivity(),
            self.evaluator_tc.sensitivity(),
            self.evaluator_et.sensitivity(),
        )
        s_avg = (s_wt + s_tc + s_et) / 3

        num = len(self.val_loader)
        h_wt = hausdorff_sum["WT"] / num
        h_tc = hausdorff_sum["TC"] / num
        h_et = hausdorff_sum["ET"] / num
        h_avg = (h_wt + h_tc + h_et) / 3

        metrics_str = (
            f"Dice: WT: {d_wt:.4f}, TC: {d_tc:.4f}, ET: {d_et:.4f}, Avg: {d_avg:.4f}.\n"
            f"IoU: WT: {i_wt:.4f}, TC: {i_tc:.4f}, ET: {i_et:.4f}, Avg: {i_avg:.4f}.\n"
            f"Sensitivity: WT: {s_wt:.4f}, TC: {s_tc:.4f}, ET: {s_et:.4f}, Avg: {s_avg:.4f}.\n"
            f"Hausdorff: WT: {h_wt:.4f}, TC: {h_tc:.4f}, ET: {h_et:.4f}, Avg: {h_avg:.4f}."
        )

        print(metrics_str + "\n")
        logging.info(metrics_str + "\n")

        self.loss_recorder.record_epoch(
            epoch + 1,
            self.last_train_loss_breakdown,
            self.last_test_loss_breakdown,
        )

        time.sleep(0.5)

        if d_avg > self.best_dice_pre:
            self.best_dice_pre = d_avg
            self.best_dice_epoch = epoch + 1
            self.best_epoch_details = metrics_str

        if i_avg > self.best_iou_pre:
            self.best_iou_pre = i_avg
            self.best_iou_epoch = epoch + 1

        self._save_epoch_weights(epoch, d_avg, s_avg)
