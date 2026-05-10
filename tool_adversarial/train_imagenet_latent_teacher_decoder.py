#!/usr/bin/env python3
"""Train a shared latent perturbation decoder from a teacher-bank dataset.

This script consumes the output of
``build_imagenet_adversarial_teacher_bank.py`` and trains:

- one shared ``LatentGridDecoder``
- one learned latent code per training sample

The goal is to distill saved adversarial perturbations into a compact latent
space that can later be fit/refined for new images.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src.ml_setup.factory import get_ml_setup_from_config
from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup_model.pretrained_models import create_torchvision_model
from py_src.model_opti_save_load import load_model_state_file


LOGGER = logging.getLogger("train_imagenet_latent_teacher_decoder")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_DATASETS = ("imagenet1k", "imagenet100", "imagenet10")


@dataclass
class TeacherBankRecord:
    split: str
    source_index: int
    source_label: int
    target_label: Optional[int]
    source_path: Optional[str]
    selected_attack_name: str
    selected_adv_path: str
    selection_rule: str
    pred_label: int
    clean_bank_path: Path
    teacher_adv_bank_path: Path
    latent_index: int = -1


@dataclass
class LossWeights:
    recon: float
    logit: float
    margin: float
    attack: float
    l2: float
    tv: float
    latent: float


class TeacherBankDataset(Dataset):
    def __init__(self, records: list[TeacherBankRecord], *, latent_index_offset: int = 0, image_transform: Optional[transforms.Compose] = None) -> None:
        self.records = records
        self.latent_index_offset = latent_index_offset
        self.image_transform = image_transform or build_teacher_bank_image_transform()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        clean_image = load_teacher_bank_image(record.clean_bank_path, self.image_transform)
        teacher_adv = load_teacher_bank_image(record.teacher_adv_bank_path, self.image_transform)
        source_label = record.source_label
        target_label_raw = record.target_label if record.target_label is not None else -1
        teacher_pred_label = record.pred_label
        target_label = target_label_raw if target_label_raw >= 0 else -1
        latent_index = record.latent_index if record.latent_index >= 0 else index + self.latent_index_offset
        return (
            latent_index,
            clean_image,
            teacher_adv,
            source_label,
            target_label,
            teacher_pred_label,
        )


def build_teacher_bank_image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_teacher_bank_image(path: Path, image_transform: transforms.Compose) -> torch.Tensor:
    with Image.open(path) as image:
        return image_transform(image.convert("RGB")) # type: ignore


class LatentGridDecoder(nn.Module):
    def __init__(
        self,
        grid_size: int,
        scale: float,
        latent_channels: int = 3,
        hidden_channels: int = 24,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.scale = scale
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.refine = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, 1),
        )

    def forward(self, latent: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
        upsampled = F.interpolate(latent, size=image_hw, mode="bilinear", align_corners=False)
        refined = self.refine(upsampled)
        return self.scale * torch.tanh(upsampled + refined)


class PerSampleLatentBank(nn.Module):
    def __init__(self, num_samples: int, channels: int, grid_size: int, init_scale: float = 1e-3) -> None:
        super().__init__()
        self.latents = nn.Parameter(init_scale * torch.randn(num_samples, channels, grid_size, grid_size))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.latents[indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a latent perturbation decoder from a teacher-bank dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_root", type=str, help="Either the teacher_bank folder or the parent adversarial root that contains a teacher_bank subfolder.")
    parser.add_argument("--output-dir", type=str, default=None, help="Folder for checkpoints and reports. Defaults to <teacher_bank_root>/latent_decoder.")
    parser.add_argument("--teacher-manifest-path", type=str, default=None, help="Optional path to selected_teacher_manifest.jsonl. If omitted, it is discovered from the input root.")
    parser.add_argument("--model-type", type=str, default=None, help="LLR2 model type name. If omitted, it will be inferred from --checkpoint when possible.")
    parser.add_argument("--dataset-type", type=str, default="imagenet1k", choices=SUPPORTED_DATASETS, help="ImageNet dataset variant used to rebuild the classifier architecture.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional LLR2 .model.pt classifier checkpoint used for the original attacks.")
    parser.add_argument("--use-torchvision-pretrained", action="store_true", help="Use torchvision ImageNet pretrained weights instead of an LLR2 checkpoint.")
    parser.add_argument("--preset", type=int, default=0, help="LLR2 ImageNet preset used when rebuilding the classifier architecture.")
    parser.add_argument("--holdout-fraction", type=float, default=0.1, help="Validation fraction used only when the teacher-bank manifest does not contain an explicit train split.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for data splitting and initialization.")
    parser.add_argument("--grid-size", type=int, default=20, help="Latent grid size.")
    parser.add_argument("--latent-channels", type=int, default=3, help="Number of channels in the learned latent grid.")
    parser.add_argument("--hidden-channels", type=int, default=24, help="Hidden channels inside the shared latent decoder.")
    parser.add_argument("--decoder-scale", type=str, default="auto", help="Scale used for decoder output in normalized space. Use 'auto' to infer it from the training deltas.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="Training dataloader workers for the image-pair teacher bank.")
    parser.add_argument("--lr-decoder", type=float, default=1e-3, help="Learning rate for the shared decoder.")
    parser.add_argument("--lr-latent", type=float, default=8e-2, help="Learning rate for the train-set latent bank.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for both optimizer parameter groups.")
    parser.add_argument("--supervision-mode", type=str, default="target_if_available_else_teacher_pred", choices=("teacher_pred", "target_if_available_else_teacher_pred"), help="How to choose class-label supervision for the attack-alignment loss.")
    parser.add_argument("--lambda-recon", type=float, default=8.0)
    parser.add_argument("--lambda-logit", type=float, default=1.0)
    parser.add_argument("--lambda-margin", type=float, default=0.5)
    parser.add_argument("--lambda-attack", type=float, default=1.0)
    parser.add_argument("--lambda-l2", type=float, default=30.0)
    parser.add_argument("--lambda-tv", type=float, default=0.8)
    parser.add_argument("--lambda-latent", type=float, default=1e-3)
    parser.add_argument("--fit-val-steps", type=int, default=0, help="If > 0, fit held-out validation latents after training for this many steps per batch.")
    parser.add_argument("--fit-val-lr", type=float, default=0.12, help="Learning rate for held-out validation latent fitting.")
    parser.add_argument("--device", type=str, default="auto", help="Computation device: auto, cuda, cpu, or cuda:0 style values.")
    parser.add_argument("--log-every", type=int, default=50, help="Emit a batch-progress log every N train batches. Use 0 to disable batch logs.")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.use_torchvision_pretrained and args.checkpoint:
        raise ValueError("choose either --checkpoint or --use-torchvision-pretrained, not both")
    if not args.checkpoint and not args.use_torchvision_pretrained:
        raise ValueError("provide either --checkpoint or --use-torchvision-pretrained")
    if args.holdout_fraction < 0 or args.holdout_fraction >= 1:
        raise ValueError("--holdout-fraction must be in [0, 1)")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.grid_size <= 0:
        raise ValueError("--grid-size must be positive")
    if args.latent_channels <= 0:
        raise ValueError("--latent-channels must be positive")
    if args.hidden_channels <= 0:
        raise ValueError("--hidden-channels must be positive")
    if args.fit_val_steps < 0:
        raise ValueError("--fit-val-steps cannot be negative")
    if args.fit_val_lr <= 0:
        raise ValueError("--fit-val-lr must be positive")
    if args.log_every < 0:
        raise ValueError("--log-every cannot be negative")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--:--"
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalized_pixel_bounds(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    low = (torch.zeros_like(mean) - mean) / std
    high = (torch.ones_like(mean) - mean) / std
    return low, high


def clamp_normalized(images: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.minimum(images, high), low)


def total_variation(delta: torch.Tensor) -> torch.Tensor:
    tv_h = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
    tv_w = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


def forward_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    outputs = model(inputs)
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    if not torch.is_tensor(outputs):
        raise TypeError(f"model forward returned unsupported type: {type(outputs)!r}")
    return outputs

def resolve_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, str, str]:
    checkpoint_model_type = None
    checkpoint_dataset_type = None
    state_dict = None

    if args.checkpoint:
        state_dict, checkpoint_model_type, checkpoint_dataset_type = load_model_state_file(args.checkpoint)

    resolved_model_type = args.model_type or checkpoint_model_type
    if resolved_model_type is None:
        raise ValueError("unable to infer model type; pass --model-type or use a checkpoint that stores model_name")

    resolved_dataset_type = args.dataset_type
    if checkpoint_dataset_type is not None and checkpoint_dataset_type != resolved_dataset_type:
        raise ValueError(
            f"checkpoint dataset type {checkpoint_dataset_type!r} does not match --dataset-type "
            f"{resolved_dataset_type!r}"
        )

    ml_setup = get_ml_setup_from_config(
        resolved_model_type,
        resolved_dataset_type,
        preset=args.preset,
        device=device,
    )

    if args.use_torchvision_pretrained:
        if resolved_dataset_type != DatasetType.imagenet1k.name:
            raise ValueError("--use-torchvision-pretrained currently expects --dataset-type imagenet1k")
        model = create_torchvision_model(resolved_model_type)
    else:
        model = ml_setup.model
        assert state_dict is not None
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"failed to load checkpoint cleanly. missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    model = model.to(device)
    model.eval()
    return model, resolved_model_type, resolved_dataset_type


def resolve_teacher_bank_root(input_root: Path) -> Path:
    input_root = input_root.resolve()
    if (input_root / "selected_teacher_manifest.jsonl").exists():
        return input_root
    candidate = input_root / "teacher_bank"
    if (candidate / "selected_teacher_manifest.jsonl").exists():
        return candidate.resolve()
    raise FileNotFoundError(
        f"could not find selected_teacher_manifest.jsonl under {input_root} or {candidate}"
    )


def resolve_manifest_relative_path(value: Optional[str], manifest_root: Path) -> Optional[Path]:
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (manifest_root / candidate).resolve()


def load_teacher_bank_records(manifest_path: Path) -> list[TeacherBankRecord]:
    records: list[TeacherBankRecord] = []
    manifest_root = manifest_path.resolve().parent
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            clean_bank_path = resolve_manifest_relative_path(row.get("clean_bank_path"), manifest_root)
            teacher_adv_bank_path = resolve_manifest_relative_path(row.get("teacher_adv_bank_path"), manifest_root)
            if clean_bank_path is None or teacher_adv_bank_path is None:
                raise ValueError(
                    "manifest row is missing clean_bank_path or teacher_adv_bank_path. "
                    f"Expected an image-pair teacher bank row: {row}"
                )
            records.append(
                TeacherBankRecord(
                    split=str(row.get("split", "unknown")),
                    source_index=int(row["source_index"]),
                    source_label=int(row["source_label"]),
                    target_label=int(row["target_label"]) if row.get("target_label") is not None else None,
                    source_path=row.get("source_path"),
                    selected_attack_name=str(row["selected_attack_name"]),
                    selected_adv_path=str(row["selected_adv_path"]),
                    selection_rule=str(row["selection_rule"]),
                    pred_label=int(row["pred_label"]),
                    clean_bank_path=clean_bank_path,
                    teacher_adv_bank_path=teacher_adv_bank_path,
                )
            )
    return records


def split_records(
    records: list[TeacherBankRecord],
    holdout_fraction: float,
    seed: int,
) -> tuple[list[TeacherBankRecord], list[TeacherBankRecord], str]:
    explicit_train = [record for record in records if record.split == "train"]
    explicit_val = [record for record in records if record.split == "val"]

    if explicit_train:
        return explicit_train, explicit_val, "manifest_train_val"

    rows = list(records)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(rows), generator=generator).tolist()
    shuffled = [rows[idx] for idx in permutation]
    val_count = 0
    if holdout_fraction > 0 and len(shuffled) > 1:
        val_count = max(1, int(round(len(shuffled) * holdout_fraction)))
        val_count = min(val_count, len(shuffled) - 1)
    train_rows = shuffled[val_count:]
    val_rows = shuffled[:val_count]
    return train_rows, val_rows, "random_holdout"


def assign_latent_indices(records: list[TeacherBankRecord]) -> None:
    for latent_index, record in enumerate(records):
        record.latent_index = latent_index


def load_teacher_bank_summary(teacher_bank_root: Path) -> dict[str, Any]:
    summary_path = teacher_bank_root / "teacher_bank_summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_decoder_scale(
    decoder_scale_arg: str,
    train_records: list[TeacherBankRecord],
    image_transform: transforms.Compose,
    teacher_bank_summary: dict[str, Any],
) -> float:
    if decoder_scale_arg != "auto":
        return float(decoder_scale_arg)

    summary_max_abs = teacher_bank_summary.get("normalized_delta_abs_max")
    if summary_max_abs is not None:
        summary_max_abs = float(summary_max_abs)
        if summary_max_abs > 0:
            return summary_max_abs

    max_abs = 0.0
    for record in train_records:
        clean_image = load_teacher_bank_image(record.clean_bank_path, image_transform)
        teacher_adv = load_teacher_bank_image(record.teacher_adv_bank_path, image_transform)
        delta = teacher_adv - clean_image
        max_abs = max(max_abs, float(delta.abs().max().item()))

    if max_abs <= 0:
        raise ValueError("could not infer a positive decoder scale from the teacher bank")
    return max_abs


def build_supervision_labels(
    target_labels: torch.Tensor,
    teacher_pred_labels: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "teacher_pred":
        return teacher_pred_labels
    if mode == "target_if_available_else_teacher_pred":
        target_mask = target_labels.ge(0)
        return torch.where(target_mask, target_labels, teacher_pred_labels)
    raise ValueError(f"unsupported supervision mode: {mode}")


def per_sample_margin(
    logits: torch.Tensor,
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
) -> torch.Tensor:
    source_values = logits.gather(1, source_labels.unsqueeze(1)).squeeze(1)
    target_values = logits.gather(1, target_labels.unsqueeze(1)).squeeze(1)
    return source_values - target_values


def distillation_losses(
    classifier: nn.Module,
    decoder: LatentGridDecoder,
    latents: torch.Tensor,
    images: torch.Tensor,
    teacher_advs: torch.Tensor,
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
    teacher_pred_labels: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    device: torch.device,
    weights: LossWeights,
    supervision_mode: str,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    images = images.to(device)
    teacher_advs = teacher_advs.to(device)
    source_labels = source_labels.to(device)
    target_labels = target_labels.to(device)
    teacher_pred_labels = teacher_pred_labels.to(device)

    teacher_delta = teacher_advs - images
    decoded_delta = decoder(latents, (images.size(2), images.size(3)))
    advs = clamp_normalized(images + decoded_delta, low, high)

    logits = forward_logits(classifier, advs)
    with torch.no_grad():
        teacher_logits = forward_logits(classifier, teacher_advs)

    supervised_labels = build_supervision_labels(target_labels, teacher_pred_labels, supervision_mode)

    recon_loss = F.mse_loss(decoded_delta, teacher_delta)
    logit_loss = F.mse_loss(logits, teacher_logits)
    margin_loss = F.mse_loss(
        per_sample_margin(logits, source_labels, supervised_labels),
        per_sample_margin(teacher_logits, source_labels, supervised_labels),
    )
    attack_loss = F.cross_entropy(logits, supervised_labels)
    l2_loss = decoded_delta.pow(2).mean()
    tv_loss = total_variation(decoded_delta)
    latent_loss = latents.pow(2).mean()

    loss = (
        weights.recon * recon_loss
        + weights.logit * logit_loss
        + weights.margin * margin_loss
        + weights.attack * attack_loss
        + weights.l2 * l2_loss
        + weights.tv * tv_loss
        + weights.latent * latent_loss
    )

    with torch.no_grad():
        probs = logits.softmax(dim=1)
        preds = logits.argmax(dim=1)
        target_known_mask = target_labels.ge(0)
        metrics = {
            "loss": float(loss.item()),
            "recon_loss": float(recon_loss.item()),
            "logit_loss": float(logit_loss.item()),
            "margin_loss": float(margin_loss.item()),
            "attack_loss": float(attack_loss.item()),
            "l2_loss": float(l2_loss.item()),
            "tv_loss": float(tv_loss.item()),
            "latent_loss": float(latent_loss.item()),
            "recon_term": float((weights.recon * recon_loss).item()),
            "logit_term": float((weights.logit * logit_loss).item()),
            "margin_term": float((weights.margin * margin_loss).item()),
            "attack_term": float((weights.attack * attack_loss).item()),
            "l2_term": float((weights.l2 * l2_loss).item()),
            "tv_term": float((weights.tv * tv_loss).item()),
            "latent_term": float((weights.latent * latent_loss).item()),
            "delta_rms": float(decoded_delta.pow(2).mean().sqrt().item()),
            "teacher_match_rate": float((preds == teacher_pred_labels).float().mean().item()),
            "source_leave_rate": float((preds != source_labels).float().mean().item()),
            "supervised_label_rate": float((preds == supervised_labels).float().mean().item()),
            "mean_supervised_confidence": float(
                probs.gather(1, supervised_labels.unsqueeze(1)).mean().item()
            ),
        }
        if target_known_mask.any():
            target_hits = preds[target_known_mask] == target_labels[target_known_mask]
            metrics["target_rate_known"] = float(target_hits.float().mean().item())
        else:
            metrics["target_rate_known"] = float("nan")
    return loss, metrics, advs.detach()


def aggregate_metric_sums(
    running_sums: dict[str, float],
    metrics: dict[str, float],
    batch_size: int,
) -> None:
    for key, value in metrics.items():
        if math.isnan(value):
            continue
        running_sums[key] = running_sums.get(key, 0.0) + value * batch_size


def finalize_metric_sums(running_sums: dict[str, float], total: int) -> dict[str, float]:
    return {key: value / total for key, value in running_sums.items()}


def format_loss_report(metrics: dict[str, float]) -> str:
    return (
        f"loss={metrics.get('loss', float('nan')):.4f} "
        f"recon_term={metrics.get('recon_term', float('nan')):.4f} "
        f"logit_term={metrics.get('logit_term', float('nan')):.4f} "
        f"margin_term={metrics.get('margin_term', float('nan')):.4f} "
        f"attack_term={metrics.get('attack_term', float('nan')):.4f} "
        f"l2_term={metrics.get('l2_term', float('nan')):.4f} "
        f"tv_term={metrics.get('tv_term', float('nan')):.4f} "
        f"latent_term={metrics.get('latent_term', float('nan')):.4f}"
    )


def train_decoder(
    classifier: nn.Module,
    decoder: LatentGridDecoder,
    latent_bank: PerSampleLatentBank,
    loader: DataLoader,
    low: torch.Tensor,
    high: torch.Tensor,
    device: torch.device,
    weights: LossWeights,
    supervision_mode: str,
    epochs: int,
    lr_decoder: float,
    lr_latent: float,
    weight_decay: float,
    log_every: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(
        [
            {"params": decoder.parameters(), "lr": lr_decoder, "weight_decay": weight_decay},
            {"params": latent_bank.parameters(), "lr": lr_latent, "weight_decay": weight_decay},
        ]
    )
    total_steps = max(1, epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        decoder.train()
        latent_bank.train()
        sums: dict[str, float] = {}
        total = 0

        for batch_index, batch in enumerate(loader, start=1):
            latent_indices, images, teacher_advs, source_labels, target_labels, teacher_pred_labels = batch
            latents = latent_bank(latent_indices.to(device))
            loss, metrics, _ = distillation_losses(
                classifier=classifier,
                decoder=decoder,
                latents=latents,
                images=images,
                teacher_advs=teacher_advs,
                source_labels=source_labels,
                target_labels=target_labels,
                teacher_pred_labels=teacher_pred_labels,
                low=low,
                high=high,
                device=device,
                weights=weights,
                supervision_mode=supervision_mode,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_size = int(images.size(0))
            total += batch_size
            aggregate_metric_sums(sums, metrics, batch_size)

            if log_every > 0 and batch_index % log_every == 0:
                LOGGER.info(
                    "epoch %d batch %d: %s leave=%.3f teacher_match=%.3f decoder_lr=%.6g latent_lr=%.6g",
                    epoch,
                    batch_index,
                    format_loss_report(metrics),
                    metrics["source_leave_rate"],
                    metrics["teacher_match_rate"],
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"],
                )

        row = {
            "epoch": epoch,
            **finalize_metric_sums(sums, total),
            "epoch_seconds": time.time() - epoch_start,
            "decoder_lr": optimizer.param_groups[0]["lr"],
            "latent_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        LOGGER.info(
            "epoch %d/%d finished in %s | %s leave=%.3f teacher_match=%.3f decoder_lr=%.6g latent_lr=%.6g",
            epoch,
            epochs,
            format_duration(row["epoch_seconds"]),
            format_loss_report(row),
            row["source_leave_rate"],
            row["teacher_match_rate"],
            row["decoder_lr"],
            row["latent_lr"],
        )
    return history


def evaluate_with_latent_bank(
    classifier: nn.Module,
    decoder: LatentGridDecoder,
    latent_bank: PerSampleLatentBank,
    loader: DataLoader,
    low: torch.Tensor,
    high: torch.Tensor,
    device: torch.device,
    weights: LossWeights,
    supervision_mode: str,
) -> dict[str, float]:
    decoder.eval()
    latent_bank.eval()
    sums: dict[str, float] = {}
    total = 0

    with torch.no_grad():
        for batch in loader:
            latent_indices, images, teacher_advs, source_labels, target_labels, teacher_pred_labels = batch
            latents = latent_bank(latent_indices.to(device))
            _, metrics, _ = distillation_losses(
                classifier=classifier,
                decoder=decoder,
                latents=latents,
                images=images,
                teacher_advs=teacher_advs,
                source_labels=source_labels,
                target_labels=target_labels,
                teacher_pred_labels=teacher_pred_labels,
                low=low,
                high=high,
                device=device,
                weights=weights,
                supervision_mode=supervision_mode,
            )
            batch_size = int(images.size(0))
            total += batch_size
            aggregate_metric_sums(sums, metrics, batch_size)

    if total == 0:
        return {}
    return finalize_metric_sums(sums, total)


def fit_latents_for_loader(
    classifier: nn.Module,
    decoder: LatentGridDecoder,
    loader: DataLoader,
    low: torch.Tensor,
    high: torch.Tensor,
    device: torch.device,
    weights: LossWeights,
    supervision_mode: str,
    grid_size: int,
    latent_channels: int,
    fit_steps: int,
    fit_lr: float,
) -> dict[str, float]:
    if fit_steps <= 0:
        return {}

    decoder.eval()
    sums: dict[str, float] = {}
    total = 0

    for batch_index, batch in enumerate(loader, start=1):
        _, images, teacher_advs, source_labels, target_labels, teacher_pred_labels = batch
        latents = torch.zeros(
            (images.size(0), latent_channels, grid_size, grid_size),
            device=device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([latents], lr=fit_lr)

        for _ in range(fit_steps):
            loss, _, _ = distillation_losses(
                classifier=classifier,
                decoder=decoder,
                latents=latents,
                images=images,
                teacher_advs=teacher_advs,
                source_labels=source_labels,
                target_labels=target_labels,
                teacher_pred_labels=teacher_pred_labels,
                low=low,
                high=high,
                device=device,
                weights=weights,
                supervision_mode=supervision_mode,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            _, metrics, _ = distillation_losses(
                classifier=classifier,
                decoder=decoder,
                latents=latents,
                images=images,
                teacher_advs=teacher_advs,
                source_labels=source_labels,
                target_labels=target_labels,
                teacher_pred_labels=teacher_pred_labels,
                low=low,
                high=high,
                device=device,
                weights=weights,
                supervision_mode=supervision_mode,
            )

        batch_size = int(images.size(0))
        total += batch_size
        aggregate_metric_sums(sums, metrics, batch_size)
        LOGGER.info(
            "val-fit batch %d: %s leave=%.3f teacher_match=%.3f",
            batch_index,
            format_loss_report(metrics),
            metrics["source_leave_rate"],
            metrics["teacher_match_rate"],
        )

    if total == 0:
        return {}
    return finalize_metric_sums(sums, total)


def save_training_checkpoint(
    path: Path,
    decoder: LatentGridDecoder,
    latent_bank: PerSampleLatentBank,
    args: argparse.Namespace,
    resolved_model_type: str,
    resolved_dataset_type: str,
    decoder_scale: float,
    train_count: int,
    val_count: int,
) -> None:
    payload = {
        "decoder_state_dict": decoder.state_dict(),
        "latent_bank_state_dict": latent_bank.state_dict(),
        "decoder_config": {
            "grid_size": args.grid_size,
            "latent_channels": args.latent_channels,
            "hidden_channels": args.hidden_channels,
            "scale": decoder_scale,
        },
        "training_config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "classifier_model_type": resolved_model_type,
        "classifier_dataset_type": resolved_dataset_type,
        "train_count": train_count,
        "val_count": val_count,
        "saved_at_unix": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    setup_logging()
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    device = resolve_device(args.device)
    input_root = Path(args.input_root).resolve()
    teacher_bank_root = resolve_teacher_bank_root(input_root)
    teacher_manifest_path = (
        Path(args.teacher_manifest_path).resolve()
        if args.teacher_manifest_path
        else teacher_bank_root / "selected_teacher_manifest.jsonl"
    )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else teacher_bank_root / "latent_decoder"
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_bank_summary = load_teacher_bank_summary(teacher_bank_root)
    image_transform = build_teacher_bank_image_transform()

    all_records = load_teacher_bank_records(teacher_manifest_path)
    if not all_records:
        raise ValueError(f"no teacher-bank records found in {teacher_manifest_path}")

    train_records, val_records, split_strategy = split_records(
        all_records,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    if not train_records:
        raise ValueError("no training records available after splitting")
    assign_latent_indices(train_records)

    decoder_scale = resolve_decoder_scale(
        args.decoder_scale,
        train_records,
        image_transform,
        teacher_bank_summary,
    )
    classifier, resolved_model_type, resolved_dataset_type = resolve_model(args, device)
    low, high = normalized_pixel_bounds(device)
    weights = LossWeights(
        recon=args.lambda_recon,
        logit=args.lambda_logit,
        margin=args.lambda_margin,
        attack=args.lambda_attack,
        l2=args.lambda_l2,
        tv=args.lambda_tv,
        latent=args.lambda_latent,
    )

    train_dataset = TeacherBankDataset(
        train_records,
        latent_index_offset=0,
        image_transform=image_transform,
    )
    val_dataset = TeacherBankDataset(
        val_records,
        latent_index_offset=0,
        image_transform=image_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    decoder = LatentGridDecoder(
        grid_size=args.grid_size,
        scale=decoder_scale,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
    ).to(device)
    latent_bank = PerSampleLatentBank(
        num_samples=len(train_dataset),
        channels=args.latent_channels,
        grid_size=args.grid_size,
    ).to(device)

    LOGGER.info(
        "training with %d train records and %d validation records | split_strategy=%s | decoder_scale=%.6f",
        len(train_dataset),
        len(val_dataset),
        split_strategy,
        decoder_scale,
    )

    run_config = {
        "input_root": str(input_root),
        "teacher_bank_root": str(teacher_bank_root),
        "teacher_manifest_path": str(teacher_manifest_path),
        "output_dir": str(output_dir),
        "classifier_model_type": resolved_model_type,
        "classifier_dataset_type": resolved_dataset_type,
        "device": str(device),
        "teacher_bank_storage_mode": "image_pairs_png",
        "split_strategy": split_strategy,
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "decoder_scale": decoder_scale,
        "scheduler": {"name": "CosineAnnealingLR", "t_max_steps": max(1, args.epochs * len(train_loader)), "eta_min": 0.0},
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(output_dir / "train_config.json", run_config)

    train_history = train_decoder(
        classifier=classifier,
        decoder=decoder,
        latent_bank=latent_bank,
        loader=train_loader,
        low=low,
        high=high,
        device=device,
        weights=weights,
        supervision_mode=args.supervision_mode,
        epochs=args.epochs,
        lr_decoder=args.lr_decoder,
        lr_latent=args.lr_latent,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
    )

    final_train_metrics = evaluate_with_latent_bank(
        classifier=classifier,
        decoder=decoder,
        latent_bank=latent_bank,
        loader=train_eval_loader,
        low=low,
        high=high,
        device=device,
        weights=weights,
        supervision_mode=args.supervision_mode,
    )
    val_fit_metrics = fit_latents_for_loader(
        classifier=classifier,
        decoder=decoder,
        loader=val_loader,
        low=low,
        high=high,
        device=device,
        weights=weights,
        supervision_mode=args.supervision_mode,
        grid_size=args.grid_size,
        latent_channels=args.latent_channels,
        fit_steps=args.fit_val_steps,
        fit_lr=args.fit_val_lr,
    ) if len(val_dataset) > 0 and args.fit_val_steps > 0 else {}

    checkpoint_path = output_dir / "latent_teacher_decoder_checkpoint.pt"
    save_training_checkpoint(
        path=checkpoint_path,
        decoder=decoder,
        latent_bank=latent_bank,
        args=args,
        resolved_model_type=resolved_model_type,
        resolved_dataset_type=resolved_dataset_type,
        decoder_scale=decoder_scale,
        train_count=len(train_dataset),
        val_count=len(val_dataset),
    )

    report = {
        "input_root": str(input_root),
        "teacher_bank_root": str(teacher_bank_root),
        "teacher_manifest_path": str(teacher_manifest_path),
        "output_dir": str(output_dir),
        "classifier_model_type": resolved_model_type,
        "classifier_dataset_type": resolved_dataset_type,
        "device": str(device),
        "teacher_bank_storage_mode": "image_pairs_png",
        "split_strategy": split_strategy,
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "decoder_scale": decoder_scale,
        "loss_weights": vars(weights),
        "train_history": train_history,
        "final_train_metrics": final_train_metrics,
        "val_fit_metrics": val_fit_metrics,
        "selected_attack_counts_train": dict(
            sorted({name: sum(record.selected_attack_name == name for record in train_records) for name in {r.selected_attack_name for r in train_records}}.items())
        ),
        "selected_attack_counts_val": dict(
            sorted({name: sum(record.selected_attack_name == name for record in val_records) for name in {r.selected_attack_name for r in val_records}}.items())
        ),
        "saved_at_unix": time.time(),
    }
    write_json(output_dir / "latent_teacher_decoder_report.json", report)

    LOGGER.info("checkpoint written to %s", checkpoint_path)
    LOGGER.info("report written to %s", output_dir / "latent_teacher_decoder_report.json")
    LOGGER.info(
        "final train %s leave=%.3f teacher_match=%.3f",
        format_loss_report(final_train_metrics),
        final_train_metrics.get("source_leave_rate", float("nan")),
        final_train_metrics.get("teacher_match_rate", float("nan")),
    )
    if val_fit_metrics:
        LOGGER.info(
            "val-fit %s leave=%.3f teacher_match=%.3f",
            format_loss_report(val_fit_metrics),
            val_fit_metrics.get("source_leave_rate", float("nan")),
            val_fit_metrics.get("teacher_match_rate", float("nan")),
        )


if __name__ == "__main__":
    main()
