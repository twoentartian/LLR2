#!/usr/bin/env python3
"""Predict labels for saved ImageNet adversarial examples and summarize success.

This script is meant to evaluate image folders produced by
``generate_imagenet_adversarial_examples.py`` or similar tools.

It recursively scans a folder for images, predicts labels with an ImageNet
classifier, and computes success rates when source and/or target labels can be
inferred from either:

- a nearby ``manifest.jsonl`` file produced by the generator script, or
- the folder / filename layout:
  - ``<attack>/class_0005/*.png`` for source labels
  - ``*_target_0017.png`` for targeted labels

Example:

```bash
python3 tool_adversarial/predict_imagenet_adversarial_examples.py \
  /home/tyd/git/LLR2/artifacts/imagenet_adv \
  --model-type resnet50 \
  --use-torchvision-pretrained
```
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src.ml_setup.factory import get_ml_setup_from_config
from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup_model.pretrained_models import create_torchvision_model
from py_src.model_opti_save_load import load_model_state_file


LOGGER = logging.getLogger("predict_imagenet_adversarial_examples")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_DATASETS = ("imagenet1k", "imagenet100", "imagenet10")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

CLASS_DIR_PATTERN = re.compile(r"^class_(\d+)$")
INTEGER_DIR_PATTERN = re.compile(r"^(\d+)$")
TARGET_PATTERN = re.compile(r"_target_(\d+)(?=\.[^.]+$)")


@dataclass
class ImageRecord:
    image_path: Path
    attack_name: str
    source_label: Optional[int]
    target_label: Optional[int]
    source_index: Optional[int]
    source_path: Optional[str]


class ImagePredictionDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict labels for saved adversarial ImageNet examples and compute success rates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Folder containing saved adversarial images. Images are scanned recursively.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="LLR2 model type name. If omitted, it will be inferred from --checkpoint when possible.",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="imagenet1k",
        choices=SUPPORTED_DATASETS,
        help="ImageNet dataset variant used to build the checkpoint architecture.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional LLR2 .model.pt checkpoint to load.",
    )
    parser.add_argument(
        "--use-torchvision-pretrained",
        action="store_true",
        help="Use torchvision ImageNet pretrained weights instead of an LLR2 checkpoint.",
    )
    parser.add_argument(
        "--preset",
        type=int,
        default=0,
        help="LLR2 ImageNet preset used when building the model architecture.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional manifest.jsonl path from the generator script. If omitted, the script searches nearby parents.",
    )
    parser.add_argument(
        "--success-mode",
        type=str,
        default="auto",
        choices=("auto", "untargeted", "targeted", "none"),
        help="Which success criterion to emphasize in the summary. The script always reports both untargeted and "
             "targeted rates when labels are available.",
    )
    parser.add_argument(
        "--exclude-attack-names",
        nargs="*",
        default=[],
        help="Optional attack/group names to skip, for example: clean",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for prediction.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Dataloader workers.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional limit on the number of discovered images to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device: auto, cuda, cpu, or cuda:0 style values.",
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        default=None,
        help="Optional resize to apply before normalization. Leave unset for already-cropped adversarial images.",
    )
    parser.add_argument(
        "--center-crop-size",
        type=int,
        default=None,
        help="Optional center crop to apply before normalization.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Folder where prediction_results.jsonl and prediction_summary.json will be written. "
             "Defaults to the input folder.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Emit an INFO log every N prediction batches. Use 0 to disable periodic logs.",
    )
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
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")


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


class ProgressTracker:
    def __init__(self, total_items: int) -> None:
        self.total_items = total_items
        self.start_time = time.time()
        self.last_line_length = 0

    def clear(self) -> None:
        if self.last_line_length == 0:
            return
        sys.stderr.write("\r" + (" " * self.last_line_length) + "\r")
        sys.stderr.flush()
        self.last_line_length = 0

    def update(self, completed_items: int, stage: str) -> None:
        elapsed = time.time() - self.start_time
        rate = completed_items / elapsed if elapsed > 0 and completed_items > 0 else 0.0
        remaining = max(self.total_items - completed_items, 0)
        eta = remaining / rate if rate > 0 else None
        percent = 100.0 * completed_items / self.total_items if self.total_items > 0 else 100.0

        line = (
            f"[progress] {percent:5.1f}% | images {completed_items}/{self.total_items} | "
            f"elapsed {format_duration(elapsed)} | eta {format_duration(eta)} | {stage}"
        )
        padding = max(self.last_line_length - len(line), 0)
        sys.stderr.write("\r" + line + (" " * padding))
        sys.stderr.flush()
        self.last_line_length = len(line)

    def finish(self, completed_items: int, stage: str) -> None:
        self.update(completed_items, stage)
        sys.stderr.write("\n")
        sys.stderr.flush()
        self.last_line_length = 0


def find_default_manifest(input_dir: Path) -> Optional[Path]:
    candidate_parents = [input_dir, *list(input_dir.parents)[:3]]
    for parent in candidate_parents:
        candidate = parent / "manifest.jsonl"
        if candidate.exists():
            return candidate
    return None


def load_manifest_index(manifest_path: Optional[Path]) -> dict[Path, dict[str, Any]]:
    if manifest_path is None or not manifest_path.exists():
        return {}

    index: dict[Path, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            saved_path = row.get("saved_path")
            if not saved_path:
                continue
            index[Path(saved_path).resolve()] = row
    return index


def parse_label_from_part(part: str) -> Optional[int]:
    match = CLASS_DIR_PATTERN.match(part)
    if match:
        return int(match.group(1))
    match = INTEGER_DIR_PATTERN.match(part)
    if match:
        return int(match.group(1))
    return None


def infer_source_label(image_path: Path, input_dir: Path) -> Optional[int]:
    relative = image_path.relative_to(input_dir)
    for part in reversed(relative.parts[:-1]):
        label = parse_label_from_part(part)
        if label is not None:
            return label

    for part in reversed(image_path.parts[:-1]):
        label = parse_label_from_part(part)
        if label is not None:
            return label
    return None


def infer_target_label(image_path: Path) -> Optional[int]:
    match = TARGET_PATTERN.search(image_path.name)
    if match:
        return int(match.group(1))
    return None


def infer_attack_name(image_path: Path, input_dir: Path) -> str:
    relative = image_path.relative_to(input_dir)
    parts = relative.parts[:-1]
    if parts:
        first_label_index = None
        for idx, part in enumerate(parts):
            if parse_label_from_part(part) is not None:
                first_label_index = idx
                break
        if first_label_index is not None and first_label_index > 0:
            return parts[0]
        if first_label_index is None and parts[0]:
            return parts[0]

    if parse_label_from_part(input_dir.name) is not None:
        return input_dir.parent.name
    return input_dir.name


def build_record_from_path(
    image_path: Path,
    input_dir: Path,
    manifest_index: dict[Path, dict[str, Any]],
) -> ImageRecord:
    manifest_row = manifest_index.get(image_path.resolve())
    if manifest_row is not None:
        return ImageRecord(
            image_path=image_path,
            attack_name=str(manifest_row.get("attack") or infer_attack_name(image_path, input_dir)),
            source_label=manifest_row.get("label"),
            target_label=manifest_row.get("target_label"),
            source_index=manifest_row.get("source_index"),
            source_path=manifest_row.get("source_path"),
        )

    return ImageRecord(
        image_path=image_path,
        attack_name=infer_attack_name(image_path, input_dir),
        source_label=infer_source_label(image_path, input_dir),
        target_label=infer_target_label(image_path),
        source_index=None,
        source_path=None,
    )


def discover_images(input_dir: Path) -> list[Path]:
    image_paths = [
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(image_paths)


def build_transform(args: argparse.Namespace):
    steps: list[Any] = []
    if args.resize_size is not None:
        steps.append(transforms.Resize(args.resize_size))
    if args.center_crop_size is not None:
        steps.append(transforms.CenterCrop(args.center_crop_size))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)

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


def selected_success(
    pred_label: int,
    source_label: Optional[int],
    target_label: Optional[int],
    success_mode: str,
) -> tuple[Optional[bool], Optional[str]]:
    untargeted_success = pred_label != source_label if source_label is not None else None
    targeted_success = pred_label == target_label if target_label is not None else None

    if success_mode == "untargeted":
        return untargeted_success, "untargeted" if untargeted_success is not None else None
    if success_mode == "targeted":
        return targeted_success, "targeted" if targeted_success is not None else None
    if success_mode == "none":
        return None, None

    if targeted_success is not None:
        return targeted_success, "targeted"
    if untargeted_success is not None:
        return untargeted_success, "untargeted"
    return None, None


def init_group_summary() -> dict[str, Any]:
    return {
        "total_images": 0,
        "source_label_known": 0,
        "target_label_known": 0,
        "selected_criterion_known": 0,
        "source_accuracy_count": 0,
        "untargeted_success_count": 0,
        "targeted_success_count": 0,
        "selected_success_count": 0,
        "avg_top1_confidence_sum": 0.0,
        "selected_mode_counts": {"untargeted": 0, "targeted": 0},
    }


def finalize_group_summary(group: dict[str, Any]) -> dict[str, Any]:
    total = group["total_images"]
    source_known = group["source_label_known"]
    target_known = group["target_label_known"]
    selected_known = group["selected_criterion_known"]

    group["avg_top1_confidence"] = (
        group["avg_top1_confidence_sum"] / total if total else 0.0
    )
    group["source_accuracy"] = (
        group["source_accuracy_count"] / source_known if source_known else None
    )
    group["untargeted_success_rate"] = (
        group["untargeted_success_count"] / source_known if source_known else None
    )
    group["targeted_success_rate"] = (
        group["targeted_success_count"] / target_known if target_known else None
    )
    group["selected_success_rate"] = (
        group["selected_success_count"] / selected_known if selected_known else None
    )
    del group["avg_top1_confidence_sum"]
    return group


def forward_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    outputs = model(inputs)
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    if not torch.is_tensor(outputs):
        raise TypeError(f"model forward returned unsupported type: {type(outputs)!r}")
    return outputs


def main() -> None:
    setup_logging()
    args = parse_args()
    validate_args(args)

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input path is not a folder: {input_dir}")

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else find_default_manifest(input_dir)
    manifest_index = load_manifest_index(manifest_path)
    if manifest_path is not None and manifest_path.exists():
        LOGGER.info("using manifest: %s", manifest_path)

    image_paths = discover_images(input_dir)
    if args.num_samples is not None:
        image_paths = image_paths[:args.num_samples]
    if not image_paths:
        raise ValueError(f"no images found under {input_dir}")

    excluded_attack_names = {name.strip() for name in args.exclude_attack_names if name.strip()}
    records = [
        build_record_from_path(image_path, input_dir, manifest_index)
        for image_path in image_paths
    ]
    if excluded_attack_names:
        records = [record for record in records if record.attack_name not in excluded_attack_names]
    if not records:
        raise ValueError("all discovered images were filtered out by --exclude-attack-names")

    model, resolved_model_type, resolved_dataset_type = resolve_model(args, device)
    transform = build_transform(args)
    dataset = ImagePredictionDataset(records, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    results_path = output_dir / "prediction_results.jsonl"
    summary_path = output_dir / "prediction_summary.json"
    progress = ProgressTracker(len(records))

    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path) if manifest_path is not None and manifest_path.exists() else None,
        "model_type": resolved_model_type,
        "dataset_type": resolved_dataset_type,
        "success_mode": args.success_mode,
        "total_images": len(records),
        "attack_groups": {},
    }

    progress.update(0, "starting")
    processed = 0
    with results_path.open("w", encoding="utf-8") as results_file:
        for batch_index, (images, indices) in enumerate(loader, start=1):
            progress.update(processed, f"predicting batch {batch_index}")
            images = images.to(device, non_blocking=True)

            with torch.no_grad():
                logits = forward_logits(model, images)
                probabilities = logits.softmax(dim=1)
                pred_confidences, pred_labels = probabilities.max(dim=1)

            batch_size = int(pred_labels.shape[0])
            for offset in range(batch_size):
                record = records[int(indices[offset].item())]
                pred_label = int(pred_labels[offset].detach().cpu().item())
                pred_confidence = float(pred_confidences[offset].detach().cpu().item())

                untargeted_success = (
                    pred_label != record.source_label if record.source_label is not None else None
                )
                targeted_success = (
                    pred_label == record.target_label if record.target_label is not None else None
                )
                selected_value, selected_mode = selected_success(
                    pred_label,
                    record.source_label,
                    record.target_label,
                    args.success_mode,
                )

                group = summary["attack_groups"].setdefault(record.attack_name, init_group_summary())
                group["total_images"] += 1
                group["avg_top1_confidence_sum"] += pred_confidence
                if record.source_label is not None:
                    group["source_label_known"] += 1
                    if pred_label == record.source_label:
                        group["source_accuracy_count"] += 1
                    if untargeted_success:
                        group["untargeted_success_count"] += 1
                if record.target_label is not None:
                    group["target_label_known"] += 1
                    if targeted_success:
                        group["targeted_success_count"] += 1
                if selected_value is not None:
                    group["selected_criterion_known"] += 1
                    if selected_value:
                        group["selected_success_count"] += 1
                if selected_mode is not None:
                    group["selected_mode_counts"][selected_mode] += 1

                result_row = {
                    "image_path": str(record.image_path),
                    "attack": record.attack_name,
                    "source_label": record.source_label,
                    "target_label": record.target_label,
                    "source_index": record.source_index,
                    "source_path": record.source_path,
                    "pred_label": pred_label,
                    "pred_confidence": pred_confidence,
                    "source_correct": (
                        pred_label == record.source_label if record.source_label is not None else None
                    ),
                    "untargeted_success": untargeted_success,
                    "targeted_success": targeted_success,
                    "selected_success": selected_value,
                    "selected_mode": selected_mode,
                }
                results_file.write(json.dumps(result_row) + "\n")

            processed += batch_size
            progress.update(processed, f"completed batch {batch_index}")
            if args.log_every > 0 and batch_index % args.log_every == 0:
                progress.clear()
                LOGGER.info(
                    "processed %d / %d images across %d batches",
                    processed,
                    len(records),
                    batch_index,
                )

    summary["attack_groups"] = {
        attack_name: finalize_group_summary(group)
        for attack_name, group in sorted(summary["attack_groups"].items())
    }
    output_summary = {
        **summary,
        "generated_at_unix": time.time(),
    }
    summary_path.write_text(json.dumps(output_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    progress.finish(processed, "finished")

    LOGGER.info("prediction results written to %s", results_path)
    LOGGER.info("prediction summary written to %s", summary_path)
    for attack_name, group in output_summary["attack_groups"].items():
        LOGGER.info(
            "%s: total=%d source_acc=%s untargeted=%s targeted=%s selected=%s",
            attack_name,
            group["total_images"],
            f"{group['source_accuracy']:.4f}" if group["source_accuracy"] is not None else "n/a",
            f"{group['untargeted_success_rate']:.4f}" if group["untargeted_success_rate"] is not None else "n/a",
            f"{group['targeted_success_rate']:.4f}" if group["targeted_success_rate"] is not None else "n/a",
            f"{group['selected_success_rate']:.4f}" if group["selected_success_rate"] is not None else "n/a",
        )


if __name__ == "__main__":
    main()
