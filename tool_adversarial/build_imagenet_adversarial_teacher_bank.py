#!/usr/bin/env python3
"""Build a training-ready teacher bank from saved ImageNet adversarial images.

This is the next step after generating FGSM / PGD / CW examples:

1. load saved adversarial images from a generator output folder
2. pair each adversarial image with its clean ImageNet sample
3. choose one teacher attack per source image
4. save clean/teacher-adv image pairs plus a manifest for later latent-perturbation training

The script expects a generator-style ``manifest.jsonl`` so it can recover the
source dataset index for each saved adversarial image.
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
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src.ml_setup.factory import get_ml_setup_from_config
from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup_model.pretrained_models import create_torchvision_model
from py_src.model_opti_save_load import load_model_state_file


LOGGER = logging.getLogger("build_imagenet_adversarial_teacher_bank")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_DATASETS = ("imagenet1k", "imagenet100", "imagenet10")
SUPPORTED_ATTACKS = ("fgsm", "pgd", "cw")
TARGET_PATTERN = re.compile(r"_target_(\d+)(?=\.[^.]+$)")


@dataclass
class AttackCandidate:
    attack_name: str
    saved_path: Path
    split: str
    source_index: int
    source_label: int
    target_label: Optional[int]
    source_path: Optional[str]
    manifest_pixel_l2: Optional[float]
    manifest_pixel_linf: Optional[float]
    manifest_adv_pred: Optional[int]


@dataclass
class ScoredCandidate:
    candidate: AttackCandidate
    adv_tensor: torch.Tensor
    pred_label: int
    top1_confidence: float
    source_confidence: Optional[float]
    target_confidence: Optional[float]
    untargeted_success: bool
    targeted_success: Optional[bool]
    reload_pixel_l2: float
    reload_pixel_linf: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clean/adversarial teacher bank from saved ImageNet attacks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "adversarial_root",
        type=str,
        help="Folder containing the generated adversarial examples.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Folder where teacher-bank shards and metadata will be saved. Defaults to <adversarial_root>/teacher_bank.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional path to manifest.jsonl. If omitted, the script searches near the adversarial root.",
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
        help="ImageNet dataset variant to use when rebuilding the clean images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional LLR2 .model.pt checkpoint to use for rescoring the saved adversarial images.",
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
        help="LLR2 ImageNet preset used when rebuilding clean samples.",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="auto",
        choices=("auto", "untargeted", "targeted"),
        help="How to choose the winning teacher when multiple attacks exist for one source image.",
    )
    parser.add_argument(
        "--include-attacks",
        nargs="*",
        default=list(SUPPORTED_ATTACKS),
        choices=SUPPORTED_ATTACKS,
        help="Attack folders to consider from the adversarial root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=("train", "val"),
        help="Optional split filter. If omitted, all manifest rows under the root are considered.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional limit on the number of source-image groups to process after grouping.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=256,
        help="Deprecated legacy shard setting. Ignored because teacher-bank pairs are now saved as per-sample PNG files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device: auto, cuda, cpu, or cuda:0 style values.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Emit an INFO log every N processed groups. Use 0 to disable periodic logs.",
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
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    if args.max_groups is not None and args.max_groups <= 0:
        raise ValueError("--max-groups must be positive")
    if args.log_every < 0:
        raise ValueError("--log-every cannot be negative")


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
    def __init__(self, total_groups: int) -> None:
        self.total_groups = total_groups
        self.start_time = time.time()
        self.last_line_length = 0

    def clear(self) -> None:
        if self.last_line_length == 0:
            return
        sys.stderr.write("\r" + (" " * self.last_line_length) + "\r")
        sys.stderr.flush()
        self.last_line_length = 0

    def update(self, completed_groups: int, selected_count: int, stage: str) -> None:
        elapsed = time.time() - self.start_time
        rate = completed_groups / elapsed if elapsed > 0 and completed_groups > 0 else 0.0
        remaining = max(self.total_groups - completed_groups, 0)
        eta = remaining / rate if rate > 0 else None
        percent = 100.0 * completed_groups / self.total_groups if self.total_groups > 0 else 100.0
        line = (
            f"[progress] {percent:5.1f}% | groups {completed_groups}/{self.total_groups} | "
            f"selected {selected_count} | elapsed {format_duration(elapsed)} | "
            f"eta {format_duration(eta)} | {stage}"
        )
        padding = max(self.last_line_length - len(line), 0)
        sys.stderr.write("\r" + line + (" " * padding))
        sys.stderr.flush()
        self.last_line_length = len(line)

    def finish(self, completed_groups: int, selected_count: int, stage: str) -> None:
        self.update(completed_groups, selected_count, stage)
        sys.stderr.write("\n")
        sys.stderr.flush()
        self.last_line_length = 0


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def find_default_manifest(adversarial_root: Path) -> Optional[Path]:
    for parent in [adversarial_root, *list(adversarial_root.parents)[:3]]:
        candidate = parent / "manifest.jsonl"
        if candidate.exists():
            return candidate
    return None


def infer_target_label_from_name(file_name: str) -> Optional[int]:
    match = TARGET_PATTERN.search(file_name)
    if match:
        return int(match.group(1))
    return None


def load_manifest_candidates(
    manifest_path: Path,
    adversarial_root: Path,
    include_attacks: set[str],
    split_filter: Optional[str],
) -> list[AttackCandidate]:
    candidates: list[AttackCandidate] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            saved_path_value = row.get("saved_path")
            source_index = row.get("source_index")
            source_label = row.get("label")
            attack_name = row.get("attack")
            split = row.get("split")
            if not saved_path_value or source_index is None or source_label is None or not attack_name or not split:
                continue

            saved_path = Path(saved_path_value).resolve()
            if not saved_path.exists():
                continue
            if not path_is_within(saved_path, adversarial_root):
                continue
            if attack_name not in include_attacks:
                continue
            if split_filter is not None and split != split_filter:
                continue

            target_label = row.get("target_label")
            if target_label is None:
                target_label = infer_target_label_from_name(saved_path.name)

            candidates.append(
                AttackCandidate(
                    attack_name=str(attack_name),
                    saved_path=saved_path,
                    split=str(split),
                    source_index=int(source_index),
                    source_label=int(source_label),
                    target_label=int(target_label) if target_label is not None else None,
                    source_path=row.get("source_path"),
                    manifest_pixel_l2=float(row["pixel_l2"]) if row.get("pixel_l2") is not None else None,
                    manifest_pixel_linf=float(row["pixel_linf"]) if row.get("pixel_linf") is not None else None,
                    manifest_adv_pred=int(row["adv_pred"]) if row.get("adv_pred") is not None else None,
                )
            )
    return candidates


def build_adv_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=images.device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device, dtype=torch.float32).view(1, 3, 1, 1)
    return images * std + mean


def forward_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    outputs = model(inputs)
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    if not torch.is_tensor(outputs):
        raise TypeError(f"model forward returned unsupported type: {type(outputs)!r}")
    return outputs

def resolve_model_and_setup(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, Any, str, str]:
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
    return model, ml_setup, resolved_model_type, resolved_dataset_type


def build_group_key(candidate: AttackCandidate) -> tuple[str, int, Optional[int]]:
    return candidate.split, candidate.source_index, candidate.target_label


def load_adv_tensor(path: Path, transform: transforms.Compose) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        return transform(image)


def score_candidates(
    model: nn.Module,
    clean_tensor: torch.Tensor,
    candidates: list[AttackCandidate],
    transform: transforms.Compose,
    source_label: int,
    device: torch.device,
) -> list[ScoredCandidate]:
    adv_tensors = [load_adv_tensor(candidate.saved_path, transform) for candidate in candidates]
    adv_batch = torch.stack(adv_tensors).to(device)
    clean_batch = clean_tensor.unsqueeze(0).repeat(len(candidates), 1, 1, 1).to(device)

    with torch.no_grad():
        logits = forward_logits(model, adv_batch)
        probs = logits.softmax(dim=1)
        pred_confidences, pred_labels = probs.max(dim=1)

    clean_pixels = denormalize_images(clean_batch).clamp(0.0, 1.0)
    adv_pixels = denormalize_images(adv_batch).clamp(0.0, 1.0)
    deltas = adv_pixels - clean_pixels
    l2_values = deltas.reshape(deltas.size(0), -1).norm(p=2, dim=1)
    linf_values = deltas.abs().reshape(deltas.size(0), -1).max(dim=1).values

    scored: list[ScoredCandidate] = []
    for idx, candidate in enumerate(candidates):
        pred_label = int(pred_labels[idx].item())
        target_confidence = None
        targeted_success = None
        if candidate.target_label is not None and candidate.target_label < probs.size(1):
            target_confidence = float(probs[idx, candidate.target_label].item())
            targeted_success = pred_label == candidate.target_label

        scored.append(
            ScoredCandidate(
                candidate=candidate,
                adv_tensor=adv_tensors[idx],
                pred_label=pred_label,
                top1_confidence=float(pred_confidences[idx].item()),
                source_confidence=float(probs[idx, source_label].item()),
                target_confidence=target_confidence,
                untargeted_success=pred_label != source_label,
                targeted_success=targeted_success,
                reload_pixel_l2=float(l2_values[idx].item()),
                reload_pixel_linf=float(linf_values[idx].item()),
            )
        )
    return scored


def ranking_l2(row: ScoredCandidate) -> float:
    if row.candidate.manifest_pixel_l2 is not None:
        return row.candidate.manifest_pixel_l2
    return row.reload_pixel_l2


def choose_teacher(
    scored_candidates: list[ScoredCandidate],
    selection_mode: str,
) -> tuple[ScoredCandidate, str]:
    target_label = scored_candidates[0].candidate.target_label
    mode = selection_mode
    if mode == "auto":
        mode = "targeted" if target_label is not None else "untargeted"

    if mode == "targeted":
        successful = [row for row in scored_candidates if row.targeted_success]
        if successful:
            return min(successful, key=ranking_l2), "targeted_success_min_l2"
        ranked = [
            row for row in scored_candidates
            if row.target_confidence is not None
        ]
        if ranked:
            return max(ranked, key=lambda row: (row.target_confidence, -ranking_l2(row))), "targeted_max_target_confidence"
        return min(scored_candidates, key=ranking_l2), "targeted_fallback_min_l2"

    successful = [row for row in scored_candidates if row.untargeted_success]
    if successful:
        return min(successful, key=ranking_l2), "untargeted_success_min_l2"
    ranked = [row for row in scored_candidates if row.source_confidence is not None]
    if ranked:
        return min(ranked, key=lambda row: (row.source_confidence, ranking_l2(row))), "untargeted_min_source_confidence"
    return min(scored_candidates, key=ranking_l2), "untargeted_fallback_min_l2"


def denormalize_to_pil(image: torch.Tensor) -> Image.Image:
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    pixels = image.detach().cpu().to(dtype=torch.float32) * std + mean
    pixels = pixels.clamp(0.0, 1.0)
    array = (
        pixels.mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    return Image.fromarray(array)


def build_clean_bank_rel_path(split_name: str, source_label: int, source_index: int) -> Path:
    return Path("clean_images") / split_name / f"class_{source_label:04d}" / f"source_{source_index:08d}.png"


def build_teacher_adv_bank_rel_path(
    split_name: str,
    source_label: int,
    source_index: int,
    target_label: Optional[int],
    attack_name: str,
) -> Path:
    stem = f"source_{source_index:08d}_attack_{attack_name}"
    if target_label is not None:
        stem = f"{stem}_target_{target_label:04d}"
    return Path("teacher_adv_images") / split_name / f"class_{source_label:04d}" / f"{stem}.png"


def save_teacher_bank_images(
    output_dir: Path,
    clean_tensor: torch.Tensor,
    teacher_adv_tensor: torch.Tensor,
    split_name: str,
    source_label: int,
    source_index: int,
    target_label: Optional[int],
    attack_name: str,
    saved_clean_rel_paths: set[str],
) -> tuple[str, str]:
    clean_rel_path = build_clean_bank_rel_path(split_name, source_label, source_index)
    teacher_adv_rel_path = build_teacher_adv_bank_rel_path(
        split_name,
        source_label,
        source_index,
        target_label,
        attack_name,
    )

    clean_abs_path = output_dir / clean_rel_path
    teacher_adv_abs_path = output_dir / teacher_adv_rel_path
    clean_key = clean_rel_path.as_posix()

    if clean_key not in saved_clean_rel_paths:
        clean_abs_path.parent.mkdir(parents=True, exist_ok=True)
        denormalize_to_pil(clean_tensor).save(clean_abs_path, format="PNG")
        saved_clean_rel_paths.add(clean_key)

    teacher_adv_abs_path.parent.mkdir(parents=True, exist_ok=True)
    denormalize_to_pil(teacher_adv_tensor).save(teacher_adv_abs_path, format="PNG")
    return clean_key, teacher_adv_rel_path.as_posix()


def main() -> None:
    setup_logging()
    args = parse_args()
    validate_args(args)

    adversarial_root = Path(args.adversarial_root).resolve()
    if not adversarial_root.exists():
        raise FileNotFoundError(f"adversarial root does not exist: {adversarial_root}")
    if not adversarial_root.is_dir():
        raise NotADirectoryError(f"adversarial root is not a folder: {adversarial_root}")

    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else find_default_manifest(adversarial_root)
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError(
            "could not find manifest.jsonl. This step requires the generator manifest so it can map "
            "saved adversarial images back to their clean ImageNet samples."
        )

    output_dir = Path(args.output_dir).resolve() if args.output_dir else adversarial_root / "teacher_bank"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    model, ml_setup, resolved_model_type, resolved_dataset_type = resolve_model_and_setup(args, device)
    adv_transform = build_adv_transform()
    include_attacks = set(args.include_attacks)

    manifest_candidates = load_manifest_candidates(manifest_path, adversarial_root, include_attacks, args.split)
    if not manifest_candidates:
        raise ValueError("no manifest rows matched the requested adversarial root / filters")

    grouped_candidates: dict[tuple[str, int, Optional[int]], list[AttackCandidate]] = defaultdict(list)
    for candidate in manifest_candidates:
        grouped_candidates[build_group_key(candidate)].append(candidate)

    group_items = sorted(grouped_candidates.items(), key=lambda item: item[0])
    if args.max_groups is not None:
        group_items = group_items[:args.max_groups]
    if not group_items:
        raise ValueError("no groups remained after filtering")

    train_dataset = ml_setup.training_data
    val_dataset = ml_setup.testing_data

    summary: dict[str, Any] = {
        "adversarial_root": str(adversarial_root),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "model_type": resolved_model_type,
        "dataset_type": resolved_dataset_type,
        "selection_mode": args.selection_mode,
        "include_attacks": sorted(include_attacks),
        "split_filter": args.split,
        "group_count": len(group_items),
        "selected_count": 0,
        "selected_attack_counts": {},
        "selection_rule_counts": {},
        "target_label_known_groups": 0,
        "untargeted_success_count": 0,
        "targeted_success_count": 0,
        "targeted_success_known_count": 0,
        "warnings": [],
    }

    selected_attack_counter: Counter[str] = Counter()
    selection_rule_counter: Counter[str] = Counter()
    warned_train_split = False
    saved_clean_rel_paths: set[str] = set()
    max_abs_delta = 0.0

    progress = ProgressTracker(len(group_items))
    progress.update(0, 0, "starting")
    completed_groups = 0

    selected_manifest_path = output_dir / "selected_teacher_manifest.jsonl"
    with selected_manifest_path.open("w", encoding="utf-8") as selected_manifest_file:
        for group_key, candidates in group_items:
            split_name, source_index, target_label = group_key
            completed_groups += 1
            progress.update(completed_groups - 1, summary["selected_count"], f"scoring source {source_index}")

            if split_name == "train" and not warned_train_split:
                warning = (
                    "manifest rows come from the train split. If the original adversarial images were generated "
                    "with stochastic train-time augmentation, the clean tensors rebuilt from the dataset index "
                    "may not exactly match the original attack input. Val split is safer for this step."
                )
                LOGGER.warning(warning)
                summary["warnings"].append(warning)
                warned_train_split = True

            split_dataset = train_dataset if split_name == "train" else val_dataset
            clean_tensor, dataset_label = split_dataset[source_index]
            source_label = int(dataset_label)
            if source_label != candidates[0].source_label:
                warning = (
                    f"manifest label mismatch at split={split_name} index={source_index}: "
                    f"manifest={candidates[0].source_label} dataset={source_label}"
                )
                LOGGER.warning(warning)
                summary["warnings"].append(warning)

            scored_candidates = score_candidates(
                model=model,
                clean_tensor=clean_tensor,
                candidates=candidates,
                transform=adv_transform,
                source_label=source_label,
                device=device,
            )
            selected, selection_rule = choose_teacher(scored_candidates, args.selection_mode)

            selected_attack_counter[selected.candidate.attack_name] += 1
            selection_rule_counter[selection_rule] += 1
            summary["selected_count"] += 1
            summary["untargeted_success_count"] += int(selected.untargeted_success)
            if target_label is not None:
                summary["target_label_known_groups"] += 1
            if selected.targeted_success is not None:
                summary["targeted_success_known_count"] += 1
                summary["targeted_success_count"] += int(selected.targeted_success)

            delta_tensor = selected.adv_tensor - clean_tensor
            max_abs_delta = max(max_abs_delta, float(delta_tensor.abs().max().item()))
            clean_bank_path, teacher_adv_bank_path = save_teacher_bank_images(
                output_dir=output_dir,
                clean_tensor=clean_tensor,
                teacher_adv_tensor=selected.adv_tensor,
                split_name=split_name,
                source_label=source_label,
                source_index=source_index,
                target_label=target_label,
                attack_name=selected.candidate.attack_name,
                saved_clean_rel_paths=saved_clean_rel_paths,
            )

            manifest_row = {
                "split": split_name,
                "source_index": source_index,
                "source_label": source_label,
                "target_label": target_label,
                "source_path": selected.candidate.source_path,
                "selected_attack_name": selected.candidate.attack_name,
                "candidate_attack_names": [row.candidate.attack_name for row in scored_candidates],
                "selected_adv_path": str(selected.candidate.saved_path),
                "clean_bank_path": clean_bank_path,
                "teacher_adv_bank_path": teacher_adv_bank_path,
                "selection_rule": selection_rule,
                "pred_label": selected.pred_label,
                "source_confidence": selected.source_confidence,
                "target_confidence": selected.target_confidence,
                "untargeted_success": selected.untargeted_success,
                "targeted_success": selected.targeted_success,
                "manifest_pixel_l2": selected.candidate.manifest_pixel_l2,
                "manifest_pixel_linf": selected.candidate.manifest_pixel_linf,
                "reload_pixel_l2": selected.reload_pixel_l2,
                "reload_pixel_linf": selected.reload_pixel_linf,
                "bank_format": "image_pairs_png",
            }
            selected_manifest_file.write(json.dumps(manifest_row) + "\n")

            progress.update(completed_groups, summary["selected_count"], f"selected {selected.candidate.attack_name}")
            if args.log_every > 0 and completed_groups % args.log_every == 0:
                progress.clear()
                LOGGER.info(
                    "processed %d / %d groups, selected=%d",
                    completed_groups,
                    len(group_items),
                    summary["selected_count"],
                )

    summary["selected_attack_counts"] = dict(sorted(selected_attack_counter.items()))
    summary["selection_rule_counts"] = dict(sorted(selection_rule_counter.items()))
    summary["bank_format"] = "image_pairs_png"
    summary["image_format"] = "png"
    summary["clean_image_count"] = len(saved_clean_rel_paths)
    summary["teacher_adv_image_count"] = summary["selected_count"]
    summary["normalized_delta_abs_max"] = max_abs_delta
    summary["untargeted_success_rate"] = (
        summary["untargeted_success_count"] / summary["selected_count"] if summary["selected_count"] else 0.0
    )
    summary["targeted_success_rate"] = (
        summary["targeted_success_count"] / summary["targeted_success_known_count"]
        if summary["targeted_success_known_count"]
        else None
    )
    summary["generated_at_unix"] = time.time()

    summary_path = output_dir / "teacher_bank_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    progress.finish(completed_groups, summary["selected_count"], "finished")

    LOGGER.info("teacher bank manifest written to %s", selected_manifest_path)
    LOGGER.info("teacher bank summary written to %s", summary_path)
    LOGGER.info(
        "selected=%d untargeted_success_rate=%.4f targeted_success_rate=%s",
        summary["selected_count"],
        summary["untargeted_success_rate"],
        f"{summary['targeted_success_rate']:.4f}" if summary["targeted_success_rate"] is not None else "n/a",
    )


if __name__ == "__main__":
    main()
