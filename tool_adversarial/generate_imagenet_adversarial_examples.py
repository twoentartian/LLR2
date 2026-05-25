#!/usr/bin/env python3
"""Generate and save adversarial ImageNet examples.

This script plugs into LLR2's existing ImageNet dataset setup and supports
three attack families out of the box:

- FGSM
- PGD
- CW (L2-style Carlini-Wagner)

Typical usage with an exported LLR2 checkpoint:

```bash
python3 tool_adversarial/generate_imagenet_adversarial_examples.py \
  --model-type resnet50 \
  --checkpoint checkpoints/resnet50_imagenet1k_v2.model.pt \
  --output-dir artifacts/imagenet_adv \
  --attacks fgsm pgd cw \
  --split val \
  --num-samples 1000
```

Typical usage with torchvision's pretrained weights:

```bash
python3 tool_adversarial/generate_imagenet_adversarial_examples.py \
  --model-type resnet50 \
  --use-torchvision-pretrained \
  --output-dir artifacts/imagenet_adv \
  --attacks fgsm pgd cw \
  --split val \
  --num-samples 1000
```
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image, to_tensor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src.ml_setup.dataloader_util import DataloaderConfig, build_dataloader, move_batch_to_device
from py_src.ml_setup.factory import get_ml_setup_from_config
from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup_model.pretrained_models import create_torchvision_model
from py_src.model_opti_save_load import load_model_state_file


LOGGER = logging.getLogger("generate_imagenet_adversarial_examples")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_DATASETS = ("imagenet1k", "imagenet100", "imagenet10")
SUPPORTED_ATTACKS = ("fgsm", "pgd", "cw")


class IndexedDataset(Dataset):
    """Wrap a dataset so each item also returns its original index."""

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset) # type: ignore

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        return image, label, index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adversarial examples for ImageNet datasets and save them to disk.",formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--model-type",type=str,default=None,help="LLR2 model type name (for example: resnet50, resnet18_bn, vit_b_32). ""If omitted, it will be inferred from --checkpoint when possible.",)
    parser.add_argument("--dataset-type",type=str,default="imagenet1k",choices=SUPPORTED_DATASETS,help="ImageNet dataset variant to attack.",)
    parser.add_argument("--checkpoint",type=str,default=None,help="Optional LLR2 .model.pt checkpoint. If provided, the state dict is loaded into an LLR2 model.",)
    parser.add_argument("--use-torchvision-pretrained",action="store_true",help="Use torchvision's DEFAULT ImageNet pretrained weights instead of an LLR2 checkpoint.",)
    parser.add_argument("--output-dir",type=str,required=True,help="Folder where adversarial images, manifests, and summaries will be saved.",)
    parser.add_argument("--attacks",nargs="+",default=list(SUPPORTED_ATTACKS),choices=SUPPORTED_ATTACKS,help="Attack methods to generate.",)
    parser.add_argument("--split",type=str,default="val",choices=("train", "val"),help="Dataset split to process.",)
    parser.add_argument("--preset",type=int,default=0,help="LLR2 ImageNet preset. In this repo, preset=1 uses the classic 256->224 eval recipe, while preset=0/2 uses the recipe-v2 eval sizes.",)
    parser.add_argument("--batch-size",type=int,default=16,help="Batch size used for attack generation.",)
    parser.add_argument("--num-workers",type=int,default=4,help="Dataloader workers.",)
    parser.add_argument("--num-samples",type=int,default=None,help="Optional limit on the number of samples to process from the selected split.",)
    parser.add_argument("--device",type=str,default="auto",help="Computation device: auto, cuda, cpu, or cuda:0 style values.",)
    parser.add_argument("--use-dali",action="store_true",help="Use the DALI-backed ImageNet loader when supported by your environment.",)
    parser.add_argument("--dali-device-id",type=int,default=0,help="GPU device id for the DALI pipeline.",)
    parser.add_argument("--attack-all-samples",action="store_true",help="Attack every sample. By default, only cleanly classified samples are attacked.",)
    parser.add_argument("--targeted",action="store_true",help="Generate targeted attacks instead of untargeted attacks.",)
    parser.add_argument("--target-mode",type=str,default="random",choices=("random", "least_likely", "fixed"),help="How to choose target labels when --targeted is enabled.",)
    parser.add_argument("--fixed-target-label",type=int,default=None,help="Required when using --targeted --target-mode fixed.",)
    parser.add_argument("--save-originals",action="store_true",help="Also save the corresponding clean images.",)
    parser.add_argument("--save-successful-only",action="store_true",help="Only write adversarial images for successful attacks.",)
    parser.add_argument("--fix-manifest",action="store_true",help="Rebuild manifest.jsonl from existing adversarial PNGs while skipping already-generated outputs.",)
    parser.add_argument("--epsilon",type=float,default=8.0 / 255.0,help="FGSM / PGD Linf budget in pixel space [0, 1].",)
    parser.add_argument("--pgd-alpha",type=float,default=2.0 / 255.0,help="PGD step size in pixel space [0, 1].",)
    parser.add_argument("--pgd-steps",type=int,default=10,help="Number of PGD steps.",)
    parser.add_argument("--no-pgd-random-start",action="store_true",help="Disable the random PGD initialization inside the epsilon ball.",)
    parser.add_argument("--cw-steps",type=int,default=200,help="Number of optimization steps for CW.",)
    parser.add_argument("--cw-lr",type=float,default=0.01,help="Learning rate for CW optimization.",)
    parser.add_argument("--cw-c-values",type=str,default="0.01,0.1,1.0,10.0",help="Comma-separated tradeoff values tried by CW.",)
    parser.add_argument("--cw-kappa",type=float,default=0.0,help="CW confidence margin.",)
    parser.add_argument("--log-every",type=int,default=10,help="Log progress every N processed batches.",)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def parse_float_list(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one float value")
    return parsed


def validate_args(args: argparse.Namespace) -> None:
    if args.targeted and args.target_mode == "fixed" and args.fixed_target_label is None:
        raise ValueError("--fixed-target-label is required when using --target-mode fixed")
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive")
    if args.pgd_alpha <= 0:
        raise ValueError("--pgd-alpha must be positive")
    if args.pgd_steps <= 0:
        raise ValueError("--pgd-steps must be positive")
    if args.cw_steps <= 0:
        raise ValueError("--cw-steps must be positive")
    if args.cw_lr <= 0:
        raise ValueError("--cw-lr must be positive")
    if args.use_dali and args.device == "cpu":
        raise ValueError("--use-dali requires a CUDA device")
    if args.use_torchvision_pretrained and args.checkpoint:
        raise ValueError("choose either --checkpoint or --use-torchvision-pretrained, not both")
    if not args.checkpoint and not args.use_torchvision_pretrained:
        raise ValueError("provide either --checkpoint or --use-torchvision-pretrained")


def safe_len(obj: Any) -> Optional[int]:
    try:
        return len(obj)
    except (TypeError, AttributeError):
        return None


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--:--"

    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ProgressTracker:
    def __init__(
        self,
        total_samples: Optional[int],
        total_batches: Optional[int],
        total_work_units: Optional[float] = None,
    ):
        self.total_samples = total_samples
        self.total_batches = total_batches
        self.total_work_units = total_work_units
        self.start_time = time.time()
        self.last_line_length = 0

    def clear(self) -> None:
        if self.last_line_length == 0:
            return

        sys.stderr.write("\r" + (" " * self.last_line_length) + "\r")
        sys.stderr.flush()
        self.last_line_length = 0

    def update(
        self,
        completed_samples: int,
        current_batch: int,
        eligible_samples: int,
        skipped_samples: int,
        stage: Optional[str] = None,
        completed_work_units: Optional[float] = None,
        total_work_units: Optional[float] = None,
    ) -> None:
        elapsed = time.time() - self.start_time
        eta = None
        percent_text = "--.-%"
        sample_text = f"samples {completed_samples}"
        effective_total_work_units = (
            total_work_units if total_work_units is not None else self.total_work_units
        )
        if completed_work_units is None:
            completed_work_units = float(completed_samples)
        rate = completed_work_units / elapsed if elapsed > 0 and completed_work_units > 0 else 0.0

        if effective_total_work_units is not None and effective_total_work_units > 0:
            remaining = max(effective_total_work_units - completed_work_units, 0.0)
            eta = remaining / rate if rate > 0 else None
            percent = min(100.0, 100.0 * completed_work_units / effective_total_work_units)
            percent_text = f"{percent:5.1f}%"
        elif self.total_samples is not None and self.total_samples > 0:
            remaining = max(self.total_samples - completed_samples, 0)
            eta = remaining / rate if rate > 0 else None
            percent = min(100.0, 100.0 * completed_samples / self.total_samples)
            percent_text = f"{percent:5.1f}%"

        if self.total_samples is not None and self.total_samples > 0:
            sample_text = f"samples {completed_samples}/{self.total_samples}"

        batch_text = f"batch {current_batch}"
        if self.total_batches is not None and self.total_batches > 0:
            batch_text = f"batch {min(current_batch, self.total_batches)}/{self.total_batches}"

        line = (
            f"[progress] {percent_text} | {sample_text} | {batch_text} | "
            f"eligible {eligible_samples} | skipped {skipped_samples} | "
            f"elapsed {format_duration(elapsed)} | eta {format_duration(eta)}"
        )
        if stage:
            line += f" | {stage}"

        padding = max(self.last_line_length - len(line), 0)
        sys.stderr.write("\r" + line + (" " * padding))
        sys.stderr.flush()
        self.last_line_length = len(line)

    def finish(
        self,
        completed_samples: int,
        current_batch: int,
        eligible_samples: int,
        skipped_samples: int,
        stage: Optional[str] = None,
        completed_work_units: Optional[float] = None,
        total_work_units: Optional[float] = None,
    ) -> None:
        self.update(
            completed_samples,
            current_batch,
            eligible_samples,
            skipped_samples,
            stage=stage,
            completed_work_units=completed_work_units,
            total_work_units=total_work_units,
        )
        sys.stderr.write("\n")
        sys.stderr.flush()
        self.last_line_length = 0


def imagenet_mean_tensor(device: torch.device) -> torch.Tensor:
    return torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)


def imagenet_std_tensor(device: torch.device) -> torch.Tensor:
    return torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    mean = imagenet_mean_tensor(images.device)
    std = imagenet_std_tensor(images.device)
    return images * std + mean


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    mean = imagenet_mean_tensor(images.device)
    std = imagenet_std_tensor(images.device)
    return (images - mean) / std


def normalized_pixel_bounds(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = imagenet_mean_tensor(device)
    std = imagenet_std_tensor(device)
    low = (torch.zeros_like(mean) - mean) / std
    high = (torch.ones_like(mean) - mean) / std
    return low, high


def normalized_epsilon(pixel_epsilon: float, device: torch.device) -> torch.Tensor:
    std = imagenet_std_tensor(device)
    return torch.full((1, 3, 1, 1), pixel_epsilon, device=device, dtype=torch.float32) / std


def clamp_tensor(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.minimum(x, high), low)


def forward_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    outputs = model(inputs)
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    if not torch.is_tensor(outputs):
        raise TypeError(f"model forward returned unsupported type: {type(outputs)!r}")
    return outputs


def build_target_labels(
    clean_logits: torch.Tensor,
    labels: torch.Tensor,
    targeted: bool,
    target_mode: str,
    fixed_target_label: Optional[int],
) -> Optional[torch.Tensor]:
    if not targeted:
        return None

    num_classes = clean_logits.shape[1]
    if target_mode == "fixed":
        assert fixed_target_label is not None
        if fixed_target_label < 0 or fixed_target_label >= num_classes:
            raise ValueError(
                f"--fixed-target-label must be in [0, {num_classes - 1}] for the selected model"
            )
        targets = torch.full_like(labels, fixed_target_label)
        if torch.any(targets.eq(labels)):
            raise ValueError("fixed target label must differ from the ground-truth label for every attacked sample")
        return targets

    if target_mode == "random":
        offsets = torch.randint(1, num_classes, labels.shape, device=labels.device)
        return (labels + offsets) % num_classes

    if target_mode == "least_likely":
        sorted_labels = clean_logits.argsort(dim=1, descending=False)
        targets = sorted_labels[:, 0].clone()
        same_mask = targets.eq(labels)
        if same_mask.any():
            targets[same_mask] = sorted_labels[same_mask, 1]
        return targets

    raise ValueError(f"unsupported target mode: {target_mode}")


def filter_fixed_target_label_matches(
    eligible_mask: torch.Tensor,
    labels: torch.Tensor,
    targeted: bool,
    target_mode: str,
    fixed_target_label: Optional[int],
) -> tuple[torch.Tensor, int]:
    if not targeted or target_mode != "fixed":
        return eligible_mask, 0

    assert fixed_target_label is not None
    conflict_mask = labels.eq(fixed_target_label)
    skipped_mask = eligible_mask & conflict_mask
    if not skipped_mask.any():
        return eligible_mask, 0
    return eligible_mask & ~conflict_mask, int(skipped_mask.sum().item())


def attack_success_mask(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    targeted: bool,
    target_labels: Optional[torch.Tensor],
) -> torch.Tensor:
    if targeted:
        assert target_labels is not None
        return predictions.eq(target_labels)
    return predictions.ne(labels)


def fgsm_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x0 = inputs.detach()
    x = x0.clone().detach().requires_grad_(True)
    logits = forward_logits(model, x)
    loss_labels = target_labels if targeted else labels
    assert loss_labels is not None
    loss = F.cross_entropy(logits, loss_labels)
    if targeted:
        loss = -loss
    gradients = torch.autograd.grad(loss, x)[0]

    eps = normalized_epsilon(epsilon, x.device)
    data_low, data_high = normalized_pixel_bounds(x.device)
    lower = torch.maximum(x0 - eps, data_low)
    upper = torch.minimum(x0 + eps, data_high)
    adv = x + eps * gradients.sign()
    return clamp_tensor(adv.detach(), lower, upper)


def pgd_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    step_size: float,
    steps: int,
    random_start: bool = True,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x0 = inputs.detach()
    eps = normalized_epsilon(epsilon, x0.device)
    alpha = normalized_epsilon(step_size, x0.device)
    data_low, data_high = normalized_pixel_bounds(x0.device)
    lower = torch.maximum(x0 - eps, data_low)
    upper = torch.minimum(x0 + eps, data_high)

    if random_start:
        adv = x0 + (torch.rand_like(x0) * 2.0 - 1.0) * eps
        adv = clamp_tensor(adv, lower, upper)
    else:
        adv = x0.clone()

    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        logits = forward_logits(model, adv)
        loss_labels = target_labels if targeted else labels
        assert loss_labels is not None
        loss = F.cross_entropy(logits, loss_labels)
        if targeted:
            loss = -loss
        gradients = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * gradients.sign()
        adv = clamp_tensor(adv, lower, upper)

    return adv.detach()


def atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))


def inverse_tanh_space(pixel_images: torch.Tensor) -> torch.Tensor:
    clamped = pixel_images.clamp(0.0, 1.0) * 2.0 - 1.0
    clamped = clamped.clamp(-0.999999, 0.999999)
    return atanh(clamped)


def tanh_space_to_pixels(w: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.tanh(w) + 1.0)


def cw_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
    learning_rate: float,
    c_values: list[float],
    kappa: float = 0.0,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x0_norm = inputs.detach()
    x0_pixel = denormalize_images(x0_norm).clamp(0.0, 1.0)
    batch_size = x0_norm.shape[0]

    best_success = torch.zeros(batch_size, dtype=torch.bool, device=x0_norm.device)
    best_success_l2 = torch.full((batch_size,), float("inf"), dtype=torch.float32, device=x0_norm.device)
    best_success_adv = x0_pixel.clone()

    best_objective = torch.full((batch_size,), float("inf"), dtype=torch.float32, device=x0_norm.device)
    best_objective_adv = x0_pixel.clone()

    for c_value in c_values:
        w = inverse_tanh_space(x0_pixel).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=learning_rate)

        for _ in range(steps):
            adv_pixel = tanh_space_to_pixels(w)
            adv_norm = normalize_images(adv_pixel)
            logits = forward_logits(model, adv_norm)

            if targeted:
                assert target_labels is not None
                target_logit = logits.gather(1, target_labels.unsqueeze(1)).squeeze(1)
                masked = logits.clone()
                masked.scatter_(1, target_labels.unsqueeze(1), float("-inf"))
                other_logit = masked.max(dim=1).values
                margin_loss = torch.clamp(other_logit - target_logit + kappa, min=0.0)
            else:
                true_logit = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
                masked = logits.clone()
                masked.scatter_(1, labels.unsqueeze(1), float("-inf"))
                other_logit = masked.max(dim=1).values
                margin_loss = torch.clamp(true_logit - other_logit + kappa, min=0.0)

            l2 = (adv_pixel - x0_pixel).reshape(batch_size, -1).pow(2).sum(dim=1)
            objective = l2 + c_value * margin_loss

            optimizer.zero_grad(set_to_none=True)
            objective.sum().backward()
            optimizer.step()

            with torch.no_grad():
                predictions = logits.argmax(dim=1)
                success = attack_success_mask(predictions, labels, targeted, target_labels)

                better_success = success & (l2 < best_success_l2)
                best_success_l2 = torch.where(better_success, l2, best_success_l2)
                best_success = best_success | success
                best_success_adv = torch.where(
                    better_success.view(-1, 1, 1, 1),
                    adv_pixel.detach(),
                    best_success_adv,
                )

                better_objective = objective < best_objective
                best_objective = torch.where(better_objective, objective, best_objective)
                best_objective_adv = torch.where(
                    better_objective.view(-1, 1, 1, 1),
                    adv_pixel.detach(),
                    best_objective_adv,
                )

    final_pixel = torch.where(best_success.view(-1, 1, 1, 1), best_success_adv, best_objective_adv)
    return normalize_images(final_pixel).detach()


def maybe_get_dataset_sample_path(dataset: Any, index: int) -> Optional[str]:
    source_dataset = getattr(dataset, "base_dataset", dataset)
    samples = getattr(source_dataset, "samples", None)
    if samples is not None and 0 <= index < len(samples):
        sample_path = samples[index][0]
        return str(Path(sample_path))

    imgs = getattr(source_dataset, "imgs", None)
    if imgs is not None and 0 <= index < len(imgs):
        sample_path = imgs[index][0]
        return str(Path(sample_path))

    return None


def file_stem_from_source(source_path: Optional[str], index: int) -> str:
    if source_path is None:
        return f"sample_{index:08d}"
    return Path(source_path).stem


def build_output_path(
    output_dir: Path,
    folder_name: str,
    label: int,
    index: int,
    source_path: Optional[str],
    target_label: Optional[int] = None,
) -> Path:
    class_dir = f"class_{label:04d}"
    stem = file_stem_from_source(source_path, index)
    target_suffix = f"_target_{target_label:04d}" if target_label is not None else ""
    file_name = f"{index:08d}_{stem}{target_suffix}.png"
    return output_dir / folder_name / class_dir / file_name


def save_image_tensor(pixel_image: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image = to_pil_image(pixel_image.detach().cpu().clamp(0.0, 1.0))
    pil_image.save(output_path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_manifest_record(
    attack_name: str,
    split: str,
    source_index: int,
    source_path: Optional[str],
    label: int,
    clean_pred: int,
    adv_pred: int,
    clean_correct: bool,
    success: bool,
    l2_distance: float,
    linf_distance: float,
    saved_path: Optional[Path],
    targeted: bool,
    target_label: Optional[int],
) -> dict[str, Any]:
    return {
        "attack": attack_name,
        "split": split,
        "source_index": source_index,
        "source_path": source_path,
        "label": label,
        "clean_pred": clean_pred,
        "adv_pred": adv_pred,
        "clean_correct": clean_correct,
        "success": success,
        "targeted": targeted,
        "target_label": target_label,
        "pixel_l2": l2_distance,
        "pixel_linf": linf_distance,
        "saved_path": str(saved_path) if saved_path is not None else None,
    }


def collect_source_metadata(
    eligible_indices: Optional[torch.Tensor],
    eligible_batch_positions: torch.Tensor,
    batch_start_index: int,
    dataset_for_metadata: Any,
) -> tuple[list[int], list[Optional[str]]]:
    source_indices: list[int] = []
    source_paths: list[Optional[str]] = []
    for sample_offset in range(int(eligible_batch_positions.shape[0])):
        source_index = (
            int(eligible_indices[sample_offset].detach().cpu().item())
            if eligible_indices is not None
            else batch_start_index + int(eligible_batch_positions[sample_offset].detach().cpu().item())
        )
        source_indices.append(source_index)
        source_paths.append(maybe_get_dataset_sample_path(dataset_for_metadata, source_index))
    return source_indices, source_paths


def build_attack_output_paths(
    output_dir: Path,
    attack_name: str,
    labels: torch.Tensor,
    source_indices: list[int],
    source_paths: list[Optional[str]],
    targeted: bool,
    target_labels: Optional[torch.Tensor],
) -> list[Path]:
    output_paths: list[Path] = []
    for sample_offset, source_index in enumerate(source_indices):
        label_value = int(labels[sample_offset].detach().cpu().item())
        target_label_value = (
            int(target_labels[sample_offset].detach().cpu().item())
            if targeted and target_labels is not None
            else None
        )
        output_paths.append(
            build_output_path(
                output_dir,
                attack_name,
                label_value,
                source_index,
                source_paths[sample_offset],
                target_label=target_label_value,
            )
        )
    return output_paths


def partition_existing_output_paths(output_paths: list[Path]) -> tuple[list[int], list[int]]:
    pending_positions: list[int] = []
    existing_positions: list[int] = []
    for sample_offset, output_path in enumerate(output_paths):
        if output_path.exists():
            existing_positions.append(sample_offset)
        else:
            pending_positions.append(sample_offset)
    return pending_positions, existing_positions


def save_missing_clean_originals(
    output_dir: Path,
    clean_pixels: torch.Tensor,
    labels: torch.Tensor,
    source_indices: list[int],
    source_paths: list[Optional[str]],
    sample_positions: list[int],
) -> None:
    for sample_offset in sample_positions:
        label_value = int(labels[sample_offset].detach().cpu().item())
        clean_path = build_output_path(
            output_dir,
            "clean",
            label_value,
            source_indices[sample_offset],
            source_paths[sample_offset],
            target_label=None,
        )
        if not clean_path.exists():
            save_image_tensor(clean_pixels[sample_offset], clean_path)


def load_saved_adv_images(output_paths: list[Path], device: torch.device) -> torch.Tensor:
    images: list[torch.Tensor] = []
    for output_path in output_paths:
        with Image.open(output_path) as image:
            pixel_tensor = to_tensor(image.convert("RGB"))
        images.append(normalize_images(pixel_tensor.unsqueeze(0)).squeeze(0))
    if not images:
        raise ValueError("expected at least one saved adversarial image to load")
    return torch.stack(images, dim=0).to(device=device, dtype=torch.float32)


def score_attack_batch(
    model: nn.Module,
    adv_images: torch.Tensor,
    clean_pixels: torch.Tensor,
    labels: torch.Tensor,
    targeted: bool,
    target_labels: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        adv_logits = forward_logits(model, adv_images)
        adv_predictions = adv_logits.argmax(dim=1)

    adv_pixels = denormalize_images(adv_images).clamp(0.0, 1.0)
    deltas = adv_pixels - clean_pixels
    l2_distances = deltas.reshape(deltas.shape[0], -1).norm(p=2, dim=1)
    linf_distances = deltas.abs().reshape(deltas.shape[0], -1).max(dim=1).values
    success_mask = attack_success_mask(
        adv_predictions,
        labels,
        targeted,
        target_labels,
    )
    return adv_predictions, adv_pixels, l2_distances, linf_distances, success_mask


def write_manifest_rows(
    manifest_file,
    attack_name: str,
    split: str,
    source_indices: list[int],
    source_paths: list[Optional[str]],
    labels: torch.Tensor,
    clean_predictions: torch.Tensor,
    adv_predictions: torch.Tensor,
    success_mask: torch.Tensor,
    l2_distances: torch.Tensor,
    linf_distances: torch.Tensor,
    saved_paths: list[Optional[Path]],
    targeted: bool,
    target_labels: Optional[torch.Tensor],
) -> None:
    for sample_offset, source_index in enumerate(source_indices):
        label_value = int(labels[sample_offset].detach().cpu().item())
        clean_pred_value = int(clean_predictions[sample_offset].detach().cpu().item())
        adv_pred_value = int(adv_predictions[sample_offset].detach().cpu().item())
        clean_correct = bool(clean_pred_value == label_value)
        success = bool(success_mask[sample_offset].detach().cpu().item())
        target_label_value = (
            int(target_labels[sample_offset].detach().cpu().item())
            if target_labels is not None
            else None
        )
        record = build_manifest_record(
            attack_name=attack_name,
            split=split,
            source_index=source_index,
            source_path=source_paths[sample_offset],
            label=label_value,
            clean_pred=clean_pred_value,
            adv_pred=adv_pred_value,
            clean_correct=clean_correct,
            success=success,
            l2_distance=float(l2_distances[sample_offset].detach().cpu().item()),
            linf_distance=float(linf_distances[sample_offset].detach().cpu().item()),
            saved_path=saved_paths[sample_offset],
            targeted=targeted,
            target_label=target_label_value,
        )
        manifest_file.write(json.dumps(record) + "\n")


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
        use_dali=args.use_dali,
        dali_device_id=args.dali_device_id,
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


def build_eval_dataloader(ml_setup: Any, args: argparse.Namespace) -> tuple[Any, Any]:
    dataset = ml_setup.training_data if args.split == "train" else ml_setup.testing_data
    is_train = args.split == "train"
    dataloader_cfg = DataloaderConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    if hasattr(dataset, "build_dataloader") and callable(dataset.build_dataloader):
        dataloader = build_dataloader(
            dataset,
            ml_setup.default_batch_size,
            config=dataloader_cfg,
            is_train=is_train,
            default_collate_fn=ml_setup.default_collate_fn,
        )
        return dataloader, dataset

    indexed_dataset = IndexedDataset(dataset)
    dataloader = build_dataloader(
        indexed_dataset,
        ml_setup.default_batch_size,
        config=dataloader_cfg,
        is_train=is_train,
        default_collate_fn=ml_setup.default_collate_fn_val or ml_setup.default_collate_fn,
    )
    return dataloader, indexed_dataset


def unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(batch, (tuple, list)):
        if len(batch) == 3:
            images, labels, indices = batch
            return images, labels, indices
        if len(batch) == 2:
            images, labels = batch
            return images, labels, None
    if isinstance(batch, dict):
        if {"data", "label", "index"}.issubset(batch):
            return batch["data"], batch["label"], batch["index"]
        if {"data", "label"}.issubset(batch):
            return batch["data"], batch["label"], None
    raise TypeError(f"unsupported batch structure: {type(batch)!r}")


def update_summary_attack_stats(
    stats: dict[str, Any],
    attack_name: str,
    success_mask: torch.Tensor,
    l2_distances: torch.Tensor,
    linf_distances: torch.Tensor,
    saved_count: int,
) -> None:
    attack_stats = stats["attacks"][attack_name]
    batch_size = int(success_mask.numel())
    success_count = int(success_mask.sum().item())

    attack_stats["attempted"] += batch_size
    attack_stats["success"] += success_count
    attack_stats["saved"] += saved_count
    attack_stats["l2_sum"] += float(l2_distances.sum().item())
    attack_stats["linf_sum"] += float(linf_distances.sum().item())

    if success_count > 0:
        attack_stats["successful_l2_sum"] += float(l2_distances[success_mask].sum().item())
        attack_stats["successful_linf_sum"] += float(linf_distances[success_mask].sum().item())


def finalize_summary(stats: dict[str, Any]) -> dict[str, Any]:
    for attack_name, attack_stats in stats["attacks"].items():
        attempted = attack_stats["attempted"]
        success = attack_stats["success"]
        attack_stats["success_rate"] = (success / attempted) if attempted else 0.0
        attack_stats["mean_l2"] = (attack_stats["l2_sum"] / attempted) if attempted else 0.0
        attack_stats["mean_linf"] = (attack_stats["linf_sum"] / attempted) if attempted else 0.0
        attack_stats["mean_successful_l2"] = (
            attack_stats["successful_l2_sum"] / success if success else 0.0
        )
        attack_stats["mean_successful_linf"] = (
            attack_stats["successful_linf_sum"] / success if success else 0.0
        )

        del attack_stats["l2_sum"]
        del attack_stats["linf_sum"]
        del attack_stats["successful_l2_sum"]
        del attack_stats["successful_linf_sum"]

    return stats


def total_skipped_samples(stats: dict[str, Any]) -> int:
    return int(
        stats.get("skipped_clean_misclassified", 0)
        + stats.get("skipped_fixed_target_label_matches", 0)
    )


def build_attack_work_profile(
    attack_names: list[str],
    pgd_steps: int,
    cw_steps: int,
    cw_c_values: list[float],
) -> dict[str, tuple[float, float]]:
    profile: dict[str, tuple[float, float]] = {}
    for attack_name in attack_names:
        if attack_name == "fgsm":
            profile[attack_name] = (2.0, 1.0)
        elif attack_name == "pgd":
            profile[attack_name] = (float(2 * pgd_steps), 1.0)
        elif attack_name == "cw":
            profile[attack_name] = (float(2 * cw_steps * len(cw_c_values)), 1.0)
        else:
            raise ValueError(f"unsupported attack: {attack_name}")
    return profile


def main() -> None:
    setup_logging()
    args = parse_args()
    validate_args(args)

    device = resolve_device(args.device)
    if args.use_dali and device.type != "cuda":
        raise ValueError("--use-dali requires a CUDA device")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cw_c_values = parse_float_list(args.cw_c_values)
    attack_work_profile = build_attack_work_profile(
        args.attacks,
        args.pgd_steps,
        args.cw_steps,
        cw_c_values,
    )
    clean_eval_work_units = 1.0
    total_attack_work_units_per_sample = sum(
        generation_units + scoring_units
        for generation_units, scoring_units in attack_work_profile.values()
    )
    model, ml_setup, resolved_model_type, resolved_dataset_type = resolve_model_and_setup(args, device)
    dataloader, dataset_for_metadata = build_eval_dataloader(ml_setup, args)
    dataset_length = safe_len(dataset_for_metadata)
    planned_total_samples = dataset_length
    if args.num_samples is not None:
        planned_total_samples = min(dataset_length, args.num_samples) if dataset_length is not None else args.num_samples
    planned_total_batches = (
        math.ceil(planned_total_samples / args.batch_size)
        if planned_total_samples is not None and args.batch_size > 0
        else safe_len(dataloader)
    )
    planned_total_work_units = (
        planned_total_samples * (clean_eval_work_units + total_attack_work_units_per_sample)
        if planned_total_samples is not None
        else None
    )
    progress = ProgressTracker(
        planned_total_samples,
        planned_total_batches,
        total_work_units=planned_total_work_units,
    )

    manifest_path = output_dir / "manifest.jsonl"
    config_path = output_dir / "run_config.json"
    summary_path = output_dir / "summary.json"

    run_config = {
        "timestamp_unix": time.time(),
        "model_type": resolved_model_type,
        "dataset_type": resolved_dataset_type,
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "use_torchvision_pretrained": args.use_torchvision_pretrained,
        "output_dir": str(output_dir),
        "attacks": args.attacks,
        "split": args.split,
        "preset": args.preset,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_samples": args.num_samples,
        "device": str(device),
        "use_dali": args.use_dali,
        "dali_device_id": args.dali_device_id,
        "attack_all_samples": args.attack_all_samples,
        "targeted": args.targeted,
        "target_mode": args.target_mode,
        "fixed_target_label": args.fixed_target_label,
        "save_originals": args.save_originals,
        "save_successful_only": args.save_successful_only,
        "fix_manifest": args.fix_manifest,
        "epsilon": args.epsilon,
        "pgd_alpha": args.pgd_alpha,
        "pgd_steps": args.pgd_steps,
        "pgd_random_start": not args.no_pgd_random_start,
        "cw_steps": args.cw_steps,
        "cw_lr": args.cw_lr,
        "cw_c_values": cw_c_values,
        "cw_kappa": args.cw_kappa,
    }
    write_json(config_path, run_config)

    summary: dict[str, Any] = {
        "model_type": resolved_model_type,
        "dataset_type": resolved_dataset_type,
        "split": args.split,
        "processed_samples": 0,
        "eligible_samples": 0,
        "skipped_clean_misclassified": 0,
        "skipped_fixed_target_label_matches": 0,
        "manifest_rebuilt": args.fix_manifest,
        "attacks": {
            attack_name: {
                "attempted": 0,
                "success": 0,
                "saved": 0,
                "skipped_existing": 0,
                "l2_sum": 0.0,
                "linf_sum": 0.0,
                "successful_l2_sum": 0.0,
                "successful_linf_sum": 0.0,
            }
            for attack_name in args.attacks
        },
    }

    processed_samples = 0
    batch_index = 0
    completed_work_units = 0.0
    remaining_total_work_units = planned_total_work_units
    progress.update(
        completed_samples=0,
        current_batch=0,
        eligible_samples=0,
        skipped_samples=total_skipped_samples(summary),
        stage="starting",
        completed_work_units=completed_work_units,
        total_work_units=remaining_total_work_units,
    )

    manifest_mode = "w" if args.fix_manifest else ("a" if manifest_path.exists() else "w")
    if args.fix_manifest:
        LOGGER.info(
            "rebuilding manifest at %s from existing outputs and any newly generated attacks",
            manifest_path,
        )
    with manifest_path.open(manifest_mode, encoding="utf-8") as manifest_file:
        for raw_batch in dataloader:
            batch_index += 1
            batch = move_batch_to_device(raw_batch, device)
            images, labels, indices = unpack_batch(batch)

            images = images.float()
            labels = labels.long()
            batch_size = int(labels.shape[0])

            if args.num_samples is not None and processed_samples >= args.num_samples:
                break

            if args.num_samples is not None:
                remaining = args.num_samples - processed_samples
                if remaining <= 0:
                    break
                if batch_size > remaining:
                    images = images[:remaining]
                    labels = labels[:remaining]
                    if indices is not None:
                        indices = indices[:remaining]
                    batch_size = remaining

            batch_start_index = processed_samples
            progress.update(
                completed_samples=processed_samples,
                current_batch=batch_index,
                eligible_samples=summary["eligible_samples"],
                skipped_samples=total_skipped_samples(summary),
                stage="evaluating clean predictions",
                completed_work_units=completed_work_units,
                total_work_units=remaining_total_work_units,
            )
            with torch.no_grad():
                clean_logits = forward_logits(model, images)
                clean_predictions = clean_logits.argmax(dim=1)
            completed_work_units += batch_size * clean_eval_work_units

            clean_correct_mask = clean_predictions.eq(labels)
            if args.attack_all_samples:
                eligible_mask = torch.ones_like(clean_correct_mask, dtype=torch.bool)
            else:
                eligible_mask = clean_correct_mask

            skipped = int((~eligible_mask).sum().item())
            summary["skipped_clean_misclassified"] += skipped
            eligible_mask, skipped_fixed_target = filter_fixed_target_label_matches(
                eligible_mask,
                labels,
                args.targeted,
                args.target_mode,
                args.fixed_target_label,
            )
            summary["skipped_fixed_target_label_matches"] += skipped_fixed_target
            skipped_attack_samples = batch_size - int(eligible_mask.sum().item())
            if remaining_total_work_units is not None:
                remaining_total_work_units -= skipped_attack_samples * total_attack_work_units_per_sample
            eligible_batch_positions = torch.arange(batch_size, device=labels.device)[eligible_mask]

            if not eligible_mask.any():
                summary["processed_samples"] += batch_size
                processed_samples += batch_size
                progress.update(
                    completed_samples=processed_samples,
                    current_batch=batch_index,
                    eligible_samples=summary["eligible_samples"],
                    skipped_samples=total_skipped_samples(summary),
                    stage="skipped batch (no attack-eligible samples)",
                    completed_work_units=completed_work_units,
                    total_work_units=remaining_total_work_units,
                )
                if args.log_every > 0 and batch_index % args.log_every == 0:
                    progress.clear()
                    LOGGER.info(
                        "processed %d samples (%d batches), batch skipped with no attack-eligible samples "
                        "(clean_misclassified=%d, fixed_target_label_matches=%d)",
                        processed_samples,
                        batch_index,
                        skipped,
                        skipped_fixed_target,
                    )
                continue

            eligible_images = images[eligible_mask]
            eligible_labels = labels[eligible_mask]
            eligible_clean_predictions = clean_predictions[eligible_mask]
            eligible_clean_logits = clean_logits[eligible_mask]
            eligible_indices = indices[eligible_mask] if indices is not None else None
            target_labels = build_target_labels(
                eligible_clean_logits,
                eligible_labels,
                args.targeted,
                args.target_mode,
                args.fixed_target_label,
            )

            summary["eligible_samples"] += int(eligible_labels.shape[0])
            clean_pixels = denormalize_images(eligible_images).clamp(0.0, 1.0)
            eligible_source_indices, eligible_source_paths = collect_source_metadata(
                eligible_indices,
                eligible_batch_positions,
                batch_start_index,
                dataset_for_metadata,
            )

            for attack_name in args.attacks:
                progress.update(
                    completed_samples=processed_samples,
                    current_batch=batch_index,
                    eligible_samples=summary["eligible_samples"],
                    skipped_samples=total_skipped_samples(summary),
                    stage=f"checking existing {attack_name}",
                    completed_work_units=completed_work_units,
                    total_work_units=remaining_total_work_units,
                )
                output_paths = build_attack_output_paths(
                    output_dir,
                    attack_name,
                    eligible_labels,
                    eligible_source_indices,
                    eligible_source_paths,
                    args.targeted,
                    target_labels,
                )
                pending_positions, existing_positions = partition_existing_output_paths(output_paths)
                summary["attacks"][attack_name]["skipped_existing"] += len(existing_positions)
                attack_generation_units, attack_scoring_units = attack_work_profile[attack_name]
                attack_total_work_units = attack_generation_units + attack_scoring_units
                if remaining_total_work_units is not None:
                    remaining_total_work_units -= len(existing_positions) * attack_total_work_units
                if args.save_originals and existing_positions:
                    save_missing_clean_originals(
                        output_dir,
                        clean_pixels,
                        eligible_labels,
                        eligible_source_indices,
                        eligible_source_paths,
                        existing_positions,
                    )
                if args.fix_manifest and existing_positions:
                    existing_positions_tensor = torch.tensor(
                        existing_positions,
                        device=eligible_labels.device,
                        dtype=torch.long,
                    )
                    existing_output_paths = [output_paths[sample_offset] for sample_offset in existing_positions]
                    existing_images = load_saved_adv_images(existing_output_paths, device)
                    existing_clean_pixels = clean_pixels.index_select(0, existing_positions_tensor)
                    existing_labels = eligible_labels.index_select(0, existing_positions_tensor)
                    existing_clean_predictions = eligible_clean_predictions.index_select(0, existing_positions_tensor)
                    existing_target_labels = (
                        target_labels.index_select(0, existing_positions_tensor)
                        if target_labels is not None
                        else None
                    )
                    (
                        existing_adv_predictions,
                        _,
                        existing_l2_distances,
                        existing_linf_distances,
                        existing_success_mask,
                    ) = score_attack_batch(
                        model,
                        existing_images,
                        existing_clean_pixels,
                        existing_labels,
                        args.targeted,
                        existing_target_labels,
                    )
                    write_manifest_rows(
                        manifest_file=manifest_file,
                        attack_name=attack_name,
                        split=args.split,
                        source_indices=[eligible_source_indices[sample_offset] for sample_offset in existing_positions],
                        source_paths=[eligible_source_paths[sample_offset] for sample_offset in existing_positions],
                        labels=existing_labels,
                        clean_predictions=existing_clean_predictions,
                        adv_predictions=existing_adv_predictions,
                        success_mask=existing_success_mask,
                        l2_distances=existing_l2_distances,
                        linf_distances=existing_linf_distances,
                        saved_paths=existing_output_paths,
                        targeted=args.targeted,
                        target_labels=existing_target_labels,
                    )
                    update_summary_attack_stats(
                        summary,
                        attack_name,
                        existing_success_mask,
                        existing_l2_distances,
                        existing_linf_distances,
                        len(existing_output_paths),
                    )

                if not pending_positions:
                    continue

                pending_positions_tensor = torch.tensor(
                    pending_positions,
                    device=eligible_labels.device,
                    dtype=torch.long,
                )
                pending_images = eligible_images.index_select(0, pending_positions_tensor)
                pending_labels = eligible_labels.index_select(0, pending_positions_tensor)
                pending_clean_predictions = eligible_clean_predictions.index_select(0, pending_positions_tensor)
                pending_target_labels = (
                    target_labels.index_select(0, pending_positions_tensor)
                    if target_labels is not None
                    else None
                )

                progress.update(
                    completed_samples=processed_samples,
                    current_batch=batch_index,
                    eligible_samples=summary["eligible_samples"],
                    skipped_samples=total_skipped_samples(summary),
                    stage=f"running {attack_name}",
                    completed_work_units=completed_work_units,
                    total_work_units=remaining_total_work_units,
                )
                if attack_name == "fgsm":
                    adv_images = fgsm_attack(
                        model,
                        pending_images,
                        pending_labels,
                        epsilon=args.epsilon,
                        targeted=args.targeted,
                        target_labels=pending_target_labels,
                    )
                elif attack_name == "pgd":
                    adv_images = pgd_attack(
                        model,
                        pending_images,
                        pending_labels,
                        epsilon=args.epsilon,
                        step_size=args.pgd_alpha,
                        steps=args.pgd_steps,
                        random_start=not args.no_pgd_random_start,
                        targeted=args.targeted,
                        target_labels=pending_target_labels,
                    )
                elif attack_name == "cw":
                    adv_images = cw_attack(
                        model,
                        pending_images,
                        pending_labels,
                        steps=args.cw_steps,
                        learning_rate=args.cw_lr,
                        c_values=cw_c_values,
                        kappa=args.cw_kappa,
                        targeted=args.targeted,
                        target_labels=pending_target_labels,
                    )
                else:
                    raise ValueError(f"unsupported attack: {attack_name}")
                completed_work_units += len(pending_positions) * attack_generation_units

                pending_clean_pixels = clean_pixels.index_select(0, pending_positions_tensor)

                progress.update(
                    completed_samples=processed_samples,
                    current_batch=batch_index,
                    eligible_samples=summary["eligible_samples"],
                    skipped_samples=total_skipped_samples(summary),
                    stage=f"scoring and saving {attack_name}",
                    completed_work_units=completed_work_units,
                    total_work_units=remaining_total_work_units,
                )
                (
                    adv_predictions,
                    adv_pixels,
                    l2_distances,
                    linf_distances,
                    success_mask,
                ) = score_attack_batch(
                    model,
                    adv_images,
                    pending_clean_pixels,
                    pending_labels,
                    args.targeted,
                    pending_target_labels,
                )

                saved_count = 0
                saved_paths: list[Optional[Path]] = []
                for local_offset, sample_offset in enumerate(pending_positions):
                    source_index = eligible_source_indices[sample_offset]
                    source_path = eligible_source_paths[sample_offset]
                    label_value = int(pending_labels[local_offset].detach().cpu().item())
                    success = bool(success_mask[local_offset].detach().cpu().item())

                    saved_path: Optional[Path] = None
                    if not args.save_successful_only or success:
                        saved_path = output_paths[sample_offset]
                        save_image_tensor(adv_pixels[local_offset], saved_path)
                        saved_count += 1

                        if args.save_originals:
                            clean_path = build_output_path(
                                output_dir,
                                "clean",
                                label_value,
                                source_index,
                                source_path,
                                target_label=None,
                            )
                            if not clean_path.exists():
                                save_image_tensor(pending_clean_pixels[local_offset], clean_path)
                    saved_paths.append(saved_path)

                write_manifest_rows(
                    manifest_file=manifest_file,
                    attack_name=attack_name,
                    split=args.split,
                    source_indices=[eligible_source_indices[sample_offset] for sample_offset in pending_positions],
                    source_paths=[eligible_source_paths[sample_offset] for sample_offset in pending_positions],
                    labels=pending_labels,
                    clean_predictions=pending_clean_predictions,
                    adv_predictions=adv_predictions,
                    success_mask=success_mask,
                    l2_distances=l2_distances,
                    linf_distances=linf_distances,
                    saved_paths=saved_paths,
                    targeted=args.targeted,
                    target_labels=pending_target_labels,
                )

                update_summary_attack_stats(
                    summary,
                    attack_name,
                    success_mask,
                    l2_distances,
                    linf_distances,
                    saved_count,
                )
                completed_work_units += len(pending_positions) * attack_scoring_units

            summary["processed_samples"] += batch_size
            processed_samples += batch_size
            progress.update(
                completed_samples=processed_samples,
                current_batch=batch_index,
                eligible_samples=summary["eligible_samples"],
                skipped_samples=total_skipped_samples(summary),
                stage="completed batch",
                completed_work_units=completed_work_units,
                total_work_units=remaining_total_work_units,
            )
            if args.log_every > 0 and batch_index % args.log_every == 0:
                progress.clear()
                LOGGER.info(
                    "processed %d samples across %d batches (eligible=%d, skipped=%d)",
                    processed_samples,
                    batch_index,
                    summary["eligible_samples"],
                    total_skipped_samples(summary),
                )

            if args.num_samples is not None and processed_samples >= args.num_samples:
                break

    finalized_summary = finalize_summary(summary)
    write_json(summary_path, finalized_summary)

    progress.finish(
        completed_samples=processed_samples,
        current_batch=batch_index,
        eligible_samples=summary["eligible_samples"],
        skipped_samples=total_skipped_samples(summary),
        stage="finished",
        completed_work_units=completed_work_units,
        total_work_units=remaining_total_work_units,
    )
    LOGGER.info("finished. outputs written to %s", output_dir)
    if finalized_summary["skipped_fixed_target_label_matches"] > 0:
        LOGGER.info(
            "skipped %d samples because their ground-truth label matched --fixed-target-label=%d",
            finalized_summary["skipped_fixed_target_label_matches"],
            args.fixed_target_label,
        )
    for attack_name, attack_stats in finalized_summary["attacks"].items():
        LOGGER.info(
            "%s: attempted=%d success=%d success_rate=%.4f saved=%d skipped_existing=%d",
            attack_name,
            attack_stats["attempted"],
            attack_stats["success"],
            attack_stats["success_rate"],
            attack_stats["saved"],
            attack_stats["skipped_existing"],
        )


if __name__ == "__main__":
    main()
