#!/usr/bin/env python3
"""Generate ImageNet adversarial examples with a trained latent teacher decoder.

This script is the next stage after
``train_imagenet_latent_teacher_decoder.py``. It loads:

- a trained latent decoder checkpoint
- a classifier (LLR2 checkpoint or torchvision pretrained model)
- an ImageNet split from the LLR2 dataset setup

Then it optimizes a fresh latent code for each selected image, decodes that
latent into a perturbation, and saves the resulting adversarial images plus a
manifest/summary for later evaluation.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src.ml_setup.dataloader_util import DataloaderConfig, build_dataloader, move_batch_to_device
from py_src.ml_setup.factory import get_ml_setup_from_config
from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup_model.pretrained_models import create_torchvision_model
from py_src.model_opti_save_load import load_model_state_file


LOGGER = logging.getLogger("generate_imagenet_latent_adversarial_examples")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_DATASETS = ("imagenet1k", "imagenet100", "imagenet10")
SUPPORTED_TARGET_MODES = ("random", "least_likely", "fixed")
SUPPORTED_LATENT_INIT_MODES = ("zero", "random", "mean_train_latent")


class IndexedDataset(Dataset):
    """Wrap a dataset so each item also returns its original index."""

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        return image, label, index


class ProgressTracker:
    def __init__(self, total_samples: Optional[int], total_batches: Optional[int]):
        self.total_samples = total_samples
        self.total_batches = total_batches
        self.start_time = time.time()
        self.last_line_length = 0

    def clear(self) -> None:
        if self.last_line_length == 0:
            return
        sys.stderr.write("\r" + (" " * self.last_line_length) + "\r")
        sys.stderr.flush()
        self.last_line_length = 0

    def update(self, completed_samples: int, current_batch: int, eligible_samples: int, skipped_samples: int, stage: Optional[str] = None) -> None:
        elapsed = time.time() - self.start_time
        rate = completed_samples / elapsed if elapsed > 0 and completed_samples > 0 else 0.0
        eta = None
        percent_text = "--.-%"
        sample_text = f"samples {completed_samples}"

        if self.total_samples is not None and self.total_samples > 0:
            remaining = max(self.total_samples - completed_samples, 0)
            eta = remaining / rate if rate > 0 else None
            percent = min(100.0, 100.0 * completed_samples / self.total_samples)
            percent_text = f"{percent:5.1f}%"
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

    def finish(self, completed_samples: int, current_batch: int, eligible_samples: int, skipped_samples: int, stage: Optional[str] = None) -> None:
        self.update(completed_samples, current_batch, eligible_samples, skipped_samples, stage=stage)
        sys.stderr.write("\n")
        sys.stderr.flush()
        self.last_line_length = 0


class LatentGridDecoder(nn.Module):
    def __init__(self, grid_size: int, scale: float, latent_channels: int = 3, hidden_channels: int = 24) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate adversarial ImageNet examples by optimizing latents in a trained perturbation decoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("latent_decoder_checkpoint", type=str, help="Path to latent_teacher_decoder_checkpoint.pt.")
    parser.add_argument("--model-type", type=str, default=None, help="LLR2 model type name. If omitted, it is inferred from the classifier checkpoint or latent decoder checkpoint.")
    parser.add_argument("--dataset-type", type=str, default=None, help="ImageNet dataset variant. If omitted, it is inferred from the latent decoder checkpoint when possible.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional LLR2 classifier .model.pt checkpoint.")
    parser.add_argument("--use-torchvision-pretrained", action="store_true", help="Use torchvision ImageNet pretrained weights instead of an LLR2 classifier checkpoint.")
    parser.add_argument("--output-dir", type=str, required=True, help="Folder where adversarial images, manifests, and summaries will be saved.")
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"), help="Dataset split to process.")
    parser.add_argument("--preset", type=int, default=0, help="LLR2 ImageNet preset.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size used for latent optimization.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--num-samples", type=int, default=None, help="Optional limit on the number of samples to process from the selected split.")
    parser.add_argument("--device", type=str, default="auto", help="Computation device: auto, cuda, cpu, or cuda:0 style values.")
    parser.add_argument("--use-dali", action="store_true", help="Use the DALI-backed ImageNet loader when supported by your environment.")
    parser.add_argument("--dali-device-id", type=int, default=0, help="GPU device id for the DALI pipeline.")
    parser.add_argument("--attack-all-samples", action="store_true", help="Attack every sample. By default, only cleanly classified samples are attacked.")
    parser.add_argument("--targeted", action="store_true", help="Generate targeted attacks instead of untargeted attacks.")
    parser.add_argument("--target-mode", type=str, default="random", choices=SUPPORTED_TARGET_MODES, help="How to choose target labels when --targeted is enabled.")
    parser.add_argument("--fixed-target-label", type=int, default=None, help="Required when using --targeted --target-mode fixed.")
    parser.add_argument("--save-originals", action="store_true", help="Also save the corresponding clean images.")
    parser.add_argument("--save-successful-only", action="store_true", help="Only write adversarial images for successful attacks.")
    parser.add_argument("--attack-name", type=str, default="latent_decoder", help="Folder/manifest attack name used for saved outputs.")
    parser.add_argument("--steps", type=int, default=80, help="Number of latent optimization steps per batch.")
    parser.add_argument("--lr", type=float, default=0.08, help="Learning rate for latent optimization.")
    parser.add_argument("--epsilon", type=float, default=None, help="Optional hard Linf budget in pixel space [0, 1]. If omitted, only valid-image clamping is applied.")
    parser.add_argument("--latent-init-mode", type=str, default="zero", choices=SUPPORTED_LATENT_INIT_MODES, help="How to initialize the optimized latents.")
    parser.add_argument("--latent-init-scale", type=float, default=1e-3, help="Stddev used when --latent-init-mode random.")
    parser.add_argument("--lambda-attack", type=float, default=1.0, help="Weight for the attack objective.")
    parser.add_argument("--lambda-l2", type=float, default=30.0, help="Weight for decoded delta L2 regularization.")
    parser.add_argument("--lambda-tv", type=float, default=0.8, help="Weight for decoded delta total variation regularization.")
    parser.add_argument("--lambda-latent", type=float, default=1e-3, help="Weight for latent L2 regularization.")
    parser.add_argument("--lambda-anchor", type=float, default=0.0, help="Weight for staying near the latent initialization.")
    parser.add_argument("--log-every", type=int, default=10, help="Log progress every N processed batches.")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")


def safe_len(obj: Any) -> Optional[int]:
    try:
        return len(obj)
    except (TypeError, AttributeError):
        return None


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


def imagenet_mean_tensor(device: torch.device) -> torch.Tensor:
    return torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)


def imagenet_std_tensor(device: torch.device) -> torch.Tensor:
    return torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    return images * imagenet_std_tensor(images.device) + imagenet_mean_tensor(images.device)


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    return (images - imagenet_mean_tensor(images.device)) / imagenet_std_tensor(images.device)


def normalized_pixel_bounds(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = imagenet_mean_tensor(device)
    std = imagenet_std_tensor(device)
    low = (torch.zeros_like(mean) - mean) / std
    high = (torch.ones_like(mean) - mean) / std
    return low, high


def normalized_epsilon(pixel_epsilon: float, device: torch.device) -> torch.Tensor:
    return torch.full((1, 3, 1, 1), pixel_epsilon, device=device, dtype=torch.float32) / imagenet_std_tensor(device)


def clamp_tensor(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.minimum(x, high), low)


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


def validate_args(args: argparse.Namespace) -> None:
    if args.use_torchvision_pretrained and args.checkpoint:
        raise ValueError("choose either --checkpoint or --use-torchvision-pretrained, not both")
    if not args.checkpoint and not args.use_torchvision_pretrained:
        raise ValueError("provide either --checkpoint or --use-torchvision-pretrained")
    if args.dataset_type is not None and args.dataset_type not in SUPPORTED_DATASETS:
        raise ValueError(f"--dataset-type must be one of: {', '.join(SUPPORTED_DATASETS)}")
    if args.targeted and args.target_mode == "fixed" and args.fixed_target_label is None:
        raise ValueError("--fixed-target-label is required when using --targeted --target-mode fixed")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.epsilon is not None and args.epsilon <= 0:
        raise ValueError("--epsilon must be positive when provided")
    if args.latent_init_scale < 0:
        raise ValueError("--latent-init-scale cannot be negative")
    if args.use_dali and args.device == "cpu":
        raise ValueError("--use-dali requires a CUDA device")


def load_latent_decoder_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, Any], LatentGridDecoder]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    decoder_config = payload.get("decoder_config")
    decoder_state_dict = payload.get("decoder_state_dict")
    if not isinstance(decoder_config, dict) or decoder_state_dict is None:
        raise ValueError(f"{path} is missing decoder_config or decoder_state_dict")

    decoder = LatentGridDecoder(
        grid_size=int(decoder_config["grid_size"]),
        scale=float(decoder_config["scale"]),
        latent_channels=int(decoder_config["latent_channels"]),
        hidden_channels=int(decoder_config["hidden_channels"]),
    )
    decoder.load_state_dict(decoder_state_dict, strict=True)
    decoder = decoder.to(device)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return payload, decoder


def initialize_latents(
    batch_size: int,
    latent_channels: int,
    grid_size: int,
    device: torch.device,
    init_mode: str,
    init_scale: float,
    decoder_checkpoint: dict[str, Any],
) -> torch.Tensor:
    if init_mode == "zero":
        return torch.zeros((batch_size, latent_channels, grid_size, grid_size), device=device, dtype=torch.float32)
    if init_mode == "random":
        return init_scale * torch.randn((batch_size, latent_channels, grid_size, grid_size), device=device, dtype=torch.float32)
    if init_mode == "mean_train_latent":
        latent_bank_state = decoder_checkpoint.get("latent_bank_state_dict")
        if not isinstance(latent_bank_state, dict) or "latents" not in latent_bank_state:
            raise ValueError("latent decoder checkpoint does not contain latent_bank_state_dict['latents']")
        mean_latent = latent_bank_state["latents"].to(device=device, dtype=torch.float32).mean(dim=0, keepdim=True)
        return mean_latent.repeat(batch_size, 1, 1, 1)
    raise ValueError(f"unsupported latent init mode: {init_mode}")


def build_target_labels(clean_logits: torch.Tensor, labels: torch.Tensor, targeted: bool, target_mode: str, fixed_target_label: Optional[int]) -> Optional[torch.Tensor]:
    if not targeted:
        return None

    num_classes = clean_logits.shape[1]
    if target_mode == "fixed":
        assert fixed_target_label is not None
        if fixed_target_label < 0 or fixed_target_label >= num_classes:
            raise ValueError(f"--fixed-target-label must be in [0, {num_classes - 1}] for the selected model")
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


def attack_success_mask(predictions: torch.Tensor, labels: torch.Tensor, targeted: bool, target_labels: Optional[torch.Tensor]) -> torch.Tensor:
    if targeted:
        assert target_labels is not None
        return predictions.eq(target_labels)
    return predictions.ne(labels)


def latent_attack(
    model: nn.Module,
    decoder: LatentGridDecoder,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    target_labels: Optional[torch.Tensor],
    decoder_checkpoint: dict[str, Any],
    device: torch.device,
    steps: int,
    lr: float,
    epsilon: Optional[float],
    init_mode: str,
    init_scale: float,
    lambda_attack: float,
    lambda_l2: float,
    lambda_tv: float,
    lambda_latent: float,
    lambda_anchor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x0 = inputs.detach().to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.long)
    if target_labels is not None:
        target_labels = target_labels.to(device=device, dtype=torch.long)

    init_latents = initialize_latents(
        batch_size=x0.size(0),
        latent_channels=decoder.latent_channels,
        grid_size=decoder.grid_size,
        device=device,
        init_mode=init_mode,
        init_scale=init_scale,
        decoder_checkpoint=decoder_checkpoint,
    )
    anchor = init_latents.detach()
    latents = init_latents.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([latents], lr=lr)

    data_low, data_high = normalized_pixel_bounds(device)
    if epsilon is not None:
        eps = normalized_epsilon(epsilon, device)
        lower = torch.maximum(x0 - eps, data_low)
        upper = torch.minimum(x0 + eps, data_high)
    else:
        lower = data_low
        upper = data_high

    final_logits = None
    final_delta = None
    advs = x0
    for _ in range(steps):
        delta = decoder(latents, (x0.size(2), x0.size(3)))
        advs = clamp_tensor(x0 + delta, lower, upper)
        logits = forward_logits(model, advs)

        if target_labels is not None:
            attack_loss = F.cross_entropy(logits, target_labels)
        else:
            true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            masked = logits.clone()
            masked.scatter_(1, labels.unsqueeze(1), float("-inf"))
            other_logits = masked.max(dim=1).values
            attack_loss = (true_logits - other_logits).mean()

        l2_loss = delta.pow(2).mean()
        tv_loss = total_variation(delta)
        latent_loss = latents.pow(2).mean()
        anchor_loss = (latents - anchor).pow(2).mean()
        loss = (
            lambda_attack * attack_loss
            + lambda_l2 * l2_loss
            + lambda_tv * tv_loss
            + lambda_latent * latent_loss
            + lambda_anchor * anchor_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        final_logits = logits.detach()
        final_delta = delta.detach()

    assert final_logits is not None and final_delta is not None
    with torch.no_grad():
        predictions = final_logits.argmax(dim=1)
        probabilities = final_logits.softmax(dim=1)
        success = attack_success_mask(predictions, labels, target_labels is not None, target_labels)
        source_confidence = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        target_confidence = probabilities.gather(1, target_labels.unsqueeze(1)).squeeze(1) if target_labels is not None else None

        metrics: dict[str, torch.Tensor] = {
            "predictions": predictions,
            "success": success,
            "source_confidence": source_confidence,
            "decoded_delta_rms": final_delta.reshape(final_delta.size(0), -1).pow(2).mean(dim=1).sqrt(),
        }
        if target_confidence is not None:
            metrics["target_confidence"] = target_confidence
        if target_labels is None:
            masked = final_logits.clone()
            masked.scatter_(1, labels.unsqueeze(1), float("-inf"))
            metrics["untargeted_margin"] = final_logits.gather(1, labels.unsqueeze(1)).squeeze(1) - masked.max(dim=1).values
        else:
            masked = final_logits.clone()
            masked.scatter_(1, target_labels.unsqueeze(1), float("-inf"))
            metrics["targeted_margin"] = masked.max(dim=1).values - final_logits.gather(1, target_labels.unsqueeze(1)).squeeze(1)
    return advs.detach(), metrics


def resolve_model_and_setup(
    args: argparse.Namespace,
    decoder_checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, Any, str, str]:
    checkpoint_model_type = None
    checkpoint_dataset_type = None
    state_dict = None
    if args.checkpoint:
        state_dict, checkpoint_model_type, checkpoint_dataset_type = load_model_state_file(args.checkpoint)

    resolved_model_type = args.model_type or checkpoint_model_type or decoder_checkpoint.get("classifier_model_type")
    if resolved_model_type is None:
        raise ValueError("unable to infer model type; pass --model-type or use a checkpoint/latent decoder checkpoint that stores it")

    resolved_dataset_type = args.dataset_type or checkpoint_dataset_type or decoder_checkpoint.get("classifier_dataset_type")
    if resolved_dataset_type is None:
        raise ValueError("unable to infer dataset type; pass --dataset-type or use a checkpoint/latent decoder checkpoint that stores it")
    if resolved_dataset_type not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset type {resolved_dataset_type!r}; expected one of {', '.join(SUPPORTED_DATASETS)}")
    if checkpoint_dataset_type is not None and checkpoint_dataset_type != resolved_dataset_type:
        raise ValueError(f"classifier checkpoint dataset type {checkpoint_dataset_type!r} does not match resolved dataset type {resolved_dataset_type!r}")

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
                f"failed to load classifier checkpoint cleanly. missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
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


def maybe_get_dataset_sample_path(dataset: Any, index: int) -> Optional[str]:
    source_dataset = getattr(dataset, "base_dataset", dataset)
    samples = getattr(source_dataset, "samples", None)
    if samples is not None and 0 <= index < len(samples):
        return str(Path(samples[index][0]))
    imgs = getattr(source_dataset, "imgs", None)
    if imgs is not None and 0 <= index < len(imgs):
        return str(Path(imgs[index][0]))
    return None


def file_stem_from_source(source_path: Optional[str], index: int) -> str:
    if source_path is None:
        return f"sample_{index:08d}"
    return Path(source_path).stem


def build_output_path(output_dir: Path, folder_name: str, label: int, index: int, source_path: Optional[str], target_label: Optional[int] = None) -> Path:
    stem = file_stem_from_source(source_path, index)
    class_dir = f"class_{label:04d}"
    target_suffix = f"_target_{target_label:04d}" if target_label is not None else ""
    file_name = f"{index:08d}_{stem}{target_suffix}.png"
    return output_dir / folder_name / class_dir / file_name


def save_image_tensor(pixel_image: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(pixel_image.detach().cpu().clamp(0.0, 1.0)).save(output_path)


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
    source_confidence: float,
    target_confidence: Optional[float],
    decoded_delta_rms: float,
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
        "source_confidence": source_confidence,
        "target_confidence": target_confidence,
        "decoded_delta_rms": decoded_delta_rms,
    }


def update_summary_stats(
    attack_stats: dict[str, Any],
    success_mask: torch.Tensor,
    l2_distances: torch.Tensor,
    linf_distances: torch.Tensor,
    source_confidences: torch.Tensor,
    target_confidences: Optional[torch.Tensor],
    decoded_delta_rms: torch.Tensor,
    saved_count: int,
) -> None:
    batch_size = int(success_mask.numel())
    success_count = int(success_mask.sum().item())
    attack_stats["attempted"] += batch_size
    attack_stats["success"] += success_count
    attack_stats["saved"] += saved_count
    attack_stats["l2_sum"] += float(l2_distances.sum().item())
    attack_stats["linf_sum"] += float(linf_distances.sum().item())
    attack_stats["source_confidence_sum"] += float(source_confidences.sum().item())
    attack_stats["decoded_delta_rms_sum"] += float(decoded_delta_rms.sum().item())
    if target_confidences is not None:
        attack_stats["target_confidence_sum"] += float(target_confidences.sum().item())
        attack_stats["target_confidence_known"] += int(target_confidences.numel())

    if success_count > 0:
        attack_stats["successful_l2_sum"] += float(l2_distances[success_mask].sum().item())
        attack_stats["successful_linf_sum"] += float(linf_distances[success_mask].sum().item())


def finalize_summary(summary: dict[str, Any], attack_name: str) -> dict[str, Any]:
    attack_stats = summary["attacks"][attack_name]
    attempted = attack_stats["attempted"]
    success = attack_stats["success"]
    attack_stats["success_rate"] = (success / attempted) if attempted else 0.0
    attack_stats["mean_l2"] = (attack_stats["l2_sum"] / attempted) if attempted else 0.0
    attack_stats["mean_linf"] = (attack_stats["linf_sum"] / attempted) if attempted else 0.0
    attack_stats["mean_successful_l2"] = (attack_stats["successful_l2_sum"] / success) if success else 0.0
    attack_stats["mean_successful_linf"] = (attack_stats["successful_linf_sum"] / success) if success else 0.0
    attack_stats["mean_source_confidence"] = (attack_stats["source_confidence_sum"] / attempted) if attempted else 0.0
    attack_stats["mean_decoded_delta_rms"] = (attack_stats["decoded_delta_rms_sum"] / attempted) if attempted else 0.0
    attack_stats["mean_target_confidence"] = (
        attack_stats["target_confidence_sum"] / attack_stats["target_confidence_known"]
        if attack_stats["target_confidence_known"]
        else None
    )

    del attack_stats["l2_sum"]
    del attack_stats["linf_sum"]
    del attack_stats["successful_l2_sum"]
    del attack_stats["successful_linf_sum"]
    del attack_stats["source_confidence_sum"]
    del attack_stats["target_confidence_sum"]
    del attack_stats["target_confidence_known"]
    del attack_stats["decoded_delta_rms_sum"]
    return summary


def total_skipped_samples(summary: dict[str, Any]) -> int:
    return int(
        summary.get("skipped_clean_misclassified", 0)
        + summary.get("skipped_fixed_target_label_matches", 0)
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    validate_args(args)

    device = resolve_device(args.device)
    latent_checkpoint_path = Path(args.latent_decoder_checkpoint).resolve()
    if not latent_checkpoint_path.exists():
        raise FileNotFoundError(f"latent decoder checkpoint does not exist: {latent_checkpoint_path}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    decoder_checkpoint, decoder = load_latent_decoder_checkpoint(latent_checkpoint_path, device)
    model, ml_setup, resolved_model_type, resolved_dataset_type = resolve_model_and_setup(args, decoder_checkpoint, device)
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
    progress = ProgressTracker(planned_total_samples, planned_total_batches)

    manifest_path = output_dir / "manifest.jsonl"
    config_path = output_dir / "run_config.json"
    summary_path = output_dir / "summary.json"

    run_config = {
        "timestamp_unix": time.time(),
        "latent_decoder_checkpoint": str(latent_checkpoint_path),
        "decoder_config": decoder_checkpoint.get("decoder_config"),
        "classifier_model_type": resolved_model_type,
        "classifier_dataset_type": resolved_dataset_type,
        "classifier_checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "use_torchvision_pretrained": args.use_torchvision_pretrained,
        "output_dir": str(output_dir),
        "attack_name": args.attack_name,
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
        "steps": args.steps,
        "lr": args.lr,
        "epsilon": args.epsilon,
        "latent_init_mode": args.latent_init_mode,
        "latent_init_scale": args.latent_init_scale,
        "lambda_attack": args.lambda_attack,
        "lambda_l2": args.lambda_l2,
        "lambda_tv": args.lambda_tv,
        "lambda_latent": args.lambda_latent,
        "lambda_anchor": args.lambda_anchor,
    }
    write_json(config_path, run_config)

    summary: dict[str, Any] = {
        "latent_decoder_checkpoint": str(latent_checkpoint_path),
        "classifier_model_type": resolved_model_type,
        "classifier_dataset_type": resolved_dataset_type,
        "split": args.split,
        "attack_name": args.attack_name,
        "processed_samples": 0,
        "eligible_samples": 0,
        "skipped_clean_misclassified": 0,
        "skipped_fixed_target_label_matches": 0,
        "attacks": {
            args.attack_name: {
                "attempted": 0,
                "success": 0,
                "saved": 0,
                "l2_sum": 0.0,
                "linf_sum": 0.0,
                "successful_l2_sum": 0.0,
                "successful_linf_sum": 0.0,
                "source_confidence_sum": 0.0,
                "target_confidence_sum": 0.0,
                "target_confidence_known": 0,
                "decoded_delta_rms_sum": 0.0,
            }
        },
    }

    processed_samples = 0
    batch_index = 0
    progress.update(
        completed_samples=0,
        current_batch=0,
        eligible_samples=0,
        skipped_samples=total_skipped_samples(summary),
        stage="starting",
    )

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
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
            )
            with torch.no_grad():
                clean_logits = forward_logits(model, images)
                clean_predictions = clean_logits.argmax(dim=1)

            clean_correct_mask = clean_predictions.eq(labels)
            eligible_mask = torch.ones_like(clean_correct_mask, dtype=torch.bool) if args.attack_all_samples else clean_correct_mask
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

            progress.update(
                completed_samples=processed_samples,
                current_batch=batch_index,
                eligible_samples=summary["eligible_samples"],
                skipped_samples=total_skipped_samples(summary),
                stage=f"running {args.attack_name}",
            )
            adv_images, attack_metrics = latent_attack(
                model=model,
                decoder=decoder,
                inputs=eligible_images,
                labels=eligible_labels,
                target_labels=target_labels,
                decoder_checkpoint=decoder_checkpoint,
                device=device,
                steps=args.steps,
                lr=args.lr,
                epsilon=args.epsilon,
                init_mode=args.latent_init_mode,
                init_scale=args.latent_init_scale,
                lambda_attack=args.lambda_attack,
                lambda_l2=args.lambda_l2,
                lambda_tv=args.lambda_tv,
                lambda_latent=args.lambda_latent,
                lambda_anchor=args.lambda_anchor,
            )

            progress.update(
                completed_samples=processed_samples,
                current_batch=batch_index,
                eligible_samples=summary["eligible_samples"],
                skipped_samples=total_skipped_samples(summary),
                stage=f"scoring and saving {args.attack_name}",
            )
            with torch.no_grad():
                adv_logits = forward_logits(model, adv_images)
                adv_predictions = adv_logits.argmax(dim=1)

            clean_pixels = denormalize_images(eligible_images).clamp(0.0, 1.0)
            adv_pixels = denormalize_images(adv_images).clamp(0.0, 1.0)
            deltas = adv_pixels - clean_pixels
            l2_distances = deltas.reshape(deltas.shape[0], -1).norm(p=2, dim=1)
            linf_distances = deltas.abs().reshape(deltas.shape[0], -1).max(dim=1).values
            success_mask = attack_success_mask(adv_predictions, eligible_labels, args.targeted, target_labels)

            saved_count = 0
            for sample_offset in range(int(eligible_labels.shape[0])):
                source_index = (
                    int(eligible_indices[sample_offset].detach().cpu().item())
                    if eligible_indices is not None
                    else batch_start_index + int(eligible_batch_positions[sample_offset].detach().cpu().item())
                )
                source_path = maybe_get_dataset_sample_path(dataset_for_metadata, source_index)
                label_value = int(eligible_labels[sample_offset].detach().cpu().item())
                clean_pred_value = int(eligible_clean_predictions[sample_offset].detach().cpu().item())
                adv_pred_value = int(adv_predictions[sample_offset].detach().cpu().item())
                clean_correct = bool(clean_pred_value == label_value)
                success = bool(success_mask[sample_offset].detach().cpu().item())
                target_label_value = int(target_labels[sample_offset].detach().cpu().item()) if target_labels is not None else None

                saved_path: Optional[Path] = None
                if not args.save_successful_only or success:
                    saved_path = build_output_path(
                        output_dir,
                        args.attack_name,
                        label_value,
                        source_index,
                        source_path,
                        target_label=target_label_value if args.targeted else None,
                    )
                    save_image_tensor(adv_pixels[sample_offset], saved_path)
                    saved_count += 1

                    if args.save_originals:
                        clean_path = build_output_path(output_dir, "clean", label_value, source_index, source_path, target_label=None)
                        if not clean_path.exists():
                            save_image_tensor(clean_pixels[sample_offset], clean_path)

                record = build_manifest_record(
                    attack_name=args.attack_name,
                    split=args.split,
                    source_index=source_index,
                    source_path=source_path,
                    label=label_value,
                    clean_pred=clean_pred_value,
                    adv_pred=adv_pred_value,
                    clean_correct=clean_correct,
                    success=success,
                    l2_distance=float(l2_distances[sample_offset].detach().cpu().item()),
                    linf_distance=float(linf_distances[sample_offset].detach().cpu().item()),
                    saved_path=saved_path,
                    targeted=args.targeted,
                    target_label=target_label_value,
                    source_confidence=float(attack_metrics["source_confidence"][sample_offset].detach().cpu().item()),
                    target_confidence=(
                        float(attack_metrics["target_confidence"][sample_offset].detach().cpu().item())
                        if "target_confidence" in attack_metrics
                        else None
                    ),
                    decoded_delta_rms=float(attack_metrics["decoded_delta_rms"][sample_offset].detach().cpu().item()),
                )
                manifest_file.write(json.dumps(record) + "\n")

            update_summary_stats(
                summary["attacks"][args.attack_name],
                success_mask=success_mask,
                l2_distances=l2_distances,
                linf_distances=linf_distances,
                source_confidences=attack_metrics["source_confidence"],
                target_confidences=attack_metrics.get("target_confidence"),
                decoded_delta_rms=attack_metrics["decoded_delta_rms"],
                saved_count=saved_count,
            )

            summary["processed_samples"] += batch_size
            processed_samples += batch_size
            progress.update(
                completed_samples=processed_samples,
                current_batch=batch_index,
                eligible_samples=summary["eligible_samples"],
                skipped_samples=total_skipped_samples(summary),
                stage=f"finished {args.attack_name} batch",
            )
            if args.log_every > 0 and batch_index % args.log_every == 0:
                progress.clear()
                LOGGER.info(
                    "processed %d samples (%d batches) | eligible=%d | success_rate_so_far=%.4f",
                    processed_samples,
                    batch_index,
                    summary["eligible_samples"],
                    (
                        summary["attacks"][args.attack_name]["success"] / summary["attacks"][args.attack_name]["attempted"]
                        if summary["attacks"][args.attack_name]["attempted"]
                        else 0.0
                    ),
                )

    finalize_summary(summary, args.attack_name)
    summary["generated_at_unix"] = time.time()
    write_json(summary_path, summary)
    progress.finish(
        completed_samples=processed_samples,
        current_batch=batch_index,
        eligible_samples=summary["eligible_samples"],
        skipped_samples=total_skipped_samples(summary),
        stage="finished",
    )

    attack_stats = summary["attacks"][args.attack_name]
    LOGGER.info("manifest written to %s", manifest_path)
    LOGGER.info("summary written to %s", summary_path)
    if summary["skipped_fixed_target_label_matches"] > 0:
        LOGGER.info(
            "skipped %d samples because their ground-truth label matched --fixed-target-label=%d",
            summary["skipped_fixed_target_label_matches"],
            args.fixed_target_label,
        )
    LOGGER.info(
        "attack=%s attempted=%d success=%d success_rate=%.4f mean_l2=%.4f mean_linf=%.4f",
        args.attack_name,
        attack_stats["attempted"],
        attack_stats["success"],
        attack_stats["success_rate"],
        attack_stats["mean_l2"],
        attack_stats["mean_linf"],
    )


if __name__ == "__main__":
    main()
