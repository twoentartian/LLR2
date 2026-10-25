#!/usr/bin/env python3
"""Symmetric train/validation misalignment measurement.

The input data are combined, shuffled once, and split into two equally sized
halves A and B.  By default, partition A is treated as ``train`` and
partition B as ``val``.  The default optimizer update normalizes the train and
val gradients independently per parameter tensor, combines train descent with
val ascent, and rescales the result to the norm of the train gradient.

The raw ``L_A - L_B`` and previous exchange-symmetric objective remain
available with ``--objective difference`` and
``--objective mean_relative_gap``.  The relative-flatness report is computed
separately on the two original losses and then averaged.  Flatness always
uses the deterministic level-0 transform, so with ``--augmentation`` it is
measured on the unaugmented halves rather than on random augmented views.

Training loaders reshuffle each half every epoch by default (``--shuffle``):
with a fixed order, ViT-B/32 on ImageNet fit only the batches next to its
current position in the epoch. Each half's order depends on its own seed and
the epoch, not on its train/val role. Evaluation and flatness keep the fixed
partition order. ``--no-shuffle`` restores the fixed training order.

Each invocation evaluates both orientations from the same initial model:
partition A as train/B as val, then B as train/A as val.

Each orientation retains the five epochs with the largest absolute accuracy
gap and the five with the largest absolute loss gap. Their checkpoint union
is measured for relative flatness after training, in addition to the final
model; an epoch selected by both rankings is measured only once.

Every epoch also records per-weight-tensor sample variance in a direction-
specific CSV, using scientific notation with four significant digits.

For models with a tokenizer module (such as CCT), epoch-end evaluation also
reports its exact nonzero output-element ratio on each complete partition.

Gradient geometry is accumulated on the device and copied to the host once
per epoch. CUDA AMP, when enabled, covers training and evaluation, but not
the second-order relative-flatness measurements.

CUDA gradient combination is also fused with torch.compile by default when
model compilation is enabled. Epoch weights are tensor inputs, so changing
them does not trigger recompilation. --no-compile-gradient-combination keeps
the original eager calculation; compilation failures also fall back to it.

The normalized two-sided update accepts arithmetic gradient-weight expressions
in the variable ``f`` (the normalized epoch fraction). The defaults are the
constant expressions ``0.5`` and ``0.5`` for train and val respectively.

Supported datasets are MNIST, CIFAR-10, CIFAR-100, ImageNet-1k/21k-p, and a modular
dataset folder containing train.txt/val.txt (and optionally test.txt).
ImageNet defaults to the shared high-accuracy ImageNet DALI pipeline: CPU
decoding and GPU preprocessing for training, evaluation and flatness. Other
datasets keep PyTorch; --loader_backend pytorch selects the PyTorch path.
"""

from __future__ import annotations

import argparse
import ast
import csv
import contextlib
import json
import logging
import math
import os
import operator
import re
import secrets
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from misalignment.epoch_order import ORDER_RNG, EpochOrderSampler, epoch_order
from misalignment.augmentation import (
    IMAGENET_DATASETS,
    build_augmentation,
    describe_augmentation,
    parse_augmentation_level,
)
from misalignment.optimizer import build_optimizer_and_scheduler
from misalignment.checkpoints import GapCheckpointTracker
from misalignment.variance import EpochWeightVarianceRecorder
from misalignment.tokenizer_stats import TOKENIZER_RATIO_FIELDS, TokenizerOutputMonitor
from misalignment.token_norms import TokenNormMonitor, TokenNormRecorder
from misalignment.gradient_geometry import GRADIENT_GEOMETRY_FIELDS, GradientGeometryAccumulator
from misalignment.gradient_combination import NormalizedGradientCombiner, combine_normalized_gradient_tensors
from misalignment.dali_loader import (
    DaliImageNetLoader, close_dali_loaders, loader_runtime_info, resolve_loader_backend,
)

from py_src.ml_setup import get_ml_setup_from_config
from py_src.ml_setup.grokking import arithmetic_addition_grokking, build_grokking_model
from py_src.ml_setup_dataset import ArithmeticDataset, DatasetSetup, DatasetType
from py_src.model_opti_save_load import load_model_state_file, save_model_state
from py_src.util import set_seed, setup_logging
from py_src import distributed
from py_src.precision import inference_autocast, inference_dtype
from py_src.ml_setup.vit import VIT_LEGACY_PRESET
from py_src.ml_setup.convnext import CONVNEXT_OFFICIAL_PRESET

logger = logging.getLogger("misalignment_measurement_2")

# The objective is expressed in terms of non-negative losses.  The unit offset
# makes the logarithm and relative-gap denominator well behaved as a loss
# approaches zero; it is fixed rather than exposed as a task-specific
# hyperparameter.
OBJECTIVE_OFFSET = 1.0


@dataclass
class DatasetBundle:
    partition_a: Dataset
    partition_b: Dataset
    ml_setup: Any
    dataset_name: str
    dataset_type_name: str
    source_examples: int
    permutation: torch.Tensor
    excluded_indices: torch.Tensor
    modular: bool
    eq_position: int | None = None
    model_preset: int | None = None
    # Unaugmented views of partition_a/partition_b for relative flatness;
    # None means the training partitions are already deterministic.
    flatness_partition_a: Dataset | None = None
    flatness_partition_b: Dataset | None = None


@dataclass
class LossMetrics:
    loss: float
    accuracy: float
    examples: int
    tokenizer_nonzero_ratio: float | None = None
    token_norms: dict[str, dict] | None = None


def _parse_augmentation(value: str) -> int:
    return parse_augmentation_level(value)


def _normalize_dataset_name(value: str) -> str:
    name = value.lower()
    return "imagenet21k_p" if name in {"imagenet21k", "imagenet21k-p", "imagenet21kp"} else name


def _default_model_type(dataset_name: str) -> str:
    return {
        "mnist": "lenet5",
        "cifar10": "resnet18_bn",
        "cifar100": "resnet18_bn",
        "noise_cifar10": "resnet18_bn",
        "imagenet1k": "resnet18_bn",
        "imagenet21k_p": "vit_b_32",
        "modular": "transformer_for_grokking",
    }[dataset_name]


def _infer_modulus(dataset_path: Path, requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    match = re.search(r"modulus(\d+)", dataset_path.name)
    if match is None:
        raise ValueError("Could not infer modular modulus; pass --modulus")
    return int(match.group(1))


class AugmentedDataset(Dataset):
    """Bind augmentation randomness to source index and epoch, not A/B name.

    Exchanging the halves therefore keeps each image's augmentation identical.
    ``epoch`` lives in a shared-memory tensor: persistent DataLoader workers
    hold a copy of the dataset taken when they start, so a plain attribute
    set by the main process afterwards would never reach them and every
    epoch would reuse the first epoch's views.
    """

    def __init__(self, dataset: Dataset, transform, seed: int, stochastic: bool) -> None:
        self.dataset = dataset
        self.transform = transform
        self.seed = int(seed)
        self.stochastic = stochastic
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    @epoch.setter
    def epoch(self, value: int) -> None:
        self._epoch.fill_(int(value))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        if self.stochastic:
            # All bundled torchvision transformations use the CPU torch RNG.
            # Seeding its default generator avoids changing the CUDA RNG.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self.seed + 1_000_003 * self.epoch + int(index))
                image = self.transform(image)
        else:
            image = self.transform(image)
        return image, target


def _set_augmentation_epoch(dataset: Dataset, epoch: int) -> None:
    if isinstance(dataset, Subset):
        _set_augmentation_epoch(dataset.dataset, epoch)
    elif isinstance(dataset, AugmentedDataset):
        dataset.epoch = epoch


def _set_loader_epoch(loader, epoch: int) -> None:
    """Advance augmentation views and, for shuffled training loaders, the order.

    DALI loaders read the epoch from their dataset and reorder themselves.
    """
    _set_augmentation_epoch(loader.dataset, epoch)
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, EpochOrderSampler):
        sampler.set_epoch(epoch)
    if isinstance(loader, _DeviceCachedLoader):
        loader.set_epoch(epoch)


def _split_indices(count: int, *, seed: int, odd_size_policy: str) -> tuple[torch.Tensor, torch.Tensor]:
    if count < 2:
        raise ValueError(f"The combined dataset must contain at least two examples, got {count}")
    if count % 2 and odd_size_policy == "error":
        raise ValueError(f"Cannot divide {count} examples equally; use --odd_size_policy drop")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(count, generator=generator)
    used = count - count % 2
    return permutation, permutation[used:]


def _split_dataset(dataset: Dataset, *, seed: int, odd_size_policy: str) -> tuple[Dataset, Dataset, torch.Tensor, torch.Tensor]:
    count = len(dataset)
    permutation, excluded = _split_indices(count, seed=seed, odd_size_policy=odd_size_policy)
    midpoint = count // 2
    first = permutation[:midpoint].tolist()
    second = permutation[midpoint:2 * midpoint].tolist()
    return Subset(dataset, first), Subset(dataset, second), permutation, excluded


def _flatness_partitions(
    ml_setup, dataset_name: str, augmentation: int, source: Dataset,
    partition_a: Subset, partition_b: Subset, seed: int,
) -> tuple[Dataset, Dataset]:
    """Same A/B examples as training, but with the deterministic level-0 transform.

    Relative flatness is a property of the loss on the data themselves; with
    stochastic augmentation it would otherwise depend on whichever random
    views the flatness pass happens to draw.
    """
    if augmentation == 0:
        return partition_a, partition_b
    clean = AugmentedDataset(source, build_augmentation(ml_setup, dataset_name, 0), seed=seed, stochastic=False)
    return Subset(clean, partition_a.indices), Subset(clean, partition_b.indices)


def _load_modular_bundle(args, model_type: str) -> DatasetBundle:
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"modular dataset folder does not exist: {dataset_path}")
    modulus = _infer_modulus(dataset_path, args.modulus)
    preferred_files = [dataset_path / name for name in ("train.txt", "val.txt", "test.txt")]
    files = [path for path in preferred_files if path.is_file()]
    if not files:
        files = sorted(path for path in dataset_path.glob("*.txt") if path.name != "tokenizer.txt")
    if len(files) < 2:
        raise ValueError(f"expected at least two modular .txt files in {dataset_path}")
    tokenizer_path = dataset_path / "tokenizer.txt"
    tokenizer_arg = str(tokenizer_path) if tokenizer_path.is_file() else None
    loaded = [
        ArithmeticDataset.load_from_file(
            str(path),
            modulus,
            name=str(dataset_path),
            train=True,
            tokenizer_path=tokenizer_arg,
        )
        for path in files
    ]
    tokenizer = loaded[0].tokenizer
    data = torch.cat([item.data for item in loaded], dim=0)
    permutation, excluded = _split_indices(data.shape[0], seed=args.split_seed, odd_size_policy=args.odd_size_policy)
    midpoint = data.shape[0] // 2
    data_a = ArithmeticDataset(str(dataset_path), data[permutation[:midpoint]], modulus, True, tokenizer=tokenizer)
    data_b = ArithmeticDataset(str(dataset_path), data[permutation[midpoint:2 * midpoint]], modulus, False, tokenizer=tokenizer)
    custom_setup = DatasetSetup(DatasetType.arithmetic_addition, data_a, data_b)
    ml_setup = arithmetic_addition_grokking(override_dataset=custom_setup)
    if any(value is not None for value in (args.m_nlayer, args.m_n_heads, args.m_d_model, args.m_context_len, args.m_pos_encoding)):
        ml_setup.model = build_grokking_model(
            data_a, n_layers=args.m_nlayer, n_heads=args.m_n_heads, d_model=args.m_d_model,
            context_len=args.m_context_len, position_encoding=args.m_pos_encoding,
        )
    if model_type != "transformer_for_grokking":
        raise ValueError("modular datasets require --model_type transformer_for_grokking")
    eq_positions = (data_a.data[0] == tokenizer.stoi["="]).nonzero(as_tuple=False).flatten()
    if eq_positions.numel() != 1:
        raise ValueError("each modular equation must contain exactly one '=' token")
    eq_position = int(eq_positions.item()) - 1
    return DatasetBundle(
        partition_a=ml_setup.training_data,
        partition_b=ml_setup.testing_data,
        ml_setup=ml_setup,
        dataset_name=str(dataset_path),
        dataset_type_name=DatasetType.arithmetic_addition.name,
        source_examples=int(data.shape[0]),
        permutation=permutation,
        excluded_indices=excluded,
        modular=True,
        eq_position=eq_position,
    )


def _load_image_bundle(args, dataset_name: str, model_type: str, augmentation: int) -> DatasetBundle:
    # Bind augmentation below instead of using the high-accuracy loader's
    # batch transforms. Mixup/CutMix is not used in this experiment.
    factory_preset = getattr(args, "model_preset", None)
    if factory_preset is None:
        if model_type.startswith("convnext") and dataset_name == "imagenet21k_p":
            factory_preset = CONVNEXT_OFFICIAL_PRESET
        elif dataset_name in IMAGENET_DATASETS:
            factory_preset = VIT_LEGACY_PRESET if model_type.startswith("vit_b_32") else 1
        else:
            factory_preset = 0
    ml_setup = get_ml_setup_from_config(model_type, dataset_type=dataset_name, preset=factory_preset)
    common_transform = build_augmentation(ml_setup, dataset_name, augmentation)
    if not hasattr(ml_setup.training_data, "transform") or not hasattr(ml_setup.testing_data, "transform"):
        raise TypeError(f"{dataset_name} datasets do not expose torchvision transforms")
    ml_setup.training_data.transform = None
    ml_setup.testing_data.transform = None
    source = ConcatDataset([ml_setup.training_data, ml_setup.testing_data])
    combined = AugmentedDataset(source, common_transform, seed=args.random_seed, stochastic=augmentation > 0)
    partition_a, partition_b, permutation, excluded = _split_dataset(
        combined, seed=args.split_seed, odd_size_policy=args.odd_size_policy,
    )
    flatness_partition_a, flatness_partition_b = _flatness_partitions(
        ml_setup, dataset_name, augmentation, source, partition_a, partition_b, args.random_seed,
    )
    return DatasetBundle(
        partition_a=partition_a,
        partition_b=partition_b,
        ml_setup=ml_setup,
        dataset_name=dataset_name,
        dataset_type_name=dataset_name,
        source_examples=len(combined),
        permutation=permutation,
        excluded_indices=excluded,
        modular=False,
        model_preset=factory_preset,
        flatness_partition_a=flatness_partition_a,
        flatness_partition_b=flatness_partition_b,
    )


def load_dataset_bundle(args) -> DatasetBundle:
    dataset_name = _normalize_dataset_name(args.dataset)
    model_type = args.model_type or _default_model_type(dataset_name)
    augmentation = _parse_augmentation(args.augmentation)
    if dataset_name == "modular":
        if augmentation != 0:
            raise ValueError("modular datasets support only --augmentation none/0")
        return _load_modular_bundle(args, model_type)
    if dataset_name not in {"mnist", "cifar10", "cifar100", "noise_cifar10", *IMAGENET_DATASETS}:
        raise ValueError(f"unsupported dataset {dataset_name!r}")
    return _load_image_bundle(args, dataset_name, model_type, augmentation)


def _move_batch(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, dict):
        return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    if isinstance(batch, (tuple, list)):
        return tuple(value.to(device, non_blocking=True) if torch.is_tensor(value) else value for value in batch)
    raise TypeError(f"unsupported batch type: {type(batch).__name__}")


def _loss_and_accuracy(
    model: nn.Module,
    batch: Any,
    criterion: nn.Module,
    *,
    modular: bool,
    eq_position: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if modular:
        text = batch["text"]
        target = batch["target"]
        output = model(x=text)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        logits = logits.transpose(-2, -1)
        if eq_position is None:
            raise ValueError("eq_position is required for modular loss")
        logits = logits[..., eq_position + 1 :]
        target = target[..., eq_position + 1 :]
        loss = F.cross_entropy(logits, target, reduction="mean")
        accuracy = (logits.argmax(dim=-2) == target).all(dim=-1).float().mean()
        return loss, accuracy

    inputs, target = batch[0], batch[1]
    output = model(inputs)
    logits = output[0] if isinstance(output, (tuple, list)) else output
    loss = criterion(logits, target)
    accuracy = (logits.argmax(dim=1) == target).float().mean()
    return loss, accuracy


@torch.no_grad()
def evaluate_partition(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    *,
    device: torch.device,
    modular: bool,
    eq_position: int | None,
    tokenizer_monitor: TokenizerOutputMonitor | None = None,
    token_norm_monitor: TokenNormMonitor | None = None,
    precision: str = "bf16",
    progress_enabled: bool | None = None,
    progress_label: str | None = None,
) -> LossMetrics:
    """Evaluate a complete partition with BF16 independent of training AMP.

    Loss/accuracy totals stay on the device until the partition is complete.
    Relative-flatness measurements use a separate, full-precision path.
    """
    distributed.broadcast_model(model, buffers_only=True)
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    example_count = 0
    capture_context = tokenizer_monitor.capture() if tokenizer_monitor is not None else contextlib.nullcontext()
    norm_capture_context = token_norm_monitor.capture() if token_norm_monitor is not None else contextlib.nullcontext()
    autocast_context = inference_autocast(device, precision)
    with capture_context, norm_capture_context, autocast_context:
        batches = loader
        if progress_label is not None:
            batches = tqdm(
                loader, total=len(loader), desc=progress_label, unit="batch", dynamic_ncols=True,
                mininterval=0.5,
                disable=not (distributed.is_main() and (sys.stderr.isatty() if progress_enabled is None else progress_enabled)),
            )
        for raw_batch in batches:
            batch = _move_batch(raw_batch, device)
            loss, accuracy = _loss_and_accuracy(model, batch, criterion, modular=modular, eq_position=eq_position)
            count = int(batch["text"].shape[0] if modular else batch[0].shape[0])
            totals.add_(torch.stack((loss, accuracy)).to(dtype=torch.float64), alpha=count)
            example_count += count
    totals = torch.cat((totals, totals.new_tensor([example_count])))
    distributed.sum_tensor(totals)
    loss_sum, correct_sum, example_count = totals.cpu().tolist()
    example_count = int(example_count)
    if tokenizer_monitor is not None:
        tokenizer_monitor.synchronize()
    if token_norm_monitor is not None:
        token_norm_monitor.synchronize()
    if example_count == 0:
        raise ValueError("cannot evaluate an empty partition")
    return LossMetrics(
        loss=loss_sum / example_count, accuracy=correct_sum / example_count, examples=example_count,
        tokenizer_nonzero_ratio=tokenizer_monitor.nonzero_ratio if tokenizer_monitor is not None else None,
        token_norms=token_norm_monitor.statistics if token_norm_monitor is not None else None,
    )


def _append_tokenizer_metrics(row: dict[str, float], train: LossMetrics, val: LossMetrics) -> str:
    """Add captured ratios to the epoch row and return its log suffix."""
    if train.tokenizer_nonzero_ratio is None and val.tokenizer_nonzero_ratio is None:
        return ""
    for name, metrics in zip(TOKENIZER_RATIO_FIELDS, (train, val), strict=True):
        ratio = metrics.tokenizer_nonzero_ratio
        row[name] = ratio if ratio is not None else float("nan")
    return "".join(f" {name}={row[name]:.3E}" for name in TOKENIZER_RATIO_FIELDS)


def _record_token_norms(recorder, epoch: int, train: LossMetrics, val: LossMetrics) -> str:
    if recorder is not None:
        recorder.record(epoch, train.token_norms, val.token_norms)
    parts = []
    for role, metrics in (("train", train), ("val", val)):
        if metrics.token_norms is not None:
            patch = metrics.token_norms["patch"]
            parts.append(
                f" {role}_patch_norm_mean={patch['mean']:.4g}"
                f" {role}_patch_norm_max={patch['max']:.4g}"
                f" {role}_patch_high_norm_ratio={patch['high_norm_ratio']:.3E}"
            )
    return "".join(parts)


def _symmetric_objective(loss_a: torch.Tensor, loss_b: torch.Tensor) -> torch.Tensor:
    """Return the bounded, exchange-symmetric objective used for training.

    The mean-loss term prevents the optimizer from making one partition's
    cross-entropy arbitrarily large merely to increase the mismatch.  The
    relative-gap term still prefers unequal losses at a fixed mean loss.
    """
    total_loss = loss_a + loss_b
    relative_gap = torch.abs(loss_a - loss_b) / (total_loss + OBJECTIVE_OFFSET)
    # log((LA + LB + 1) / 2) is a stable log-mean surrogate.  Using the same
    # denominator as relative_gap keeps the analytic dataset-level gradient
    # exact and bounded near zero loss.
    return torch.log((total_loss + OBJECTIVE_OFFSET) / 2.0) - relative_gap


def _objective_value_and_coefficients(
    loss_a: float,
    loss_b: float,
    *,
    objective_mode: str = "mean_relative_gap",
    offset: float = OBJECTIVE_OFFSET,
) -> tuple[float, float, float, float, float]:
    """Evaluate the scalar objective and its derivatives w.r.t. A/B losses.

    ``train_symmetric_objective`` first evaluates each complete partition and
    then accumulates mini-batch gradients.  These analytic coefficients make
    that accumulation exactly match the gradient of the dataset-level
    objective without retaining the evaluation graph.
    """
    if objective_mode in {"difference", "normalized_two_sided"}:
        # A is the train partition and B is the val partition.  Both modes
        # report the raw difference; normalized_two_sided uses it only as a
        # diagnostic because its actual update is constructed layer by layer.
        total = loss_a + loss_b
        relative_gap = abs(loss_a - loss_b) / (total + offset)
        return loss_a - loss_b, relative_gap, 1.0, -1.0, 0.5 * total
    if objective_mode != "mean_relative_gap":
        raise ValueError(f"unknown objective mode {objective_mode!r}")
    if loss_a < 0.0 or loss_b < 0.0:
        raise ValueError("the symmetric objective requires non-negative losses")
    if offset <= 0.0:
        raise ValueError("objective offset must be positive")
    total = loss_a + loss_b
    denominator = total + offset
    absolute_difference = abs(loss_a - loss_b)
    relative_gap = absolute_difference / denominator
    objective = math.log(denominator / 2.0) - relative_gap
    mean_coefficient = 1.0 / denominator
    if loss_a > loss_b:
        gap_coefficient_a = (2.0 * loss_b + offset) / (denominator * denominator)
        gap_coefficient_b = -(2.0 * loss_a + offset) / (denominator * denominator)
    elif loss_a < loss_b:
        gap_coefficient_a = -(2.0 * loss_b + offset) / (denominator * denominator)
        gap_coefficient_b = (2.0 * loss_a + offset) / (denominator * denominator)
    else:
        # Use the symmetric zero subgradient of abs(x) at x=0.
        gap_coefficient_a = 0.0
        gap_coefficient_b = 0.0
    coefficient_a = mean_coefficient - gap_coefficient_a
    coefficient_b = mean_coefficient - gap_coefficient_b
    return objective, relative_gap, coefficient_a, coefficient_b, 0.5 * total


def _write_csv_rows(
    path: str,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Rewrite a small experiment CSV with compact numeric values."""
    if not distributed.is_main():
        return
    formatted_rows = [
        {
            fieldname: format(value, ".3E" if fieldname in TOKENIZER_RATIO_FIELDS else ".5g")
            if isinstance(value, float) else value
            for fieldname, value in row.items()
        }
        for row in rows
    ]
    with open(path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(formatted_rows)
        outfile.flush()


def _paired_epoch_progress(loader_a, loader_b, *, epoch: int, epochs: int,
                           label: str = "", enabled: bool | None = None):
    """Show paired optimizer-update speed without reading CUDA tensors.

    Auto mode keeps redirected logs plain; --progress can force rendering.
    One progress unit is one pair of A/B batches, i.e. one optimizer update.
    """
    show_progress = distributed.is_main() and (sys.stderr.isatty() if enabled is None else enabled)
    return tqdm(
        zip(loader_a, loader_b, strict=True), total=len(loader_a),
        desc=f"{label} epoch {epoch + 1}/{epochs} train".strip(),
        unit="batch", dynamic_ncols=True, mininterval=0.5, disable=not show_progress,
    )


EPOCH_TIMING_FIELDS = (
    "train_seconds", "eval_train_seconds", "eval_val_seconds",
    "baseline_eval_seconds", "checkpoint_seconds", "epoch_seconds",
)


TRAINING_LOSS_FIELDS = ("running_train_loss", "running_val_loss")


def _epoch_training_losses(totals: torch.Tensor, counts: list[int]) -> dict[str, float]:
    """Read sample-weighted training-mode losses with one transfer per epoch."""
    if distributed.enabled():
        values = torch.cat((totals, totals.new_tensor(counts)))
        distributed.sum_tensor(values)
        totals = values[:2]
        counts = values[2:].tolist()
    return {
        name: total / count if count else float("nan")
        for name, total, count in zip(TRAINING_LOSS_FIELDS, totals.cpu().tolist(), counts, strict=True)
    }


def _epoch_evaluation_start(epochs: int, last_10_percent: bool) -> int:
    """First zero-based evaluation epoch, rounding the final tenth up."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    return epochs - (epochs + 9) // 10 if last_10_percent else 0


def _finish_epoch_timing(row, *, epoch_start, train_seconds, eval_train_seconds,
                         eval_val_seconds, baseline_eval_seconds, epoch_callback):
    row.update(train_seconds=train_seconds, eval_train_seconds=eval_train_seconds,
               eval_val_seconds=eval_val_seconds, baseline_eval_seconds=baseline_eval_seconds)
    checkpoint_start = time.perf_counter()
    if epoch_callback is not None:
        epoch_callback(row)
    row["checkpoint_seconds"] = time.perf_counter() - checkpoint_start
    row["epoch_seconds"] = time.perf_counter() - epoch_start
    return "".join(f" {name}={row[name]:.2f}" for name in EPOCH_TIMING_FIELDS)


def train_symmetric_objective(
    model: nn.Module,
    loader_a: DataLoader,
    loader_b: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
    modular: bool,
    eq_position: int | None,
    optimizer,
    scheduler,
    epochs: int,
    report_interval: int,
    csv_path: str,
    objective_mode: str,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
    scaler=None,
    eval_precision: str = "bf16",
    epoch_callback: Callable[[dict[str, float]], None] | None = None,
    tokenizer_monitor: TokenizerOutputMonitor | None = None,
    token_norm_monitor: TokenNormMonitor | None = None,
    token_norm_recorder: TokenNormRecorder | None = None,
    progress_enabled: bool | None = None,
    progress_label: str = "",
    eval_loader_a: Iterable | None = None,
    eval_loader_b: Iterable | None = None,
    final_metrics_callback: Callable[[LossMetrics, LossMetrics], None] | None = None,
    eval_last_10_percent: bool = False,
    gradient_accumulate_every: int = 1,
) -> list[dict[str, float]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if gradient_accumulate_every <= 0:
        raise ValueError("gradient_accumulate_every must be positive")
    eval_loader_a = loader_a if eval_loader_a is None else eval_loader_a
    eval_loader_b = loader_b if eval_loader_b is None else eval_loader_b
    first_eval_epoch = _epoch_evaluation_start(epochs, eval_last_10_percent)
    rows: list[dict[str, float]] = []
    optimization_fields = [
        "epoch", "evaluated", "loss_a", "loss_b", "train_accuracy", "val_accuracy", "mean_loss",
        "abs_accuracy_gap", "abs_loss_gap", "relative_gap", "objective", "batches", "optimizer_updates",
        "learning_rate",
    ]
    if tokenizer_monitor is not None and tokenizer_monitor.available:
        optimization_fields.extend(TOKENIZER_RATIO_FIELDS)
    optimization_fields.extend(TRAINING_LOSS_FIELDS)
    optimization_fields.extend(EPOCH_TIMING_FIELDS)
    _write_csv_rows(csv_path, rows, optimization_fields)
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        epoch_label = f"{progress_label} epoch {epoch + 1}/{epochs}".strip()
        model.train()
        _set_loader_epoch(loader_a, epoch)
        _set_loader_epoch(loader_b, epoch)
        # First evaluate the two complete halves to obtain the objective
        # coefficients.  Each paired mini-batch is then differentiated and
        # applied immediately, matching ordinary mini-batch optimization.
        baseline_start = time.perf_counter()
        baseline_a = evaluate_partition(
            model, eval_loader_a, criterion, device=device, modular=modular, eq_position=eq_position,
            precision=eval_precision,
            progress_enabled=progress_enabled, progress_label=f"{epoch_label} baseline eval train",
        )
        baseline_b = evaluate_partition(
            model, eval_loader_b, criterion, device=device, modular=modular, eq_position=eq_position,
            precision=eval_precision,
            progress_enabled=progress_enabled, progress_label=f"{epoch_label} baseline eval val",
        )
        baseline_eval_seconds = time.perf_counter() - baseline_start
        _, _, coefficient_a, coefficient_b, _ = (
            _objective_value_and_coefficients(
                baseline_a.loss, baseline_b.loss, objective_mode=objective_mode
            )
        )
        model.train()
        batches = 0
        updates = 0
        microbatches_per_epoch = len(loader_a)
        training_loss_totals = torch.zeros(2, dtype=torch.float64, device=device)
        training_example_counts = [0, 0]
        training_start = time.perf_counter()
        for raw_a, raw_b in _paired_epoch_progress(
            loader_a, loader_b, epoch=epoch, epochs=epochs,
            label=progress_label, enabled=progress_enabled,
        ):
            distributed.broadcast_model(model, buffers_only=True)
            batch_a = _move_batch(raw_a, device)
            batch_b = _move_batch(raw_b, device)
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_enabled and device.type == "cuda" and amp_dtype is not None
                else contextlib.nullcontext()
            )
            with autocast_context:
                loss_a, _ = _loss_and_accuracy(
                    model, batch_a, criterion, modular=modular, eq_position=eq_position
                )
                loss_b, _ = _loss_and_accuracy(
                    model, batch_b, criterion, modular=modular, eq_position=eq_position
                )
            count_a = int(batch_a["text"].shape[0] if modular else batch_a[0].shape[0])
            count_b = int(batch_b["text"].shape[0] if modular else batch_b[0].shape[0])
            if count_a != count_b:
                raise RuntimeError("paired A/B batches must contain equal numbers of examples")
            training_loss_totals.add_(torch.stack((loss_a.detach(), loss_b.detach())).to(torch.float64), alpha=count_a)
            training_example_counts[0] += count_a
            training_example_counts[1] += count_b
            surrogate = (
                coefficient_a * (count_a / len(loader_a.dataset)) * loss_a
                + coefficient_b * (count_b / len(loader_b.dataset)) * loss_b
            )
            if scaler is not None:
                surrogate = scaler.scale(surrogate)
            # The surrogate is already weighted by count/partition size, so
            # summing microbatch gradients gives the group's gradient.
            if batches % gradient_accumulate_every == 0:
                optimizer.zero_grad(set_to_none=True)
            surrogate.backward()
            batches += 1
            if batches % gradient_accumulate_every and batches != microbatches_per_epoch:
                continue
            if distributed.enabled():
                parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
                for parameter in parameters:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                # The surrogate already divides by the full partition size.
                distributed.reduce_gradients([parameter.grad for parameter in parameters], average=False)
            if scaler is not None:
                previous_scale = scaler.get_scale()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None and (scaler is None or scaler.get_scale() >= previous_scale):
                scheduler.step()
            updates += 1
        running_losses = _epoch_training_losses(training_loss_totals, training_example_counts)
        train_seconds = time.perf_counter() - training_start
        evaluated = epoch >= first_eval_epoch
        eval_train_seconds = eval_val_seconds = 0.0
        if evaluated:
            eval_start = time.perf_counter()
            current_a = evaluate_partition(
                model, eval_loader_a, criterion, device=device, modular=modular, eq_position=eq_position,
                tokenizer_monitor=tokenizer_monitor, token_norm_monitor=token_norm_monitor,
                precision=eval_precision,
                progress_enabled=progress_enabled, progress_label=f"{epoch_label} eval train",
            )
            eval_train_seconds = time.perf_counter() - eval_start
            eval_start = time.perf_counter()
            current_b = evaluate_partition(
                model, eval_loader_b, criterion, device=device, modular=modular, eq_position=eq_position,
                tokenizer_monitor=tokenizer_monitor, token_norm_monitor=token_norm_monitor,
                precision=eval_precision,
                progress_enabled=progress_enabled, progress_label=f"{epoch_label} eval val",
            )
            eval_val_seconds = time.perf_counter() - eval_start
        else:
            current_a = current_b = LossMetrics(loss=float("nan"), accuracy=float("nan"), examples=0)
        if epoch == epochs - 1 and final_metrics_callback is not None:
            final_metrics_callback(current_a, current_b)
        mean_a = current_a.loss
        mean_b = current_b.loss
        (
            mean_objective,
            relative_gap_current,
            _,
            _,
            mean_loss_current,
        ) = _objective_value_and_coefficients(
            mean_a, mean_b, objective_mode=objective_mode
        ) if evaluated else (float("nan"),) * 5
        row = {
            "epoch": float(epoch),
            "evaluated": float(evaluated),
            **running_losses,
            "loss_a": mean_a,
            "loss_b": mean_b,
            "train_accuracy": current_a.accuracy,
            "val_accuracy": current_b.accuracy,
            "mean_loss": mean_loss_current,
            "abs_accuracy_gap": abs(current_a.accuracy - current_b.accuracy),
            "abs_loss_gap": abs(mean_a - mean_b),
            "relative_gap": relative_gap_current,
            "objective": mean_objective,
            "batches": float(batches),
            "optimizer_updates": float(updates),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        tokenizer_log = "".join(f" {name}={row[name]:.6g}" for name in TRAINING_LOSS_FIELDS)
        if evaluated:
            tokenizer_log += _append_tokenizer_metrics(row, current_a, current_b)
            tokenizer_log += _record_token_norms(token_norm_recorder, epoch, current_a, current_b)
        rows.append(row)
        tokenizer_log += _finish_epoch_timing(
            row, epoch_start=epoch_start, train_seconds=train_seconds,
            eval_train_seconds=eval_train_seconds, eval_val_seconds=eval_val_seconds,
            baseline_eval_seconds=baseline_eval_seconds, epoch_callback=epoch_callback,
        )
        _write_csv_rows(csv_path, rows, optimization_fields)
        if epoch % max(1, report_interval) == 0 or epoch == epochs - 1:
            if not evaluated:
                logger.info("epoch %d/%d: eval skipped lr=%.6g%s", epoch, epochs,
                            optimizer.param_groups[0]["lr"], tokenizer_log)
                continue
            logger.info(
                "epoch %d/%d: eval_train_loss=%.6g eval_val_loss=%.6g train_acc=%.6g val_acc=%.6g mean_loss=%.6g abs_gap=%.6g relative_gap=%.6g objective=%.6g lr=%.6g%s",
                epoch,
                epochs,
                mean_a,
                mean_b,
                current_a.accuracy,
                current_b.accuracy,
                mean_loss_current,
                abs(mean_a - mean_b),
                relative_gap_current,
                mean_objective,
                optimizer.param_groups[0]["lr"],
                tokenizer_log,
            )
    return rows


def _compute_partition_gradients(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
    modular: bool,
    eq_position: int | None,
    parameters: list[nn.Parameter],
) -> list[torch.Tensor]:
    """Compute the full-partition gradient without changing model parameters."""
    gradients = [torch.zeros_like(parameter) for parameter in parameters]
    example_count = len(loader.dataset)
    if example_count == 0:
        raise ValueError("cannot differentiate an empty partition")
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        loss, _ = _loss_and_accuracy(
            model, batch, criterion, modular=modular, eq_position=eq_position
        )
        count = int(batch["text"].shape[0] if modular else batch[0].shape[0])
        batch_gradients = torch.autograd.grad(
            loss, parameters, allow_unused=True, retain_graph=False
        )
        weight = count / example_count
        for accumulator, gradient in zip(gradients, batch_gradients, strict=True):
            if gradient is not None:
                accumulator.add_(gradient.detach(), alpha=weight)
    return gradients


def _compute_batch_gradients(
    model: nn.Module,
    batch: Any,
    criterion: nn.Module,
    *,
    device: torch.device,
    modular: bool,
    eq_position: int | None,
    parameters: list[nn.Parameter],
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
    scaler=None,
    loss_total: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Compute a mean-loss gradient for one mini-batch.

    Unlike :func:`_compute_partition_gradients`, this helper deliberately does
    not apply a dataset-size weight.  The caller uses the resulting gradient
    for one optimizer update immediately after the paired train/validation
    mini-batches have been evaluated.
    Optional loss totals capture detached, unscaled losses for epoch logging.
    """
    autocast_context = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_enabled and device.type == "cuda" and amp_dtype is not None
        else contextlib.nullcontext()
    )
    with autocast_context:
        loss, _ = _loss_and_accuracy(
            model, batch, criterion, modular=modular, eq_position=eq_position
        )
        if loss_total is not None:
            count = int(batch["text"].shape[0] if modular else batch[0].shape[0])
            loss_total.add_(loss.detach().to(dtype=loss_total.dtype), alpha=count)
        if scaler is not None:
            loss = scaler.scale(loss)
    gradients = torch.autograd.grad(
        loss, parameters, allow_unused=True, retain_graph=False
    )
    return [
        gradient.detach() if gradient is not None else torch.zeros_like(parameter)
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ]


def _apply_gradient_update(optimizer, scheduler, parameters, gradients, scaler=None) -> None:
    """Assign a manually constructed gradient and perform one optimizer step."""
    optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient
    if scaler is not None:
        previous_scale = scaler.get_scale()
        # ``gradients`` are still loss-scaled.  Unscaling here lets GradScaler
        # detect overflow before the optimizer update while preserving the
        # normalized train/val combination above.
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    if scheduler is not None and (scaler is None or scaler.get_scale() >= previous_scale):
        scheduler.step()


@torch.no_grad()
def _combine_normalized_train_val_gradients(
    model: nn.Module,
    train_gradients: list[torch.Tensor],
    val_gradients: list[torch.Tensor],
    *,
    parameters: list[nn.Parameter],
    train_weight: float,
    val_weight: float,
    collect_geometry: bool = True,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Build the per-parameter train-descent plus val-ascent gradient.

    For every parameter tensor, both gradients are normalized first.  The
    signed directions are then weighted and added (train minus val because the
    optimizer itself performs descent), and the result is rescaled to the
    original train-gradient norm. Zero norms are handled with device-side
    masks, not scalar host reads. Returned gradients are ready to be assigned
    to ``parameter.grad`` before one Adam/SGD step.

    Diagnostics are a device tensor whose columns follow
    ``GRADIENT_GEOMETRY_FIELDS``. On unsampled geometry epochs only the cosine
    column is returned; the optimization CSV still reports it every epoch.
    """
    if train_weight < 0.0 or val_weight < 0.0:
        raise ValueError("gradient weights must be non-negative")
    return combine_normalized_gradient_tensors(
        parameters, train_gradients, val_gradients, (train_weight, val_weight), collect_geometry,
    )


def _gradient_geometry_sample_epochs(epochs: int) -> set[int]:
    """Select at most 100 evenly spaced epochs, including the first and last."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    points = min(100, epochs)
    return {index * (epochs - 1) // max(1, points - 1) for index in range(points)}


_GRADIENT_WEIGHT_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_GRADIENT_WEIGHT_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_GRADIENT_WEIGHT_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "max": max,
    "min": min,
}


def _compile_gradient_weight_expression(expression: str, *, name: str) -> Callable[[float], float]:
    """Compile a safe arithmetic expression of ``f`` into a weight function."""
    source = str(expression).strip()
    if not source:
        raise ValueError(f"{name} must not be empty")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"invalid {name} expression {source!r}: {error.msg}") from error

    def evaluate(node: ast.AST, fraction: float) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body, fraction)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "f":
            return fraction
        if isinstance(node, ast.UnaryOp) and type(node.op) in _GRADIENT_WEIGHT_UNARY_OPERATORS:
            return _GRADIENT_WEIGHT_UNARY_OPERATORS[type(node.op)](evaluate(node.operand, fraction))
        if isinstance(node, ast.BinOp) and type(node.op) in _GRADIENT_WEIGHT_BINARY_OPERATORS:
            return _GRADIENT_WEIGHT_BINARY_OPERATORS[type(node.op)](
                evaluate(node.left, fraction), evaluate(node.right, fraction)
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _GRADIENT_WEIGHT_FUNCTIONS:
            if node.keywords:
                raise ValueError(f"{name} does not support keyword arguments")
            function = _GRADIENT_WEIGHT_FUNCTIONS[node.func.id]
            return float(function(*(evaluate(argument, fraction) for argument in node.args)))
        raise ValueError(
            f"unsupported syntax in {name} expression {source!r}; use arithmetic with f"
        )

    def weight(fraction: float) -> float:
        try:
            value = float(evaluate(tree, float(fraction)))
        except (ArithmeticError, TypeError, ValueError) as error:
            raise ValueError(f"could not evaluate {name} expression {source!r} at f={fraction:g}: {error}") from error
        if not math.isfinite(value):
            raise ValueError(f"{name} expression {source!r} returned non-finite value at f={fraction:g}")
        if value < 0.0:
            raise ValueError(f"{name} expression {source!r} returned negative value {value:g} at f={fraction:g}")
        return value

    # Validate the endpoints before training starts, while still allowing
    # expressions whose value changes between them.
    weight(0.0)
    weight(1.0)
    return weight


def train_normalized_two_sided(
    model: nn.Module,
    loader_a: DataLoader,
    loader_b: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
    modular: bool,
    eq_position: int | None,
    optimizer,
    scheduler,
    epochs: int,
    report_interval: int,
    csv_path: str,
    gradient_geometry_csv_path: str,
    train_gradient_weight_function: str = "0.5",
    val_gradient_weight_function: str = "0.5",
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
    scaler=None,
    eval_precision: str = "bf16",
    epoch_callback: Callable[[dict[str, float]], None] | None = None,
    tokenizer_monitor: TokenizerOutputMonitor | None = None,
    token_norm_monitor: TokenNormMonitor | None = None,
    token_norm_recorder: TokenNormRecorder | None = None,
    gradient_combiner: NormalizedGradientCombiner | None = None,
    progress_enabled: bool | None = None,
    progress_label: str = "",
    eval_loader_a: Iterable | None = None,
    eval_loader_b: Iterable | None = None,
    final_metrics_callback: Callable[[LossMetrics, LossMetrics], None] | None = None,
    eval_last_10_percent: bool = False,
    gradient_accumulate_every: int = 1,
) -> list[dict[str, float]]:
    """Run the normalized train-descent/val-ascent update from initialization.

    With ``gradient_accumulate_every=k`` the raw train and val gradients of
    ``k`` consecutive paired microbatches are averaged (weighted by examples)
    at one parameter state before the per-tensor normalization, so each
    update sees the gradient of the full effective batch. A shorter final
    group in an epoch is applied as its own update.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if gradient_accumulate_every <= 0:
        raise ValueError("gradient_accumulate_every must be positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    parameter_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    train_weight_function = _compile_gradient_weight_expression(
        train_gradient_weight_function, name="train gradient weight"
    )
    val_weight_function = _compile_gradient_weight_expression(
        val_gradient_weight_function, name="val gradient weight"
    )
    eval_loader_a = loader_a if eval_loader_a is None else eval_loader_a
    eval_loader_b = loader_b if eval_loader_b is None else eval_loader_b
    first_eval_epoch = _epoch_evaluation_start(epochs, eval_last_10_percent)
    rows: list[dict[str, float]] = []
    geometry_rows: list[dict[str, float | str]] = []
    geometry_sample_epochs = _gradient_geometry_sample_epochs(epochs)
    optimization_fields = [
        "epoch", "evaluated", "loss_a", "loss_b", "train_accuracy", "val_accuracy", "mean_loss",
        "abs_accuracy_gap", "abs_loss_gap", "relative_gap", "objective", "batches", "optimizer_updates",
        "learning_rate", "train_val_gradient_cosine", "train_gradient_weight", "val_gradient_weight",
    ]
    if tokenizer_monitor is not None and tokenizer_monitor.available:
        optimization_fields.extend(TOKENIZER_RATIO_FIELDS)
    geometry_fields = [
        "epoch", "parameter", "train_gradient_norm", "val_gradient_norm",
        "combined_gradient_norm", "parameter_norm", "train_val_gradient_cosine",
        "train_gradient_weight", "val_gradient_weight", "relative_train_gradient",
        "relative_combined_gradient",
    ]
    optimization_fields.extend(EPOCH_TIMING_FIELDS)
    optimization_fields.extend(TRAINING_LOSS_FIELDS)
    _write_csv_rows(csv_path, rows, optimization_fields)
    _write_csv_rows(gradient_geometry_csv_path, geometry_rows, geometry_fields)
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        epoch_label = f"{progress_label} epoch {epoch + 1}/{epochs}".strip()
        # Keep dropout and other train-time layers enabled during optimization.
        # evaluate_partition switches back to eval mode for epoch metrics.
        model.train()
        _set_loader_epoch(loader_a, epoch)
        _set_loader_epoch(loader_b, epoch)
        progress = epoch / max(1, epochs - 1)
        train_gradient_weight = train_weight_function(progress)
        val_gradient_weight = val_weight_function(progress)
        # Update immediately after every paired mini-batch.  Both gradients
        # are evaluated before the step, so they refer to the same parameter
        # state; the next pair then sees the updated model and optimizer state.
        # This matches ordinary mini-batch training and is intentionally
        # different from accumulating a full-partition gradient and taking a
        # single optimizer step per epoch. With accumulation, a group of
        # microbatches plays the role of one mini-batch.
        batches = 0
        updates = 0
        microbatches_per_epoch = len(loader_a)
        pending_train = pending_val = None
        pending_counts = [0, 0]
        collect_geometry = epoch in geometry_sample_epochs
        statistic_fields = GRADIENT_GEOMETRY_FIELDS if collect_geometry else ("train_val_gradient_cosine",)
        geometry_accumulator = GradientGeometryAccumulator(parameter_names, statistic_fields, device)
        training_loss_totals = torch.zeros(2, dtype=torch.float64, device=device)
        train_loss_total, val_loss_total = training_loss_totals.unbind()
        training_example_counts = [0, 0]
        training_start = time.perf_counter()
        for raw_a, raw_b in _paired_epoch_progress(
            loader_a, loader_b, epoch=epoch, epochs=epochs,
            label=progress_label, enabled=progress_enabled,
        ):
            distributed.broadcast_model(model, buffers_only=True)
            batch_a = _move_batch(raw_a, device)
            train_gradients = _compute_batch_gradients(
                model, batch_a, criterion, device=device, modular=modular,
                eq_position=eq_position, parameters=parameters,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype, scaler=scaler,
                loss_total=train_loss_total,
            )
            count_a = int(batch_a["text"].shape[0] if modular else batch_a[0].shape[0])
            training_example_counts[0] += count_a
            if val_gradient_weight == 0.0:
                # Avoid an unnecessary validation forward/backward pass.  In
                # particular, this keeps train-only runs from consuming extra
                # dropout RNG values and makes them a faithful train baseline.
                val_gradients = None
                count_b = 0
            else:
                batch_b = _move_batch(raw_b, device)
                val_gradients = _compute_batch_gradients(
                    model, batch_b, criterion, device=device, modular=modular,
                    eq_position=eq_position, parameters=parameters,
                    amp_enabled=amp_enabled, amp_dtype=amp_dtype, scaler=scaler,
                    loss_total=val_loss_total,
                )
                count_b = int(batch_b["text"].shape[0] if modular else batch_b[0].shape[0])
                training_example_counts[1] += count_b
            batches += 1
            if gradient_accumulate_every > 1:
                # Parameters are unchanged within a group, so the example-
                # weighted mean of the microbatch-mean gradients equals the
                # mean-loss gradient of the whole group. GradScaler's scale is
                # also constant until the update below.
                if pending_train is None:
                    pending_train = [gradient.mul_(count_a) for gradient in train_gradients]
                    pending_val = (None if val_gradients is None
                                   else [gradient.mul_(count_b) for gradient in val_gradients])
                    pending_counts = [count_a, count_b]
                else:
                    torch._foreach_add_(pending_train, train_gradients, alpha=count_a)
                    if val_gradients is not None:
                        torch._foreach_add_(pending_val, val_gradients, alpha=count_b)
                    pending_counts[0] += count_a
                    pending_counts[1] += count_b
                group_end = batches % gradient_accumulate_every == 0 or batches == microbatches_per_epoch
                if not group_end:
                    continue
                torch._foreach_div_(pending_train, float(pending_counts[0]))
                if pending_val is not None:
                    torch._foreach_div_(pending_val, float(pending_counts[1]))
                train_gradients, val_gradients = pending_train, pending_val
                pending_train = pending_val = None
            # Training shards have equal per-step sample counts on every rank.
            distributed.reduce_gradients(train_gradients)
            if val_gradients is None:
                val_gradients = [torch.zeros_like(gradient) for gradient in train_gradients]
            else:
                distributed.reduce_gradients(val_gradients)
            if gradient_combiner is None:
                combined_gradients, batch_stats = _combine_normalized_train_val_gradients(
                    model, train_gradients, val_gradients, parameters=parameters,
                    train_weight=train_gradient_weight, val_weight=val_gradient_weight,
                    collect_geometry=collect_geometry,
                )
            else:
                combined_gradients, batch_stats = gradient_combiner(
                    train_gradients, val_gradients,
                    train_weight=train_gradient_weight, val_weight=val_gradient_weight,
                    collect_geometry=collect_geometry,
                )
            _apply_gradient_update(
                optimizer, scheduler, parameters, combined_gradients, scaler=scaler
            )
            updates += 1

            # Keep the geometry CSV schema unchanged while reporting the mean
            # geometry observed over the optimizer updates in this epoch.
            geometry_accumulator.update(batch_stats)

        gradient_stats = geometry_accumulator.averages()
        for stats in gradient_stats:
            stats["train_gradient_weight"] = train_gradient_weight
            stats["val_gradient_weight"] = val_gradient_weight
        running_losses = _epoch_training_losses(training_loss_totals, training_example_counts)
        train_seconds = time.perf_counter() - training_start
        baseline_eval_seconds = 0.0
        evaluated = epoch >= first_eval_epoch
        eval_train_seconds = eval_val_seconds = 0.0
        if evaluated:
            eval_start = time.perf_counter()
            current_a = evaluate_partition(
                model, eval_loader_a, criterion, device=device, modular=modular, eq_position=eq_position,
                tokenizer_monitor=tokenizer_monitor, token_norm_monitor=token_norm_monitor,
                precision=eval_precision,
                progress_enabled=progress_enabled, progress_label=f"{epoch_label} eval train",
            )
            eval_train_seconds = time.perf_counter() - eval_start
            eval_start = time.perf_counter()
            current_b = evaluate_partition(
                model, eval_loader_b, criterion, device=device, modular=modular, eq_position=eq_position,
                tokenizer_monitor=tokenizer_monitor, token_norm_monitor=token_norm_monitor,
                precision=eval_precision,
                progress_enabled=progress_enabled, progress_label=f"{epoch_label} eval val",
            )
            eval_val_seconds = time.perf_counter() - eval_start
        else:
            current_a = current_b = LossMetrics(loss=float("nan"), accuracy=float("nan"), examples=0)
        if epoch == epochs - 1 and final_metrics_callback is not None:
            final_metrics_callback(current_a, current_b)
        mean_loss = 0.5 * (current_a.loss + current_b.loss)
        absolute_gap = abs(current_a.loss - current_b.loss)
        relative_gap = absolute_gap / (current_a.loss + current_b.loss + OBJECTIVE_OFFSET)
        objective = current_a.loss - current_b.loss
        finite_cosines = [
            float(stats["train_val_gradient_cosine"])
            for stats in gradient_stats
            if math.isfinite(float(stats["train_val_gradient_cosine"]))
        ]
        mean_gradient_cosine = (
            sum(finite_cosines) / len(finite_cosines) if finite_cosines else float("nan")
        )
        row = {
            "epoch": float(epoch),
            "evaluated": float(evaluated),
            **running_losses,
            "loss_a": current_a.loss,
            "loss_b": current_b.loss,
            "train_accuracy": current_a.accuracy,
            "val_accuracy": current_b.accuracy,
            "mean_loss": mean_loss,
            "abs_accuracy_gap": abs(current_a.accuracy - current_b.accuracy),
            "abs_loss_gap": absolute_gap,
            "relative_gap": relative_gap,
            "objective": objective,
            "batches": float(batches),
            "optimizer_updates": float(updates),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_val_gradient_cosine": mean_gradient_cosine,
            "train_gradient_weight": train_gradient_weight,
            "val_gradient_weight": val_gradient_weight,
        }
        tokenizer_log = "".join(f" {name}={row[name]:.6g}" for name in TRAINING_LOSS_FIELDS)
        if evaluated:
            tokenizer_log += _append_tokenizer_metrics(row, current_a, current_b)
            tokenizer_log += _record_token_norms(token_norm_recorder, epoch, current_a, current_b)
        rows.append(row)
        tokenizer_log += _finish_epoch_timing(
            row, epoch_start=epoch_start, train_seconds=train_seconds,
            eval_train_seconds=eval_train_seconds, eval_val_seconds=eval_val_seconds,
            baseline_eval_seconds=baseline_eval_seconds, epoch_callback=epoch_callback,
        )
        _write_csv_rows(csv_path, rows, optimization_fields)
        if epoch in geometry_sample_epochs:
            for stats in gradient_stats:
                geometry_rows.append({"epoch": float(epoch), **stats})
            _write_csv_rows(gradient_geometry_csv_path, geometry_rows, geometry_fields)
        if epoch % max(1, report_interval) == 0 or epoch == epochs - 1:
            if not evaluated:
                logger.info("epoch %d/%d: eval skipped lr=%.6g train_w=%.6g val_w=%.6g%s",
                            epoch, epochs, optimizer.param_groups[0]["lr"], train_gradient_weight,
                            val_gradient_weight, tokenizer_log)
                continue
            logger.info(
                "epoch %d/%d: eval_train_loss=%.6g eval_val_loss=%.6g train_acc=%.6g val_acc=%.6g mean_loss=%.6g abs_gap=%.6g relative_gap=%.6g objective=train-val=%.6g lr=%.6g train_w=%.6g val_w=%.6g%s",
                epoch,
                epochs,
                current_a.loss,
                current_b.loss,
                current_a.accuracy,
                current_b.accuracy,
                mean_loss,
                absolute_gap,
                relative_gap,
                objective,
                optimizer.param_groups[0]["lr"],
                train_gradient_weight,
                val_gradient_weight,
                tokenizer_log,
            )
    return rows


def _rademacher_like(parameter: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    bits = torch.randint(
        0, 2, parameter.shape, device=parameter.device, generator=generator, dtype=torch.int8
    )
    return bits.to(parameter.dtype).mul_(2).sub_(1)


def _resolve_flatness_layers(model: nn.Module, requested: str) -> list[str]:
    parameters = dict(model.named_parameters())
    matrix_names = [name for name, parameter in parameters.items() if parameter.ndim == 2 and name.endswith(".weight")]
    names = [part.strip() for part in requested.split(",") if part.strip()]
    if not names or names == ["last_matrix_weight"]:
        if not matrix_names:
            raise ValueError("model has no rank-2 weight matrix for relative flatness")
        return [matrix_names[-1]]
    if names == ["all_matrix_weights"]:
        return [name for name in matrix_names if name != "embedding.weight"]
    missing = [name for name in names if name not in parameters]
    if missing:
        raise ValueError(f"unknown relative-flatness layer(s): {missing}; available matrices={matrix_names}")
    invalid = [name for name in names if parameters[name].ndim != 2]
    if invalid:
        raise ValueError(f"relative flatness requires rank-2 parameters, got {invalid}")
    return names


def measure_relative_flatness(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
    modular: bool,
    eq_position: int | None,
    layer_names: list[str],
    hutchinson_samples: int,
    max_batches: int,
    seed: int,
) -> list[dict[str, Any]]:
    if hutchinson_samples <= 0:
        raise ValueError("relative-flatness samples must be positive")
    parameters = dict(model.named_parameters())
    selected = [parameters[name] for name in layer_names]
    estimates: dict[str, list[float]] = {name: [] for name in layer_names}
    gradient_norms: dict[str, list[float]] = {name: [] for name in layer_names}
    loss_total = 0.0
    accuracy_total = 0.0
    examples = 0
    model.eval()
    for batch_index, raw_batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = _move_batch(raw_batch, device)
        loss, accuracy = _loss_and_accuracy(model, batch, criterion, modular=modular, eq_position=eq_position)
        count = int(batch["text"].shape[0] if modular else batch[0].shape[0])
        # Keep only the graph needed for the Hessian-vector products.  The
        # last product does not need to retain it, which makes the allocator
        # release the activation graph before the next batch is loaded.
        gradients = torch.autograd.grad(loss, selected, create_graph=True, retain_graph=True, allow_unused=True)
        loss_total += float(loss.detach().item()) * count
        accuracy_total += float(accuracy.detach().item()) * count
        examples += count
        for layer_index, (name, parameter, gradient) in enumerate(zip(layer_names, selected, gradients, strict=True)):
            if gradient is None:
                continue
            gradient_norms[name].append(float(gradient.detach().norm().item()))
            generator = torch.Generator(device=parameter.device)
            generator.manual_seed(int(seed) + 1_000_003 * layer_index + batch_index)
            weight = parameter.detach()
            for sample_index in range(hutchinson_samples):
                probe = _rademacher_like(parameter, generator)
                hessian_probe = torch.autograd.grad(
                    (gradient * probe).sum(),
                    parameter,
                    retain_graph=not (layer_index == len(layer_names) - 1 and sample_index == hutchinson_samples - 1),
                    allow_unused=True,
                )[0]
                if hessian_probe is None:
                    del probe
                    continue
                relative_probe = weight @ (weight.transpose(0, 1) @ probe)
                estimates[name].append(float((relative_probe * hessian_probe).sum().detach().item()))
                del hessian_probe, relative_probe, probe
        del gradient
        # ``autograd.grad`` does not populate parameter.grad, but the local
        # tensors still keep the current batch and its graph alive until the
        # next loop iteration unless they are deleted explicitly.
        del gradients, loss, accuracy, batch
        model.zero_grad(set_to_none=True)
    if examples == 0:
        raise ValueError("relative-flatness loader produced no examples")
    results: list[dict[str, Any]] = []
    for name, parameter in zip(layer_names, selected, strict=True):
        values = torch.tensor(estimates[name], dtype=torch.float64)
        mean = float(values.mean().item()) if values.numel() else float("nan")
        std = float(values.std(unbiased=True).item()) if values.numel() > 1 else 0.0
        results.append({
            "layer": name,
            "weight_shape": list(parameter.shape),
            "examples": examples,
            "batches": max_batches if max_batches > 0 else -1,
            "hutchinson_samples": hutchinson_samples,
            "loss": loss_total / examples,
            "accuracy": accuracy_total / examples,
            "weight_norm": float(parameter.detach().norm().item()),
            "gradient_norm": sum(gradient_norms[name]) / max(1, len(gradient_norms[name])),
            "signed_estimate": mean,
            "sample_std": std,
            "standard_error": std / math.sqrt(values.numel()) if values.numel() else float("nan"),
            "positive_part": max(0.0, mean) if math.isfinite(mean) else float("nan"),
        })
    model.zero_grad(set_to_none=True)
    return results


def _measure_direction_relative_flatness(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
    modular: bool,
    eq_position: int | None,
    layers: list[str],
    train_partition_name: str,
    val_partition_name: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Measure one model using the same settings for final and top-gap points."""
    flatness_train = measure_relative_flatness(
        model, train_loader, criterion,
        device=device, modular=modular, eq_position=eq_position,
        layer_names=layers, hutchinson_samples=args.relative_flatness_samples,
        max_batches=args.relative_flatness_batches, seed=args.relative_flatness_seed + 11,
    )
    flatness_val = measure_relative_flatness(
        model, val_loader, criterion,
        device=device, modular=modular, eq_position=eq_position,
        layer_names=layers, hutchinson_samples=args.relative_flatness_samples,
        max_batches=args.relative_flatness_batches, seed=args.relative_flatness_seed + 29,
    )
    rows = [
        {"partition": partition_name, "role": role, **value}
        for role, partition_name, values in (
            ("train", train_partition_name, flatness_train),
            ("val", val_partition_name, flatness_val),
        )
        for value in values
    ]
    by_partition = {train_partition_name: flatness_train, val_partition_name: flatness_val}
    by_layer = {}
    for name in layers:
        train_value = next(item for item in flatness_train if item["layer"] == name)
        val_value = next(item for item in flatness_val if item["layer"] == name)
        by_layer[name] = {
            "mean_signed_estimate": (train_value["signed_estimate"] + val_value["signed_estimate"]) / 2,
            "mean_positive_part": (train_value["positive_part"] + val_value["positive_part"]) / 2,
        }
    report = {
        "layers": layers,
        "samples": args.relative_flatness_samples,
        "batch_size": train_loader.batch_size,
        "batches": args.relative_flatness_batches,
        "symmetric_by_layer": by_layer,
        "partitions": {
            "train": flatness_train,
            "val": flatness_val,
            "a": by_partition["a"],
            "b": by_partition["b"],
        },
    }
    measure = {
        "definition": "mean_positive_relative_flatness_across_the_two_equal_halves",
        "by_layer": {name: values["mean_positive_part"] for name, values in by_layer.items()},
        "overall_mean": sum(values["mean_positive_part"] for values in by_layer.values()) / max(1, len(by_layer)),
    }
    return report, measure, rows


def _measure_gap_checkpoint_union(
    *,
    tracker: GapCheckpointTracker,
    model: nn.Module,
    direction_name: str,
    final_epoch: int,
    final_checkpoint_path: Path,
    final_result: tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]],
    measure_model: Callable[[], tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Load each retained epoch once and restore the final model afterwards."""
    selection = tracker.selection()
    measured_points = []
    csv_rows = []
    csv_path = tracker.output_folder / f"selected_relative_flatness_{direction_name}.csv"
    json_path = tracker.output_folder / f"selected_checkpoints_{direction_name}.json"
    summary = {
        **selection,
        "relative_flatness_data": "same_loaders_and_probe_seeds_as_final_model",
        "relative_flatness_settings": {
            key: final_result[0][key] for key in ("layers", "samples", "batch_size", "batches")
        },
        "union": measured_points,
    }
    try:
        for index, point in enumerate(selection["union"]):
            logger.info(
                "measuring selected checkpoint direction=%s epoch=%d (%d/%d): accuracy_gap=%.6g loss_gap=%.6g",
                direction_name, point["epoch"], index + 1, len(selection["union"]),
                point["abs_accuracy_gap"], point["abs_loss_gap"],
            )
            if point["epoch"] == final_epoch:
                # The final model was already measured with identical settings.
                report, measure, rows = final_result
            else:
                state, _, _ = load_model_state_file(
                    str(tracker.output_folder / point["checkpoint"]), map_location="cpu"
                )
                model.load_state_dict(state, strict=True)
                del state
                report, measure, rows = measure_model()
            measured_points.append({**point, "relative_flatness": report, "misalignment_measure": measure})
            for row in rows:
                layer_summary = report["symmetric_by_layer"][row["layer"]]
                csv_rows.append({
                    **point,
                    "selected_by": "+".join(point["selected_by"]),
                    **row,
                    "symmetric_layer_signed_estimate": layer_summary["mean_signed_estimate"],
                    "symmetric_layer_positive_part": layer_summary["mean_positive_part"],
                    "misalignment_measure": measure["overall_mean"],
                })
            # Persist completed measurements even if a later Hessian pass fails.
            _write_csv_rows(str(csv_path), csv_rows, list(csv_rows[0]) if csv_rows else [])
            _write_json(str(json_path), summary)
    finally:
        state, _, _ = load_model_state_file(str(final_checkpoint_path), map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()
    # An all-non-finite run may have no qualifying checkpoints.
    if not measured_points:
        _write_csv_rows(str(csv_path), [], ["epoch", "checkpoint", "role", "layer", "signed_estimate"])
        _write_json(str(json_path), summary)
    return summary


def _build_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    device: torch.device,
    num_workers: int,
    seed: int,
    prefetch_factor: int = 4,
    loader_backend: str = "pytorch",
    augmentation_config: dict[str, Any] | None = None,
    is_train: bool = True,
    distributed_loader: bool = True,
    shuffle: bool = False,
) -> DataLoader | DaliImageNetLoader:
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive")
    if loader_backend == "dali":
        if augmentation_config is None:
            raise ValueError("DALI loading requires augmentation_config")
        return DaliImageNetLoader(
            dataset, batch_size, device=device, num_workers=num_workers, seed=seed,
            prefetch_factor=prefetch_factor, augmentation_config=augmentation_config,
            is_train=is_train, distributed_loader=distributed_loader, shuffle=shuffle,
        )
    if loader_backend != "pytorch":
        raise ValueError(f"unsupported loader backend {loader_backend!r}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    sampler = None
    if distributed_loader and distributed.enabled():
        batch_size = distributed.local_batch_size(batch_size)
        sampler = (EpochOrderSampler(len(dataset), seed=seed, shuffle=shuffle,
                                     world_size=distributed.world_size(), rank=distributed.rank())
                   if is_train else distributed.EvaluationSampler(dataset))
    elif shuffle:
        sampler = EpochOrderSampler(len(dataset), seed=seed, shuffle=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        # The order comes from the sampler: the per-epoch order of
        # misalignment/epoch_order.py, or the fixed partition order. Both are
        # independent of the train/val role, keeping the A/B exchange exact;
        # augmentation randomness is seeded by source index and epoch.
        shuffle=False,
        sampler=sampler,
        generator=generator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        # This is batches prefetched per worker, not a fraction of the data.
        # Single-process loading does not support multiprocessing prefetch.
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        **({"multiprocessing_context": "spawn"} if num_workers > 0 and distributed.enabled() else {}),
    )


class _DeviceCachedLoader:
    """Batches materialized once on the training device.

    Modular partitions are small integer tensors, so caching the fixed-order
    batches avoids per-epoch worker start-up and per-example collation. With
    ``shuffle`` the cached examples are regrouped in ``epoch_order`` instead.
    """

    def __init__(self, loader: DataLoader, device: torch.device, *, shuffle: bool = False, seed: int = 0) -> None:
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.batches = [_move_batch(batch, device) for batch in loader]
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0
        self._examples = ({key: torch.cat([batch[key] for batch in self.batches]) for key in self.batches[0]}
                          if shuffle and self.batches else None)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        if self._examples is None:
            return iter(self.batches)
        size = next(iter(self._examples.values())).shape[0]
        order = torch.tensor(epoch_order(size, seed=self.seed, epoch=self.epoch, shuffle=True),
                             device=next(iter(self._examples.values())).device)
        return iter([{key: value[order[start:start + self.batch_size]] for key, value in self._examples.items()}
                     for start in range(0, size, self.batch_size)])

    def __len__(self) -> int:
        return len(self.batches)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    if not distributed.is_main():
        return
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2, sort_keys=True, default=str)


def _git_provenance() -> dict[str, Any]:
    """Return the Git revision used for this run when the source is in Git.

    Result folders are often copied away from the checkout, so recording the
    revision in the result metadata is more useful than relying on the path to
    the source script.  Missing Git metadata must not prevent an experiment
    from running (for example, when the script is packaged or copied alone).
    """
    repository = Path(__file__).resolve().parents[1]
    provenance: dict[str, Any] = {
        "repository": str(repository),
        "commit": None,
        "describe": None,
        "branch": None,
        "dirty": None,
    }
    try:
        def run_git(*arguments: str) -> str:
            result = subprocess.run(
                ["git", *arguments],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()

        provenance["commit"] = run_git("rev-parse", "HEAD")
        provenance["describe"] = run_git("describe", "--always", "--dirty", "--long")
        provenance["branch"] = run_git("branch", "--show-current") or None
        # Ignore unrelated untracked data directories; ``dirty`` describes
        # tracked source changes relative to the recorded commit.
        provenance["dirty"] = bool(run_git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError) as error:
        provenance["error"] = f"{type(error).__name__}: {error}"
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure symmetric train/validation misalignment.")
    parser.add_argument(
        "--dataset",
        "--dataset_type",
        dest="dataset",
        type=_normalize_dataset_name,
        choices=["mnist", "cifar10", "cifar100", "noise_cifar10", "imagenet1k", "imagenet21k_p", "modular"],
        required=True,
    )
    parser.add_argument("--dataset_path", default=None, help="modular dataset folder containing train.txt/val.txt")
    parser.add_argument("-m", "--model_type", default=None, help="MLSetup model name; defaults depend on the dataset")
    parser.add_argument("--model_preset", type=int, default=None,
                        help="MLSetup/criterion preset (default: ViT ImageNet v3, ConvNeXt 21k-p 4, "
                             "other ImageNet v1, otherwise 0); "
                             "image augmentation remains controlled by --augmentation")
    parser.add_argument("--augmentation", "--data_augmentation", default="none", help="none/0 or an integer augmentation level")
    parser.add_argument("--label_smoothing", type=float, default=None,
                        help="override the preset criterion with CrossEntropyLoss(label_smoothing=x) for training, "
                             "evaluation and flatness (default: keep the --model_preset criterion; image datasets only)")
    parser.add_argument(
        "--objective",
        choices=["normalized_two_sided", "difference", "mean_relative_gap"],
        default="normalized_two_sided",
        help="normalized train-descent/val-ascent update, raw train-val difference, or the previous symmetric objective",
    )
    parser.add_argument("--modulus", type=int, default=None)
    parser.add_argument(
        "--split_seed", type=int, default=None,
        help="seed for shuffling the combined dataset; defaults to the resolved --random_seed",
    )
    parser.add_argument(
        "--shuffle", action=argparse.BooleanOptionalAction, default=True,
        help="reshuffle each half's training order every epoch (default); --no-shuffle keeps the fixed "
             "partition order. Evaluation and flatness always use the fixed order",
    )
    parser.add_argument(
        "--odd_size_policy",
        choices=["drop", "error"],
        default="drop",
        help="when the combined source count is odd, drop one shuffled example (default) or fail",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="seed for initialization, dropout and augmentation; omitted draws a fresh seed per run (recorded in summary.json)",
    )
    parser.add_argument("--batch_size", type=int, default=None,
                        help="global paired mini-batch size per partition across all GPUs")
    parser.add_argument("--gradient_accumulate_every", type=int, default=1,
                        help="paired microbatches averaged per optimizer update (default: 1); the effective "
                             "batch per partition is --batch_size times this value, and LR schedules count updates")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--optimizer_preset", type=int, default=0)
    parser.add_argument("--optimizer", choices=["auto", "sgd", "adam", "adamw"], default="auto")
    parser.add_argument(
        "--scheduler",
        "--lr_schedule",
        "--learning_rate_schedule",
        dest="scheduler",
        choices=["fixed", "cosine", "auto", "none", "onecycle"],
        default="fixed",
        help="learning-rate schedule: fixed (default), cosine annealing, or legacy auto/none/onecycle modes",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=None,
        help="linear warmup epochs for fixed/cosine schedules (default: dataset preset, 10 for modular)",
    )
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=8,
                        help="loader workers / DALI decoder threads per pipeline (default: 8)")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="batch size for full-partition epoch evaluation; defaults to training batch size")
    parser.add_argument("--eval_precision", choices=["bf16", "fp32"], default="bf16",
                        help="forward-only evaluation precision, independent of --amp; "
                             "bf16 uses native CUDA support and falls back to fp32 elsewhere")
    parser.add_argument("--eval_num_workers", type=int, default=None,
                        help="decoder/loader workers for epoch evaluation; defaults to --num_workers")
    parser.add_argument("--eval_last_10_percent", "--eval-last-10-percent", action="store_true",
                        help="only run epoch-end train/val evaluation in the final ceil(epochs/10) epochs; "
                             "symmetric-objective baseline evaluations required for updates are retained")
    parser.add_argument(
        "--loader_backend", "--loader-backend", choices=["auto", "pytorch", "dali"], default="auto",
        help="auto (default): DALI for ImageNet, PyTorch otherwise; DALI requires CUDA and NVIDIA DALI",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="PyTorch batches prefetched per worker, or DALI pipeline prefetch depth (default: 4)",
    )
    parser.add_argument("--report_interval", type=int, default=1)
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=None,
        help="show per-epoch tqdm progress, batch/s and ETA (default: automatic for interactive terminals)",
    )
    parser.add_argument(
        "--token_norms", action=argparse.BooleanOptionalAction, default=True,
        help="record pre-LayerNorm token norms during ViT epoch evaluation (default: enabled for ViT)",
    )
    parser.add_argument(
        "--token_norm_threshold", type=float, default=150.0,
        help="fixed L2 threshold for reporting high-norm token ratios; use the same value for both models",
    )
    parser.add_argument(
        "--train_gradient_weight_function",
        "--train_gradient_weight",
        "--train-gradient-weight-function",
        dest="train_gradient_weight_function",
        default="0.5",
        help="arithmetic expression for train gradient weight as a function of f; f is the normalized epoch fraction",
    )
    parser.add_argument(
        "--val_gradient_weight_function",
        "--val_gradient_weight",
        "--val-gradient-weight-function",
        dest="val_gradient_weight_function",
        default="0.5",
        help="arithmetic expression for val gradient weight as a function of f; f is the normalized epoch fraction",
    )
    parser.add_argument("--relative_flatness_layers", default="last_matrix_weight")
    parser.add_argument("--relative_flatness_samples", type=int, default=8)
    parser.add_argument(
        "--relative_flatness_batches",
        type=int,
        default=-1,
        help="number of batches for relative flatness; -1 uses all batches",
    )
    parser.add_argument(
        "--relative_flatness_batch_size",
        type=int,
        default=1024,
        help="microbatch size used only for second-order relative-flatness measurement",
    )
    parser.add_argument("--relative_flatness_seed", type=int, default=2718)
    parser.add_argument("--relative_flatness", action=argparse.BooleanOptionalAction, default=None,
                        help="measure flatness after training (default: on for single device, off for torchrun)")
    parser.add_argument("--skip_relative_flatness", dest="relative_flatness", action="store_false",
                        default=argparse.SUPPRESS, help="save models and defer flatness measurement")
    parser.add_argument(
        "--flatness_from_checkpoint",
        default=None,
        help="skip training and measure flatness for an existing saved model checkpoint",
    )
    parser.add_argument("--output_folder_name", "-o", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--torch_compile",
        "--compile",
        dest="torch_compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile the training model with torch.compile (default: enabled; use --no-compile to disable)",
    )
    parser.add_argument(
        "--compile_gradient_combination", "--compile-gradient-combination",
        dest="compile_gradient_combination",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="fuse CUDA normalized gradient combination with torch.compile (default: follows --torch_compile); use --no-compile-gradient-combination for eager combination",
    )
    parser.add_argument(
        "--amp",
        dest="amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA training automatic mixed precision (default: enabled; use --no-amp to disable; ignored on CPU); "
             "evaluation uses --eval_precision and relative flatness stays full precision",
    )
    parser.add_argument(
        "--directions",
        choices=["both", "a_as_train_b_as_val", "b_as_train_a_as_val"],
        default="both",
        help="train/val orientations to run (default: both)",
    )
    parser.add_argument("--num_threads", type=int, default=None)
    parser.add_argument("--m_nlayer", type=int, default=None)
    parser.add_argument("--m_n_heads", type=int, default=None)
    parser.add_argument("--m_d_model", type=int, default=None)
    parser.add_argument("--m_context_len", type=int, default=None)
    parser.add_argument("--m_pos_encoding", choices=["default", "trainable"], default=None)
    return parser.parse_args()


def _run_direction(
    *,
    direction_name: str,
    train_partition_name: str,
    val_partition_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    flatness_train_loader: DataLoader,
    flatness_val_loader: DataLoader,
    bundle: DatasetBundle,
    model: nn.Module,
    initial_state: dict[str, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
    epochs: int,
    args: argparse.Namespace,
    output_folder: Path,
    model_type_name: str,
    dataset_type_name: str,
    eval_train_loader: Iterable | None = None,
    eval_val_loader: Iterable | None = None,
) -> dict[str, Any]:
    """Run one train/val orientation from the same initial state."""
    set_seed(args.random_seed + distributed.rank())
    model.load_state_dict(initial_state, strict=True)
    optimizer, scheduler, _ = build_optimizer_and_scheduler(
        bundle.ml_setup,
        model,
        bundle.partition_a,
        batch_size=batch_size,
        preset=args.optimizer_preset,
        epochs=epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        gradient_accumulate_every=getattr(args, "gradient_accumulate_every", 1),
        warmup_epochs=args.warmup_epochs,
    )
    runtime_model = model
    compile_enabled = False
    if args.torch_compile:
        if hasattr(torch, "compile"):
            try:
                runtime_model = torch.compile(model)
                compile_enabled = True
                logger.info("torch.compile enabled for direction=%s", direction_name)
            except Exception as exc:
                logger.warning("torch.compile unavailable; using eager model: %s", exc)
        else:
            logger.warning("torch.compile is not available in this PyTorch build")

    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = None
    scaler = None
    if amp_enabled:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if amp_dtype == torch.float16:
            try:
                scaler = torch.amp.GradScaler("cuda", enabled=True)
            except (AttributeError, TypeError):
                scaler = torch.cuda.amp.GradScaler(enabled=True)
        logger.info(
            "CUDA AMP enabled for training, direction=%s: dtype=%s grad_scaler=%s; relative flatness remains full precision",
            direction_name,
            amp_dtype,
            scaler is not None,
        )
    else:
        logger.info("CUDA AMP disabled for direction=%s", direction_name)
    eval_precision = getattr(args, "eval_precision", "bf16")
    eval_dtype = inference_dtype(device, eval_precision)
    logger.info("evaluation precision, direction=%s: requested=%s effective=%s",
                direction_name, eval_precision, eval_dtype or torch.float32)
    logger.info(
        "starting direction=%s: train=partition_%s val=partition_%s",
        direction_name,
        train_partition_name,
        val_partition_name,
    )
    if getattr(args, "eval_last_10_percent", False):
        first_eval_epoch = _epoch_evaluation_start(epochs, True)
        logger.info("epoch-end evaluation restricted to epochs %d-%d (1-based); gap checkpoints use these epochs",
                    first_eval_epoch + 1, epochs)
    optimization_path = output_folder / f"optimization_{direction_name}.csv"
    geometry_path = output_folder / f"gradient_geometry_{direction_name}.csv"
    variance_path = output_folder / f"weight_variance_{direction_name}.csv"
    flatness_path = output_folder / f"relative_flatness_{direction_name}.csv"
    initial_model_path = output_folder / f"initial_{direction_name}.model.pt"
    final_model_path = output_folder / f"final_{direction_name}.model.pt"
    if distributed.is_main():
        save_model_state(
            str(initial_model_path),
            {name: value.detach().cpu().clone() for name, value in initial_state.items()},
            model_type_name,
            dataset_type_name,
        )
    gap_tracker = GapCheckpointTracker(
        model, output_folder, direction_name, model_type_name, dataset_type_name,
    )
    variance_recorder = EpochWeightVarianceRecorder(model, variance_path)

    def record_epoch(row: dict[str, float]) -> None:
        variance_recorder.record(int(row["epoch"]))
        if row["evaluated"]:
            gap_tracker.observe(row)

    # torch.compile is lazy: install the fixed hook before its first forward.
    # Captures are active only in epoch-end/final evaluation, never during the
    # training forwards/backwards or second-order flatness measurements.
    tokenizer_monitor = TokenizerOutputMonitor(model)
    token_norm_monitor = TokenNormMonitor(
        model, threshold=getattr(args, "token_norm_threshold", 150.0),
        enabled=getattr(args, "token_norms", True),
    )
    token_norm_path = output_folder / f"token_norms_{direction_name}.csv"
    token_norm_recorder = TokenNormRecorder(token_norm_path, token_norm_monitor)
    gradient_combiner = None
    final_metrics = None

    def retain_final_metrics(train, val):
        nonlocal final_metrics
        final_metrics = (train, val)

    try:
        if args.objective == "normalized_two_sided":
            combine_compile = getattr(args, "compile_gradient_combination", None)
            gradient_combiner = NormalizedGradientCombiner(
                [parameter for parameter in runtime_model.parameters() if parameter.requires_grad],
                enabled=args.torch_compile if combine_compile is None else combine_compile,
            )
            rows = train_normalized_two_sided(
                runtime_model,
                train_loader,
                val_loader,
                criterion,
                device=device,
                modular=bundle.modular,
                eq_position=bundle.eq_position,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=epochs,
                report_interval=args.report_interval,
                csv_path=str(optimization_path),
                gradient_geometry_csv_path=str(geometry_path),
                eval_precision=eval_precision,
                train_gradient_weight_function=args.train_gradient_weight_function,
                val_gradient_weight_function=args.val_gradient_weight_function,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                scaler=scaler,
                epoch_callback=record_epoch,
                tokenizer_monitor=tokenizer_monitor, token_norm_monitor=token_norm_monitor,
                token_norm_recorder=token_norm_recorder,
                gradient_combiner=gradient_combiner,
                progress_enabled=getattr(args, "progress", None), progress_label=direction_name,
                eval_loader_a=eval_train_loader, eval_loader_b=eval_val_loader,
                final_metrics_callback=retain_final_metrics,
                eval_last_10_percent=getattr(args, "eval_last_10_percent", False),
                gradient_accumulate_every=getattr(args, "gradient_accumulate_every", 1),
            )
        else:
            rows = train_symmetric_objective(
                runtime_model,
                train_loader,
                val_loader,
                criterion,
                device=device,
                modular=bundle.modular,
                eq_position=bundle.eq_position,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=epochs,
                report_interval=args.report_interval,
                csv_path=str(optimization_path),
                objective_mode=args.objective,
                eval_precision=eval_precision,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                scaler=scaler,
                epoch_callback=record_epoch,
                tokenizer_monitor=tokenizer_monitor, token_norm_monitor=token_norm_monitor,
                token_norm_recorder=token_norm_recorder,
                progress_enabled=getattr(args, "progress", None), progress_label=direction_name,
                eval_loader_a=eval_train_loader, eval_loader_b=eval_val_loader,
                final_metrics_callback=retain_final_metrics,
                eval_last_10_percent=getattr(args, "eval_last_10_percent", False),
                gradient_accumulate_every=getattr(args, "gradient_accumulate_every", 1),
            )
        if distributed.is_main():
            save_model_state(str(final_model_path), model.state_dict(), model_type_name, dataset_type_name)

        if final_metrics is None:
            raise RuntimeError("training did not provide final epoch evaluation metrics")
        final_train, final_val = final_metrics

    finally:
        tokenizer_monitor.close()
        token_norm_monitor.close()
        # Do not keep the training decoder/preprocessing pools resident while
        # building the separate second-order flatness microbatch pipelines.
        close_dali_loaders(train_loader, val_loader, eval_train_loader, eval_val_loader)
    do_flatness = getattr(args, "relative_flatness", True)
    if do_flatness:
        layers = _resolve_flatness_layers(model, args.relative_flatness_layers)

        def measure_current_model():
            try:
                return _measure_direction_relative_flatness(
                    model, flatness_train_loader, flatness_val_loader, criterion,
                    device=device, modular=bundle.modular, eq_position=bundle.eq_position,
                    layers=layers, train_partition_name=train_partition_name,
                    val_partition_name=val_partition_name, args=args,
                )
            finally:
                close_dali_loaders(flatness_train_loader, flatness_val_loader)

        final_flatness_result = measure_current_model()
        relative_flatness, misalignment_measure, flatness_rows = final_flatness_result
        _write_csv_rows(str(flatness_path), flatness_rows,
                        list(flatness_rows[0].keys()) if flatness_rows else [])
        gap_checkpoints = _measure_gap_checkpoint_union(
            tracker=gap_tracker, model=model, direction_name=direction_name,
            final_epoch=epochs - 1, final_checkpoint_path=final_model_path,
            final_result=final_flatness_result, measure_model=measure_current_model,
        )
    else:
        relative_flatness = misalignment_measure = None
        gap_checkpoints = {**gap_tracker.selection(), "relative_flatness_status": "deferred"}
        _write_json(str(output_folder / f"selected_checkpoints_{direction_name}.json"), gap_checkpoints)
    final_objective, final_relative_gap, _, _, final_mean_loss = _objective_value_and_coefficients(
        final_train.loss, final_val.loss, objective_mode=args.objective
    )
    final_by_partition = {
        "a": asdict(final_train if train_partition_name == "a" else final_val),
        "b": asdict(final_train if train_partition_name == "b" else final_val),
    }
    return {
        "direction": direction_name,
        "train_partition": train_partition_name,
        "val_partition": val_partition_name,
        "dataset": bundle.dataset_name,
        "dataset_type": dataset_type_name,
        "source_examples": bundle.source_examples,
        "half_examples": len(bundle.partition_a),
        "split_seed": args.split_seed,
        "random_seed": args.random_seed,
        "augmentation": _parse_augmentation(args.augmentation),
        "augmentation_config": describe_augmentation(args.dataset, args.augmentation),
        "relative_flatness_augmentation_config": describe_augmentation(args.dataset, 0),
        "data_order": {
            # Callers that build their own loaders may pass args without it.
            "training_shuffle": getattr(args, "shuffle", False),
            "training_order": ORDER_RNG if getattr(args, "shuffle", False) else "fixed partition order",
            "evaluation_and_flatness_order": "fixed partition order",
        },
        "excluded_source_indices": bundle.excluded_indices.tolist(),
        "model_type": model_type_name,
        "model_preset": bundle.model_preset,
        "criterion": {"type": type(criterion).__name__,
                      "label_smoothing": getattr(criterion, "label_smoothing", None)},
        "device": str(device),
        "runtime": {
            "world_size": distributed.world_size(),
            "local_batch_size": distributed.local_batch_size(batch_size),
            "batch_size_definition": "global_per_partition_before_accumulation",
            "gradient_accumulate_every": getattr(args, "gradient_accumulate_every", 1),
            "effective_batch_size": batch_size * getattr(args, "gradient_accumulate_every", 1),
            "training_sampler_padding": (math.ceil(len(bundle.partition_a) / distributed.world_size())
                                         * distributed.world_size() - len(bundle.partition_a)),
            "loader": {
                "train": loader_runtime_info(train_loader),
                "val": loader_runtime_info(val_loader),
                "eval_train": loader_runtime_info(eval_train_loader if eval_train_loader is not None else train_loader),
                "eval_val": loader_runtime_info(eval_val_loader if eval_val_loader is not None else val_loader),
                "flatness_train": loader_runtime_info(flatness_train_loader) if do_flatness else None,
                "flatness_val": loader_runtime_info(flatness_val_loader) if do_flatness else None,
            },
            "torch_compile": compile_enabled,
            "gradient_combination": gradient_combiner.runtime_info() if gradient_combiner is not None else None,
            "amp": amp_enabled,
            "evaluation_amp": eval_dtype is not None,
            "eval_precision": eval_precision,
            "evaluation_amp_dtype": str(eval_dtype) if eval_dtype is not None else None,
            "relative_flatness_amp": False,
            "amp_dtype": str(amp_dtype) if amp_dtype is not None else None,
            "grad_scaler": scaler is not None,
        },
        "batch_size": batch_size,
        "epochs": epochs,
        "epoch_evaluation": {
            "policy": "last_10_percent" if getattr(args, "eval_last_10_percent", False) else "every_epoch",
            "first_epoch": _epoch_evaluation_start(epochs, getattr(args, "eval_last_10_percent", False)),
            "last_epoch": epochs - 1,
            "epoch_definition": "zero_based_epoch_after_updates",
            "gap_checkpoint_scope": "evaluated_epochs_only",
        },
        "objective_mode": args.objective,
        "gradient_weight_functions": {
            "train": args.train_gradient_weight_function,
            "val": args.val_gradient_weight_function,
            "variable": "f=epoch/(epochs-1) for epochs>1, otherwise 0",
        },
        "optimizer": getattr(optimizer, "misalignment_config", None),
        "files": {
            "optimization": optimization_path.name,
            "gradient_geometry": geometry_path.name if args.objective == "normalized_two_sided" else None,
            "weight_variance": variance_path.name,
            "relative_flatness": flatness_path.name if do_flatness else None,
            "initial_model": initial_model_path.name,
            "final_model": final_model_path.name,
            "gap_checkpoint_selection": str(gap_tracker.manifest_path.relative_to(output_folder)),
            "selected_relative_flatness": f"selected_relative_flatness_{direction_name}.csv" if do_flatness else None,
            "selected_checkpoint_summary": f"selected_checkpoints_{direction_name}.json",
            "token_norms": token_norm_path.name if token_norm_monitor.available else None,
        },
        "final": {
            "train": asdict(final_train),
            "val": asdict(final_val),
            "partition_a": final_by_partition["a"],
            "partition_b": final_by_partition["b"],
            "mean_loss": final_mean_loss,
            "absolute_accuracy_gap": abs(final_train.accuracy - final_val.accuracy),
            "absolute_loss_gap": abs(final_train.loss - final_val.loss),
            "relative_gap": final_relative_gap,
            "objective": final_objective,
        },
        "relative_flatness": relative_flatness,
        "relative_flatness_status": "measured" if do_flatness else "deferred",
        "misalignment_measure": misalignment_measure,
        "gap_checkpoints": gap_checkpoints,
        "weight_variance": {
            "layers": variance_recorder.layer_names,
            "correction": 1,
            "format": ".3E (4 significant digits)",
            "epoch_definition": "zero_based_epoch_after_updates",
        },
        "tokenizer_diagnostics": {
            "enabled": tokenizer_monitor.available,
            "definition": "nonzero_output_elements/total_output_elements_before_positional_embeddings",
            "measurement": "epoch_end_full_partition_eval_mode",
            "format": ".3E (4 significant digits)",
        },
        "final_metrics_source": "last_epoch_full_partition_evaluation",
        "token_norm_diagnostics": token_norm_monitor.metadata(),
        "optimization_last_row": rows[-1] if rows else None,
    }


def _measure_flatness_from_checkpoint(
    *,
    checkpoint_path: Path,
    output_folder: Path,
    bundle: DatasetBundle,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    flatness_loader_a: DataLoader,
    flatness_loader_b: DataLoader,
    args: argparse.Namespace,
    model_type_name: str,
    dataset_type_name: str,
) -> None:
    """Measure flatness for an already trained A-as-train/B-as-val model.

    This path deliberately does not construct an optimizer or execute any
    training epochs.  It is used when a long training run already produced
    ``final_a_as_train_b_as_val.model.pt`` and only its flatness is missing.
    """
    state, checkpoint_model_type, checkpoint_dataset_type = load_model_state_file(
        str(checkpoint_path), map_location="cpu"
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    layers = _resolve_flatness_layers(model, args.relative_flatness_layers)
    flatness_a = measure_relative_flatness(
        model,
        flatness_loader_a,
        criterion,
        device=device,
        modular=bundle.modular,
        eq_position=bundle.eq_position,
        layer_names=layers,
        hutchinson_samples=args.relative_flatness_samples,
        max_batches=args.relative_flatness_batches,
        seed=args.relative_flatness_seed + 11,
    )
    flatness_b = measure_relative_flatness(
        model,
        flatness_loader_b,
        criterion,
        device=device,
        modular=bundle.modular,
        eq_position=bundle.eq_position,
        layer_names=layers,
        hutchinson_samples=args.relative_flatness_samples,
        max_batches=args.relative_flatness_batches,
        seed=args.relative_flatness_seed + 29,
    )
    rows = []
    for partition_name, values in (("a", flatness_a), ("b", flatness_b)):
        rows.extend({"partition": partition_name, "role": "checkpoint_flatness", **value} for value in values)
    flatness_path = output_folder / "relative_flatness_a_as_train_b_as_val.csv"
    _write_csv_rows(str(flatness_path), rows, list(rows[0].keys()) if rows else [])
    by_layer = {}
    for name in layers:
        value_a = next(item for item in flatness_a if item["layer"] == name)
        value_b = next(item for item in flatness_b if item["layer"] == name)
        by_layer[name] = {
            "partition_a_signed_estimate": value_a["signed_estimate"],
            "partition_b_signed_estimate": value_b["signed_estimate"],
            "mean_signed_estimate": (value_a["signed_estimate"] + value_b["signed_estimate"]) / 2,
            "mean_positive_part": (value_a["positive_part"] + value_b["positive_part"]) / 2,
        }
    summary = {
        "schema_version": 1,
        "mode": "flatness_from_checkpoint",
        "git_provenance": _git_provenance(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_model_type": checkpoint_model_type,
        "checkpoint_dataset_type": checkpoint_dataset_type,
        "model_type": model_type_name,
        "dataset": bundle.dataset_name,
        "dataset_type": dataset_type_name,
        "source_examples": bundle.source_examples,
        "half_examples": len(bundle.partition_a),
        "split_seed": args.split_seed,
        "random_seed": args.random_seed,
        "loader": {
            "partition_a": loader_runtime_info(flatness_loader_a),
            "partition_b": loader_runtime_info(flatness_loader_b),
        },
        "relative_flatness": {
            "layers": layers,
            "samples": args.relative_flatness_samples,
            "batch_size": flatness_loader_a.batch_size,
            "batches": args.relative_flatness_batches,
            "partitions": {"a": flatness_a, "b": flatness_b},
            "by_layer": by_layer,
        },
        "files": {"checkpoint": checkpoint_path.name, "relative_flatness": flatness_path.name},
    }
    _write_json(str(output_folder / "flatness_from_checkpoint.json"), summary)
    logger.info("flatness computed from checkpoint=%s; results written to %s", checkpoint_path, output_folder)


def _resolve_criterion(preset_criterion: nn.Module | None, label_smoothing: float | None) -> nn.Module:
    """Return the preset criterion, or plain CE with an explicit smoothing."""
    criterion = preset_criterion or nn.CrossEntropyLoss()
    if label_smoothing is None:
        return criterion
    if type(criterion) is not nn.CrossEntropyLoss:
        raise ValueError(f"--label_smoothing requires a CrossEntropyLoss preset criterion, got {type(criterion).__name__}")
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def main() -> None:
    args = parse_args()
    distributed_device = distributed.initialize(use_cpu=args.device == "cpu")
    if args.relative_flatness is None:
        args.relative_flatness = not distributed.enabled()
    if distributed.enabled() and (args.relative_flatness or args.flatness_from_checkpoint):
        raise ValueError("Multi-GPU execution supports training/evaluation; run flatness separately on one GPU")
    if not distributed.is_main():
        args.progress = False
    if args.dataset == "modular" and not args.dataset_path:
        raise ValueError("--dataset_path is required for --dataset modular")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.gradient_accumulate_every <= 0:
        raise ValueError("--gradient_accumulate_every must be positive")
    if args.label_smoothing is not None:
        if not 0.0 <= args.label_smoothing < 1.0:
            raise ValueError("--label_smoothing must be in [0, 1)")
        if args.dataset == "modular":
            raise ValueError("--label_smoothing is not supported for modular datasets")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        raise ValueError("--eval_batch_size must be positive")
    if args.eval_num_workers is not None and args.eval_num_workers < 0:
        raise ValueError("--eval_num_workers must be non-negative")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch_factor must be positive")
    if args.relative_flatness_batches == 0 or args.relative_flatness_samples <= 0:
        raise ValueError("relative-flatness batches must be non-zero and samples must be positive")
    if args.relative_flatness_batch_size <= 0:
        raise ValueError("--relative_flatness_batch_size must be positive")
    if args.num_threads is not None:
        torch.set_num_threads(max(1, args.num_threads))
    setup_logging(logger, "main")
    if not distributed.is_main():
        logger.setLevel(logging.WARNING)
    if args.random_seed is None:
        # Each invocation gets an independent seed.  It is still applied
        # explicitly so both directions share the same dropout stream and the
        # run can be reproduced from the recorded value.
        args.random_seed = secrets.randbelow(2**31)
        logger.info("random seed not given; drew random_seed = %d", args.random_seed)
    else:
        logger.info("random_seed = %d", args.random_seed)
    args.random_seed = distributed.shared_seed(args.random_seed)
    if args.split_seed is None:
        args.split_seed = args.random_seed
    logger.info("split_seed = %d", args.split_seed)
    set_seed(args.random_seed)
    git_provenance = _git_provenance()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    if distributed_device is not None:
        device = distributed_device
    logger.info("device = %s", device)
    loader_backend = resolve_loader_backend(args.dataset, args.loader_backend)
    if loader_backend == "dali" and device.type != "cuda":
        raise ValueError("DALI ImageNet loading requires CUDA; use --loader_backend pytorch on CPU")
    augmentation_config = describe_augmentation(args.dataset, args.augmentation)
    logger.info("loader backend = %s (requested=%s)", loader_backend, args.loader_backend)

    bundle = load_dataset_bundle(args)
    model = bundle.ml_setup.model.to(device)
    distributed.broadcast_model(model)
    criterion = _resolve_criterion(bundle.ml_setup.criterion, args.label_smoothing)
    logger.info("criterion = %s(label_smoothing=%s)", type(criterion).__name__,
                getattr(criterion, "label_smoothing", None))
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif bundle.modular:
        # The shuffled partitions are equal-sized; use a capped full batch
        # for modular data, matching the grokking experiment scripts.
        batch_size = min(len(bundle.partition_a), 65536)
    else:
        batch_size = int(getattr(bundle.ml_setup, "default_batch_size", 64) or 64)
    # Modular batches are cached on the device and reordered there.
    sampler_shuffle = args.shuffle and not bundle.modular
    loader_a = _build_loader(
        bundle.partition_a, batch_size, device=device, num_workers=args.num_workers,
        seed=args.random_seed + 101, prefetch_factor=args.prefetch_factor,
        loader_backend=loader_backend, augmentation_config=augmentation_config, shuffle=sampler_shuffle,
    )
    loader_b = _build_loader(
        bundle.partition_b, batch_size, device=device, num_workers=args.num_workers,
        seed=args.random_seed + 202, prefetch_factor=args.prefetch_factor,
        loader_backend=loader_backend, augmentation_config=augmentation_config, shuffle=sampler_shuffle,
    )
    if bundle.modular:
        loader_a = _DeviceCachedLoader(loader_a, device, shuffle=args.shuffle, seed=args.random_seed + 101)
        loader_b = _DeviceCachedLoader(loader_b, device, shuffle=args.shuffle, seed=args.random_seed + 202)
    eval_batch_size = args.eval_batch_size or batch_size
    eval_num_workers = args.num_workers if args.eval_num_workers is None else args.eval_num_workers
    eval_loader_a, eval_loader_b = loader_a, loader_b
    if eval_batch_size != batch_size or eval_num_workers != args.num_workers or distributed.enabled():
        eval_loader_a = _build_loader(
            bundle.partition_a, eval_batch_size, device=device, num_workers=eval_num_workers,
            seed=args.random_seed + 101, prefetch_factor=args.prefetch_factor,
            loader_backend=loader_backend, augmentation_config=augmentation_config,
            is_train=False,
        )
        eval_loader_b = _build_loader(
            bundle.partition_b, eval_batch_size, device=device, num_workers=eval_num_workers,
            seed=args.random_seed + 202, prefetch_factor=args.prefetch_factor,
            loader_backend=loader_backend, augmentation_config=augmentation_config,
            is_train=False,
        )
    logger.info("epoch evaluation batch_size=%d num_workers=%d", eval_batch_size, eval_num_workers)
    # Training on the whole modular partition is intentional, but using the
    # same multi-million-example batch for create_graph=True flatness would
    # retain all forward activations and trigger a CUDA OOM.  Flatness is an
    # average over deterministic microbatches instead.
    flatness_batch_size = min(batch_size, args.relative_flatness_batch_size)
    # Flatness always sees the deterministic level-0 transform, never the
    # stochastic training augmentation.
    flatness_augmentation_config = describe_augmentation(args.dataset, 0)
    flatness_loader_a = _build_loader(
        bundle.partition_a if bundle.flatness_partition_a is None else bundle.flatness_partition_a,
        flatness_batch_size,
        device=device,
        num_workers=args.num_workers,
        seed=args.random_seed + 303,
        prefetch_factor=args.prefetch_factor,
        loader_backend=loader_backend, augmentation_config=flatness_augmentation_config,
        is_train=False, distributed_loader=False,
    ) if args.relative_flatness or args.flatness_from_checkpoint else None
    flatness_loader_b = _build_loader(
        bundle.partition_b if bundle.flatness_partition_b is None else bundle.flatness_partition_b,
        flatness_batch_size,
        device=device,
        num_workers=args.num_workers,
        seed=args.random_seed + 404,
        prefetch_factor=args.prefetch_factor,
        loader_backend=loader_backend, augmentation_config=flatness_augmentation_config,
        is_train=False, distributed_loader=False,
    ) if args.relative_flatness or args.flatness_from_checkpoint else None
    model_type_name = bundle.ml_setup.model_type.name
    dataset_type_name = bundle.dataset_type_name
    if args.flatness_from_checkpoint is not None:
        checkpoint_path = Path(args.flatness_from_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"flatness checkpoint does not exist: {checkpoint_path}")
        if args.output_folder_name is None:
            output_folder = checkpoint_path.parent
        else:
            output_folder = Path(args.output_folder_name)
            if not output_folder.is_absolute():
                output_folder = Path.cwd() / output_folder
            output_folder = output_folder.resolve()
        output_folder.mkdir(parents=True, exist_ok=True)
        (output_folder / "flatness_command.txt").write_text(" ".join([sys.executable, *sys.argv]), encoding="utf-8")
        try:
            _measure_flatness_from_checkpoint(
                checkpoint_path=checkpoint_path,
                output_folder=output_folder,
                bundle=bundle,
                model=model,
                criterion=criterion,
                device=device,
                flatness_loader_a=flatness_loader_a,
                flatness_loader_b=flatness_loader_b,
                args=args,
                model_type_name=model_type_name,
                dataset_type_name=dataset_type_name,
            )
        finally:
            close_dali_loaders(loader_a, loader_b, flatness_loader_a, flatness_loader_b)
        return
    _, _, default_epochs = build_optimizer_and_scheduler(
        bundle.ml_setup,
        model,
        bundle.partition_a,
        batch_size=batch_size,
        preset=args.optimizer_preset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        gradient_accumulate_every=args.gradient_accumulate_every,
        warmup_epochs=args.warmup_epochs,
    )
    epochs = args.epochs if args.epochs is not None else int(default_epochs)
    output_folder = (
        Path(args.output_folder_name)
        if args.output_folder_name is not None
        else Path(f"misalignment_measurement_2_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_%f')}")
    )
    output_folder = output_folder if output_folder.is_absolute() else Path.cwd() / output_folder
    output_folder = Path(distributed.broadcast_object(str(output_folder)))
    if distributed.is_main():
        output_folder.mkdir(parents=True, exist_ok=False)
        (output_folder / "command.txt").write_text(" ".join([sys.executable, *sys.argv]), encoding="utf-8")
        torch.save(bundle.permutation, output_folder / "split_permutation.pt")
    distributed.barrier()
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    logger.info(
        "dataset=%s source_examples=%d half_size=%d augmentation=%s batch_size=%d gradient_accumulate_every=%d "
        "effective_batch_size=%d flatness_batch_size=%d epochs=%d",
        bundle.dataset_name,
        bundle.source_examples,
        len(bundle.partition_a),
        args.augmentation,
        batch_size,
        args.gradient_accumulate_every,
        batch_size * args.gradient_accumulate_every,
        flatness_batch_size,
        epochs,
    )
    if (model_type_name == "convnext_tiny" and dataset_type_name == "imagenet21k_p"
            and batch_size * args.gradient_accumulate_every != 4096):
        logger.warning("ConvNeXt 21k-p recipe assumes effective batch 4096 (LR 4e-3); got %d",
                       batch_size * args.gradient_accumulate_every)
    logger.info("objective mode = %s; directions = %s", args.objective, args.directions)
    run_specs = tuple(
        spec
        for spec in (
            ("a_as_train_b_as_val", "a", "b", loader_a, loader_b),
            ("b_as_train_a_as_val", "b", "a", loader_b, loader_a),
        )
        if args.directions in ("both", spec[0])
    )
    runs = {}
    for direction_name, train_partition_name, val_partition_name, train_loader, val_loader in run_specs:
        runs[direction_name] = _run_direction(
            direction_name=direction_name,
            train_partition_name=train_partition_name,
            val_partition_name=val_partition_name,
            train_loader=train_loader,
            val_loader=val_loader,
            eval_train_loader=(eval_loader_a if train_partition_name == "a" else eval_loader_b),
            eval_val_loader=(eval_loader_b if val_partition_name == "b" else eval_loader_a),
            flatness_train_loader=(flatness_loader_a if train_partition_name == "a" else flatness_loader_b),
            flatness_val_loader=(flatness_loader_b if val_partition_name == "b" else flatness_loader_a),
            bundle=bundle,
            model=model,
            initial_state=initial_state,
            criterion=criterion,
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            args=args,
            output_folder=output_folder,
            model_type_name=model_type_name,
            dataset_type_name=dataset_type_name,
        )
    measure_values = {
        name: run["misalignment_measure"]["overall_mean"] for name, run in runs.items()
        if run["misalignment_measure"] is not None
    }
    measure_items = list(measure_values.values())
    aggregate = {
        "schema_version": 2,
        "git_provenance": git_provenance,
        "dataset": bundle.dataset_name,
        "dataset_type": dataset_type_name,
        "source_examples": bundle.source_examples,
        "half_examples": len(bundle.partition_a),
        "split_seed": args.split_seed,
        "random_seed": args.random_seed,
        "permutation_file": "split_permutation.pt",
        "loader_backend": loader_backend,
        "run_order": list(runs),
        "runs": runs,
        "aggregate": {
            "misalignment_measure_by_direction": measure_values,
            "mean_misalignment_measure": sum(measure_items) / len(measure_items) if measure_items else None,
            "absolute_direction_difference": (
                abs(measure_items[0] - measure_items[1]) if len(measure_items) == 2 else None
            ),
        },
    }
    _write_json(str(output_folder / "summary.json"), aggregate)
    logger.info("finished; results written to %s", output_folder)


if __name__ == "__main__":
    try:
        main()
    finally:
        distributed.shutdown()
