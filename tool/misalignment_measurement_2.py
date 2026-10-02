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
separately on the two original losses and then averaged.

Supported datasets are MNIST, CIFAR-10, CIFAR-100, ImageNet-1k, and a modular
dataset folder containing train.txt/val.txt (and optionally test.txt).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from misalignment.augmentation import (
    build_augmentation,
    describe_augmentation,
    parse_augmentation_level,
)
from misalignment.optimizer import build_optimizer_and_scheduler

from py_src.ml_setup import get_ml_setup_from_config
from py_src.ml_setup.grokking import arithmetic_addition_grokking, build_grokking_model
from py_src.ml_setup_dataset import ArithmeticDataset, DatasetSetup, DatasetType
from py_src.model_opti_save_load import save_model_state
from py_src.util import set_seed, setup_logging

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


@dataclass
class LossMetrics:
    loss: float
    accuracy: float
    examples: int


def _parse_augmentation(value: str) -> int:
    return parse_augmentation_level(value)


def _default_model_type(dataset_name: str) -> str:
    return {
        "mnist": "lenet5",
        "cifar10": "resnet18_bn",
        "cifar100": "resnet18_bn",
        "imagenet1k": "resnet18_bn",
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
    Worker processes are restarted on each iteration to observe ``epoch``.
    """

    def __init__(self, dataset: Dataset, transform, seed: int, stochastic: bool) -> None:
        self.dataset = dataset
        self.transform = transform
        self.seed = int(seed)
        self.stochastic = stochastic
        self.epoch = 0

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
    # Use preset 1 to construct a plain cross-entropy MLSetup.  Augmentation
    # is applied below explicitly, so ImageNet level 2 does not silently add
    # Mixup/CutMix or label smoothing to this experiment.
    factory_preset = 1 if dataset_name == "imagenet1k" else 0
    ml_setup = get_ml_setup_from_config(model_type, dataset_type=dataset_name, preset=factory_preset)
    common_transform = build_augmentation(ml_setup, dataset_name, augmentation)
    if not hasattr(ml_setup.training_data, "transform") or not hasattr(ml_setup.testing_data, "transform"):
        raise TypeError(f"{dataset_name} datasets do not expose torchvision transforms")
    ml_setup.training_data.transform = None
    ml_setup.testing_data.transform = None
    combined = AugmentedDataset(
        ConcatDataset([ml_setup.training_data, ml_setup.testing_data]),
        common_transform, seed=args.random_seed, stochastic=augmentation > 0,
    )
    partition_a, partition_b, permutation, excluded = _split_dataset(
        combined, seed=args.split_seed, odd_size_policy=args.odd_size_policy,
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
    )


def load_dataset_bundle(args) -> DatasetBundle:
    dataset_name = args.dataset.lower()
    model_type = args.model_type or _default_model_type(dataset_name)
    augmentation = _parse_augmentation(args.augmentation)
    if dataset_name == "modular":
        if augmentation != 0:
            raise ValueError("modular datasets support only --augmentation none/0")
        return _load_modular_bundle(args, model_type)
    if dataset_name not in {"mnist", "cifar10", "cifar100", "imagenet1k"}:
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
) -> LossMetrics:
    model.eval()
    loss_sum = 0.0
    correct_sum = 0.0
    example_count = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        loss, accuracy = _loss_and_accuracy(model, batch, criterion, modular=modular, eq_position=eq_position)
        count = int(batch["text"].shape[0] if modular else batch[0].shape[0])
        loss_sum += float(loss.item()) * count
        correct_sum += float(accuracy.item()) * count
        example_count += count
    if example_count == 0:
        raise ValueError("cannot evaluate an empty partition")
    return LossMetrics(loss=loss_sum / example_count, accuracy=correct_sum / example_count, examples=example_count)


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
    """Rewrite a small experiment CSV so completed epochs are immediately durable."""
    with open(path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        outfile.flush()


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
) -> list[dict[str, float]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    rows: list[dict[str, float]] = []
    optimization_fields = [
        "epoch", "loss_a", "loss_b", "train_accuracy", "val_accuracy", "mean_loss",
        "abs_loss_gap", "relative_gap", "objective", "batches", "learning_rate",
    ]
    _write_csv_rows(csv_path, rows, optimization_fields)
    model.eval()
    for epoch in range(epochs):
        _set_augmentation_epoch(loader_a.dataset, epoch)
        _set_augmentation_epoch(loader_b.dataset, epoch)
        # First evaluate the two complete halves.  The derivative of the
        # dataset-level objective is then accumulated over mini-batches, so
        # memory use stays bounded while the update still corresponds to the
        # objective of the whole halves.
        baseline_a = evaluate_partition(
            model, loader_a, criterion, device=device, modular=modular, eq_position=eq_position
        )
        baseline_b = evaluate_partition(
            model, loader_b, criterion, device=device, modular=modular, eq_position=eq_position
        )
        _, _, coefficient_a, coefficient_b, _ = (
            _objective_value_and_coefficients(
                baseline_a.loss, baseline_b.loss, objective_mode=objective_mode
            )
        )
        optimizer.zero_grad(set_to_none=True)
        batches = 0
        for raw_a, raw_b in zip(loader_a, loader_b, strict=True):
            batch_a = _move_batch(raw_a, device)
            batch_b = _move_batch(raw_b, device)
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
            surrogate = (
                coefficient_a * (count_a / len(loader_a.dataset)) * loss_a
                + coefficient_b * (count_b / len(loader_b.dataset)) * loss_b
            )
            surrogate.backward()
            batches += 1
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        current_a = evaluate_partition(
            model, loader_a, criterion, device=device, modular=modular, eq_position=eq_position
        )
        current_b = evaluate_partition(
            model, loader_b, criterion, device=device, modular=modular, eq_position=eq_position
        )
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
        )
        row = {
            "epoch": float(epoch),
            "loss_a": mean_a,
            "loss_b": mean_b,
            "train_accuracy": current_a.accuracy,
            "val_accuracy": current_b.accuracy,
            "mean_loss": mean_loss_current,
            "abs_loss_gap": abs(mean_a - mean_b),
            "relative_gap": relative_gap_current,
            "objective": mean_objective,
            "batches": float(batches),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        rows.append(row)
        _write_csv_rows(csv_path, rows, optimization_fields)
        if epoch % max(1, report_interval) == 0 or epoch == epochs - 1:
            logger.info(
                "epoch %d/%d: train_loss=%.6g val_loss=%.6g train_acc=%.6g val_acc=%.6g mean_loss=%.6g abs_gap=%.6g relative_gap=%.6g objective=%.6g lr=%.6g",
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


def _combine_normalized_train_val_gradients(
    model: nn.Module,
    train_gradients: list[torch.Tensor],
    val_gradients: list[torch.Tensor],
    *,
    parameters: list[nn.Parameter],
    train_weight: float,
    val_weight: float,
) -> tuple[list[torch.Tensor], list[dict[str, float | str]]]:
    """Build the per-parameter train-descent plus val-ascent gradient.

    For every parameter tensor, both gradients are normalized first.  The
    signed directions are then weighted and added (train minus val because the
    optimizer itself performs descent), and the result is rescaled to the
    original train-gradient norm.  The returned tensors are detached and ready
    to be assigned to ``parameter.grad`` before one Adam/SGD step.
    """
    if train_weight < 0.0 or val_weight < 0.0:
        raise ValueError("gradient weights must be non-negative")
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    combined: list[torch.Tensor] = []
    stats: list[dict[str, float | str]] = []
    for name, parameter, train_gradient, val_gradient in zip(
        names, parameters, train_gradients, val_gradients, strict=True
    ):
        train_norm = float(train_gradient.norm().item())
        val_norm = float(val_gradient.norm().item())
        parameter_norm = float(parameter.detach().norm().item())
        if train_norm == 0.0:
            combined_gradient = torch.zeros_like(train_gradient)
            cosine = float("nan")
        elif val_norm == 0.0:
            combined_gradient = train_gradient.detach().clone()
            cosine = float("nan")
        else:
            train_unit = train_gradient / train_norm
            val_unit = val_gradient / val_norm
            signed_unit_sum = train_weight * train_unit - val_weight * val_unit
            signed_unit_norm = float(signed_unit_sum.norm().item())
            if signed_unit_norm == 0.0:
                combined_gradient = torch.zeros_like(train_gradient)
            else:
                combined_gradient = signed_unit_sum * (train_norm / signed_unit_norm)
            cosine = float(torch.sum(train_gradient * val_gradient).item() / (train_norm * val_norm))
        combined_norm = float(combined_gradient.norm().item())
        combined.append(combined_gradient.detach())
        stats.append({
            "parameter": name,
            "train_gradient_norm": train_norm,
            "val_gradient_norm": val_norm,
            "combined_gradient_norm": combined_norm,
            "parameter_norm": parameter_norm,
            "train_val_gradient_cosine": cosine,
            "train_gradient_weight": train_weight,
            "val_gradient_weight": val_weight,
            "relative_train_gradient": train_norm / max(parameter_norm, 1e-12),
            "relative_combined_gradient": combined_norm / max(parameter_norm, 1e-12),
        })
    return combined, stats


def _gradient_geometry_sample_epochs(epochs: int) -> set[int]:
    """Select at most 100 evenly spaced epochs, including the first and last."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    points = min(100, epochs)
    return {index * (epochs - 1) // max(1, points - 1) for index in range(points)}


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
) -> list[dict[str, float]]:
    """Run the normalized train-descent/val-ascent update from initialization."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    rows: list[dict[str, float]] = []
    geometry_rows: list[dict[str, float | str]] = []
    geometry_sample_epochs = _gradient_geometry_sample_epochs(epochs)
    optimization_fields = [
        "epoch", "loss_a", "loss_b", "train_accuracy", "val_accuracy", "mean_loss",
        "abs_loss_gap", "relative_gap", "objective", "batches", "learning_rate",
        "train_val_gradient_cosine", "train_gradient_weight", "val_gradient_weight",
    ]
    geometry_fields = [
        "epoch", "parameter", "train_gradient_norm", "val_gradient_norm",
        "combined_gradient_norm", "parameter_norm", "train_val_gradient_cosine",
        "train_gradient_weight", "val_gradient_weight", "relative_train_gradient",
        "relative_combined_gradient",
    ]
    _write_csv_rows(csv_path, rows, optimization_fields)
    _write_csv_rows(gradient_geometry_csv_path, geometry_rows, geometry_fields)
    model.eval()
    for epoch in range(epochs):
        _set_augmentation_epoch(loader_a.dataset, epoch)
        _set_augmentation_epoch(loader_b.dataset, epoch)
        # Both gradients are computed before optimizer.step(), so they see the
        # exact same parameter state.  The optimizer state is updated only once
        # with the combined gradient below.
        train_gradients = _compute_partition_gradients(
            model, loader_a, criterion, device=device, modular=modular,
            eq_position=eq_position, parameters=parameters,
        )
        val_gradients = _compute_partition_gradients(
            model, loader_b, criterion, device=device, modular=modular,
            eq_position=eq_position, parameters=parameters,
        )
        progress = epoch / max(1, epochs - 1)
        # Ramp the validation contribution linearly from zero to one half.
        # The train contribution is reduced at the same rate so that the
        # total gradient weight is one throughout the run.
        val_gradient_weight = 0.5 * progress
        train_gradient_weight = 1.0 - 0.5 * progress
        combined_gradients, gradient_stats = _combine_normalized_train_val_gradients(
            model,
            train_gradients,
            val_gradients,
            parameters=parameters,
            train_weight=train_gradient_weight,
            val_weight=val_gradient_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(parameters, combined_gradients, strict=True):
            parameter.grad = gradient
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        current_a = evaluate_partition(
            model, loader_a, criterion, device=device, modular=modular, eq_position=eq_position
        )
        current_b = evaluate_partition(
            model, loader_b, criterion, device=device, modular=modular, eq_position=eq_position
        )
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
            "loss_a": current_a.loss,
            "loss_b": current_b.loss,
            "train_accuracy": current_a.accuracy,
            "val_accuracy": current_b.accuracy,
            "mean_loss": mean_loss,
            "abs_loss_gap": absolute_gap,
            "relative_gap": relative_gap,
            "objective": objective,
            "batches": float(len(loader_a) + len(loader_b)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_val_gradient_cosine": mean_gradient_cosine,
            "train_gradient_weight": train_gradient_weight,
            "val_gradient_weight": val_gradient_weight,
        }
        rows.append(row)
        _write_csv_rows(csv_path, rows, optimization_fields)
        if epoch in geometry_sample_epochs:
            for stats in gradient_stats:
                geometry_rows.append({"epoch": float(epoch), **stats})
            _write_csv_rows(gradient_geometry_csv_path, geometry_rows, geometry_fields)
        if epoch % max(1, report_interval) == 0 or epoch == epochs - 1:
            logger.info(
                "epoch %d/%d: train_loss=%.6g val_loss=%.6g train_acc=%.6g val_acc=%.6g mean_loss=%.6g abs_gap=%.6g relative_gap=%.6g objective=train-val=%.6g lr=%.6g train_w=%.6g val_w=%.6g",
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
            for _ in range(hutchinson_samples):
                probe = _rademacher_like(parameter, generator)
                hessian_probe = torch.autograd.grad(
                    (gradient * probe).sum(),
                    parameter,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if hessian_probe is None:
                    continue
                relative_probe = weight @ (weight.transpose(0, 1) @ probe)
                estimates[name].append(float((relative_probe * hessian_probe).sum().detach().item()))
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


def _build_loader(dataset: Dataset, batch_size: int, *, device: torch.device, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        # The split itself is randomized.  Keeping the order fixed makes the
        # A/B exchange test deterministic; augmentation randomness is seeded
        # by source index and epoch in AugmentedDataset.
        shuffle=False,
        generator=generator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2, sort_keys=True, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure symmetric train/validation misalignment.")
    parser.add_argument(
        "--dataset",
        "--dataset_type",
        dest="dataset",
        choices=["mnist", "cifar10", "cifar100", "imagenet1k", "modular"],
        required=True,
    )
    parser.add_argument("--dataset_path", default=None, help="modular dataset folder containing train.txt/val.txt")
    parser.add_argument("-m", "--model_type", default=None, help="MLSetup model name; defaults depend on the dataset")
    parser.add_argument("--augmentation", "--data_augmentation", default="none", help="none/0 or an integer augmentation level")
    parser.add_argument(
        "--objective",
        choices=["normalized_two_sided", "difference", "mean_relative_gap"],
        default="normalized_two_sided",
        help="normalized train-descent/val-ascent update, raw train-val difference, or the previous symmetric objective",
    )
    parser.add_argument("--modulus", type=int, default=None)
    parser.add_argument("--split_seed", type=int, default=1729)
    parser.add_argument(
        "--odd_size_policy",
        choices=["drop", "error"],
        default="drop",
        help="when the combined source count is odd, drop one shuffled example (default) or fail",
    )
    parser.add_argument("--random_seed", type=int, default=1729)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--optimizer_preset", type=int, default=0)
    parser.add_argument("--optimizer", choices=["auto", "sgd", "adam", "adamw"], default="auto")
    parser.add_argument("--scheduler", choices=["auto", "none", "cosine", "onecycle"], default="auto")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--report_interval", type=int, default=1)
    parser.add_argument("--relative_flatness_layers", default="last_matrix_weight")
    parser.add_argument("--relative_flatness_samples", type=int, default=4)
    parser.add_argument("--relative_flatness_batches", type=int, default=1)
    parser.add_argument("--relative_flatness_seed", type=int, default=2718)
    parser.add_argument("--output_folder_name", "-o", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num_threads", type=int, default=None)
    parser.add_argument("--m_nlayer", type=int, default=None)
    parser.add_argument("--m_n_heads", type=int, default=None)
    parser.add_argument("--m_d_model", type=int, default=None)
    parser.add_argument("--m_context_len", type=int, default=None)
    parser.add_argument("--m_pos_encoding", choices=["default", "trainable"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "modular" and not args.dataset_path:
        raise ValueError("--dataset_path is required for --dataset modular")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")
    if args.relative_flatness_batches == 0 or args.relative_flatness_samples <= 0:
        raise ValueError("relative-flatness batches must be non-zero and samples must be positive")
    if args.num_threads is not None:
        torch.set_num_threads(max(1, args.num_threads))
    set_seed(args.random_seed)
    setup_logging(logger, "main")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    logger.info("device = %s", device)

    bundle = load_dataset_bundle(args)
    model = bundle.ml_setup.model.to(device)
    criterion = bundle.ml_setup.criterion or nn.CrossEntropyLoss()
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif bundle.modular:
        # The shuffled partitions are equal-sized; use a capped full batch
        # for modular data, matching the grokking experiment scripts.
        batch_size = min(len(bundle.partition_a), 65536)
    else:
        batch_size = int(getattr(bundle.ml_setup, "default_batch_size", 64) or 64)
    loader_a = _build_loader(bundle.partition_a, batch_size, device=device, num_workers=args.num_workers, seed=args.random_seed + 101)
    loader_b = _build_loader(bundle.partition_b, batch_size, device=device, num_workers=args.num_workers, seed=args.random_seed + 202)
    model_type_name = bundle.ml_setup.model_type.name
    dataset_type_name = bundle.dataset_type_name
    optimizer, scheduler, default_epochs = build_optimizer_and_scheduler(
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
    )
    epochs = args.epochs if args.epochs is not None else int(default_epochs)
    output_folder = (
        Path(args.output_folder_name)
        if args.output_folder_name is not None
        else Path(f"misalignment_measurement_2_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_%f')}")
    )
    output_folder = output_folder if output_folder.is_absolute() else Path.cwd() / output_folder
    output_folder.mkdir(parents=True, exist_ok=False)
    (output_folder / "command.txt").write_text(" ".join([sys.executable, *sys.argv]), encoding="utf-8")
    torch.save(bundle.permutation, output_folder / "split_permutation.pt")
    save_model_state(
        str(output_folder / "initial.model.pt"),
        {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        model_type_name,
        dataset_type_name,
    )
    logger.info(
        "dataset=%s source_examples=%d half_size=%d augmentation=%s batch_size=%d epochs=%d",
        bundle.dataset_name,
        bundle.source_examples,
        len(bundle.partition_a),
        args.augmentation,
        batch_size,
        epochs,
    )
    logger.info("objective mode = %s (partition A=train, partition B=val)", args.objective)

    if args.objective == "normalized_two_sided":
        rows = train_normalized_two_sided(
            model,
            loader_a,
            loader_b,
            criterion,
            device=device,
            modular=bundle.modular,
            eq_position=bundle.eq_position,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            report_interval=args.report_interval,
            csv_path=str(output_folder / "optimization.csv"),
            gradient_geometry_csv_path=str(output_folder / "gradient_geometry.csv"),
        )
    else:
        rows = train_symmetric_objective(
            model,
            loader_a,
            loader_b,
            criterion,
            device=device,
            modular=bundle.modular,
            eq_position=bundle.eq_position,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            report_interval=args.report_interval,
            csv_path=str(output_folder / "optimization.csv"),
            objective_mode=args.objective,
        )
    save_model_state(str(output_folder / "final.model.pt"), model.state_dict(), model_type_name, dataset_type_name)

    final_a = evaluate_partition(model, loader_a, criterion, device=device, modular=bundle.modular, eq_position=bundle.eq_position)
    final_b = evaluate_partition(model, loader_b, criterion, device=device, modular=bundle.modular, eq_position=bundle.eq_position)
    layers = _resolve_flatness_layers(model, args.relative_flatness_layers)
    flatness_a = measure_relative_flatness(
        model, loader_a, criterion, device=device, modular=bundle.modular, eq_position=bundle.eq_position,
        layer_names=layers, hutchinson_samples=args.relative_flatness_samples,
        max_batches=args.relative_flatness_batches, seed=args.relative_flatness_seed + 11,
    )
    flatness_b = measure_relative_flatness(
        model, loader_b, criterion, device=device, modular=bundle.modular, eq_position=bundle.eq_position,
        layer_names=layers, hutchinson_samples=args.relative_flatness_samples,
        max_batches=args.relative_flatness_batches, seed=args.relative_flatness_seed + 29,
    )
    flatness_rows = []
    for partition, values in (("a", flatness_a), ("b", flatness_b)):
        for value in values:
            flatness_rows.append({"partition": partition, **value})
    by_layer = {}
    for name in layers:
        value_a = next(item for item in flatness_a if item["layer"] == name)
        value_b = next(item for item in flatness_b if item["layer"] == name)
        by_layer[name] = {
            "mean_signed_estimate": (value_a["signed_estimate"] + value_b["signed_estimate"]) / 2,
            "mean_positive_part": (value_a["positive_part"] + value_b["positive_part"]) / 2,
        }
    with open(output_folder / "relative_flatness.csv", "w", newline="", encoding="utf-8") as outfile:
        fieldnames = list(flatness_rows[0].keys()) if flatness_rows else []
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flatness_rows)
    final_objective, final_relative_gap, _, _, final_mean_loss = _objective_value_and_coefficients(
        final_a.loss, final_b.loss, objective_mode=args.objective
    )
    summary = {
        "dataset": bundle.dataset_name,
        "dataset_type": dataset_type_name,
        "source_examples": bundle.source_examples,
        "half_examples": len(bundle.partition_a),
        "split_seed": args.split_seed,
        "random_seed": args.random_seed,
        "augmentation": _parse_augmentation(args.augmentation),
        "augmentation_config": describe_augmentation(args.dataset, args.augmentation),
        "excluded_source_indices": bundle.excluded_indices.tolist(),
        "model_type": model_type_name,
        "device": str(device),
        "batch_size": batch_size,
        "epochs": epochs,
        "objective_mode": args.objective,
        "optimizer": getattr(optimizer, "misalignment_config", None),
        "final": {
            "partition_a": asdict(final_a),
            "partition_b": asdict(final_b),
            "mean_loss": final_mean_loss,
            "absolute_loss_gap": abs(final_a.loss - final_b.loss),
            "relative_gap": final_relative_gap,
            "objective": final_objective,
        },
        "relative_flatness": {
            "layers": layers,
            "samples": args.relative_flatness_samples,
            "batches": args.relative_flatness_batches,
            "symmetric_by_layer": by_layer,
            "partitions": {"a": flatness_a, "b": flatness_b},
        },
        "misalignment_measure": {
            "definition": "mean_positive_relative_flatness_across_the_two_equal_halves",
            "by_layer": {name: values["mean_positive_part"] for name, values in by_layer.items()},
            "overall_mean": sum(values["mean_positive_part"] for values in by_layer.values()) / max(1, len(by_layer)),
        },
        "optimization_last_row": rows[-1] if rows else None,
    }
    _write_json(str(output_folder / "summary.json"), summary)
    logger.info("finished; results written to %s", output_folder)


if __name__ == "__main__":
    main()
