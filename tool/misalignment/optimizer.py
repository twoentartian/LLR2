"""Optimizer presets for the symmetric misalignment experiment.

The misalignment objectives are optimized after every paired mini-batch.
Scheduler durations therefore count the number of mini-batch updates in each
epoch. These presets are local to this experiment and do not change
project-wide MLSetup interfaces.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MisalignmentOptimizerConfig:
    """Resolved optimizer values, suitable for saving with an experiment."""

    dataset: str
    model: str
    preset: int
    optimizer: str
    scheduler: str
    learning_rate: float
    weight_decay: float
    momentum: float
    betas: tuple[float, float]
    eps: float
    epochs: int
    updates_per_epoch: int
    warmup_epochs: int
    minimum_learning_rate: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _type_name(value) -> str:
    return str(getattr(value, "name", value)).lower()


class MisalignmentOptimizationSetup:
    """Dataset-level presets following ``complete_ml_setup`` conventions."""

    @staticmethod
    def resolve_config(
        ml_setup,
        *,
        preset: int = 0,
        epochs: int | None = None,
        learning_rate: float | None = None,
        weight_decay: float | None = None,
        optimizer: str = "auto",
        scheduler: str = "fixed",
        updates_per_epoch: int = 1,
    ) -> MisalignmentOptimizerConfig:
        preset = int(preset)
        if preset not in (0, 1):
            raise ValueError(f"misalignment optimizer preset must be 0 or 1, got {preset}")
        model_name = _type_name(ml_setup.model_type)
        dataset_name = _type_name(ml_setup.dataset_type)
        modular = dataset_name == "modular" or dataset_name.startswith("arithmetic_")
        if modular:
            default_optimizer, default_lr, default_wd, default_epochs = "adamw", 1e-3, 0.0, 3000
            minimum_lr_ratio, default_warmup = 0.1, 10
        elif dataset_name in ("mnist", "mnist_224"):
            default_optimizer, default_lr, default_wd, default_epochs = "sgd", 0.01, 0.0, 20
            minimum_lr_ratio, default_warmup = 0.0, 0
        elif dataset_name in ("cifar10", "cifar10_224", "cifar100", "cifar100_224"):
            default_optimizer, default_lr, default_wd, default_epochs = "sgd", 0.1, 5e-4, 70
            if preset == 1:
                default_lr, default_wd = 0.2, 1e-4
            minimum_lr_ratio, default_warmup = 0.0, 0
        elif dataset_name == "imagenet1k":
            default_optimizer, default_lr, default_wd, default_epochs = "sgd", 0.1, 1e-4, 100
            minimum_lr_ratio, default_warmup = 0.01, 0
        else:
            raise NotImplementedError(f"No misalignment optimizer preset for {dataset_name!r}")

        optimizer_name = "auto" if optimizer is None else str(optimizer).lower()
        if optimizer_name == "auto":
            optimizer_name = default_optimizer
        if optimizer_name not in ("sgd", "adam", "adamw"):
            raise ValueError("optimizer must be auto, sgd, adam, or adamw")
        scheduler_name = "fixed" if scheduler is None else str(scheduler).lower()
        if scheduler_name == "auto":
            scheduler_name = "none" if dataset_name.startswith("mnist") else "onecycle" if dataset_name.startswith("cifar") else "cosine"
        # ``fixed`` is the explicit, constant-learning-rate mode used by the
        # misalignment experiment.  Keep the historical ``none`` spelling as
        # a backwards-compatible alias for the same behavior.
        if scheduler_name not in ("fixed", "none", "cosine", "onecycle"):
            raise ValueError("scheduler must be fixed, cosine, auto, none, or onecycle")

        actual_epochs = default_epochs if epochs is None else int(epochs)
        actual_updates_per_epoch = int(updates_per_epoch)
        if actual_updates_per_epoch <= 0:
            raise ValueError("updates_per_epoch must be positive")
        actual_lr = default_lr if learning_rate is None else float(learning_rate)
        actual_wd = default_wd if weight_decay is None else float(weight_decay)
        if actual_epochs <= 0:
            raise ValueError("epochs must be positive")
        if not math.isfinite(actual_lr) or actual_lr <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(actual_wd) or actual_wd < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        warmup_epochs = min(default_warmup, actual_epochs - 1) if scheduler_name == "cosine" else 0
        return MisalignmentOptimizerConfig(
            dataset=dataset_name,
            model=model_name,
            preset=preset,
            optimizer=optimizer_name,
            scheduler=scheduler_name,
            learning_rate=actual_lr,
            weight_decay=actual_wd,
            momentum=0.9,
            betas=(0.9, 0.98) if modular else (0.9, 0.999),
            eps=1e-8,
            epochs=actual_epochs,
            updates_per_epoch=actual_updates_per_epoch,
            warmup_epochs=warmup_epochs,
            # For a fixed schedule the effective minimum is the initial LR;
            # the dataset-specific ratio only applies to cosine annealing.
            minimum_learning_rate=(
                actual_lr
                if scheduler_name in ("fixed", "none")
                else actual_lr * minimum_lr_ratio
            ),
        )

    @staticmethod
    def get_optimizer_lr_scheduler_epoch(
        ml_setup,
        model,
        preset: int = 0,
        *,
        epochs: int | None = None,
        learning_rate: float | None = None,
        weight_decay: float | None = None,
        optimizer: str = "auto",
        scheduler: str = "fixed",
        updates_per_epoch: int = 1,
    ):
        config = MisalignmentOptimizationSetup.resolve_config(
            ml_setup,
            preset=preset,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            scheduler=scheduler,
            updates_per_epoch=updates_per_epoch,
        )
        if config.optimizer == "sgd":
            optimizer_instance = torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        else:
            optimizer_class = torch.optim.AdamW if config.optimizer == "adamw" else torch.optim.Adam
            optimizer_instance = optimizer_class(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=config.betas,
                eps=config.eps,
            )
        scheduler_instance = None
        if config.scheduler == "onecycle":
            scheduler_instance = torch.optim.lr_scheduler.OneCycleLR(
                optimizer_instance,
                max_lr=config.learning_rate,
                steps_per_epoch=config.updates_per_epoch,
                epochs=config.epochs,
            )
        elif config.scheduler == "cosine":
            total_updates = config.epochs * config.updates_per_epoch
            warmup_updates = config.warmup_epochs * config.updates_per_epoch
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_instance,
                T_max=max(1, total_updates - warmup_updates),
                eta_min=config.minimum_learning_rate,
            )
            if warmup_updates:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer_instance,
                    start_factor=1e-8,
                    end_factor=1.0,
                    total_iters=warmup_updates,
                )
                scheduler_instance = torch.optim.lr_scheduler.SequentialLR(
                    optimizer_instance,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_updates],
                )
            else:
                scheduler_instance = cosine
        # Keep the three-value return used by existing callers, while exposing
        # exact resolved values for summary/checkpoint metadata.
        optimizer_instance.misalignment_config = config.as_dict()
        return optimizer_instance, scheduler_instance, config.epochs


def build_optimizer_and_scheduler(
    ml_setup,
    model,
    training_dataset,
    *,
    batch_size: int,
    preset: int = 0,
    epochs: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    optimizer: str = "auto",
    scheduler: str = "fixed",
):
    """Build a preset while preserving the original wrapper's call shape."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(training_dataset) == 0:
        raise ValueError("training_dataset must contain at least one example")
    updates_per_epoch = max(1, math.ceil(len(training_dataset) / batch_size))
    return MisalignmentOptimizationSetup.get_optimizer_lr_scheduler_epoch(
        ml_setup,
        model,
        preset=preset,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer=optimizer,
        scheduler=scheduler,
        updates_per_epoch=updates_per_epoch,
    )
