from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional
from enum import Enum, auto

import torch
import torch.nn as nn

from py_src.adapters import ModelAdapter
from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup_model import ModelType
from .dataloader_util import DataloaderConfig, build_dataloader


class ApplicationType(Enum):
    classifier = auto()
    diffusion = auto()
    clip = auto()
    unknown = auto()

@dataclass
class MLSetup:
    """Everything needed to train or evaluate a model."""

    # ---- model + adapter --------------------------------------------------
    model: nn.Module = None                    # type: ignore # the raw nn.Module
    adapter: ModelAdapter = None               # type: ignore # how the engine talks to this model
    model_type: ModelType = None               # type: ignore 
    application_type: ApplicationType = ApplicationType.unknown

    # ---- dataset ----------------------------------------------------------
    training_data: Any = None                  # map-style Dataset, IterableDataset, or DataLoader
    testing_data: Any = None
    dataset_type: DatasetType = None          # type: ignore 

    # ---- training defaults ------------------------------------------------
    default_batch_size: int = 0
    criterion: Optional[nn.Module] = None
    default_collate_fn: Optional[Callable] = None
    default_collate_fn_val: Optional[Callable] = None
    default_sampler_fn: Optional[Callable] = None
    default_prefetch_factor: int = 4
    has_normalization_layer: bool = False
    gradient_accumulate_every: int = 1
    max_grad_norm: Optional[float] = None

    # ---- dataloader overrides (pre-built loaders) -------------------------
    override_train_loader: Optional[Iterable] = None
    override_test_loader: Optional[Iterable] = None

    # ---- special functions ------------------------------------------------
    difussion_generate_sample: Optional[Callable[..., None]] = None

    # ------------------------------------------------------------------
    # Public API: get dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self, config: Optional[DataloaderConfig] = None, ignore_override=False) -> Iterable:
        """Build (or return) a training dataloader.

        If ``override_train_loader`` is set, it is returned as-is
        (config is ignored in that case).
        """
        if self.override_train_loader is not None and not ignore_override:
            return self.override_train_loader
        return build_dataloader(
            dataset=self.training_data,
            default_batch_size=self.default_batch_size,
            default_prefetch_factor=self.default_prefetch_factor,
            config=config,
            is_train=True,
            default_collate_fn=self.default_collate_fn,
            default_sampler_fn=self.default_sampler_fn,
        )

    def val_dataloader(self, config: Optional[DataloaderConfig] = None, ignore_override=False) -> Iterable:
        """Build (or return) a validation / test dataloader."""
        if self.override_test_loader is not None and not ignore_override:
            return self.override_test_loader
        collate = self.default_collate_fn_val or self.default_collate_fn
        return build_dataloader(
            dataset=self.testing_data,
            default_batch_size=self.default_batch_size,
            default_prefetch_factor=self.default_prefetch_factor,
            config=config,
            is_train=False,
            default_collate_fn=collate,
        )

    def description(self) -> str:
        return f"{self.model_type}@{self.dataset_type}"
