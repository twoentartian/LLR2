"""Execution engine – platform-aware train / val loops.

Supports: single CUDA GPU, Apple MPS, CPU, and multi-GPU via
``torch.nn.parallel.DistributedDataParallel``.

Usage::

    from py_src_new.engine import Device, train, val

    device = Device.auto()                          # picks best available
    # device = Device("cuda:0")
    # device = Device("mps")

    train_loader = setup.train_dataloader()
    result = train(setup.adapter, train_loader, optimizer, lr_scheduler,
                   device=device, scaler=scaler)
    val_result = val(setup.adapter, val_loader, device=device)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from py_src.adapters import ModelAdapter


# ---------------------------------------------------------------------------
# Device abstraction
# ---------------------------------------------------------------------------

class Device:
    """Thin wrapper around ``torch.device`` with helpers for AMP and DDP."""

    def __init__(self, device: str | torch.device):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

    # -- convenience constructors ------------------------------------------

    @classmethod
    def auto(cls) -> "Device":
        """Pick the best single device available."""
        if torch.cuda.is_available():
            return cls("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return cls("mps")
        return cls("cpu")

    @classmethod
    def cuda(cls, index: int = 0) -> "Device":
        return cls(f"cuda:{index}")

    @classmethod
    def mps(cls) -> "Device":
        return cls("mps")

    @classmethod
    def cpu(cls) -> "Device":
        return cls("cpu")

    # -- AMP helpers -------------------------------------------------------

    @property
    def supports_amp(self) -> bool:
        return self.device.type in ("cuda",)

    def make_scaler(self) -> Optional[torch.amp.GradScaler]: # type: ignore
        if self.supports_amp:
            return torch.amp.GradScaler(self.device.type) # type: ignore
        return None

    # -- DDP helpers -------------------------------------------------------

    @staticmethod
    def init_distributed(backend: str = "nccl"):
        """Initialise the default process group for DDP."""
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

    def wrap_ddp(self, model: torch.nn.Module) -> DDP:
        """Wrap *model* in ``DistributedDataParallel``."""
        model = model.to(self.device)
        return DDP(model, device_ids=[self.device.index] if self.device.type == "cuda" else None)

    @staticmethod
    def make_distributed_sampler(dataset, shuffle: bool = True) -> DistributedSampler:
        return DistributedSampler(dataset, shuffle=shuffle)

    def __repr__(self) -> str:
        return f"Device({self.device})"


# ---------------------------------------------------------------------------
# Train result
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    total_loss: float = 0.0
    total_count: int = 0
    total_correct: Optional[int] = None
    iterations: int = 0

    @property
    def avg_loss(self) -> float:
        return self.total_loss / self.total_count if self.total_count else 0.0

    @property
    def accuracy(self) -> Optional[float]:
        if self.total_correct is None:
            return None
        return self.total_correct / self.total_count if self.total_count else 0.0


@dataclass
class ValResult:
    total_loss: float = 0.0
    total_count: int = 0
    total_correct: Optional[int] = None
    extra: dict = field(default_factory=dict)

    @property
    def avg_loss(self) -> float:
        return self.total_loss / self.total_count if self.total_count else 0.0

    @property
    def accuracy(self) -> Optional[float]:
        if self.total_correct is None:
            return None
        return self.total_correct / self.total_count if self.total_count else 0.0


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(
    adapter: ModelAdapter,
    dataloader: Iterable,
    optimizer: Optional[torch.optim.Optimizer],
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    *,
    device: Device,
    scaler: Optional[torch.amp.GradScaler] = None, # type: ignore
    backpropagation: bool = True,
    max_steps: Optional[int] = None,
) -> TrainResult:
    """Run one epoch of training (or up to *max_steps* batches).

    Parameters
    ----------
    adapter:
        A :class:`ModelAdapter` that knows how to do a single train step.
    dataloader:
        Any iterable that yields batches.
    optimizer / lr_scheduler:
        Can be ``None`` when ``backpropagation=False`` (forward-only mode).
    device:
        The :class:`Device` to run on.
    scaler:
        An AMP ``GradScaler`` (pass ``None`` to disable mixed precision).
    backpropagation:
        Set to ``False`` to only compute the forward pass (no weight updates).
    max_steps:
        Stop after this many batches (``None`` = full epoch).
    """
    model = adapter.get_model()
    model.train()
    model.to(device.device)

    result = TrainResult()
    step_limit = max_steps if max_steps is not None else sys.maxsize

    step_counter = 0
    for batch_idx, batch in enumerate(dataloader):
        if step_counter >= step_limit:
            break

        out = adapter.train_step(
            batch=batch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device.device,
            scaler=scaler,
            backpropagation=backpropagation,
        )

        result.total_loss += out.loss * out.sample_count
        result.total_count += out.sample_count
        if out.correct_count is not None:
            result.total_correct = (result.total_correct or 0) + out.correct_count
        result.iterations += 1

        adapter.post_train_step()
        step_counter += 1

    return result


# ---------------------------------------------------------------------------
# Val
# ---------------------------------------------------------------------------

def val(
    adapter: ModelAdapter,
    dataloader: Iterable,
    *,
    device: Device,
) -> ValResult:
    """Run one epoch of validation.

    Parameters
    ----------
    adapter:
        A :class:`ModelAdapter` that knows how to do a single val step.
    dataloader:
        Any iterable that yields batches.
    device:
        The :class:`Device` to run on.
    """
    model = adapter.get_model()
    model.eval()
    model.to(device.device)

    adapter.on_validation_epoch_start()

    result = ValResult()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            out = adapter.val_step(batch, batch_idx, device.device)
            result.total_loss += out.loss * out.sample_count
            result.total_count += out.sample_count
            if out.correct_count is not None:
                result.total_correct = (result.total_correct or 0) + out.correct_count
            result.extra.update(out.extra)

    # let the adapter aggregate (e.g. Lightning collects metrics internally)
    epoch_metrics = adapter.on_validation_epoch_end()
    if epoch_metrics:
        if "loss" in epoch_metrics:
            result.total_loss = epoch_metrics["loss"] * result.total_count
        if "correct_count" in epoch_metrics:
            result.total_correct = epoch_metrics["correct_count"]
        result.extra.update(epoch_metrics)

    return result
