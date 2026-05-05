"""Model adapters - normalise different model families behind one interface.

The execution engine calls ``adapter.train_step(...)`` / ``adapter.val_step(...)``
without knowing whether the underlying model is a plain ``nn.Module``, a
``LightningModule``, or a diffusion wrapper.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn
import lightning as L

from py_src.types import StepOutput


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_device(x: Any, device: torch.device) -> Any:
    """Recursively move tensors (in dicts / lists / tuples) to *device*."""
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    return x


def _classification_target_indices(label: torch.Tensor, num_classes: int) -> Optional[torch.Tensor]:
    """Return class indices for hard or soft classification targets."""
    if label.ndim == 1:
        return label
    if label.ndim == 2 and label.size(1) == num_classes:
        return label.argmax(dim=1)
    return None


def _step_optimizer_with_scaler(
    scaler: torch.amp.GradScaler, # type: ignore
    optimizer: torch.optim.Optimizer,
) -> bool:
    """Run ``scaler.step`` / ``scaler.update`` and report whether the step happened.

    ``GradScaler`` skips the underlying optimizer step when it detects inf/nan
    gradients. In that case we must not advance the LR scheduler either.
    """
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() >= old_scale


def _resolve_max_grad_norm(
    explicit_max_grad_norm: Optional[float],
    default_max_grad_norm: Optional[float] = None,
) -> Optional[float]:
    if explicit_max_grad_norm is not None:
        return explicit_max_grad_norm
    return default_max_grad_norm


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ModelAdapter(ABC):
    """Uniform interface consumed by the execution engine."""

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the underlying ``nn.Module`` (for ``.to(device)``, state_dict, etc.)."""

    @abstractmethod
    def train_step(
        self,
        batch: Any,
        batch_idx: int,
        optimizer: Optional[torch.optim.Optimizer],
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        device: torch.device,
        scaler: Optional[torch.amp.GradScaler], # type: ignore
        backpropagation: bool = True,
        *,
        zero_grad: bool = True,
        step_optimizer: bool = True,
        loss_divisor: float = 1.0,
        max_grad_norm: Optional[float] = None,
    ) -> StepOutput:
        """One training iteration on *batch*."""

    @abstractmethod
    def val_step(
        self,
        batch: Any,
        batch_idx: int,
        device: torch.device,
    ) -> StepOutput:
        """One validation iteration on *batch*."""

    # Optional hooks (default no-op) ---------------------

    def on_validation_epoch_start(self) -> None:
        pass

    def on_validation_epoch_end(self) -> dict:
        """Return aggregated validation metrics (if the model accumulates them internally)."""
        return {}

    def post_train_step(self) -> None:
        """Called after every training step (e.g. EMA update)."""


# ---------------------------------------------------------------------------
# Standard PyTorch models  (ResNet, CCT, ViT, …)
# ---------------------------------------------------------------------------

class StandardAdapter(ModelAdapter):
    """Adapter for plain ``nn.Module`` models with a standard
    ``criterion(model(x), y)`` training loop.
    """

    def __init__(self, model: nn.Module, criterion: nn.Module, clip_grad_norm: Optional[float] = None):
        self._model = model
        self._criterion = criterion
        self._clip_grad_norm = clip_grad_norm

    def get_model(self) -> nn.Module:
        return self._model

    # ---- training ----

    def train_step(
        self, batch, batch_idx, optimizer, lr_scheduler, device, scaler,
        backpropagation=True,
        *,
        zero_grad: bool = True,
        step_optimizer: bool = True,
        loss_divisor: float = 1.0,
        max_grad_norm: Optional[float] = None,
    ) -> StepOutput:
        data, label = batch
        data = data.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        if backpropagation and optimizer is not None and zero_grad:
            optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None
        optimizer_was_run = False
        clip_grad_norm = _resolve_max_grad_norm(max_grad_norm, self._clip_grad_norm)
        if use_amp:
            with torch.amp.autocast(device.type):               # type: ignore
                outputs = self._model(data)
                loss = self._criterion(outputs, label)
            if backpropagation:
                assert optimizer is not None
                assert scaler is not None
                loss_for_backward = loss / loss_divisor
                scaler.scale(loss_for_backward).backward()
                if step_optimizer:
                    if clip_grad_norm is not None:
                        scaler.unscale_(optimizer)              # type: ignore
                        nn.utils.clip_grad_norm_(self._model.parameters(), clip_grad_norm)
                    optimizer_was_run = _step_optimizer_with_scaler(scaler, optimizer)
        else:
            outputs = self._model(data)
            loss = self._criterion(outputs, label)
            if backpropagation and optimizer is not None:
                loss_for_backward = loss / loss_divisor
                loss_for_backward.backward()
                if step_optimizer:
                    if clip_grad_norm is not None:
                        nn.utils.clip_grad_norm_(self._model.parameters(), clip_grad_norm)
                    optimizer.step()
                    optimizer_was_run = True

        if backpropagation and step_optimizer and lr_scheduler is not None and optimizer_was_run:
            lr_scheduler.step()

        # Accuracy is well-defined for standard classification logits even if
        # the training loss is label-smoothed or soft-target based.
        correct = None
        if outputs.ndim >= 2:
            _, predicted = torch.max(outputs, 1)
            target_indices = _classification_target_indices(label, outputs.size(1))
            if target_indices is not None:
                correct = (predicted == target_indices).sum().item()

        return StepOutput(
            loss=loss.item(),
            sample_count=label.size(0),
            correct_count=correct, # type: ignore
            optimizer_was_run=optimizer_was_run,
        )

    # ---- validation ----

    def val_step(self, batch, batch_idx, device) -> StepOutput:
        data, label = batch
        data = data.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        with torch.no_grad():
            outputs = self._model(data)
            loss = self._criterion(outputs, label)

        correct = None
        if outputs.ndim >= 2:
            _, predicted = torch.max(outputs, 1)
            target_indices = _classification_target_indices(label, outputs.size(1))
            if target_indices is not None:
                correct = (predicted == target_indices).sum().item()

        variance = outputs.var(dim=0, unbiased=False).mean().item()

        return StepOutput(
            loss=loss.item(),
            sample_count=label.size(0),
            correct_count=correct, # type: ignore
            extra={"variance": variance},
        )


# ---------------------------------------------------------------------------
# Lightning adapter
# ---------------------------------------------------------------------------

class LightningAdapter(ModelAdapter):
    """Adapter for ``lightning.LightningModule`` models (e.g. NanoCLIP).

    Delegates to the module's own ``training_step`` / ``validation_step``.
    """

    def __init__(self, model: L.LightningModule):
        self._model = model

    def get_model(self) -> nn.Module:
        return self._model

    # ---- training ----

    def train_step(
        self, batch, batch_idx, optimizer, lr_scheduler, device, scaler,
        backpropagation=True,
        *,
        zero_grad: bool = True,
        step_optimizer: bool = True,
        loss_divisor: float = 1.0,
        max_grad_norm: Optional[float] = None,
    ) -> StepOutput:
        batch = _to_device(batch, device)

        if backpropagation and optimizer is not None and zero_grad:
            optimizer.zero_grad(set_to_none=True)

        loss, batch_accuracy = self._model.training_step(batch, batch_idx)  # type: ignore

        optimizer_was_run = False
        if backpropagation:
            loss_for_backward = loss / loss_divisor                                                # type: ignore
            loss_for_backward.backward()                                                           # type: ignore
            if step_optimizer:
                assert optimizer is not None
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self._model.parameters(), max_grad_norm)
                self._model.optimizer_step(0, batch_idx, optimizer, optimizer_closure=None)        # type: ignore
                optimizer_was_run = True
            if step_optimizer and lr_scheduler is not None and optimizer_was_run:
                lr_scheduler.step()

        # infer batch size
        if isinstance(batch, (tuple, list)):
            bs = batch[0].size(0)
        elif isinstance(batch, torch.Tensor):
            bs = batch.size(0)
        else:
            bs = 1

        return StepOutput(
            loss=loss.item(),                                                                      # type: ignore
            sample_count=bs,
            correct_count=int(batch_accuracy.item() * bs) if batch_accuracy is not None else None, # type: ignore
            optimizer_was_run=optimizer_was_run,
        )

    # ---- validation ----

    def on_validation_epoch_start(self) -> None:
        self._model.on_validation_epoch_start()

    def val_step(self, batch, batch_idx, device) -> StepOutput:
        batch = _to_device(batch, device)

        if isinstance(batch, (tuple, list)):
            bs = batch[0].size(0)
        elif isinstance(batch, torch.Tensor):
            bs = batch.size(0)
        else:
            bs = 1

        with torch.no_grad():
            self._model.validation_step(batch, batch_idx)

        return StepOutput(loss=0.0, sample_count=bs)

    def on_validation_epoch_end(self) -> dict:
        self._model.on_validation_epoch_end()
        loss, correct = self._model.get_validation_result() # type: ignore
        return {"loss": loss, "correct_count": correct}


# ---------------------------------------------------------------------------
# Diffusion adapter  (wraps DDPM's GaussianDiffusion)
# ---------------------------------------------------------------------------

class DiffusionAdapter(ModelAdapter):
    """Adapter for diffusion models (e.g. DDPM).

    The ``GaussianDiffusion`` forward returns the loss directly,
    so ``criterion`` is not used in the traditional sense.
    EMA update happens in ``post_train_step``.
    """

    def __init__(self, diffusion_model: nn.Module, clip_grad_norm: Optional[float] = None):
        self._model = diffusion_model
        self._clip_grad_norm = clip_grad_norm

    def get_model(self) -> nn.Module:
        return self._model

    # ---- training ----

    def train_step(
        self, batch, batch_idx, optimizer, lr_scheduler, device, scaler,
        backpropagation=True,
        *,
        zero_grad: bool = True,
        step_optimizer: bool = True,
        loss_divisor: float = 1.0,
        max_grad_norm: Optional[float] = None,
    ) -> StepOutput:
        data, label = batch
        data = data.to(device, non_blocking=True)

        if backpropagation and optimizer is not None and zero_grad:
            optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None
        optimizer_was_run = False
        clip_grad_norm = _resolve_max_grad_norm(max_grad_norm, self._clip_grad_norm)
        if use_amp:
            with torch.amp.autocast(device.type): # type: ignore
                loss = self._model(data)
            if backpropagation:
                assert optimizer is not None
                assert scaler is not None
                loss_for_backward = loss / loss_divisor
                scaler.scale(loss_for_backward).backward()
                if step_optimizer:
                    if clip_grad_norm is not None:
                        scaler.unscale_(optimizer)              # type: ignore
                        nn.utils.clip_grad_norm_(self._model.parameters(), clip_grad_norm)
                    optimizer_was_run = _step_optimizer_with_scaler(scaler, optimizer)
        else:
            loss = self._model(data)
            if backpropagation and optimizer is not None:
                loss_for_backward = loss / loss_divisor
                loss_for_backward.backward()
                if step_optimizer:
                    if clip_grad_norm is not None:
                        nn.utils.clip_grad_norm_(self._model.parameters(), clip_grad_norm)
                    optimizer.step()
                    optimizer_was_run = True

        if backpropagation and step_optimizer and lr_scheduler is not None and optimizer_was_run:
            lr_scheduler.step()

        return StepOutput(
            loss=loss.item(),
            sample_count=data.size(0),
            optimizer_was_run=optimizer_was_run,
        )

    def post_train_step(self) -> None:
        """Update EMA weights after each training step."""
        if hasattr(self._model, "update_ema"):
            self._model.update_ema() # type: ignore

    # ---- validation ----

    def val_step(self, batch, batch_idx, device) -> StepOutput:
        data, label = batch
        data = data.to(device, non_blocking=True)
        with torch.no_grad():
            loss = self._model(data)
        return StepOutput(loss=loss.item(), sample_count=data.size(0))


# ---------------------------------------------------------------------------
# Custom step adapter  (for grokking and other custom training loops)
# ---------------------------------------------------------------------------

class CustomStepAdapter(ModelAdapter):
    """Adapter that delegates to user-provided step functions.

    This covers models like the grokking transformer that need
    a completely custom training loop.
    """

    def __init__(
        self,
        model: nn.Module,
        train_step_fn,  # (batch_idx, batch, model, optimizer, lr_scheduler, extra_ctx) -> StepOutput
        val_step_fn,    # (batch_idx, batch, model, extra_ctx) -> StepOutput
        extra_ctx: Any = None,
    ):
        self._model = model
        self._train_step_fn = train_step_fn
        self._val_step_fn = val_step_fn
        self.extra_ctx = extra_ctx

    def get_model(self) -> nn.Module:
        return self._model

    def train_step(
        self, batch, batch_idx, optimizer, lr_scheduler, device, scaler,
        backpropagation=True,
        *,
        zero_grad: bool = True,
        step_optimizer: bool = True,
        loss_divisor: float = 1.0,
        max_grad_norm: Optional[float] = None,
    ) -> StepOutput:
        batch = _to_device(batch, device)
        effective_optimizer = optimizer if backpropagation else None
        effective_lr_scheduler = lr_scheduler if backpropagation else None
        return self._train_step_fn(
            batch_idx,
            batch,
            self._model,
            effective_optimizer,
            effective_lr_scheduler,
            self.extra_ctx,
        )

    def val_step(self, batch, batch_idx, device) -> StepOutput:
        batch = _to_device(batch, device)
        return self._val_step_fn(batch_idx, batch, self._model, self.extra_ctx)


def clone_adapter_for_model(
    adapter: ModelAdapter,
    model: nn.Module,
    *,
    criterion: Optional[nn.Module] = None,
) -> ModelAdapter:
    """Create an adapter of the same family bound to *model*.

    Services often reuse a shared model instance whose weights change over time.
    They need an adapter that matches the original ML setup without forcing
    every model family through ``StandardAdapter``.
    """
    if isinstance(adapter, StandardAdapter):
        criterion_to_use = criterion if criterion is not None else adapter._criterion
        return StandardAdapter(model, criterion_to_use, clip_grad_norm=adapter._clip_grad_norm)
    if isinstance(adapter, LightningAdapter):
        if not isinstance(model, L.LightningModule):
            raise TypeError("LightningAdapter requires a LightningModule model")
        return LightningAdapter(model)
    if isinstance(adapter, DiffusionAdapter):
        return DiffusionAdapter(model, clip_grad_norm=adapter._clip_grad_norm)
    if isinstance(adapter, CustomStepAdapter):
        return CustomStepAdapter(
            model,
            adapter._train_step_fn,
            adapter._val_step_fn,
            extra_ctx=adapter.extra_ctx,
        )
    raise TypeError(f"Unsupported adapter type: {type(adapter).__name__}")
