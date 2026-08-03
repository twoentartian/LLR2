"""Adaptive Gradient Clipping (AGC) for optimizer-based training.

AGC clips the gradient of each output unit relative to the corresponding
parameter norm. It is installed as an optimizer step pre-hook, so it works
with LLR2's existing model adapters without changing their training loops.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


def _validate_hyperparameters(clip_factor: float, eps: float) -> None:
    if clip_factor <= 0.0:
        raise ValueError("AGC clip_factor must be positive")
    if eps <= 0.0:
        raise ValueError("AGC eps must be positive")


def unitwise_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Return an L2 norm per output unit, retaining broadcast dimensions."""

    norm_input = (
        tensor.float()
        if tensor.dtype in (torch.float16, torch.bfloat16)
        else tensor
    )
    if tensor.ndim <= 1:
        return torch.linalg.vector_norm(norm_input)
    return torch.linalg.vector_norm(
        norm_input,
        dim=tuple(range(1, tensor.ndim)),
        keepdim=True,
    )


@torch.no_grad()
def adaptive_clip_grad_(
    parameters: Iterable[nn.Parameter] | nn.Parameter,
    clip_factor: float = 0.01,
    eps: float = 1e-3,
) -> None:
    """Clip gradients in place using AGC's unit-wise relative threshold.

    For each unit, the multiplier is

    ``min(1, clip_factor * max(parameter_norm, eps) / (gradient_norm + eps))``.
    """

    _validate_hyperparameters(clip_factor, eps)
    if isinstance(parameters, nn.Parameter):
        parameters = (parameters,)

    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse:
            raise TypeError("AGC does not support sparse gradients")
        if not torch.is_floating_point(parameter) or not torch.is_floating_point(
            gradient
        ):
            raise TypeError("AGC requires floating-point parameters and gradients")

        parameter_norm = unitwise_norm(parameter.detach()).clamp(min=eps)
        gradient_norm = unitwise_norm(gradient.detach())
        max_gradient_norm = parameter_norm * clip_factor
        multiplier = (max_gradient_norm / (gradient_norm + eps)).clamp(max=1.0)
        gradient.mul_(multiplier.to(device=gradient.device, dtype=gradient.dtype))


def _is_normalization_module(module: nn.Module) -> bool:
    normalization_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.GroupNorm,
        nn.LayerNorm,
    )
    if isinstance(module, normalization_types):
        return True
    rms_norm_type = getattr(nn, "RMSNorm", None)
    return rms_norm_type is not None and isinstance(module, rms_norm_type)


class AdaptiveGradientClipper:
    """Install stateless AGC immediately before an optimizer update.

    Bias parameters and parameters owned by normalization modules are excluded
    by default. The eligible set is restricted to trainable parameters present
    in the attached optimizer.
    """

    def __init__(
        self,
        clip_factor: float = 0.01,
        eps: float = 1e-3,
        *,
        exclude_bias: bool = True,
        exclude_normalization: bool = True,
    ):
        _validate_hyperparameters(clip_factor, eps)
        self.clip_factor = float(clip_factor)
        self.eps = float(eps)
        self.exclude_bias = exclude_bias
        self.exclude_normalization = exclude_normalization

        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._parameters: dict[str, nn.Parameter] = {}
        self._excluded_parameter_names: tuple[str, ...] = ()
        self._step_pre_hook_handle: Optional[RemovableHandle] = None

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of parameters whose gradients AGC clips."""

        return tuple(self._parameters)

    @property
    def excluded_parameter_names(self) -> tuple[str, ...]:
        """Names excluded as biases or normalization parameters."""

        return self._excluded_parameter_names

    def attach(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> "AdaptiveGradientClipper":
        """Attach AGC to a model optimizer and return this clipper."""

        self.detach()
        optimizer_parameter_ids = {
            id(parameter)
            for parameter_group in optimizer.param_groups
            for parameter in parameter_group["params"]
        }

        excluded_parameter_ids = set()
        for _, module in model.named_modules():
            is_normalization = _is_normalization_module(module)
            for parameter_name, parameter in module.named_parameters(
                recurse=False
            ):
                is_excluded = (
                    self.exclude_bias and parameter_name == "bias"
                ) or (self.exclude_normalization and is_normalization)
                if is_excluded:
                    excluded_parameter_ids.add(id(parameter))

        eligible_parameters = {}
        excluded_parameter_names = []
        for parameter_name, parameter in model.named_parameters():
            if (
                not parameter.requires_grad
                or id(parameter) not in optimizer_parameter_ids
            ):
                continue
            if id(parameter) in excluded_parameter_ids:
                excluded_parameter_names.append(parameter_name)
            else:
                eligible_parameters[parameter_name] = parameter

        if not eligible_parameters:
            raise ValueError("AGC found no eligible trainable optimizer parameters")

        self._parameters = eligible_parameters
        self._excluded_parameter_names = tuple(excluded_parameter_names)
        self._optimizer = optimizer
        self._step_pre_hook_handle = optimizer.register_step_pre_hook(
            self._optimizer_step_pre_hook,
        )
        return self

    def detach(self) -> None:
        """Remove AGC's optimizer hook, if attached."""

        if self._step_pre_hook_handle is not None:
            self._step_pre_hook_handle.remove()
        self._step_pre_hook_handle = None
        self._optimizer = None
        self._parameters = {}
        self._excluded_parameter_names = ()

    def _optimizer_step_pre_hook(self, optimizer, _args, _kwargs):
        if optimizer is not self._optimizer:
            raise RuntimeError("AGC hook invoked by an unexpected optimizer")
        adaptive_clip_grad_(
            self._parameters.values(),
            clip_factor=self.clip_factor,
            eps=self.eps,
        )
        return None


__all__ = [
    "AdaptiveGradientClipper",
    "adaptive_clip_grad_",
    "unitwise_norm",
]
