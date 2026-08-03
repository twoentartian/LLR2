"""Buffered decentralized FedProx-style training support.

This module intentionally avoids changing the model adapters. It installs an
optimizer step pre-hook that adds the gradient of the proximal objective

    (mu / 2) * ||w - w_reference||^2

immediately before each optimizer update. The reference is the mean of the
peer-model buffer consumed by the most recent averaging event. It excludes the
local model, so it can exert a non-zero proximal force on the first local step
after decentralized mixing.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from .dfedavgm import DFedAvgMAverager


class DecentralizedFedProxAverager(DFedAvgMAverager):
    """DFedAvgM mixing with a buffered peer-average proximal reference.

    ``variance_corrector`` and the other :class:`DFedAvgMAverager` options are
    accepted through ``**kwargs``. Consequently, the same class supports both
    decentralized FedProx-style experiments with and without variance
    correction.

    Call :meth:`attach` once with the node's model and optimizer before the
    first averaging event. The proximal term is applied through the optimizer,
    so task-loss reporting remains unchanged. Its latest value is available as
    :attr:`last_proximal_loss`.

    Because the hook runs immediately before ``optimizer.step()``, it also runs
    after ``GradScaler`` has unscaled AMP gradients. If gradient clipping is
    enabled by an adapter, the proximal gradient is added after that clipping.
    """

    def __init__(self, mu: float, *args, **kwargs):
        if mu < 0.0:
            raise ValueError("FedProx mu must be non-negative")
        super().__init__(*args, **kwargs)
        self.mu = float(mu)
        self.last_proximal_loss = 0.0

        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._parameters: dict[str, nn.Parameter] = {}
        self._reference_model_stat: Optional[dict[str, torch.Tensor]] = None
        self._step_pre_hook_handle: Optional[RemovableHandle] = None

    def attach(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> "DecentralizedFedProxAverager":
        """Attach the proximal-gradient hook to one node model and optimizer."""

        self.detach()
        optimizer_parameter_ids = {
            id(parameter)
            for parameter_group in optimizer.param_groups
            for parameter in parameter_group["params"]
        }
        self._parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) in optimizer_parameter_ids
        }
        if not self._parameters:
            raise ValueError("optimizer does not contain any trainable model parameters")

        self._optimizer = optimizer
        self._step_pre_hook_handle = optimizer.register_step_pre_hook(
            self._optimizer_step_pre_hook,
        )
        return self

    def detach(self) -> None:
        """Remove a previously installed optimizer hook, if any."""

        if self._step_pre_hook_handle is not None:
            self._step_pre_hook_handle.remove()
        self._step_pre_hook_handle = None
        self._optimizer = None
        self._parameters = {}

    def get_reference_model_stat(self) -> Optional[dict[str, torch.Tensor]]:
        """Return a defensive copy of the latest peer-average reference."""

        return copy.deepcopy(self._reference_model_stat)

    def get_model(self, self_model: dict, *args, **kwargs) -> dict:
        """Capture the peer-buffer mean, then perform normal DFedAvgM mixing."""

        if self._optimizer is None:
            raise RuntimeError(
                "DecentralizedFedProxAverager.attach() must be called before averaging"
            )
        self._capture_peer_average_reference()
        return super().get_model(self_model, *args, **kwargs)

    def _capture_peer_average_reference(self) -> None:
        assert self.model_buffer is not None
        assert self.model_counter > 0

        reference_model_stat = {}
        for name, parameter in self._parameters.items():
            if name not in self.model_buffer:
                continue
            reference_model_stat[name] = (
                self.model_buffer[name]
                .detach()
                .to(device=parameter.device, dtype=parameter.dtype)
                .div(self.model_counter)
                .clone()
            )
        if not reference_model_stat:
            raise RuntimeError("peer-model buffer has no trainable model parameters")
        self._reference_model_stat = reference_model_stat

    def _optimizer_step_pre_hook(self, optimizer, _args, _kwargs):
        if optimizer is not self._optimizer:
            raise RuntimeError("FedProx hook invoked by an unexpected optimizer")
        self._add_proximal_gradient()
        return None

    @torch.no_grad()
    def _add_proximal_gradient(self) -> None:
        reference_model_stat = self._reference_model_stat
        if reference_model_stat is None or self.mu == 0.0:
            self.last_proximal_loss = 0.0
            return

        squared_distance = 0.0
        for name, parameter in self._parameters.items():
            reference = reference_model_stat.get(name)
            if reference is None:
                continue
            difference = parameter.detach() - reference
            squared_distance += float(difference.square().sum().item())
            proximal_gradient = difference.mul(self.mu)

            if parameter.grad is None:
                parameter.grad = proximal_gradient
            elif parameter.grad.is_sparse:
                raise TypeError("FedProx does not support sparse parameter gradients")
            else:
                parameter.grad.add_(proximal_gradient)

        self.last_proximal_loss = 0.5 * self.mu * squared_distance


__all__ = ["DecentralizedFedProxAverager"]
