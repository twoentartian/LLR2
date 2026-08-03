"""Decentralized stochastic gradient tracking (DSGT).

The implementation follows the synchronous adapt-then-combine recursion

    X[k+1] = W (X[k] - eta Y[k])
    Y[k+1] = W Y[k] + G[k+1] - G[k]

without changing LLR2's model adapters. An optimizer step pre-hook replaces
the local task gradient with the gradient tracker. Communication payloads
contain both the stepped model and tracker state.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from .core import ModelAverager
from .variance import VarianceCorrector
from py_src.special_torch_layers import (
    is_ignored_layer_averaging,
    is_ignored_layer_variance_correction,
)


_PAYLOAD_ALGORITHM_KEY = "algorithm"
_PAYLOAD_ALGORITHM_VALUE = "DSGT"
_PAYLOAD_MODEL_KEY = "model"
_PAYLOAD_TRACKER_KEY = "gradient_tracker"


class DSGTModelAverager(ModelAverager):
    """Synchronous DSGT model mixing and stochastic-gradient tracking.

    Each node starts with an independently initialized model and zero tracker.
    The first optimizer step is suppressed so the first communication computes
    ``X[1] = W X[0]``. Subsequent task gradients are converted to
    ``Y[k] = W Y[k-1] + G[k] - G[k-1]`` immediately before the optimizer step.
    Plain SGD is enforced by default, which gives the ordinary DSGT model
    recursion. Set ``enforce_plain_sgd=False`` to retain the tracker recursion
    while applying it through another local optimizer. That is an empirical
    optimizer variant and does not implement the paper's exact model update.

    If a variance corrector is supplied, it is applied only to the mixed model.
    The tracker is always mixed linearly and is never variance-corrected.
    ``FollowConservative`` gives the weighted target variance described for the
    DSGT+VC empirical extension because its conservative value is the effective
    local mixing weight.
    """

    def __init__(
        self,
        self_weight: Optional[float] = None,
        *args,
        enforce_plain_sgd: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self_weight is not None and not 0.0 <= self_weight <= 1.0:
            raise ValueError("self_weight must be between 0 and 1")
        self.self_weight = self_weight
        self.enforce_plain_sgd = enforce_plain_sgd

        self.model_buffer: Optional[dict] = None
        self.tracker_buffer: Optional[dict] = None
        self.model_counter = 0

        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._parameters: dict[str, nn.Parameter] = {}
        self._tracker: dict[str, torch.Tensor] = {}
        self._mixed_tracker: dict[str, torch.Tensor] = {}
        self._previous_gradient: dict[str, torch.Tensor] = {}
        self._initial_exchange_pending = True
        self._has_stepped_since_communication = False
        self._step_pre_hook_handle: Optional[RemovableHandle] = None

    def attach(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> "DSGTModelAverager":
        """Attach DSGT to one model and its local optimizer."""

        if self.enforce_plain_sgd:
            self._validate_optimizer(optimizer)
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

        self._tracker = {
            name: torch.zeros_like(parameter)
            for name, parameter in self._parameters.items()
        }
        self._mixed_tracker = copy.deepcopy(self._tracker)
        self._previous_gradient = copy.deepcopy(self._tracker)
        self._initial_exchange_pending = True
        self._has_stepped_since_communication = False
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

    def get_communication_payload(self, model_stat: dict) -> dict:
        if self._optimizer is None:
            raise RuntimeError("DSGTModelAverager.attach() must be called before sending")
        return {
            _PAYLOAD_ALGORITHM_KEY: _PAYLOAD_ALGORITHM_VALUE,
            _PAYLOAD_MODEL_KEY: model_stat,
            _PAYLOAD_TRACKER_KEY: {
                name: tensor.detach().clone()
                for name, tensor in self._tracker.items()
            },
        }

    def add_model(self, communication_payload: dict) -> None:
        model_stat, tracker_stat = self._parse_payload(communication_payload)
        with torch.no_grad():
            if self.model_buffer is None:
                self.model_buffer = copy.deepcopy(model_stat)
                for layer_name in list(self.model_buffer):
                    if is_ignored_layer_averaging(layer_name):
                        del self.model_buffer[layer_name]
                self.tracker_buffer = copy.deepcopy(tracker_stat)
            else:
                self.model_buffer = ModelAverager._iadd_two_model(
                    self.model_buffer,
                    model_stat,
                    check_same_keys=False,
                )
                assert self.tracker_buffer is not None
                self.tracker_buffer = ModelAverager._iadd_two_model(
                    self.tracker_buffer,
                    tracker_stat,
                )
            self.model_counter += 1
            if self.variance_corrector is not None:
                self.variance_corrector.add_variance(model_stat)

    def get_model(self, self_model: dict, *args, **kwargs) -> dict:
        del args, kwargs
        if self._optimizer is None:
            raise RuntimeError("DSGTModelAverager.attach() must be called before averaging")
        assert self.model_buffer is not None
        assert self.tracker_buffer is not None
        assert self.model_counter > 0

        with torch.no_grad():
            local_weight, peer_weight = self._get_mixing_weights()
            device = ModelAverager._get_device_from_model_stat(self.model_buffer)
            local_model = copy.deepcopy(self_model)
            ModelAverager._move_state_dict(local_model, device)
            output = copy.deepcopy(local_model)

            for layer_name in output:
                if layer_name not in self.model_buffer:
                    continue
                output[layer_name] = (
                    output[layer_name] * local_weight
                    + self.model_buffer[layer_name] * peer_weight
                )

            mixed_tracker = copy.deepcopy(self._tracker)
            tracker_device = ModelAverager._get_device_from_model_stat(
                self.tracker_buffer
            )
            ModelAverager._move_state_dict(mixed_tracker, tracker_device)
            for name in mixed_tracker:
                mixed_tracker[name] = (
                    mixed_tracker[name] * local_weight
                    + self.tracker_buffer[name] * peer_weight
                )

            if self.variance_corrector is not None:
                target_variance = self.variance_corrector.get_variance(
                    local_model,
                    local_weight,
                )
                for layer_name, single_layer_variance in target_variance.items():
                    if is_ignored_layer_variance_correction(layer_name):
                        continue
                    if layer_name not in output:
                        continue
                    output[layer_name] = VarianceCorrector.scale_tensor_to_variance(
                        output[layer_name],
                        single_layer_variance,
                    )

            self._mixed_tracker = {
                name: tensor.to(
                    device=self._parameters[name].device,
                    dtype=self._parameters[name].dtype,
                )
                for name, tensor in mixed_tracker.items()
            }
            self.model_buffer = None
            self.tracker_buffer = None
            self.model_counter = 0
            self._has_stepped_since_communication = False
            return output

    def get_model_count(self) -> int:
        return self.model_counter

    def get_tracker_stat(self) -> dict[str, torch.Tensor]:
        """Return a defensive copy of the current local gradient tracker."""

        return copy.deepcopy(self._tracker)

    def get_mixed_tracker_stat(self) -> dict[str, torch.Tensor]:
        """Return the latest linearly mixed tracker used by the next update."""

        return copy.deepcopy(self._mixed_tracker)

    def _get_mixing_weights(self) -> tuple[float, float]:
        if self.self_weight is None:
            local_weight = 1.0 / (self.model_counter + 1)
        else:
            local_weight = self.self_weight
        peer_weight = (1.0 - local_weight) / self.model_counter
        return local_weight, peer_weight

    def _optimizer_step_pre_hook(self, optimizer, _args, _kwargs):
        if optimizer is not self._optimizer:
            raise RuntimeError("DSGT hook invoked by an unexpected optimizer")
        self._apply_gradient_tracking()
        return None

    @torch.no_grad()
    def _apply_gradient_tracking(self) -> None:
        if self._has_stepped_since_communication:
            raise RuntimeError(
                "ordinary DSGT permits one optimizer step per communication round"
            )
        if self._initial_exchange_pending:
            for parameter in self._parameters.values():
                # ``grad = None`` suppresses the complete optimizer update,
                # including AdamW's decoupled weight decay. A zero gradient
                # would still change independently initialized model weights.
                parameter.grad = None
            self._initial_exchange_pending = False
            self._has_stepped_since_communication = True
            return

        for name, parameter in self._parameters.items():
            if parameter.grad is not None and parameter.grad.is_sparse:
                raise TypeError("DSGT does not support sparse parameter gradients")
            current_gradient = (
                torch.zeros_like(parameter)
                if parameter.grad is None
                else parameter.grad.detach().clone()
            )
            tracker = (
                self._mixed_tracker[name]
                + current_gradient
                - self._previous_gradient[name]
            )
            self._tracker[name] = tracker.detach().clone()
            self._previous_gradient[name] = current_gradient
            parameter.grad = tracker
        self._has_stepped_since_communication = True

    @staticmethod
    def _validate_optimizer(optimizer: torch.optim.Optimizer) -> None:
        if not isinstance(optimizer, torch.optim.SGD):
            raise TypeError("DSGT requires torch.optim.SGD")
        learning_rates = set()
        for parameter_group in optimizer.param_groups:
            learning_rates.add(float(parameter_group["lr"]))
            if float(parameter_group.get("momentum", 0.0)) != 0.0:
                raise ValueError("DSGT SGD requires zero momentum")
            if float(parameter_group.get("dampening", 0.0)) != 0.0:
                raise ValueError("DSGT SGD requires zero dampening")
            if float(parameter_group.get("weight_decay", 0.0)) != 0.0:
                raise ValueError("DSGT SGD requires zero weight decay")
            if bool(parameter_group.get("nesterov", False)):
                raise ValueError("DSGT SGD does not support Nesterov momentum")
            if bool(parameter_group.get("maximize", False)):
                raise ValueError("DSGT SGD does not support maximize mode")
        if len(learning_rates) != 1:
            raise ValueError("DSGT requires one shared SGD learning rate")

    @staticmethod
    def _parse_payload(communication_payload: dict) -> tuple[dict, dict]:
        if (
            communication_payload.get(_PAYLOAD_ALGORITHM_KEY)
            != _PAYLOAD_ALGORITHM_VALUE
        ):
            raise ValueError("received a non-DSGT communication payload")
        model_stat = communication_payload.get(_PAYLOAD_MODEL_KEY)
        tracker_stat = communication_payload.get(_PAYLOAD_TRACKER_KEY)
        if not isinstance(model_stat, dict) or not isinstance(tracker_stat, dict):
            raise TypeError("DSGT payload must contain model and tracker state dicts")
        return model_stat, tracker_stat


__all__ = ["DSGTModelAverager"]
