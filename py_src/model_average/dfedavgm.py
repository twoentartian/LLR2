"""DFedAvgM decentralized model mixing."""

from __future__ import annotations

import copy
from typing import Optional

import torch

from .core import ModelAverager
from .variance import VarianceCorrector
from py_src.special_torch_layers import (
    is_ignored_layer_averaging,
    is_ignored_layer_variance_correction,
)


class DFedAvgMAverager(ModelAverager):
    """DFedAvgM's decentralized model-mixing step.

    DFedAvgM performs heavy-ball momentum during local SGD and then applies
    ``x_i <- sum_j w_ij z_j`` over the node's closed neighborhood. This class
    implements that second step. When ``self_weight`` is ``None``, the local
    model and every received neighbor model receive equal weight. Otherwise,
    ``self_weight`` is assigned to the local model and the remaining weight is
    divided equally among the received neighbor models.

    The equal-weight default is a symmetric mixing matrix for regular graphs.
    A fixed self weight is also symmetric when every node uses the same value
    on a regular graph. Momentum must be configured on the node's local SGD
    optimizer; it is intentionally not applied to model deltas here. Set
    ``enforce_heavy_ball_sgd=False`` to use only DFedAvgM's mixing step with a
    different local training recipe. In that mode, optimizer state is kept.

    When a variance corrector is supplied, correction is applied after mixing.
    Correction modes that include the local model use DFedAvgM's effective
    local mixing weight.
    """

    def __init__(
        self,
        self_weight: Optional[float] = None,
        *args,
        enforce_heavy_ball_sgd: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self_weight is not None and not 0.0 <= self_weight <= 1.0:
            raise ValueError("self_weight must be between 0 and 1")
        self.self_weight = self_weight
        self.enforce_heavy_ball_sgd = enforce_heavy_ball_sgd
        self.model_buffer: Optional[dict] = None
        self.model_counter = 0

    def add_model(self, model_stat: dict) -> None:
        with torch.no_grad():
            if self.model_buffer is None:
                self.model_buffer = copy.deepcopy(model_stat)
                for layer_name in list(self.model_buffer):
                    if is_ignored_layer_averaging(layer_name):
                        del self.model_buffer[layer_name]
            else:
                self.model_buffer = ModelAverager._iadd_two_model(
                    self.model_buffer,
                    model_stat,
                    check_same_keys=False,
                )
            self.model_counter += 1
            if self.variance_corrector is not None:
                self.variance_corrector.add_variance(model_stat)

    def get_model(self, self_model: dict, *args, **kwargs) -> dict:
        del args, kwargs
        assert self.model_buffer is not None
        assert self.model_counter > 0
        with torch.no_grad():
            device = ModelAverager._get_device_from_model_stat(self.model_buffer)
            self_model = copy.deepcopy(self_model)
            ModelAverager._move_state_dict(self_model, device)
            output = copy.deepcopy(self_model)

            if self.self_weight is None:
                local_weight = 1.0 / (self.model_counter + 1)
            else:
                local_weight = self.self_weight
            neighbor_weight = (1.0 - local_weight) / self.model_counter

            for layer_name in output:
                if layer_name not in self.model_buffer:
                    continue
                output[layer_name] = (
                    output[layer_name] * local_weight
                    + self.model_buffer[layer_name] * neighbor_weight
                )

            if self.variance_corrector is not None:
                target_variance = self.variance_corrector.get_variance(
                    self_model,
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

            self.model_buffer = None
            self.model_counter = 0
            return output

    def get_model_count(self) -> int:
        return self.model_counter

    def on_after_averaging(
        self,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> None:
        """Reset heavy-ball velocity for the next local-training round."""

        if optimizer is None:
            return
        if not self.enforce_heavy_ball_sgd:
            return
        if not isinstance(optimizer, torch.optim.SGD):
            raise TypeError("DFedAvgM local training requires torch.optim.SGD")
        for parameter_group in optimizer.param_groups:
            momentum = float(parameter_group.get("momentum", 0.0))
            if not 0.0 <= momentum < 1.0:
                raise ValueError("DFedAvgM SGD momentum must be in [0, 1)")
            if float(parameter_group.get("dampening", 0.0)) != 0.0:
                raise ValueError("DFedAvgM SGD requires zero dampening")
            if bool(parameter_group.get("nesterov", False)):
                raise ValueError(
                    "DFedAvgM uses heavy-ball rather than Nesterov momentum"
                )
        for optimizer_state in optimizer.state.values():
            optimizer_state.pop("momentum_buffer", None)


__all__ = ["DFedAvgMAverager"]
