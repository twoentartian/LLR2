"""Model-state movement and simulator averaging utilities."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch

from py_src.special_torch_layers import is_ignored_layer_averaging
from py_src.model_variance_correct import VarianceCorrectionType, VarianceCorrector
from py_src.special_torch_layers import is_ignored_layer_variance_correction


# ---------------------------------------------------------------------------
# Tensor-level helpers
# ---------------------------------------------------------------------------

def move_tensor_toward(
    layer_name: str,
    src_tensor: torch.Tensor,
    dest_tensor: torch.Tensor,
    step: float,
    adoptive_step: float,
    random_scale: Optional[float] = None,
) -> torch.Tensor:
    """Move *src_tensor* one step toward *dest_tensor*.

    The actual step size is ``max(step, ||dest-src|| * adoptive_step)``.
    """
    if torch.equal(src_tensor, dest_tensor):
        return src_tensor

    diff = dest_tensor - src_tensor
    norm = torch.norm(diff)
    adoptive_part = norm * adoptive_step
    real_step = step if step > adoptive_part else adoptive_part
    direction = diff / norm
    if random_scale is not None:
        direction = direction * torch.rand_like(direction) * random_scale
    return src_tensor + direction * real_step


# ---------------------------------------------------------------------------
# State-dict-level movement
# ---------------------------------------------------------------------------

def move_model_state_toward(
    src_model_stat: dict,
    dest_model_stat: dict,
    step: float,
    adoptive_step: float,
    ratio_step_per_layer: Optional[Dict[str, float]] = None,
    enable_merge_bias_with_weight: bool = False,
    ignore_layers: Optional[List[str]] = None,
    move_layer: Optional[List[str]] = None,
    random_scale: Optional[float] = None,
) -> dict:
    """Move every (selected) layer in *src_model_stat* toward *dest_model_stat*.

    Parameters
    ----------
    src_model_stat / dest_model_stat:
        Source and destination state dicts (tensors may be on any device).
    step:
        Absolute step size (geodesic direction).
    adoptive_step:
        Fraction of the current distance added to *step*.
    ratio_step_per_layer:
        Optional per-layer extra step sizes (added to *step*).
    enable_merge_bias_with_weight:
        Concatenate the weight and its matching bias before computing the move
        direction so that bias follows the weight direction.
    ignore_layers:
        Layer names to leave untouched (mutually exclusive with *move_layer*).
    move_layer:
        Explicit allow-list of layers to move (mutually exclusive with *ignore_layers*).
    """
    assert not (ignore_layers is not None and move_layer is not None), \
        "only one of ignore_layers / move_layer may be provided"

    if ignore_layers is None:
        ignore_layers = []
    if move_layer is None:
        move_layer = list(src_model_stat.keys())

    output = copy.deepcopy(src_model_stat)
    processed: set = set()

    for layer_name in move_layer:
        if layer_name in processed or layer_name in ignore_layers:
            continue

        total_step = step if ratio_step_per_layer is None else step + ratio_step_per_layer.get(layer_name, 0.0)
        moved = False

        # Optionally merge weight + bias into a single vector
        if enable_merge_bias_with_weight and 'weight' in layer_name:
            bias_name = layer_name.replace('weight', 'bias')
            if bias_name in move_layer:
                processed.add(layer_name)
                processed.add(bias_name)
                w_src = src_model_stat[layer_name]
                b_src = src_model_stat[bias_name]
                w_dst = dest_model_stat[layer_name]
                b_dst = dest_model_stat[bias_name]
                src_cat = torch.cat((w_src.flatten(), b_src.flatten()))
                dst_cat = torch.cat((w_dst.flatten(), b_dst.flatten()))
                out_cat = move_tensor_toward(layer_name, src_cat, dst_cat, total_step, adoptive_step, random_scale)
                w_out, b_out = torch.split(out_cat, [w_src.numel(), b_src.numel()])
                output[layer_name] = w_out.reshape(w_src.shape)
                output[bias_name] = b_out.reshape(b_src.shape)
                moved = True

        if not moved:
            processed.add(layer_name)
            output[layer_name] = move_tensor_toward(
                layer_name,
                src_model_stat[layer_name],
                dest_model_stat[layer_name],
                total_step, adoptive_step, random_scale,
            )

    return output


class ModelAverager:
    def __init__(self, variance_corrector: Optional[VarianceCorrector] = None):
        self.variance_corrector = variance_corrector

    def add_model(self, model_stat: dict) -> None:
        raise NotImplementedError

    def get_model(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    def get_model_count(self) -> int:
        raise NotImplementedError

    @staticmethod
    def _iadd_two_model(
        src: dict,
        addition: dict,
        *,
        weight_src: float = 1.0,
        weight_addition: float = 1.0,
        check_same_keys: bool = True,
    ) -> dict:
        with torch.no_grad():
            assert (not check_same_keys) or (set(src.keys()) == set(addition.keys()))
            for layer_name in src.keys():
                addition_tensor = addition[layer_name]
                if src[layer_name].device != addition_tensor.device:
                    addition_tensor = addition_tensor.to(src[layer_name].device)
                if weight_src == 1.0 and weight_addition == 1.0:
                    src[layer_name] += addition_tensor
                else:
                    src[layer_name] = (
                        src[layer_name] * weight_src + addition_tensor * weight_addition
                    )
        return src

    @staticmethod
    def _move_state_dict(state_dict: dict, device: torch.device) -> None:
        for key, value in state_dict.items():
            state_dict[key] = value.to(device)

    @staticmethod
    def _get_device_from_model_stat(state_dict: dict) -> torch.device:
        return next(iter(state_dict.values())).device


class StandardModelAverager(ModelAverager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.variance_corrector is not None:
            vc_type = self.variance_corrector.variance_correction_type
            assert vc_type == VarianceCorrectionType.FollowOthers
        self.model_buffer: Optional[dict] = None
        self.model_counter = 0

    def add_model(self, model_stat: dict) -> None:
        with torch.no_grad():
            if self.model_buffer is None:
                self.model_buffer = copy.deepcopy(model_stat)
                for layer_name in list(self.model_buffer.keys()):
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
        assert self.model_buffer is not None
        with torch.no_grad():
            device = ModelAverager._get_device_from_model_stat(self.model_buffer)
            self_model = copy.deepcopy(self_model)
            ModelAverager._move_state_dict(self_model, device)

            output = copy.deepcopy(self_model)
            for layer_name in output:
                if layer_name not in self.model_buffer:
                    continue
                output[layer_name] = self.model_buffer[layer_name] / self.model_counter

            if self.variance_corrector is not None:
                target_variance = self.variance_corrector.get_variance()
                for layer_name, single_layer_variance in target_variance.items():
                    if is_ignored_layer_variance_correction(layer_name):
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


class ConservativeModelAverager(ModelAverager):
    def __init__(self, conservative: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0.0 <= conservative <= 1.0
        self.conservative = conservative
        self.model_buffer: Optional[dict] = None
        self.model_counter = 0

    def add_model(self, model_stat: dict) -> None:
        with torch.no_grad():
            if self.model_buffer is None:
                self.model_buffer = copy.deepcopy(model_stat)
                for layer_name in list(self.model_buffer.keys()):
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
        assert self.model_buffer is not None
        with torch.no_grad():
            device = ModelAverager._get_device_from_model_stat(self.model_buffer)
            self_model = copy.deepcopy(self_model)
            ModelAverager._move_state_dict(self_model, device)

            averaged = copy.deepcopy(self_model)
            for layer_name in averaged:
                if layer_name not in self.model_buffer:
                    continue
                averaged[layer_name] = self.model_buffer[layer_name] / self.model_counter

            output = ModelAverager._iadd_two_model(
                self_model,
                averaged,
                weight_src=self.conservative,
                weight_addition=1 - self.conservative,
            )
            if self.variance_corrector is not None:
                target_variance = self.variance_corrector.get_variance(
                    self_model,
                    self.conservative,
                )
                for layer_name, single_layer_variance in target_variance.items():
                    if is_ignored_layer_variance_correction(layer_name):
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
