"""Variance correction utilities.

Ported from DFL_torch/py_src/model_variance_correct.py.
"""

from __future__ import annotations

import copy
import enum
from typing import Dict, Optional

import torch

from py_src.special_torch_layers import is_ignored_layer_variance_correction


class VarianceCorrectionType(enum.Enum):
    FollowSelfVariance = 0
    FollowConservative = 1
    FollowOthers = 2


class VarianceCorrector:
    def __init__(self, variance_correction_type: VarianceCorrectionType):
        self.variance_correction_type = variance_correction_type
        self.variance_record: Optional[Dict[str, float]] = None
        self.model_counter: int = 0

    # ------------------------------------------------------------------

    @staticmethod
    def calculate_variance_for_tensor(tensor: torch.Tensor) -> float:
        if tensor.numel() == 1:
            return 0.0
        return torch.var(tensor).item()

    def add_variance(self, model_stat: dict) -> None:
        if self.variance_correction_type in (
            VarianceCorrectionType.FollowConservative,
            VarianceCorrectionType.FollowOthers,
        ):
            if self.variance_record is None:
                self.variance_record = {name: 0.0 for name in model_stat}
            for name, param in model_stat.items():
                if is_ignored_layer_variance_correction(name):
                    continue
                self.variance_record[name] += self.calculate_variance_for_tensor(param)
            self.model_counter += 1

    def get_variance(
        self,
        self_model_stat: Optional[dict] = None,
        conservative: Optional[float] = None,
    ) -> Dict[str, float]:
        output: Dict[str, float] = {}

        if self.variance_correction_type == VarianceCorrectionType.FollowSelfVariance:
            assert self_model_stat is not None and conservative is not None
            for name, param in self_model_stat.items():
                output[name] = self.calculate_variance_for_tensor(param)

        elif self.variance_correction_type == VarianceCorrectionType.FollowConservative:
            assert self_model_stat is not None and conservative is not None and self.variance_record is not None
            c = float(conservative)
            self_var = {name: self.calculate_variance_for_tensor(p) for name, p in self_model_stat.items()}
            for name, var in self.variance_record.items():
                output[name] = self_var[name] * c + (var / self.model_counter) * (1 - c)

        elif self.variance_correction_type == VarianceCorrectionType.FollowOthers:
            assert self.variance_record is not None
            for name, var in self.variance_record.items():
                output[name] = var / self.model_counter

        # Reset accumulators
        self.model_counter = 0
        self.variance_record = None
        return output

    # ------------------------------------------------------------------

    @staticmethod
    def scale_tensor_to_variance(layer_tensor: torch.Tensor, target_variance: float) -> torch.Tensor:
        epsilon = 1e-4
        with torch.no_grad():
            if layer_tensor.numel() == 1:
                return layer_tensor
            mean = torch.mean(layer_tensor)
            cur_var = torch.var(layer_tensor)
            scale = torch.sqrt((target_variance + epsilon) / (cur_var + epsilon))
            return (layer_tensor - mean) * scale + mean

    @staticmethod
    def scale_model_stat_to_variance(
        model_stat: dict,
        target_variance: Dict[str, float],
        ignore_layer_list: Optional[list] = None,
    ) -> dict:
        out = copy.deepcopy(model_stat)
        ignore = set(ignore_layer_list) if ignore_layer_list else set()
        for layer_name, tvar in target_variance.items():
            if is_ignored_layer_variance_correction(layer_name):
                continue
            if layer_name in ignore:
                continue
            if layer_name not in model_stat:
                continue
            out[layer_name] = VarianceCorrector.scale_tensor_to_variance(model_stat[layer_name], tvar)
        return out
