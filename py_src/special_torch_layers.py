"""Utility functions for inspecting and classifying model layers.

Ported from DFL_torch/py_src/special_torch_layers.py.
"""

import re
import torch.nn as nn


# ---------------------------------------------------------------------------
# Keyword helpers
# ---------------------------------------------------------------------------

def is_keyword_in_layer_name(layer_name: str, keywords) -> bool:
    for kw in keywords:
        if kw in layer_name:
            return True
    return False


def is_layer_index_with_keyword(layer_name: str, keywords) -> bool:
    for kw in keywords:
        m = re.search(r'[^.]+\.(\d+)', layer_name)
        layer_index = int(m.group(1)) if m else None
        m2 = re.search(r'[^.]+\.(\d+)', kw)
        layer_index_in_keyword = int(m2.group(1)) if m2 else None
        if layer_name.startswith(kw) and layer_index == layer_index_in_keyword:
            return True
    return False


# ---------------------------------------------------------------------------
# Layer filtering predicates
# ---------------------------------------------------------------------------

_IGNORE_AVERAGING = ["num_batches_tracked", "running_mean", "running_var"]
_IGNORE_VARIANCE_CORRECTION = ["num_batches_tracked", "running_mean", "running_var"]


def is_ignored_layer_averaging(layer_name: str) -> bool:
    return is_keyword_in_layer_name(layer_name, _IGNORE_AVERAGING)


def is_ignored_layer_variance_correction(layer_name: str) -> bool:
    return is_keyword_in_layer_name(layer_name, _IGNORE_VARIANCE_CORRECTION)


# ---------------------------------------------------------------------------
# Layer search
# ---------------------------------------------------------------------------

def find_layers_according_to_name_and_keyword(
    model_state_dict,
    layer_names=None,
    layer_name_keywords=None,
    layer_name_match_layer_index=None,
):
    """Return (found_layers, ignored_layers) based on name/keyword matching."""
    _names = layer_names or []
    _keywords = layer_name_keywords or []
    _index_kws = layer_name_match_layer_index or []

    found = []
    for layer_name in model_state_dict.keys():
        if layer_name in _names:
            found.append(layer_name)
        elif is_keyword_in_layer_name(layer_name, _keywords):
            found.append(layer_name)
        elif is_layer_index_with_keyword(layer_name, _index_kws):
            found.append(layer_name)

    ignored = [l for l in model_state_dict.keys() if l not in found]
    return found, ignored


# ---------------------------------------------------------------------------
# Normalization layer detection
# ---------------------------------------------------------------------------

class NormalizationLayerResults:
    def __init__(self):
        self.batch_normalization = []
        self.layer_normalization = []
        self.group_normalization = []
        self.instance_normalization = []


def find_normalization_layers(model: nn.Module) -> NormalizationLayerResults:
    """Walk ``model.named_modules()`` and collect normalisation layer names."""
    out = NormalizationLayerResults()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            out.batch_normalization.append(name)
        if isinstance(module, nn.LayerNorm):
            out.layer_normalization.append(name)
        if isinstance(module, nn.GroupNorm):
            out.group_normalization.append(name)
        if isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            out.instance_normalization.append(name)
    return out
