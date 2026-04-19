"""Weight-space movement utilities.

Ported from DFL_torch/py_src/model_average.py — only the functions used by
``find_high_accuracy_path.py`` are included.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch

from py_src.special_torch_layers import is_ignored_layer_averaging


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
