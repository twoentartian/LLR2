"""Bound layer-wise weight variance by a reference model.

The reference is either a ``.model.pt`` file (variances are computed from its
tensors) or a ``.csv`` variance record written by ``ServiceVarianceRecorder``
(the last row is used; only the layers it has columns for are bounded).

Every eligible parameter tensor of a training model whose variance exceeds the
variance of the same tensor in a reference model is rescaled around its mean so
its variance equals the reference variance. Tensors whose variance is already at
or below the reference are left untouched, so the reference acts as an upper
bound only.

Parameters owned by normalization modules (BatchNorm, GroupNorm, LayerNorm,
InstanceNorm, RMSNorm, and custom modules whose class name contains "Norm") are
excluded, as are buffers (running statistics, fixed positional tables, ...),
non-floating tensors, and single-element tensors.
"""

from __future__ import annotations

import csv
import logging
from typing import Optional

import torch
import torch.nn as nn

from py_src.model_opti_save_load import model_name_key, stat_dict_key

_NORMALIZATION_TYPES: tuple[type, ...] = (
    nn.modules.batchnorm._NormBase,  # BatchNorm*, SyncBatchNorm, InstanceNorm*
    nn.GroupNorm,
    nn.LayerNorm,
    nn.LocalResponseNorm,
)
if hasattr(nn, "RMSNorm"):
    _NORMALIZATION_TYPES = _NORMALIZATION_TYPES + (nn.RMSNorm,)


def is_normalization_module(module: nn.Module) -> bool:
    if isinstance(module, _NORMALIZATION_TYPES):
        return True
    # Custom norms such as ConvNeXt's LayerNorm2d or a hand-written RMSNorm.
    return "norm" in type(module).__name__.lower()


def normalization_parameter_names(model: nn.Module) -> set[str]:
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if not is_normalization_module(module):
            continue
        prefix = f"{module_name}." if module_name else ""
        for param_name, _ in module.named_parameters(recurse=False):
            names.add(prefix + param_name)
    return names


def load_reference_state_dict(path: str) -> tuple[dict, Optional[str]]:
    """Load a ``.model.pt`` file (``save_model_state`` format) or a raw state dict."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict) and stat_dict_key in data:
        return data[stat_dict_key], data.get(model_name_key)
    if isinstance(data, dict):
        return data, None
    raise TypeError(f"unsupported reference model file content in {path}: {type(data)!r}")


def load_final_variance_from_csv(path: str) -> dict[str, float]:
    """Return ``{layer: variance}`` from the last row of a variance CSV.

    The CSV is the per-node file written by ``ServiceVarianceRecorder``:
    columns ``tick,phase,<layer>,...`` with one row per record.
    """
    with open(path, newline="") as csv_file:
        rows = [row for row in csv.reader(csv_file) if row and any(cell.strip() for cell in row)]
    if len(rows) < 2:
        raise ValueError(f"variance CSV {path} has no data rows")
    header, final_row = rows[0], rows[-1]
    if len(final_row) != len(header):
        raise ValueError(
            f"last row of variance CSV {path} has {len(final_row)} columns, header has {len(header)}"
        )
    non_layer_columns = {"tick", "phase"}
    return {
        name.strip(): float(value)
        for name, value in zip(header, final_row)
        if name.strip() not in non_layer_columns
    }


class ReferenceVarianceBound:
    """Clamp each eligible layer's variance to at most the reference variance."""

    # Ignore float rounding so a layer just scaled to the bound is not
    # reported as over the bound on the next pass.
    relative_tolerance = 1e-5

    def __init__(
        self,
        reference_variance: dict[str, float | torch.Tensor],
        template_model: nn.Module,
        device: torch.device | str | None = None,
        *,
        require_all_layers: bool = True,
    ):
        """Bound ``template_model``'s eligible layers by ``reference_variance``.

        ``reference_variance`` maps parameter names to their maximum variance.
        Entries for normalization or ineligible parameters are ignored. With
        ``require_all_layers`` every eligible layer must have an entry;
        otherwise layers without one stay unbounded (``unbounded_layer_names``).
        """
        excluded_norm = normalization_parameter_names(template_model)
        self.excluded_normalization_names: tuple[str, ...] = tuple(sorted(excluded_norm))

        bounded: dict[str, torch.Tensor] = {}
        unbounded: list[str] = []
        for name, param in template_model.named_parameters():
            if name in excluded_norm:
                continue
            if not param.is_floating_point() or param.numel() <= 1:
                continue
            if name not in reference_variance:
                if require_all_layers:
                    raise KeyError(f"reference has no variance for layer {name!r}")
                unbounded.append(name)
                continue
            variance = torch.as_tensor(reference_variance[name], dtype=torch.float32)
            if variance.numel() != 1 or not torch.isfinite(variance) or variance < 0:
                raise ValueError(f"invalid reference variance for layer {name!r}: {reference_variance[name]!r}")
            bounded[name] = variance.reshape(()).to(device if device is not None else param.device)
        if not bounded:
            raise ValueError("no eligible layers found for the reference variance bound")
        self.reference_variance = bounded
        self.unbounded_layer_names: tuple[str, ...] = tuple(unbounded)

    @classmethod
    def from_state_dict(
        cls,
        reference_state_dict: dict,
        template_model: nn.Module,
        device: torch.device | str | None = None,
    ) -> "ReferenceVarianceBound":
        template_shapes = {name: tuple(p.shape) for name, p in template_model.named_parameters()}
        reference_variance: dict[str, torch.Tensor] = {}
        for name, tensor in reference_state_dict.items():
            if name not in template_shapes or not torch.is_tensor(tensor) or not tensor.is_floating_point():
                continue
            if tuple(tensor.shape) != template_shapes[name]:
                raise ValueError(
                    f"reference tensor {name!r} has shape {tuple(tensor.shape)}, "
                    f"expected {template_shapes[name]}"
                )
            if tensor.numel() > 1:
                reference_variance[name] = torch.var(tensor.detach().to(torch.float32))
        return cls(reference_variance, template_model, device, require_all_layers=True)

    @classmethod
    def from_csv(
        cls,
        path: str,
        template_model: nn.Module,
        device: torch.device | str | None = None,
    ) -> "ReferenceVarianceBound":
        """Use the last row of a ``ServiceVarianceRecorder`` CSV as the bound."""
        return cls(
            load_final_variance_from_csv(path),
            template_model,
            device,
            require_all_layers=False,
        )

    @classmethod
    def from_file(
        cls,
        path: str,
        template_model: nn.Module,
        device: torch.device | str | None = None,
        *,
        expected_model_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> "ReferenceVarianceBound":
        """Load a bound from a ``.csv`` variance record or a ``.model.pt`` file."""
        if str(path).lower().endswith(".csv"):
            return cls.from_csv(path, template_model, device)
        state_dict, model_name = load_reference_state_dict(path)
        if (
            logger is not None
            and expected_model_name is not None
            and model_name is not None
            and model_name != expected_model_name
        ):
            logger.warning(
                "reference model %s was saved as %s, but the simulation uses %s",
                path,
                model_name,
                expected_model_name,
            )
        return cls.from_state_dict(state_dict, template_model, device)

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(self.reference_variance)

    def apply(self, model: nn.Module) -> int:
        """Rescale over-variance layers of ``model`` in place.

        Returns the number of layers that were scaled down.
        """
        parameters = dict(model.named_parameters())
        scaled = []
        with torch.no_grad():
            for name, reference_variance in self.reference_variance.items():
                param = parameters[name]
                value = param.detach().to(torch.float32)
                mean = value.mean()
                variance = value.var()
                over = variance > reference_variance * (1.0 + self.relative_tolerance)
                scale = torch.sqrt(reference_variance / variance.clamp_min(torch.finfo(torch.float32).tiny))
                # Only shrink; layers within the bound keep their exact values.
                # torch.where avoids a per-layer GPU->CPU sync on ``over``.
                param.copy_(torch.where(over, (value - mean) * scale + mean, value).to(param.dtype))
                scaled.append(over)
        return int(torch.stack(scaled).sum().item())
