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

Modes (``mode``):

* ``"always"``: every check clamps any layer above its reference variance.
* ``"wait_until_below"``: tracked per model (node) and per layer. A layer is
  not clamped until its variance first drops to or below the reference
  variance; from then on it is clamped like in ``"always"``. Layers that start
  at or below the reference are enabled immediately.
"""

from __future__ import annotations

import csv
import logging
from typing import Hashable, NamedTuple, Optional

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


MODE_ALWAYS = "always"
MODE_WAIT_UNTIL_BELOW = "wait_until_below"
MODES = (MODE_ALWAYS, MODE_WAIT_UNTIL_BELOW)


class BoundResult(NamedTuple):
    scaled: int
    """Layers scaled down by this call."""
    newly_enabled: int
    """Layers whose bound was enabled by this call (``wait_until_below`` only)."""
    enabled: int
    """Layers whose bound is enabled after this call."""


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
        mode: str = MODE_ALWAYS,
    ):
        """Bound ``template_model``'s eligible layers by ``reference_variance``.

        ``reference_variance`` maps parameter names to their maximum variance.
        Entries for normalization or ineligible parameters are ignored. With
        ``require_all_layers`` every eligible layer must have an entry;
        otherwise layers without one stay unbounded (``unbounded_layer_names``).
        ``mode`` is one of ``MODES`` (see the module docstring).
        """
        if mode not in MODES:
            raise ValueError(f"unknown reference variance mode {mode!r}, expected one of {MODES}")
        self.mode = mode
        # wait_until_below: per model key, per layer, whether the bound is enabled.
        self._enabled: dict[Hashable, dict[str, torch.Tensor]] = {}
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
        *,
        mode: str = MODE_ALWAYS,
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
        return cls(reference_variance, template_model, device, require_all_layers=True, mode=mode)

    @classmethod
    def from_csv(
        cls,
        path: str,
        template_model: nn.Module,
        device: torch.device | str | None = None,
        *,
        mode: str = MODE_ALWAYS,
    ) -> "ReferenceVarianceBound":
        """Use the last row of a ``ServiceVarianceRecorder`` CSV as the bound."""
        return cls(
            load_final_variance_from_csv(path),
            template_model,
            device,
            require_all_layers=False,
            mode=mode,
        )

    @classmethod
    def from_file(
        cls,
        path: str,
        template_model: nn.Module,
        device: torch.device | str | None = None,
        *,
        mode: str = MODE_ALWAYS,
        expected_model_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> "ReferenceVarianceBound":
        """Load a bound from a ``.csv`` variance record or a ``.model.pt`` file."""
        if str(path).lower().endswith(".csv"):
            return cls.from_csv(path, template_model, device, mode=mode)
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
        return cls.from_state_dict(state_dict, template_model, device, mode=mode)

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(self.reference_variance)

    def apply(self, model: nn.Module, key: Hashable = None) -> BoundResult:
        """Rescale over-variance layers of ``model`` in place.

        ``key`` identifies the model (e.g. the node name) for the per-layer
        enable state of ``wait_until_below``; it is ignored in ``always``.
        """
        parameters = dict(model.named_parameters())
        wait = self.mode == MODE_WAIT_UNTIL_BELOW
        enabled_state = self._enabled.setdefault(key, {}) if wait else None
        scaled, newly_enabled, enabled = [], [], []
        with torch.no_grad():
            for name, reference_variance in self.reference_variance.items():
                param = parameters[name]
                value = param.detach().to(torch.float32)
                mean = value.mean()
                variance = value.var()
                over = variance > reference_variance * (1.0 + self.relative_tolerance)
                if wait:
                    was_enabled = enabled_state.get(name)
                    if was_enabled is None:
                        was_enabled = torch.zeros((), dtype=torch.bool, device=variance.device)
                    now_enabled = was_enabled | (variance <= reference_variance)
                    enabled_state[name] = now_enabled
                    newly_enabled.append(now_enabled & ~was_enabled)
                    enabled.append(now_enabled)
                    over = over & now_enabled
                scale = torch.sqrt(reference_variance / variance.clamp_min(torch.finfo(torch.float32).tiny))
                # Only shrink; layers within the bound keep their exact values.
                # torch.where avoids a per-layer GPU->CPU sync on ``over``.
                param.copy_(torch.where(over, (value - mean) * scale + mean, value).to(param.dtype))
                scaled.append(over)
        if not wait:
            count = int(torch.stack(scaled).sum().item())
            return BoundResult(count, 0, len(self.reference_variance))
        counts = torch.stack(
            [torch.stack(scaled).sum(), torch.stack(newly_enabled).sum(), torch.stack(enabled).sum()]
        ).tolist()
        return BoundResult(*(int(c) for c in counts))
