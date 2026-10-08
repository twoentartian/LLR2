"""Fused, device-only normalized train-descent / validation-ascent updates.

Only the pure tensor calculation is compiled. Optimizer/scaler steps and
epoch bookkeeping stay outside it, so compilation failure can safely fall
back without applying an update twice or changing Adam's state.
"""

from __future__ import annotations

import logging
import math

import torch
from torch import nn


logger = logging.getLogger("misalignment_measurement_2")


def _weighted_gradient_pair(train_gradient, val_gradient, weights, norms=None):
    if isinstance(weights, torch.Tensor):
        train_weight = weights[0].to(dtype=train_gradient.dtype)
        val_weight = weights[1].to(dtype=val_gradient.dtype)
    else:
        train_weight, val_weight = weights
    if norms is None:
        train_norm, val_norm = train_gradient.norm(), val_gradient.norm()
    else:
        train_norm, val_norm = norms
    train_zero, val_zero = train_norm == 0, val_norm == 0
    train_unit = train_gradient / torch.where(train_zero, 1.0, train_norm)
    val_unit = val_gradient / torch.where(val_zero, 1.0, val_norm)
    return train_weight * train_unit, val_weight * val_unit, train_norm, val_norm, train_zero, val_zero


def _rescale_gradient(train_gradient, train_direction, val_direction, train_norm, val_norm,
                      train_zero=None, val_zero=None):
    if train_zero is None:
        train_zero = train_norm == 0
    if val_zero is None:
        val_zero = val_norm == 0
    signed_unit_sum = train_direction - val_direction
    # Retain the actual norm: the analytic cosine identity is ill-conditioned
    # when directions cancel and would change the update's zero handling.
    signed_unit_norm = signed_unit_sum.norm()
    safe_signed_norm = torch.where(signed_unit_norm == 0, 1.0, signed_unit_norm)
    rescale = torch.where(signed_unit_norm == 0, 0.0, train_norm / safe_signed_norm)
    combined_gradient = signed_unit_sum * rescale
    combined_gradient = torch.where(val_zero, train_gradient, combined_gradient)
    return torch.where(train_zero, 0.0, combined_gradient).detach()


@torch.no_grad()
def _normalize_and_weight_gradients(
    parameters: list[torch.Tensor],
    train_gradients: list[torch.Tensor],
    val_gradients: list[torch.Tensor],
    weights: tuple[float, float] | torch.Tensor,
    collect_geometry: bool = True,
    norms: tuple[list[torch.Tensor], ...] | None = None,
) -> tuple[list[torch.Tensor], ...]:
    """Normalize/weight each side and compute independent reductions.

    Tensor weights are runtime inputs rather than Python-float constants:
    changing epoch weights must not compile another graph. The compiled path
    accepts FP32/FP64 parameter models (including ordinary CUDA AMP, whose
    parameters/gradients stay FP32). Explicit FP16/BF16 parameter models keep
    the eager path: casting scalar weights to those dtypes changes rounding.
    """
    weighted_train, weighted_val = [], []
    train_norms, val_norms, dot_products = [], [], []
    parameter_norms = []
    for index, (parameter, train_gradient, val_gradient) in enumerate(zip(
        parameters, train_gradients, val_gradients, strict=True
    )):
        pair_norms = None if norms is None else (norms[0][index], norms[1][index])
        train_direction, val_direction, train_norm, val_norm, _, _ = _weighted_gradient_pair(train_gradient, val_gradient, weights, pair_norms)
        weighted_train.append(train_direction)
        weighted_val.append(val_direction)
        train_norms.append(train_norm)
        val_norms.append(val_norm)
        dot_products.append(torch.sum(train_gradient * val_gradient))
        if collect_geometry:
            parameter_norms.append(parameter.detach().norm() if norms is None else norms[2][index])
    return weighted_train, weighted_val, train_norms, val_norms, dot_products, parameter_norms


@torch.no_grad()
def _merge_weighted_gradients(
    train_gradients: list[torch.Tensor],
    normalized: tuple[list[torch.Tensor], ...],
    collect_geometry: bool = True,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Merge already-rounded weighted directions, rescale and pack metrics.

    Keeping weighted multiplication in a separate compiled call prevents
    multiply-add contraction across the subtraction even on PyTorch 2.7,
    which does not expose per-graph control of Triton's FP fusion. No global
    environment/compiler setting or optimizer behavior needs to change.
    """
    weighted_train, weighted_val, train_norms, val_norms, dot_products, parameter_norms = normalized
    combined, combined_norms = [], []
    for train_gradient, train_direction, val_direction, train_norm, val_norm in zip(
        train_gradients, weighted_train, weighted_val, train_norms, val_norms, strict=True,
    ):
        combined_gradient = _rescale_gradient(train_gradient, train_direction, val_direction, train_norm, val_norm)
        combined.append(combined_gradient)
        if collect_geometry:
            combined_norms.append(combined_gradient.norm())

    return combined, _pack_geometry(train_norms, val_norms, dot_products, parameter_norms, combined_norms, collect_geometry)


def _pack_geometry(train_norms, val_norms, dot_products, parameter_norms, combined_norms, collect_geometry):
    train_norm_values = torch.stack(train_norms).to(dtype=torch.float64)
    val_norm_values = torch.stack(val_norms).to(dtype=torch.float64)
    valid_cosine = (train_norm_values != 0) & (val_norm_values != 0)
    denominator = torch.where(valid_cosine, train_norm_values * val_norm_values, 1.0)
    cosine_values = torch.where(
        valid_cosine,
        torch.stack(dot_products).to(dtype=torch.float64) / denominator,
        float("nan"),
    )
    if not collect_geometry:
        return cosine_values.unsqueeze(1)
    parameter_norm_values = torch.stack(parameter_norms).to(dtype=torch.float64)
    combined_norm_values = torch.stack(combined_norms).to(dtype=torch.float64)
    relative_denominator = parameter_norm_values.clamp_min(1e-12)
    stats = torch.stack((
        train_norm_values, val_norm_values, combined_norm_values,
        parameter_norm_values, cosine_values,
        train_norm_values / relative_denominator,
        combined_norm_values / relative_denominator,
    ), dim=1)
    return stats


@torch.no_grad()
def combine_normalized_gradient_tensors(
    parameters: list[torch.Tensor],
    train_gradients: list[torch.Tensor],
    val_gradients: list[torch.Tensor],
    weights: tuple[float, float] | torch.Tensor,
    collect_geometry: bool = True,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Uncompiled reference/fallback with the original update and metrics."""
    # Keep the eager loop interleaved per parameter, as before optimization:
    # the fallback need not materialize both weighted directions for the
    # entire model at once. Only the compiled path uses the stage boundary.
    combined, train_norms, val_norms, dot_products = [], [], [], []
    parameter_norms, combined_norms = [], []
    for parameter, train, val in zip(parameters, train_gradients, val_gradients, strict=True):
        train_direction, val_direction, train_norm, val_norm, train_zero, val_zero = _weighted_gradient_pair(train, val, weights)
        gradient = _rescale_gradient(train, train_direction, val_direction, train_norm, val_norm, train_zero, val_zero)
        combined.append(gradient)
        train_norms.append(train_norm)
        val_norms.append(val_norm)
        dot_products.append(torch.sum(train * val))
        if collect_geometry:
            parameter_norms.append(parameter.detach().norm())
            combined_norms.append(gradient.norm())
    return combined, _pack_geometry(train_norms, val_norms, dot_products, parameter_norms, combined_norms, collect_geometry)


class NormalizedGradientCombiner:
    """Per-direction compiled combiner with lazy, permanent eager fallback.

    Native CUDA foreach norms batch the reductions using a consistent path
    on both sides; pointwise/reduction work after them is compiled/fused.
    Each of the two stages has two normal variants (full geometry/cosine only).
    No CUDA graphs are used: returned gradients may be retained by the
    optimizer until the next update, and parameters change every mini-batch.
    """

    def __init__(self, parameters: list[nn.Parameter], *, enabled: bool = True) -> None:
        if not parameters:
            raise ValueError("gradient combination needs at least one parameter")
        self.parameters = parameters
        self.requested = enabled
        self.compiled = None
        self._compiled_normalize = None
        self.fallback_reason: str | None = None
        self._weight_values: tuple[float, float] | None = None
        self._weights: torch.Tensor | None = None
        self._warmed_geometry_modes: set[bool] = set()
        if not enabled:
            return
        if any(parameter.device.type != "cuda" for parameter in parameters):
            self.fallback_reason = "compiled gradient combination requires CUDA parameters"
            return
        if any(parameter.dtype not in (torch.float32, torch.float64) for parameter in parameters):
            self.fallback_reason = "non-FP32/FP64 parameters retain eager scalar-weight rounding"
            logger.info("%s", self.fallback_reason)
            return
        if not hasattr(torch, "compile"):
            self.fallback_reason = "torch.compile is unavailable"
            logger.warning("%s; using eager gradient combination", self.fallback_reason)
            return
        try:
            options = {
                "emulate_precision_casts": True,
                "triton.cudagraphs": False,
                # Fuse independent parameter operations horizontally too,
                # rather than retaining hundreds of per-layer launches.
                "combo_kernels": True,
            }
            self._compiled_normalize = torch.compile(
                _normalize_and_weight_gradients, fullgraph=True, dynamic=False, options=options,
            )
            self.compiled = torch.compile(
                _merge_weighted_gradients, fullgraph=True, dynamic=False, options=options,
            )
            logger.info("gradient combination fusion enabled; first use of each geometry mode will compile")
        except Exception as error:
            self._fallback(error)

    def _fallback(self, error: Exception) -> None:
        self.compiled = None
        self._compiled_normalize = None
        self.fallback_reason = f"{type(error).__name__}: {error}"
        logger.warning("gradient combination compilation failed; using eager for the rest of this direction: %s", error)

    @torch.no_grad()
    def __call__(
        self,
        train_gradients: list[torch.Tensor],
        val_gradients: list[torch.Tensor],
        *,
        train_weight: float,
        val_weight: float,
        collect_geometry: bool = True,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        values = (float(train_weight), float(val_weight))
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("gradient weights must be finite and non-negative")
        if len(train_gradients) != len(self.parameters) or len(val_gradients) != len(self.parameters):
            raise ValueError("gradient and parameter list lengths must match")
        if self.compiled is not None:
            if values != self._weight_values:
                # One tiny transfer per weight change (normally once/epoch),
                # never a per-parameter or per-batch device scalar read.
                self._weights = torch.tensor(values, dtype=torch.float64, device=self.parameters[0].device)
                self._weight_values = values
            try:
                # Native multi-tensor norms use the same reduction path for
                # each side. Compiler-selected split reductions can round
                # identical train/val gradients differently, turning exact
                # cancellation into a residual rescaled to a full update.
                norms = (
                    list(torch._foreach_norm(train_gradients)),
                    list(torch._foreach_norm(val_gradients)),
                    list(torch._foreach_norm(self.parameters)) if collect_geometry else [],
                )
                normalized = self._compiled_normalize(
                    self.parameters, train_gradients, val_gradients, self._weights, collect_geometry, norms,
                )
                result = self.compiled(train_gradients, normalized, collect_geometry)
            except Exception as error:
                # The compiled function has no mutations or optimizer step.
                normalized = None  # Release stage-one temporaries before the eager fallback.
                self._fallback(error)
            else:
                if collect_geometry not in self._warmed_geometry_modes:
                    logger.info("gradient combination fusion ready (collect_geometry=%s)", collect_geometry)
                    self._warmed_geometry_modes.add(collect_geometry)
                return result
        return combine_normalized_gradient_tensors(
            self.parameters, train_gradients, val_gradients, values, collect_geometry,
        )

    def runtime_info(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "enabled": self.compiled is not None and bool(self._warmed_geometry_modes),
            "backend": "inductor" if self.compiled is not None else "eager",
            "fusion_stages": 2,
            "horizontal_fusion": self.compiled is not None,
            "norm_backend": "cuda_foreach" if self.compiled is not None else "eager",
            "fallback_reason": self.fallback_reason,
        }
