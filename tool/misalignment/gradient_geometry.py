"""Device-side epoch statistics for the normalized two-sided update."""

from __future__ import annotations

import torch


GRADIENT_GEOMETRY_FIELDS = (
    "train_gradient_norm",
    "val_gradient_norm",
    "combined_gradient_norm",
    "parameter_norm",
    "train_val_gradient_cosine",
    "relative_train_gradient",
    "relative_combined_gradient",
)


class GradientGeometryAccumulator:
    """Average finite batch values, with only one host transfer per epoch.

    Counts are per parameter and per field: zero-gradient cosine NaNs and
    other nonfinite diagnostics must not change the existing CSV averages.
    Float64 accumulation matches the previous Python-float accumulation.
    """

    def __init__(self, parameter_names: list[str], fields: tuple[str, ...], device: torch.device) -> None:
        self.parameter_names = parameter_names
        self.fields = fields
        shape = (len(parameter_names), len(fields))
        self.sums = torch.zeros(shape, dtype=torch.float64, device=device)
        self.counts = torch.zeros(shape, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        if values.shape != self.sums.shape:
            raise ValueError("gradient geometry shape does not match the accumulator")
        values = values.detach().to(dtype=torch.float64)
        finite = torch.isfinite(values)
        self.sums.add_(torch.where(finite, values, 0.0))
        self.counts.add_(finite)

    @torch.no_grad()
    def averages(self) -> list[dict[str, float | str]]:
        means = torch.where(
            self.counts > 0,
            self.sums / self.counts.clamp_min(1),
            float("nan"),
        )
        # The sole device-to-host copy; never called inside the minibatch loop.
        host_values = means.cpu().tolist()
        return [
            {"parameter": name, **dict(zip(self.fields, values, strict=True))}
            for name, values in zip(self.parameter_names, host_values, strict=True)
        ]
