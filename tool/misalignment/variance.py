"""Record epoch-end weight variance without changing the shared service API."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn


class EpochWeightVarianceRecorder:
    """Append one CSV row per epoch, with one column per weight tensor.

    Like ServiceVarianceRecorder, select state-dict names containing `weight`
    and compute sample variance (torch.var's default correction=1). Use the
    original model, not a torch.compile wrapper, to preserve layer names.
    Values use .3E: one digit before the decimal plus three after it gives
    four significant digits. A tensor with fewer than two elements has an
    undefined sample variance and is recorded as NAN.
    """

    def __init__(self, model: nn.Module, csv_path: Path) -> None:
        self.model = model
        self.csv_path = Path(csv_path)
        self.layer_names = [name for name in model.state_dict() if "weight" in name]
        with self.csv_path.open("x", newline="", encoding="utf-8") as outfile:
            csv.writer(outfile).writerow(["epoch", *self.layer_names])

    @torch.no_grad()
    def record(self, epoch: int) -> None:
        state = self.model.state_dict()
        # Reduce on the weights' devices; only scalar results move to CPU.
        # Grouping by device avoids a separate GPU synchronization per layer.
        groups: dict[torch.device, list[tuple[int, torch.Tensor]]] = {}
        values = [float("nan")] * len(self.layer_names)
        for index, name in enumerate(self.layer_names):
            weight = state[name]
            if weight.numel() < 2:
                continue
            variance = torch.var(weight, correction=1)
            groups.setdefault(variance.device, []).append((index, variance))
        for items in groups.values():
            variances = torch.stack([variance for _, variance in items]).cpu().tolist()
            for (index, _), variance in zip(items, variances, strict=True):
                values[index] = float(variance)
        with self.csv_path.open("a", newline="", encoding="utf-8") as outfile:
            csv.writer(outfile).writerow([int(epoch), *(f"{value:.3E}" for value in values)])
