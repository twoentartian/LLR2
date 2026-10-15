"""Streaming ViT token norms measured before the final LayerNorm."""

from __future__ import annotations

from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path

import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer
from py_src import distributed


class TokenNormMonitor:
    """Observe evaluation forwards without retaining token embeddings.

    A fixed pre-hook on encoder.ln works for torchvision ViT and the register
    variant, including torch.compile. Training and Hessian forwards are not
    captured. Counts and histograms are accumulated on the model's device.
    """

    def __init__(self, model: nn.Module, *, threshold: float = 150.0, enabled: bool = True) -> None:
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("token norm threshold must be finite and positive")
        while hasattr(model, "_orig_mod"):
            model = model._orig_mod
        self.available = enabled and isinstance(model, VisionTransformer)
        self.threshold = float(threshold)
        self.num_register_tokens = int(getattr(model, "num_register_tokens", 0))
        self.groups = ["patch", "cls", *[f"register_{i}" for i in range(self.num_register_tokens)]]
        # Fixed logarithmic bins plus the chosen threshold; the last bin has
        # no upper bound, so very large norms are never silently discarded.
        self.histogram_edges = sorted({float(2**i) for i in range(13)} | {self.threshold})
        self.active = False
        self._handle = None
        if self.available:
            device = next(model.parameters()).device
            self._edges = torch.tensor(self.histogram_edges, dtype=torch.float32, device=device)
            # count, sum, sum of squares, maximum, count above threshold
            self._stats = torch.zeros((len(self.groups), 5), dtype=torch.float64, device=device)
            self._stats[:, 3].fill_(-float("inf"))
            self._histograms = torch.zeros(
                (len(self.groups), len(self.histogram_edges) + 1), dtype=torch.int64, device=device,
            )
            self._handle = model.encoder.ln.register_forward_pre_hook(self._observe)

    def _observe(self, module, inputs) -> None:
        if not self.active:
            return
        tokens = inputs[0].detach()
        norms = tokens.float().norm(dim=-1)
        values = [norms[:, 1 + self.num_register_tokens :].reshape(-1), norms[:, 0]]
        values.extend(norms[:, 1 + i] for i in range(self.num_register_tokens))
        for index, group_norms in enumerate(values):
            group_norms = group_norms.contiguous()
            self._stats[index, 0].add_(group_norms.numel())
            self._stats[index, 1].add_(group_norms.sum(dtype=torch.float64))
            self._stats[index, 2].add_(group_norms.double().square().sum())
            self._stats[index, 3].copy_(torch.maximum(self._stats[index, 3], group_norms.amax()))
            self._stats[index, 4].add_((group_norms > self.threshold).sum())
            bins = torch.bucketize(group_norms, self._edges, right=True)
            self._histograms[index].scatter_add_(0, bins, torch.ones_like(bins))

    @contextmanager
    def capture(self):
        if self.active:
            raise RuntimeError("token norm captures cannot be nested")
        if self.available:
            self._stats.zero_()
            self._stats[:, 3].fill_(-float("inf"))
            self._histograms.zero_()
            self.active = True
        try:
            yield self
        finally:
            self.active = False

    @property
    def statistics(self) -> dict[str, dict] | None:
        if not self.available:
            return None
        stats = self._stats.cpu().tolist()
        histograms = self._histograms.cpu().tolist()
        if not stats[0][0]:
            return None
        result = {}
        for group, (count, total, squared, maximum, high_count), histogram in zip(
            self.groups, stats, histograms, strict=True,
        ):
            mean = total / count
            variance = squared / count - mean * mean
            result[group] = {
                "count": int(count), "mean": mean, "std": math.sqrt(max(variance, 0.0)),
                "max": maximum, "high_norm_ratio": high_count / count,
                "histogram_counts": histogram,
            }
        return result

    def synchronize(self) -> None:
        if self.available and distributed.enabled():
            maxima = self._stats[:, 3].clone()
            torch.distributed.all_reduce(maxima, op=torch.distributed.ReduceOp.MAX)
            self._stats[:, 3].zero_()
            distributed.sum_tensor(self._stats)
            self._stats[:, 3].copy_(maxima)
            distributed.sum_tensor(self._histograms)

    def metadata(self) -> dict:
        return {
            "enabled": self.available,
            "layer": "encoder.ln.input",
            "norm": "L2 of each token in FP32 before final LayerNorm",
            "measurement": "epoch_end_full_partition_eval_mode",
            "num_register_tokens": self.num_register_tokens,
            "high_norm_threshold": self.threshold,
            "high_norm_comparison": ">",
            "histogram_edges": self.histogram_edges,
            "histogram_bins": "[0, first_edge), [edge_i, edge_i+1), [last_edge, infinity)",
            "std_definition": "population standard deviation across tokens",
        }

    def close(self) -> None:
        self.active = False
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class TokenNormRecorder:
    """Persist one row per epoch, train/val role and token group."""

    FIELDS = ("epoch", "role", "token_group", "count", "mean", "std", "max",
              "high_norm_ratio", "histogram_counts")

    def __init__(self, path: Path, monitor: TokenNormMonitor) -> None:
        self.path = path
        self.monitor = monitor
        self.rows = []
        if monitor.available:
            self._write()

    def record(self, epoch: int, train: dict | None, val: dict | None) -> None:
        if not self.monitor.available:
            return
        for role, statistics in (("train", train), ("val", val)):
            for group, values in (statistics or {}).items():
                self.rows.append({
                    "epoch": epoch, "role": role, "token_group": group,
                    **values, "histogram_counts": json.dumps(values["histogram_counts"]),
                })
        self._write()

    def _write(self) -> None:
        if not distributed.is_main():
            return
        with self.path.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
