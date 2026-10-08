"""Epoch-end tokenizer activation diagnostics, without retaining activations."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn


TOKENIZER_RATIO_FIELDS = ("train_tokenizer_nonzero_ratio", "val_tokenizer_nonzero_ratio")


class TokenizerOutputMonitor:
    """Count exact nonzeros in a model's tokenizer output during evaluation.

    Install before the first compiled forward and keep the hook fixed during
    training. When inactive, the hook does nothing. When active, update fixed
    int64 counters on the device: no activation copies, .item() calls or Python
    counter mutations inside the forward hook. This lets torch.compile cache
    the training and diagnostic evaluation paths without per-epoch re-tracing.
    """

    def __init__(self, model: nn.Module) -> None:
        while hasattr(model, "_orig_mod"):
            model = model._orig_mod
        tokenizer = getattr(model, "tokenizer", None)
        self.available = isinstance(tokenizer, nn.Module)
        self.active = False
        self._counts = None
        self._handle = None
        if self.available:
            parameter = next(model.parameters(), None)
            device = parameter.device if parameter is not None else torch.device("cpu")
            self._counts = torch.zeros(2, dtype=torch.int64, device=device)
            self._handle = tokenizer.register_forward_hook(self._observe)

    def _observe(self, module, inputs, output) -> None:
        if not self.active:
            return
        if not isinstance(output, torch.Tensor):
            raise TypeError("tokenizer nonzero measurement requires a tensor output")
        # Count the returned tokenizer tensor, before positional embeddings or
        # classifier processing. Counts can exceed 2**31 over ImageNet epochs.
        self._counts[0].add_(torch.count_nonzero(output.detach()))
        self._counts[1].add_(output.numel())

    @contextmanager
    def capture(self):
        if self.active:
            raise RuntimeError("tokenizer captures cannot be nested")
        if self.available:
            self._counts.zero_()
            self.active = True
        try:
            yield self
        finally:
            self.active = False

    @property
    def nonzero_ratio(self) -> float | None:
        if not self.available:
            return None
        nonzero, elements = self._counts.tolist()
        return nonzero / elements if elements else None

    def close(self) -> None:
        self.active = False
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
