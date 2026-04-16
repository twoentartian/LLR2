"""Shared types for py_src_new."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional


# ---------------------------------------------------------------------------
# Step output – returned by every train / val step
# ---------------------------------------------------------------------------

@dataclass
class StepOutput:
    """Result of a single training or validation step (one batch)."""
    loss: float
    sample_count: int
    correct_count: Optional[int] = None
    extra: dict = field(default_factory=dict)
