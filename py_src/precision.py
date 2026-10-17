"""Precision policy for forward-only evaluation, independent of training AMP."""

from contextlib import nullcontext

import torch


def inference_dtype(device: torch.device, precision: str = "bf16") -> torch.dtype | None:
    """Prefer native CUDA BF16; keep other devices and explicit FP32 in FP32."""
    if precision not in ("bf16", "fp32"):
        raise ValueError(f"Unknown evaluation precision {precision!r}; expected bf16 or fp32")
    if precision == "bf16" and device.type == "cuda":
        with torch.cuda.device(device):
            if torch.cuda.is_bf16_supported(including_emulation=False):
                return torch.bfloat16
    return None


def inference_autocast(device: torch.device, precision: str = "bf16"):
    """Scope autocast to inference without changing model or optimizer storage."""
    dtype = inference_dtype(device, precision)
    if device.type in ("cuda", "cpu"):
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)
    return nullcontext()
