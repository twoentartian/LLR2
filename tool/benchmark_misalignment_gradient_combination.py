#!/usr/bin/env python3
"""Benchmark only gradient combination, without downloading a dataset.

Uses the real CCT-14 ImageNet parameter shapes and synthetic FP32 gradients.
Compilation/warmup are excluded from timing; this is not an epoch benchmark.
Run in the same conda/CUDA environment as misalignment_measurement_2.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from py_src.third_party.compact_transformers.src.cct import cct_14_7x2_224
from misalignment.gradient_combination import NormalizedGradientCombiner, combine_normalized_gradient_tensors


def measure_pair(eager, fused, iterations: int, repeats: int):
    samples = {"eager": [], "fused": []}
    for repeat in range(repeats):
        order = (("eager", eager), ("fused", fused))
        if repeat % 2:
            order = order[::-1]
        for name, fn in order:
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(iterations):
                fn()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - started) * 1000 / iterations)
    return tuple({"median_ms": statistics.median(samples[name]), "samples_ms": samples[name]} for name in ("eager", "fused"))


def kernel_count(fn) -> int:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        fn()
        torch.cuda.synchronize()
    return sum(
        event.device_type == torch.autograd.DeviceType.CUDA
        and not event.name.startswith(("Memcpy", "Memset"))
        for event in profile.events()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile_kernels", action="store_true")
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.warmup) <= 0:
        parser.error("iterations, repeats and warmup must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    model = cct_14_7x2_224(pretrained=False, num_classes=1000).cuda()
    parameters = [p for p in model.parameters() if p.requires_grad]
    train = [torch.randn_like(p) for p in parameters]
    val = [torch.randn_like(p) for p in parameters]
    combiner = NormalizedGradientCombiner(parameters)
    result = {
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "parameter_tensors": len(parameters), "parameter_elements": sum(p.numel() for p in parameters),
        "gradient_dtype": str(train[0].dtype), "weights": [.8, .2],
        "iterations": args.iterations, "repeats": args.repeats, "geometry_modes": {},
    }
    profiling_functions = []
    for collect in (True, False):
        def eager(collect_geometry=collect):
            return combine_normalized_gradient_tensors(parameters, train, val, (.8, .2), collect_geometry)

        def fused(collect_geometry=collect):
            return combiner(train, val, train_weight=.8, val_weight=.2, collect_geometry=collect_geometry)

        torch.cuda.synchronize()
        started = time.perf_counter()
        actual, actual_stats = fused()
        torch.cuda.synchronize()
        first_call_s = time.perf_counter() - started
        if not combiner.runtime_info()["enabled"]:
            raise RuntimeError(f"combination did not compile: {combiner.fallback_reason}")
        expected, expected_stats = eager()
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_stats, expected_stats, rtol=1e-5, atol=1e-6, equal_nan=True)
        del actual, expected, actual_stats, expected_stats
        # Validate cancellation at large reduction sizes as well as in the
        # small unit tests; a tiny residual would be rescaled to a full step.
        identical_val = [g.clone() for g in train]
        cancelled, _ = combiner(train, identical_val, train_weight=.2, val_weight=.2, collect_geometry=collect)
        for gradient in cancelled:
            if torch.count_nonzero(gradient).item() != 0:
                raise AssertionError("exact train/val cancellation was not preserved")
        del identical_val, cancelled
        for _ in range(args.warmup):
            eager()
            fused()
        eager_time, fused_time = measure_pair(eager, fused, args.iterations, args.repeats)
        mode = {
            "first_fused_call_seconds_including_compilation": first_call_s,
            "eager": eager_time, "fused": fused_time,
            "speedup": eager_time["median_ms"] / fused_time["median_ms"],
        }
        profiling_functions.append((mode, eager, fused))
        result["geometry_modes"][str(collect)] = mode
        logging.info("geometry=%s: eager %.3f ms, fused %.3f ms, %.2fx", collect,
                     eager_time["median_ms"], fused_time["median_ms"], mode["speedup"])
    # Profile after all timings: profiler startup/cleanup must not disturb
    # subsequent timing samples, especially the many-launch eager path.
    if args.profile_kernels:
        for mode, eager, fused in profiling_functions:
            mode["eager_cuda_kernels"] = kernel_count(eager)
            mode["fused_cuda_kernels"] = kernel_count(fused)
    result["runtime"] = combiner.runtime_info()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
