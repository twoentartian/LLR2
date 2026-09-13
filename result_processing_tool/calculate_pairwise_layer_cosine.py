#!/usr/bin/env python3
"""Mean layer-wise cosine similarity over all unordered checkpoint pairs.

Example (from repository root):
  python3 result_processing_tool/calculate_pairwise_layer_cosine.py tool/high_accuracy_BNN
  python3 result_processing_tool/calculate_pairwise_layer_cosine.py MODELS \
      -o mean.csv --pairs-file pairs.csv --key-regex '^(conv|fc)'

Default selection/grouping is identical to plot_layer_weight_pca.py: concatenate
each module's weight and bias, excluding BN running statistics. Each pair has
equal weight in the mean; self-pairs and duplicate reverse pairs are excluded.
Undefined zero-norm pairs are NaN, excluded from the mean, and counted explicitly.
If no valid pair exists for a layer its mean/std/min/max are NaN.
Uses temporary disk caching and feature-blocked N x N Gram matrices: O(N**2)
matrix memory, plus one checkpoint and an N x block-size working block.
Dependencies: torch, numpy. No dataset or GPU required.
"""

import argparse
from contextlib import ExitStack
import csv
from pathlib import Path
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from result_processing_tool.model_weight_utils import cached_weights, natural_key


def pairwise_cosines(matrix: np.ndarray, block_size: int = 65536) -> np.ndarray:
    """Cosine matrix; zero-norm rows/columns are NaN, not artificial zeros."""
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1 or block_size < 1:
        raise ValueError("Need >=2 models, >=1 feature and a positive block size")
    n = matrix.shape[0]
    gram = np.zeros((n, n), dtype=np.float64)
    for start in range(0, matrix.shape[1], block_size):
        block = np.asarray(matrix[:, start:start + block_size], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Non-finite weights")
        gram += block @ block.T
    if not np.isfinite(gram).all():
        raise ValueError("Gram matrix overflow; weight magnitudes are too large")
    norms = np.sqrt(np.maximum(np.diag(gram), 0))
    valid = norms > 0
    # Divide in two steps to avoid overflowing the product of norms.
    np.divide(gram, norms[:, None], out=gram, where=valid[:, None])
    np.divide(gram, norms[None, :], out=gram, where=valid[None, :])
    gram[~valid, :] = np.nan
    gram[:, ~valid] = np.nan
    np.clip(gram, -1, 1, out=gram)
    return gram


def calculate(args) -> Path:
    source = args.input_dir.resolve()
    if not source.is_dir():
        raise ValueError(f"Not a directory: {source}")
    files = sorted({p.resolve() for p in source.glob(args.pattern) if p.is_file()}, key=natural_key)
    if len(files) < 2:
        raise ValueError("Need at least two model checkpoints")
    output = (args.output or source / "layer_cosine_summary.csv").resolve()
    pairs_path = args.pairs_file.resolve() if args.pairs_file else None
    if output == pairs_path or output in files or pairs_path in files:
        raise ValueError("Output paths must be distinct and must not overwrite checkpoints")
    for path in [output, pairs_path]:
        if path:
            if path.exists():
                raise ValueError(f"Output already exists: {path}; choose another output path")
            path.parent.mkdir(parents=True, exist_ok=True)
    with cached_weights(files, key_regex=args.key_regex, exclude_regex=args.exclude_regex,
                        layer_level=args.layer_level, exclude_bias=args.exclude_bias,
                        cache_dir=args.cache_dir) as (cache, layers, slices, metadata):
        rows, cols = np.triu_indices(len(files), k=1)
        with ExitStack() as stack:
            summary = csv.writer(stack.enter_context(output.open("x", newline="", encoding="utf-8")))
            summary.writerow(["layer", "model_type", "dataset_type", "model_count", "parameter_count",
                              "pair_count", "valid_pair_count", "undefined_pair_count",
                              "mean_cosine", "std_cosine", "min_cosine", "max_cosine"])
            pairs = None
            if pairs_path:
                pairs = csv.writer(stack.enter_context(pairs_path.open("x", newline="", encoding="utf-8")))
                pairs.writerow(["layer", "model_a", "model_b", "cosine"])
            for layer, (start, end) in slices.items():
                cosines = pairwise_cosines(cache[:, start:end], args.block_size)[rows, cols]
                values = cosines[np.isfinite(cosines)]
                stats = ([values.mean(), values.std(), values.min(), values.max()]
                         if values.size else [float("nan")] * 4)
                summary.writerow([layer, *metadata, len(files), end - start, len(rows),
                                  values.size, len(rows) - values.size, *stats])
                if pairs:
                    pairs.writerows((layer, str(files[i]), str(files[j]), float(value))
                                    for i, j, value in zip(rows, cols, cosines))
                print(f"{layer}: mean={stats[0]:.8f}, valid={values.size}/{len(rows)}", flush=True)
    print(f"Saved layer means: {output}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="Summary CSV (default: INPUT/layer_cosine_summary.csv)")
    parser.add_argument("--pairs-file", type=Path, help="Optional detailed CSV for every layer and pair")
    parser.add_argument("--pattern", default="*.model.pt")
    parser.add_argument("--key-regex")
    parser.add_argument("--exclude-regex")
    parser.add_argument("--exclude-bias", action="store_true")
    parser.add_argument("--layer-level", type=int)
    parser.add_argument("--block-size", type=int, default=65536)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.block_size, args.threads, args.layer_level or 1) < 1 or args.layer_level == 0:
        parser.error("block-size, threads and layer-level must be positive")
    torch.set_num_threads(args.threads)
    try:
        calculate(args)
    except (ValueError, OSError, re.error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
