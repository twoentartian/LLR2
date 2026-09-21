#!/usr/bin/env python3
"""Mean layer-wise cosine similarity over all unordered checkpoint pairs.

Example (from repository root):
  python3 result_processing_tool/calculate_pairwise_layer_cosine.py tool/high_accuracy_BNN
  python3 result_processing_tool/calculate_pairwise_layer_cosine.py MODELS \
      --key-regex '^(conv|fc)'

Default selection/grouping is identical to plot_layer_weight_pca.py: concatenate
each module's weight and bias, excluding BN running statistics. Each pair has
equal weight in the mean; self-pairs and duplicate reverse pairs are excluded.
Undefined zero-norm pairs are NaN, excluded from the mean, and counted explicitly.
If no valid pair exists for a layer its mean/std/min/max are NaN. By default,
the summary CSV, detailed pair CSV, and one-page distribution PDF are written
to INPUT/layer_cosine_summary.csv, INPUT/layer_cosine_pairs.csv, and
INPUT/layer_cosine_distribution.pdf respectively. The PDF contains one
histogram subplot per layer on one page.
Uses temporary disk caching and feature-blocked N x N Gram matrices: O(N**2)
matrix memory, plus one checkpoint and an N x block-size working block.
Dependencies: torch, numpy, matplotlib. No dataset or GPU required.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter


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
    output_is_default = args.output is None
    pairs_is_default = args.pairs_file is None
    output = (args.output or source / "layer_cosine_summary.csv").resolve()
    pairs_path = (args.pairs_file or source / "layer_cosine_pairs.csv").resolve()
    distribution_path = (args.distribution_file or source / "layer_cosine_distribution.pdf").resolve()
    if output == pairs_path or output in files or pairs_path in files:
        raise ValueError("Output paths must be distinct and must not overwrite checkpoints")
    if distribution_path in files:
        raise ValueError("Distribution output must not overwrite a checkpoint")
    for path, is_default in [(output, output_is_default), (pairs_path, pairs_is_default)]:
        if path:
            if path.exists() and not is_default:
                raise ValueError(f"Output already exists: {path}; choose another output path")
            path.parent.mkdir(parents=True, exist_ok=True)
    distribution_path.parent.mkdir(parents=True, exist_ok=True)
    with cached_weights(files, key_regex=args.key_regex, exclude_regex=args.exclude_regex,
                        layer_level=args.layer_level, exclude_bias=args.exclude_bias,
                        cache_dir=args.cache_dir) as (cache, layers, slices, metadata):
        rows, cols = np.triu_indices(len(files), k=1)
        with ExitStack() as stack:
            summary_mode = "w" if output_is_default else "x"
            pairs_mode = "w" if pairs_is_default else "x"
            summary = csv.writer(stack.enter_context(output.open(summary_mode, newline="", encoding="utf-8")))
            summary.writerow(["layer", "model_type", "dataset_type", "model_count", "parameter_count",
                              "pair_count", "valid_pair_count", "undefined_pair_count",
                              "mean_cosine", "std_cosine", "min_cosine", "max_cosine"])
            pairs = None
            pairs = csv.writer(stack.enter_context(pairs_path.open(pairs_mode, newline="", encoding="utf-8")))
            pairs.writerow(["layer", "model_a", "model_b", "cosine"])
            distributions = []
            for layer, (start, end) in slices.items():
                cosines = pairwise_cosines(cache[:, start:end], args.block_size)[rows, cols]
                values = cosines[np.isfinite(cosines)]
                stats = ([values.mean(), values.std(), values.min(), values.max()]
                         if values.size else [float("nan")] * 4)
                summary.writerow([layer, *metadata, len(files), end - start, len(rows),
                                  values.size, len(rows) - values.size, *stats])
                pairs.writerows((layer, str(files[i]), str(files[j]), float(value))
                                for i, j, value in zip(rows, cols, cosines))
                distributions.append((layer, values.copy(), stats[0], len(rows) - values.size))
                print(f"{layer}: mean={stats[0]:.8f}, valid={values.size}/{len(rows)}", flush=True)
        _write_distribution_pdf(distribution_path, distributions, metadata, len(files),
                                bins=args.bins, range_padding=args.range_padding)
    print(f"Saved layer means: {output}", flush=True)
    print(f"Saved pair details: {pairs_path}", flush=True)
    print(f"Saved layer distributions: {distribution_path}", flush=True)
    return output


def _automatic_plot_range(values: np.ndarray, padding_fraction: float) -> tuple[float, float]:
    """Choose a narrow, per-layer x range while keeping constant layers visible."""
    if values.size == 0:
        return -1.0, 1.0
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = maximum - minimum
    if span <= np.finfo(np.float64).eps:
        half_width = max(max(abs(minimum), 1.0) * 1e-6, 1e-6)
        return minimum - half_width, maximum + half_width
    padding = span * padding_fraction
    return minimum - padding, maximum + padding


def _write_distribution_pdf(path: Path, distributions: list[tuple], metadata: tuple,
                            model_count: int, bins: int = 60,
                            range_padding: float = 0.05) -> None:
    """Write one-page PDF containing one histogram subplot per layer."""
    if not distributions:
        raise ValueError("No layer distributions to plot")
    columns = min(4, max(1, len(distributions)))
    rows = (len(distributions) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 3.2 * rows),
                             squeeze=False, constrained_layout=True)
    axes_flat = axes.ravel()
    for axis, (layer, values, mean, undefined) in zip(axes_flat, distributions):
        if values.size:
            lower, upper = _automatic_plot_range(values, range_padding)
            axis.hist(values, bins=bins, range=(lower, upper), color="#3478b8",
                      alpha=0.82, edgecolor="white", linewidth=0.35)
            axis.axvline(mean, color="#c0392b", linewidth=1.4, label=f"mean={mean:.8g}")
            axis.legend(fontsize=8, frameon=False, loc="upper left")
            axis.set_xlim(lower, upper)
        else:
            axis.text(0.5, 0.5, "No valid pairs", ha="center", va="center",
                      transform=axis.transAxes)
        axis.set_title(layer, fontsize=10)
        axis.set_xlabel("Pairwise cosine", fontsize=8)
        axis.set_ylabel("Count", fontsize=8)
        axis.tick_params(labelsize=8)
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.6g"))
        axis.grid(alpha=0.18)
        if undefined:
            axis.text(0.98, 0.96, f"undefined={undefined}", ha="right", va="top",
                      transform=axis.transAxes, fontsize=7, color="#666666")
    for axis in axes_flat[len(distributions):]:
        axis.set_visible(False)
    model, dataset = metadata
    fig.suptitle(f"Pairwise layer cosine distributions | {model or 'unknown model'} / "
                 f"{dataset or 'unknown dataset'} | {model_count} models", fontsize=14)
    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="Summary CSV (default: INPUT/layer_cosine_summary.csv)")
    parser.add_argument("--pairs-file", type=Path,
                        help="Detailed CSV (default: INPUT/layer_cosine_pairs.csv)")
    parser.add_argument("--distribution-file", type=Path,
                        help="One-page layer distribution PDF (default: INPUT/layer_cosine_distribution.pdf)")
    parser.add_argument("--bins", type=int, default=60,
                        help="Histogram bins per layer (default: 60)")
    parser.add_argument("--range-padding", type=float, default=0.05,
                        help="Fractional padding around each layer's min/max (default: 0.05)")
    parser.add_argument("--pattern", default="*.model.pt")
    parser.add_argument("--key-regex")
    parser.add_argument("--exclude-regex")
    parser.add_argument("--exclude-bias", action="store_true")
    parser.add_argument("--layer-level", type=int)
    parser.add_argument("--block-size", type=int, default=65536)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if (min(args.block_size, args.threads, args.bins, args.layer_level or 1) < 1
            or args.layer_level == 0 or args.range_padding < 0
            or not np.isfinite(args.range_padding)):
        parser.error("block-size, threads and bins must be positive; range-padding must be finite and nonnegative")
    torch.set_num_threads(args.threads)
    try:
        calculate(args)
    except (ValueError, OSError, re.error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
