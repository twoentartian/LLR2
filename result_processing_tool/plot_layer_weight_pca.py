#!/usr/bin/env python3
"""Compare checkpoints with one independent 2-D weight PCA per layer.

Run from the repository root:
    python3 result_processing_tool/plot_layer_weight_pca.py tool/high_accuracy_BNN
    python3 result_processing_tool/plot_layer_weight_pca.py tool/high_accuracy_BNN \
        --key-regex '^(conv|fc)' -o result_processing_tool/bnn_conv_fc_pca

Uses the checkpoint format produced by generate_high_accuracy_model.py. Each
row/sample is one model, each column/feature is one flattened parameter. By
default, floating tensors are grouped by owning module (weight + bias), using
the same parameter heuristic as calculate_cosine_similarity.py: common BN/EMA
buffers and integer tensors are excluded. Custom floating buffers cannot be
distinguished from parameters using state_dict alone; use --key-regex or
--exclude-regex to control selection. Parameter-free modules produce no plot.

PCA centers each feature across models, without standardization, normalization,
neuron permutation alignment, or BNN binarization. BNN checkpoints store latent
full-precision weights. Different architectures/datasets must be run separately.
No MLSetup/dataset construction or GPU is needed.

Selected weights are cached on disk as float64, one checkpoint loaded at a time.
Exact PCA uses a feature-blocked, centered sample Gram matrix, requiring
O(N * block_size + N**2) working memory, O(N * P) temporary disk, and
O(N**2 * P + N**3) work per layer (N models, P layer parameters). This targets
wide layers with fewer models than parameters. It is not a streaming solution
for arbitrarily large N. The temporary cache is removed on completion/failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from result_processing_tool.model_weight_utils import (
    load_checkpoint, natural_key, select_layers,
)


def pca_2d(matrix: np.ndarray, block_size: int = 65536) -> tuple[np.ndarray, np.ndarray, int]:
    """Return N x 2 scores, explained variance ratios, and numerical rank.

    Diagonalize X_centered @ X_centered.T rather than a P x P covariance.
    The eigenvalues are squared singular values, so U * sqrt(eigenvalue)
    gives the same scores as centered SVD (up to sign/degenerate rotations).
    Rank-zero/rank-one inputs are padded with zeros; no NaN ratios are emitted.
    """
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("PCA needs at least two models and one feature")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    n, features = matrix.shape
    gram = np.zeros((n, n), dtype=np.float64)
    for start in range(0, features, block_size):
        block = np.array(matrix[:, start:start + block_size], dtype=np.float64, copy=True)
        if not np.isfinite(block).all():
            raise ValueError("PCA input contains NaN or infinity")
        # Subtract a reference first so identical inputs center to exact zero.
        block -= block[0:1].copy()
        block -= block.mean(axis=0, keepdims=True)
        gram += block @ block.T
    if not np.isfinite(gram).all():
        raise ValueError("PCA Gram matrix overflow; input magnitudes are too large")
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    vectors = vectors[:, order]
    tolerance = np.finfo(np.float64).eps * n * eigenvalues[0]
    rank = min(int(np.count_nonzero(eigenvalues > tolerance)), n - 1, features)
    scores = np.zeros((n, 2), dtype=np.float64)
    ratios = np.zeros(2, dtype=np.float64)
    total = eigenvalues.sum()
    for component in range(min(2, rank)):
        scores[:, component] = vectors[:, component] * np.sqrt(eigenvalues[component])
        # Deterministic sign: the largest-magnitude sample score is positive.
        pivot = np.argmax(np.abs(scores[:, component]))
        if scores[pivot, component] < 0:
            scores[:, component] *= -1
        ratios[component] = eigenvalues[component] / total
    return scores, ratios, rank


def plot_layer(path: Path, name: str, scores: np.ndarray, ratios: np.ndarray,
               labels: list[str], metadata: tuple, features: int, rank: int,
               annotate: bool, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.scatter(scores[:, 0], scores[:, 1], c=np.arange(len(labels)),
               cmap="turbo", s=48, edgecolors="white", linewidths=0.6)
    if annotate:
        for label, (x, y) in zip(labels, scores):
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    model, dataset = metadata
    ax.set_title(f"{name}\n{model or 'unknown model'} / {dataset or 'unknown dataset'}"
                 f" | {len(labels)} models | {features:,} parameters")
    ax.set_xlabel(f"PC1 ({ratios[0]:.2%} explained variance)")
    ax.set_ylabel(f"PC2 ({ratios[1]:.2%} explained variance)")
    ax.grid(alpha=0.2)
    ax.margins(0.15)
    if rank < 2:
        ax.text(0.02, 0.98, "Identical weights: all points overlap" if rank == 0
                else "Rank 1: PC2 is zero", transform=ax.transAxes, va="top", fontsize=9)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    source = args.input_dir.resolve()
    if not source.is_dir():
        raise ValueError(f"Input directory does not exist: {source}")
    files = sorted((p for p in source.glob(args.pattern) if p.is_file()), key=natural_key)
    if len(files) < 2:
        raise ValueError(f"Need at least two checkpoints matching {args.pattern!r} in {source}")
    first, metadata = load_checkpoint(files[0])
    layers = select_layers(first, args.key_regex, args.exclude_regex, args.layer_level, args.exclude_bias)
    if not layers:
        raise ValueError("No floating parameter tensors matched the selection")
    shapes = {key: tuple(first[key].shape) for keys in layers.values() for key in keys}
    slices = {}
    width = 0
    for name, keys in layers.items():
        size = sum(first[key].numel() for key in keys)
        slices[name] = (width, width + size)
        width += size
    del first
    output = (args.output_dir or source / "layer_weight_pca").resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Avoid mixing stale plots from different selections in a previous run.
    if any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}. Choose another -o directory.")
    labels = [p.relative_to(source).as_posix().removesuffix(".model.pt") for p in files]
    print(f"{metadata[0]} / {metadata[1]}: {len(files)} models, {len(layers)} layers", flush=True)
    print(f"Temporary weight cache: {len(files) * width * 8 / 2**30:.2f} GiB", flush=True)
    summary = {"model_type": metadata[0], "dataset_type": metadata[1],
               "model_count": len(files), "checkpoints": [str(p) for p in files],
               "selection": {"key_regex": args.key_regex, "exclude_regex": args.exclude_regex,
                             "layer_level": args.layer_level, "exclude_bias": args.exclude_bias},
               "method": "Feature-centered PCA via sample Gram matrix; no standardization",
               "layers": []}
    with tempfile.TemporaryDirectory(prefix="layer_weight_pca_", dir=args.cache_dir) as temp:
        cache = np.memmap(Path(temp) / "weights.dat", mode="w+", dtype=np.float64,
                          shape=(len(files), width))
        try:
            for row, path in enumerate(files):
                state, current_metadata = load_checkpoint(path)
                if current_metadata != metadata:
                    raise ValueError(f"{path}: model/dataset mismatch: {current_metadata} != {metadata}")
                current_layers = select_layers(state, args.key_regex, args.exclude_regex,
                                               args.layer_level, args.exclude_bias)
                current_keys = {key for keys in current_layers.values() for key in keys}
                if current_keys != set(shapes):
                    raise ValueError(f"{path}: selected parameter keys differ; "
                                     f"missing={sorted(set(shapes) - current_keys)}, "
                                     f"extra={sorted(current_keys - set(shapes))}")
                for name, keys in layers.items():
                    offset = slices[name][0]
                    for key in keys:
                        value = state[key]
                        if tuple(value.shape) != shapes[key]:
                            raise ValueError(f"{path}: shape mismatch for {key}: "
                                             f"{tuple(value.shape)} != {shapes[key]}")
                        flat = value.detach().reshape(-1).to(dtype=torch.float64).numpy()
                        if not np.isfinite(flat).all():
                            raise ValueError(f"{path}: {key} contains NaN or infinity")
                        cache[row, offset:offset + flat.size] = flat
                        offset += flat.size
                del state, value, flat
                print(f"Loaded {row + 1}/{len(files)}: {path.name}", flush=True)
            cache.flush()
            for index, (name, keys) in enumerate(layers.items(), start=1):
                start, end = slices[name]
                scores, ratios, rank = pca_2d(cache[:, start:end], args.block_size)
                slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)[:120]
                stem = f"{index:03d}_{slug}"
                plot_layer(output / f"{stem}.png", name, scores, ratios, labels, metadata,
                           end - start, rank, args.annotate, args.dpi)
                with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["model_id", "checkpoint", "PC1", "PC2"])
                    for label, path, point in zip(labels, files, scores):
                        writer.writerow([label, str(path), *point])
                summary["layers"].append({
                    "name": name, "parameter_keys": keys, "parameter_shapes": {k: shapes[k] for k in keys},
                    "feature_count": end - start, "rank": rank,
                    "explained_variance_ratio": ratios.tolist(),
                    "image": f"{stem}.png", "coordinates": f"{stem}.csv",
                })
                print(f"PCA {index}/{len(layers)}: {name}, explained variance "
                      f"{ratios.sum():.2%}", flush=True)
        finally:
            # Close the mapping before TemporaryDirectory removes it on Windows.
            cache._mmap.close()
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(f"Saved {len(layers)} plots and coordinate tables to {output}", flush=True)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Directory containing checkpoints of one model/dataset")
    parser.add_argument("-o", "--output-dir", type=Path, help="Empty output directory (default: INPUT/layer_weight_pca)")
    parser.add_argument("--pattern", default="*.model.pt", help="Glob relative to input; use **/*.model.pt for recursion")
    parser.add_argument("--key-regex", help="Include only matching parameter keys")
    parser.add_argument("--exclude-regex", help="Exclude matching parameter keys")
    parser.add_argument("--exclude-bias", action="store_true", help="Omit bias tensors")
    parser.add_argument("--layer-level", type=int, help="Group by this many module path components instead of full module name")
    parser.add_argument("--block-size", type=int, default=65536, help="Features per PCA block (default: 65536)")
    parser.add_argument("--cache-dir", type=Path, help="Existing directory for temporary disk cache")
    parser.add_argument("--annotate", action=argparse.BooleanOptionalAction, default=True, help="Label points with model IDs")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for PyTorch")
    args = parser.parse_args(argv)
    if min(args.block_size, args.dpi, args.threads) < 1 or (args.layer_level is not None and args.layer_level < 1):
        parser.error("block-size, dpi, threads and layer-level must be positive")
    torch.set_num_threads(args.threads)
    try:
        run(args)
    except (ValueError, OSError, re.error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
