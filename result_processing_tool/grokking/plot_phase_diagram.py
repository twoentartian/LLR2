"""
Plot the grokking phase diagram from results produced by
generate_grokking_phase_diagram.py.

This script is fully decoupled from training: it only reads the
per-cell meta.json / 00.log.csv files and the top-level grid_spec.json,
then produces phase_diagram.png in the same output folder.

Usage
-----
python plot_grokking_phase_diagram.py -o phase_diagram_output

Optional overrides
------------------
# Change classification thresholds
python plot_grokking_phase_diagram.py -o phase_diagram_output \
    --high_acc 0.99 --grokking_val_threshold 0.5

# Save to a custom path
python plot_grokking_phase_diagram.py -o phase_diagram_output \
    --save_path my_figure.png
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("plot_grokking_phase_diagram")

# ---------------------------------------------------------------------------
# Phase colours (paper-inspired)
# ---------------------------------------------------------------------------
PHASE_COLORS = {
    "comprehension": "#2196F3",   # blue
    "grokking":      "#4CAF50",   # green
    "memorization":  "#FF9800",   # orange
    "confusion":     "#F44336",   # red
    "unknown":       "#9E9E9E",   # grey
}

PHASE_DISPLAY = {
    "comprehension": "Comp.",
    "grokking":      "Grok.",
    "memorization":  "Memo.",
    "confusion":     "Conf.",
    "unknown":       "?",
}


# ---------------------------------------------------------------------------
# Phase classification
# ---------------------------------------------------------------------------

def classify_phase(log_csv_path: str,
                   high_acc: float,
                   grokking_val_threshold: float) -> str:
    """
    Read a per-cell training log CSV and return the learning phase label.

    CSV columns (written by train_grokking):
        epoch, training_loss, training_accuracy,
        validation_loss, validation_accuracy, lrs

    Parameters
    ----------
    high_acc : float
        Minimum final accuracy to be considered "high" (e.g. 0.95).
    grokking_val_threshold : float
        When train accuracy first crosses `high_acc`, val accuracy must be
        below this value (absolute, not relative) to be labelled grokking
        rather than comprehension.  E.g. 0.5 means val was still below 50 %
        when train already generalised.
    """
    try:
        df = pd.read_csv(log_csv_path)
    except Exception as exc:
        logger.warning(f"Could not read {log_csv_path}: {exc}")
        return "unknown"

    train_acc = df["training_accuracy"].values
    val_acc   = df["validation_accuracy"].values

    final_train = float(train_acc[-1])
    final_val   = float(val_acc[-1])

    train_high = final_train >= high_acc
    val_high   = final_val   >= high_acc

    if not train_high and not val_high:
        return "confusion"

    if train_high and not val_high:
        return "memorization"

    if train_high and val_high:
        # Find the first epoch where train crosses the threshold
        crossed = np.argmax(train_acc >= high_acc)
        val_at_cross = float(val_acc[crossed])
        if val_at_cross >= grokking_val_threshold:
            return "comprehension"   # val was already reasonable when train peaked
        else:
            return "grokking"        # val only caught up much later

    # train low, val high — rare edge case
    return "comprehension"


# ---------------------------------------------------------------------------
# Discovery: find all cell directories in an output folder
# ---------------------------------------------------------------------------

def discover_cells(output_folder: str):
    """
    Walk output_folder and return a list of dicts, one per cell:
        {"lr": float, "wd": float, "log_csv": str}

    The lr/wd are read from meta.json inside each cell directory.
    Directories without meta.json are silently skipped.
    """
    cells = []
    for entry in sorted(Path(output_folder).iterdir()):
        if not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        log_path  = entry / "00.log.csv"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if "learning_rate" not in meta or "weight_decay" not in meta:
            logger.warning(f"meta.json in {entry} missing lr/wd keys, skipping")
            continue
        cells.append({
            "lr":      float(meta["learning_rate"]),
            "wd":      float(meta["weight_decay"]),
            "log_csv": str(log_path) if log_path.exists() else None,
        })
    return cells


def load_grid_spec(output_folder: str):
    """
    Load the ordered lr/wd axes from grid_spec.json if present.
    Falls back to sorted unique values discovered from cell meta.json files.
    """
    spec_path = Path(output_folder) / "grid_spec.json"
    if spec_path.exists():
        with open(spec_path) as f:
            spec = json.load(f)
        return spec["learning_rates"], spec["weight_decays"]
    return None, None   # caller will derive axes from discovered cells


# ---------------------------------------------------------------------------
# Core plotting
# ---------------------------------------------------------------------------

def plot_phase_diagram(output_folder: str,
                       high_acc: float,
                       grokking_val_threshold: float,
                       save_path: str = None,
                       annotate: bool = True):
    """
    Build and save the phase diagram.

    Parameters
    ----------
    output_folder : str
        Root directory produced by generate_grokking_phase_diagram.py.
    high_acc : float
        Accuracy threshold for "high" (used in classify_phase).
    grokking_val_threshold : float
        Val-acc threshold at train-crossing epoch to distinguish
        grokking from comprehension.
    save_path : str or None
        Where to write the PNG.  Defaults to <output_folder>/phase_diagram.png.
    annotate : bool
        Whether to print the phase label in each cell.
    """
    # ---- discover cells ----
    cells = discover_cells(output_folder)
    if not cells:
        logger.error(f"No cell directories with meta.json found in {output_folder}")
        sys.exit(1)
    logger.info(f"Found {len(cells)} cell directories")

    # ---- build axis arrays ----
    grid_lrs, grid_wds = load_grid_spec(output_folder)
    if grid_lrs is None:
        grid_lrs = sorted(set(c["lr"] for c in cells))
        grid_wds = sorted(set(c["wd"] for c in cells))
        logger.info("grid_spec.json not found — axes inferred from cell metadata")

    n_lr = len(grid_lrs)
    n_wd = len(grid_wds)

    # Map value → index for fast lookup
    lr_idx = {lr: i for i, lr in enumerate(grid_lrs)}
    wd_idx = {wd: j for j, wd in enumerate(grid_wds)}

    # ---- classify each cell ----
    phase_matrix = np.full((n_lr, n_wd), fill_value="unknown", dtype=object)

    for cell in cells:
        lr, wd = cell["lr"], cell["wd"]
        # Find closest grid point (guards against floating-point mismatch)
        i = min(range(n_lr), key=lambda k: abs(grid_lrs[k] - lr))
        j = min(range(n_wd), key=lambda k: abs(grid_wds[k] - wd))

        if cell["log_csv"] is None:
            logger.warning(f"lr={lr:.4e} wd={wd:.4e}: log CSV missing, marking unknown")
            continue
        phase_matrix[i, j] = classify_phase(
            cell["log_csv"], high_acc, grokking_val_threshold)

    # ---- save classification summary ----
    summary = {}
    for i, lr in enumerate(grid_lrs):
        for j, wd in enumerate(grid_wds):
            summary[f"lr={lr:.4e},wd={wd:.4e}"] = phase_matrix[i, j]
    summary_path = os.path.join(output_folder, "phase_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Classification summary saved to {summary_path}")

    # ---- build RGBA colour matrix ----
    color_matrix = np.zeros((n_lr, n_wd, 4))
    for i in range(n_lr):
        for j in range(n_wd):
            phase = phase_matrix[i, j]
            color_matrix[i, j] = matplotlib.colors.to_rgba(
                PHASE_COLORS.get(phase, PHASE_COLORS["unknown"]))

    # ---- figure ----
    cell_size = 0.55   # inches per cell
    fig_w = max(8,  n_wd * cell_size + 3)
    fig_h = max(6,  n_lr * cell_size + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(color_matrix, origin="lower", aspect="auto",
              extent=[-0.5, n_wd - 0.5, -0.5, n_lr - 0.5])

    # Cell annotations
    if annotate:
        font_size = max(4, min(8, int(cell_size * 13)))
        for i in range(n_lr):
            for j in range(n_wd):
                phase = phase_matrix[i, j]
                ax.text(j, i, PHASE_DISPLAY[phase],
                        ha="center", va="center",
                        fontsize=font_size, color="white", fontweight="bold")

    # ---- axis ticks: use log scale labels ----
    # Weight decay x-axis
    wd_labels = []
    for wd in grid_wds:
        if wd == 0.0:
            wd_labels.append("0")
        else:
            exp = int(round(np.log10(wd)))
            mantissa = wd / (10 ** exp)
            if abs(mantissa - 1.0) < 0.05:
                wd_labels.append(f"$10^{{{exp}}}$")
            else:
                wd_labels.append(f"{wd:.1e}")

    lr_labels = []
    for lr in grid_lrs:
        exp = int(round(np.log10(lr)))
        mantissa = lr / (10 ** exp)
        if abs(mantissa - 1.0) < 0.05:
            lr_labels.append(f"$10^{{{exp}}}$")
        else:
            lr_labels.append(f"{lr:.1e}")

    # Show a subset of ticks when the grid is large (avoid crowding)
    max_ticks = 10
    def sparse_ticks(n):
        if n <= max_ticks:
            return list(range(n))
        step = max(1, n // max_ticks)
        return list(range(0, n, step))

    wd_tick_pos = sparse_ticks(n_wd)
    lr_tick_pos = sparse_ticks(n_lr)

    ax.set_xticks(wd_tick_pos)
    ax.set_xticklabels([wd_labels[j] for j in wd_tick_pos],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(lr_tick_pos)
    ax.set_yticklabels([lr_labels[i] for i in lr_tick_pos], fontsize=8)

    ax.set_xlabel("Weight Decay", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title(
        f"Grokking Phase Diagram  ({n_lr}×{n_wd} grid)\n"
        f"(cf. Figure 6, Liu et al. 2022)",
        fontsize=12)

    # ---- legend ----
    legend_handles = [
        mpatches.Patch(color=PHASE_COLORS["comprehension"], label="Comprehension"),
        mpatches.Patch(color=PHASE_COLORS["grokking"],      label="Grokking"),
        mpatches.Patch(color=PHASE_COLORS["memorization"],  label="Memorization"),
        mpatches.Patch(color=PHASE_COLORS["confusion"],     label="Confusion"),
        mpatches.Patch(color=PHASE_COLORS["unknown"],       label="Unknown / missing"),
    ]
    ax.legend(handles=legend_handles,
              loc="upper left", bbox_to_anchor=(1.01, 1),
              borderaxespad=0, fontsize=9, framealpha=0.9)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(output_folder, "phase_diagram")
    fig.savefig(f"{save_path}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Phase diagram saved to {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the grokking phase diagram from sweep results.")

    parser.add_argument("output_folder_name",
                        help="Output folder produced by generate_grokking_phase_diagram.py")
    parser.add_argument("--save_path", default=None,
                        help="Custom path for the output PNG "
                             "(default: <output_folder>/phase_diagram.png)")
    parser.add_argument("--high_acc", type=float, default=0.95,
                        help="Accuracy threshold to be called 'high' (default: 0.95)")
    parser.add_argument("--grokking_val_threshold", type=float, default=0.5,
                        help="Val-acc value below which a cell is labelled 'grokking' "
                             "rather than 'comprehension' at the epoch when train first "
                             "exceeds --high_acc (default: 0.5)")
    parser.add_argument("--no_annotate", action="store_true",
                        help="Suppress per-cell text annotations")

    args = parser.parse_args()

    output_folder = os.path.join(os.curdir, args.output_folder_name)
    if not os.path.isdir(output_folder):
        logger.error(f"Folder not found: {output_folder}")
        sys.exit(1)

    plot_phase_diagram(
        output_folder=output_folder,
        high_acc=args.high_acc,
        grokking_val_threshold=args.grokking_val_threshold,
        save_path=args.save_path,
        annotate=not args.no_annotate,
    )
