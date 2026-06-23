"""
plot_training_logs.py
---------------------
Find all *.log.csv files under a root folder and plot training/validation
loss and accuracy curves for each one.

Usage
-----
    python plot_training_logs.py <root_folder>

For every *.log.csv found, a matching *.log.pdf is saved in the same directory.

Columns expected in each CSV:
    epoch, training_loss, training_accuracy, validation_loss, validation_accuracy
    (extra columns such as 'lrs' are ignored)

Layout
------
    Left  y-axis : accuracy  (training + validation)
    Right y-axis : loss      (training + validation)
    x-axis       : epoch
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_log_files(root: Path) -> list[Path]:
    """Return all *.log.csv files found anywhere under root (sorted)."""
    return sorted(root.rglob("*.log.csv"))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_log(csv_path: Path, max_epoch: Optional[int], override_existing=False) -> bool:
    """
    Read one *.log.csv and save a twin-axis figure next to it as *.log.pdf.
    Returns True on success.
    """
    out_pdf_path = csv_path.with_suffix("").with_suffix(".log.pdf")
    out_png_path = csv_path.with_suffix("").with_suffix(".log.png")
    out_pdf_log_path = csv_path.with_suffix("").with_suffix(".log.log_loss.pdf")
    out_png_log_path = csv_path.with_suffix("").with_suffix(".log.log_loss.png")
    if any([os.path.exists(p) for p in [out_pdf_path, out_png_path, out_pdf_log_path, out_png_log_path]]):
        if not override_existing:
            print(f"  Already exist -> {out_pdf_path}")
            return True

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"  ERROR reading CSV: {exc}")
        return False

    required = {"epoch", "training_loss", "training_accuracy",
                "validation_loss", "validation_accuracy"}
    missing = required - set(df.columns)
    if missing:
        print(f"  ERROR: missing column(s): {', '.join(sorted(missing))}")
        return False

    epoch        = df["epoch"]
    train_acc    = df["training_accuracy"]
    val_acc      = df["validation_accuracy"]
    train_loss   = df["training_loss"]
    val_loss     = df["validation_loss"]

    # ------------------------------------------------------------------ #
    # Figure with twin y-axes                                             #
    # ------------------------------------------------------------------ #
    fig, ax_acc = plt.subplots(figsize=(10, 5))
    ax_loss = ax_acc.twinx()

    # --- Accuracy (left axis) ---
    l1, = ax_acc.plot(epoch, train_acc, color="#2171b5", linewidth=1.2,
                      label="Train accuracy")
    l2, = ax_acc.plot(epoch, val_acc,   color="#6baed6", linewidth=1.2,
                      linestyle="--", label="Val accuracy")

    # --- Loss (right axis) ---
    l3, = ax_loss.plot(epoch, train_loss, color="#cb181d", linewidth=1.0,
                       alpha=0.85, label="Train loss")
    l4, = ax_loss.plot(epoch, val_loss,   color="#fb6a4a", linewidth=1.0,
                       alpha=0.85, linestyle="--", label="Val loss")

    # --- Labels ---
    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Accuracy", fontsize=11, color="#2171b5")
    ax_loss.set_ylabel("Loss",    fontsize=11, color="#cb181d")

    ax_acc.tick_params(axis="y", labelcolor="#2171b5")
    ax_loss.tick_params(axis="y", labelcolor="#cb181d")

    # --- Title: use the stem of the filename (strip .log.csv) ---
    stem = csv_path.name
    for suffix in (".log.csv",):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    ax_acc.set_title(f"{stem}  —  {csv_path.parent.name}", fontsize=12,
                     fontweight="bold")

    # --- Combined legend (all four lines) ---
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax_acc.legend(lines, labels, loc="center right", fontsize=9,
                  framealpha=0.85)

    # --- Grid on accuracy axis only (subtle) ---
    ax_acc.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_acc.grid(axis="y", which="major", linestyle=":", alpha=0.4)
    ax_acc.grid(axis="x", which="major", linestyle=":", alpha=0.3)

    ax_acc.set_xlim(left=0)
    ax_loss.set_xlim(left=0)
    if max_epoch is not None:
        ax_acc.set_xlim(right=int(max_epoch))
        ax_loss.set_xlim(right=int(max_epoch))

    fig.tight_layout()

    fig.savefig(out_pdf_path, bbox_inches="tight")
    fig.savefig(out_png_path, dpi=300, bbox_inches="tight")
    ax_loss.set_yscale("log")
    fig.savefig(out_pdf_log_path, bbox_inches="tight")
    fig.savefig(out_png_log_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved -> {out_pdf_path}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(root_folder, max_epoch=None, override_existing=False):
    root = Path(root_folder)
    if not root.is_dir():
        sys.exit(f"ERROR: not a directory: {root}")

    log_files = find_log_files(root)
    if not log_files:
        sys.exit(f"No *.log.csv files found under {root}")

    print(f"Found {len(log_files)} *.log.csv file(s) under {root}\n")

    n_ok = n_fail = 0
    for i, csv_path in enumerate(log_files, 1):
        print(f"[{i}/{len(log_files)}] {csv_path}")
        ok = plot_log(csv_path, max_epoch, override_existing=override_existing)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
        print()

    print(f"Done.  {n_ok} succeeded,  {n_fail} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot training/validation loss and accuracy from *.log.csv files. "
            "Recursively searches the given root folder."
        )
    )
    parser.add_argument("folder", help="Root folder to search for *.log.csv files")
    parser.add_argument("-e", "--epoch", help="maximum epoch")
    args = parser.parse_args()
    main(args.folder, max_epoch=args.epoch)
