#!/usr/bin/env python3
"""Validation accuracy vs. symmetry-aware training coverage (Appendix I.1).

Two sweeps on x+y mod m, several seeds each (tool/run_grokking_sweep.sh):
  * random split, training partition 15..50%           (different_train_percentage/)
  * symmetry-controlled split, 50% training, same-partition ratio 100..50%
    (anti_grokking/, chessboard_random + --chessboard_transpose_ratio)

Coverage is computed from each realized split (dataset/train.txt):
    C(T) = |{(a,b) : (a,b) in T or (b,a) in T}| / m^2
which equals Table 4's p/m + (1 - 1/m)(2p - p^2) in expectation for random splits.

Accuracies come from final_correct_position.csv (the saved final model evaluated
on every operand pair). Panel (b) restricts validation to samples whose mirror
(b,a) is NOT in the training partition, i.e. pairs that commutativity alone
cannot answer.

Usage:
  python plot_coverage_vs_accuracy.py --modulus 97 [--root <dir with anti_grokking/ and different_train_percentage/>]

Also classifies each run's phase with the paper's Appendix D criteria (Table 3).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

RANDOM_COLOR = "#2a78d6"
SYMM_COLOR = "#eb6834"
BAND_COLOR = "#e6e5e0"
LABEL_OFFSETS = {
    "above_left": (-4, 5, "right", "bottom"),
    "below_right": (4, -5, "left", "top"),
    "right": (7, 0, "left", "center"),
    "left": (-7, 0, "right", "center"),
    "above": (0, 7, "center", "bottom"),
}
# Hand placement of the parameter labels where the two curves cross, per modulus (checked by eye).
LABEL_OVERRIDES = {
    97: {
        ("random", 15): "above",
        ("random", 40): "below_right",
        ("random", 45): "left",
        ("symmetry_controlled", 100): "right",
        ("symmetry_controlled", 70): "above_left",
        ("symmetry_controlled", 60): "right",
    },
    197: {
        ("random", 15): "above",
        ("random", 45): "below_right",
        ("symmetry_controlled", 100): "right",
        ("symmetry_controlled", 60): "above_left",
    },
}


def load_mask(run_dir: Path, m: int) -> np.ndarray:
    mask = np.zeros((m, m), bool)
    with open(run_dir / "dataset" / "train.txt") as handle:
        for line in handle:
            tok = line.split()
            mask[int(tok[1]), int(tok[3])] = True
    return mask


def classify_phase(train_acc: float, val_acc: float, val_at_t95: float) -> str:
    """Paper Appendix D, Table 3. t95 = first epoch with train accuracy >= 95%."""
    if train_acc < 0.95:
        return "confusion"
    if val_acc < 0.95:
        return "memorization"
    return "comprehension" if val_at_t95 >= 0.5 else "grokking"


def summarize_run(run_dir: Path, m: int) -> dict | None:
    name = run_dir.name
    match = re.fullmatch(rf"x\+y_mod_{m}_train(\d+)(?:_chessboard_random_ratio_(\d+))?_(\d+)", name)
    if match is None or not (run_dir / "final_correct_position.csv").exists():
        return None
    train = load_mask(run_dir, m)
    val = ~train
    eye = np.eye(m, dtype=bool)
    no_mirror = val & ~train.T & ~eye

    pos = pd.read_csv(run_dir / "final_correct_position.csv")
    correct = np.full((m, m), -1)
    correct[pos["lhs"].to_numpy(), pos["rhs"].to_numpy()] = pos["correct?"].to_numpy()
    if (correct < 0).any():
        raise ValueError(f"{run_dir}: final_correct_position.csv does not cover all {m * m} pairs")

    iu = np.triu_indices(m, 1)
    log = pd.read_csv(run_dir / "0.log.csv", usecols=["epoch", "training_accuracy", "validation_accuracy"])
    epochs = log["epoch"].to_numpy()
    above = log["validation_accuracy"].to_numpy() >= 0.95
    train_above = log["training_accuracy"].to_numpy() >= 0.95
    t95 = int(train_above.argmax()) if train_above.any() else None
    val_at_t95 = float(log["validation_accuracy"].iloc[t95]) if t95 is not None else np.nan
    phase = classify_phase(float(correct[train].mean()), float(correct[val].mean()), val_at_t95)
    symmetric = match.group(2) is not None
    return {
        "sweep": "symmetry_controlled" if symmetric else "random",
        "param": int(match.group(2) if symmetric else match.group(1)),
        "seed": int(match.group(3)),
        "run": name,
        "n_train": int(train.sum()),
        "same_partition_ratio": float(np.mean(train[iu] == train.T[iu])),
        "coverage": float((train | train.T).sum() / m**2),
        "train_acc": float(correct[train].mean()),
        "val_acc": float(correct[val].mean()),
        "val_frac_mirror_in_train": float((val & train.T).sum() / val.sum()),
        "val_acc_mirror_in_train": float(correct[val & train.T].mean()) if (val & train.T).any() else np.nan,
        "val_acc_no_mirror": float(correct[no_mirror].mean()),
        "epoch_val_acc_095": int(epochs[above.argmax()]) if above.any() else np.nan,
        "epoch_train_acc_095": int(epochs[t95]) if t95 is not None else np.nan,
        "val_acc_at_train_095": val_at_t95,
        "phase": phase,
    }


def plot_panel(ax, df: pd.DataFrame, column: str, title: str, band: tuple[float, float],
               label_overrides: dict | None) -> None:
    ax.axvspan(*band, color=BAND_COLOR, zorder=0, lw=0)
    specs = (
        ("random", RANDOM_COLOR, "o", "Random split (label: train %)", "{}%"),
        ("symmetry_controlled", SYMM_COLOR, "s", "Symmetry-controlled, 50% train\n(label r: same-partition ratio %)", "r={}"),
    )
    for sweep, color, marker, label, fmt in specs:
        sub = df[df["sweep"] == sweep]
        if sub.empty:
            continue
        mean = sub.groupby("param").agg(coverage=("coverage", "mean"), y=(column, "mean")).sort_values("coverage")
        ax.scatter(sub["coverage"], sub[column], s=16, marker=marker, facecolors="none",
                   edgecolors=color, linewidths=0.8, alpha=0.75, zorder=3)
        ax.plot(mean["coverage"], mean["y"], color=color, lw=1.6, marker=marker, ms=5,
                mec="white", mew=0.8, label=label, zorder=4)
        if label_overrides is not None:
            for param, row in mean.iterrows():
                # random labels above-left of the mean, symmetry labels below-right, with hand overrides
                # where the two curves cross (placement checked for mod 97)
                default = ("above_left" if sweep == "random" else "below_right")
                where = label_overrides.get((sweep, param), default)
                dx, dy, ha, va = LABEL_OFFSETS[where]
                ax.annotate(fmt.format(param), (row["coverage"], row["y"]), xytext=(dx, dy),
                            textcoords="offset points", fontsize=6.5, color="#52514e", ha=ha, va=va)
    ax.set_title(title, fontsize=10, pad=5)
    ax.set_xlabel("Realized training coverage (per run)", fontsize=9)
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlim(right=max(0.79, df["coverage"].max() + 0.04))
    ax.yaxis.set_major_locator(ticker.FixedLocator([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.tick_params(labelsize=8)
    ax.grid(axis="both", linestyle=":", linewidth=0.6, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modulus", type=int, default=97)
    parser.add_argument("--root", type=Path, default=None, help="default: <script dir>/runs_mod<modulus>")
    parser.add_argument("--out-dir", type=Path, default=script_dir)
    parser.add_argument("--band", type=float, nargs=2, default=(0.636, 0.700),
                        help="shaded coverage interval (default: the mod-97 transition from Appendix I.1)")
    args = parser.parse_args()

    root = (args.root or script_dir / f"runs_mod{args.modulus}").expanduser().resolve()
    rows = [r for group in ("anti_grokking", "different_train_percentage")
            for d in sorted((root / group).glob("*/")) if (r := summarize_run(d, args.modulus))]
    if not rows:
        raise SystemExit(f"no runs found under {root}")
    df = pd.DataFrame(rows).sort_values(["sweep", "param", "seed"], ascending=[True, False, True])

    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"coverage_vs_accuracy_mod{args.modulus}"
    df.to_csv(out / f"{stem}_runs.csv", index=False, float_format="%.4f")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True, gridspec_kw={"wspace": 0.12})
    plot_panel(axes[0], df, "val_acc", "(a) Validation accuracy", tuple(args.band),
               label_overrides=LABEL_OVERRIDES.get(args.modulus, {}))
    plot_panel(axes[1], df, "val_acc_no_mirror", "(b) Val. samples whose mirror is not in train",
               tuple(args.band), label_overrides=None)
    axes[0].set_ylabel("Final accuracy", fontsize=9)
    n_seeds = df.groupby(["sweep", "param"]).size().max()
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls="none", marker="o", mfc="none", mec="#8a8984", ms=4))
    labels.append(f"hollow: individual seeds ({n_seeds}/config)\nfilled: seed mean")
    handles.append(plt.Rectangle((0, 0), 1, 1, color=BAND_COLOR))
    labels.append(f"coverage {args.band[0]:.3f}-{args.band[1]:.3f}")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=6.5,
               frameon=False, handlelength=1.8, columnspacing=1.4)
    fig.suptitle(f"x+y mod {args.modulus}", fontsize=10, y=1.02)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    pd.set_option("display.width", 220)
    print(df.drop(columns="run").to_string(index=False, float_format="%.3f"))
    print(f"Saved {out / stem}.pdf/.png and {stem}_runs.csv")


if __name__ == "__main__":
    main()
