"""
analyze_model_states.py
-----------------------
Analyse training/validation run folders, find key checkpoints, and compute
layer-wise cosine similarity and norms across those checkpoints.

Usage
-----
    python analyze_model_states.py <root_path>

<root_path> must contain (directly or recursively) folders that have both a
"train" and a "val" sub-folder, each containing a *.log.csv and a numbered
run sub-folder with model state .pt files.

Requirements
------------
    pip install torch pandas numpy
"""

import argparse
import sys
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from py_src import util


# ---------------------------------------------------------------------------
# Helpers – filesystem discovery
# ---------------------------------------------------------------------------

def find_experiment_folders(root: Path) -> list[Path]:
    """
    Return every directory under *root* (inclusive) that contains both a
    'train' and a 'val' sub-directory.
    """
    matches = []
    for p in sorted(root.rglob("train")):
        if p.is_dir():
            val_sibling = p.parent / "val"
            if val_sibling.is_dir():
                matches.append(p.parent)
    # De-duplicate (rglob may visit nested duplicates)
    seen = set()
    unique = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def find_log_csv(folder: Path) -> Path:
    """
    Return the first *.log.csv found directly inside *folder*.
    Tries 0.log.csv, 1.log.csv, … and falls back to any *.log.csv.
    """
    for i in range(10):
        p = folder / f"{i}.log.csv"
        if p.is_file():
            return p
    candidates = sorted(folder.glob("*.log.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No *.log.csv found in {folder}")


def find_model_stat_dir(folder: Path) -> Path:
    """
    Return the directory that holds {epoch}.model.pt files.
    Expected layout: <folder>/<run_id>/model_stat/0/
    The run_id is a single digit (0, 1, …).
    """
    for run_id_dir in sorted(folder.iterdir()):
        if not run_id_dir.is_dir():
            continue
        candidate = run_id_dir / "model_stat" / "0"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No <run_id>/model_stat/0/ directory found under {folder}"
    )


def list_epoch_checkpoints(model_stat_dir: Path) -> dict[int, Path]:
    """
    Return {epoch_int: path} for every *.model.pt file in *model_stat_dir*.
    The filename stem is the epoch number; -1 is the initial state.
    """
    ckpts = {}
    for pt in model_stat_dir.glob("*.model.pt"):
        stem = pt.stem.replace(".model", "")   # e.g. "-1" or "74000"
        try:
            ckpts[int(stem)] = pt
        except ValueError:
            pass
    return ckpts


def nearest_checkpoint(target_epoch: int, ckpts: dict[int, Path]) -> Path:
    """
    Return the checkpoint path whose epoch is closest to *target_epoch*.
    Ties broken by preferring the smaller epoch.
    """
    if not ckpts:
        raise ValueError("No checkpoints available.")
    best_epoch = min(ckpts.keys(), key=lambda e: (abs(e - target_epoch), e))
    return ckpts[best_epoch], best_epoch


# ---------------------------------------------------------------------------
# Helpers – CSV analysis
# ---------------------------------------------------------------------------

def epoch_at_fraction_of_max(series: pd.Series, fraction: float = 0.99,
                             min_max_threshold: float = 0.5) -> tuple[int, bool]:
    """
    Return (row_index, used_fallback) where row_index is the 0-based row
    position to use as the target epoch.

    If max(series) < min_max_threshold, the series never reached meaningful
    accuracy, so we fall back to the final row instead of 0.99×max.
    used_fallback=True indicates the fallback was applied.
    """
    max_val = series.max()
    if max_val < min_max_threshold:
        return int(series.index[-1]), True
    threshold = fraction * max_val
    hits = series[series >= threshold]
    if hits.empty:
        raise ValueError("Series never reaches the requested fraction of max.")
    return int(hits.index[0]), False


# ---------------------------------------------------------------------------
# Helpers – model state comparison
# ---------------------------------------------------------------------------

def load_state_dict(path: Path) -> dict:
    model_state, _, _ = util.load_model_state_file(str(path))
    return model_state


def cosine_similarity_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flattened tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return float("nan")
    return float(torch.dot(a_flat, b_flat) / (norm_a * norm_b))


def layer_norm(t: torch.Tensor) -> float:
    return float(torch.norm(t.float()))


def compare_states(states: dict[str, dict]) -> pd.DataFrame:
    """
    Given a dict of {label: state_dict} with exactly the keys
    'a', 'b', 'c', 'd', 'e' (e = initial), compute for every layer:
      - the L2 norm of each state_dict
      - the 8 specified pairwise cosine similarities:
          init_a, init_b, init_c, init_d  (e vs a/b/c/d)
          a_b, c_d, a_c, b_d

    Returns a tidy DataFrame.
    """
    PAIRS = [
        ("e", "a", "init_a"),
        ("e", "b", "init_b"),
        ("e", "c", "init_c"),
        ("e", "d", "init_d"),
        ("a", "b", "a_b"),
        ("c", "d", "c_d"),
        ("a", "c", "a_c"),
        ("b", "d", "b_d"),
    ]

    # Collect all parameter keys (union across all state dicts)
    all_keys = []
    seen_keys = set()
    for sd in states.values():
        for k in sd:
            if k not in seen_keys:
                all_keys.append(k)
                seen_keys.add(k)

    rows = []
    for key in all_keys:
        row = {"layer": key}

        # Norms – one column per checkpoint label (a/b/c/d/e)
        for label, sd in states.items():
            row[f"norm_{label}"] = layer_norm(sd[key]) if key in sd else float("nan")

        # Cosine similarities – only the 8 named pairs
        for la, lb, col_name in PAIRS:
            if key in states[la] and key in states[lb]:
                row[f"cos_sim_{col_name}"] = cosine_similarity_1d(
                    states[la][key], states[lb][key]
                )
            else:
                row[f"cos_sim_{col_name}"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core: process one experiment folder
# ---------------------------------------------------------------------------

def process_experiment(exp_folder: Path) -> bool:
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_folder}")
    print(f"{'='*70}")

    train_folder = exp_folder / "train"
    val_folder   = exp_folder / "val"

    # -----------------------------------------------------------------------
    # 1. Load CSV logs
    # -----------------------------------------------------------------------
    train_csv_path = find_log_csv(train_folder)
    val_csv_path   = find_log_csv(val_folder)
    print(f"  Train CSV : {train_csv_path.name}")
    print(f"  Val   CSV : {val_csv_path.name}")

    train_df = pd.read_csv(train_csv_path)
    val_df   = pd.read_csv(val_csv_path)

    # -----------------------------------------------------------------------
    # 2. Find target epochs
    # -----------------------------------------------------------------------
    def describe_target(label, df, col, row_idx, fallback):
        max_val = df[col].max()
        actual_val = df.iloc[row_idx][col]
        note = " [FALLBACK: max < 0.5, using final epoch]" if fallback else ""
        print(f"    {label}: epoch {int(df.iloc[row_idx]['epoch'])}  "
              f"({col}={actual_val:.4f}, max={max_val:.4f}){note}")

    # train folder: epoch when train_acc first hits 0.99 * max
    train_epoch_at_99_train_acc, fb_tt = epoch_at_fraction_of_max(
        train_df["training_accuracy"])
    # train folder: epoch when val_acc first hits 0.99 * max
    train_epoch_at_99_val_acc,   fb_tv = epoch_at_fraction_of_max(
        train_df["validation_accuracy"])
    # val folder: epoch when train_acc first hits 0.99 * max
    val_epoch_at_99_train_acc,   fb_vt = epoch_at_fraction_of_max(
        val_df["training_accuracy"])
    # val folder: epoch when val_acc first hits 0.99 * max
    val_epoch_at_99_val_acc,     fb_vv = epoch_at_fraction_of_max(
        val_df["validation_accuracy"])

    # Convert row index → actual epoch number from the 'epoch' column
    def row_to_epoch(df, row_idx):
        return int(df.iloc[row_idx]["epoch"])

    train_target_train = row_to_epoch(train_df, train_epoch_at_99_train_acc)
    train_target_val   = row_to_epoch(train_df, train_epoch_at_99_val_acc)
    val_target_train   = row_to_epoch(val_df,   val_epoch_at_99_train_acc)
    val_target_val     = row_to_epoch(val_df,   val_epoch_at_99_val_acc)

    print(f"\n  Epoch targets:")
    describe_target("train folder – train_acc", train_df, "training_accuracy",
                    train_epoch_at_99_train_acc, fb_tt)
    describe_target("train folder – val_acc  ", train_df, "validation_accuracy",
                    train_epoch_at_99_val_acc, fb_tv)
    describe_target("val   folder – train_acc", val_df, "training_accuracy",
                    val_epoch_at_99_train_acc, fb_vt)
    describe_target("val   folder – val_acc  ", val_df, "validation_accuracy",
                    val_epoch_at_99_val_acc, fb_vv)

    # -----------------------------------------------------------------------
    # 3. Locate model state directories and enumerate checkpoints
    # -----------------------------------------------------------------------
    train_model_dir = find_model_stat_dir(train_folder)
    val_model_dir   = find_model_stat_dir(val_folder)
    print(f"\n  Train model state dir: {train_model_dir}")
    print(f"  Val   model state dir: {val_model_dir}")

    train_ckpts = list_epoch_checkpoints(train_model_dir)
    val_ckpts   = list_epoch_checkpoints(val_model_dir)
    print(f"  Available checkpoints – train: {sorted(train_ckpts)}")
    print(f"  Available checkpoints – val  : {sorted(val_ckpts)}")

    # -----------------------------------------------------------------------
    # 4. Load initial state (-1) and verify they are identical
    # -----------------------------------------------------------------------
    if -1 not in train_ckpts:
        raise FileNotFoundError("Initial checkpoint (-1.model.pt) not found in train folder.")
    if -1 not in val_ckpts:
        raise FileNotFoundError("Initial checkpoint (-1.model.pt) not found in val folder.")

    init_train_sd = load_state_dict(train_ckpts[-1])
    init_val_sd   = load_state_dict(val_ckpts[-1])

    # Verify identical initial states
    train_keys = set(init_train_sd.keys())
    val_keys   = set(init_val_sd.keys())
    if train_keys != val_keys:
        print(f"  WARNING: Initial state dicts have different keys! "
              f"Train only: {train_keys - val_keys}, Val only: {val_keys - train_keys}")
        initial_states_match = False
    else:
        all_match = all(
            torch.equal(init_train_sd[k].float(), init_val_sd[k].float())
            for k in train_keys
        )
        initial_states_match = all_match
    print(f"\n  Initial model states match: {initial_states_match}")
    if not initial_states_match:
        print("  WARNING: train and val initial states differ – results may not "
              "be directly comparable.")

    # -----------------------------------------------------------------------
    # 5. Load the four target checkpoints (nearest available)
    # -----------------------------------------------------------------------
    def load_nearest(ckpts, target, label):
        path, actual = nearest_checkpoint(target, ckpts)
        print(f"    {label}: target epoch {target} → nearest checkpoint epoch {actual} ({path.name})")
        return load_state_dict(path), actual

    print("\n  Loading target checkpoints:")
    sd_a, epoch_a = load_nearest(train_ckpts, train_target_train,
                                 "(a) train-folder, 0.99×max train_acc")
    sd_b, epoch_b = load_nearest(train_ckpts, train_target_val,
                                 "(b) train-folder, 0.99×max val_acc  ")
    sd_c, epoch_c = load_nearest(val_ckpts,   val_target_train,
                                 "(c) val-folder,   0.99×max train_acc")
    sd_d, epoch_d = load_nearest(val_ckpts,   val_target_val,
                                 "(d) val-folder,   0.99×max val_acc  ")
    sd_e = init_train_sd   # (e) initial state (train = val, verified above)

    states = {
        "a": sd_a,   # train folder, 0.99×max train_acc  (ep{epoch_a})
        "b": sd_b,   # train folder, 0.99×max val_acc    (ep{epoch_b})
        "c": sd_c,   # val   folder, 0.99×max train_acc  (ep{epoch_c})
        "d": sd_d,   # val   folder, 0.99×max val_acc    (ep{epoch_d})
        "e": sd_e,   # initial state (epoch -1)
    }

    # -----------------------------------------------------------------------
    # 6. Layer-wise comparison
    # -----------------------------------------------------------------------
    print("\n  Computing layer-wise cosine similarities and norms …")
    comparison_df = compare_states(states)

    # -----------------------------------------------------------------------
    # 7. Save outputs
    # -----------------------------------------------------------------------
    out_dir = exp_folder
    # Save main comparison table
    comparison_csv = out_dir / "model_state_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\n  Saved comparison table → {comparison_csv}")

    # Save a human-readable summary
    summary_path = out_dir / "model_state_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Experiment: {exp_folder}\n")
        f.write("="*70 + "\n\n")

        f.write("INITIAL MODEL STATES MATCH: " + str(initial_states_match) + "\n\n")

        f.write("EPOCH TARGETS\n")
        f.write("-"*40 + "\n")
        def fb_note(fallback): return "  [FALLBACK: max < 0.5, used final epoch]" if fallback else ""
        f.write(f"  (a) train folder – train_acc : target epoch {train_target_train} "
                f"→ nearest ckpt epoch {epoch_a}{fb_note(fb_tt)}\n")
        f.write(f"  (b) train folder – val_acc   : target epoch {train_target_val} "
                f"→ nearest ckpt epoch {epoch_b}{fb_note(fb_tv)}\n")
        f.write(f"  (c) val   folder – train_acc : target epoch {val_target_train} "
                f"→ nearest ckpt epoch {epoch_c}{fb_note(fb_vt)}\n")
        f.write(f"  (d) val   folder – val_acc   : target epoch {val_target_val} "
                f"→ nearest ckpt epoch {epoch_d}{fb_note(fb_vv)}\n")
        f.write(f"  (e) initial state (epoch -1)\n\n")

        f.write("CHECKPOINT LABELS\n")
        f.write("-"*40 + "\n")
        f.write(f"  a = train folder, train_acc target → nearest ckpt epoch {epoch_a}{fb_note(fb_tt)}\n")
        f.write(f"  b = train folder, val_acc target   → nearest ckpt epoch {epoch_b}{fb_note(fb_tv)}\n")
        f.write(f"  c = val   folder, train_acc target → nearest ckpt epoch {epoch_c}{fb_note(fb_vt)}\n")
        f.write(f"  d = val   folder, val_acc target   → nearest ckpt epoch {epoch_d}{fb_note(fb_vv)}\n")
        f.write(f"  e = initial state (epoch -1)\n")
        f.write("\n")

        f.write("LAYER-WISE NORMS\n")
        f.write("-"*40 + "\n")
        norm_cols = [c for c in comparison_df.columns if c.startswith("norm_")]
        f.write(comparison_df[["layer"] + norm_cols].to_string(index=False))
        f.write("\n\n")

        cos_cols = [c for c in comparison_df.columns if c.startswith("cos_sim_")]
        f.write("LAYER-WISE COSINE SIMILARITIES\n")
        f.write("-"*40 + "\n")
        f.write(comparison_df[["layer"] + cos_cols].to_string(index=False))
        f.write("\n")

    print(f"  Saved summary          → {summary_path}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyse model checkpoints: find key epochs (0.99×max accuracy), "
            "verify initial model states, then compute layer-wise cosine "
            "similarity and norms across selected checkpoints."
        )
    )
    parser.add_argument(
        "root_path",
        help="Root directory to scan for experiment folders "
             "(folders containing 'train' and 'val' sub-folders).",
    )
    args = parser.parse_args()

    root = Path(args.root_path)
    if not root.is_dir():
        sys.exit(f"ERROR: not a directory: {root}")

    experiments = find_experiment_folders(root)
    if not experiments:
        sys.exit(
            f"No experiment folders (containing 'train' and 'val') "
            f"found under {root}"
        )

    print(f"Found {len(experiments)} experiment folder(s) under {root}")

    n_ok = n_fail = 0
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {exp}")
        try:
            process_experiment(exp)
            n_ok += 1
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            n_fail += 1

    print(f"\nDone. {n_ok} succeeded, {n_fail} failed.")


if __name__ == "__main__":
    main()