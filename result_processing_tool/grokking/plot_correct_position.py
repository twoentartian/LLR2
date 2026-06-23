"""
plot_correct_position.py
------------------------
Find all folders containing "final_correct_position.csv", locate the
sibling "modulus*" subfolder with train.txt / val.txt / tokenizer.txt,
and plot a correctness grid.

Usage
-----
    python plot_correct_position.py <root_folder>

Colour scheme per cell (lhs=row, rhs=col)
------------------------------------------
    Train + correct   : white
    Train + incorrect : black
    Val   + correct   : green
    Val   + incorrect : red
    Neither (missing) : grey

Output: "correct_position_plot.pdf" saved next to each
        "final_correct_position.csv".
"""

import argparse
import sys
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Vocabulary helpers  (reused from visualizing_arithemetic_dataset.py)
# ---------------------------------------------------------------------------

def load_tokens(tokenizer_path: Path) -> list[str]:
    return tokenizer_path.read_text().strip().split("\n")


def extract_operators(tokens: list[str]) -> list[str]:
    skip = {"<|eos|>", "="}

    def is_operand(t):
        if t in skip:
            return True
        if re.fullmatch(r"\d+", t):
            return True
        if re.fullmatch(r"[0-4]{5}", t):
            return True
        return False

    ops = [t for t in tokens if not is_operand(t)]
    ops.sort(key=len, reverse=True)
    return ops


def build_operand_index(tokens: list[str]) -> dict[str, int]:
    skip = {"<|eos|>", "="}
    ops = set(extract_operators(tokens))
    operand_tokens = [t for t in tokens if t not in skip and t not in ops]
    return {t: i for i, t in enumerate(operand_tokens)}


# ---------------------------------------------------------------------------
# Equation parsing  ->  set of (a_idx, b_idx) for train and val
# ---------------------------------------------------------------------------

def parse_equation(eq: str, operators: list[str],
                   operand_index: dict[str, int]):
    eq = eq.strip()
    eq = re.sub(r"<\|eos\|>", "", eq).strip()
    if not eq:
        return None
    parts = eq.split(" = ")
    if len(parts) < 2:
        return None
    lhs = parts[0].strip()
    op_found = None
    for op in operators:
        if f" {op} " in lhs:
            op_found = op
            break
    if op_found is None:
        return None
    halves = lhs.split(f" {op_found} ", maxsplit=1)
    if len(halves) != 2:
        return None
    a_idx = operand_index.get(halves[0].strip())
    b_idx = operand_index.get(halves[1].strip())
    if a_idx is None or b_idx is None:
        return None
    return a_idx, b_idx


def load_coord_set(txt_path: Path, operators: list[str],
                   operand_index: dict[str, int]) -> set[tuple[int, int]]:
    coords = set()
    for line in txt_path.read_text().splitlines():
        r = parse_equation(line, operators, operand_index)
        if r is not None:
            coords.add(r)
    return coords


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_csv_folders(root: Path) -> list[Path]:
    """
    Return all directories that contain final_correct_position.csv,
    found by searching for the file with rglob (fast anchor search).
    """
    return sorted({p.parent for p in root.rglob("final_correct_position.csv")})


def find_modulus_folder(csv_folder: Path) -> Path | None:
    """
    Return the first subfolder of csv_folder whose name starts with 'modulus'
    and that contains train.txt, val.txt, tokenizer.txt.
    """
    required = {"train.txt", "val.txt", "tokenizer.txt"}
    for sub in sorted(csv_folder.iterdir()):
        if sub.is_dir() and sub.name.startswith("modulus"):
            if required.issubset({f.name for f in sub.iterdir() if f.is_file()}):
                return sub
    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_correct_position(csv_path: Path, dataset_folder: Path,
                          out_path: Path, override_existing=False) -> bool:
    if any([os.path.exists(p) for p in [out_path]]):
        if not override_existing:
            print(f"  Already exist -> {out_path}")
            return True

    # --- Load correctness data ---
    df = pd.read_csv(csv_path)
    required_cols = {"lhs", "rhs", "correct?"}
    if not required_cols.issubset(df.columns):
        print(f"  ERROR: missing columns {required_cols - set(df.columns)}")
        return False

    # --- Load vocabulary & train/val membership ---
    tokenizer_path = dataset_folder / "tokenizer.txt"
    tokens         = load_tokens(tokenizer_path)
    operators      = extract_operators(tokens)
    operand_index  = build_operand_index(tokens)

    train_coords = load_coord_set(dataset_folder / "train.txt", operators, operand_index)
    val_coords   = load_coord_set(dataset_folder / "val.txt",   operators, operand_index)

    # --- Infer grid size ---
    n = max(df["lhs"].max(), df["rhs"].max()) + 1

    # --- Build grid ---
    # Values:  0 = train+correct   1 = train+incorrect
    #          2 = val+correct     3 = val+incorrect
    #         -1 = missing / unassigned
    grid = np.full((n, n), fill_value=-1, dtype=np.int8)

    for _, row in df.iterrows():
        a, b, correct = int(row["lhs"]), int(row["rhs"]), bool(row["correct?"])
        if not (0 <= a < n and 0 <= b < n):
            continue
        in_train = (a, b) in train_coords
        in_val   = (a, b) in val_coords
        if in_train:
            grid[a, b] = 0 if correct else 1
        elif in_val:
            grid[a, b] = 2 if correct else 3
        # else: neither (missing / overlap edge case) → stays -1

    # --- Colour map ---
    #  0 train+correct   → white
    #  1 train+incorrect → black
    #  2 val+correct     → green
    #  3 val+incorrect   → red
    # -1 missing         → mid-grey
    cmap = {
        -1: (0.60, 0.60, 0.60, 1.0),   # grey
         0: (1.00, 1.00, 1.00, 1.0),   # white
         1: (0.00, 0.00, 0.00, 1.0),   # black
         2: (0.18, 0.65, 0.18, 1.0),   # green
         3: (0.85, 0.15, 0.15, 1.0),   # red
    }
    img = np.zeros((n, n, 4))
    for v, color in cmap.items():
        img[grid == v] = color

    # --- Stats ---
    n_tc = int((grid == 0).sum())
    n_ti = int((grid == 1).sum())
    n_vc = int((grid == 2).sum())
    n_vi = int((grid == 3).sum())
    n_miss = int((grid == -1).sum())
    total  = n * n
    train_total = n_tc + n_ti
    val_total   = n_vc + n_vi

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [3, 1]})

    ax = axes[0]
    ax.imshow(img, origin="upper", aspect="equal",
              extent=[-0.5, n - 0.5, n - 0.5, -0.5])
    ax.set_xlabel("rhs  (column operand)", fontsize=11)
    ax.set_ylabel("lhs  (row operand)",    fontsize=11)
    ax.set_title(
        f"Correctness per position  —  {csv_path.parent.name}\n"
        f"(dataset: {dataset_folder.name})",
        fontsize=12, fontweight="bold",
    )
    step = max(1, n // 10)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # legend
    patches = [
        mpatches.Patch(facecolor=cmap[0][:3], edgecolor="lightgrey",
                       label=f"Train correct   ({n_tc:,})"),
        mpatches.Patch(facecolor=cmap[1][:3], edgecolor="lightgrey",
                       label=f"Train incorrect ({n_ti:,})"),
        mpatches.Patch(facecolor=cmap[2][:3],
                       label=f"Val correct     ({n_vc:,})"),
        mpatches.Patch(facecolor=cmap[3][:3],
                       label=f"Val incorrect   ({n_vi:,})"),
    ]
    if n_miss:
        patches.append(mpatches.Patch(facecolor=cmap[-1][:3],
                                      label=f"Missing         ({n_miss:,})"))
    fig.legend(handles=patches, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=len(patches),
               fontsize=9, framealpha=0.85)

    # --- Stats panel ---
    ax2 = axes[1]
    ax2.axis("off")

    def pct(x, tot):
        return f"{100*x/tot:.1f}%" if tot > 0 else "n/a"

    stats = [
        ("Grid",            f"{n} x {n} = {total:,}"),
        ("Train correct",   f"{n_tc:,}  /  {train_total:,}  ({pct(n_tc, train_total)})"),
        ("Train incorrect", f"{n_ti:,}  /  {train_total:,}  ({pct(n_ti, train_total)})"),
        ("Val correct",     f"{n_vc:,}  /  {val_total:,}  ({pct(n_vc, val_total)})"),
        ("Val incorrect",   f"{n_vi:,}  /  {val_total:,}  ({pct(n_vi, val_total)})"),
    ]
    if n_miss:
        stats.append(("Missing", f"{n_miss:,}"))

    y = 0.92
    for label, value in stats:
        ax2.text(0.03, y, label + ":", fontsize=10, fontweight="bold",
                 transform=ax2.transAxes, va="top")
        ax2.text(0.03, y - 0.045, value, fontsize=10,
                 transform=ax2.transAxes, va="top", color="#333333")
        y -= 0.13

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {out_path}")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Resolve output path (graceful read-only fallback)
# ---------------------------------------------------------------------------

def resolve_out(folder, filename, override_existing=False):
    p = folder / filename
    if p.exists() and not override_existing:
        return p  # already there, no writability check needed
    try:
        p.touch()
        p.unlink()
        return p
    except OSError:
        fallback = Path.cwd() / filename
        print(f"  (folder is read-only -- saving {filename} to {fallback})")
        return fallback


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(root_folder, override_existing=False):
    root = Path(root_folder)
    if not root.is_dir():
        sys.exit(f"ERROR: not a directory: {root}")

    csv_folders = find_csv_folders(root)
    if not csv_folders:
        sys.exit(f"No final_correct_position.csv found under {root}")

    print(f"Found {len(csv_folders)} folder(s) with final_correct_position.csv\n")

    n_ok = n_fail = 0
    for i, folder in enumerate(csv_folders, 1):
        print(f"[{i}/{len(csv_folders)}] {folder}")
        csv_path = folder / "final_correct_position.csv"

        mod_folder = find_modulus_folder(folder)
        if mod_folder is None:
            print(f"  WARNING: no 'modulus*' subfolder with required files — skipping")
            n_fail += 1
            print()
            continue

        print(f"  Dataset folder: {mod_folder.name}")
        out_path = resolve_out(folder, "correct_position_plot.pdf", override_existing=override_existing)

        try:
            ok = plot_correct_position(csv_path, mod_folder, out_path, override_existing=override_existing)
            n_ok += 1 if ok else 0
            n_fail += 0 if ok else 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            n_fail += 1
        print()

    print(f"Done.  {n_ok} succeeded,  {n_fail} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-position correctness grids from final_correct_position.csv. "
            "Recursively searches the given root folder."
        )
    )
    parser.add_argument("folder", help="Root folder to search")
    args = parser.parse_args()
    main(args.folder)
