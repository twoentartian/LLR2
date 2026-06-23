"""
visualizing_arithemetic_dataset.py
--------------
Visualise which (a, b) operand pairs belong to the training set vs the
validation set.

Usage
-----
    python visualizing_arithemetic_dataset.py <folder>

<folder> must contain exactly three files:
    train.txt      – training equations
    val.txt        – validation equations
    tokenizer.txt  – one token per line (the vocabulary)

The script reads the operator tokens directly from tokenizer.txt so it
works with any operator, including ones not seen at development time.
"""

import argparse
import re
import sys
import os
from itertools import permutations
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm

# ---------------------------------------------------------------------------
# Tokenizer / vocabulary helpers
# ---------------------------------------------------------------------------

def load_tokens(tokenizer_path: Path) -> list[str]:
    """Return the list of tokens in vocabulary order."""
    return tokenizer_path.read_text().strip().split("\n")


def extract_operators(tokens: list[str]) -> list[str]:
    """
    Return the subset of tokens that are operators (not EOS/EQ/numbers/perms).
    Sorted longest-first so greedy matching picks the most specific operator.
    """
    skip = {"<|eos|>", "="}
    # Numbers: purely numeric strings
    # Permutation tokens: exactly 5 chars, all digits (e.g. '01234')
    def is_operand(t):
        if t in skip:
            return True
        if re.fullmatch(r"\d+", t):        # plain integer
            return True
        if re.fullmatch(r"[0-4]{5}", t):   # s5 permutation
            return True
        return False

    ops = [t for t in tokens if not is_operand(t)]
    # Longest first → greedy match won't confuse e.g. "s5" vs "s5conj"
    ops.sort(key=len, reverse=True)
    return ops


def build_operand_index(tokens: list[str]) -> dict[str, int]:
    """
    Map every non-operator, non-special token to a sequential integer index.
    This covers plain numbers (0, 1, …) and s5 permutation tokens alike.
    """
    skip = {"<|eos|>", "="}
    ops = set(extract_operators(tokens))
    operand_tokens = [t for t in tokens if t not in skip and t not in ops]
    return {t: i for i, t in enumerate(operand_tokens)}


# ---------------------------------------------------------------------------
# Equation parsing
# ---------------------------------------------------------------------------

def parse_equation(eq: str, operators: list[str],
                   operand_index: dict[str, int]):
    """
    Parse one equation line and return (row_idx, col_idx).

    Format expected (spaces are token separators):
        <|eos|> a OP b = c <|eos|>

    Returns None on failure.
    """
    eq = eq.strip()
    eq = re.sub(r"<\|eos\|>", "", eq).strip()
    if not eq:
        return None

    parts = eq.split(" = ")
    if len(parts) < 2:
        return None
    lhs = parts[0].strip()
    c_str = parts[1].strip().split()[0]

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
    a_str, b_str = halves[0].strip(), halves[1].strip()

    a_idx = operand_index.get(a_str)
    b_idx = operand_index.get(b_str)
    if a_idx is None or b_idx is None:
        return None

    return a_idx, b_idx, c_str


def load_data(txt_path, operators, operand_index):
    results = []
    for line in txt_path.read_text().splitlines():
        r = parse_equation(line, operators, operand_index)
        if r is not None:
            results.append(r)
    return results


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


def build_output_grid(all_data, operand_index, n):
    """
    Returns
        val_grid   : float (n,n), NaN where missing; numeric output value
                     (or vocabulary index for non-integer tokens like s5 perms)
        label_grid : str   (n,n), raw c_str for text display
        numeric    : bool, True when all outputs are plain integers
    """
    val_grid   = np.full((n, n), np.nan)
    label_grid = np.full((n, n), "", dtype=object)
    numeric    = True

    for a, b, c_str in all_data:
        if not (0 <= a < n and 0 <= b < n):
            continue
        label_grid[a, b] = c_str
        try:
            val_grid[a, b] = int(c_str)
        except ValueError:
            c_idx = operand_index.get(c_str)
            val_grid[a, b] = float(c_idx) if c_idx is not None else np.nan
            numeric = False

    return val_grid, label_grid, numeric

def find_dataset_folders(root: Path) -> list[Path]:
    """
    Return all directories (under root, including root itself) that contain
    train.txt, val.txt, and tokenizer.txt.

    Fast path: rglob for tokenizer.txt only, then check that the other two
    required files exist in the same directory.  This avoids enumerating
    every file and directory in the tree.
    """
    others = {"train.txt", "val.txt"}
    matches = []
    for tokenizer in sorted(root.rglob("tokenizer.txt")):
        folder = tokenizer.parent
        if all((folder / name).is_file() for name in others):
            matches.append(folder)
    return matches


# ---------------------------------------------------------------------------
# Shannon entropy helpers
# ---------------------------------------------------------------------------

def shannon_entropy(values: list) -> float:
    """
    Compute Shannon entropy (in bits) of the empirical output-value
    distribution for a given list of output tokens/values.

    H = -sum_c  p(c) * log2(p(c))

    where p(c) is the fraction of equations whose output equals c,
    computed *within* the supplied partition only.
    """
    if not values:
        return 0.0
    counts = Counter(values)
    total  = len(values)
    probs  = np.array([cnt / total for cnt in counts.values()], dtype=float)
    # guard against log(0) – only non-zero probabilities contribute
    probs  = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_partition_entropies(
    train_data: list,
    val_data:   list,
) -> dict:
    """
    Compute Shannon entropy of the output-value distribution for each
    partition independently.

    Parameters
    ----------
    train_data, val_data : list of (a_idx, b_idx, c_str) triples

    Returns
    -------
    dict with keys:
        'train_entropy'     – H computed over train outputs only  (bits)
        'val_entropy'       – H computed over val outputs only    (bits)
        'train_n_classes'   – number of distinct output values in train
        'val_n_classes'     – number of distinct output values in val
        'train_n_samples'   – number of equations in train
        'val_n_samples'     – number of equations in val
        'max_possible_entropy' – log2(total distinct output classes
                                       across both partitions)  (bits)
    """
    train_outputs = [c for _, _, c in train_data]
    val_outputs   = [c for _, _, c in val_data]

    all_classes = set(train_outputs) | set(val_outputs)
    max_entropy = float(np.log2(len(all_classes))) if all_classes else 0.0

    return {
        "train_entropy":       shannon_entropy(train_outputs),
        "val_entropy":         shannon_entropy(val_outputs),
        "train_n_classes":     len(set(train_outputs)),
        "val_n_classes":       len(set(val_outputs)),
        "train_n_samples":     len(train_outputs),
        "val_n_samples":       len(val_outputs),
        "max_possible_entropy": max_entropy,
    }


def print_entropy_report(entropy_stats: dict) -> None:
    """Pretty-print the entropy statistics to stdout."""
    e     = entropy_stats
    max_e = e["max_possible_entropy"]
    n_all = int(round(2 ** max_e)) if max_e > 0 else 0

    def pct(h):
        return f"{100 * h / max_e:.1f}%" if max_e > 0 else "N/A"

    print("=" * 55)
    print("  Shannon Entropy of Output-Value Distributions")
    print("=" * 55)
    print(f"  {'Partition':<12}  {'Samples':>8}  {'Classes':>8}  {'H (bits)':>10}  {'H/H_max':>8}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")
    print(f"  {'Train':<12}  {e['train_n_samples']:>8,}  {e['train_n_classes']:>8,}"
          f"  {e['train_entropy']:>10.4f}  {pct(e['train_entropy']):>8}")
    print(f"  {'Val':<12}  {e['val_n_samples']:>8,}  {e['val_n_classes']:>8,}"
          f"  {e['val_entropy']:>10.4f}  {pct(e['val_entropy']):>8}")
    print(f"  Max possible H (log2 of {n_all} total classes): {max_e:.4f} bits")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Orbit-based conditional entropy / held-out surprisal
# ---------------------------------------------------------------------------

def build_orbit_map(data: list) -> dict:
    """
    For a commutative (or partially commutative) operation, the *orbit* of
    an input pair (x, y) under the swap symmetry is {(x,y), (y,x)}.
    Canonical key = (min(a,b), max(a,b)).

    Returns a dict mapping canonical key -> list of (a, b, c) triples
    that belong to that orbit, drawn from `data`.
    """
    orbits: dict[tuple, list] = {}
    for a, b, c in data:
        key = (min(a, b), max(a, b))
        orbits.setdefault(key, []).append((a, b, c))
    return orbits


def compute_orbit_surprisal(
    train_data: list,
    val_data:   list,
    n_output_classes: int,
) -> dict:
    """
    Compute the orbit-conditional surprisal of val given train.

    Motivation
    ----------
    Plain H(V) ignores input structure and cannot distinguish:
      (A) outputs look diverse but are perfectly predictable once you know
          the orbit [(x,y)];
      (B) outputs look equally diverse and remain random even after knowing
          the orbit.

    The right quantity is:

        S(val | train, orbit) = -(1/|val|) * sum_{(x,y)->z in val}
                                    log2 p_train(z | orbit(x,y))

    where p_train(z | orbit) is the Laplace-smoothed empirical probability
    of output z within orbit(x,y) as observed in the train set.

    Three cases for a val example (x, y) -> z:
      1. orbit seen in train AND z matches a train output for that orbit:
             p ~ 1 -> surprisal ~ 0 bits
             (commutative: train already told us the answer)
      2. orbit seen in train but z is new (non-commutative inconsistency):
             Laplace smoothing gives a small probability
      3. orbit NOT seen in train at all:
             p(z | orbit) = 1 / n_output_classes  (uniform prior)
             surprisal = log2(n_output_classes)   [maximum]

    Extreme cases (for a 50/50 random split of a commutative operation,
    p output classes):
      - If the split always puts the "known" half of each orbit in train,
        the average surprisal -> 0.
      - If train and val are perfectly complementary (no orbit overlaps),
        average surprisal = log2(p) = maximum.

    Parameters
    ----------
    train_data        : list of (a, b, c_str) triples
    val_data          : list of (a, b, c_str) triples
    n_output_classes  : total number of distinct output tokens in the vocab
                        (used as the uniform-prior denominator)

    Returns
    -------
    dict with keys:
        'avg_surprisal'          - mean -log2 p_train(z|orbit) over val (bits)
        'total_surprisal'        - sum of per-example surprisals (bits)
        'max_surprisal'          - log2(n_output_classes), the ceiling (bits)
        'frac_of_max'            - avg_surprisal / max_surprisal
        'n_val_orbit_seen'       - val examples whose orbit appeared in train
        'n_val_orbit_unseen'     - val examples whose orbit was absent in train
        'n_val_total'            - total val examples scored
        'per_orbit_surprisal'    - dict: canonical key -> avg surprisal for
                                    val examples in that orbit
    """
    if not val_data:
        return {
            "avg_surprisal":       0.0,
            "total_surprisal":     0.0,
            "max_surprisal":       float(np.log2(max(n_output_classes, 1))),
            "frac_of_max":         0.0,
            "n_val_orbit_seen":    0,
            "n_val_orbit_unseen":  0,
            "n_val_total":         0,
            "per_orbit_surprisal": {},
        }

    max_surprisal = float(np.log2(max(n_output_classes, 1)))

    # Build train orbit lookup: canonical_key -> Counter of output values
    train_orbits = build_orbit_map(train_data)
    train_output_counts: dict[tuple, Counter] = {
        key: Counter(c for _, _, c in triples)
        for key, triples in train_orbits.items()
    }

    total_surprisal   = 0.0
    n_seen            = 0
    n_unseen          = 0
    per_orbit_totals: dict[tuple, list] = {}

    for a, b, z in val_data:
        key = (min(a, b), max(a, b))
        per_orbit_totals.setdefault(key, [])

        if key not in train_output_counts:
            # Orbit completely absent from train -> uniform prior
            s = max_surprisal
            n_unseen += 1
        else:
            counts = train_output_counts[key]
            n_train_for_orbit = sum(counts.values())
            # Laplace-smoothed probability: add 1 to every class
            p_z = (counts.get(z, 0) + 1) / (n_train_for_orbit + n_output_classes)
            s   = float(-np.log2(p_z))
            n_seen += 1

        total_surprisal += s
        per_orbit_totals[key].append(s)

    n_val = len(val_data)
    avg_surprisal = total_surprisal / n_val if n_val > 0 else 0.0

    per_orbit_surprisal = {
        key: float(np.mean(vals))
        for key, vals in per_orbit_totals.items()
    }

    return {
        "avg_surprisal":       avg_surprisal,
        "total_surprisal":     total_surprisal,
        "max_surprisal":       max_surprisal,
        "frac_of_max":         avg_surprisal / max_surprisal if max_surprisal > 0 else 0.0,
        "n_val_orbit_seen":    n_seen,
        "n_val_orbit_unseen":  n_unseen,
        "n_val_total":         n_val,
        "per_orbit_surprisal": per_orbit_surprisal,
    }


def print_orbit_surprisal_report(stats: dict) -> None:
    """Pretty-print the orbit-conditional surprisal results."""
    s     = stats
    max_s = s["max_surprisal"]
    print("=" * 65)
    print("  Orbit-Conditional Surprisal  S(val | train, orbit)")
    print("=" * 65)
    print(f"  Average surprisal per val example : {s['avg_surprisal']:.6f} bits")
    print(f"  Maximum possible (uniform prior)  : {max_s:.6f} bits")
    print(f"  Fraction of maximum               : {100*s['frac_of_max']:.4f} %")
    print(f"  Val examples w/ orbit in train    : {s['n_val_orbit_seen']:,}")
    print(f"  Val examples w/ orbit NOT in train: {s['n_val_orbit_unseen']:,}")
    print(f"  Total val examples scored         : {s['n_val_total']:,}")
    print()
    print("  Interpretation:")
    print( "    0 bits  = val outputs fully predictable from train orbits")
    print(f"    {max_s:.4f} bits = val outputs carry no info from train (blind guess)")
    print("=" * 65)


def plot_orbit_surprisal(
    stats:    dict,
    out_path: Path,
    n:        int,
    title:    str = "Orbit-Conditional Surprisal  S(val | train, orbit)",
) -> None:
    """
    Two-panel figure.
    Left  : gauge bar showing avg surprisal vs maximum.
    Right : n x n heatmap of per-orbit surprisal
            (grey = not in val, green = 0 bits, red = max bits).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- left: gauge bar --------------------------------------------------
    ax = axes[0]
    max_s = stats["max_surprisal"]
    avg_s = stats["avg_surprisal"]
    frac  = stats["frac_of_max"]
    bar_color = plt.cm.RdYlGn_r(frac)

    ax.barh(["Val surprisal"], [avg_s], color=bar_color, height=0.4, zorder=3)
    ax.axvline(max_s, color="grey", linestyle="--", linewidth=1.5,
               label=f"Max = {max_s:.4f} bits (uniform prior)")
    ax.set_xlim(0, max_s * 1.25)
    ax.set_xlabel("Surprisal (bits)", fontsize=11)
    ax.set_title("Average orbit-conditional surprisal\n(val examples)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.text(avg_s + max_s * 0.02, 0,
            f"{avg_s:.4f} bits\n({100*frac:.2f}% of max)",
            va="center", fontsize=10)

    n_seen   = stats["n_val_orbit_seen"]
    n_unseen = stats["n_val_orbit_unseen"]
    n_total  = stats["n_val_total"]
    ax.text(0.5, -0.12,
            f"Val: {n_total:,} total  |  {n_seen:,} orbit seen in train  "
            f"|  {n_unseen:,} orbit unseen",
            ha="center", transform=ax.transAxes, fontsize=9, color="dimgrey")

    ax.xaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # ---- right: per-orbit surprisal heatmap -------------------------------
    ax2 = axes[1]
    grid = np.full((n, n), np.nan)
    for (r, c), s_val in stats["per_orbit_surprisal"].items():
        if 0 <= r < n and 0 <= c < n:
            grid[r, c] = s_val
            if r != c:
                grid[c, r] = s_val

    cmap2 = plt.get_cmap("RdYlGn_r").copy()
    cmap2.set_bad(color=(0.92, 0.92, 0.92))

    masked = np.ma.masked_invalid(grid)
    im = ax2.imshow(masked, origin="upper", aspect="equal",
                    extent=[-0.5, n - 0.5, n - 0.5, -0.5],
                    cmap=cmap2, vmin=0, vmax=max_s,
                    interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Surprisal per orbit (bits)", fontsize=10)

    step = max(1, n // 10)
    ticks = list(range(0, n, step))
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    ax2.set_xlabel("b  (column operand)", fontsize=11)
    ax2.set_ylabel("a  (row operand)", fontsize=11)
    ax2.set_title("Per-orbit surprisal heatmap\n"
                  "(green=0 bits, red=max bits, grey=not in val)",
                  fontsize=10, fontweight="bold")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)



def plot_entropy_bar(entropy_stats: dict, out_path: Path,
                     title: str = "Shannon Entropy per Partition") -> None:
    """
    Save a bar chart comparing the entropy of the train and val partitions.
    A dashed horizontal line marks the theoretical maximum (uniform distribution
    over all distinct output classes seen in either partition).
    """
    e      = entropy_stats
    labels = ["Train", "Val"]
    values = [e["train_entropy"], e["val_entropy"]]
    colors = [(0.20, 0.45, 0.75), (0.85, 0.25, 0.25)]
    max_e  = e["max_possible_entropy"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, width=0.45, zorder=3)

    # Annotate bars with exact value and relative % of max
    for bar, val in zip(bars, values):
        pct = f"({100 * val / max_e:.1f}% of max)" if max_e > 0 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01 * max_e,
            f"{val:.4f} bits\n{pct}",
            ha="center", va="bottom", fontsize=10,
        )

    # Max-entropy reference line
    if max_e > 0:
        ax.axhline(max_e, color="grey", linestyle="--", linewidth=1.2,
                   label=f"Max H = {max_e:.4f} bits (uniform)")
        ax.legend(fontsize=9, loc="lower right")

    ax.set_ylabel("Shannon Entropy  (bits)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max_e * 1.18 if max_e > 0 else max(values) * 1.3)
    ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Annotate sample / class counts below bars
    counts = [
        (e["train_n_samples"], e["train_n_classes"]),
        (e["val_n_samples"],   e["val_n_classes"]),
    ]
    for bar, (n_samp, n_cls) in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            -0.04 * (ax.get_ylim()[1]),
            f"n={n_samp:,}\n{n_cls} classes",
            ha="center", va="top", fontsize=8.5, color="dimgrey",
            transform=ax.get_xaxis_transform(),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 -- Train / Val split membership
# ---------------------------------------------------------------------------

def plot_splits(train_data, val_data, n, out_path, title="Train / Val Split"):
    """
    Draw an n×n grid coloured by split membership.

    Blue  → train only
    Red   → val only
    Green → overlap (both)
    White → missing
    """
    train_coords = [(a, b) for a, b, _ in train_data]
    val_coords = [(a, b) for a, b, _ in val_data]

    grid = np.full((n, n), fill_value=-1, dtype=np.int8)
    for a, b in train_coords:
        if 0 <= a < n and 0 <= b < n:
            grid[a, b] = 0
    for a, b in val_coords:
        if 0 <= a < n and 0 <= b < n:
            grid[a, b] = 2 if grid[a, b] == 0 else 1

    cmap = {
        -1: (1.00, 1.00, 1.00, 1.0),
        0: (0.20, 0.45, 0.75, 1.0),
        1: (0.85, 0.25, 0.25, 1.0),
        2: (0.20, 0.70, 0.30, 1.0),
    }
    img = np.zeros((n, n, 4))
    for v, color in cmap.items():
        img[grid == v] = color

    n_train = int((grid == 0).sum())
    n_val = int((grid == 1).sum())
    n_both = int((grid == 2).sum())
    n_miss = int((grid == -1).sum())
    total = n * n

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    ax.imshow(img, origin="upper", aspect="equal",
              extent=[-0.5, n - 0.5, n - 0.5, -0.5])
    ax.set_xlabel("b  (column operand)", fontsize=11)
    ax.set_ylabel("a  (row operand)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    step = max(1, n // 10)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    patches = [
        mpatches.Patch(color=cmap[0][:3], label=f"Train ({n_train:,})"),
        mpatches.Patch(color=cmap[1][:3], label=f"Val   ({n_val:,})"),
    ]
    if n_both:
        patches.append(mpatches.Patch(color=cmap[2][:3], label=f"Both  ({n_both:,})"))
    if n_miss:
        patches.append(mpatches.Patch(color=(0.9, 0.9, 0.9), label=f"Missing ({n_miss:,})"))
    ax.legend(handles=patches, loc="upper right", framealpha=0.85, fontsize=10)

    ax2 = axes[1]
    ax2.axis("off")
    stats = [
        ("Grid size", f"{n} x {n}  =  {total:,}"),
        ("Train cells", f"{n_train:,}  ({100 * n_train / total:.1f}%)"),
        ("Val cells", f"{n_val:,}  ({100 * n_val / total:.1f}%)"),
        ("Overlap", f"{n_both:,}  ({100 * n_both / total:.1f}%)"),
        ("Missing", f"{n_miss:,}  ({100 * n_miss / total:.1f}%)"),
    ]
    y = 0.85
    for label, value in stats:
        ax2.text(0.05, y, label + ":", fontsize=11, fontweight="bold",
                 transform=ax2.transAxes)
        ax2.text(0.55, y, value, fontsize=11, transform=ax2.transAxes)
        y -= 0.12

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2 -- Output value heatmap  +  output distribution bar chart
# ---------------------------------------------------------------------------

def plot_output_heatmap(train_data, val_data, operand_index, n, out_path,
                        title="Output Value Heatmap"):
    """
    Page 1 (left panel)  : n×n heatmap coloured by output value (viridis).
                           Grey cells have no equation for that (a, b) pair.
    Page 2 (right panel) : Bar chart of the output-value frequency distribution
                           for the full dataset (train+val), train only, and
                           val only, covering every integer from 0 to modulus-1.
                           The modulus is inferred as the number of distinct
                           operand tokens (i.e. len(operand_index)).

    Both panels are saved as pages in the single PDF `out_path`.
    """
    all_data = train_data + val_data
    val_grid, _, numeric = build_output_grid(all_data, operand_index, n)

    # ------------------------------------------------------------------ #
    #  Panel 1 – heatmap (unchanged from original)                        #
    # ------------------------------------------------------------------ #
    fig1, ax1 = plt.subplots(figsize=(9, 8))

    cmap_img = plt.get_cmap("viridis").copy()
    cmap_img.set_bad(color=(0.85, 0.85, 0.85))

    masked = np.ma.masked_invalid(val_grid)
    im = ax1.imshow(masked, origin="upper", aspect="equal",
                    extent=[-0.5, n - 0.5, n - 0.5, -0.5],
                    cmap=cmap_img, interpolation="nearest")

    cbar = fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Output value" if numeric else "Output token index",
                   fontsize=11)

    ax1.set_xlabel("b  (column operand)", fontsize=11)
    ax1.set_ylabel("a  (row operand)", fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    step = max(1, n // 10)
    ticks = list(range(0, n, step))
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)

    fig1.tight_layout()

    # ------------------------------------------------------------------ #
    #  Panel 2 – output-value distribution bar chart                      #
    # ------------------------------------------------------------------ #
    # Infer the modulus from the actual integer output values in the data.
    # Collect all outputs that are plain integers, then modulus = max + 1.
    # This avoids counting non-integer operand tokens (e.g. s5 permutations)
    # that inflate len(operand_index) beyond the true modular range.
    int_outputs = []
    for _, _, c_str in all_data:
        try:
            int_outputs.append(int(c_str))
        except ValueError:
            pass
    if int_outputs:
        modulus = max(int_outputs) + 1
    else:
        # Fallback for non-integer outputs (e.g. s5): use vocabulary size
        modulus = len(operand_index)
    x_vals  = list(range(modulus))

    def count_outputs(data, modulus):
        """Count occurrences of each integer output 0 … modulus-1."""
        counts = Counter()
        for _, _, c_str in data:
            try:
                v = int(c_str)
                if 0 <= v < modulus:
                    counts[v] += 1
            except ValueError:
                pass
        return np.array([counts.get(v, 0) for v in range(modulus)], dtype=int)

    counts_train = count_outputs(train_data, modulus)
    counts_val   = count_outputs(val_data,   modulus)
    counts_all   = counts_train + counts_val

    # Uniform-distribution reference line value
    total_all   = counts_all.sum()
    uniform_all = total_all / modulus if modulus > 0 else 0

    # ---- colour: viridis so output-value colour matches the heatmap ----
    viridis = plt.get_cmap("viridis")
    bar_colors = [viridis(v / max(modulus - 1, 1)) for v in x_vals]

    fig2, axes2 = plt.subplots(3, 1, figsize=(max(10, modulus * 0.12), 12),
                                sharex=True)

    datasets = [
        (counts_all,   f"All  (train + val, n={total_all:,})",  bar_colors),
        (counts_train, f"Train  (n={counts_train.sum():,})",    bar_colors),
        (counts_val,   f"Val    (n={counts_val.sum():,})",      bar_colors),
    ]

    for ax, (counts, subtitle, colors) in zip(axes2, datasets):
        ax.bar(x_vals, counts, color=colors, width=0.85, zorder=3)

        # Uniform reference line for this partition
        total_part = counts.sum()
        if total_part > 0 and modulus > 0:
            uniform_part = total_part / modulus
            ax.axhline(uniform_part, color="red", linestyle="--",
                       linewidth=1.2, label=f"Uniform ({uniform_part:.1f})")
            ax.legend(fontsize=9, loc="upper right")

        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    axes2[-1].set_xlabel("Output value  (0 … modulus−1)", fontsize=11)

    # Set x-ticks: show every value if modulus ≤ 50, else every 10th
    tick_step = 1 if modulus <= 50 else (5 if modulus <= 100 else 10)
    axes2[-1].set_xticks(list(range(0, modulus, tick_step)))

    fig2.suptitle(
        f"Output-Value Frequency Distribution  (modulus = {modulus})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig2.tight_layout()

    # ------------------------------------------------------------------ #
    #  Write both figures as pages into the single PDF                    #
    # ------------------------------------------------------------------ #
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out_path) as pdf:
        pdf.savefig(fig1, dpi=150, bbox_inches="tight")
        pdf.savefig(fig2, dpi=150, bbox_inches="tight")

    plt.close(fig1)
    plt.close(fig2)
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 -- Output values as numbers in each cell
# ---------------------------------------------------------------------------

def plot_output_numbers(all_data, operand_index, n, out_path,
                        title="Output Values (Numbers)"):
    """
    Prints the output value as text inside each cell.
    Background colour is the same viridis scale as the heatmap.
    White/black text chosen automatically for readability.
    """
    val_grid, label_grid, numeric = build_output_grid(all_data, operand_index, n)

    fontsize = max(3, min(10, int(180 / n)))
    fig_side = max(8, min(24, n * 0.20))
    fig, ax = plt.subplots(figsize=(fig_side, fig_side * 0.92))

    cmap_img = plt.get_cmap("viridis").copy()
    cmap_img.set_bad(color=(0.85, 0.85, 0.85))

    vmin = np.nanmin(val_grid) if not np.all(np.isnan(val_grid)) else 0
    vmax = np.nanmax(val_grid) if not np.all(np.isnan(val_grid)) else 1
    norm = Normalize(vmin=vmin, vmax=vmax)

    masked = np.ma.masked_invalid(val_grid)
    ax.imshow(masked, origin="upper", aspect="equal",
              extent=[-0.5, n - 0.5, n - 0.5, -0.5],
              cmap=cmap_img, norm=norm, interpolation="nearest")

    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap_img)
    for row in range(n):
        for col in range(n):
            label = label_grid[row, col]
            if not label:
                continue
            v = val_grid[row, col]
            if np.isnan(v):
                continue
            rgba = scalar_map.to_rgba(v)
            lum  = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(col, row, label, ha="center", va="center",
                    fontsize=fontsize, color="white" if lum < 0.5 else "black",
                    fontfamily="monospace")

    ax.set_xlabel("b  (column operand)", fontsize=11)
    ax.set_ylabel("a  (row operand)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    step = max(1, n // 10)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core: process a single folder
# ---------------------------------------------------------------------------

def process_folder(folder: Path, override_existing=False) -> bool:
    """
    Generate the three figures for one dataset folder.
    Returns True on success, False if parsing yields no data.
    """
    train_path     = folder / "train.txt"
    val_path       = folder / "val.txt"
    tokenizer_path = folder / "tokenizer.txt"

    output_path_split_plot          = resolve_out(folder, "split_plot.pdf")
    output_path_output_heatmap      = resolve_out(folder, "output_heatmap.pdf")
    output_path_output_numbers      = resolve_out(folder, "output_numbers.pdf")
    output_path_entropy_bar         = resolve_out(folder, "entropy_bar.pdf")
    output_path_conditional_entropy = resolve_out(folder, "conditional_entropy.pdf")

    if (
        os.path.exists(output_path_split_plot)
        and os.path.exists(output_path_output_heatmap)
        and os.path.exists(output_path_output_numbers)
        and os.path.exists(output_path_entropy_bar)
        and os.path.exists(output_path_conditional_entropy)
    ):
        if not override_existing:
            return True

    tokens        = load_tokens(tokenizer_path)
    operators     = extract_operators(tokens)
    operand_index = build_operand_index(tokens)
    n             = len(operand_index)

    train_data = load_data(train_path, operators, operand_index)
    val_data   = load_data(val_path,   operators, operand_index)
    all_data   = train_data + val_data

    print(f"  {len(train_data):,} train  |  {len(val_data):,} val  "
          f"|  {len(operators)} operator(s)")

    if not all_data:
        print("  WARNING: no parseable equations found, skipping.")
        return False

    # Infer actual grid size from data
    n = max(max(a, b) for a, b, _ in all_data) + 1

    # -------------------------------------------------------------------
    # Shannon entropy
    # -------------------------------------------------------------------
    entropy_stats = compute_partition_entropies(train_data, val_data)
    print_entropy_report(entropy_stats)

    if not os.path.exists(output_path_entropy_bar) or override_existing:
        plot_entropy_bar(
            entropy_stats,
            out_path=output_path_entropy_bar,
            title=f"Shannon Entropy per Partition  (grid {n}×{n})",
        )

    # -------------------------------------------------------------------
    # Orbit-conditional surprisal  S(val | train, orbit)
    # -------------------------------------------------------------------
    # n_output_classes = number of distinct output values in the vocabulary
    all_output_vals   = sorted(set(c for _, _, c in train_data + val_data))
    n_output_classes  = len(all_output_vals)

    orbit_stats = compute_orbit_surprisal(train_data, val_data, n_output_classes)
    print_orbit_surprisal_report(orbit_stats)

    if not os.path.exists(output_path_conditional_entropy) or override_existing:
        plot_orbit_surprisal(
            orbit_stats,
            out_path=output_path_conditional_entropy,
            n=n,
            title=f"Orbit-Conditional Surprisal  S(val | train, orbit)  (grid {n}x{n})",
        )

    if not os.path.exists(output_path_split_plot):
        plot_splits(
            train_data, val_data, n=n,
            out_path=output_path_split_plot,
            title=f"Train / Val Split  (grid {n}x{n})",
        )

    if not os.path.exists(output_path_output_heatmap) or override_existing:
        plot_output_heatmap(
            train_data, val_data, operand_index, n=n,
            out_path=output_path_output_heatmap,
            title=f"Output Value Heatmap  (grid {n}x{n})",
        )

    if n < 200:
        if not os.path.exists(output_path_output_numbers):
            plot_output_numbers(
                all_data, operand_index, n=n,
                out_path=output_path_output_numbers,
                title=f"Output Values  (grid {n}x{n})",
            )
    return True

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(root_folder, override_existing=False):
    root = Path(root_folder)
    if not root.is_dir():
        sys.exit(f"ERROR: not a directory: {root}")

    folders = find_dataset_folders(root)
    if not folders:
        sys.exit(
            "No subfolders containing train.txt + val.txt + tokenizer.txt "
            f"were found under {root}"
        )
    print(f"Found {len(folders)} dataset folder(s) under {root}\n")

    n_ok = n_fail = 0
    for i, folder in enumerate(folders, 1):
        print(f"[{i}/{len(folders)}] Processing: {folder}")
        try:
            ok = process_folder(folder, override_existing=override_existing)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            n_fail += 1
        print()

    print(f"Done. {n_ok} succeeded, {n_fail} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(
        "Plot train/val split and output-value grids. "
        "Pass a folder containing train.txt / val.txt / tokenizer.txt, "
        "or use --recursive to scan all matching subfolders."
    ))
    parser.add_argument(
        "folder",
        help="Root folder to process (directly or recursively)",
    )
    args = parser.parse_args()
    main(args.folder)
