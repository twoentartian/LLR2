"""
plot_correct_position.py
------------------------
Find all folders containing "final_correct_position.csv", locate the
sibling ``dataset`` (or legacy ``modulus*``) subfolder with train.txt /
val.txt / tokenizer.txt, and plot a correctness grid.

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
import json
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
    Return the generated ``dataset`` folder when present.  Older experiment
    directories used a timestamped ``modulus*`` folder, so retain that as a
    fallback for backwards compatibility.
    """
    required = {"train.txt", "val.txt", "tokenizer.txt"}
    candidates = [csv_folder / "dataset"]
    candidates.extend(
        sub for sub in sorted(csv_folder.iterdir()) if sub.is_dir() and sub.name.startswith("modulus")
    )
    for sub in candidates:
        if sub.is_dir() and required.issubset({f.name for f in sub.iterdir() if f.is_file()}):
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
# Noisy-label recovery analysis
# ---------------------------------------------------------------------------

def _load_final_model(model_path: Path, device):
    """Load the standard grokking transformer used by generate_grokking."""
    # Keep the original plotting path usable without importing torch.  The
    # dependency is loaded only for --noise_analysis.
    import torch
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from py_src.ml_setup_model.transformer_for_grokking import TransformerForGrokking
    from py_src.ml_setup_dataset.dataset_modular import ArithmeticTokenizer

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = TransformerForGrokking(
        n_layers=2,
        n_heads=4,
        d_model=128,
        max_context_len=50,
        vocab_len=2000,
        trainable_position_encoding=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, ArithmeticTokenizer


def _infer_clean_predictions(csv_path: Path, model_path: Path, modulus: int,
                             batch_size: int, device_name: str):
    """Infer the model's deterministic final predictions for every position.

    The saved final_correct_position.csv contains only whether the original
    (possibly noisy) label matched the training-time forward pass.  The model
    checkpoint is therefore evaluated again in eval mode, and its prediction
    is compared with the clean modular-arithmetic answer.
    """
    import numpy as np
    import torch

    device = torch.device(
        device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, tokenizer_cls = _load_final_model(model_path, device)
    tokenizer = tokenizer_cls(modulus)
    eos_token = tokenizer.stoi["<|eos|>"]
    equal_token = tokenizer.stoi["="]
    operator_token = tokenizer.stoi.get(
        f"x**y_mod_{modulus}", tokenizer.stoi["unknown"]
    )

    # CSV stores the human-readable operand values (0, 1, ...), whereas the
    # transformer consumes tokenizer IDs.  Build the mapping explicitly
    # instead of assuming that numeric IDs always start at 2.
    numeric_token_ids = np.asarray(
        [tokenizer.stoi[str(value)] for value in range(modulus)],
        dtype=np.int64,
    )
    token_to_value = np.full(len(tokenizer.itos), -1, dtype=np.int64)
    for token_id, token in enumerate(tokenizer.itos):
        try:
            value = int(token)
        except (TypeError, ValueError):
            continue
        if 0 <= value < modulus:
            token_to_value[token_id] = value

    df = pd.read_csv(csv_path)
    lhs = df["lhs"].to_numpy(dtype=np.int64)
    rhs = df["rhs"].to_numpy(dtype=np.int64)
    clean_rhs = np.asarray(
        [pow(int(a), int(b), modulus) for a, b in zip(lhs, rhs)],
        dtype=np.int64,
    )
    recorded_correct = df["correct?"].astype(bool).to_numpy()
    predictions = np.full(lhs.shape, -1, dtype=np.int64)

    with torch.inference_mode():
        for start in range(0, len(lhs), batch_size):
            stop = min(start + batch_size, len(lhs))
            count = stop - start
            x = torch.stack(
                (
                    torch.full((count,), eos_token, dtype=torch.long),
                    torch.from_numpy(numeric_token_ids[lhs[start:stop]].copy()),
                    torch.full((count,), operator_token, dtype=torch.long),
                    torch.from_numpy(numeric_token_ids[rhs[start:stop]].copy()),
                    torch.full((count,), equal_token, dtype=torch.long),
                    torch.full((count,), eos_token, dtype=torch.long),
                ),
                dim=1,
            ).to(device)
            # Position 4 is the first RHS token after the '=' token.
            predicted_token = model(x)[0][:, 4, :].argmax(dim=-1).cpu().numpy()
            predictions[start:stop] = token_to_value[predicted_token]

    return df, clean_rhs, predictions, recorded_correct, str(device)


def _load_noisy_dataset_labels(dataset_folder: Path, modulus: int):
    """Load the actual noisy RHS and split membership for every coordinate."""
    import numpy as np

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from py_src.ml_setup_dataset.dataset_modular import ArithmeticDataset

    tokenizer_path = dataset_folder / "tokenizer.txt"
    train_path = dataset_folder / "train.txt"
    val_path = dataset_folder / "val.txt"
    required = (tokenizer_path, train_path, val_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing dataset files: " + ", ".join(missing))

    train = ArithmeticDataset.load_from_file(
        str(train_path), modulus, name="train", train=True,
        tokenizer_path=str(tokenizer_path),
    )
    val = ArithmeticDataset.load_from_file(
        str(val_path), modulus, name="val", train=False,
        tokenizer_path=str(tokenizer_path),
    )

    numeric_token_to_value = np.full(len(train.tokenizer.itos), -1, dtype=np.int64)
    for token_id, token in enumerate(train.tokenizer.itos):
        try:
            value = int(token)
        except (TypeError, ValueError):
            continue
        if 0 <= value < modulus:
            numeric_token_to_value[token_id] = value

    noisy_rhs = np.full(modulus * modulus, -1, dtype=np.int64)
    split = np.full(modulus * modulus, -1, dtype=np.int8)
    for split_id, dataset in enumerate((train, val)):
        encoded = dataset.data.detach().cpu().numpy()
        lhs = numeric_token_to_value[encoded[:, 1]]
        rhs = numeric_token_to_value[encoded[:, 3]]
        labels = numeric_token_to_value[encoded[:, 5]]
        if np.any(lhs < 0) or np.any(rhs < 0) or np.any(labels < 0):
            raise ValueError(f"non-numeric token found while reading {dataset_folder}")
        flat = lhs * modulus + rhs
        if np.any(noisy_rhs[flat] >= 0):
            raise ValueError(f"duplicate coordinate found while reading {dataset_folder}")
        noisy_rhs[flat] = labels
        split[flat] = split_id

    if np.any(noisy_rhs < 0):
        raise ValueError(
            f"dataset {dataset_folder} does not contain all {modulus * modulus} coordinates"
        )
    return noisy_rhs, split


def plot_noisy_label_recovery(csv_path: Path, model_path: Path,
                              out_stem: Path, modulus: int = 797,
                              batch_size: int = 8192, device: str = "auto",
                              override_existing: bool = False,
                              dataset_folder: Path | None = None) -> bool:
    """Plot clean-function recovery for one final comprehension model."""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plot_path = out_stem.with_suffix(".pdf")
    if plot_path.exists() and not override_existing:
        print(f"  Already exist -> {plot_path}")
        return True

    df, clean_rhs, predictions, recorded_correct, actual_device = _infer_clean_predictions(
        csv_path, model_path, modulus, batch_size, device
    )
    if dataset_folder is None:
        raise ValueError("dataset_folder is required for noisy-label recovery analysis")
    noisy_rhs_grid, split_grid = _load_noisy_dataset_labels(dataset_folder, modulus)
    flat_coordinates = df["lhs"].to_numpy(dtype=np.int64) * modulus + df["rhs"].to_numpy(dtype=np.int64)
    noisy_rhs = noisy_rhs_grid[flat_coordinates]
    split = split_grid[flat_coordinates]

    predicted_clean = predictions == clean_rhs
    noisy_label_is_wrong = noisy_rhs != clean_rhs
    predicted_noisy = predictions == noisy_rhs
    recovered_wrong = noisy_label_is_wrong & predicted_clean
    wrong_label_fitted = noisy_label_is_wrong & predicted_noisy
    clean_and_correct = ~noisy_label_is_wrong & predicted_clean
    clean_and_misclassified = ~noisy_label_is_wrong & ~predicted_clean
    wrong_label_other = noisy_label_is_wrong & ~predicted_clean & ~predicted_noisy
    invalid_prediction = predictions < 0

    classification = np.full(len(df), "wrong_label_third_answer", dtype=object)
    classification[clean_and_correct] = "clean_label_correct"
    classification[recovered_wrong] = "wrong_label_recovered"
    classification[wrong_label_fitted] = "wrong_label_fitted"
    classification[clean_and_misclassified] = "clean_label_misclassified"
    classification[invalid_prediction] = "invalid_prediction"

    # 0: clean label predicted correctly, 1: wrong label recovered, 2: wrong
    # label fitted, 3: clean label misclassified, 4: wrong label mapped to a
    # third answer, -1: invalid token.
    grid = np.full((modulus, modulus), -1, dtype=np.int8)
    for a, b, category in zip(df["lhs"], df["rhs"], np.where(
        invalid_prediction,
        -1,
        np.where(
            clean_and_correct, 0,
            np.where(recovered_wrong, 1,
                     np.where(wrong_label_fitted, 2,
                              np.where(clean_and_misclassified, 3, 4))),
        ),
    )):
        grid[int(a), int(b)] = int(category)

    cmap = {
        -1: (0.70, 0.70, 0.70, 1.0),
         0: (0.80, 0.95, 0.80, 1.0),
         1: (1.00, 0.60, 0.05, 1.0),
         2: (0.35, 0.60, 0.95, 1.0),
         3: (0.90, 0.25, 0.25, 1.0),
         4: (0.60, 0.25, 0.70, 1.0),
    }
    image = np.zeros((modulus, modulus, 4))
    for value, color in cmap.items():
        image[grid == value] = color

    # Save a complete table as well as the two most useful subsets for direct
    # inspection.  ``correct?`` is retained for comparison with the original
    # training-time (dropout-enabled) correctness CSV.
    classified = df[["lhs", "rhs", "correct?"]].copy()
    classified["split"] = np.where(split == 0, "train", "val")
    classified["noisy_rhs"] = noisy_rhs
    classified["clean_rhs"] = clean_rhs
    classified["predicted_rhs"] = predictions
    classified["noisy_label_is_wrong"] = noisy_label_is_wrong
    classified["predicted_clean"] = predicted_clean
    classified["predicted_noisy"] = predicted_noisy
    classified["classification"] = classification
    classified.to_csv(
        out_stem.with_name(out_stem.name + "_classification.csv"), index=False
    )
    classified.loc[recovered_wrong].to_csv(
        out_stem.with_name(out_stem.name + "_recovered_wrong_labels.csv"), index=False
    )
    classified.loc[clean_and_correct].to_csv(
        out_stem.with_name(out_stem.name + "_clean_correct.csv"), index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]})
    axes[0].imshow(image, origin="upper", aspect="equal",
                   extent=[-0.5, modulus - 0.5, modulus - 0.5, -0.5])
    axes[0].set_xlabel("rhs (column operand)")
    axes[0].set_ylabel("lhs (row operand)")
    axes[0].set_title(
        f"Clean-label recovery — {csv_path.parent.name}\n"
        "orange = originally wrong label recovered as the clean answer",
        fontsize=11, fontweight="bold",
    )
    step = max(1, modulus // 10)
    axes[0].set_xticks(list(range(0, modulus, step)))
    axes[0].set_yticks(list(range(0, modulus, step)))

    stats = [
        (0, "Clean label, predicted correctly", int(clean_and_correct.sum())),
        (1, "Wrong label recovered", int(recovered_wrong.sum())),
        (2, "Wrong label fitted", int(wrong_label_fitted.sum())),
        (3, "Clean label misclassified", int(clean_and_misclassified.sum())),
        (4, "Wrong label, third answer", int(wrong_label_other.sum())),
        (-1, "Invalid/non-numeric prediction", int(invalid_prediction.sum())),
    ]
    patches = [mpatches.Patch(facecolor=cmap[k][:3], edgecolor="lightgrey", label=f"{label} ({count:,})") for k, label, count in stats if count]
    fig.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=2, fontsize=8, framealpha=0.9)

    axes[1].axis("off")
    total = len(df)
    axes[1].text(0.03, 0.94, "Deterministic eval summary", fontsize=11,
                 fontweight="bold", transform=axes[1].transAxes, va="top")
    lines = [
        ("Device", actual_device),
        ("All positions", f"{total:,}"),
        ("Predicted clean", f"{int(predicted_clean.sum()):,} ({100*predicted_clean.mean():.2f}%)"),
        ("Originally wrong labels", f"{int(noisy_label_is_wrong.sum()):,} ({100*noisy_label_is_wrong.mean():.2f}%)"),
        ("Wrong labels recovered", f"{int(recovered_wrong.sum()):,} ({100*recovered_wrong.sum()/max(1, noisy_label_is_wrong.sum()):.2f}% of wrong)"),
    ]
    y = 0.84
    for label, value in lines:
        axes[1].text(0.03, y, label + ":", fontsize=9, fontweight="bold", transform=axes[1].transAxes, va="top")
        axes[1].text(0.03, y - 0.045, value, fontsize=9, transform=axes[1].transAxes, va="top", color="#333333")
        y -= 0.13

    summary = {
        "all_positions": int(total),
        "originally_wrong_labels": int(noisy_label_is_wrong.sum()),
        "predicted_clean": int(predicted_clean.sum()),
        "clean_label_correct": int(clean_and_correct.sum()),
        "wrong_label_recovered": int(recovered_wrong.sum()),
        "wrong_label_fitted": int(wrong_label_fitted.sum()),
        "clean_label_misclassified": int(clean_and_misclassified.sum()),
        "wrong_label_third_answer": int(wrong_label_other.sum()),
        "invalid_prediction": int(invalid_prediction.sum()),
        "wrong_label_recovery_rate": float(
            recovered_wrong.sum() / max(1, noisy_label_is_wrong.sum())
        ),
        "device": actual_device,
        "dataset_folder": str(dataset_folder),
        "model_path": str(model_path),
    }
    with out_stem.with_name(out_stem.name + "_summary.json").open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, indent=2)
        outfile.write("\n")

    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {plot_path}")
    print(f"  Classification CSV -> {out_stem.with_name(out_stem.name + '_classification.csv')}")
    print(f"  Recovered wrong-label CSV -> {out_stem.with_name(out_stem.name + '_recovered_wrong_labels.csv')}")
    print(f"  Summary JSON -> {out_stem.with_name(out_stem.name + '_summary.json')}")
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

def main(root_folder, override_existing=False, noise_analysis=False,
         phase_results_path=None, modulus=797, batch_size=8192,
         device="auto"):
    root = Path(root_folder)
    if not root.is_dir():
        sys.exit(f"ERROR: not a directory: {root}")

    if noise_analysis:
        phase_path = Path(phase_results_path) if phase_results_path else root / "phase_results.json"
        if not phase_path.exists():
            sys.exit(f"ERROR: phase results not found: {phase_path}")
        with phase_path.open(encoding="utf-8") as stream:
            phases = json.load(stream)
        comprehension_cells = []
        for cell in sorted(root.glob("lr*")):
            if not cell.is_dir():
                continue
            meta_path = cell / "meta.json"
            csv_path = cell / "final_correct_position.csv"
            model_path = cell / "00.model.pt"
            if not (meta_path.exists() and csv_path.exists() and model_path.exists()):
                continue
            with meta_path.open(encoding="utf-8") as stream:
                meta = json.load(stream)
            phase_key = f"lr={float(meta['learning_rate']):.4e},wd={float(meta['weight_decay']):.4e}"
            if phases.get(phase_key) == "comprehension":
                comprehension_cells.append((cell, csv_path, model_path, int(meta.get("modulus", modulus))))
        if not comprehension_cells:
            sys.exit("No comprehension cells with final_correct_position.csv and 00.model.pt found")
        dataset_folder = find_modulus_folder(root)
        if dataset_folder is None:
            sys.exit(
                "No dataset folder with train.txt, val.txt and tokenizer.txt found "
                "under the phase-diagram root"
            )
        print(f"Dataset labels: {dataset_folder}\n")
        print(f"Found {len(comprehension_cells)} comprehension cell(s)\n")
        n_ok = n_fail = 0
        for cell, csv_path, model_path, cell_modulus in comprehension_cells:
            print(f"Cell: {cell}")
            out_stem = cell / "clean_label_recovery"
            try:
                ok = plot_noisy_label_recovery(
                    csv_path, model_path, out_stem,
                    modulus=cell_modulus,
                    batch_size=batch_size,
                    device=device,
                    override_existing=override_existing,
                    dataset_folder=dataset_folder,
                )
                n_ok += int(ok)
                n_fail += int(not ok)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                n_fail += 1
            print()
        print(f"Done.  {n_ok} succeeded,  {n_fail} failed.")
        return

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
            print(f"  WARNING: no 'dataset' or legacy 'modulus*' subfolder with required files — skipping")
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
    parser.add_argument("--override", action="store_true",
                        help="Overwrite existing plots")
    parser.add_argument("--noise_analysis", action="store_true",
                        help="Analyze clean-label recovery for comprehension cells")
    parser.add_argument("--phase_results", default=None,
                        help="Optional phase_results.json path")
    parser.add_argument("--modulus", type=int, default=797,
                        help="Fallback modular size for noise analysis")
    parser.add_argument("--batch_size", type=int, default=8192,
                        help="Inference batch size for noise analysis")
    parser.add_argument("--device", default="auto",
                        help="Inference device, e.g. auto, cpu, cuda")
    args = parser.parse_args()
    main(
        args.folder,
        override_existing=args.override,
        noise_analysis=args.noise_analysis,
        phase_results_path=args.phase_results,
        modulus=args.modulus,
        batch_size=args.batch_size,
        device=args.device,
    )
