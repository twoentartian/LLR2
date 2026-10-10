#!/usr/bin/env python3
"""Aggregate LR/weight-decay misalignment sweep summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "cell", "summary", "dataset", "augmentation", "epochs", "batch_size",
    "learning_rate", "weight_decay", "misalignment_measure",
    "misalignment_a_as_train_b_as_val", "misalignment_b_as_train_a_as_val",
    "directional_measure_difference", "a_train_loss", "a_val_loss",
    "a_train_accuracy", "a_val_accuracy", "a_abs_loss_gap", "a_abs_accuracy_gap",
    "b_train_loss", "b_val_loss", "b_train_accuracy", "b_val_accuracy",
    "b_abs_loss_gap", "b_abs_accuracy_gap",
]


def _first_run(summary: dict[str, Any]) -> dict[str, Any]:
    runs = summary.get("runs", {})
    if not runs:
        raise ValueError("summary has no runs")
    return next(iter(runs.values()))


def _run_row(run: dict[str, Any]) -> dict[str, Any]:
    final = run.get("final", {})
    train = final.get("train", {})
    val = final.get("val", {})
    return {
        "train_loss": train.get("loss"),
        "val_loss": val.get("loss"),
        "train_accuracy": train.get("accuracy"),
        "val_accuracy": val.get("accuracy"),
        "abs_loss_gap": final.get("absolute_loss_gap"),
        "abs_accuracy_gap": final.get("absolute_accuracy_gap"),
    }


def read_cell(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    first = _first_run(summary)
    optimizer = first.get("optimizer") or {}
    by_direction = summary.get("aggregate", {}).get("misalignment_measure_by_direction", {})
    measure_a = by_direction.get("a_as_train_b_as_val")
    measure_b = by_direction.get("b_as_train_a_as_val")
    row = {
        "cell": path.name,
        "summary": str(summary_path),
        "dataset": summary.get("dataset"),
        "augmentation": first.get("augmentation"),
        "epochs": first.get("epochs"),
        "batch_size": first.get("batch_size"),
        "learning_rate": optimizer.get("learning_rate"),
        "weight_decay": optimizer.get("weight_decay"),
        "misalignment_measure": summary.get("aggregate", {}).get("mean_misalignment_measure"),
        "misalignment_a_as_train_b_as_val": measure_a,
        "misalignment_b_as_train_a_as_val": measure_b,
        "directional_measure_difference": (
            abs(measure_a - measure_b) if measure_a is not None and measure_b is not None else None
        ),
    }
    row.update({f"a_{key}": value for key, value in _run_row(summary["runs"]["a_as_train_b_as_val"]).items()})
    row.update({f"b_{key}": value for key, value in _run_row(summary["runs"]["b_as_train_a_as_val"]).items()})
    return row


def analyze(input_root: Path, output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for summary_path in sorted(input_root.glob("**/summary.json")):
        if summary_path.parent == output_root:
            continue
        try:
            rows.append(read_cell(summary_path.parent))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"Skipping malformed summary {summary_path}: {exc}")
    rows.sort(key=lambda row: (
        float("inf") if row["misalignment_measure"] is None else row["misalignment_measure"],
        row["cell"],
    ))
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "sweep_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output_root / "sweep_results.json").open("w", encoding="utf-8") as stream:
        json.dump({"cells": rows, "ranked_by": "misalignment_measure"}, stream, indent=2, sort_keys=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=None)
    args = parser.parse_args()
    rows = analyze(args.input_root, args.output_root or args.input_root)
    print(f"Analyzed {len(rows)} completed cells")
    for row in rows:
        print(
            f"{row['cell']}: measure={row['misalignment_measure']:.6g} "
            f"lr={row['learning_rate']:.6g} wd={row['weight_decay']:.6g} "
            f"A/B={row['misalignment_a_as_train_b_as_val']:.6g}/"
            f"{row['misalignment_b_as_train_a_as_val']:.6g}"
        )


if __name__ == "__main__":
    main()
