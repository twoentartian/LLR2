#!/usr/bin/env python3
"""Archived repair tool: re-measure old flatness on unaugmented data.

Runs of misalignment_measurement_2.py made before the flatness loaders were
switched to the deterministic level-0 transform measured flatness on random
augmented views.  This script reloads every saved final model and gap
checkpoint of a result folder, rebuilds the same A/B split from the folder's
command.txt, and measures relative flatness with the same settings (layers,
Hutchinson samples, microbatch size, probe seeds) on the unaugmented halves.

Training is not repeated, so the models themselves are unchanged.  Results
go to a new subfolder; the original files are never modified.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import misalignment_measurement_2 as measurement
from py_src.model_opti_save_load import load_model_state_file

DIRECTIONS = (("a_as_train_b_as_val", "a", "b"), ("b_as_train_a_as_val", "b", "a"))
CHECKPOINT_EPOCH = re.compile(r"epoch_(\d+)\.model\.pt$")


def _original_args(folder: Path, device: str, num_workers: int) -> tuple[argparse.Namespace, bool]:
    command = shlex.split((folder / "command.txt").read_text(encoding="utf-8"))
    script_index = next(index for index, part in enumerate(command) if part.endswith(".py"))
    uses_cct_entry = command[script_index].endswith("misalignment_cct_entry.py")
    argv = [command[script_index], *command[script_index + 1:],
            "--device", device, "--num_workers", str(num_workers)]
    with patch.object(sys, "argv", argv):
        return measurement.parse_args(), uses_cct_entry


def _apply_cct_entry_overrides() -> None:
    import misalignment_cct_entry as entry

    measurement.build_augmentation = entry.build_augmentation
    measurement.describe_augmentation = entry.describe_augmentation


def _model_files(folder: Path, direction: str) -> list[tuple[int | None, Path]]:
    files: list[tuple[int | None, Path]] = [(None, folder / f"final_{direction}.model.pt")]
    checkpoint_folder = folder / f"gap_checkpoints_{direction}"
    if checkpoint_folder.is_dir():
        for path in sorted(checkpoint_folder.glob("epoch_*.model.pt")):
            files.append((int(CHECKPOINT_EPOCH.search(path.name).group(1)), path))
    return [(epoch, path) for epoch, path in files if path.is_file()]


def remeasure(folder: Path, *, output_subdir: str, device: str, num_workers: int, overwrite: bool) -> dict:
    output_folder = folder / output_subdir
    if output_folder.exists() and not overwrite:
        raise FileExistsError(f"{output_folder} exists; pass --overwrite to replace it")
    original = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    args, uses_cct_entry = _original_args(folder, device, num_workers)
    if uses_cct_entry:
        _apply_cct_entry_overrides()
    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                                device if device != "auto" else "cpu")
    measurement.set_seed(args.random_seed)
    bundle = measurement.load_dataset_bundle(args)
    saved_permutation = torch.load(folder / original.get("permutation_file", "split_permutation.pt"))
    if not torch.equal(saved_permutation, bundle.permutation):
        raise RuntimeError(f"{folder}: rebuilt A/B split differs from split_permutation.pt")
    model = bundle.ml_setup.model.to(torch_device)
    criterion = measurement._resolve_criterion(bundle.ml_setup.criterion, args.label_smoothing)

    first_run = original["runs"][original["run_order"][0]]
    settings = first_run["relative_flatness"]
    args.relative_flatness_samples = int(settings["samples"])
    args.relative_flatness_batches = int(settings["batches"])
    flatness_config = measurement.describe_augmentation(args.dataset, 0)
    loader_backend = measurement.resolve_loader_backend(args.dataset, args.loader_backend)
    loaders = {
        name: measurement._build_loader(
            partition if clean is None else clean, int(settings["batch_size"]),
            device=torch_device, num_workers=num_workers, seed=args.random_seed + seed_offset,
            prefetch_factor=args.prefetch_factor, loader_backend=loader_backend,
            augmentation_config=flatness_config, is_train=False, distributed_loader=False,
        )
        for name, partition, clean, seed_offset in (
            ("a", bundle.partition_a, bundle.flatness_partition_a, 303),
            ("b", bundle.partition_b, bundle.flatness_partition_b, 404),
        )
    }

    output_folder.mkdir(parents=True, exist_ok=True)
    runs = {}
    for direction, train_name, val_name in DIRECTIONS:
        if direction not in original["runs"]:
            continue
        points, csv_rows = [], []
        for epoch, path in _model_files(folder, direction):
            state, _, _ = load_model_state_file(str(path), map_location="cpu")
            model.load_state_dict(state, strict=True)
            del state
            report, measure, rows = measurement._measure_direction_relative_flatness(
                model, loaders[train_name], loaders[val_name], criterion,
                device=torch_device, modular=bundle.modular, eq_position=bundle.eq_position,
                layers=list(settings["layers"]), train_partition_name=train_name,
                val_partition_name=val_name, args=args,
            )
            train_metrics, val_metrics = report["partitions"]["train"][0], report["partitions"]["val"][0]
            point = {
                "checkpoint": str(path.relative_to(folder)),
                "epoch": "final" if epoch is None else epoch,
                "clean_train_accuracy": train_metrics["accuracy"],
                "clean_val_accuracy": val_metrics["accuracy"],
                "clean_abs_accuracy_gap": abs(train_metrics["accuracy"] - val_metrics["accuracy"]),
                "clean_abs_loss_gap": abs(train_metrics["loss"] - val_metrics["loss"]),
                "misalignment_measure": measure["overall_mean"],
            }
            print(f"{folder.name} {direction} {point['epoch']}: "
                  f"clean acc gap={point['clean_abs_accuracy_gap']:.4f} RF={measure['overall_mean']:.6g}",
                  flush=True)
            points.append({**point, "relative_flatness": report, "misalignment_measure_detail": measure})
            csv_rows.extend({**point, **row} for row in rows)
        measurement._write_csv_rows(str(output_folder / f"relative_flatness_{direction}.csv"),
                                    csv_rows, list(csv_rows[0]) if csv_rows else [])
        runs[direction] = {
            "final": next(point for point in points if point["epoch"] == "final"),
            "checkpoints": [point for point in points if point["epoch"] != "final"],
        }
    measurement.close_dali_loaders(*loaders.values())
    values = [run["final"]["misalignment_measure"] for run in runs.values()]
    summary = {
        "mode": "remeasure_flatness_on_unaugmented_data",
        "source_folder": str(folder),
        "git_provenance": measurement._git_provenance(),
        "training_augmentation": first_run.get("augmentation"),
        "relative_flatness_augmentation_config": flatness_config,
        "relative_flatness_settings": {key: settings[key] for key in ("layers", "samples", "batch_size", "batches")},
        "runs": runs,
        "aggregate": {
            "misalignment_measure_by_direction": {name: run["final"]["misalignment_measure"] for name, run in runs.items()},
            "mean_misalignment_measure": sum(values) / len(values) if values else None,
        },
    }
    measurement._write_json(str(output_folder / "summary.json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", type=Path, help="result folders containing summary.json")
    parser.add_argument("--output_subdir", default="relative_flatness_unaugmented")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    options = parser.parse_args()
    for folder in options.folders:
        remeasure(folder.resolve(), output_subdir=options.output_subdir, device=options.device,
                  num_workers=options.num_workers, overwrite=options.overwrite)


if __name__ == "__main__":
    main()
