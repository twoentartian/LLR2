#!/usr/bin/env python3
"""Re-measure saved misalignment models: accuracy/loss and relative flatness.

A result folder of misalignment_measurement_2.py (or misalignment_cct_entry.py)
is rebuilt from its ``command.txt``: the same ml_setup, model and criterion,
and the same A/B split, which is checked against ``split_permutation.pt``.
The saved final models, and optionally every gap checkpoint, are then
measured on both halves with a freely chosen image transform:

* ``--measure accuracy`` evaluates loss and accuracy (as the epoch-end
  evaluation does, so ``--augmentation none`` gives the gap on original
  images);
* ``--measure flatness`` measures relative flatness with the run's layers,
  Hutchinson samples and probe seeds unless overridden.

Each half can be subsampled (``--samples`` / ``--flatness_samples``); the
default draws a seeded random subset, and the same seed selects the same
examples in every run that shares the split. ``--sample_order first`` takes
the first examples in partition order instead, which reproduces the
original in-run flatness measurement. Training is never repeated; results go
to a new subfolder and the original files are not modified.

Examples::

    # Accuracy gap on all original (unaugmented) images, final models only.
    python remeasure_misalignment_flatness.py RESULT_FOLDER --measure accuracy

    # Flatness on 4096 random original images per half, all checkpoints.
    python remeasure_misalignment_flatness.py RESULT_FOLDER --measure flatness \\
        --samples 4096 --checkpoints all
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch.utils.data import Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import misalignment_measurement_2 as measurement
from py_src.model_opti_save_load import load_model_state_file

DIRECTIONS = {"a_as_train_b_as_val": ("a", "b"), "b_as_train_a_as_val": ("b", "a")}
CHECKPOINT_EPOCH = re.compile(r"epoch_(\d+)\.model\.pt$")
MEASUREMENTS = ("accuracy", "flatness")


def _sample_count(value: str) -> int | None:
    if value.lower() == "all":
        return None
    count = int(value)
    if count <= 0:
        raise argparse.ArgumentTypeError("sample counts must be positive or 'all'")
    return count


def _measurements(value: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = [name for name in names if name not in MEASUREMENTS]
    if not names or unknown:
        raise argparse.ArgumentTypeError(f"--measure takes a comma list of {', '.join(MEASUREMENTS)}")
    return names


def _original_args(folder: Path, options: argparse.Namespace) -> tuple[argparse.Namespace, bool, list[str]]:
    command = shlex.split((folder / "command.txt").read_text(encoding="utf-8"))
    script_index = next(index for index, part in enumerate(command) if part.endswith(".py"))
    uses_cct_entry = command[script_index].endswith("misalignment_cct_entry.py")
    argv = [command[script_index], *command[script_index + 1:], "--device", options.device,
            "--num_workers", str(options.num_workers), "--augmentation", options.augmentation]
    if options.dataset_path is not None:
        argv += ["--dataset_path", options.dataset_path]
    with patch.object(sys, "argv", argv):
        args = measurement.parse_args()
    return args, uses_cct_entry, command


def _apply_cct_entry_overrides() -> None:
    import misalignment_cct_entry as entry

    measurement.build_augmentation = entry.build_augmentation
    measurement.describe_augmentation = entry.describe_augmentation


def _subset(partition: Dataset, count: int | None, *, order: str, seed: int) -> Dataset:
    size = len(partition)
    if count is None or count >= size:
        return partition
    if order == "first":
        indices = list(range(count))
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        # Sorted, so loaders read in partition order; the set is what is random.
        indices = sorted(torch.randperm(size, generator=generator)[:count].tolist())
    return Subset(partition, indices)


def _checkpoints(folder: Path, direction: str, which: str, final_epoch: int | None) -> list[tuple[str, int | None, Path]]:
    files: list[tuple[str, int | None, Path]] = [("final", final_epoch, folder / f"final_{direction}.model.pt")]
    checkpoint_folder = folder / f"gap_checkpoints_{direction}"
    if which == "all" and checkpoint_folder.is_dir():
        for path in sorted(checkpoint_folder.glob("epoch_*.model.pt")):
            epoch = int(CHECKPOINT_EPOCH.search(path.name).group(1))
            # The final model is that epoch's checkpoint; measure it once.
            if epoch != final_epoch:
                files.append((f"epoch_{epoch}", epoch, path))
    missing = [str(path) for _, _, path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing model files: {missing}")
    return files


def _describe_criterion(criterion: torch.nn.Module) -> dict[str, Any]:
    return {"type": type(criterion).__name__, "label_smoothing": getattr(criterion, "label_smoothing", None)}


def _print_config(config: dict[str, Any]) -> None:
    print(f"== remeasure {config['source_folder']}", flush=True)
    for section, values in config.items():
        if isinstance(values, dict):
            print(f"  {section}:")
            for key, value in values.items():
                print(f"    {key}: {value}")
        elif section != "source_folder":
            print(f"  {section}: {values}")
    sys.stdout.flush()


def remeasure(folder: Path, options: argparse.Namespace) -> dict[str, Any]:
    original = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    args, uses_cct_entry, command = _original_args(folder, options)
    # A command without --random_seed/--split_seed drew them at run time;
    # summary.json records the values that were actually used.
    args.random_seed = original.get("random_seed", args.random_seed)
    args.split_seed = original.get("split_seed", args.split_seed if args.split_seed is not None else args.random_seed)
    if uses_cct_entry:
        _apply_cct_entry_overrides()
    device = torch.device("cuda" if options.device == "auto" and torch.cuda.is_available() else
                          options.device if options.device != "auto" else "cpu")
    first_run = next(iter(original["runs"].values()))
    level = measurement._parse_augmentation(options.augmentation)

    measurement.set_seed(args.random_seed)
    # The bundle carries the requested evaluation transform; the split depends
    # only on split_seed and is verified against the saved permutation.
    bundle = measurement.load_dataset_bundle(args)
    saved_permutation = torch.load(folder / original.get("permutation_file", "split_permutation.pt"))
    if not torch.equal(saved_permutation, bundle.permutation):
        raise RuntimeError(f"{folder}: rebuilt A/B split differs from split_permutation.pt")
    for partition in (bundle.partition_a, bundle.partition_b):
        measurement._set_augmentation_epoch(partition, options.augmentation_epoch)
    model = bundle.ml_setup.model.to(device)
    criterion = measurement._resolve_criterion(bundle.ml_setup.criterion, args.label_smoothing)
    recorded_criterion = first_run.get("criterion")
    criterion_info = _describe_criterion(criterion)
    if recorded_criterion is not None and recorded_criterion != criterion_info:
        raise RuntimeError(f"{folder}: criterion {criterion_info} differs from the recorded {recorded_criterion}")

    flatness_settings = first_run.get("relative_flatness") or {}
    layers_request = options.flatness_layers or ",".join(flatness_settings.get("layers") or []) or args.relative_flatness_layers
    layers = measurement._resolve_flatness_layers(model, layers_request)
    args.relative_flatness_samples = options.hutchinson_samples or int(flatness_settings.get("samples") or args.relative_flatness_samples)
    args.relative_flatness_batches = -1  # Subsampling is done through the partitions below.
    if options.flatness_seed is not None:
        args.relative_flatness_seed = options.flatness_seed
    flatness_batch_size = options.flatness_batch_size or int(flatness_settings.get("batch_size") or args.relative_flatness_batch_size)
    eval_batch_size = options.eval_batch_size or args.eval_batch_size or int(first_run.get("batch_size") or 128)
    accuracy_samples = options.samples
    flatness_samples = options.flatness_samples
    loader_backend = measurement.resolve_loader_backend(args.dataset, args.loader_backend)
    augmentation_config = measurement.describe_augmentation(args.dataset, level)
    directions = [name for name in original.get("run_order", list(original["runs"]))
                  if options.directions in ("both", name)]

    def sample_label(count):
        return "all" if count is None else f"{count} ({options.sample_order}, seed {options.sample_seed})"

    name_parts = [f"remeasure_aug-{augmentation_config['name']}"]
    if "accuracy" in options.measure:
        name_parts.append(f"acc-{'all' if accuracy_samples is None else accuracy_samples}")
    if "flatness" in options.measure:
        name_parts.append(f"flat-{'all' if flatness_samples is None else flatness_samples}")
    name_parts.append(f"{options.sample_order}{options.sample_seed}")
    output_folder = folder / (options.output_subdir or "_".join(name_parts))
    if output_folder.exists() and not options.overwrite:
        raise FileExistsError(f"{output_folder} exists; pass --overwrite or choose --output_subdir")

    final_epoch = int(first_run["epochs"]) - 1 if first_run.get("epochs") else None
    checkpoints = {direction: _checkpoints(folder, direction, options.checkpoints, final_epoch)
                   for direction in directions}
    config = {
        "source_folder": str(folder),
        "model": {
            "model_type": bundle.ml_setup.model_type.name,
            "model_preset": bundle.model_preset,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "criterion": criterion_info,
            "recorded_criterion": recorded_criterion if recorded_criterion is not None else "not recorded by this run",
        },
        "dataset": {
            "dataset": bundle.dataset_name,
            "dataset_type": bundle.dataset_type_name,
            "source_examples": bundle.source_examples,
            "half_examples": len(bundle.partition_a),
            "split_seed": args.split_seed,
            "split_check": "matches split_permutation.pt",
        },
        "original_run": {
            "script": Path(command[next(i for i, part in enumerate(command) if part.endswith('.py'))]).name,
            "training_augmentation": first_run.get("augmentation_config", {}).get("operations", first_run.get("augmentation")),
            "epochs": first_run.get("epochs"),
            "batch_size": first_run.get("batch_size"),
            "optimizer": {key: first_run.get("optimizer", {}).get(key)
                          for key in ("optimizer", "learning_rate", "weight_decay", "scheduler")},
            "random_seed": args.random_seed,
            "git_commit": (original.get("git_provenance") or {}).get("commit"),
        },
        "evaluation": {
            "measure": list(options.measure),
            "augmentation": augmentation_config["operations"],
            "augmentation_epoch": options.augmentation_epoch if level > 0 else None,
            "accuracy_samples_per_half": sample_label(accuracy_samples) if "accuracy" in options.measure else None,
            "accuracy_precision": options.eval_precision,
            "accuracy_batch_size": eval_batch_size,
            "flatness_samples_per_half": sample_label(flatness_samples) if "flatness" in options.measure else None,
            "flatness_layers": layers,
            "flatness_hutchinson_samples": args.relative_flatness_samples,
            "flatness_batch_size": flatness_batch_size,
            "flatness_probe_seed": args.relative_flatness_seed,
            "loader_backend": loader_backend,
            "device": str(device),
        },
        "checkpoints": {direction: [label for label, _, _ in items] for direction, items in checkpoints.items()},
        "output_folder": str(output_folder),
    }
    _print_config(config)
    if options.dry_run:
        return {"config": config}

    def build(partition, count, batch_size, seed_offset):
        return measurement._build_loader(
            _subset(partition, count, order=options.sample_order, seed=options.sample_seed),
            batch_size, device=device, num_workers=options.num_workers, seed=args.random_seed + seed_offset,
            prefetch_factor=args.prefetch_factor, loader_backend=loader_backend,
            augmentation_config=augmentation_config, is_train=False, distributed_loader=False,
        )

    # Loader seeds follow misalignment_measurement_2.py (evaluation 101/202,
    # flatness 303/404), so DALI augmentation views match the original run.
    partitions = {"a": bundle.partition_a, "b": bundle.partition_b}
    accuracy_loaders = ({name: build(partitions[name], accuracy_samples, eval_batch_size, offset)
                         for name, offset in (("a", 101), ("b", 202))}
                        if "accuracy" in options.measure else {})
    flatness_loaders = ({name: build(partitions[name], flatness_samples, flatness_batch_size, offset)
                         for name, offset in (("a", 303), ("b", 404))}
                        if "flatness" in options.measure else {})

    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / "command.txt").write_text(" ".join([sys.executable, *sys.argv]), encoding="utf-8")
    runs: dict[str, list[dict[str, Any]]] = {}
    csv_rows: list[dict[str, Any]] = []
    try:
        for direction in directions:
            train_name, val_name = DIRECTIONS[direction]
            runs[direction] = []
            for label, epoch, path in checkpoints[direction]:
                state, _, _ = load_model_state_file(str(path), map_location="cpu")
                model.load_state_dict(state, strict=True)
                del state
                model.eval()
                point: dict[str, Any] = {"checkpoint": str(path.relative_to(folder)), "label": label, "epoch": epoch}
                if "accuracy" in options.measure:
                    metrics = {
                        role: measurement.evaluate_partition(
                            model, accuracy_loaders[name], criterion, device=device, modular=bundle.modular,
                            eq_position=bundle.eq_position, precision=options.eval_precision,
                            progress_label=f"{direction} {label} {role}",
                        )
                        for role, name in (("train", train_name), ("val", val_name))
                    }
                    point["accuracy"] = {
                        **{role: {"partition": name, "loss": metrics[role].loss, "accuracy": metrics[role].accuracy,
                                  "examples": metrics[role].examples}
                           for role, name in (("train", train_name), ("val", val_name))},
                        "abs_accuracy_gap": abs(metrics["train"].accuracy - metrics["val"].accuracy),
                        "abs_loss_gap": abs(metrics["train"].loss - metrics["val"].loss),
                    }
                    csv_rows.extend({"direction": direction, "checkpoint": label, "epoch": epoch,
                                     "measurement": "accuracy", "role": role, **point["accuracy"][role]}
                                    for role in ("train", "val"))
                if "flatness" in options.measure:
                    report, measure, rows = measurement._measure_direction_relative_flatness(
                        model, flatness_loaders[train_name], flatness_loaders[val_name], criterion,
                        device=device, modular=bundle.modular, eq_position=bundle.eq_position, layers=layers,
                        train_partition_name=train_name, val_partition_name=val_name, args=args,
                    )
                    point["relative_flatness"] = report
                    point["misalignment_measure"] = measure
                    csv_rows.extend({"direction": direction, "checkpoint": label, "epoch": epoch,
                                     "measurement": "flatness", **row} for row in rows)
                runs[direction].append(point)
                parts = [f"{direction} {label}:"]
                if "accuracy" in point:
                    accuracy = point["accuracy"]
                    parts.append(f"train acc={accuracy['train']['accuracy']:.4f} val acc={accuracy['val']['accuracy']:.4f} "
                                 f"gap={accuracy['abs_accuracy_gap']:.4f}")
                if "misalignment_measure" in point:
                    parts.append(f"RF={point['misalignment_measure']['overall_mean']:.6g}")
                print(" ".join(parts), flush=True)
                # Persist completed measurements even if a later checkpoint fails.
                measurement._write_csv_rows(str(output_folder / "measurements.csv"), csv_rows,
                                            sorted({key for row in csv_rows for key in row}, key=str))
    finally:
        measurement.close_dali_loaders(*accuracy_loaders.values(), *flatness_loaders.values())

    finals = {direction: next(point for point in points if point["label"] == "final") for direction, points in runs.items()}
    aggregate: dict[str, Any] = {}
    if "accuracy" in options.measure:
        gaps = {direction: point["accuracy"]["abs_accuracy_gap"] for direction, point in finals.items()}
        aggregate["abs_accuracy_gap_by_direction"] = gaps
        aggregate["mean_abs_accuracy_gap"] = sum(gaps.values()) / len(gaps)
    if "flatness" in options.measure:
        values = {direction: point["misalignment_measure"]["overall_mean"] for direction, point in finals.items()}
        aggregate["misalignment_measure_by_direction"] = values
        aggregate["mean_misalignment_measure"] = sum(values.values()) / len(values)
    summary = {
        "mode": "remeasure_saved_models",
        "config": config,
        "git_provenance": measurement._git_provenance(),
        "runs": runs,
        "aggregate": aggregate,
    }
    measurement._write_json(str(output_folder / "summary.json"), summary)
    print(f"aggregate: {json.dumps(aggregate)}", flush=True)
    print(f"results written to {output_folder}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", type=Path, help="result folders containing command.txt and summary.json")
    parser.add_argument("--measure", type=_measurements, default=MEASUREMENTS,
                        help="comma list of accuracy,flatness (default: both)")
    parser.add_argument("--augmentation", default="none",
                        help="image transform for the measurement: none/0 (default) or a training level")
    parser.add_argument("--augmentation_epoch", type=int, default=0,
                        help="epoch that seeds stochastic augmentation; the final epoch reproduces its evaluation")
    parser.add_argument("--samples", type=_sample_count, default=None, metavar="N|all",
                        help="examples per half (default: all)")
    parser.add_argument("--flatness_samples", type=_sample_count, default=argparse.SUPPRESS, metavar="N|all",
                        help="examples per half for flatness (default: same as --samples)")
    parser.add_argument("--sample_order", choices=["random", "first"], default="random",
                        help="random seeded subset (default) or the first examples in partition order")
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--checkpoints", choices=["final", "all"], default="final",
                        help="final models only, or also every saved gap checkpoint")
    parser.add_argument("--directions", choices=["both", *DIRECTIONS], default="both")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="default: the run's evaluation batch size")
    parser.add_argument("--eval_precision", choices=["bf16", "fp32"], default="bf16",
                        help="accuracy evaluation precision (flatness is always full precision)")
    parser.add_argument("--flatness_batch_size", type=int, default=None, help="default: the run's flatness batch size")
    parser.add_argument("--flatness_layers", default=None, help="default: the run's flatness layers")
    parser.add_argument("--hutchinson_samples", type=int, default=None, help="default: the run's probe count")
    parser.add_argument("--flatness_seed", type=int, default=None, help="default: the run's --relative_flatness_seed")
    parser.add_argument("--dataset_path", default=None, help="override a relative modular dataset path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_subdir", default=None, help="default: derived from the measurement settings")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="print the configuration without measuring")
    options = parser.parse_args()
    if not hasattr(options, "flatness_samples"):
        options.flatness_samples = options.samples
    return options


def main() -> None:
    options = parse_args()
    for folder in options.folders:
        remeasure(folder.resolve(), options)


if __name__ == "__main__":
    main()
