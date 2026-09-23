#!/usr/bin/env python3
"""Inspect and override model/dataset metadata in ``*.model.pt`` files.

The script changes only the top-level ``model_name`` and/or ``dataset_name``
fields. Tensor weights, state_dict keys, dtypes, and all other checkpoint
fields are preserved. Target names are restricted to the names currently
defined by ``py_src.ml_setup_model.model_types.ModelType`` and
``py_src.ml_setup_dataset.dataset_types.DatasetType``.

Examples (from the repository root)::

    # Print metadata, then interactively choose whether to rewrite it:
    python3 result_processing_tool/override_model_pt_info.py MODELS

    # With no positional argument, ask for the directory interactively first:
    python3 result_processing_tool/override_model_pt_info.py

    # Rewrite metadata in place for every direct child *.model.pt:
    python3 result_processing_tool/override_model_pt_info.py MODELS \
        --model-name binary_attention_cct_7_3x1_32 --dataset-name cifar10 --in-place

    # Write renamed copies to another directory:
    python3 result_processing_tool/override_model_pt_info.py MODELS \
        --model-name bnn_floating --output-dir MODELS_RETAGGED

Use ``--recursive`` to discover checkpoints below nested directories. Existing
files in an explicit output directory are refused; in-place updates use an
atomic temporary file and replace only after ``torch.save`` succeeds.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from py_src.ml_setup_dataset.dataset_types import DatasetType
from py_src.ml_setup_model.model_types import ModelType


MODEL_NAMES = tuple(item.name for item in ModelType)
DATASET_NAMES = tuple(item.name for item in DatasetType)


def checkpoint_files(input_dir: Path, recursive: bool) -> list[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    pattern = "**/*.model.pt" if recursive else "*.model.pt"
    return sorted((path for path in input_dir.glob(pattern) if path.is_file()),
                  key=lambda path: (str(path).lower(), str(path)))


def read_checkpoint_info(path: Path) -> tuple[Any, Any, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"{path}: could not load checkpoint: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a checkpoint dictionary")
    if "state_dict" not in payload or not isinstance(payload["state_dict"], dict):
        raise ValueError(f"{path}: missing a dictionary 'state_dict' field")
    return payload.get("model_name"), payload.get("dataset_name"), payload


def write_checkpoint_info(path: Path, payload: dict[str, Any], *, model_name: str | None,
                          dataset_name: str | None, destination: Path) -> None:
    updated = dict(payload)
    if model_name is not None:
        updated["model_name"] = model_name
    if dataset_name is not None:
        updated["dataset_name"] = dataset_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.",
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(updated, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def display_name(value: Any) -> str:
    return "<missing>" if value is None else str(value)


def _ask_yes_no(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        print("\nNo interactive input received; leaving files unchanged.")
        return False
    return answer in {"y", "yes"}


def _ask_input_dir() -> Path | None:
    try:
        value = input("Input directory containing *.model.pt files: ").strip()
    except EOFError:
        print("\nNo input directory received; leaving files unchanged.")
        return None
    if not value:
        print("No input directory provided; leaving files unchanged.")
        return None
    return Path(value)


def _ask_defined_name(kind: str, names: tuple[str, ...]) -> str:
    print(f"Defined {kind} names:")
    print("  " + ", ".join(names))
    while True:
        try:
            value = input(f"New {kind} name: ").strip()
        except EOFError as exc:
            raise ValueError("interactive input ended before a new name was entered") from exc
        if value in names:
            return value
        print(f"Unknown {kind} name {value!r}; choose one of the names above.")


def _interactive_override() -> tuple[str | None, str | None, bool, Path | None]:
    """Ask for metadata changes and a write destination when no CLI override exists."""
    change_model = _ask_yes_no("Modify the model name?")
    change_dataset = _ask_yes_no("Modify the dataset name?")
    if not change_model and not change_dataset:
        return None, None, False, None
    model_name = _ask_defined_name("model", MODEL_NAMES) if change_model else None
    dataset_name = _ask_defined_name("dataset", DATASET_NAMES) if change_dataset else None
    if _ask_yes_no("Rewrite the checkpoint files in place?"):
        return model_name, dataset_name, True, None
    try:
        output_text = input("Output directory for rewritten copies (empty cancels): ").strip()
    except EOFError:
        print("\nNo output directory received; leaving files unchanged.")
        return None, None, False, None
    if not output_text:
        print("No output directory provided; leaving files unchanged.")
        return None, None, False, None
    return model_name, dataset_name, False, Path(output_text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, nargs="?",
                        help="Directory containing *.model.pt files; prompted when omitted")
    parser.add_argument("--model-name", choices=MODEL_NAMES,
                        help="New model name; must be a current ModelType name")
    parser.add_argument("--dataset-name", choices=DATASET_NAMES,
                        help="New dataset name; must be a current DatasetType name")
    parser.add_argument("--in-place", action="store_true",
                        help="Rewrite the discovered checkpoint files in place")
    parser.add_argument("--output-dir", type=Path,
                        help="Write renamed copies below this directory instead of updating input files")
    parser.add_argument("--recursive", action="store_true",
                        help="Search recursively for *.model.pt files")
    args = parser.parse_args(argv)

    input_dir = args.input_dir
    if input_dir is None:
        input_dir = _ask_input_dir()
        if input_dir is None:
            return
    input_root = input_dir.resolve()

    if args.model_name is None and args.dataset_name is None:
        if args.in_place or args.output_dir is not None:
            parser.error("--in-place/--output-dir requires --model-name and/or --dataset-name")
    elif args.in_place and args.output_dir is not None:
        parser.error("--in-place and --output-dir cannot be combined")
    elif not args.in_place and args.output_dir is None:
        parser.error("metadata overrides require --in-place or --output-dir")

    try:
        files = checkpoint_files(input_root, args.recursive)
        if not files:
            raise ValueError("No *.model.pt files found")
        print(f"Found {len(files)} checkpoint(s) in {input_root}")
        print(f"{'file':<48} {'model_name':<42} dataset_name")
        print("-" * 110)
        loaded: list[tuple[Path, Any, Any, dict[str, Any]]] = []
        for path in files:
            model_name, dataset_name, payload = read_checkpoint_info(path)
            loaded.append((path, model_name, dataset_name, payload))
            relative = path.relative_to(input_root).as_posix()
            print(f"{relative:<48} {display_name(model_name):<42} {display_name(dataset_name)}")

        model_name = args.model_name
        dataset_name = args.dataset_name
        in_place = args.in_place
        output_dir = args.output_dir
        if model_name is None and dataset_name is None:
            model_name, dataset_name, in_place, output_dir = _interactive_override()
            if model_name is None and dataset_name is None:
                return

        destination_root = input_root if in_place else output_dir.resolve()
        print("\nMetadata override:")
        print(f"  model_name: {model_name or '<unchanged>'}")
        print(f"  dataset_name: {dataset_name or '<unchanged>'}")
        print(f"  destination: {'in place' if in_place else destination_root}")
        for path, old_model, old_dataset, payload in loaded:
            destination = path if in_place else destination_root / path.relative_to(input_root)
            if not in_place and destination.exists():
                raise ValueError(f"Refusing to overwrite existing output: {destination}")
            write_checkpoint_info(path, payload, model_name=model_name,
                                 dataset_name=dataset_name, destination=destination)
            print(f"Rewrote {destination}")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
