from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from py_src.model_opti_save_load import dataset_name_key, model_name_key, stat_dict_key


def _load_symbol_from_file(module_name: str, file_path: Path, symbol_name: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {symbol_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, symbol_name)


DatasetType = _load_symbol_from_file(
    "_change_model_metadata_dataset_types",
    _REPO_ROOT / "py_src" / "ml_setup_dataset" / "dataset_types.py",
    "DatasetType",
)
ModelType = _load_symbol_from_file(
    "_change_model_metadata_model_types",
    _REPO_ROOT / "py_src" / "ml_setup_model" / "model_types.py",
    "ModelType",
)


def _model_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.name.endswith(".model.pt"))


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} is not a dictionary checkpoint")
    if stat_dict_key not in checkpoint:
        raise KeyError(f"{path} does not contain {stat_dict_key!r}")
    return checkpoint


def _print_checkpoint_table(rows: list[tuple[Path, Optional[str], Optional[str]]], folder: Path) -> None:
    name_width = max(len(path.name) for path, _, _ in rows)
    model_width = max(10, *(len(model or "<missing>") for _, model, _ in rows))
    print()
    print(f"Found {len(rows)} .model.pt file(s) in {folder}")
    print(f"{'file'.ljust(name_width)}  {'model'.ljust(model_width)}  dataset")
    print(f"{'-' * name_width}  {'-' * model_width}  {'-' * 20}")
    for path, model_name, dataset_name in rows:
        print(
            f"{path.name.ljust(name_width)}  "
            f"{(model_name or '<missing>').ljust(model_width)}  "
            f"{dataset_name or '<missing>'}"
        )


def _print_choices(title: str, names: list[str]) -> None:
    print()
    print(title)
    for index, name in enumerate(names, start=1):
        print(f"  {index:2d}. {name}")


def _prompt_choice(prompt: str, names: list[str]) -> Optional[str]:
    name_by_lower = {name.lower(): name for name in names}
    while True:
        raw_value = input(prompt).strip()
        if not raw_value:
            return None
        if raw_value.isdigit():
            index = int(raw_value)
            if 1 <= index <= len(names):
                return names[index - 1]
        else:
            match = name_by_lower.get(raw_value.lower())
            if match is not None:
                return match
        print("Please enter a listed number/name, or press Enter to keep existing values.")


def _preview_changes(
    rows: list[tuple[Path, Optional[str], Optional[str]]],
    new_model_name: Optional[str],
    new_dataset_name: Optional[str],
) -> None:
    print()
    print("Preview")
    for path, old_model_name, old_dataset_name in rows:
        target_model_name = new_model_name if new_model_name is not None else old_model_name
        target_dataset_name = new_dataset_name if new_dataset_name is not None else old_dataset_name
        print(
            f"  {path.name}: "
            f"model {old_model_name or '<missing>'} -> {target_model_name or '<missing>'}; "
            f"dataset {old_dataset_name or '<missing>'} -> {target_dataset_name or '<missing>'}"
        )


def _atomic_save(checkpoint: dict[str, Any], target_path: Path) -> None:
    fd = None
    temp_path = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f"{target_path.name}.",
            suffix=".tmp",
            dir=str(target_path.parent),
        )
        os.close(fd)
        fd = None
        temp_path = Path(temp_name)
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, target_path)
    finally:
        if fd is not None:
            os.close(fd)
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _apply_changes(
    rows: list[tuple[Path, Optional[str], Optional[str]]],
    new_model_name: Optional[str],
    new_dataset_name: Optional[str],
) -> None:
    for path, _, _ in rows:
        checkpoint = _load_checkpoint(path)
        if new_model_name is not None:
            checkpoint[model_name_key] = new_model_name
        if new_dataset_name is not None:
            checkpoint[dataset_name_key] = new_dataset_name
        _atomic_save(checkpoint, path)
        print(f"updated {path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively change model/dataset metadata in .model.pt files.",
    )
    parser.add_argument("folder", type=Path, help="folder containing .model.pt files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a folder")

    files = _model_files(folder)
    if not files:
        print(f"No .model.pt files found in {folder}")
        return

    rows = []
    for path in files:
        checkpoint = _load_checkpoint(path)
        rows.append((path, checkpoint.get(model_name_key), checkpoint.get(dataset_name_key)))

    _print_checkpoint_table(rows, folder)

    model_names = [item.name for item in ModelType]
    dataset_names = [item.name for item in DatasetType]
    _print_choices("Model types", model_names)
    new_model_name = _prompt_choice(
        "New model type for all files (blank = keep existing): ",
        model_names,
    )
    _print_choices("Dataset types", dataset_names)
    new_dataset_name = _prompt_choice(
        "New dataset type for all files (blank = keep existing): ",
        dataset_names,
    )

    if new_model_name is None and new_dataset_name is None:
        print("No changes requested.")
        return

    _preview_changes(rows, new_model_name, new_dataset_name)
    answer = input("Apply these changes? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        print("Aborted; no files changed.")
        return

    _apply_changes(rows, new_model_name, new_dataset_name)
    print("Done.")


if __name__ == "__main__":
    main()
