#!/usr/bin/env python3
"""Generate evenly spaced linear interpolations between two LLR2 .model.pt files."""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup_model import ModelType
from py_src.model_opti_save_load import load_model_state_file, save_model_state
from py_src.util import setup_logging

logger = logging.getLogger("linear_interpolate_models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evenly spaced linear interpolations between two .model.pt files",
    )
    parser.add_argument("start_model", type=Path, help="path to the starting .model.pt file")
    parser.add_argument("end_model", type=Path, help="path to the ending .model.pt file")
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=10,
        help="number of equal interpolation intervals; size=10 writes alpha 0.0..1.0 in steps of 0.1",
    )
    parser.add_argument("-d", "--dataset", default=None, help="Dataset override")
    parser.add_argument(
        "-o",
        "--output_folder_name",
        default=None,
        help="output folder for the interpolated checkpoints",
    )
    return parser.parse_args()


def _checkpoint_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".model.pt"):
        return name[: -len(".model.pt")]
    return path.stem


def _resolve_output_folder(args: argparse.Namespace) -> Path:
    if args.output_folder_name is not None:
        return Path(args.output_folder_name).expanduser().resolve()

    start_stem = _checkpoint_stem(args.start_model)
    end_stem = _checkpoint_stem(args.end_model)
    folder_name = f"linear_interpolation_{start_stem}_to_{end_stem}_size_{args.size}"
    return (Path.cwd() / folder_name).resolve()


def _coerce_model_type(name: Optional[str], source: str) -> Optional[ModelType]:
    if name is None:
        return None
    try:
        return ModelType[name]
    except KeyError as exc:
        raise RuntimeError(
            f"{source} contains unknown model type {name!r}. Valid options: {[item.name for item in ModelType]}"
        ) from exc


def _coerce_dataset_type(name: Optional[str], source: str) -> Optional[DatasetType]:
    if name is None:
        return None
    try:
        return DatasetType[name]
    except KeyError as exc:
        raise RuntimeError(
            f"{source} contains unknown dataset type {name!r}. Valid options: {[item.name for item in DatasetType]}"
        ) from exc


def _resolve_model_type(
    start_name: Optional[str],
    end_name: Optional[str],
) -> ModelType:
    start_type = _coerce_model_type(start_name, "start checkpoint")
    end_type = _coerce_model_type(end_name, "end checkpoint")

    if start_type is None and end_type is None:
        raise RuntimeError("both checkpoints are missing model type metadata")
    if start_type is None:
        return end_type  # type: ignore[return-value]
    if end_type is None:
        return start_type
    if start_type != end_type:
        raise RuntimeError(
            f"model type mismatch: start={start_type.name!r} end={end_type.name!r}"
        )
    return start_type


def _resolve_dataset_type(
    start_name: Optional[str],
    end_name: Optional[str],
    override_name: Optional[str],
) -> DatasetType:
    start_type = _coerce_dataset_type(start_name, "start checkpoint")
    end_type = _coerce_dataset_type(end_name, "end checkpoint")
    override_type = _coerce_dataset_type(override_name, "dataset override")

    metadata_types = {item for item in (start_type, end_type) if item is not None}
    if len(metadata_types) > 1:
        raise RuntimeError(
            "dataset type mismatch between checkpoints: "
            f"start={start_type.name if start_type is not None else None!r} "
            f"end={end_type.name if end_type is not None else None!r}"
        )

    metadata_type = next(iter(metadata_types), None)
    if override_type is not None:
        if metadata_type is not None and override_type != metadata_type:
            raise RuntimeError(
                "dataset override conflicts with checkpoint metadata: "
                f"override={override_type.name!r} metadata={metadata_type.name!r}"
            )
        return override_type

    if metadata_type is None:
        raise RuntimeError(
            "dataset metadata is missing in both checkpoints; provide -d/--dataset"
        )
    return metadata_type


def _copy_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return copy.deepcopy(value)


def _validate_state_dicts(
    start_state: dict[str, Any],
    end_state: dict[str, Any],
) -> None:
    start_keys = set(start_state.keys())
    end_keys = set(end_state.keys())
    if start_keys != end_keys:
        missing_in_end = sorted(start_keys - end_keys)
        missing_in_start = sorted(end_keys - start_keys)
        details: list[str] = []
        if missing_in_end:
            details.append(f"missing in end: {missing_in_end[:10]}")
        if missing_in_start:
            details.append(f"missing in start: {missing_in_start[:10]}")
        raise RuntimeError("state_dict keys do not match; " + "; ".join(details))

    for key in start_state.keys():
        start_value = start_state[key]
        end_value = end_state[key]

        if torch.is_tensor(start_value) != torch.is_tensor(end_value):
            raise RuntimeError(f"key {key!r} is a tensor in only one checkpoint")

        if torch.is_tensor(start_value):
            if start_value.shape != end_value.shape:
                raise RuntimeError(
                    f"shape mismatch for key {key!r}: {tuple(start_value.shape)} vs {tuple(end_value.shape)}"
                )
            if start_value.dtype != end_value.dtype:
                raise RuntimeError(
                    f"dtype mismatch for key {key!r}: {start_value.dtype} vs {end_value.dtype}"
                )
        else:
            if type(start_value) is not type(end_value):
                raise RuntimeError(
                    f"type mismatch for key {key!r}: {type(start_value).__name__} vs {type(end_value).__name__}"
                )
            if start_value != end_value:
                raise RuntimeError(
                    f"non-tensor metadata for key {key!r} differs between checkpoints"
                )


def _floating_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _interpolate_tensor(
    start_value: torch.Tensor,
    end_value: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    start_cpu = start_value.detach().cpu()
    end_cpu = end_value.detach().cpu()

    if start_cpu.dtype.is_floating_point:
        compute_dtype = _floating_compute_dtype(start_cpu.dtype)
        blended = torch.lerp(
            start_cpu.to(dtype=compute_dtype),
            end_cpu.to(dtype=compute_dtype),
            alpha,
        )
        return blended.to(dtype=start_cpu.dtype)

    if torch.equal(start_cpu, end_cpu):
        return start_cpu.clone()

    blended = torch.round(
        (1.0 - alpha) * start_cpu.to(dtype=torch.float64)
        + alpha * end_cpu.to(dtype=torch.float64)
    )
    return blended.to(dtype=start_cpu.dtype)


def _interpolate_state_dict(
    start_state: dict[str, Any],
    end_state: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, start_value in start_state.items():
        end_value = end_state[key]
        if torch.is_tensor(start_value):
            output[key] = _interpolate_tensor(start_value, end_value, alpha)
        else:
            output[key] = _copy_value(start_value)
    return output


def _write_manifest(
    output_folder: Path,
    rows: list[tuple[int, float, str]],
    start_model: Path,
    end_model: Path,
    model_type: ModelType,
    dataset_type: DatasetType,
) -> None:
    manifest_path = output_folder / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["index", "alpha", "path", "model_type", "dataset_type", "start_model", "end_model"])
        for index, alpha, path in rows:
            writer.writerow(
                [
                    index,
                    f"{alpha:.10f}",
                    path,
                    model_type.name,
                    dataset_type.name,
                    str(start_model.resolve()),
                    str(end_model.resolve()),
                ]
            )


def main() -> None:
    args = parse_args()
    setup_logging(logger, "main")

    if args.size < 1:
        raise RuntimeError(f"--size must be at least 1, got {args.size}")

    for path in (args.start_model, args.end_model):
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")

    start_state, start_model_name, start_dataset_name = load_model_state_file(str(args.start_model))
    end_state, end_model_name, end_dataset_name = load_model_state_file(str(args.end_model))

    model_type = _resolve_model_type(start_model_name, end_model_name)
    dataset_type = _resolve_dataset_type(start_dataset_name, end_dataset_name, args.dataset)
    _validate_state_dicts(start_state, end_state)

    output_folder = _resolve_output_folder(args)
    output_folder.mkdir(parents=True, exist_ok=True)

    logger.info(
        "interpolating %s -> %s with model=%s dataset=%s size=%d output=%s",
        args.start_model,
        args.end_model,
        model_type.name,
        dataset_type.name,
        args.size,
        output_folder,
    )

    manifest_rows: list[tuple[int, float, str]] = []
    for index in range(args.size + 1):
        alpha = index / args.size
        interpolated_state = _interpolate_state_dict(start_state, end_state, alpha)
        output_path = output_folder / f"{index:04d}_alpha_{alpha:.6f}.model.pt"
        save_model_state(
            str(output_path),
            interpolated_state,
            model_type.name,
            dataset_type.name,
        )
        manifest_rows.append((index, alpha, str(output_path)))

    _write_manifest(
        output_folder,
        manifest_rows,
        args.start_model,
        args.end_model,
        model_type,
        dataset_type,
    )
    logger.info("wrote %d interpolated checkpoints to %s", len(manifest_rows), output_folder)


if __name__ == "__main__":
    main()
