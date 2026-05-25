#!/usr/bin/env python3
"""Measure train/val metrics along the linear interpolation between two LLR2 checkpoints."""

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

from py_src.engine import Device, train as engine_train, val as engine_val
from py_src.ml_setup import MLSetup, get_ml_setup_from_config
from py_src.ml_setup.dataloader_util import DataloaderConfig
from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup_model import ModelType
from py_src.model_opti_save_load import load_model_state_file
from py_src.util import setup_logging

logger = logging.getLogger("linear_interpolate_models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure train/val metrics along the linear interpolation between two .model.pt files",
    )
    parser.add_argument("start_model", type=Path, help="path to the starting .model.pt file")
    parser.add_argument("end_model", type=Path, help="path to the ending .model.pt file")
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=10,
        help="number of equal interpolation intervals; size=10 evaluates alpha 0.0..1.0 in steps of 0.1",
    )
    parser.add_argument("-d", "--dataset", default=None, help="Dataset override")
    parser.add_argument(
        "-o",
        "--output_folder_name",
        default=None,
        help="output folder for the CSV report",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=None,
        help="batch size override for both train and val loaders",
    )
    parser.add_argument(
        "--train_dataset_size",
        type=int,
        default=None,
        help="optional training dataset subsample size",
    )
    parser.add_argument(
        "--val_dataset_size",
        type=int,
        default=None,
        help="optional validation dataset subsample size",
    )
    parser.add_argument("--cpu", action="store_true", help="force CPU evaluation")
    parser.add_argument(
        "--dali",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use NVIDIA DALI for ImageNet dataloading when supported by the setup",
    )
    parser.add_argument("--dali_device_id", type=int, default=0, help="CUDA device id used by DALI pipelines")
    parser.add_argument("-P", "--torch_preset_version", type=int, default=None, help="factory preset index")
    parser.add_argument("-w", "--worker", type=int, default=0, help="dataloader worker count")
    parser.add_argument("--prefetch_factor", type=int, default=None, help="dataloader prefetch factor")
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
    folder_name = f"linear_interpolation_metrics_{start_stem}_to_{end_stem}_size_{args.size}"
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
            output[key] = copy.deepcopy(start_value)
    return output


def _build_ml_setup(
    model_type: ModelType,
    dataset_type: DatasetType,
    args: argparse.Namespace,
) -> MLSetup:
    return get_ml_setup_from_config(
        model_type.name,
        dataset_type=dataset_type.name,
        preset=args.torch_preset_version or 1,
        use_dali=args.dali,
        dali_device_id=args.dali_device_id,
    )


def _build_train_dataloader(ml_setup: MLSetup, args: argparse.Namespace):
    if ml_setup.training_data is None:
        return None

    loader_setup = copy.copy(ml_setup)
    loader_setup.override_train_loader = None
    loader_config = DataloaderConfig(
        batch_size=args.batch_size or ml_setup.default_batch_size,
        num_workers=args.worker,
        num_samples=args.train_dataset_size,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )
    return loader_setup.train_dataloader(loader_config, ignore_override=False)


def _build_val_dataloader(ml_setup: MLSetup, args: argparse.Namespace):
    if ml_setup.testing_data is None:
        return None

    loader_setup = copy.copy(ml_setup)
    loader_setup.override_test_loader = None
    loader_config = DataloaderConfig(
        batch_size=args.batch_size or ml_setup.default_batch_size,
        num_workers=args.worker,
        num_samples=args.val_dataset_size,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )
    return loader_setup.val_dataloader(loader_config, ignore_override=False)


def _format_optional_metric(value: Optional[float], *, missing: bool = False) -> str:
    if missing or value is None:
        return ""
    return f"{value:.10e}"


def _format_count(value: Optional[int]) -> str:
    if value is None:
        return ""
    return str(value)


def _evaluate_train_metrics(model, adapter, dataloader, device_obj: Device):
    if dataloader is None:
        return None
    result = engine_train(
        adapter,
        dataloader,
        optimizer=None,
        lr_scheduler=None,
        device=device_obj,
        scaler=None,
        backpropagation=False,
        training_mode=False,
    )
    if result.total_count == 0:
        return None
    return result


def _evaluate_val_metrics(adapter, dataloader, device_obj: Device):
    if dataloader is None:
        return None
    result = engine_val(adapter, dataloader, device=device_obj)
    if result.total_count == 0:
        return None
    return result


def _write_metrics_csv(
    output_path: Path,
    rows: list[dict[str, str]],
) -> None:
    fieldnames = [
        "index",
        "alpha",
        "train_loss",
        "train_accuracy",
        "train_count",
        "val_loss",
        "val_accuracy",
        "val_count",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    setup_logging(logger, "main")

    if args.size < 1:
        raise RuntimeError(f"--size must be at least 1, got {args.size}")
    if args.cpu and args.dali:
        raise RuntimeError("--dali requires CUDA; do not combine it with --cpu")

    for path in (args.start_model, args.end_model):
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")

    start_state, start_model_name, start_dataset_name = load_model_state_file(str(args.start_model))
    end_state, end_model_name, end_dataset_name = load_model_state_file(str(args.end_model))

    model_type = _resolve_model_type(start_model_name, end_model_name)
    dataset_type = _resolve_dataset_type(start_dataset_name, end_dataset_name, args.dataset)
    _validate_state_dicts(start_state, end_state)

    logger.info(
        "building ML setup for model=%s dataset=%s",
        model_type.name,
        dataset_type.name,
    )
    ml_setup = _build_ml_setup(model_type, dataset_type, args)

    device_obj = Device.cpu() if args.cpu else Device.auto()
    model = ml_setup.model.to(device_obj.device)
    adapter = ml_setup.adapter

    train_dataloader = _build_train_dataloader(ml_setup, args)
    val_dataloader = _build_val_dataloader(ml_setup, args)

    output_folder = _resolve_output_folder(args)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_folder / "interpolation_metrics.csv"

    logger.info(
        "measuring interpolation %s -> %s with size=%d on device=%s output=%s",
        args.start_model,
        args.end_model,
        args.size,
        device_obj.device,
        output_csv_path,
    )

    rows: list[dict[str, str]] = []
    for index in range(args.size + 1):
        alpha = index / args.size
        interpolated_state = _interpolate_state_dict(start_state, end_state, alpha)
        model.load_state_dict(interpolated_state)

        train_result = _evaluate_train_metrics(model, adapter, train_dataloader, device_obj)
        val_result = _evaluate_val_metrics(adapter, val_dataloader, device_obj)

        row = {
            "index": str(index),
            "alpha": f"{alpha:.10f}",
            "train_loss": _format_optional_metric(
                None if train_result is None else train_result.avg_loss,
                missing=train_result is None,
            ),
            "train_accuracy": _format_optional_metric(
                None if train_result is None else train_result.accuracy,
                missing=train_result is None,
            ),
            "train_count": _format_count(None if train_result is None else train_result.total_count),
            "val_loss": _format_optional_metric(
                None if val_result is None else val_result.avg_loss,
                missing=val_result is None,
            ),
            "val_accuracy": _format_optional_metric(
                None if val_result is None else val_result.accuracy,
                missing=val_result is None,
            ),
            "val_count": _format_count(None if val_result is None else val_result.total_count),
        }
        rows.append(row)

        logger.info(
            "point %d/%d alpha=%.6f train_loss=%s train_acc=%s val_loss=%s val_acc=%s",
            index,
            args.size,
            alpha,
            row["train_loss"] or "NA",
            row["train_accuracy"] or "NA",
            row["val_loss"] or "NA",
            row["val_accuracy"] or "NA",
        )

    _write_metrics_csv(output_csv_path, rows)
    logger.info("wrote metrics CSV to %s", output_csv_path)


if __name__ == "__main__":
    main()
