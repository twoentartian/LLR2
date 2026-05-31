#!/usr/bin/env python3

"""Measure model test accuracy and loss for an LLR2 checkpoint.

Ported from ``DFL_torch/tool/measure_model_test_accuracy.py`` and adapted to
LLR2's MLSetup + engine.val() evaluation flow.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src import ml_setup
from py_src.engine import Device, ValResult, val as engine_val
from py_src.ml_setup.dataloader_util import DataloaderConfig
from py_src.model_opti_save_load import load_model_state_file


def _name_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def _resolve_model_type(cli_model_type: str, file_model_type: Optional[str]) -> str:
    if cli_model_type == "auto":
        if file_model_type is None:
            raise SystemExit("model_type is not stored in the model file. Please pass --model_type.")
        return file_model_type

    if file_model_type is not None and file_model_type != cli_model_type:
        print(
            f"WARNING: model_type in CLI ({cli_model_type}) and model file ({file_model_type}) mismatch. "
            f"Using CLI value."
        )
    return cli_model_type


def _resolve_dataset_type(cli_dataset_type: str, file_dataset_type: Optional[str]) -> str:
    if cli_dataset_type == "auto":
        return file_dataset_type if file_dataset_type is not None else "default"

    if file_dataset_type is not None and file_dataset_type != cli_dataset_type:
        print(
            f"WARNING: dataset_type in CLI ({cli_dataset_type}) and model file ({file_dataset_type}) mismatch. "
            f"Using CLI value."
        )
    return cli_dataset_type


def _resolve_device(args: argparse.Namespace) -> Device:
    if args.cpu:
        return Device.cpu()
    if args.device is not None:
        return Device(args.device)
    return Device.auto()


def _build_dataloader_config(
    *,
    batch_size: int,
    num_workers: int,
    num_samples: Optional[int],
    shuffle: bool,
    device: Device,
) -> DataloaderConfig:
    worker_count = max(0, int(num_workers))
    return DataloaderConfig(
        batch_size=None if batch_size <= 0 else batch_size,
        num_workers=worker_count,
        num_samples=num_samples if num_samples is None or num_samples > 0 else None,
        shuffle=shuffle,
        pin_memory=device.device.type == "cuda",
        prefetch_factor=4 if worker_count > 0 else None,
        persistent_workers=True if worker_count > 0 else None,
    )


def evaluate_test_data(
    current_ml_setup,
    *,
    device: Device,
    batch_size: int,
    num_workers: int,
    num_samples: Optional[int],
) -> ValResult:
    if current_ml_setup.testing_data is None:
        raise SystemExit("MLSetup has no testing_data to evaluate.")

    dataloader = current_ml_setup.val_dataloader(
        _build_dataloader_config(
            batch_size=batch_size,
            num_workers=num_workers,
            num_samples=num_samples,
            shuffle=False,
            device=device,
        )
    )
    return engine_val(current_ml_setup.adapter, dataloader, device=device)


def evaluate_training_data(
    current_ml_setup,
    *,
    device: Device,
    batch_size: int,
    num_workers: int,
    num_samples: Optional[int],
) -> ValResult:
    if current_ml_setup.training_data is None:
        raise SystemExit("MLSetup has no training_data to evaluate.")

    dataloader = current_ml_setup.train_dataloader(
        _build_dataloader_config(
            batch_size=batch_size,
            num_workers=num_workers,
            num_samples=num_samples,
            shuffle=True,
            device=device,
        )
    )
    return engine_val(current_ml_setup.adapter, dataloader, device=device)


def get_layer_variances(state_dict: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, value in state_dict.items():
        if torch.is_tensor(value) and value.dtype.is_floating_point:
            output[name] = value.detach().float().var(unbiased=False).item()
    return output


def _format_accuracy(result: ValResult) -> str:
    if result.accuracy is None:
        return "n/a"
    return f"{result.accuracy:.8f}"


def _write_report(
    output_path: Path,
    *,
    model_file_path: Path,
    model_type: str,
    dataset_type: str,
    test_result: ValResult,
    train_result: Optional[ValResult],
    layer_variances: dict[str, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write(f"model_file={model_file_path}\n")
        outfile.write(f"model_type={model_type}\n")
        outfile.write(f"dataset_type={dataset_type}\n")
        outfile.write(f"test loss={test_result.avg_loss}, test acc={_format_accuracy(test_result)}\n")
        if train_result is not None:
            outfile.write(f"train loss={train_result.avg_loss}, train acc={_format_accuracy(train_result)}\n")
        outfile.write("\nLayer Variance List:\n")
        for layer_name, layer_var in layer_variances.items():
            outfile.write(f"layer {layer_name}: {layer_var}\n")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure model test accuracy and loss.")
    parser.add_argument("model_file", type=str, help="path to an LLR2 .model.pt file")
    parser.add_argument("-m", "--model_type", type=str, default="auto", help="model type, or auto to read from file")
    parser.add_argument("-d", "--dataset_type", type=str, default="auto", help="dataset type, or auto to read from file")
    parser.add_argument("-t", "--training", action="store_true", help="also evaluate the training dataset")
    parser.add_argument("-P", "--torch_preset_version", type=int, default=0, help="PyTorch/ImageNet preset version")
    parser.add_argument("-b", "--batch_size", type=int, default=100, help="batch size; <=0 uses MLSetup default")
    parser.add_argument("-c", "--core", type=int, default=os.cpu_count() or 1, help="number of CPU cores/dataloader workers")
    parser.add_argument("--num_samples", type=int, default=None, help="optional number of samples to evaluate")
    parser.add_argument("--cpu", action="store_true", help="force CPU execution")
    parser.add_argument("--device", type=str, default=None, help="explicit torch device, for example cuda:1")
    parser.add_argument("--dali", action=argparse.BooleanOptionalAction, default=False, help="use NVIDIA DALI for ImageNet dataloading")
    parser.add_argument("--dali_device_id", type=int, default=0, help="CUDA device id used by DALI pipelines")
    parser.add_argument("-o", "--output", type=str, default=None, help="output report path; defaults to <model_file>.txt")
    parser.add_argument("--no_output", action="store_true", help="print results only; do not write a report file")
    parser.add_argument("--non_strict", action="store_true", help="load state_dict with strict=False")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    model_file_path = Path(args.model_file)
    if not model_file_path.exists():
        raise SystemExit(f"file not found: {model_file_path}")

    model_state, file_model_type, file_dataset_type = load_model_state_file(str(model_file_path))
    file_model_type = _name_or_none(file_model_type)
    file_dataset_type = _name_or_none(file_dataset_type)

    model_type = _resolve_model_type(args.model_type, file_model_type)
    dataset_type = _resolve_dataset_type(args.dataset_type, file_dataset_type)

    if args.cpu and args.dali:
        raise SystemExit("--dali requires CUDA; remove --cpu or rerun without --dali.")

    current_ml_setup = ml_setup.get_ml_setup_from_config(
        model_type,
        dataset_type=dataset_type,
        preset=int(args.torch_preset_version),
        use_dali=bool(args.dali),
        dali_device_id=int(args.dali_device_id),
    )

    model = current_ml_setup.model
    model.load_state_dict(model_state, strict=not args.non_strict)

    device = _resolve_device(args)
    worker_count = min(max(int(args.core), 0), 8)

    test_result = evaluate_test_data(
        current_ml_setup,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=worker_count,
        num_samples=args.num_samples,
    )
    print(f"test loss={test_result.avg_loss}, test acc={_format_accuracy(test_result)}")

    train_result = None
    if args.training:
        train_result = evaluate_training_data(
            current_ml_setup,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=worker_count,
            num_samples=args.num_samples,
        )
        print(f"train loss={train_result.avg_loss}, train acc={_format_accuracy(train_result)}")

    if not args.no_output:
        output_path = Path(args.output) if args.output is not None else Path(f"{model_file_path}.txt")
        _write_report(
            output_path,
            model_file_path=model_file_path,
            model_type=model_type,
            dataset_type=dataset_type,
            test_result=test_result,
            train_result=train_result,
            layer_variances=get_layer_variances(model_state),
        )
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
