import argparse
import io
import os
import re
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional

import lmdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src import ml_setup, model_opti_save_load
from py_src.adapters import ModelAdapter, clone_adapter_for_model
from py_src.engine import Device, val as engine_val
from py_src.ml_setup.dataloader_util import DataloaderConfig, build_dataloader


def _unwrap_model_payload(payload):
    if (
        isinstance(payload, dict)
        and model_opti_save_load.stat_dict_key in payload
    ):
        return (
            payload[model_opti_save_load.stat_dict_key],
            payload.get(model_opti_save_load.model_name_key),
            payload.get(model_opti_save_load.dataset_name_key),
        )
    return payload, None, None


def get_model_weights_from_file(
    path: Path,
    tick: Optional[int] = None,
) -> tuple[dict, Optional[str], Optional[str], str]:
    model_weights = None
    model_name = None
    dataset_name = None
    output_name = None

    if path.is_file():
        print(f"[info] '{path}' is a file.")
        model_weights, model_name, dataset_name = model_opti_save_load.load_model_state_file(str(path))
        output_name = str(path)
    elif path.is_dir():
        print(f"[info] '{path}' is a folder.")
        lmdb_data_path = path / "data.mdb"
        lmdb_lock_path = path / "lock.mdb"
        if not (lmdb_data_path.exists() and lmdb_lock_path.exists()):
            print("[error] LMDB files are missing", file=sys.stderr)
            sys.exit(1)
        if tick is None:
            raise AssertionError("tick has to be provided in LMDB mode")
        lmdb_index = int(tick)
        env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            all_ticks = set()
            for key, value in cursor:
                match = re.search(r"/(\d+)\.model\.pt$", key.decode())
                if match is None:
                    continue
                current_tick = int(match.group(1))
                all_ticks.add(current_tick)
                if current_tick == lmdb_index:
                    buffer = io.BytesIO(value)
                    payload = torch.load(buffer, map_location="cpu", weights_only=True)
                    model_weights, model_name, dataset_name = _unwrap_model_payload(payload)
                    break
            if model_weights is None:
                print(f"[error] tick is not in the lmdb, all ticks: {sorted(all_ticks)}", file=sys.stderr)
                sys.exit(1)
        output_name = f"{path}_tick{lmdb_index}"
    else:
        print(f"[error] Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    assert model_weights is not None
    assert output_name is not None
    return model_weights, model_name, dataset_name, output_name


def _build_eval_dataloader(
    current_ml_setup,
    device: Device,
    number_of_core: int,
) -> Iterable:
    num_workers = min(8, int(number_of_core))
    eval_collate_fn = current_ml_setup.default_collate_fn_val or current_ml_setup.default_collate_fn
    config = DataloaderConfig(
        batch_size=current_ml_setup.default_batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=device.device.type == "cuda",
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else None,
        collate_fn=eval_collate_fn,
    )
    return build_dataloader(
        current_ml_setup.training_data,
        default_batch_size=current_ml_setup.default_batch_size,
        config=config,
        is_train=False,
        default_collate_fn=eval_collate_fn,
    )


def _clone_state_dict(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, tensor.detach().clone())
        for name, tensor in model.state_dict().items()
    )


def _per_param_unit_dirs(model: torch.nn.Module) -> list[Optional[torch.Tensor]]:
    dirs: list[Optional[torch.Tensor]] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            dirs.append(None)
            continue
        direction = torch.randn_like(parameter)
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
        dirs.append(direction)
    return dirs


def _per_param_l2_norms(model: torch.nn.Module) -> list[Optional[float]]:
    norms: list[Optional[float]] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            norms.append(None)
        else:
            norms.append(max(torch.linalg.vector_norm(parameter.detach()).item(), 1e-12))
    return norms


@torch.inference_mode()
def compute_loss_of_point(
    adapter: ModelAdapter,
    dataloader: Iterable,
    device: Device,
) -> float:
    result = engine_val(adapter, dataloader, device=device)
    return result.avg_loss


@torch.inference_mode()
def loss_landscape_sharpness(
    model: torch.nn.Module,
    adapter: ModelAdapter,
    dataloader: Iterable,
    *,
    change_ratio: Optional[list[float]] = None,
    sample_count: int = 100,
    device: Device,
) -> dict[float, list[float]]:
    if change_ratio is None:
        change_ratio = [0.001, 0.002, 0.003]

    baseline_loss = compute_loss_of_point(adapter, dataloader, device)
    original_state = _clone_state_dict(model)
    model.eval()
    model.to(device.device)

    norms = _per_param_l2_norms(model)
    output: dict[float, list[float]] = {}
    for ratio in change_ratio:
        output[ratio] = []
        for sample_idx in range(sample_count):
            model.load_state_dict(original_state, strict=True)
            directions = _per_param_unit_dirs(model)
            for parameter, direction, norm in zip(model.parameters(), directions, norms):
                if parameter.requires_grad and direction is not None and norm is not None:
                    parameter.add_(direction, alpha=ratio * norm)

            perturbed_loss = compute_loss_of_point(adapter, dataloader, device)
            delta_loss = abs(perturbed_loss - baseline_loss)
            print(f"[info] finish ratio {ratio}, sample {sample_idx}.")
            output[ratio].append(delta_loss)

    model.load_state_dict(original_state, strict=True)
    return output


def _draw_result(csv_path: str) -> None:
    dataframe = pd.read_csv(csv_path)
    data = dataframe.iloc[:, 1:].values
    x_values = dataframe.columns[1:].astype(float)
    mean_values = np.mean(data, axis=0)
    std_values = np.std(data, axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(x_values, mean_values, "b-", linewidth=2, label="Mean")
    plt.fill_between(
        x_values,
        mean_values - std_values,
        mean_values + std_values,
        alpha=0.3,
        color="blue",
        label="+/- 1 Std Dev",
    )
    plt.xlabel("change ratio", fontsize=12)
    plt.ylabel("delta loss", fontsize=12)
    plt.title("Loss landscape sharpness", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    plt.tight_layout()
    output_path = f"{csv_path}.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Plot saved as '{output_path}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate loss-landscape sharpness by sampling perturbed points.",
    )
    parser.add_argument(
        "model_weights_path",
        type=str,
        nargs="?",
        default=None,
        help="file containing the model weights, can be a .model.pt file or a lmdb directory.",
    )
    parser.add_argument("-t", "--tick", type=int, help="specify the model weights tick index for a lmdb file.")
    parser.add_argument("-m", "--model", type=str, default=None, help="specify the model name")
    parser.add_argument("-d", "--dataset", type=str, default=None, help="specify the dataset name")
    parser.add_argument("--cpu", action="store_true", help="force using CPU")
    parser.add_argument(
        "-r",
        "--change_ratio",
        type=float,
        default=[0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128],
        nargs="+",
        help="specify the list of change ratio",
    )
    parser.add_argument("-s", "--sample_count", type=int, default=100, help="specify the number of samples")
    parser.add_argument("-c", "--core", type=int, default=os.cpu_count(), help="specify the number of CPU cores to use")
    parser.add_argument("--draw", type=str, help="plot the result file")

    args = parser.parse_args()

    if args.draw is not None:
        _draw_result(args.draw)
        return

    if args.model_weights_path is None:
        raise SystemExit("model_weights_path is required")

    device = Device.cpu() if args.cpu else Device.auto()

    model_weight_file_path = Path(args.model_weights_path).expanduser().resolve()
    tick = None if args.tick is None else int(args.tick)
    model_weights, model_name_from_model, dataset_name_from_model, output_name = get_model_weights_from_file(
        model_weight_file_path,
        tick,
    )

    if model_name_from_model is not None and args.model is not None:
        assert model_name_from_model == args.model, (
            f"model name mismatch {model_name_from_model} != {args.model}"
        )
    model_name = model_name_from_model if model_name_from_model is not None else args.model

    if dataset_name_from_model is not None and args.dataset is not None:
        assert dataset_name_from_model == args.dataset, (
            f"dataset name mismatch {dataset_name_from_model} != {args.dataset}"
        )
    dataset_name = dataset_name_from_model if dataset_name_from_model is not None else args.dataset

    if model_name is None or dataset_name is None:
        raise SystemExit("model and dataset must be provided either in the weight file or via CLI.")

    current_ml_setup = ml_setup.get_ml_setup_from_config(model_name, dataset_type=dataset_name)
    dataloader = _build_eval_dataloader(current_ml_setup, device, int(args.core))

    target_model: torch.nn.Module = deepcopy(current_ml_setup.model)
    target_model.load_state_dict(model_weights, strict=True)
    target_model.to(device.device)
    adapter = clone_adapter_for_model(
        current_ml_setup.adapter,
        target_model,
        criterion=current_ml_setup.criterion,
    )

    result = loss_landscape_sharpness(
        target_model,
        adapter,
        dataloader,
        change_ratio=list(args.change_ratio),
        sample_count=int(args.sample_count),
        device=device,
    )
    print(f"[info] final result is \n{result}.")

    result_df = pd.DataFrame({k: [float(x) for x in v] for k, v in result.items()})
    result_df.to_csv(f"{output_name}.loss_sharpness.csv")


if __name__ == "__main__":
    main()
