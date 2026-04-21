#!/usr/bin/env python3

import argparse
import io
import logging
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional

import lmdb
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src import ml_setup, model_opti_save_load
from py_src.adapters import clone_adapter_for_model
from py_src.engine import Device, val as engine_val
from py_src.ml_setup.dataloader_util import DataloaderConfig, build_dataloader, move_batch_to_device


def setup_logger(verbosity: int) -> logging.Logger:
    level = logging.INFO if verbosity <= 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("relative_loss_landscape_sharpness")


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
        model_weights, model_name, dataset_name = model_opti_save_load.load_model_state_file(str(path))
        output_name = str(path)
    elif path.is_dir():
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
                print(f"[error] tick {lmdb_index} not in lmdb, all ticks: {sorted(all_ticks)}", file=sys.stderr)
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


def find_last_linear(model: nn.Module) -> nn.Linear:
    last_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is None:
        raise TypeError("No nn.Linear found in model; cannot compute relative loss-landscape sharpness.")
    return last_linear


class _CaptureInputHook:
    def __init__(self):
        self.x = None

    def __call__(self, module, inputs, output):
        self.x = inputs[0]


def _is_cross_entropy_mean(criterion: Optional[nn.Module]) -> bool:
    if not isinstance(criterion, nn.CrossEntropyLoss):
        return False
    return getattr(criterion, "reduction", "mean") == "mean"


def _extract_inputs_and_targets(batch):
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError("Expected batches shaped like (inputs, targets, ...).")
    return batch[0], batch[1]


@torch.inference_mode()
def relative_flatness_kappa_tr_cross_entropy(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    *,
    device: Device,
    max_batches: Optional[int] = None,
    log_every: int = 10,
    logger: Optional[logging.Logger] = None,
) -> float:
    if logger is None:
        logger = logging.getLogger("relative_loss_landscape_sharpness")

    if not _is_cross_entropy_mean(criterion):
        raise TypeError(
            "This fast kappa implementation supports nn.CrossEntropyLoss(reduction='mean') only.",
        )

    model.eval()
    model.to(device.device)

    head = find_last_linear(model)
    weight = head.weight.detach().to(device.device)

    logger.info(
        "Using final head: %s (out_features=%s, in_features=%s)",
        head.__class__.__name__,
        head.out_features,
        head.in_features,
    )

    logger.info("Building Gram matrix G = W W^T ...")
    gram = weight @ weight.t()
    diag_gram = torch.diag(gram)

    capture = _CaptureInputHook()
    handle = head.register_forward_hook(capture)

    total = 0.0
    total_count = 0
    used_batches = 0

    try:
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and max_batches > 0 and batch_idx >= max_batches:
                break

            batch = move_batch_to_device(batch, device.device)
            inputs, _targets = _extract_inputs_and_targets(batch)

            capture.x = None
            logits = model(inputs)
            phi = capture.x
            if phi is None:
                raise RuntimeError(
                    "Failed to capture penultimate features via hook. "
                    "The selected last nn.Linear might not be used in forward.",
                )

            probabilities = torch.softmax(logits, dim=1)
            phi_norm2 = (phi * phi).sum(dim=1)
            probabilities_gram = probabilities @ gram
            p_g_p = (probabilities_gram * probabilities).sum(dim=1)
            diag_term = (probabilities * diag_gram).sum(dim=1)
            trace_term = diag_term - p_g_p
            contribution = (phi_norm2 * trace_term).sum()

            batch_size = inputs.size(0)
            total += float(contribution.item())
            total_count += batch_size
            used_batches += 1

            if log_every > 0 and ((batch_idx + 1) % log_every == 0):
                logger.info(
                    "Progress: batch %s%s, samples=%s, running kappa=%.6e",
                    batch_idx + 1,
                    f"/{max_batches}" if (max_batches is not None and max_batches > 0) else "",
                    total_count,
                    total / max(total_count, 1),
                )
    finally:
        handle.remove()

    kappa = total / max(total_count, 1)
    logger.info("Done. Used batches=%s, samples=%s, kappa=%.6e", used_batches, total_count, kappa)
    return float(kappa)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Relative loss-landscape sharpness kappa^phi_Tr(w) for classifiers with a final Linear head. "
            "Uses engine.val() for the baseline evaluation path and an analytic Hessian trace formula "
            "for the final relative-flatness metric."
        ),
    )
    parser.add_argument(
        "model_weights_path",
        type=str,
        nargs="?",
        default=None,
        help="file containing model weights, can be a .model.pt file or an lmdb directory.",
    )
    parser.add_argument("-t", "--tick", type=int, help="model weights tick index for lmdb mode.")
    parser.add_argument("-m", "--model", type=str, default=None, help="model name")
    parser.add_argument("-d", "--dataset", type=str, default=None, help="dataset name")
    parser.add_argument("--cpu", action="store_true", help="force using CPU")
    parser.add_argument("--max-batches", type=int, default=100, help="use only first N batches (<=0 means all)")
    parser.add_argument("--log-every", type=int, default=10, help="log progress every N batches (<=0 disables)")
    parser.add_argument("-c", "--core", type=int, default=os.cpu_count(), help="number of CPU cores to use")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity")
    parser.add_argument(
        "-P",
        "--torch_preset_version",
        type=int,
        default=0,
        help="LLR2 preset version forwarded to get_ml_setup_from_config().",
    )

    args = parser.parse_args()
    logger = setup_logger(args.verbose)

    if args.model_weights_path is None:
        raise SystemExit("model_weights_path is required")

    device = Device.cpu() if args.cpu else Device.auto()
    logger.info("Device: %s", device.device)

    model_weight_file_path = Path(args.model_weights_path).expanduser().resolve()
    tick = None if args.tick is None else int(args.tick)
    model_weights, model_name_from_file, dataset_name_from_file, output_name = get_model_weights_from_file(
        model_weight_file_path,
        tick,
    )

    model_name = model_name_from_file if model_name_from_file is not None else args.model
    dataset_name = dataset_name_from_file if dataset_name_from_file is not None else args.dataset
    if model_name is None or dataset_name is None:
        raise SystemExit("model and dataset must be provided either in the weight file or via --model/--dataset")

    if model_name_from_file is not None and args.model is not None:
        assert model_name_from_file == args.model, (
            f"model name mismatch {model_name_from_file} != {args.model}"
        )
    if dataset_name_from_file is not None and args.dataset is not None:
        assert dataset_name_from_file == args.dataset, (
            f"dataset name mismatch {dataset_name_from_file} != {args.dataset}"
        )

    logger.info("Model: %s, Dataset: %s", model_name, dataset_name)

    current_ml_setup = ml_setup.get_ml_setup_from_config(
        model_name,
        dataset_type=dataset_name,
        preset=int(args.torch_preset_version),
    )
    dataloader = _build_eval_dataloader(current_ml_setup, device, int(args.core))

    target_model: nn.Module = deepcopy(current_ml_setup.model)
    target_model.load_state_dict(model_weights, strict=True)
    target_model.to(device.device)

    adapter = clone_adapter_for_model(
        current_ml_setup.adapter,
        target_model,
        criterion=current_ml_setup.criterion,
    )
    baseline_result = engine_val(adapter, dataloader, device=device)
    logger.info(
        "Baseline eval via engine.val(): loss=%.6e accuracy=%s",
        baseline_result.avg_loss,
        "n/a" if baseline_result.accuracy is None else f"{baseline_result.accuracy:.6f}",
    )

    criterion = current_ml_setup.criterion
    if criterion is None:
        raise TypeError("This script requires MLSetup.criterion to be defined.")

    max_batches = None if (args.max_batches is None or args.max_batches <= 0) else int(args.max_batches)
    logger.info("Computing kappa^phi_Tr(w) ...")
    kappa = relative_flatness_kappa_tr_cross_entropy(
        model=target_model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        max_batches=max_batches,
        log_every=int(args.log_every),
        logger=logger,
    )

    log_info = (
        f"Relative loss-landscape sharpness kappa^phi_Tr(w): {kappa:.6e} "
        f"(batch_size={current_ml_setup.default_batch_size}, max_batches={max_batches}) "
        f"model_type={current_ml_setup.model_type.name}, dataset_type={current_ml_setup.dataset_type.name}\n"
    )
    print(log_info, end="")

    output_path = f"{output_name}.relative_loss_landscape_sharpness.log"
    with open(output_path, "a", encoding="utf-8") as outfile:
        outfile.write(log_info)
    logger.info("Wrote: %s", output_path)


if __name__ == "__main__":
    main()
