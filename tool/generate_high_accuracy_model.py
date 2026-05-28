import argparse
import concurrent.futures
import copy
import json
import logging
import math
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from PIL import Image
import lightning as L
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src.util import setup_logging, set_seed, re_initialize_model
from py_src.model_opti_save_load import save_model_state, save_optimizer_state, load_model_state_file
from py_src.ml_setup import ApplicationType, get_ml_setup_from_config, MLSetup
from py_src.complete_ml_setup import FastTrainingSetup, TransferTrainingSetup
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_dataset import DatasetType
from py_src.engine import Device, train, val
from py_src.ml_setup.dataloader_util import DataloaderConfig
import py_src.adapters as _adapters

logger = logging.getLogger("generate_high_accuracy_model")

# ---------------------------------------------------------------------------
# Opposite-direction initialisation
# ---------------------------------------------------------------------------

def initialize_model_to_opposite_direction(
    model: torch.nn.Module,
    reference_state: dict,
    reference_model_type: str,
    reference_dataset_type: Optional[str],
    arg_ml_setup: MLSetup,
    child_logger: logging.Logger,
):
    if reference_model_type is not None and reference_model_type != arg_ml_setup.model_type.name:
        raise RuntimeError(
            f"reference model type mismatch: current={arg_ml_setup.model_type.name}, "
            f"checkpoint={reference_model_type}"
        )
    if reference_dataset_type is not None and reference_dataset_type != arg_ml_setup.dataset_type.name:
        child_logger.warning(
            "opposite-init checkpoint dataset %s differs from current dataset %s",
            reference_dataset_type, arg_ml_setup.dataset_type.name,
        )

    initialized, skipped = 0, 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in reference_state:
                raise RuntimeError(f"reference checkpoint missing parameter: {name}")
            ref = reference_state[name]
            if not torch.is_tensor(ref):
                raise RuntimeError(f"reference entry {name} is not a tensor")
            if ref.shape != param.shape:
                raise RuntimeError(f"shape mismatch for {name}: model={tuple(param.shape)}, ref={tuple(ref.shape)}")
            if not param.dtype.is_floating_point or not ref.dtype.is_floating_point:
                continue
            cur_norm = param.detach().float().norm().item()
            ref_norm = ref.detach().float().norm().item()
            if cur_norm <= 0.0 or ref_norm <= 0.0:
                skipped += 1
                continue
            opposite = -ref.detach().to(device=param.device, dtype=torch.float32) / ref_norm
            param.copy_((opposite * cur_norm).to(dtype=param.dtype))
            initialized += 1

    child_logger.info(
        "initialized %d parameter tensors toward the opposite direction of %s@%s; skipped %d zero-norm",
        initialized, reference_model_type, reference_dataset_type, skipped,
    )


def maybe_enable_torch_compile(
    model: torch.nn.Module,
    adapter,
    device: Device,
    child_logger: logging.Logger,
    enable_torch_compile: bool,
) -> None:
    if not enable_torch_compile:
        child_logger.info("torch.compile disabled")
        return
    if device.device.type != "cuda":
        child_logger.info("torch.compile skipped: device=%s", device.device.type)
        return
    if not hasattr(torch, "compile"):
        child_logger.info("torch.compile unavailable in this PyTorch build")
        return
    if not isinstance(adapter, (_adapters.StandardAdapter, _adapters.DiffusionAdapter)):
        child_logger.info(
            "torch.compile skipped: adapter %s is not supported in this training path",
            type(adapter).__name__,
        )
        return

    try:
        compiled_model = torch.compile(model)
        if isinstance(compiled_model, torch.nn.Module):
            adapter._model = compiled_model # type: ignore[attr-defined]
            child_logger.info("torch.compile enabled")
        else:
            child_logger.info("torch.compile returned a non-Module wrapper; using the original module")
    except Exception as exc:
        child_logger.info("torch.compile skipped: %s", exc)


class _ProgressIterable:
    def __init__(self, iterable, progress_bar):
        self._iterable = iterable
        self._progress_bar = progress_bar

    def __iter__(self):
        for batch in self._iterable:
            yield batch
            self._progress_bar.update(1)


# ---------------------------------------------------------------------------
# Core training routine (single process, no multiprocessing)
# ---------------------------------------------------------------------------

def training_model(
    output_folder, index, number_of_models,
    arg_ml_setup: MLSetup,
    arg_use_cpu: bool,
    random_seed: int,
    arg_worker_count: int,
    arg_total_cpu_count: int,
    arg_save_format: str,
    arg_save_interval: int,
    arg_amp: bool,
    arg_compile: bool,
    arg_preset: int,
    arg_epoch_override,
    transfer_learn_model_path,
    init_model_path,
    opposite_init_model_path,
    disable_reinit: bool,
    enable_validation: bool,
):
    thread_per_process = arg_total_cpu_count // arg_worker_count
    thread_per_process = min(thread_per_process, 8)
    torch.set_num_threads(thread_per_process)

    child_logger = logging.getLogger(f"generate_high_accuracy_model.{index}")
    setup_logging(child_logger, str(index))

    if random_seed is not None:
        set_seed(random_seed)
        child_logger.info("random seed = %d", random_seed)

    device = Device.cpu() if arg_use_cpu else Device.auto()

    digit_width = len(str(number_of_models))
    model: torch.nn.Module = copy.deepcopy(arg_ml_setup.model)
    adapter = copy.deepcopy(arg_ml_setup.adapter)

    # rebind the deep-copied adapter to the deep-copied model
    if isinstance(adapter, (_adapters.StandardAdapter, _adapters.DiffusionAdapter,
                             _adapters.LightningAdapter, _adapters.CustomStepAdapter)):
        adapter._model = model # type: ignore

    num_workers = min(thread_per_process, 8)

    dataloader = arg_ml_setup.train_dataloader(DataloaderConfig(num_workers=num_workers))
    steps_per_epoch = len(dataloader) # type: ignore

    if enable_validation and arg_ml_setup.application_type==ApplicationType.classifier:
        dataloader_test = arg_ml_setup.val_dataloader(DataloaderConfig(num_workers=num_workers))
    else:
        dataloader_test = None

    epochs = arg_epoch_override  # may be None

    if arg_save_format != "none":
        ckpt_folder = os.path.join(output_folder, str(index))
        os.makedirs(ckpt_folder)
    else:
        ckpt_folder = None

    # --- initialise weights and select optimizer ---

    optimizer, lr_scheduler = None, None

    if transfer_learn_model_path is None:
        if disable_reinit:
            child_logger.info("re-initialisation disabled")
        else:
            if init_model_path is not None:
                state, init_model_type, init_dataset_type = load_model_state_file(init_model_path)
                child_logger.info("loading initial weights from %s (model=%s, dataset=%s)",
                                  init_model_path, init_model_type, init_dataset_type)
                model.load_state_dict(state)
            else:
                child_logger.info("re-initialising model")
                re_initialize_model(model)
                if opposite_init_model_path is not None:
                    ref_state, ref_model_type, ref_dataset_type = load_model_state_file(opposite_init_model_path)
                    child_logger.info("rotating init toward opposite direction of %s", opposite_init_model_path)
                    assert ref_model_type is not None, f"ref_model_type is {ref_model_type}"
                    initialize_model_to_opposite_direction(
                        model, ref_state, ref_model_type, ref_dataset_type, arg_ml_setup, child_logger
                    )
            model.to(device.device)

        child_logger.info("mode: ||||||||    TRAIN FROM INITIALIZATION    ||||||||")
        if isinstance(model, L.LightningModule):
            optimizer_lit, lr_scheduler_lit = model.configure_optimizers()  # type: ignore
            optimizer_cfg, lr_scheduler_cfg, epochs_cfg = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(
                arg_ml_setup, model, arg_preset, override_steps_per_epoch=steps_per_epoch
            )
            optimizer = optimizer_lit if optimizer_cfg is None else optimizer_cfg
            lr_scheduler = lr_scheduler_lit if lr_scheduler_cfg is None else lr_scheduler_cfg
            if epochs is None:
                epochs = epochs_cfg
        else:
            optimizer, lr_scheduler, epochs_cfg = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(
                arg_ml_setup, model, arg_preset, override_steps_per_epoch=steps_per_epoch
            )
            if epochs is None:
                epochs = epochs_cfg

    else:
        state, src_model_type, src_dataset_type = load_model_state_file(transfer_learn_model_path)
        child_logger.info("transfer learning from %s (model=%s, dataset=%s)",
                          transfer_learn_model_path, src_model_type, src_dataset_type)
        model.load_state_dict(state)
        model.to(device.device)
        child_logger.info("mode: ||||||||    TRANSFER TRAINING    ||||||||")
        src_dt = DatasetType[src_dataset_type] if src_dataset_type else None
        assert src_dt is not None, f"dataset type is None for {transfer_learn_model_path}"
        optimizer, lr_scheduler, epochs_cfg = TransferTrainingSetup.get_optimizer_lr_scheduler_epoch(
            src_dt,
            arg_ml_setup,
            model,
            arg_preset,
            override_steps_per_epoch=steps_per_epoch,
        ) # type: ignore
        if epochs is None:
            epochs = epochs_cfg

    log_csv_path = os.path.join(output_folder, f"{str(index).zfill(digit_width)}.log.csv")
    log_csv = open(log_csv_path, "w")
    log_csv.write("epoch,training_loss,training_accuracy,validation_loss,validation_accuracy,lrs\n")
    log_csv.flush()

    child_logger.info("begin training for %d epochs", epochs)
    child_logger.info("steps_per_epoch = %d", steps_per_epoch)

    if hasattr(model, "set_batches_per_epoch"):
        model.set_batches_per_epoch(len(dataloader)) # type: ignore

    maybe_enable_torch_compile(model, adapter, device, child_logger, arg_compile)

    scaler = device.make_scaler() if arg_amp else None

    for epoch in range(epochs):
        progress_enabled = arg_worker_count == 1
        progress_context = tqdm(
            total=steps_per_epoch,
            desc=f"epoch {epoch + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        ) if progress_enabled else nullcontext()
        with progress_context as progress_bar:
            train_iterable = (
                _ProgressIterable(dataloader, progress_bar)
                if progress_enabled and progress_bar is not None
                else dataloader
            )
            train_result = train(
                adapter,
                train_iterable,
                optimizer, # type: ignore
                lr_scheduler, # type: ignore
                device=device,
                scaler=scaler,
                gradient_accumulate_every=arg_ml_setup.gradient_accumulate_every,
                max_grad_norm=arg_ml_setup.max_grad_norm,
            ) # type: ignore

        lrs = [pg["lr"] for pg in optimizer.param_groups] # type: ignore

        if dataloader_test is None:
            acc_str = f"{train_result.accuracy:.4f}" if train_result.accuracy is not None else "n/a"
            child_logger.info(
                "epoch[%d] training loss=%.4f accuracy=%s lrs=%s",
                epoch, train_result.avg_loss, acc_str, lrs,
            )
            train_acc = train_result.accuracy if train_result.accuracy is not None else math.nan
            log_csv.write(f"{epoch},{train_result.avg_loss:.4e},{train_acc:.4e},{math.nan},{math.nan},{lrs}\n")
        else:
            val_result = val(adapter, dataloader_test, device=device)
            child_logger.info(
                "epoch[%d] (train) loss=%.4f acc=%s  (val) loss=%.4f acc=%s  lrs=%s",
                epoch,
                train_result.avg_loss,
                f"{train_result.accuracy:.4f}" if train_result.accuracy is not None else "n/a",
                val_result.avg_loss,
                f"{val_result.accuracy:.4f}" if val_result.accuracy is not None else "n/a",
                lrs,
            )
            train_acc = train_result.accuracy if train_result.accuracy is not None else math.nan
            val_acc = val_result.accuracy if val_result.accuracy is not None else math.nan
            log_csv.write(
                f"{epoch},{train_result.avg_loss:.4e},{train_acc:.4e},"
                f"{val_result.avg_loss:.3e},{val_acc:.4e},{lrs}\n"
            )
        log_csv.flush()

        if ckpt_folder is not None and epoch % arg_save_interval == 0:
            ckpt_path = os.path.join(ckpt_folder, f"epoch{epoch}.pt")
            save_model_state(
                ckpt_path, model.state_dict(),
                arg_ml_setup.model_type.name, arg_ml_setup.dataset_type.name,
            )

        # DDPM: generate sample images every 10 epochs
        if arg_ml_setup.application_type == ApplicationType.diffusion and epoch%10 == 0:
            if arg_ml_setup.difussion_generate_sample is not None:
                sample_count = 10
                with torch.no_grad():
                    arg_ml_setup.difussion_generate_sample(model, output_folder, epoch, device.device, sample_count)

    child_logger.info("training complete")
    log_csv.flush()
    log_csv.close()

    model_out = os.path.join(output_folder, f"{str(index).zfill(digit_width)}.model.pt")
    opt_out = os.path.join(output_folder, f"{str(index).zfill(digit_width)}.optimizer.pt")
    save_model_state(model_out, model.state_dict(), arg_ml_setup.model_type.name, arg_ml_setup.dataset_type.name)
    save_optimizer_state(opt_out, optimizer.state_dict(), arg_ml_setup.model_type.name, arg_ml_setup.dataset_type.name) # type: ignore

    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description="Generate high-accuracy model checkpoints")
    parser.add_argument("-n", "--number_of_models", type=int, default=1)
    parser.add_argument("-c", "--core", type=int, default=os.cpu_count(),
                        help="total number of CPU cores available")
    parser.add_argument("-w", "--worker", type=int, default=1,
                        help="number of models to train in parallel")
    parser.add_argument("-m", "--model_type", type=str, default="lenet5")
    parser.add_argument("-d", "--dataset_type", type=str, default="default")
    parser.add_argument("--cpu", action="store_true", help="force CPU training")
    parser.add_argument("--dali", action=argparse.BooleanOptionalAction, default=False, help="use NVIDIA DALI for ImageNet dataloading")
    parser.add_argument("--dali_device_id", type=int, default=0, help="CUDA device id used by DALI pipelines")
    parser.add_argument("-o", "--output_folder_name", default=None)
    parser.add_argument("--save_format", type=str, default="none", choices=["none", "file"],
                        help="save per-epoch checkpoints (file) or skip (none)")
    parser.add_argument("--save_interval", type=int, default=1, help="checkpoint every N epochs")
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable torch.compile on CUDA when supported",
    )
    parser.add_argument("-s", "--random_seed", type=int, default=None)
    parser.add_argument("-i", "--start_index", type=int, default=0)
    parser.add_argument("-P", "--preset", type=int, default=1, help="training hyperparameter preset index")
    parser.add_argument("-e", "--epoch", type=int, default=None, help="override epoch count")
    parser.add_argument("-tl", "--transfer_learn", type=str, default=None,
                        help="checkpoint path for transfer learning")
    parser.add_argument("-init", "--initial_model", type=str, default=None,
                        help="checkpoint path to use as initial weights")
    parser.add_argument("--opposite_init_model", type=str, default=None,
                        help="checkpoint whose opposite direction is used after re-init")
    parser.add_argument("--disable_reinit", action="store_true", help="skip weight re-initialisation")
    parser.add_argument("--enable_eval", action="store_true", help="evaluate on validation set each epoch")

    args = parser.parse_args()
    if args.cpu and args.dali:
        parser.error("--dali requires CUDA; do not combine it with --cpu")

    number_of_models = args.number_of_models
    worker_count = args.worker
    total_cpu_cores = args.core

    setup_logging(logger, "main")
    logger.info("logging ready")

    if args.opposite_init_model is not None:
        if args.transfer_learn is not None:
            raise RuntimeError("--opposite_init_model cannot be combined with --transfer_learn")
        if args.initial_model is not None:
            raise RuntimeError("--opposite_init_model cannot be combined with --initial_model")
        if args.disable_reinit:
            raise RuntimeError("--opposite_init_model requires re-initialisation; cannot use with --disable_reinit")
        if not os.path.exists(args.opposite_init_model):
            raise FileNotFoundError(f"{args.opposite_init_model} does not exist")

    current_ml_setup = get_ml_setup_from_config(
        args.model_type,
        args.dataset_type,
        args.preset,
        use_dali=args.dali,
        dali_device_id=args.dali_device_id,
    )
    logger.info("model: %s  dataset: %s", current_ml_setup.model_type.name, current_ml_setup.dataset_type.name)

    if args.output_folder_name is None:
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_folder_path = os.path.join(os.curdir, f"{os.path.basename(__file__)}_{time_str}")
    else:
        output_folder_path = os.path.join(os.curdir, args.output_folder_name)
    os.mkdir(output_folder_path)

    info = {
        "model_type": current_ml_setup.model_type.name,
        "dataset_type": current_ml_setup.dataset_type.name,
        "model_count": number_of_models,
        "generated_by_cpu": args.cpu,
        "use_dali": args.dali,
        "dali_device_id": args.dali_device_id,
        "torch_compile": args.compile,
        "opposite_init_model": args.opposite_init_model,
    }
    with open(os.path.join(output_folder_path, "info.json"), "w") as f:
        json.dump(info, f)

    if worker_count > number_of_models:
        worker_count = number_of_models

    task_args = [
        (output_folder_path, i, number_of_models,
         current_ml_setup, args.cpu, args.random_seed,
         worker_count, total_cpu_cores,
         args.save_format, args.save_interval, args.amp, args.compile, args.preset, args.epoch,
         args.transfer_learn, args.initial_model, args.opposite_init_model,
         args.disable_reinit, args.enable_eval)
        for i in range(args.start_index, args.start_index + number_of_models)
    ]

    if worker_count == 1:
        for task in task_args:
            training_model(*task)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(training_model, *task) for task in task_args]
            for future in concurrent.futures.as_completed(futures):
                future.result()
