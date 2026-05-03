from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Optional

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src.model_opti_save_load import load_model_state_file, save_model_state
from py_src.ml_setup_dataset.dataset_modular import ArithmeticDataset, ArithmeticIterator
from py_src.ml_setup.dataloader_util import cache_dataloader_on_device
from py_src.ml_setup_model.transformer_for_grokking import TransformerForGrokking
from py_src.service import record_model_stat
import py_src.service.record_weights_difference as record_weights_difference
from py_src.util import re_initialize_model, set_seed, setup_logging


logger = logging.getLogger("generate_grokking")
SPEED_REPORT_INTERVAL = 100
SPLIT_CHOICES = [
    "random",
    "chessboard",
    "updown",
    "leftright",
    "tl_to_br",
    "tr_to_bl",
    "interlace_row",
    "interlace_col",
    "chessboard_random",
]


@dataclass
class GrokkingStepOutput:
    loss_value: float
    sample_count: int
    correct_count: int
    correct_location: torch.Tensor


class GrokkingParameters:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.dataset_name = None
        self.logger = None
        self.output_folder_path = None
        self.early_stop = None
        self.weight_decay = None
        self.learning_rate = None
        self.warmup_epoch = None
        self.total_epoch = None
        self.min_lr = None
        self.ineffective_train_stop = None
        self.ineffective_train_stop_window = None
        self.high_loss_train_stop = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.save_format = None
        self.save_interval = None
        self.save_name = None
        self.record_weight_norm_interval = None
        self.early_stop_train_accuracy = 1.00
        self.early_stop_val_accuracy = 1.00
        self.disable_validation = False

    def set_ml_env(self, model, model_name, dataset_name, tokenizer):
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer

    def set_dataloader(self, train_dataloader, val_dataloader):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def set_ml_hyperparameter(self, learning_rate=1e-3, weight_decay=0.0, min_lr=1e-4, warmup_epoch=10, total_epoch=150000):
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.min_lr = float(min_lr)
        self.warmup_epoch = int(warmup_epoch)
        self.total_epoch = int(total_epoch)

    def set_env(self, output_path, early_stop, logger=None):
        self.output_folder_path = output_path
        self.logger = logger
        self.early_stop = early_stop

    def set_early_stop_thresholds(self, train_accuracy=0.96, val_accuracy=0.96):
        self.early_stop_train_accuracy = float(train_accuracy)
        self.early_stop_val_accuracy = float(val_accuracy)

    def set_disable_validation(self, enabled=True):
        self.disable_validation = enabled

    def set_model_save(self, save_name, save_format="none", save_interval=500, record_weight_norm_interval=100):
        self.save_name = save_name
        self.save_format = save_format
        self.save_interval = save_interval
        self.record_weight_norm_interval = record_weight_norm_interval

    def set_ineffective_train_stop(self, enabled=True, window=1000):
        self.ineffective_train_stop = enabled
        self.ineffective_train_stop_window = window

    def set_high_loss_train_stop(self, enabled=True):
        self.high_loss_train_stop = enabled


def normalize_expression(expression: str, modulus: int) -> str:
    expr = expression.replace(" ", "")
    if "_mod_" not in expr and ("x" in expr or "y" in expr):
        return f"{expr}_mod_{modulus}"
    return expr


def loading_dataset_from(path: str, modulus: Optional[int] = None):
    match = re.search(r"modulus(\d+)", Path(path).name)
    actual_modulus = int(match.group(1)) if match is not None else modulus
    if actual_modulus is None:
        raise ValueError("Could not infer modulus from dataset path; pass --modulus or use a modulus{N} folder name")
    train_dataset = ArithmeticDataset.load_from_file(
        f"{path}/train.txt",
        actual_modulus,
        name=path,
        train=True,
        tokenizer_path=f"{path}/tokenizer.txt",
    )
    val_dataset = ArithmeticDataset.load_from_file(
        f"{path}/val.txt",
        actual_modulus,
        name=path,
        train=False,
        tokenizer_path=f"{path}/tokenizer.txt",
    )
    return train_dataset, val_dataset


def generate_dataset(output_folder_path, train_pct, expression, modulus, train_split_type, operand_length):
    normalized_expression = normalize_expression(expression, modulus)
    train_dataset, val_dataset = ArithmeticDataset.splits(
        train_pct=train_pct,
        operator=normalized_expression,
        train_split_type=train_split_type,
        modulus=modulus,
        operand_length=operand_length,
    )
    name = train_dataset.name
    train_dataset.save_to_file(os.path.join(output_folder_path, name, "train.txt"))
    val_dataset.save_to_file(os.path.join(output_folder_path, name, "val.txt"))
    train_dataset.tokenizer.save_tokens(os.path.join(output_folder_path, name, "tokenizer.txt"))
    return train_dataset, val_dataset


def default_batch_size(dataset):
    return ArithmeticIterator.calculate_batchsize(len(dataset), batchsize_hint=0)


def build_grokking_model(dataset, *, n_layers=None, n_heads=None, d_model=None, context_len=None, position_encoding=None):
    min_context_len = int(dataset.data.shape[1] - 1)
    actual_context_len = max(50, min_context_len) if context_len is None else max(context_len, min_context_len)
    return TransformerForGrokking(
        n_layers=2 if n_layers is None else n_layers,
        n_heads=4 if n_heads is None else n_heads,
        d_model=128 if d_model is None else d_model,
        max_context_len=actual_context_len,
        vocab_len=len(dataset.tokenizer),
        trainable_position_encoding=position_encoding == "trainable",
    )


def _check_ineffective_train_stop(train_loss_history: deque, current_epoch: int, *, window=1000, min_epoch=1000, loss_threshold=1.5, cv_threshold=0.02) -> bool:
    if current_epoch < min_epoch or len(train_loss_history) < window:
        return False
    history = list(train_loss_history)
    mean = sum(history) / len(history)
    if mean <= loss_threshold:
        return False
    variance = sum((value - mean) ** 2 for value in history) / len(history)
    return ((variance ** 0.5) / mean) < cv_threshold


def _check_loss_above_initial(loss_above_initial_counter, initial_loss, current_loss, current_epoch, total_epoch, *, initial_loss_multiplier=1.1, lookback_ratio=0.1):
    threshold = initial_loss * initial_loss_multiplier
    loss_above_initial_counter = loss_above_initial_counter + 1 if current_loss > threshold else 0
    required = max(1, int(total_epoch * lookback_ratio))
    return (current_epoch >= required and loss_above_initial_counter >= required), loss_above_initial_counter


def grokking_step(batch_index, batch, model, optimizer, lr_scheduler, tokenizer, *, train=False, scaler=None):
    del batch_index
    if train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    x = batch["text"]
    y = batch["target"]
    use_amp = train and x.device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    autocast_context = torch.autocast(device_type=x.device.type, dtype=amp_dtype) if use_amp else contextlib.nullcontext()
    grad_context = contextlib.nullcontext() if train else torch.inference_mode()
    with grad_context:
        with autocast_context:
            y_hat, _, _ = model(x=x)
            y_hat = y_hat.transpose(-2, -1)
            eq_token_index = tokenizer.stoi["="]
            eq_position_tensor = torch.nonzero(y[0, :] == eq_token_index, as_tuple=False)
            eq_position = int(eq_position_tensor.squeeze())
            y_rhs = y[..., eq_position + 1:]
            y_hat_rhs = y_hat[..., eq_position + 1:]
            loss = torch.nn.functional.cross_entropy(y_hat_rhs, y_rhs, reduction="mean")

    y_hat_max = y_hat_rhs.argmax(dim=-2)
    row_accuracy = (y_hat_max == y_rhs).all(dim=-1)
    correct_count = int(row_accuracy.int().sum().item())

    if train and optimizer is not None:
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

    return GrokkingStepOutput(
        loss_value=float(loss.item()),
        sample_count=int(y.shape[0]),
        correct_count=correct_count,
        correct_location=row_accuracy.detach().cpu(),
    )


def _write_final_correct_position(output_folder_path, tokenizer, model, train_batches, val_batches):
    def token_to_repr(token_id: int) -> str:
        token = tokenizer.itos[token_id]
        try:
            return str(int(token))
        except ValueError:
            return token

    rows = []
    for batch_idx, batch in enumerate(chain(train_batches, val_batches)):
        output = grokking_step(batch_idx, batch, model, None, None, tokenizer, train=False)
        lhs_tokens = batch["text"][:, 1].detach().cpu().tolist()
        rhs_tokens = batch["text"][:, 3].detach().cpu().tolist()
        for lhs, rhs, correct in zip(lhs_tokens, rhs_tokens, output.correct_location.tolist()):
            rows.append((token_to_repr(lhs), token_to_repr(rhs), int(bool(correct))))

    rows.sort(key=lambda item: (item[0], item[1]))
    with open(os.path.join(output_folder_path, "final_correct_position.csv"), "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["lhs", "rhs", "correct?"])
        writer.writerows(rows)


def train_grokking(parameters: GrokkingParameters):
    assert parameters.model is not None
    assert parameters.learning_rate is not None
    assert parameters.weight_decay is not None
    assert parameters.total_epoch is not None
    assert parameters.warmup_epoch is not None
    assert parameters.min_lr is not None
    assert parameters.train_dataloader is not None
    assert parameters.disable_validation or parameters.val_dataloader is not None
    assert parameters.output_folder_path is not None
    assert parameters.tokenizer is not None
    assert parameters.model_name is not None
    assert parameters.dataset_name is not None
    assert parameters.save_name is not None
    assert parameters.save_format is not None
    assert parameters.save_interval is not None
    assert parameters.record_weight_norm_interval is not None

    optimizer = torch.optim.AdamW(
        parameters.model.parameters(),
        weight_decay=parameters.weight_decay,
        lr=parameters.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=parameters.warmup_epoch)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=parameters.total_epoch - parameters.warmup_epoch,
        eta_min=parameters.min_lr,
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[parameters.warmup_epoch])

    device = next(parameters.model.parameters()).device
    if parameters.logger is not None:
        parameters.logger.info("caching train/val batches on %s", device)
    cached_train = cache_dataloader_on_device(parameters.train_dataloader, device)
    if parameters.disable_validation:
        cached_val = []
        if parameters.logger is not None:
            parameters.logger.info("validation evaluation disabled; train metrics will be mirrored into val columns")
    else:
        assert parameters.val_dataloader is not None
        cached_val = cache_dataloader_on_device(parameters.val_dataloader, device)
    if parameters.logger is not None:
        parameters.logger.info("cached %d train batches, %d val batches", len(cached_train), len(cached_val))

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16)) if use_amp else None  # type: ignore

    record_model_service = None
    if parameters.save_format != "none":
        record_model_service = record_model_stat.ModelStatRecorder(1, parameters.model_name, parameters.dataset_name)
        model_state_path = os.path.join(parameters.output_folder_path, parameters.save_name)
        os.makedirs(model_state_path, exist_ok=True)
        record_model_service.initialize_without_runtime_parameters(
            [0],
            model_state_path,
            save_format=parameters.save_format,
            lmdb_db_name=parameters.save_name,
        )

    distance_to_origin_service = record_weights_difference.ServiceDistanceToOriginRecorder(1, [0])
    distance_to_origin_service.initialize_without_runtime_parameters({0: parameters.model.state_dict()}, parameters.output_folder_path, logger=parameters.logger)

    log_csv_path = os.path.join(parameters.output_folder_path, f"{parameters.save_name}.log.csv")
    log_csv_file = open(log_csv_path, "w", encoding="utf-8")
    log_csv_file.write("epoch,training_loss,training_accuracy,validation_loss,validation_accuracy,lrs\n")
    log_csv_file.flush()

    speed_window_start_time = time.time()
    speed_window_start_epoch = 0
    train_loss_history = deque(maxlen=parameters.ineffective_train_stop_window if parameters.ineffective_train_stop else 1)
    initial_train_loss = None
    loss_above_initial_counter = 0

    for epoch in range(parameters.total_epoch):
        train_loss_sum = 0.0
        train_count = 0
        train_correct = 0

        if epoch == 0 and record_model_service is not None:
            record_model_service.trigger_without_runtime_parameters(-1, [0], [parameters.model.state_dict()])

        for batch_idx, batch in enumerate(cached_train):
            output = grokking_step(
                batch_idx,
                batch,
                parameters.model,
                optimizer,
                lr_scheduler,
                parameters.tokenizer,
                train=True,
                scaler=scaler,
            )
            train_loss_sum += output.loss_value * output.sample_count
            train_count += output.sample_count
            train_correct += output.correct_count

        train_accuracy = train_correct / train_count
        train_loss = train_loss_sum / train_count
        if parameters.disable_validation:
            val_accuracy = train_accuracy
            val_loss = train_loss
        else:
            total_val_loss = 0.0
            val_correct = 0
            val_count = 0
            for batch_idx, batch in enumerate(cached_val):
                output = grokking_step(batch_idx, batch, parameters.model, None, None, parameters.tokenizer, train=False)
                total_val_loss += output.loss_value * output.sample_count
                val_correct += output.correct_count
                val_count += output.sample_count
            val_accuracy = val_correct / val_count
            val_loss = total_val_loss / val_count
        lrs = [param_group["lr"] for param_group in optimizer.param_groups]

        if parameters.logger is not None:
            parameters.logger.info(
                "epoch[%d] loss,accuracy= (train) %.4f,%.4f (val) %.4f,%.4f lrs=%s",
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
                lrs,
            )
        log_csv_file.write(f"{epoch},{train_loss:.4e},{train_accuracy:.4e},{val_loss:.3e},{val_accuracy:.4e},{lrs}\n")
        if epoch % 100 == 0:
            log_csv_file.flush()

        epochs_since_report = epoch - speed_window_start_epoch + 1
        if epochs_since_report >= SPEED_REPORT_INTERVAL:
            elapsed = time.time() - speed_window_start_time
            epochs_remaining = parameters.total_epoch - epoch - 1
            time_per_epoch = elapsed / epochs_since_report
            eta_seconds = time_per_epoch * epochs_remaining
            if parameters.logger is not None:
                parameters.logger.info(
                    "[Speed] last %d epochs: %.1fs (%.2fs/epoch) | remaining: %d epochs, ETA: %s",
                    epochs_since_report,
                    elapsed,
                    time_per_epoch,
                    epochs_remaining,
                    time.strftime("%H:%M:%S", time.gmtime(eta_seconds)),
                )
            speed_window_start_time = time.time()
            speed_window_start_epoch = epoch + 1

        if (
            parameters.early_stop
            and train_accuracy >= parameters.early_stop_train_accuracy
            and val_accuracy >= parameters.early_stop_val_accuracy
        ):
            break

        train_loss_history.append(float(train_loss))
        if parameters.ineffective_train_stop and _check_ineffective_train_stop(
            train_loss_history,
            epoch,
            window=parameters.ineffective_train_stop_window or 1000,
        ):
            if parameters.logger is not None:
                parameters.logger.info(
                    "ineffective_train_stop triggered at epoch %d: mean loss over last %d epochs = %.4f",
                    epoch,
                    parameters.ineffective_train_stop_window,
                    sum(train_loss_history) / len(train_loss_history),
                )
            break

        if initial_train_loss is None:
            initial_train_loss = float(train_loss)
        if parameters.high_loss_train_stop:
            should_stop, loss_above_initial_counter = _check_loss_above_initial(
                loss_above_initial_counter,
                initial_train_loss,
                float(train_loss),
                epoch,
                parameters.total_epoch,
            )
            if should_stop:
                if parameters.logger is not None:
                    parameters.logger.info("loss_above_initial_stop triggered at epoch %d", epoch)
                break

        model_stat = None
        if record_model_service is not None and epoch % parameters.save_interval == 0:
            model_stat = parameters.model.state_dict()
            record_model_service.trigger_without_runtime_parameters(epoch, [0], [model_stat])
        if epoch % parameters.record_weight_norm_interval == 0:
            model_stat = parameters.model.state_dict() if model_stat is None else model_stat
            distance_to_origin_service.trigger_without_runtime_parameters(epoch, {0: model_stat})

    log_csv_file.flush()
    log_csv_file.close()
    _write_final_correct_position(parameters.output_folder_path, parameters.tokenizer, parameters.model, cached_train, cached_val)
    save_model_state(
        os.path.join(parameters.output_folder_path, f"{parameters.save_name}.model.pt"),
        parameters.model.state_dict(),
        parameters.model_name,
        parameters.dataset_name,
    )


def initialize_model_for_training(model, *, transfer_learn_path, init_model_path, disable_reinit, device):
    if transfer_learn_path is not None:
        state, model_name, dataset_name = load_model_state_file(transfer_learn_path)
        logger.info("load model weights for transfer learning, original model type: %s, dataset type: %s", model_name, dataset_name)
        model.load_state_dict(state)
    elif init_model_path is not None:
        state, model_name, dataset_name = load_model_state_file(init_model_path)
        logger.info("load model weights for initialization, original model type: %s, dataset type: %s", model_name, dataset_name)
        model.load_state_dict(state)
    elif disable_reinit:
        logger.info("re-initialize model is disabled")
    else:
        logger.info("re-initialize model")
        re_initialize_model(model)
    model.to(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate grokking models")
    parser.add_argument("-n", "--number_of_models", type=int, default=1)
    parser.add_argument("-c", "--core", type=int, default=4)
    parser.add_argument("-w", "--worker", type=int, default=1, help="kept for CLI compatibility; training is sequential")
    parser.add_argument("-o", "--output_folder_name", default=None)
    parser.add_argument("-m", "--model_type", type=str, default="transformer_for_grokking")
    parser.add_argument("-dpath", "--dataset_path", type=str, default=None)
    parser.add_argument("-dexp", "--dataset_exp", type=str, default=None)
    parser.add_argument("--modulus", type=int, default=97)
    parser.add_argument("-tp", "--train_pct", type=float, default=50)
    parser.add_argument("-st", "--split_type", type=str, default="random", choices=SPLIT_CHOICES)
    parser.add_argument("-ol", "--operand_length", type=int, default=None)
    parser.add_argument("-lr", "--learning_rate", type=float, default=None)
    parser.add_argument("-minlr", "--min_lr", type=float, default=None)
    parser.add_argument("-epoch", "--epoch", type=int, default=None)
    parser.add_argument("-wd", "--weight_decay", type=float, default=None)
    parser.add_argument("-bs", "--batchsize", type=int, default=None)
    parser.add_argument("--save_format", type=str, default="none", choices=["none", "file", "lmdb"])
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--record_weight_norm", type=int, default=None)
    parser.add_argument("-s", "--random_seed", type=int, default=None)
    parser.add_argument("-i", "--start_index", type=int, default=0)
    parser.add_argument("-t", "--transfer_learn", type=str, default=None)
    parser.add_argument("--init_model", type=str, default=None)
    parser.add_argument("--disable_reinit", action="store_true")
    parser.add_argument("--inverse_train_val", action="store_true")
    parser.add_argument("--disable_validation", action="store_true")
    parser.add_argument("--m_nlayer", default=None, type=int)
    parser.add_argument("--m_n_heads", default=None, type=int)
    parser.add_argument("--m_d_model", default=None, type=int)
    parser.add_argument("--m_context_len", default=None, type=int)
    parser.add_argument("--m_pos_encoding", default=None, type=str, choices=["default", "trainable"])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_type != "transformer_for_grokking":
        raise ValueError("generate_grokking.py only supports --model_type transformer_for_grokking")

    torch.set_num_threads(max(1, min(args.core, 8)))
    setup_logging(logger, "main")
    logger.info("logging setup complete")
    if args.worker != 1:
        logger.warning("--worker is accepted for CLI compatibility but this port trains sequentially")
    if args.random_seed is not None:
        set_seed(args.random_seed)
        logger.info("random seed = %d", args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device = %s", device)
    if args.output_folder_name is None:
        output_folder_path = os.path.join(os.curdir, f"generate_grokking_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}")
    else:
        output_folder_path = os.path.join(os.curdir, args.output_folder_name)
    os.makedirs(output_folder_path, exist_ok=False)
    with open(os.path.join(output_folder_path, "command.txt"), "w", encoding="utf-8") as outfile:
        outfile.write(" ".join([sys.executable, *sys.argv]))

    if args.dataset_path is not None:
        train_ds, val_ds = loading_dataset_from(args.dataset_path, modulus=args.modulus)
    else:
        if args.dataset_exp is None:
            raise ValueError("--dataset_exp is required when --dataset_path is not provided")
        train_ds, val_ds = generate_dataset(output_folder_path, args.train_pct, args.dataset_exp, args.modulus, args.split_type, args.operand_length)

    if args.inverse_train_val:
        logger.info("inverse training/validation mode enabled; forcing number_of_models to 2")
        args.number_of_models = 2
    batch_size = default_batch_size(train_ds) if args.batchsize is None else args.batchsize
    logger.info("batch size = %s", batch_size)

    run_indices = list(range(args.start_index, args.start_index + args.number_of_models))
    digit_width = max(1, len(str(run_indices[-1]))) if run_indices else 1
    inverse_initial_state = None

    for run_offset, run_index in enumerate(run_indices):
        save_name = str(run_index).zfill(digit_width)
        model = build_grokking_model(
            train_ds,
            n_layers=args.m_nlayer,
            n_heads=args.m_n_heads,
            d_model=args.m_d_model,
            context_len=args.m_context_len,
            position_encoding=args.m_pos_encoding,
        )

        if args.inverse_train_val:
            if run_offset == 0:
                initialize_model_for_training(model, transfer_learn_path=args.transfer_learn, init_model_path=args.init_model, disable_reinit=args.disable_reinit, device=device)
                inverse_initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                train_dl = ArithmeticIterator(train_ds, device, batchsize_hint=batch_size)
                val_dl = ArithmeticIterator(val_ds, device, batchsize_hint=batch_size, shuffle=False)
                output_folder_path_current = os.path.join(output_folder_path, "train")
                logger.info("inverse train/val mode: currently training on train partition")
            elif run_offset == 1:
                assert inverse_initial_state is not None
                model.load_state_dict(inverse_initial_state)
                model.to(device)
                train_dl = ArithmeticIterator(val_ds, device, batchsize_hint=batch_size)
                val_dl = ArithmeticIterator(train_ds, device, batchsize_hint=batch_size, shuffle=False)
                output_folder_path_current = os.path.join(output_folder_path, "val")
                logger.info("inverse train/val mode: currently training on val partition")
            else:
                raise NotImplementedError("inverse_train_val only supports exactly two runs")
        else:
            initialize_model_for_training(model, transfer_learn_path=args.transfer_learn, init_model_path=args.init_model, disable_reinit=args.disable_reinit, device=device)
            train_dl = ArithmeticIterator(train_ds, device, batchsize_hint=batch_size)
            val_dl = ArithmeticIterator(val_ds, device, batchsize_hint=batch_size, shuffle=False)
            output_folder_path_current = output_folder_path

        os.makedirs(output_folder_path_current, exist_ok=True)
        total_epoch = 150000 if args.epoch is None else args.epoch
        record_weight_norm_interval = max(1, total_epoch // 2000) if args.record_weight_norm is None else args.record_weight_norm

        params = GrokkingParameters()
        params.set_env(output_folder_path_current, False, logger=logger)
        params.set_ml_env(model, args.model_type, train_ds.name, train_ds.tokenizer)
        params.set_ml_hyperparameter(
            learning_rate=1e-3 if args.learning_rate is None else args.learning_rate,
            weight_decay=0.0 if args.weight_decay is None else args.weight_decay,
            min_lr=1e-4 if args.min_lr is None else args.min_lr,
            warmup_epoch=10,
            total_epoch=total_epoch,
        )
        params.set_dataloader(train_dl, val_dl)
        if args.disable_validation:
            params.set_disable_validation()
        params.set_model_save(save_name, save_format=args.save_format, save_interval=args.save_interval, record_weight_norm_interval=record_weight_norm_interval)
        params.set_ineffective_train_stop()
        params.set_high_loss_train_stop()
        train_grokking(params)


if __name__ == "__main__":
    main()
