from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src.model_opti_save_load import save_model_state
from py_src.ml_setup.dataloader_util import cache_dataloader_on_device
from py_src.ml_setup_dataset.dataset_modular import ArithmeticDataset, ArithmeticIterator
import py_src.service.record_weights_difference as record_weights_difference
from py_src.util import set_seed, setup_logging

from generate_grokking import (
    SPLIT_CHOICES,
    GrokkingParameters,
    build_grokking_model,
    default_batch_size,
    initialize_model_for_training,
    loading_dataset_from,
    normalize_expression,
)


logger = logging.getLogger("generate_grokking_find_in_low_loss_region")
REQUIREMENT_CHOICES = ["fit", "shift", "mismatch"]
REQUIREMENT_TO_CODE = {
    "fit": 0,
    "shift": 1,
    "mismatch": 2,
}
MISMATCH_MARGIN = 0.0
SPEED_REPORT_INTERVAL = 100


@dataclass
class RunResult:
    run_index: int
    save_name: str
    success: bool
    success_epoch: int | None
    final_epoch: int | None
    final_train_accuracy: float
    final_val_accuracy: float
    best_train_accuracy: float
    best_val_accuracy: float
    final_train_fit_accuracy: float
    final_val_fit_accuracy: float
    best_train_fit_accuracy: float
    best_val_fit_accuracy: float
    log_csv_path: str
    model_path: str


@dataclass
class TrainStepOutput:
    loss_value: float
    sample_count: int


@dataclass
class ObjectiveMetrics:
    loss: float
    accuracy: float


@dataclass
class PartitionMetrics:
    loss: float
    requirement_score: float
    fit_accuracy: float


def _clone_dataset(dataset: ArithmeticDataset, data: torch.Tensor, *, name_suffix: str) -> ArithmeticDataset:
    return ArithmeticDataset(
        f"{dataset.name}_{name_suffix}",
        data,
        dataset.modulus,
        dataset.train,
        tokenizer=dataset.tokenizer,
    )


def _merge_datasets(*datasets: ArithmeticDataset, name_suffix: str) -> ArithmeticDataset:
    if len(datasets) == 0:
        raise ValueError("At least one dataset is required for merging")
    base = datasets[0]
    merged_data = torch.cat([dataset.data for dataset in datasets], dim=0)
    return ArithmeticDataset(
        f"{base.name}_{name_suffix}",
        merged_data,
        base.modulus,
        train=True,
        tokenizer=base.tokenizer,
    )


def _empty_dataset_like(dataset: ArithmeticDataset, *, name_suffix: str) -> ArithmeticDataset:
    if dataset.data.ndim == 1:
        width = int(dataset.data.shape[0])
    elif dataset.data.ndim >= 2:
        width = int(dataset.data.shape[1])
    else:
        width = 0
    empty_data = torch.empty((0, width), dtype=dataset.data.dtype)
    return ArithmeticDataset(
        f"{dataset.name}_{name_suffix}",
        empty_data,
        dataset.modulus,
        train=False,
        tokenizer=dataset.tokenizer,
    )


def build_dataset_without_saving(train_pct: float, expression: str, modulus: int, split_type: str, operand_length: int | None):
    normalized_expression = normalize_expression(expression, modulus)
    return ArithmeticDataset.splits(
        train_pct=train_pct,
        operator=normalized_expression,
        train_split_type=split_type,  # type: ignore[arg-type]
        modulus=modulus,
        operand_length=operand_length,
    )


def _shift_numeric_token(token: str, modulus: int, label_shift: int) -> str | None:
    if not token.isdigit():
        return None
    value = int(token)
    if token != str(value) or not (0 <= value < modulus):
        return None
    return str((value + label_shift) % modulus)


def _shift_rhs_labels(dataset: ArithmeticDataset, *, label_shift: int, partition_name: str) -> torch.Tensor:
    shifted = dataset.data.clone()
    eq_token_id = dataset.tokenizer.stoi["="]
    eos_token_id = dataset.tokenizer.stoi["<|eos|>"]
    shifted_token_count = 0

    for row_idx, row in enumerate(shifted):
        eq_positions = torch.nonzero(row == eq_token_id, as_tuple=False).flatten()
        if eq_positions.numel() != 1:
            raise ValueError(f"{partition_name} row {row_idx} does not contain exactly one '=' token")
        eq_position = int(eq_positions.item())

        eos_positions = torch.nonzero(row == eos_token_id, as_tuple=False).flatten()
        eos_positions = eos_positions[eos_positions > eq_position]
        if eos_positions.numel() == 0:
            raise ValueError(f"{partition_name} row {row_idx} does not contain an EOS token after '='")
        rhs_end = int(eos_positions[0].item())

        row_shift_count = 0
        for position in range(eq_position + 1, rhs_end):
            token_id = int(row[position].item())
            token = dataset.tokenizer.itos[token_id]
            shifted_token = _shift_numeric_token(token, dataset.modulus, label_shift)
            if shifted_token is None:
                raise ValueError(
                    f"{partition_name} row {row_idx} has non-modular RHS token '{token}'. "
                    "This script only supports shifting modular-integer outputs."
                )
            row[position] = dataset.tokenizer.stoi[shifted_token]
            row_shift_count += 1

        if row_shift_count == 0:
            raise ValueError(f"{partition_name} row {row_idx} has no RHS tokens to shift")
        shifted_token_count += row_shift_count

    logger.info(
        "shifted %d RHS tokens in %s by %+d modulo %d",
        shifted_token_count,
        partition_name,
        label_shift,
        dataset.modulus,
    )
    return shifted


def transform_dataset_for_requirement(
    dataset: ArithmeticDataset,
    *,
    requirement: str,
    label_shift: int,
    partition_name: str,
) -> ArithmeticDataset:
    if requirement == "fit":
        return _clone_dataset(dataset, dataset.data.clone(), name_suffix=f"{partition_name}_fit")
    if requirement == "shift":
        shifted = _shift_rhs_labels(dataset, label_shift=label_shift, partition_name=partition_name)
        return _clone_dataset(dataset, shifted, name_suffix=f"{partition_name}_shift")
    if requirement == "mismatch":
        return _clone_dataset(dataset, dataset.data.clone(), name_suffix=f"{partition_name}_mismatch")
    raise ValueError(f"Unknown requirement: {requirement}")


def save_requirement_datasets(
    output_folder_path: str,
    train_dataset: ArithmeticDataset,
    val_dataset: ArithmeticDataset,
):
    dataset_dir = os.path.join(output_folder_path, "region_dataset")
    train_dataset.save_to_file(os.path.join(dataset_dir, "train.txt"))
    val_dataset.save_to_file(os.path.join(dataset_dir, "val.txt"))
    train_dataset.tokenizer.save_tokens(os.path.join(dataset_dir, "tokenizer.txt"))
    return dataset_dir


def _csv_float(row: dict[str, str], key: str, *, fallback_key: str | None = None) -> float:
    value = row.get(key)
    if value not in (None, ""):
        return float(value)
    if fallback_key is not None:
        return float(row[fallback_key])
    raise KeyError(f"CSV column '{key}' is missing")


def summarize_run(log_csv_path: str, model_path: str, *, run_index: int, save_name: str, success_threshold: float) -> RunResult:
    with open(log_csv_path, "r", newline="", encoding="utf-8") as infile:
        rows = list(csv.DictReader(infile))
    if not rows:
        raise ValueError(f"No training rows were written to {log_csv_path}")

    success_row = next(
        (
            row
            for row in rows
            if float(row["training_accuracy"]) >= success_threshold
            and float(row["validation_accuracy"]) >= success_threshold
        ),
        None,
    )
    final_row = rows[-1]

    return RunResult(
        run_index=run_index,
        save_name=save_name,
        success=success_row is not None,
        success_epoch=None if success_row is None else int(success_row["epoch"]),
        final_epoch=int(final_row["epoch"]),
        final_train_accuracy=float(final_row["training_accuracy"]),
        final_val_accuracy=float(final_row["validation_accuracy"]),
        best_train_accuracy=max(float(row["training_accuracy"]) for row in rows),
        best_val_accuracy=max(float(row["validation_accuracy"]) for row in rows),
        final_train_fit_accuracy=_csv_float(final_row, "train_fit_accuracy", fallback_key="training_accuracy"),
        final_val_fit_accuracy=_csv_float(final_row, "val_fit_accuracy", fallback_key="validation_accuracy"),
        best_train_fit_accuracy=max(_csv_float(row, "train_fit_accuracy", fallback_key="training_accuracy") for row in rows),
        best_val_fit_accuracy=max(_csv_float(row, "val_fit_accuracy", fallback_key="validation_accuracy") for row in rows),
        log_csv_path=log_csv_path,
        model_path=model_path,
    )


def write_summary(output_folder_path: str, payload: dict):
    with open(os.path.join(output_folder_path, "summary.json"), "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2)


def _check_loss_above_initial(loss_above_initial_counter, initial_loss, current_loss, current_epoch, total_epoch, *, initial_loss_multiplier=1.1, lookback_ratio=0.1):
    threshold = initial_loss * initial_loss_multiplier
    loss_above_initial_counter = loss_above_initial_counter + 1 if current_loss > threshold else 0
    required = max(1, int(total_epoch * lookback_ratio))
    return (current_epoch >= required and loss_above_initial_counter >= required), loss_above_initial_counter


def _build_requirement_codes(train_size: int, train_requirement: str, val_size: int, val_requirement: str) -> torch.Tensor:
    train_codes = torch.full((train_size,), REQUIREMENT_TO_CODE[train_requirement], dtype=torch.long)
    val_codes = torch.full((val_size,), REQUIREMENT_TO_CODE[val_requirement], dtype=torch.long)
    return torch.cat([train_codes, val_codes], dim=0)


def _cache_partition_batches(dataset: ArithmeticDataset, device: torch.device, batchsize_hint: int | float, *, shuffle: bool) -> list[dict[str, torch.Tensor]]:
    if len(dataset) == 0:
        return []
    dataloader = ArithmeticIterator(dataset, device, batchsize_hint=batchsize_hint, shuffle=shuffle)
    return cache_dataloader_on_device(dataloader, device)


def _cache_merged_training_batches(
    dataset: ArithmeticDataset,
    requirement_codes: torch.Tensor,
    device: torch.device,
    batchsize_hint: int | float,
) -> list[dict[str, torch.Tensor]]:
    actual_batchsize = ArithmeticIterator.calculate_batchsize(len(dataset), batchsize_hint=batchsize_hint)
    permutation = torch.randperm(len(dataset))
    batches: list[dict[str, torch.Tensor]] = []
    for batch_begin in range(0, len(dataset), actual_batchsize):
        indices = permutation[batch_begin : batch_begin + actual_batchsize]
        batch_data = dataset.data[indices]
        batches.append(
            {
                "text": batch_data[:, :-1].to(device),
                "target": batch_data[:, 1:].to(device),
                "requirement_codes": requirement_codes[indices].to(device),
            }
        )
    return batches


def _rhs_logits_and_targets(model, batch: dict[str, torch.Tensor], tokenizer, *, train: bool):
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
    return y_hat_rhs, y_rhs


def _fit_loss_per_sample(y_hat_rhs: torch.Tensor, y_rhs: torch.Tensor) -> torch.Tensor:
    per_token_loss = torch.nn.functional.cross_entropy(y_hat_rhs, y_rhs, reduction="none")
    return per_token_loss.mean(dim=-1)


def _mismatch_loss_per_sample(y_hat_rhs: torch.Tensor, y_rhs: torch.Tensor) -> torch.Tensor:
    correct_logits = y_hat_rhs.gather(dim=1, index=y_rhs.unsqueeze(1)).squeeze(1)
    correct_mask = torch.zeros_like(y_hat_rhs, dtype=torch.bool)
    correct_mask.scatter_(1, y_rhs.unsqueeze(1), True)
    max_wrong_logits = y_hat_rhs.masked_fill(correct_mask, float("-inf")).amax(dim=1)
    per_token_loss = torch.relu(correct_logits - max_wrong_logits + MISMATCH_MARGIN)
    return per_token_loss.mean(dim=-1)


def _objective_loss_per_sample(y_hat_rhs: torch.Tensor, y_rhs: torch.Tensor, requirement_codes: torch.Tensor) -> torch.Tensor:
    fit_loss = _fit_loss_per_sample(y_hat_rhs, y_rhs)
    mismatch_loss = _mismatch_loss_per_sample(y_hat_rhs, y_rhs)
    mismatch_mask = requirement_codes == REQUIREMENT_TO_CODE["mismatch"]
    return torch.where(mismatch_mask, mismatch_loss, fit_loss)


def _row_accuracy(y_hat_rhs: torch.Tensor, y_rhs: torch.Tensor) -> torch.Tensor:
    y_hat_max = y_hat_rhs.argmax(dim=-2)
    return (y_hat_max == y_rhs).all(dim=-1)


def _partition_training_step(batch, model, optimizer, lr_scheduler, tokenizer, *, scaler=None) -> TrainStepOutput:
    optimizer.zero_grad(set_to_none=True)
    y_hat_rhs, y_rhs = _rhs_logits_and_targets(model, batch, tokenizer, train=True)
    per_sample_loss = _objective_loss_per_sample(y_hat_rhs, y_rhs, batch["requirement_codes"])
    loss = per_sample_loss.mean()

    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()

    return TrainStepOutput(
        loss_value=float(loss.item()),
        sample_count=int(y_rhs.shape[0]),
    )


def _evaluate_batches(model, tokenizer, batches: list[dict[str, torch.Tensor]], *, objective: str) -> ObjectiveMetrics:
    if len(batches) == 0:
        return ObjectiveMetrics(loss=0.0, accuracy=1.0)

    total_loss = 0.0
    total_count = 0
    total_correct = 0
    for batch in batches:
        y_hat_rhs, y_rhs = _rhs_logits_and_targets(model, batch, tokenizer, train=False)
        if objective == "fit":
            per_sample_loss = _fit_loss_per_sample(y_hat_rhs, y_rhs)
        elif objective == "mismatch":
            per_sample_loss = _mismatch_loss_per_sample(y_hat_rhs, y_rhs)
        else:
            raise ValueError(f"Unknown objective: {objective}")
        correct_location = _row_accuracy(y_hat_rhs, y_rhs)
        total_loss += float(per_sample_loss.sum().item())
        total_count += int(y_rhs.shape[0])
        total_correct += int(correct_location.int().sum().item())

    return ObjectiveMetrics(
        loss=total_loss / total_count,
        accuracy=total_correct / total_count,
    )


def _evaluate_partition(
    model,
    tokenizer,
    *,
    original_batches: list[dict[str, torch.Tensor]],
    required_batches: list[dict[str, torch.Tensor]],
    requirement: str,
) -> PartitionMetrics:
    if len(original_batches) == 0 and len(required_batches) == 0:
        return PartitionMetrics(loss=0.0, requirement_score=1.0, fit_accuracy=1.0)

    fit_metrics = _evaluate_batches(model, tokenizer, original_batches, objective="fit")
    if requirement == "fit":
        return PartitionMetrics(loss=fit_metrics.loss, requirement_score=fit_metrics.accuracy, fit_accuracy=fit_metrics.accuracy)
    if requirement == "shift":
        shift_metrics = _evaluate_batches(model, tokenizer, required_batches, objective="fit")
        return PartitionMetrics(loss=shift_metrics.loss, requirement_score=shift_metrics.accuracy, fit_accuracy=fit_metrics.accuracy)
    if requirement == "mismatch":
        mismatch_metrics = _evaluate_batches(model, tokenizer, original_batches, objective="mismatch")
        return PartitionMetrics(loss=mismatch_metrics.loss, requirement_score=1.0 - fit_metrics.accuracy, fit_accuracy=fit_metrics.accuracy)
    raise ValueError(f"Unknown requirement: {requirement}")


def _write_final_correct_position(output_folder_path, tokenizer, model, train_batches, val_batches):
    def token_to_repr(token_id: int) -> str:
        token = tokenizer.itos[token_id]
        try:
            return str(int(token))
        except ValueError:
            return token

    rows = []
    for batch in [*train_batches, *val_batches]:
        y_hat_rhs, y_rhs = _rhs_logits_and_targets(model, batch, tokenizer, train=False)
        correct_location = _row_accuracy(y_hat_rhs, y_rhs)
        lhs_tokens = batch["text"][:, 1].detach().cpu().tolist()
        rhs_tokens = batch["text"][:, 3].detach().cpu().tolist()
        for lhs, rhs, correct in zip(lhs_tokens, rhs_tokens, correct_location.tolist()):
            rows.append((token_to_repr(lhs), token_to_repr(rhs), int(bool(correct))))

    rows.sort(key=lambda item: (item[0], item[1]))
    with open(os.path.join(output_folder_path, "final_correct_position.csv"), "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["lhs", "rhs", "correct?"])
        writer.writerows(rows)


def train_grokking_partition_requirements(
    parameters: GrokkingParameters,
    *,
    merged_train_dataset: ArithmeticDataset,
    original_train_dataset: ArithmeticDataset,
    original_val_dataset: ArithmeticDataset,
    required_train_dataset: ArithmeticDataset,
    required_val_dataset: ArithmeticDataset,
    batchsize_hint: int | float,
    train_requirement: str,
    val_requirement: str,
):
    assert parameters.model is not None
    assert parameters.learning_rate is not None
    assert parameters.weight_decay is not None
    assert parameters.total_epoch is not None
    assert parameters.warmup_epoch is not None
    assert parameters.min_lr is not None
    assert parameters.output_folder_path is not None
    assert parameters.tokenizer is not None
    assert parameters.model_name is not None
    assert parameters.dataset_name is not None
    assert parameters.save_name is not None
    assert parameters.record_weight_norm_interval is not None

    model = parameters.model
    device = next(model.parameters()).device
    if parameters.logger is not None:
        parameters.logger.info("caching merged training batches and partition-eval batches on %s", device)

    requirement_codes = _build_requirement_codes(
        len(required_train_dataset),
        train_requirement,
        len(required_val_dataset),
        val_requirement,
    )
    cached_train = _cache_merged_training_batches(merged_train_dataset, requirement_codes, device, batchsize_hint)
    cached_train_original = _cache_partition_batches(original_train_dataset, device, batchsize_hint, shuffle=False)
    cached_val_original = _cache_partition_batches(original_val_dataset, device, batchsize_hint, shuffle=False)
    cached_train_required = _cache_partition_batches(required_train_dataset, device, batchsize_hint, shuffle=False)
    cached_val_required = _cache_partition_batches(required_val_dataset, device, batchsize_hint, shuffle=False)
    if parameters.logger is not None:
        parameters.logger.info(
            "cached %d merged train batches, %d/%d train/val original eval batches, %d/%d train/val requirement eval batches",
            len(cached_train),
            len(cached_train_original),
            len(cached_val_original),
            len(cached_train_required),
            len(cached_val_required),
        )

    steps_per_epoch = max(1, len(cached_train))
    warmup_steps = parameters.warmup_epoch * steps_per_epoch
    total_steps = parameters.total_epoch * steps_per_epoch
    cosine_steps = max(1, total_steps - warmup_steps)
    if parameters.logger is not None:
        parameters.logger.info(
            "lr scheduler stepping per iteration: steps_per_epoch=%d warmup_steps=%d total_steps=%d",
            steps_per_epoch,
            warmup_steps,
            total_steps,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=parameters.weight_decay,
        lr=parameters.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=parameters.min_lr,
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    runtime_model = model
    if hasattr(torch, "compile"):
        try:
            compiled_model = torch.compile(model)
            if isinstance(compiled_model, torch.nn.Module):
                runtime_model = compiled_model
                if parameters.logger is not None:
                    parameters.logger.info("torch.compile enabled")
            elif parameters.logger is not None:
                parameters.logger.info("torch.compile returned a non-Module wrapper; using the original module")
        except Exception as exc:
            if parameters.logger is not None:
                parameters.logger.info("torch.compile skipped: %s", exc)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16)) if use_amp else None  # type: ignore[arg-type]

    distance_to_origin_service = record_weights_difference.ServiceDistanceToOriginRecorder(1, [0])
    distance_to_origin_service.initialize_without_runtime_parameters({0: model.state_dict()}, parameters.output_folder_path, logger=parameters.logger)

    log_csv_path = os.path.join(parameters.output_folder_path, f"{parameters.save_name}.log.csv")
    log_csv_file = open(log_csv_path, "w", encoding="utf-8")
    log_csv_file.write(
        "epoch,training_loss,training_accuracy,validation_loss,validation_accuracy,"
        "train_fit_accuracy,val_fit_accuracy,merged_objective_loss,lrs\n"
    )
    log_csv_file.flush()

    speed_window_start_time = time.time()
    speed_window_start_epoch = 0
    initial_merged_objective_loss = None
    loss_above_initial_counter = 0

    for epoch in range(parameters.total_epoch):
        merged_loss_sum = 0.0
        merged_count = 0
        for batch in cached_train:
            output = _partition_training_step(
                batch,
                runtime_model,
                optimizer,
                lr_scheduler,
                parameters.tokenizer,
                scaler=scaler,
            )
            merged_loss_sum += output.loss_value * output.sample_count
            merged_count += output.sample_count
        merged_objective_loss = merged_loss_sum / merged_count

        train_metrics = _evaluate_partition(
            runtime_model,
            parameters.tokenizer,
            original_batches=cached_train_original,
            required_batches=cached_train_required,
            requirement=train_requirement,
        )
        val_metrics = _evaluate_partition(
            runtime_model,
            parameters.tokenizer,
            original_batches=cached_val_original,
            required_batches=cached_val_required,
            requirement=val_requirement,
        )
        lrs = [param_group["lr"] for param_group in optimizer.param_groups]

        if parameters.logger is not None:
            parameters.logger.info(
                "epoch[%d] merged_loss=%.4f train(loss,score,fit)=%.4f,%.4f,%.4f val(loss,score,fit)=%.4f,%.4f,%.4f lrs=%s",
                epoch,
                merged_objective_loss,
                train_metrics.loss,
                train_metrics.requirement_score,
                train_metrics.fit_accuracy,
                val_metrics.loss,
                val_metrics.requirement_score,
                val_metrics.fit_accuracy,
                lrs,
            )
        log_csv_file.write(
            f"{epoch},{train_metrics.loss:.4e},{train_metrics.requirement_score:.4e},"
            f"{val_metrics.loss:.4e},{val_metrics.requirement_score:.4e},"
            f"{train_metrics.fit_accuracy:.4e},{val_metrics.fit_accuracy:.4e},"
            f"{merged_objective_loss:.4e},{lrs}\n"
        )
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
            and train_metrics.requirement_score >= parameters.early_stop_train_accuracy
            and val_metrics.requirement_score >= parameters.early_stop_val_accuracy
        ):
            break

        if initial_merged_objective_loss is None:
            initial_merged_objective_loss = float(merged_objective_loss)
        if parameters.high_loss_train_stop:
            should_stop, loss_above_initial_counter = _check_loss_above_initial(
                loss_above_initial_counter,
                initial_merged_objective_loss,
                float(merged_objective_loss),
                epoch,
                parameters.total_epoch,
            )
            if should_stop:
                if parameters.logger is not None:
                    parameters.logger.info("loss_above_initial_stop triggered at epoch %d", epoch)
                break

        if epoch % parameters.record_weight_norm_interval == 0:
            distance_to_origin_service.trigger_without_runtime_parameters(epoch, {0: model.state_dict()})

    log_csv_file.flush()
    log_csv_file.close()
    _write_final_correct_position(parameters.output_folder_path, parameters.tokenizer, runtime_model, cached_train_original, cached_val_original)
    save_model_state(
        os.path.join(parameters.output_folder_path, f"{parameters.save_name}.model.pt"),
        model.state_dict(),
        parameters.model_name,
        parameters.dataset_name,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a grokking model inside a requested low-loss region",
    )
    parser.add_argument("-c", "--core", type=int, default=4)
    parser.add_argument("-o", "--output_folder_name", default=None)
    parser.add_argument("-m", "--model_type", type=str, default="transformer_for_grokking")
    parser.add_argument("-dpath", "--dataset_path", type=str, default=None)
    parser.add_argument("-dexp", "--dataset_exp", type=str, default=None)
    parser.add_argument("--modulus", type=int, default=97)
    parser.add_argument("-tp", "--train_pct", type=float, default=50)
    parser.add_argument("-st", "--split_type", type=str, default="random", choices=SPLIT_CHOICES)
    parser.add_argument("-ol", "--operand_length", type=int, default=None)
    parser.add_argument("-train", "--train_requirement", type=str, default="fit", choices=REQUIREMENT_CHOICES)
    parser.add_argument("-val", "--val_requirement", type=str, default="fit", choices=REQUIREMENT_CHOICES)
    parser.add_argument("--label_shift", type=int, default=1)
    parser.add_argument("--success_threshold", type=float, default=0.99)
    parser.add_argument("-lr", "--learning_rate", type=float, default=None)
    parser.add_argument("-minlr", "--min_lr", type=float, default=None)
    parser.add_argument("-epoch", "--epoch", type=int, default=None)
    parser.add_argument("-wd", "--weight_decay", type=float, default=None)
    parser.add_argument("-bs", "--batchsize", type=int, default=None)
    parser.add_argument("--record_weight_norm", type=int, default=None)
    parser.add_argument("-s", "--random_seed", type=int, default=None)
    parser.add_argument("--init_model", type=str, default=None)
    parser.add_argument("--disable_reinit", action="store_true")
    parser.add_argument("--m_nlayer", default=None, type=int)
    parser.add_argument("--m_n_heads", default=None, type=int)
    parser.add_argument("--m_d_model", default=None, type=int)
    parser.add_argument("--m_context_len", default=None, type=int)
    parser.add_argument("--m_pos_encoding", default=None, type=str, choices=["default", "trainable"])
    parser.add_argument("--enable_high_loss_train_stop", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_type != "transformer_for_grokking":
        raise ValueError("generate_grokking_find_in_low_loss_region.py only supports --model_type transformer_for_grokking")
    if not (0.0 < args.success_threshold <= 1.0):
        raise ValueError("--success_threshold must be in the interval (0, 1]")

    torch.set_num_threads(max(1, min(args.core, 8)))
    setup_logging(logger, "main")
    logger.info("logging setup complete")
    if args.random_seed is not None:
        set_seed(args.random_seed)
        logger.info("random seed = %d", args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device = %s", device)
    if args.output_folder_name is None:
        output_folder_path = os.path.join(os.curdir, f"generate_grokking_find_in_low_loss_region_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}")
    else:
        output_folder_path = os.path.join(os.curdir, args.output_folder_name)
    os.makedirs(output_folder_path, exist_ok=False)
    with open(os.path.join(output_folder_path, "command.txt"), "w", encoding="utf-8") as outfile:
        outfile.write(" ".join([sys.executable, *sys.argv]))

    if args.dataset_path is not None:
        train_dataset, val_dataset = loading_dataset_from(args.dataset_path, modulus=args.modulus)
    else:
        if args.dataset_exp is None:
            raise ValueError("--dataset_exp is required when --dataset_path is not provided")
        train_dataset, val_dataset = build_dataset_without_saving(
            args.train_pct,
            args.dataset_exp,
            args.modulus,
            args.split_type,
            args.operand_length,
        )

    effective_shift = args.label_shift % train_dataset.modulus
    if effective_shift != args.label_shift:
        logger.info("label shift reduced modulo %d: %d -> %d", train_dataset.modulus, args.label_shift, effective_shift)
    if effective_shift == 0 and (args.train_requirement == "shift" or args.val_requirement == "shift"):
        logger.warning("effective label shift is 0, so 'shift' becomes equivalent to 'fit'")

    required_train_dataset = transform_dataset_for_requirement(
        train_dataset,
        requirement=args.train_requirement,
        label_shift=effective_shift,
        partition_name="train",
    )
    required_val_dataset = transform_dataset_for_requirement(
        val_dataset,
        requirement=args.val_requirement,
        label_shift=effective_shift,
        partition_name="val",
    )
    merged_train_dataset = _merge_datasets(required_train_dataset, required_val_dataset, name_suffix="merged_train")
    empty_val_dataset = _empty_dataset_like(required_val_dataset, name_suffix="empty_val")
    requirement_dataset_dir = save_requirement_datasets(
        output_folder_path,
        merged_train_dataset,
        empty_val_dataset,
    )
    logger.info(
        "training on merged dataset: %d train-partition examples + %d val-partition examples = %d total",
        len(required_train_dataset),
        len(required_val_dataset),
        len(merged_train_dataset),
    )
    logger.info(
        "partition requirements: train=%s val=%s",
        args.train_requirement,
        args.val_requirement,
    )

    batch_size = default_batch_size(merged_train_dataset) if args.batchsize is None else args.batchsize
    logger.info("batch size = %s", batch_size)
    total_epoch = 10000 if args.epoch is None else args.epoch
    record_weight_norm_interval = max(1, total_epoch // 2000) if args.record_weight_norm is None else args.record_weight_norm

    run_results: list[RunResult] = []
    found = False
    run_index = 0
    save_name = "0"
    logger.info("starting run %s", save_name)
    model = build_grokking_model(
        merged_train_dataset,
        n_layers=args.m_nlayer,
        n_heads=args.m_n_heads,
        d_model=args.m_d_model,
        context_len=args.m_context_len,
        position_encoding=args.m_pos_encoding,
    )
    initialize_model_for_training(
        model,
        transfer_learn_path=None,
        init_model_path=args.init_model,
        disable_reinit=args.disable_reinit,
        device=device,
    )

    params = GrokkingParameters()
    params.set_env(output_folder_path, True, logger=logger)
    params.set_early_stop_thresholds(train_accuracy=args.success_threshold, val_accuracy=args.success_threshold)
    params.set_ml_env(model, args.model_type, merged_train_dataset.name, merged_train_dataset.tokenizer)
    params.set_ml_hyperparameter(
        learning_rate=1e-3 if args.learning_rate is None else args.learning_rate,
        weight_decay=0.0 if args.weight_decay is None else args.weight_decay,
        min_lr=1e-5 if args.min_lr is None else args.min_lr,
        warmup_epoch=10,
        total_epoch=total_epoch,
    )
    params.set_model_save(
        save_name,
        save_format="none",
        save_interval=1,
        record_weight_norm_interval=record_weight_norm_interval,
    )
    if args.enable_high_loss_train_stop:
        params.set_high_loss_train_stop()

    train_grokking_partition_requirements(
        params,
        merged_train_dataset=merged_train_dataset,
        original_train_dataset=train_dataset,
        original_val_dataset=val_dataset,
        required_train_dataset=required_train_dataset,
        required_val_dataset=required_val_dataset,
        batchsize_hint=batch_size,
        train_requirement=args.train_requirement,
        val_requirement=args.val_requirement,
    )
    numbered_model_path = os.path.join(output_folder_path, f"{save_name}.model.pt")
    final_model_path = os.path.join(output_folder_path, "final.model.pt")
    save_model_state(final_model_path, model.state_dict(), args.model_type, merged_train_dataset.name)
    if not os.path.exists(numbered_model_path):
        save_model_state(numbered_model_path, model.state_dict(), args.model_type, merged_train_dataset.name)

    run_result = summarize_run(
        os.path.join(output_folder_path, f"{save_name}.log.csv"),
        final_model_path,
        run_index=run_index,
        save_name=save_name,
        success_threshold=args.success_threshold,
    )
    run_results.append(run_result)
    logger.info(
        "run %s finished: success=%s final(score train,val)=(%.4f, %.4f) final(fit train,val)=(%.4f, %.4f) best(score train,val)=(%.4f, %.4f)",
        save_name,
        run_result.success,
        run_result.final_train_accuracy,
        run_result.final_val_accuracy,
        run_result.final_train_fit_accuracy,
        run_result.final_val_fit_accuracy,
        run_result.best_train_accuracy,
        run_result.best_val_accuracy,
    )

    if run_result.success:
        found = True
        logger.info("found a model satisfying the requested region at epoch %s", run_result.success_epoch)

    summary = {
        "possible": found,
        "model_type": args.model_type,
        "success_threshold": args.success_threshold,
        "train_requirement": args.train_requirement,
        "val_requirement": args.val_requirement,
        "mismatch_margin": MISMATCH_MARGIN,
        "label_shift": effective_shift,
        "dataset_name": train_dataset.name,
        "dataset_path": args.dataset_path,
        "dataset_exp": args.dataset_exp,
        "modulus": train_dataset.modulus,
        "train_examples": len(merged_train_dataset),
        "val_examples": 0,
        "train_partition_examples": len(required_train_dataset),
        "val_partition_examples": len(required_val_dataset),
        "validation_disabled": False,
        "saved_val_dataset_empty": True,
        "requirement_dataset_dir": requirement_dataset_dir,
        "final_model_path": final_model_path,
        "numbered_model_path": numbered_model_path,
        "runs_attempted": len(run_results),
        "runs_requested": 1,
        "runs": [asdict(run_result) for run_result in run_results],
    }
    write_summary(output_folder_path, summary)

    if found:
        logger.info("result: found a model that matches the requested region")
    else:
        logger.info("result: no model matching the requested region was found in %d attempt(s)", len(run_results))


if __name__ == "__main__":
    main()
