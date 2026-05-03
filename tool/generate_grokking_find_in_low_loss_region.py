from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src.ml_setup_dataset.dataset_modular import ArithmeticDataset, ArithmeticIterator
from py_src.util import set_seed, setup_logging

from generate_grokking import (
    SPLIT_CHOICES,
    GrokkingParameters,
    build_grokking_model,
    default_batch_size,
    generate_dataset,
    initialize_model_for_training,
    loading_dataset_from,
    train_grokking,
)


logger = logging.getLogger("generate_grokking_in_region")
REQUIREMENT_CHOICES = ["fit", "shift"]


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
    log_csv_path: str
    model_path: str


def _clone_dataset(dataset: ArithmeticDataset, data: torch.Tensor, *, name_suffix: str) -> ArithmeticDataset:
    return ArithmeticDataset(
        f"{dataset.name}_{name_suffix}",
        data,
        dataset.modulus,
        dataset.train,
        tokenizer=dataset.tokenizer,
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
    raise ValueError(f"Unknown requirement: {requirement}")


def save_requirement_datasets(output_folder_path: str, train_dataset: ArithmeticDataset, val_dataset: ArithmeticDataset):
    dataset_dir = os.path.join(output_folder_path, "region_dataset")
    train_dataset.save_to_file(os.path.join(dataset_dir, "train.txt"))
    val_dataset.save_to_file(os.path.join(dataset_dir, "val.txt"))
    train_dataset.tokenizer.save_tokens(os.path.join(dataset_dir, "tokenizer.txt"))
    return dataset_dir


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
        log_csv_path=log_csv_path,
        model_path=model_path,
    )


def write_summary(output_folder_path: str, payload: dict):
    with open(os.path.join(output_folder_path, "summary.json"), "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a grokking model toward a requested train/validation fit region",
    )
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
    parser.add_argument("--train_requirement", type=str, default="fit", choices=REQUIREMENT_CHOICES)
    parser.add_argument("--val_requirement", type=str, default="fit", choices=REQUIREMENT_CHOICES)
    parser.add_argument("--label_shift", type=int, default=1)
    parser.add_argument("--success_threshold", type=float, default=1.0)
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
    parser.add_argument("--m_nlayer", default=None, type=int)
    parser.add_argument("--m_n_heads", default=None, type=int)
    parser.add_argument("--m_d_model", default=None, type=int)
    parser.add_argument("--m_context_len", default=None, type=int)
    parser.add_argument("--m_pos_encoding", default=None, type=str, choices=["default", "trainable"])
    parser.add_argument("--enable_ineffective_training_stop", action="store_true")
    parser.add_argument("--enable_high_loss_train_stop", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_type != "transformer_for_grokking":
        raise ValueError("generate_grokking_in_region.py only supports --model_type transformer_for_grokking")
    if args.number_of_models < 1:
        raise ValueError("--number_of_models must be at least 1")
    if not (0.0 < args.success_threshold <= 1.0):
        raise ValueError("--success_threshold must be in the interval (0, 1]")

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
        output_folder_path = os.path.join(os.curdir, f"generate_grokking_in_region_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}")
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
        train_dataset, val_dataset = generate_dataset(
            output_folder_path,
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
    requirement_dataset_dir = save_requirement_datasets(output_folder_path, required_train_dataset, required_val_dataset)

    batch_size = default_batch_size(required_train_dataset) if args.batchsize is None else args.batchsize
    logger.info("batch size = %s", batch_size)
    total_epoch = 150000 if args.epoch is None else args.epoch
    record_weight_norm_interval = max(1, total_epoch // 2000) if args.record_weight_norm is None else args.record_weight_norm

    run_indices = list(range(args.start_index, args.start_index + args.number_of_models))
    digit_width = max(1, len(str(run_indices[-1]))) if run_indices else 1
    run_results: list[RunResult] = []
    found = False

    for run_index in run_indices:
        save_name = str(run_index).zfill(digit_width)
        logger.info("starting run %s", save_name)
        model = build_grokking_model(
            required_train_dataset,
            n_layers=args.m_nlayer,
            n_heads=args.m_n_heads,
            d_model=args.m_d_model,
            context_len=args.m_context_len,
            position_encoding=args.m_pos_encoding,
        )
        initialize_model_for_training(
            model,
            transfer_learn_path=args.transfer_learn,
            init_model_path=args.init_model,
            disable_reinit=args.disable_reinit,
            device=device,
        )

        train_dataloader = ArithmeticIterator(required_train_dataset, device, batchsize_hint=batch_size)
        val_dataloader = ArithmeticIterator(required_val_dataset, device, batchsize_hint=batch_size, shuffle=False)

        params = GrokkingParameters()
        params.set_env(output_folder_path, True, logger=logger)
        params.set_early_stop_thresholds(train_accuracy=args.success_threshold, val_accuracy=args.success_threshold)
        params.set_ml_env(model, args.model_type, required_train_dataset.name, required_train_dataset.tokenizer)
        params.set_ml_hyperparameter(
            learning_rate=1e-3 if args.learning_rate is None else args.learning_rate,
            weight_decay=0.0 if args.weight_decay is None else args.weight_decay,
            min_lr=1e-4 if args.min_lr is None else args.min_lr,
            warmup_epoch=10,
            total_epoch=total_epoch,
        )
        params.set_dataloader(train_dataloader, val_dataloader)
        params.set_model_save(
            save_name,
            save_format=args.save_format,
            save_interval=args.save_interval,
            record_weight_norm_interval=record_weight_norm_interval,
        )
        if args.enable_ineffective_training_stop:
            params.set_ineffective_train_stop()
        if args.enable_high_loss_train_stop:
            params.set_high_loss_train_stop()

        train_grokking(params)

        run_result = summarize_run(
            os.path.join(output_folder_path, f"{save_name}.log.csv"),
            os.path.join(output_folder_path, f"{save_name}.model.pt"),
            run_index=run_index,
            save_name=save_name,
            success_threshold=args.success_threshold,
        )
        run_results.append(run_result)
        logger.info(
            "run %s finished: success=%s final(train,val)=(%.4f, %.4f) best(train,val)=(%.4f, %.4f)",
            save_name,
            run_result.success,
            run_result.final_train_accuracy,
            run_result.final_val_accuracy,
            run_result.best_train_accuracy,
            run_result.best_val_accuracy,
        )

        if run_result.success:
            found = True
            logger.info("found a model satisfying the requested region at epoch %s", run_result.success_epoch)
            break

    summary = {
        "possible": found,
        "model_type": args.model_type,
        "success_threshold": args.success_threshold,
        "train_requirement": args.train_requirement,
        "val_requirement": args.val_requirement,
        "label_shift": effective_shift,
        "dataset_name": train_dataset.name,
        "dataset_path": args.dataset_path,
        "dataset_exp": args.dataset_exp,
        "modulus": train_dataset.modulus,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "requirement_dataset_dir": requirement_dataset_dir,
        "runs_attempted": len(run_results),
        "runs_requested": args.number_of_models,
        "runs": [asdict(run_result) for run_result in run_results],
    }
    write_summary(output_folder_path, summary)

    if found:
        logger.info("result: found a model that matches the requested region")
    else:
        logger.info("result: no model matching the requested region was found in %d attempt(s)", len(run_results))


if __name__ == "__main__":
    main()
