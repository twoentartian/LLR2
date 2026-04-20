from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src.util import setup_logging, set_seed
from py_src.ml_setup_dataset.dataset_modular import ArithmeticIterator

from generate_grokking import (
    SPLIT_CHOICES,
    GrokkingParameters,
    build_grokking_model,
    default_batch_size,
    generate_dataset,
    loading_dataset_from,
    train_grokking,
)


logger = logging.getLogger("generate_grokking_phase_diagram")

DEFAULT_LR_MIN = 1e-5
DEFAULT_LR_MAX = 1e-2
DEFAULT_N_LR = 11
DEFAULT_WD_MIN = 0.0
DEFAULT_WD_MAX = 10.0
DEFAULT_N_WD = 11


def make_lr_grid(lr_min: float, lr_max: float, n: int) -> list[float]:
    return list(np.logspace(np.log10(lr_min), np.log10(lr_max), n))


def make_wd_grid(wd_min: float, wd_max: float, n: int) -> list[float]:
    return list(np.linspace(wd_min, wd_max, n))


def cell_output_dir(base: str, lr: float, wd: float) -> str:
    return os.path.join(base, f"lr{lr:.4e}_wd{wd:.4e}")


def cell_log_csv(base: str, lr: float, wd: float) -> str:
    return os.path.join(cell_output_dir(base, lr, wd), "00.log.csv")


def save_cell_metadata(cell_dir: str, lr: float, wd: float, extra: dict | None = None):
    payload = {"learning_rate": lr, "weight_decay": wd}
    if extra:
        payload.update(extra)
    with open(os.path.join(cell_dir, "meta.json"), "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2)


def read_phase_from_log(log_csv_path: str, *, high_acc: float = 0.95, grokking_val_threshold: float = 0.5) -> str:
    if not os.path.exists(log_csv_path):
        return "unknown"
    with open(log_csv_path, "r", newline="", encoding="utf-8") as infile:
        rows = list(csv.DictReader(infile))
    if not rows:
        return "unknown"
    train_acc = [float(row["training_accuracy"]) for row in rows]
    val_acc = [float(row["validation_accuracy"]) for row in rows]
    final_train = train_acc[-1]
    final_val = val_acc[-1]
    if final_train < high_acc and final_val < high_acc:
        return "confusion"
    if final_train >= high_acc and final_val < high_acc:
        return "memorization"
    if final_train >= high_acc and final_val >= high_acc:
        crossed = next((index for index, value in enumerate(train_acc) if value >= high_acc), len(train_acc) - 1)
        return "comprehension" if val_acc[crossed] >= grokking_val_threshold else "grokking"
    return "comprehension"


def train_cell(args, lr, wd, train_ds, val_ds, device):
    cell_dir = cell_output_dir(args.output_folder_path, lr, wd)
    os.makedirs(cell_dir, exist_ok=True)
    save_cell_metadata(
        cell_dir,
        lr,
        wd,
        extra={
            "modulus": args.modulus,
            "train_pct": args.train_pct,
            "epoch": args.epoch,
            "model_type": args.model_type,
        },
    )

    model = build_grokking_model(
        train_ds,
        n_layers=args.m_nlayer,
        n_heads=args.m_n_heads,
        d_model=args.m_d_model,
        context_len=args.m_context_len,
        position_encoding=args.m_pos_encoding,
    )
    model.to(device)

    batch_size = default_batch_size(train_ds) if args.batchsize is None else args.batchsize
    train_dl = ArithmeticIterator(train_ds, device, batchsize_hint=batch_size)
    val_dl = ArithmeticIterator(val_ds, device, batchsize_hint=batch_size, shuffle=False)

    params = GrokkingParameters()
    params.set_env(cell_dir, True, logger=logger)
    params.set_ml_env(model, args.model_type, train_ds.name, train_ds.tokenizer)
    params.set_ml_hyperparameter(
        learning_rate=lr,
        weight_decay=wd,
        min_lr=lr,
        warmup_epoch=10,
        total_epoch=args.epoch,
    )
    params.set_dataloader(train_dl, val_dl)
    params.set_model_save("00", save_format="none", record_weight_norm_interval=max(1, args.epoch // 1000))
    if args.enable_ineffective_training_stop:
        params.set_ineffective_train_stop()
        params.set_high_loss_train_stop()
    logger.info("  -> training cell lr=%.4e wd=%.4e", lr, wd)
    train_grokking(params)


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep LR x WD to reproduce the grokking phase diagram")
    parser.add_argument("-o", "--output_folder_name", default=None)
    parser.add_argument("--lr_min", type=float, default=DEFAULT_LR_MIN)
    parser.add_argument("--lr_max", type=float, default=DEFAULT_LR_MAX)
    parser.add_argument("--n_lr", type=int, default=DEFAULT_N_LR)
    parser.add_argument("--wd_max", type=float, default=DEFAULT_WD_MAX)
    parser.add_argument("--n_wd", type=int, default=DEFAULT_N_WD)
    parser.add_argument("-dpath", "--dataset_path", type=str, default=None)
    parser.add_argument("-dexp", "--dataset_exp", type=str, default=None)
    parser.add_argument("--modulus", type=int, default=97)
    parser.add_argument("-tp", "--train_pct", type=float, default=50)
    parser.add_argument("-st", "--split_type", type=str, default="random", choices=SPLIT_CHOICES)
    parser.add_argument("-ol", "--operand_length", type=int, default=None)
    parser.add_argument("-epoch", "--epoch", type=int, default=100000)
    parser.add_argument("-bs", "--batchsize", type=int, default=None)
    parser.add_argument("-m", "--model_type", type=str, default="transformer_for_grokking")
    parser.add_argument("--m_nlayer", default=None, type=int)
    parser.add_argument("--m_n_heads", default=None, type=int)
    parser.add_argument("--m_d_model", default=None, type=int)
    parser.add_argument("--m_context_len", default=None, type=int)
    parser.add_argument("--m_pos_encoding", default=None, type=str, choices=["default", "trainable"])
    parser.add_argument("-rs", "--random_seed", type=int, default=None)
    parser.add_argument("--enable_ineffective_training_stop", action="store_true")
    parser.add_argument("--enable_skip_larger_wd_after_confusion", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_type != "transformer_for_grokking":
        raise ValueError("generate_grokking_phase_diagram.py only supports --model_type transformer_for_grokking")

    setup_logging(logger, "main")
    logger.info("phase diagram sweep starting")
    if args.random_seed is not None:
        set_seed(args.random_seed)
        logger.info("random seed = %d", args.random_seed)

    learning_rates = make_lr_grid(args.lr_min, args.lr_max, args.n_lr)
    weight_decays = make_wd_grid(DEFAULT_WD_MIN, args.wd_max, args.n_wd)
    logger.info("LR grid (%d pts, log-spaced): %.2e .. %.2e", args.n_lr, args.lr_min, args.lr_max)
    logger.info("WD grid (%d pts, linear-spaced): %.2f .. %.2f", args.n_wd, DEFAULT_WD_MIN, args.wd_max)

    if args.output_folder_name is None:
        args.output_folder_path = os.path.join(os.curdir, f"generate_grokking_phase_diagram_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    else:
        args.output_folder_path = os.path.join(os.curdir, args.output_folder_name)
    os.makedirs(args.output_folder_path, exist_ok=True)

    with open(os.path.join(args.output_folder_path, "command.txt"), "w", encoding="utf-8") as outfile:
        outfile.write(" ".join(sys.argv))
    with open(os.path.join(args.output_folder_path, "grid_spec.json"), "w", encoding="utf-8") as outfile:
        json.dump({"learning_rates": learning_rates, "weight_decays": weight_decays}, outfile, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset_path is not None:
        train_ds, val_ds = loading_dataset_from(args.dataset_path, modulus=args.modulus)
    else:
        if args.dataset_exp is None:
            raise ValueError("--dataset_exp is required when --dataset_path is not provided")
        train_ds, val_ds = generate_dataset(
            args.output_folder_path,
            args.train_pct,
            args.dataset_exp,
            args.modulus,
            args.split_type,
            args.operand_length,
        )

    total_cells = len(learning_rates) * len(weight_decays)
    done = 0
    skipped_by_confusion = 0

    for lr in learning_rates:
        consecutive_confusion = 0
        skip_rest_of_row = False
        for wd in weight_decays:
            done += 1
            if skip_rest_of_row:
                logger.info("[%d/%d] lr=%.4e wd=%.4e -> skipped (confusion rule)", done, total_cells, lr, wd)
                skipped_by_confusion += 1
                continue

            log_csv_path = cell_log_csv(args.output_folder_path, lr, wd)
            if os.path.exists(log_csv_path):
                logger.info("[%d/%d] lr=%.4e wd=%.4e -> already done, skipping", done, total_cells, lr, wd)
            else:
                logger.info("[%d/%d] lr=%.4e wd=%.4e", done, total_cells, lr, wd)
                train_cell(args, lr, wd, train_ds, val_ds, device)

            if args.enable_skip_larger_wd_after_confusion:
                phase = read_phase_from_log(log_csv_path)
                if phase == "confusion":
                    consecutive_confusion += 1
                    if consecutive_confusion >= 2:
                        skip_rest_of_row = True
                        logger.info("  -> two consecutive confusion cells at lr=%.4e, skipping remaining WD values in this row", lr)
                else:
                    consecutive_confusion = 0

    logger.info("Sweep complete. Cells skipped by confusion rule: %d/%d", skipped_by_confusion, total_cells)


if __name__ == "__main__":
    main()
