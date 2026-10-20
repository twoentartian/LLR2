#!/usr/bin/env python3
"""Train randomly initialized Pythia-14M on prepared Pile A/B, single node 1/2/4 GPUs.

Example: torchrun --standalone --nproc_per_node=4 tool/train_pythia.py \
    --data /path/to/ab_seed42 --output /path/to/run
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import itertools
import json
import logging
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from py_src import distributed
from py_src.language_model.pile import ExactTokenBatchSampler, PileTokenDataset
from py_src.language_model.pythia import build_pythia

LOGGER = logging.getLogger("train_pythia")
CHECKPOINT_VERSION = 1
STOP_REQUESTED = False


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def scheduled_lr(consumed_tokens, total_tokens, *, peak, minimum, warmup_fraction):
    """Token-based adaptation of Pythia's 1%-warmup cosine schedule.

    With complete constant-size batches this is the original step schedule;
    shortened one-pass training rescales the schedule to its actual token budget.
    """
    warmup = max(1, int(total_tokens * warmup_fraction))
    if consumed_tokens < warmup:
        return peak * consumed_tokens / warmup
    progress = min(1.0, (consumed_tokens - warmup) / max(1, total_tokens - warmup))
    return minimum + (peak - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def amp_context(device, precision):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def save_checkpoint(path, model, optimizer, *, progress, identity, device):
    state = rng_state(device)
    if distributed.enabled():
        states = [None] * distributed.world_size()
        torch.distributed.all_gather_object(states, state)
    else:
        states = [state]
    if distributed.is_main():
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "checkpoint_version": CHECKPOINT_VERSION,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "progress": progress,
                "identity": identity,
                "rng_states": states,
            },
            temporary,
        )
        os.replace(temporary, path)
        LOGGER.info(
            "Checkpoint: step=%d tokens=%d path=%s",
            progress["step"],
            progress["tokens"],
            path,
        )
    distributed.barrier()


@torch.no_grad()
def evaluate(model, dataset, *, device, precision, sequences, seed, batch_size):
    # A fixed subset is explicitly a TRAINING probe unless a disjoint partition
    # was selected with --eval-partition. Never call training data validation.
    indices = np.random.Generator(np.random.PCG64(seed)).choice(
        len(dataset), size=min(sequences, len(dataset)), replace=False
    )
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        sampler=distributed.EvaluationSampler(subset),
        generator=torch.Generator().manual_seed(seed),
    )
    was_training = model.training
    model.eval()
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    for batch in loader:
        inputs, labels = (batch[key].to(device) for key in ("input_ids", "labels"))
        with amp_context(device, precision):
            loss, correct = model(inputs, labels)
        totals += torch.stack((loss.detach(), correct, labels.ne(-100).sum())).to(
            torch.float64
        )
    distributed.sum_tensor(totals)
    nll, correct, tokens = totals.cpu().tolist()
    model.train(was_training)
    return {
        "nll": nll / tokens,
        "perplexity": math.exp(min(700, nll / tokens)),
        "token_accuracy": correct / tokens,
        "target_tokens": int(tokens),
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--partition", choices=("A", "B", "AB"), default="AB")
    result.add_argument("--eval-partition", choices=("A", "B"))
    result.add_argument("--sequence-length", type=int, default=2048)
    result.add_argument("--epochs", type=int, default=1)
    result.add_argument(
        "--max-tokens",
        type=int,
        help="Exact total target-token cap across all epochs/ranks",
    )
    result.add_argument(
        "--micro-batch-size",
        type=int,
        default=16,
        help="Sequences per GPU, before accumulation",
    )
    result.add_argument("--gradient-accumulation", type=int, default=16)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--min-learning-rate", type=float, default=1e-4)
    result.add_argument("--warmup-fraction", type=float, default=0.01)
    result.add_argument("--weight-decay", type=float, default=0.1)
    result.add_argument("--clip-grad-norm", type=float, default=1.0)
    result.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    result.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    result.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--log-interval", type=int, default=10)
    result.add_argument("--save-interval", type=int, default=1000)
    result.add_argument("--eval-interval", type=int, default=1000)
    result.add_argument("--eval-sequences", type=int, default=128)
    result.add_argument("--resume", type=Path)
    result.add_argument(
        "--max-steps",
        type=int,
        help="Stop cleanly at this absolute optimizer step (benchmark/testing)",
    )
    result.add_argument("--cpu", action="store_true")
    result.add_argument(
        "--model-config",
        type=Path,
        help="Local GPTNeoX architecture JSON only; no weights",
    )
    result.add_argument(
        "--tokenizer-dir",
        type=Path,
        help="Optional local tokenizer assets to copy with the exported model",
    )
    return result


def train(args):
    global STOP_REQUESTED
    STOP_REQUESTED = False
    for name in (
        "sequence_length",
        "epochs",
        "micro_batch_size",
        "gradient_accumulation",
        "log_interval",
        "save_interval",
        "eval_interval",
        "eval_sequences",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if (
        args.workers < 0
        or args.seed < 0
        or args.weight_decay < 0
        or args.clip_grad_norm <= 0
    ):
        raise ValueError("Invalid worker count, seed, weight decay, or clipping norm")
    if (
        not 0 <= args.warmup_fraction < 1
        or not 0 <= args.min_learning_rate <= args.learning_rate
        or args.learning_rate <= 0
    ):
        raise ValueError("Invalid learning-rate schedule")
    if (
        args.max_tokens is not None
        and args.max_tokens <= 0
        or args.max_steps is not None
        and args.max_steps <= 0
    ):
        raise ValueError("Token/step caps must be positive")
    if args.eval_partition and args.eval_partition in args.partition:
        raise ValueError("Held-out evaluation partition must be excluded from training")
    device = distributed.initialize(use_cpu=args.cpu)
    device = device or torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "Native BF16 is required; use --precision fp32 otherwise"
            )
    elif not args.cpu:
        raise RuntimeError("CUDA unavailable; pass --cpu explicitly for tests")
    logging.basicConfig(
        level=logging.INFO if distributed.is_main() else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # Seed BEFORE constructing any model, including after a torchrun launch.
    seed_everything(args.seed)
    dataset = PileTokenDataset(
        args.data, partitions=args.partition, sequence_length=args.sequence_length
    )
    if args.model_config:
        from transformers import GPTNeoXConfig

        config = GPTNeoXConfig.from_json_file(str(args.model_config))
    else:
        config = None
    raw_model = build_pythia(config, attention=args.attention).to(device)
    config = raw_model.model.config
    if config.vocab_size != dataset.manifest["vocabulary_size"]:
        raise ValueError("Model and prepared dataset vocabulary sizes differ")
    if args.sequence_length > config.max_position_embeddings:
        raise ValueError("Sequence length exceeds the architecture's context limit")
    decay = [p for p in raw_model.parameters() if p.ndim >= 2]
    no_decay = [p for p in raw_model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=False,
    )
    budget = min(
        dataset.target_tokens * args.epochs,
        args.max_tokens or dataset.target_tokens * args.epochs,
    )
    identity = {
        "dataset": dataset.fingerprint,
        "partition": args.partition,
        "sequence_length": args.sequence_length,
        "epochs": args.epochs,
        "token_budget": budget,
        "seed": args.seed,
        "world_size": distributed.world_size(),
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "precision": args.precision,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_fraction": args.warmup_fraction,
        "weight_decay": args.weight_decay,
        "clip_grad_norm": args.clip_grad_norm,
        "attention": args.attention,
        "model_config": json.loads(config.to_json_string(use_diff=False)),
        "initialization": "NeoX small_init + wang_init residual projections",
    }
    progress = {"epoch": 0, "next_micro": 0, "step": 0, "tokens": 0}
    resume_rng = None
    if args.resume:
        # Only load our own trusted training artifacts (includes NumPy/Python RNG).
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION
            or checkpoint["identity"] != identity
        ):
            raise ValueError(
                "Checkpoint configuration/data differs; use the same training settings to resume"
            )
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        progress = checkpoint["progress"]
        resume_rng = checkpoint["rng_states"][distributed.rank()]
        del checkpoint
    problem = None
    if distributed.is_main():
        if (args.output / "info.json").exists() and not args.resume:
            problem = (
                "Output already contains a run; use --resume or a new output directory"
            )
        else:
            args.output.mkdir(parents=True, exist_ok=True)
    problem = distributed.broadcast_object(problem)
    if problem:
        raise ValueError(problem)
    distributed.barrier()
    if distributed.is_main() and not args.resume:
        atomic_json(
            args.output / "info.json",
            dict(
                **identity,
                pretrained_weights=False,
                parameters=sum(p.numel() for p in raw_model.parameters()),
                evaluation="heldout" if args.eval_partition else "training_probe",
                eval_partition=args.eval_partition,
            ),
        )
        torch.save(
            {
                "model": raw_model.model.state_dict(),
                "config": identity["model_config"],
                "initialization": identity["initialization"],
            },
            args.output / "initial_model.pt",
        )
    if distributed.is_main() and args.resume and (args.output / "train.csv").exists():
        path = args.output / "train.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            rows = [row for row in reader if int(row["step"]) <= progress["step"]]
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    compiled = (
        torch.compile(raw_model)
        if args.compile and device.type == "cuda"
        else raw_model
    )
    model = (
        DistributedDataParallel(
            compiled,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        if distributed.enabled()
        else compiled
    )
    if resume_rng is not None:
        restore_rng(resume_rng, device)

    def request_stop(signum, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True

    for signum in (signal.SIGTERM, getattr(signal, "SIGUSR1", signal.SIGTERM)):
        signal.signal(signum, request_stop)
    eval_dataset = (
        PileTokenDataset(
            args.data,
            partitions=args.eval_partition,
            sequence_length=args.sequence_length,
        )
        if args.eval_partition
        else dataset
    )
    eval_name = "heldout" if args.eval_partition else "training_probe"
    LOGGER.info(
        "Pythia parameters=%d; target tokens=%d; per-GPU microbatch=%d; effective sequences=%d; %s evaluation",
        sum(p.numel() for p in raw_model.parameters()),
        budget,
        args.micro_batch_size,
        args.micro_batch_size * distributed.world_size() * args.gradient_accumulation,
        eval_name,
    )
    log_totals = torch.zeros(3, device=device, dtype=torch.float64)
    log_start = time.monotonic()
    stop = args.max_steps is not None and progress["step"] >= args.max_steps
    last_eval = None
    try:
        for epoch in range(progress["epoch"], args.epochs):
            start_micro = progress["next_micro"] if epoch == progress["epoch"] else 0
            sampler = ExactTokenBatchSampler(
                dataset,
                micro_batch_size=args.micro_batch_size,
                rank=distributed.rank(),
                world_size=distributed.world_size(),
                seed=args.seed,
                epoch=epoch,
                start_micro=start_micro,
                token_limit=budget - progress["tokens"],
            )
            loader_kwargs = {
                "batch_sampler": sampler,
                "num_workers": args.workers,
                "pin_memory": device.type == "cuda",
                "generator": torch.Generator().manual_seed(args.seed + epoch),
            }
            if args.workers:
                loader_kwargs.update(
                    multiprocessing_context="spawn", persistent_workers=True
                )
            loader = DataLoader(dataset, **loader_kwargs)
            iterator = iter(loader)
            next_micro = start_micro
            while not stop and progress["tokens"] < budget:
                window = list(itertools.islice(iterator, args.gradient_accumulation))
                if not window:
                    break
                local_tokens = sum(
                    int(batch["labels"].ne(-100).sum()) for batch in window
                )
                count = torch.tensor(local_tokens, dtype=torch.int64, device=device)
                distributed.sum_tensor(count)
                global_tokens = int(count.item())
                if not global_tokens:
                    raise ValueError("Optimizer window contains no prediction targets")
                lr = scheduled_lr(
                    progress["tokens"] + global_tokens,
                    budget,
                    peak=args.learning_rate,
                    minimum=args.min_learning_rate,
                    warmup_fraction=args.warmup_fraction,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                model.train()
                for i, batch in enumerate(window):
                    sync = (
                        model.no_sync()
                        if distributed.enabled() and i + 1 < len(window)
                        else contextlib.nullcontext()
                    )
                    inputs, labels = (
                        batch[key].to(device, non_blocking=True)
                        for key in ("input_ids", "labels")
                    )
                    with sync:
                        with amp_context(device, args.precision):
                            loss, correct = model(inputs, labels)
                            # DDP averages rank gradients; compensate for that and
                            # normalize by ACTUAL global target tokens, including tails.
                            scaled = loss * (distributed.world_size() / global_tokens)
                        scaled.backward()
                    log_totals[0] += loss.detach().to(torch.float64)
                    log_totals[1] += correct.to(torch.float64)
                log_totals[2] += local_tokens
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(), args.clip_grad_norm
                )
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError("Non-finite language-model gradient")
                optimizer.step()
                progress["step"] += 1
                progress["tokens"] += global_tokens
                next_micro += len(window)
                progress["epoch"], progress["next_micro"] = epoch, next_micro
                if next_micro == sampler.total_micro_steps:
                    progress["epoch"], progress["next_micro"] = epoch + 1, 0
                if (
                    progress["step"] % args.log_interval == 0
                    or progress["tokens"] == budget
                ):
                    distributed.sum_tensor(log_totals)
                    loss_sum, correct_sum, token_count = log_totals.cpu().tolist()
                    elapsed = max(time.monotonic() - log_start, 1e-9)
                    row = {
                        "step": progress["step"],
                        "epoch": epoch,
                        "target_tokens": progress["tokens"],
                        "nll": loss_sum / token_count,
                        "token_accuracy": correct_sum / token_count,
                        "learning_rate": lr,
                        "tokens_per_second": token_count / elapsed,
                    }
                    if distributed.is_main():
                        path = args.output / "train.csv"
                        new = not path.exists()
                        with path.open("a", newline="") as stream:
                            writer = csv.DictWriter(stream, fieldnames=list(row))
                            if new:
                                writer.writeheader()
                            writer.writerow(row)
                        LOGGER.info(
                            "step=%d tokens=%d/%d nll=%.5f acc=%.4f lr=%.3g tokens/s=%.0f",
                            row["step"],
                            row["target_tokens"],
                            budget,
                            row["nll"],
                            row["token_accuracy"],
                            lr,
                            row["tokens_per_second"],
                        )
                    log_totals.zero_()
                    log_start = time.monotonic()
                if progress["step"] % args.eval_interval == 0:
                    pause_start = time.monotonic()
                    last_eval = evaluate(
                        raw_model,
                        eval_dataset,
                        device=device,
                        precision=args.precision,
                        sequences=args.eval_sequences,
                        seed=args.seed + 100000,
                        batch_size=args.micro_batch_size,
                    )
                    if distributed.is_main():
                        atomic_json(
                            args.output / "evaluation.json",
                            dict(kind=eval_name, step=progress["step"], **last_eval),
                        )
                        LOGGER.info("%s %s", eval_name, last_eval)
                    log_start += time.monotonic() - pause_start
                stop_tensor = torch.tensor(
                    int(STOP_REQUESTED), dtype=torch.int64, device=device
                )
                distributed.sum_tensor(stop_tensor)
                stop = bool(stop_tensor.item()) or (
                    args.max_steps is not None and progress["step"] >= args.max_steps
                )
                if (
                    progress["step"] % args.save_interval == 0
                    or stop
                    or progress["tokens"] == budget
                ):
                    pause_start = time.monotonic()
                    save_checkpoint(
                        args.output / "training_checkpoint.pt",
                        raw_model,
                        optimizer,
                        progress=progress.copy(),
                        identity=identity,
                        device=device,
                    )
                    log_start += time.monotonic() - pause_start
                if stop:
                    break
            del iterator, loader
            if stop or progress["tokens"] >= budget:
                break
        save_checkpoint(
            args.output / "training_checkpoint.pt",
            raw_model,
            optimizer,
            progress=progress.copy(),
            identity=identity,
            device=device,
        )
        if not STOP_REQUESTED:
            last_eval = evaluate(
                raw_model,
                eval_dataset,
                device=device,
                precision=args.precision,
                sequences=args.eval_sequences,
                seed=args.seed + 100000,
                batch_size=args.micro_batch_size,
            )
        if distributed.is_main():
            # These are OUR trained weights, exported in the standard HF layout.
            raw_model.model.save_pretrained(args.output / "model")
            if args.tokenizer_dir and args.tokenizer_dir.is_dir():
                import shutil

                for name in (
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ):
                    if (args.tokenizer_dir / name).is_file():
                        shutil.copy2(
                            args.tokenizer_dir / name, args.output / "model" / name
                        )
            atomic_json(
                args.output / "summary.json",
                dict(
                    **progress,
                    token_budget=budget,
                    complete=progress["tokens"] >= budget,
                    evaluation_kind=eval_name,
                    evaluation=last_eval,
                ),
            )
        distributed.barrier()
        return progress
    finally:
        distributed.shutdown()


def main(argv=None):
    args = parser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
