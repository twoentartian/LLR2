#!/usr/bin/env python3
"""Normalized train-descent/val-ascent misalignment for Pythia-14M on Pile A/B.

Language-model counterpart of ``misalignment_measurement_2.py`` with
``--objective normalized_two_sided``. Both orientations start from the same
seeded random initialization as ``train_pythia.py`` (equal seeds give equal
initial weights). At every optimizer step the full effective-batch train and
val gradients are computed at the same parameters, normalized per parameter
tensor, weighted, combined as train descent plus val ascent and rescaled to the
train-gradient norm. The ordinary Pythia recipe of ``train_pythia.py`` (AdamW,
1% warmup then cosine on train tokens, clip norm 1, BF16) is then applied to
the combined gradient. One epoch is one pass over the train partition; the val
partition stream restarts with a new order if it runs out first.

A partition holds ~100B tokens and cannot be evaluated in full repeatedly, so
fixed random subsets of A and B (identical for both orientations) are evaluated
every ``--eval-interval`` optimizer steps and at the end. These evaluation
points take the role of epochs in the optimization CSV, the per-parameter
gradient geometry (averaged over the steps since the previous point), the
weight variance and the top-k accuracy/loss-gap checkpoints. CSV columns keep
the image script's names: ``loss_a``/``train_accuracy`` refer to the TRAIN
partition of that orientation and ``loss_b``/``val_accuracy`` to its VAL one.
Relative flatness is not measured here; selected checkpoints are saved.

A training checkpoint is written after every evaluation point and when a stop
is requested (SIGUSR1/SIGTERM or ``--max-steps``); ``--resume`` continues an
interrupted orientation exactly and skips finished ones.

Example: torchrun --standalone --nproc_per_node=4 tool/misalignment_pythia.py \
    --data /path/to/ab_seed42 --output /path/to/run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import signal
import sys
import time
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
for path in (TOOL_DIR.parent, TOOL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from misalignment.gradient_combination import NormalizedGradientCombiner
from misalignment.gradient_geometry import GRADIENT_GEOMETRY_FIELDS, GradientGeometryAccumulator
from py_src import distributed
from py_src.language_model.pile import ExactTokenBatchSampler, PileTokenDataset
from py_src.language_model.pythia import build_pythia
from py_src.model_opti_save_load import save_model_state
from train_pythia import amp_context, atomic_json, restore_rng, rng_state, scheduled_lr, seed_everything

LOGGER = logging.getLogger("misalignment_pythia")
CHECKPOINT_VERSION = 1
MODEL_TYPE = "pythia_14m"
DATASET_TYPE = "pile_deduped"
DIRECTIONS = {"a_as_train_b_as_val": ("A", "B"), "b_as_train_a_as_val": ("B", "A")}
# Each partition's training order depends on its own seed offset and epoch,
# never on its train/val role, as in misalignment_measurement_2.py.
ORDER_SEED_OFFSET = {"A": 101_000, "B": 202_000}
EVAL_SEED_OFFSET = {"A": 0, "B": 1}
OPTIMIZATION_FIELDS = (
    "step", "epoch", "train_target_tokens", "loss_a", "loss_b", "train_accuracy", "val_accuracy",
    "mean_loss", "abs_accuracy_gap", "abs_loss_gap", "relative_gap", "objective", "learning_rate",
    "train_val_gradient_cosine", "train_gradient_weight", "val_gradient_weight",
    "running_train_loss", "running_val_loss", "running_train_accuracy", "running_val_accuracy",
    "tokens_per_second", "eval_seconds",
)
TRAIN_LOG_FIELDS = (
    "step", "train_target_tokens", "train_nll", "train_accuracy", "val_nll", "val_accuracy",
    "learning_rate", "tokens_per_second",
)
GEOMETRY_FIELDS = ("step", "parameter", *GRADIENT_GEOMETRY_FIELDS, "train_gradient_weight", "val_gradient_weight")
OBJECTIVE_OFFSET = 1.0
STOP_REQUESTED = False


def append_csv(path, fields, rows):
    new = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def truncate_csv(path, step):
    """Drop rows written after the checkpoint being resumed."""
    if not path.exists():
        return
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = [row for row in reader if int(float(row["step"])) <= step]
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


class PartitionStream:
    """Consume one partition's rank shard in microbatches across epochs.

    ``epoch`` and ``next_micro`` count consumed microbatches only, so they
    reconstruct the exact position after a resume regardless of prefetching.
    """

    def __init__(self, dataset, *, args, device, seed, epoch, next_micro, epochs=None, token_limit=None):
        self.dataset, self.args, self.device, self.seed = dataset, args, device, seed
        self.epoch, self.next_micro = epoch, next_micro
        self.epochs = epochs
        self.token_limit = token_limit
        self.iterator = None

    def _open(self):
        sampler = ExactTokenBatchSampler(
            self.dataset,
            micro_batch_size=self.args.micro_batch_size,
            rank=distributed.rank(),
            world_size=distributed.world_size(),
            seed=self.seed,
            epoch=self.epoch,
            start_micro=self.next_micro,
            token_limit=None if self.token_limit is None else self.token_limit(),
        )
        kwargs = {"batch_sampler": sampler, "num_workers": self.args.workers,
                  "pin_memory": self.device.type == "cuda"}
        if self.args.workers:
            kwargs.update(multiprocessing_context="spawn", prefetch_factor=4)
        self.iterator = iter(DataLoader(self.dataset, **kwargs))

    def take(self, count):
        batches = []
        while len(batches) < count:
            if self.iterator is None:
                if self.epochs is not None and self.epoch >= self.epochs:
                    break
                if self.token_limit is not None and self.token_limit() <= 0:
                    break
                self._open()
            try:
                batches.append(next(self.iterator))
                self.next_micro += 1
            except StopIteration:
                self.iterator = None
                self.epoch, self.next_micro = self.epoch + 1, 0
        return batches

    def close(self):
        self.iterator = None


def window_gradients(model, parameters, window, *, device, precision, totals):
    """Local sum-NLL gradients of one rank's microbatches at fixed parameters."""
    for parameter in parameters:
        parameter.grad = None
    local_tokens = 0
    for batch in window:
        local_tokens += int(batch["labels"].ne(-100).sum())
        inputs, labels = (batch[key].to(device, non_blocking=True) for key in ("input_ids", "labels"))
        with amp_context(device, precision):
            loss, correct = model(inputs, labels)
        loss.backward()
        totals[0] += loss.detach().to(torch.float64)
        totals[1] += correct.to(torch.float64)
    totals[2] += local_tokens
    gradients = [p.grad if p.grad is not None else torch.zeros_like(p) for p in parameters]
    for parameter in parameters:
        parameter.grad = None
    return gradients, local_tokens


def load_eval_subset(dataset, *, sequences, seed, args, device):
    """Cache this rank's shard of a fixed random subset on the device."""
    indices = np.random.Generator(np.random.PCG64(seed)).choice(
        len(dataset), size=min(sequences, len(dataset)), replace=False
    )
    shard = Subset(dataset, indices[distributed.rank()::distributed.world_size()].tolist())
    kwargs = {"batch_size": 64, "num_workers": args.workers}
    if args.workers:
        kwargs.update(multiprocessing_context="spawn")
    parts = list(DataLoader(shard, **kwargs))
    if not parts:
        return indices, None
    return indices, {key: torch.cat([part[key] for part in parts]).to(device) for key in ("input_ids", "labels")}


@torch.no_grad()
def evaluate_cached(model, cached, *, device, precision, batch_size):
    was_training = model.training
    model.eval()
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    if cached is not None:
        for start in range(0, len(cached["labels"]), batch_size):
            inputs = cached["input_ids"][start:start + batch_size]
            labels = cached["labels"][start:start + batch_size]
            with amp_context(device, precision):
                loss, correct = model(inputs, labels)
            totals += torch.stack((loss.detach(), correct, labels.ne(-100).sum())).to(torch.float64)
    distributed.sum_tensor(totals)
    model.train(was_training)
    nll, correct, tokens = totals.cpu().tolist()
    return {"loss": nll / tokens, "accuracy": correct / tokens, "target_tokens": int(tokens)}


class GapCheckpoints:
    """Resumable union of top-k accuracy-gap and loss-gap evaluation points.

    Same ranking rules as misalignment.checkpoints.GapCheckpointTracker
    (absolute train-val gaps, ties prefer the earlier point, a file is deleted
    only after it leaves both rankings), keyed by optimizer step instead of
    epoch and persisted so a resumed run keeps its selection.
    """

    def __init__(self, folder, model, *, top_k, resume_step=None):
        self.folder, self.model, self.top_k = Path(folder), model, top_k
        self.manifest_path = self.folder / "selection.json"
        self.points = {}
        if not distributed.is_main():
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        if resume_step is not None and self.manifest_path.exists():
            union = json.loads(self.manifest_path.read_text())["union"]
            self.points = {int(point["step"]): point for point in union}
            # A crash between observing a point and checkpointing may leave
            # points newer than the checkpoint; they will be observed again.
            for step in [step for step in self.points if step > resume_step]:
                (self.folder.parent / self.points.pop(step)["checkpoint"]).unlink(missing_ok=True)
        self._write()

    def _rankings(self, points):
        accuracy = sorted(points, key=lambda step: (-points[step]["abs_accuracy_gap"], step))[: self.top_k]
        loss = sorted(points, key=lambda step: (-points[step]["abs_loss_gap"], step))[: self.top_k]
        return accuracy, loss

    def observe(self, row):
        if not distributed.is_main():
            return
        step = int(row["step"])
        metrics = {key: float(row[key]) for key in ("loss_a", "loss_b", "train_accuracy", "val_accuracy")}
        if not all(math.isfinite(value) for value in metrics.values()):
            LOGGER.warning("Skipping non-finite gap checkpoint at step %d", step)
            return
        point = {
            "step": step,
            "epoch": float(row["epoch"]),
            "train_loss": metrics["loss_a"],
            "val_loss": metrics["loss_b"],
            "train_accuracy": metrics["train_accuracy"],
            "val_accuracy": metrics["val_accuracy"],
            "abs_accuracy_gap": float(row["abs_accuracy_gap"]),
            "abs_loss_gap": float(row["abs_loss_gap"]),
            "learning_rate": float(row["learning_rate"]),
            "checkpoint": f"{self.folder.name}/step_{step:07d}.model.pt",
        }
        candidates = {**self.points, step: point}
        accuracy, loss = self._rankings(candidates)
        retained = set(accuracy) | set(loss)
        if step not in retained:
            return
        state = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
        save_model_state(str(self.folder.parent / point["checkpoint"]), state, MODEL_TYPE, DATASET_TYPE)
        retired = [self.points[item]["checkpoint"] for item in set(self.points) - retained]
        self.points = {item: candidates[item] for item in retained}
        self._write()
        for name in retired:
            (self.folder.parent / name).unlink()

    def selection(self):
        accuracy, loss = self._rankings(self.points)
        points = {
            step: {
                **point,
                "accuracy_gap_rank": accuracy.index(step) + 1 if step in accuracy else None,
                "loss_gap_rank": loss.index(step) + 1 if step in loss else None,
            }
            for step, point in self.points.items()
        }
        return {
            "top_k": self.top_k,
            "gap_definition": "absolute_train_minus_val",
            "point_definition": "optimizer_step_after_update",
            "tie_break": "earlier_step_first",
            "top_accuracy_gap": [points[step] for step in accuracy],
            "top_loss_gap": [points[step] for step in loss],
            "union": [points[step] for step in sorted(points)],
        }

    def _write(self):
        atomic_json(self.manifest_path, self.selection())


@torch.no_grad()
def record_weight_variance(model, path, step):
    if not distributed.is_main():
        return
    state = model.state_dict()
    names = [name for name in state if "weight" in name]
    variances = torch.stack([
        torch.var(state[name].float(), correction=1) if state[name].numel() > 1
        else torch.tensor(float("nan"), device=state[name].device)
        for name in names
    ]).cpu().tolist()
    append_csv(path, ("step", *names), [{"step": step, **{n: f"{v:.3E}" for n, v in zip(names, variances)}}])


def save_training_checkpoint(path, raw_model, optimizer, *, progress, identity, device, interval):
    # Per rank: RNG plus the partial evaluation interval (local running loss
    # sums and the device-side gradient-geometry sums), so a resumed interval
    # reports the same averages as an uninterrupted one.
    state = {"rng": rng_state(device), "interval": {key: value.cpu() for key, value in interval.items()}}
    if distributed.enabled():
        states = [None] * distributed.world_size()
        torch.distributed.all_gather_object(states, state)
    else:
        states = [state]
    if distributed.is_main():
        temporary = path.with_suffix(".tmp")
        torch.save({
            "checkpoint_version": CHECKPOINT_VERSION,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "progress": progress,
            "identity": identity,
            "rank_states": states,
        }, temporary)
        os.replace(temporary, path)
        LOGGER.info("Checkpoint: step=%d train_tokens=%d path=%s", progress["step"], progress["train_tokens"], path)
    distributed.barrier()


def build_optimizer(raw_model, args):
    decay = [p for p in raw_model.parameters() if p.ndim >= 2]
    no_decay = [p for p in raw_model.parameters() if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=False,
    )


def run_direction(name, *, args, identity, raw_model, model, initial_state, eval_sets, datasets, device):
    """Train one orientation; returns its summary, or None if stopped early."""
    global STOP_REQUESTED
    train_name, val_name = DIRECTIONS[name]
    train_dataset, val_dataset = datasets[train_name], datasets[val_name]
    output = args.output
    paths = {
        "optimization": output / f"optimization_{name}.csv",
        "train_log": output / f"train_{name}.csv",
        "geometry": output / f"gradient_geometry_{name}.csv",
        "variance": output / f"weight_variance_{name}.csv",
        "checkpoint": output / f"training_checkpoint_{name}.pt",
        "initial": output / f"initial_{name}.model.pt",
        "final": output / f"final_{name}.model.pt",
        "summary": output / f"summary_{name}.json",
    }
    budget = min(train_dataset.target_tokens * args.epochs, args.max_tokens or train_dataset.target_tokens * args.epochs)
    seed_everything(args.seed)
    raw_model.load_state_dict(initial_state, strict=True)
    optimizer = build_optimizer(raw_model, args)
    progress = {"step": 0, "train_tokens": 0, "val_tokens": 0, "train_epoch": 0, "train_next_micro": 0,
                "val_epoch": 0, "val_next_micro": 0}
    resumed = args.resume and paths["checkpoint"].exists()
    rank_state = None
    if resumed:
        # Only load our own trusted artifacts (includes NumPy/Python RNG states).
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION or checkpoint["identity"] != identity:
            raise ValueError(f"{paths['checkpoint']} was written with different settings")
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        progress = checkpoint["progress"]
        rank_state = checkpoint["rank_states"][distributed.rank()]
        restore_rng(rank_state["rng"], device)
        del checkpoint
        if distributed.is_main():
            for key in ("optimization", "train_log", "geometry", "variance"):
                truncate_csv(paths[key], progress["step"])
        LOGGER.info("Resuming %s at step %d (%d/%d train tokens)", name, progress["step"],
                    progress["train_tokens"], budget)
    elif distributed.is_main():
        for key in ("optimization", "train_log", "geometry", "variance"):
            paths[key].unlink(missing_ok=True)
        save_model_state(str(paths["initial"]), {k: v.cpu().clone() for k, v in initial_state.items()},
                         MODEL_TYPE, DATASET_TYPE)
    tracker = GapCheckpoints(output / f"gap_checkpoints_{name}", raw_model, top_k=args.gap_top_k,
                             resume_step=progress["step"] if resumed else None)
    parameters = [p for p in raw_model.parameters() if p.requires_grad]
    parameter_names = [n for n, p in raw_model.named_parameters() if p.requires_grad]
    combiner = NormalizedGradientCombiner(parameters, enabled=args.compile and device.type == "cuda")
    weights = {"train_gradient_weight": args.train_gradient_weight, "val_gradient_weight": args.val_gradient_weight}
    train_stream = PartitionStream(
        train_dataset, args=args, device=device, seed=args.seed + ORDER_SEED_OFFSET[train_name],
        epoch=progress["train_epoch"], next_micro=progress["train_next_micro"], epochs=args.epochs,
        token_limit=lambda: budget - progress["train_tokens"],
    )
    val_stream = PartitionStream(
        val_dataset, args=args, device=device, seed=args.seed + ORDER_SEED_OFFSET[val_name],
        epoch=progress["val_epoch"], next_micro=progress["val_next_micro"],
    )

    def evaluation_row(running, geometry, tokens_per_second):
        start = time.monotonic()
        train = evaluate_cached(raw_model, eval_sets[train_name], device=device, precision=args.precision,
                                batch_size=args.eval_batch_size)
        val = evaluate_cached(raw_model, eval_sets[val_name], device=device, precision=args.precision,
                              batch_size=args.eval_batch_size)
        cosines = [float(row["train_val_gradient_cosine"]) for row in geometry]
        cosines = [value for value in cosines if math.isfinite(value)]
        loss_sum = train["loss"] + val["loss"]
        return {
            "step": progress["step"],
            "epoch": progress["train_tokens"] / train_dataset.target_tokens,
            "train_target_tokens": progress["train_tokens"],
            "loss_a": train["loss"],
            "loss_b": val["loss"],
            "train_accuracy": train["accuracy"],
            "val_accuracy": val["accuracy"],
            "mean_loss": 0.5 * loss_sum,
            "abs_accuracy_gap": abs(train["accuracy"] - val["accuracy"]),
            "abs_loss_gap": abs(train["loss"] - val["loss"]),
            "relative_gap": abs(train["loss"] - val["loss"]) / (loss_sum + OBJECTIVE_OFFSET),
            "objective": train["loss"] - val["loss"],
            "learning_rate": optimizer.param_groups[0]["lr"] if progress["step"] else 0.0,
            "train_val_gradient_cosine": sum(cosines) / len(cosines) if cosines else float("nan"),
            **weights,
            **running,
            "tokens_per_second": tokens_per_second,
            "eval_seconds": time.monotonic() - start,
        }

    def write_evaluation(row, geometry):
        if distributed.is_main():
            row = {key: row[key] for key in OPTIMIZATION_FIELDS}
            append_csv(paths["optimization"], OPTIMIZATION_FIELDS, [row])
            if geometry:
                append_csv(paths["geometry"], GEOMETRY_FIELDS,
                           [{"step": row["step"], **stats, **weights} for stats in geometry])
            LOGGER.info(
                "%s eval step=%d epoch=%.4f train_loss=%.5f val_loss=%.5f train_acc=%.5f val_acc=%.5f "
                "acc_gap=%.5f loss_gap=%.5f cosine=%.4f",
                name, row["step"], row["epoch"], row["loss_a"], row["loss_b"], row["train_accuracy"],
                row["val_accuracy"], row["abs_accuracy_gap"], row["abs_loss_gap"], row["train_val_gradient_cosine"],
            )
        record_weight_variance(raw_model, paths["variance"], progress["step"])

    # interval_totals columns: train nll/correct/tokens, val nll/correct/tokens.
    interval_totals = torch.zeros(6, device=device, dtype=torch.float64)
    geometry = GradientGeometryAccumulator(parameter_names, GRADIENT_GEOMETRY_FIELDS, device)
    if rank_state is not None:
        interval_totals.copy_(rank_state["interval"]["totals"])
        geometry.sums.copy_(rank_state["interval"]["geometry_sums"])
        geometry.counts.copy_(rank_state["interval"]["geometry_counts"])

    def checkpoint():
        save_training_checkpoint(
            paths["checkpoint"], raw_model, optimizer, progress=progress.copy(), identity=identity, device=device,
            interval={"totals": interval_totals, "geometry_sums": geometry.sums, "geometry_counts": geometry.counts},
        )

    if not resumed:
        running = {key: float("nan") for key in
                   ("running_train_loss", "running_val_loss", "running_train_accuracy", "running_val_accuracy")}
        write_evaluation(evaluation_row(running, [], float("nan")), [])
        checkpoint()
    LOGGER.info("%s: train=%s val=%s; train target tokens=%d; weights train=%g val=%g", name, train_name,
                val_name, budget, args.train_gradient_weight, args.val_gradient_weight)
    log_totals = torch.zeros(6, device=device, dtype=torch.float64)
    log_start = interval_start = time.monotonic()
    if args.max_steps is not None and progress["step"] >= args.max_steps and progress["train_tokens"] < budget:
        LOGGER.info("%s already at --max-steps %d", name, args.max_steps)
        return None
    model.train()
    try:
        while progress["train_tokens"] < budget:
            train_window = train_stream.take(args.gradient_accumulation)
            if not train_window:
                break
            val_window = val_stream.take(len(train_window))
            step_totals = torch.zeros(6, device=device, dtype=torch.float64)
            train_gradients, train_local = window_gradients(
                model, parameters, train_window, device=device, precision=args.precision, totals=step_totals[:3])
            val_gradients, val_local = window_gradients(
                model, parameters, val_window, device=device, precision=args.precision, totals=step_totals[3:])
            counts = torch.tensor([train_local, val_local], dtype=torch.int64, device=device)
            distributed.sum_tensor(counts)
            train_tokens, val_tokens = (int(value) for value in counts.tolist())
            if not train_tokens or not val_tokens:
                raise ValueError("Optimizer window contains no prediction targets")
            # Sum over ranks, then divide by ACTUAL global target tokens.
            for gradients, tokens in ((train_gradients, train_tokens), (val_gradients, val_tokens)):
                distributed.reduce_gradients(gradients, average=False)
                torch._foreach_div_(gradients, float(tokens))
            lr = scheduled_lr(progress["train_tokens"] + train_tokens, budget, peak=args.learning_rate,
                              minimum=args.min_learning_rate, warmup_fraction=args.warmup_fraction)
            for group in optimizer.param_groups:
                group["lr"] = lr
            combined, stats = combiner(train_gradients, val_gradients, train_weight=args.train_gradient_weight,
                                       val_weight=args.val_gradient_weight, collect_geometry=True)
            geometry.update(stats)
            del train_gradients, val_gradients
            for parameter, gradient in zip(parameters, combined, strict=True):
                parameter.grad = gradient
            # Clipping the combined gradient preserves the normal recipe: its
            # per-tensor norms equal the train gradient's.
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad_norm)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError("Non-finite combined language-model gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            interval_totals += step_totals
            log_totals += step_totals
            progress["step"] += 1
            progress["train_tokens"] += train_tokens
            progress["val_tokens"] += val_tokens
            progress["train_epoch"], progress["train_next_micro"] = train_stream.epoch, train_stream.next_micro
            progress["val_epoch"], progress["val_next_micro"] = val_stream.epoch, val_stream.next_micro
            finished = progress["train_tokens"] >= budget
            if progress["step"] % args.log_interval == 0 or finished:
                distributed.sum_tensor(log_totals)
                values = log_totals.cpu().tolist()
                elapsed = max(time.monotonic() - log_start, 1e-9)
                log_row = {
                    "step": progress["step"], "train_target_tokens": progress["train_tokens"],
                    "train_nll": values[0] / values[2], "train_accuracy": values[1] / values[2],
                    "val_nll": values[3] / values[5], "val_accuracy": values[4] / values[5],
                    "learning_rate": lr, "tokens_per_second": (values[2] + values[5]) / elapsed,
                }
                if distributed.is_main():
                    append_csv(paths["train_log"], TRAIN_LOG_FIELDS, [log_row])
                    LOGGER.info("%s step=%d tokens=%d/%d train_nll=%.5f val_nll=%.5f train_acc=%.4f val_acc=%.4f "
                                "lr=%.3g tokens/s=%.0f", name, log_row["step"], log_row["train_target_tokens"], budget,
                                log_row["train_nll"], log_row["val_nll"], log_row["train_accuracy"],
                                log_row["val_accuracy"], lr, log_row["tokens_per_second"])
                log_totals.zero_()
                log_start = time.monotonic()
            stop_tensor = torch.tensor(int(STOP_REQUESTED), dtype=torch.int64, device=device)
            distributed.sum_tensor(stop_tensor)
            stopped = bool(stop_tensor.item()) or (args.max_steps is not None and progress["step"] >= args.max_steps)
            if progress["step"] % args.eval_interval == 0 or finished:
                pause = time.monotonic()
                distributed.sum_tensor(interval_totals)
                values = interval_totals.cpu().tolist()
                running = {
                    "running_train_loss": values[0] / values[2], "running_val_loss": values[3] / values[5],
                    "running_train_accuracy": values[1] / values[2], "running_val_accuracy": values[4] / values[5],
                }
                stats_rows = geometry.averages()
                row = evaluation_row(running, stats_rows, (values[2] + values[5]) / max(pause - interval_start, 1e-9))
                write_evaluation(row, stats_rows)
                tracker.observe(row)
                interval_totals.zero_()
                geometry = GradientGeometryAccumulator(parameter_names, GRADIENT_GEOMETRY_FIELDS, device)
                checkpoint()
                log_start += time.monotonic() - pause
                interval_start = time.monotonic()
            elif stopped:
                checkpoint()
            if stopped and not finished:
                LOGGER.info("%s stopped at step %d; resume with --resume", name, progress["step"])
                return None
    finally:
        train_stream.close()
        val_stream.close()
    if progress["train_tokens"] < budget:
        raise RuntimeError(f"{name}: train partition exhausted before the token budget")
    if distributed.is_main():
        save_model_state(str(paths["final"]), {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()},
                         MODEL_TYPE, DATASET_TYPE)
        with paths["optimization"].open(newline="") as stream:
            last = list(csv.DictReader(stream))[-1]
        summary = {
            "direction": name,
            "train_partition": train_name,
            "val_partition": val_name,
            "complete": True,
            "progress": progress,
            "token_budget": budget,
            "final": {key: float(value) for key, value in last.items()},
            "gap_checkpoints": tracker.selection(),
            "gradient_combination": combiner.runtime_info(),
            "files": {key: path.name for key, path in paths.items() if key != "checkpoint"},
        }
        atomic_json(paths["summary"], summary)
    distributed.barrier()
    return json.loads(paths["summary"].read_text())


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--directions", choices=("both", *DIRECTIONS), default="both")
    result.add_argument("--train-gradient-weight", type=float, default=0.8)
    result.add_argument("--val-gradient-weight", type=float, default=0.2)
    result.add_argument("--sequence-length", type=int, default=2048)
    result.add_argument("--epochs", type=int, default=1, help="Passes over the train partition")
    result.add_argument("--max-tokens", type=int, help="Train-partition target-token cap per orientation")
    result.add_argument("--micro-batch-size", type=int, default=16, help="Sequences per GPU per partition")
    result.add_argument("--gradient-accumulation", type=int, default=16)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--min-learning-rate", type=float, default=1e-4)
    result.add_argument("--warmup-fraction", type=float, default=0.01)
    result.add_argument("--weight-decay", type=float, default=0.1)
    result.add_argument("--clip-grad-norm", type=float, default=1.0)
    result.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    result.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    result.add_argument("--workers", type=int, default=0, help="DataLoader workers per GPU per partition stream")
    result.add_argument("--seed", type=int, default=42, help="Initialization and data-order seed")
    result.add_argument("--eval-interval", type=int, default=500, help="Optimizer steps between evaluation points")
    result.add_argument("--eval-sequences", type=int, default=8192, help="Fixed evaluation sequences per partition")
    result.add_argument("--eval-seed", type=int, default=2026)
    result.add_argument("--eval-batch-size", type=int, default=16)
    result.add_argument("--gap-top-k", type=int, default=5)
    result.add_argument("--log-interval", type=int, default=10)
    result.add_argument("--max-steps", type=int, help="Stop resumably at this step of the current orientation")
    result.add_argument("--resume", action="store_true", help="Continue the run in --output")
    result.add_argument("--cpu", action="store_true")
    result.add_argument("--model-config", type=Path, help="Local GPTNeoX architecture JSON only; no weights")
    return result


def train(args):
    global STOP_REQUESTED
    STOP_REQUESTED = False
    for name in ("sequence_length", "epochs", "micro_batch_size", "gradient_accumulation", "eval_interval",
                 "eval_sequences", "eval_batch_size", "gap_top_k", "log_interval"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("train_gradient_weight", "val_gradient_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if args.workers < 0 or args.seed < 0 or args.weight_decay < 0 or args.clip_grad_norm <= 0:
        raise ValueError("Invalid worker count, seed, weight decay, or clipping norm")
    if not 0 <= args.warmup_fraction < 1 or not 0 <= args.min_learning_rate <= args.learning_rate:
        raise ValueError("Invalid learning-rate schedule")
    if (args.max_tokens is not None and args.max_tokens <= 0) or (args.max_steps is not None and args.max_steps <= 0):
        raise ValueError("Token/step caps must be positive")
    device = distributed.initialize(use_cpu=args.cpu)
    device = device or torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Native BF16 is required; use --precision fp32 otherwise")
    elif not args.cpu:
        raise RuntimeError("CUDA unavailable; pass --cpu explicitly for tests")
    logging.basicConfig(level=logging.INFO if distributed.is_main() else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    # Seed BEFORE constructing the model, in the same order as train_pythia.py,
    # so the same seed reproduces the ordinary run's initial weights.
    seed_everything(args.seed)
    datasets = {name: PileTokenDataset(args.data, partitions=name, sequence_length=args.sequence_length)
                for name in ("A", "B")}
    config = None
    if args.model_config:
        from transformers import GPTNeoXConfig

        config = GPTNeoXConfig.from_json_file(str(args.model_config))
    raw_model = build_pythia(config, attention=args.attention).to(device)
    config = raw_model.model.config
    if config.vocab_size != datasets["A"].manifest["vocabulary_size"]:
        raise ValueError("Model and prepared dataset vocabulary sizes differ")
    if args.sequence_length > config.max_position_embeddings:
        raise ValueError("Sequence length exceeds the architecture's context limit")
    initial_state = {name: value.detach().clone() for name, value in raw_model.state_dict().items()}
    directions = list(DIRECTIONS) if args.directions == "both" else [args.directions]
    identity = {
        "dataset": datasets["A"].fingerprint,
        "objective": "normalized_two_sided",
        "train_gradient_weight": args.train_gradient_weight,
        "val_gradient_weight": args.val_gradient_weight,
        "sequence_length": args.sequence_length,
        "epochs": args.epochs,
        "max_tokens": args.max_tokens,
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
        "eval_interval": args.eval_interval,
        "eval_sequences": args.eval_sequences,
        "eval_seed": args.eval_seed,
        "gap_top_k": args.gap_top_k,
        "model_config": json.loads(config.to_json_string(use_diff=False)),
        "initialization": "NeoX small_init + wang_init residual projections",
    }
    problem = None
    if distributed.is_main():
        info = args.output / "info.json"
        if args.resume:
            if not info.exists():
                problem = "--resume needs an existing run in --output"
            elif json.loads(info.read_text())["identity"] != identity:
                problem = "Run in --output used different settings; resume needs identical settings"
        elif info.exists():
            problem = "Output already contains a run; use --resume or a new output directory"
        else:
            args.output.mkdir(parents=True, exist_ok=True)
            atomic_json(info, {
                "identity": identity,
                "directions": directions,
                "parameters": sum(p.numel() for p in raw_model.parameters()),
                "pretrained_weights": False,
                "partition_tokens": {name: dataset.target_tokens for name, dataset in datasets.items()},
                "evaluation": "fixed random subsets of each partition (training data of one side, "
                              "ascended data of the other); not held-out data",
                "csv_convention": "loss_a/train_accuracy = train partition, loss_b/val_accuracy = val partition",
            })
            (args.output / "command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n")
    problem = distributed.broadcast_object(problem)
    if problem:
        raise ValueError(problem)
    distributed.barrier()
    compiled = torch.compile(raw_model) if args.compile and device.type == "cuda" else raw_model

    def request_stop(signum, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True

    for signum in (signal.SIGTERM, getattr(signal, "SIGUSR1", signal.SIGTERM)):
        signal.signal(signum, request_stop)
    try:
        eval_sets = {}
        for name, dataset in datasets.items():
            indices, eval_sets[name] = load_eval_subset(
                dataset, sequences=args.eval_sequences, seed=args.eval_seed + EVAL_SEED_OFFSET[name],
                args=args, device=device)
            if distributed.is_main():
                np.save(args.output / f"eval_indices_{name}.npy", indices)
        LOGGER.info("Pythia parameters=%d; partitions A=%d B=%d target tokens; effective sequences per partition=%d; "
                    "eval sequences per partition=%d every %d steps", sum(p.numel() for p in raw_model.parameters()),
                    datasets["A"].target_tokens, datasets["B"].target_tokens,
                    args.micro_batch_size * distributed.world_size() * args.gradient_accumulation,
                    args.eval_sequences, args.eval_interval)
        runs = {}
        for name in directions:
            summary_path = args.output / f"summary_{name}.json"
            if args.resume and summary_path.exists():
                LOGGER.info("%s already finished; skipping", name)
                runs[name] = json.loads(summary_path.read_text())
                continue
            result = run_direction(name, args=args, identity=identity, raw_model=raw_model, model=compiled,
                                   initial_state=initial_state, eval_sets=eval_sets, datasets=datasets, device=device)
            if result is None:
                return None
            runs[name] = result
        if distributed.is_main():
            atomic_json(args.output / "summary.json", {
                "objective": "normalized_two_sided",
                "run_order": list(runs),
                "final_absolute_accuracy_gap": {name: run["final"]["abs_accuracy_gap"] for name, run in runs.items()},
                "final_absolute_loss_gap": {name: run["final"]["abs_loss_gap"] for name, run in runs.items()},
                "runs": runs,
            })
        distributed.barrier()
        return runs
    finally:
        distributed.shutdown()


def main(argv=None):
    train(parser().parse_args(argv))


if __name__ == "__main__":
    main()
