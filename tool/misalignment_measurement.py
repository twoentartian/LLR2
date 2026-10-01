from __future__ import annotations

"""Measure train/validation low-loss-region misalignment.

The experiment has three stages:

1. Train a Grokking Transformer on the training partition only.
2. Starting from that train-only solution, search for a nearby model that is
   low-loss on both partitions.  The layer-normalized distance to the best
   feasible candidate found along this optimization path is an upper-bound/
   local proxy for the distance from the train-only region to the overlap
   region; it is not a global shortest-distance computation.
3. Around the train-only solution, estimate the local fraction of perturbations
   that remain train-low-loss but are validation-high-loss.  This is a local
   conditional-volume proxy for the non-overlap region.  Finite-radius loss
   increases are reported as auxiliary sharpness values.
4. Estimate Petzka et al.'s layer-wise relative flatness on train and
   validation losses with Hutchinson Hessian-vector products.  Relative
   flatness is reported as an auxiliary, reparameterization-aware curvature
   statistic rather than as a direct alignment measure.

The script deliberately does not call a train-fit/validation-mismatch point a
boundary point.  The thresholds supplied on the command line define the
low-loss regions explicitly, and every reported metric is checked against
those thresholds.
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_grokking import loading_dataset_from

from py_src.ml_setup.grokking import build_grokking_model
from py_src.model_opti_save_load import save_model_state
from py_src.util import set_seed, setup_logging

logger = logging.getLogger("misalignment_measurement")


@dataclass
class DatasetMetrics:
    loss: float
    accuracy: float
    low_loss: bool


@dataclass
class LocalProbeResult:
    radius: float
    samples: int
    train_low_fraction: float
    val_low_fraction: float
    overlap_fraction: float
    train_only_fraction: float
    conditional_train_only_fraction: float
    train_finite_radius_sharpness: float
    val_finite_radius_sharpness: float


@dataclass
class RelativeFlatnessResult:
    layer: str
    weight_shape: list[int]
    examples: int
    hutchinson_samples: int
    loss: float
    accuracy: float
    weight_norm: float
    gradient_norm: float
    signed_estimate: float
    sample_std: float
    standard_error: float
    positive_part: float


def _parse_radii(value: str) -> list[float]:
    radii = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not radii or any(radius < 0 for radius in radii):
        raise ValueError("--local_radii must be a comma-separated list of non-negative values")
    return radii


def _parse_layer_names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("--relative_flatness_layers must contain at least one layer name")
    if "all_matrix_weights" in names and len(names) != 1:
        raise ValueError("all_matrix_weights cannot be combined with explicit layer names")
    return names


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _load_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def _batch_for_dataset(dataset, device: torch.device) -> dict[str, torch.Tensor]:
    if len(dataset) == 0:
        raise ValueError(f"Dataset {dataset.name!r} is empty")
    data = dataset.data.to(device)
    return {"text": data[:, :-1], "target": data[:, 1:]}


def _rhs_logits_and_targets(model, batch: dict[str, torch.Tensor], tokenizer):
    logits, _, _ = model(x=batch["text"])
    logits = logits.transpose(-2, -1)
    eq_token_index = tokenizer.stoi["="]
    eq_positions = torch.nonzero(batch["target"][0] == eq_token_index, as_tuple=False).flatten()
    if eq_positions.numel() != 1:
        raise ValueError("Expected exactly one '=' token in each arithmetic sequence")
    eq_position = int(eq_positions.item())
    return logits[..., eq_position + 1 :], batch["target"][..., eq_position + 1 :]


def _loss_and_accuracy(model, batch, tokenizer, *, enable_grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if enable_grad:
        logits_rhs, targets_rhs = _rhs_logits_and_targets(model, batch, tokenizer)
        loss = F.cross_entropy(logits_rhs, targets_rhs, reduction="mean")
    else:
        with torch.inference_mode():
            logits_rhs, targets_rhs = _rhs_logits_and_targets(model, batch, tokenizer)
            loss = F.cross_entropy(logits_rhs, targets_rhs, reduction="mean")
    predictions = logits_rhs.argmax(dim=-2)
    accuracy = (predictions == targets_rhs).all(dim=-1).float().mean()
    return loss, accuracy


def _metrics(model, batch, tokenizer, threshold: float) -> DatasetMetrics:
    model.eval()
    loss, accuracy = _loss_and_accuracy(model, batch, tokenizer, enable_grad=False)
    loss_value = float(loss.item())
    return DatasetMetrics(
        loss=loss_value,
        accuracy=float(accuracy.item()),
        low_loss=loss_value <= threshold,
    )


def _make_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    warmup_steps = max(0, warmup_steps)
    total_steps = max(1, total_steps)

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        cosine_steps = max(1, total_steps - warmup_steps)
        cosine_step = min(max(0, step - warmup_steps), cosine_steps)
        progress = cosine_step / cosine_steps
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _write_csv(path: str, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(list(header))
        writer.writerows(rows)


def train_train_only(
    model,
    train_batch,
    val_batch,
    tokenizer,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    min_lr: float,
    warmup_epochs: int,
    train_threshold: float,
    accuracy_threshold: float,
    report_interval: int,
    log_path: str,
):
    if epochs <= 0:
        raise ValueError("--fit_epoch must be positive")
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    scheduler = _make_scheduler(
        optimizer,
        warmup_epochs,
        epochs,
        min_lr / learning_rate if learning_rate > 0 else 0.0,
    )
    rows: list[list[object]] = []
    best_state = _clone_state(model)
    best_train_loss = float("inf")
    success_epoch: int | None = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss, _ = _loss_and_accuracy(model, train_batch, tokenizer, enable_grad=True)
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % max(1, report_interval) == 0 or epoch == epochs - 1:
            train_metrics = _metrics(model, train_batch, tokenizer, train_threshold)
            val_metrics = _metrics(model, val_batch, tokenizer, float("inf"))
            rows.append(
                [
                    epoch,
                    train_metrics.loss,
                    train_metrics.accuracy,
                    val_metrics.loss,
                    val_metrics.accuracy,
                    optimizer.param_groups[0]["lr"],
                ]
            )
            if train_metrics.loss < best_train_loss:
                best_train_loss = train_metrics.loss
                best_state = _clone_state(model)
            if (
                success_epoch is None
                and train_metrics.low_loss
                and train_metrics.accuracy >= accuracy_threshold
            ):
                success_epoch = epoch
                logger.info(
                    "train-only fit threshold reached at epoch %d: loss=%.4g accuracy=%.4f",
                    epoch,
                    train_metrics.loss,
                    train_metrics.accuracy,
                )
                break

    _load_state(model, best_state)
    _write_csv(
        log_path,
        ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate"],
        rows,
    )
    return best_state, success_epoch, rows


def _normalized_distance(
    reference_state: dict[str, torch.Tensor],
    candidate_state: dict[str, torch.Tensor],
    floor: float,
) -> float:
    ratios: list[torch.Tensor] = []
    for name, reference in reference_state.items():
        candidate = candidate_state[name]
        if not torch.is_floating_point(reference):
            continue
        reference_norm = reference.norm()
        candidate_norm = (candidate - reference).norm()
        ratios.append(candidate_norm / torch.clamp(reference_norm, min=floor))
    if not ratios:
        return 0.0
    return float(torch.stack(ratios).square().mean().sqrt().item())


def _relative_random_perturbation(
    model: torch.nn.Module,
    base_state: dict[str, torch.Tensor],
    radius: float,
    floor: float,
) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            base = base_state[name].to(device=parameter.device, dtype=parameter.dtype)
            noise = torch.randn_like(parameter)
            noise_norm = noise.norm()
            if noise_norm == 0:
                continue
            base_norm = torch.clamp(base.norm(), min=floor)
            parameter.copy_(base + noise * (radius * base_norm / noise_norm))


def probe_local_geometry(
    model,
    train_batch,
    val_batch,
    tokenizer,
    base_state: dict[str, torch.Tensor],
    *,
    radii: list[float],
    samples: int,
    train_threshold: float,
    val_threshold: float,
    normalization_floor: float,
    csv_path: str,
) -> list[LocalProbeResult]:
    if samples <= 0:
        raise ValueError("--local_samples must be positive")
    results: list[LocalProbeResult] = []
    model.eval()
    for radius in radii:
        train_low = 0
        val_low = 0
        overlap = 0
        train_only = 0
        train_sharpness = 0.0
        val_sharpness = 0.0
        _load_state(model, base_state)
        base_train = _metrics(model, train_batch, tokenizer, train_threshold)
        base_val = _metrics(model, val_batch, tokenizer, val_threshold)
        for _ in range(samples):
            _load_state(model, base_state)
            _relative_random_perturbation(model, base_state, radius, normalization_floor)
            perturbed_train = _metrics(model, train_batch, tokenizer, train_threshold)
            perturbed_val = _metrics(model, val_batch, tokenizer, val_threshold)
            train_is_low = perturbed_train.low_loss
            val_is_low = perturbed_val.low_loss
            train_low += int(train_is_low)
            val_low += int(val_is_low)
            overlap += int(train_is_low and val_is_low)
            train_only += int(train_is_low and not val_is_low)
            train_sharpness = max(train_sharpness, perturbed_train.loss - base_train.loss)
            val_sharpness = max(val_sharpness, perturbed_val.loss - base_val.loss)
        result = LocalProbeResult(
            radius=radius,
            samples=samples,
            train_low_fraction=train_low / samples,
            val_low_fraction=val_low / samples,
            overlap_fraction=overlap / samples,
            train_only_fraction=train_only / samples,
            conditional_train_only_fraction=train_only / train_low if train_low else float("nan"),
            train_finite_radius_sharpness=train_sharpness,
            val_finite_radius_sharpness=val_sharpness,
        )
        results.append(result)
        logger.info(
            "local radius %.4g: train-low=%.3f val-low=%.3f overlap=%.3f conditional-train-only=%.3f",
            radius,
            result.train_low_fraction,
            result.val_low_fraction,
            result.overlap_fraction,
            result.conditional_train_only_fraction,
        )
    _load_state(model, base_state)
    _write_csv(
        csv_path,
        list(asdict(results[0]).keys()) if results else [],
        ([asdict(result).values() for result in results] if results else []),
    )
    return results


def _resolve_relative_flatness_layers(model: torch.nn.Module, requested: list[str]) -> list[str]:
    parameters = dict(model.named_parameters())
    if requested == ["all_matrix_weights"]:
        names = [
            name
            for name, parameter in parameters.items()
            if name.endswith(".weight") and name != "embedding.weight" and parameter.ndim == 2
        ]
    else:
        names = requested
    missing = [name for name in names if name not in parameters]
    if missing:
        available = [name for name, parameter in parameters.items() if parameter.ndim == 2]
        raise ValueError(
            f"Unknown relative-flatness layer(s): {missing}. Available matrix parameters: {available}"
        )
    invalid = [name for name in names if parameters[name].ndim != 2]
    if invalid:
        raise ValueError(f"Relative flatness requires rank-2 weight matrices, got: {invalid}")
    return names


def _subset_batch(batch: dict[str, torch.Tensor], max_examples: int, seed: int) -> dict[str, torch.Tensor]:
    size = int(batch["text"].shape[0])
    if max_examples <= 0 or max_examples >= size:
        return batch
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(size, generator=generator)[:max_examples].to(batch["text"].device)
    return {key: value[indices] for key, value in batch.items()}


def _rademacher_like(parameter: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    bits = torch.randint(
        0,
        2,
        parameter.shape,
        device=parameter.device,
        generator=generator,
        dtype=torch.int8,
    )
    return bits.to(dtype=parameter.dtype).mul_(2).sub_(1)


def measure_relative_flatness(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    tokenizer,
    *,
    layer_names: list[str],
    hutchinson_samples: int,
    seed: int,
) -> list[RelativeFlatnessResult]:
    """Estimate Definition 3 of Petzka et al. with Hessian-vector products.

    For a selected weight matrix W with row-block Hessian H, relative flatness
    is Tr((W W^T \u2297 I) H).  A Rademacher probe Z gives the unbiased estimate
    <(W W^T) Z, H Z>.  This avoids materializing the full Hessian.
    """
    if hutchinson_samples <= 0:
        raise ValueError("--relative_flatness_samples must be positive")
    parameters = dict(model.named_parameters())
    selected_parameters = [parameters[name] for name in layer_names]
    model.eval()
    model.zero_grad(set_to_none=True)
    loss, accuracy = _loss_and_accuracy(model, batch, tokenizer, enable_grad=True)
    gradients = torch.autograd.grad(
        loss,
        selected_parameters,
        create_graph=True,
        retain_graph=True,
    )
    results: list[RelativeFlatnessResult] = []
    for layer_index, (name, parameter, gradient) in enumerate(
        zip(layer_names, selected_parameters, gradients, strict=True)
    ):
        generator = torch.Generator(device=parameter.device)
        generator.manual_seed(seed + 1_000_003 * layer_index)
        estimates: list[float] = []
        weight = parameter.detach()
        for _ in range(hutchinson_samples):
            probe = _rademacher_like(parameter, generator)
            hessian_probe = torch.autograd.grad(
                (gradient * probe).sum(),
                parameter,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if hessian_probe is None:
                estimate = torch.zeros((), device=parameter.device, dtype=parameter.dtype)
            else:
                relative_probe = weight @ (weight.transpose(0, 1) @ probe)
                estimate = (relative_probe * hessian_probe).sum()
            estimates.append(float(estimate.detach().item()))

        estimates_tensor = torch.tensor(estimates, dtype=torch.float64)
        signed_estimate = float(estimates_tensor.mean().item())
        if hutchinson_samples > 1:
            sample_std = float(estimates_tensor.std(unbiased=True).item())
            standard_error = sample_std / math.sqrt(hutchinson_samples)
        else:
            sample_std = 0.0
            standard_error = 0.0
        result = RelativeFlatnessResult(
            layer=name,
            weight_shape=list(parameter.shape),
            examples=int(batch["text"].shape[0]),
            hutchinson_samples=hutchinson_samples,
            loss=float(loss.detach().item()),
            accuracy=float(accuracy.detach().item()),
            weight_norm=float(weight.norm().item()),
            gradient_norm=float(gradient.detach().norm().item()),
            signed_estimate=signed_estimate,
            sample_std=sample_std,
            standard_error=standard_error,
            positive_part=max(0.0, signed_estimate),
        )
        results.append(result)
        logger.info(
            "relative flatness layer=%s estimate=%.6g stderr=%.3g gradient_norm=%.3g",
            name,
            result.signed_estimate,
            result.standard_error,
            result.gradient_norm,
        )
    del gradients
    del loss
    model.zero_grad(set_to_none=True)
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def measure_checkpoint_relative_flatness(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    train_batch: dict[str, torch.Tensor],
    val_batch: dict[str, torch.Tensor],
    tokenizer,
    *,
    layer_names: list[str],
    hutchinson_samples: int,
    max_examples: int,
    seed: int,
) -> dict[str, list[RelativeFlatnessResult]]:
    _load_state(model, state)
    train_subset = _subset_batch(train_batch, max_examples, seed + 11)
    val_subset = _subset_batch(val_batch, max_examples, seed + 29)
    logger.info(
        "measuring relative flatness on %d train and %d val examples",
        train_subset["text"].shape[0],
        val_subset["text"].shape[0],
    )
    train_results = measure_relative_flatness(
        model,
        train_subset,
        tokenizer,
        layer_names=layer_names,
        hutchinson_samples=hutchinson_samples,
        seed=seed,
    )
    val_results = measure_relative_flatness(
        model,
        val_subset,
        tokenizer,
        layer_names=layer_names,
        hutchinson_samples=hutchinson_samples,
        seed=seed,
    )
    _load_state(model, state)
    return {"train": train_results, "val": val_results}


def search_nearest_overlap(
    model,
    train_batch,
    val_batch,
    tokenizer,
    train_state: dict[str, torch.Tensor],
    *,
    epochs: int,
    learning_rate: float,
    train_threshold: float,
    val_threshold: float,
    penalty: float,
    normalization_floor: float,
    report_interval: int,
    csv_path: str,
):
    if epochs <= 0:
        raise ValueError("--overlap_epoch must be positive")
    _load_state(model, train_state)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.98))
    best_feasible_state: dict[str, torch.Tensor] | None = None
    best_feasible_distance = float("inf")
    best_violation_state = _clone_state(model)
    best_violation = float("inf")
    rows: list[list[object]] = []

    for epoch in range(epochs):
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        train_loss, _ = _loss_and_accuracy(model, train_batch, tokenizer, enable_grad=True)
        val_loss, _ = _loss_and_accuracy(model, val_batch, tokenizer, enable_grad=True)
        train_violation = F.relu(train_loss / max(train_threshold, normalization_floor) - 1.0)
        objective = val_loss / max(val_threshold, normalization_floor) + penalty * train_violation.square()
        objective.backward()
        optimizer.step()

        # Check the updated parameters every epoch. In particular, do not
        # wait for ``report_interval``: with a large reporting interval the
        # optimizer can enter the overlap region thousands of steps before
        # the old loop notices it. These are inference-only passes, so the
        # stopping check does not affect the overlap-search gradients.
        updated_train_loss, updated_train_accuracy = _loss_and_accuracy(
            model, train_batch, tokenizer, enable_grad=False
        )
        updated_val_loss, updated_val_accuracy = _loss_and_accuracy(
            model, val_batch, tokenizer, enable_grad=False
        )
        train_metrics = DatasetMetrics(
            loss=float(updated_train_loss.item()),
            accuracy=float(updated_train_accuracy.item()),
            low_loss=float(updated_train_loss.item()) <= train_threshold,
        )
        val_metrics = DatasetMetrics(
            loss=float(updated_val_loss.item()),
            accuracy=float(updated_val_accuracy.item()),
            low_loss=float(updated_val_loss.item()) <= val_threshold,
        )
        feasible_now = train_metrics.low_loss and val_metrics.low_loss
        should_report = epoch % max(1, report_interval) == 0 or epoch == epochs - 1

        if should_report or feasible_now:
            state = _clone_state(model)
            distance = _normalized_distance(train_state, state, normalization_floor)
            violation = max(0.0, train_metrics.loss / max(train_threshold, normalization_floor) - 1.0) ** 2
            violation += max(0.0, val_metrics.loss / max(val_threshold, normalization_floor) - 1.0) ** 2
            rows.append([epoch, objective.item(), train_metrics.loss, val_metrics.loss, distance, violation])
            if violation < best_violation:
                best_violation = violation
                best_violation_state = state
            if train_metrics.low_loss and val_metrics.low_loss and distance < best_feasible_distance:
                best_feasible_distance = distance
                best_feasible_state = state
            if epoch % max(1, report_interval * 10) == 0 or epoch == epochs - 1:
                logger.info(
                    "overlap search epoch %d: train_loss=%.4g val_loss=%.4g distance=%.4g feasible=%s",
                    epoch,
                    train_metrics.loss,
                    val_metrics.loss,
                    distance,
                    train_metrics.low_loss and val_metrics.low_loss,
                )

            if feasible_now:
                logger.info(
                    "overlap entered at epoch %d: train_loss=%.4g val_loss=%.4g distance=%.4g; "
                    "stopping overlap search and proceeding to flatness measurement",
                    epoch,
                    train_metrics.loss,
                    val_metrics.loss,
                    distance,
                )
                break

    selected_state = best_feasible_state if best_feasible_state is not None else best_violation_state
    _load_state(model, selected_state)
    selected_train = _metrics(model, train_batch, tokenizer, train_threshold)
    selected_val = _metrics(model, val_batch, tokenizer, val_threshold)
    selected_distance = _normalized_distance(train_state, selected_state, normalization_floor)
    _write_csv(
        csv_path,
        ["epoch", "objective", "train_loss", "val_loss", "normalized_distance", "constraint_violation"],
        rows,
    )
    return selected_state, selected_train, selected_val, selected_distance, best_feasible_state is not None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure train/validation low-loss-region misalignment for a Grokking Transformer"
    )
    parser.add_argument("-dpath", "--dataset_path", required=True)
    parser.add_argument("-o", "--output_folder_name", default=None)
    parser.add_argument("--modulus", type=int, default=97)
    parser.add_argument("-m", "--model_type", default="transformer_for_grokking")
    parser.add_argument("--m_nlayer", type=int, default=None)
    parser.add_argument("--m_n_heads", type=int, default=None)
    parser.add_argument("--m_d_model", type=int, default=None)
    parser.add_argument("--m_context_len", type=int, default=None)
    parser.add_argument("--m_pos_encoding", choices=["default", "trainable"], default=None)
    parser.add_argument("--fit_epoch", type=int, default=10000)
    parser.add_argument("--overlap_epoch", type=int, default=3000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--overlap_learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_epoch", type=int, default=10)
    parser.add_argument("--overlap_penalty", type=float, default=100.0)
    parser.add_argument("--low_loss_threshold", type=float, default=1e-2)
    parser.add_argument("--train_loss_threshold", type=float, default=None)
    parser.add_argument("--val_loss_threshold", type=float, default=None)
    parser.add_argument("--fit_accuracy_threshold", type=float, default=0.99)
    parser.add_argument("--local_radii", default="0.01,0.03,0.1,0.3")
    parser.add_argument("--local_samples", type=int, default=32)
    parser.add_argument("--normalization_floor", type=float, default=1e-8)
    parser.add_argument(
        "--relative_flatness_layers",
        default="linear.weight",
        help="Comma-separated rank-2 parameter names, or all_matrix_weights",
    )
    parser.add_argument("--relative_flatness_samples", type=int, default=8)
    parser.add_argument(
        "--relative_flatness_examples",
        type=int,
        default=512,
        help="Examples per partition for Hessian estimates; <=0 uses the full partition",
    )
    parser.add_argument("--relative_flatness_seed", type=int, default=1729)
    parser.add_argument("--skip_relative_flatness", action="store_true")
    parser.add_argument("--report_interval", type=int, default=100)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num_threads", type=int, default=None)
    parser.add_argument("--skip_overlap_search", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_type != "transformer_for_grokking":
        raise ValueError("Only transformer_for_grokking is supported")
    if not (0.0 < args.fit_accuracy_threshold <= 1.0):
        raise ValueError("--fit_accuracy_threshold must be in (0, 1]")
    if args.low_loss_threshold <= 0:
        raise ValueError("--low_loss_threshold must be positive")
    if args.normalization_floor <= 0:
        raise ValueError("--normalization_floor must be positive")
    if args.learning_rate <= 0 or args.overlap_learning_rate <= 0 or args.min_lr <= 0:
        raise ValueError("learning rates must be positive")
    if args.min_lr > args.learning_rate:
        raise ValueError("--min_lr must not exceed --learning_rate")

    if args.num_threads is not None:
        torch.set_num_threads(max(1, args.num_threads))
    setup_logging(logger, "main")
    if args.random_seed is not None:
        set_seed(args.random_seed)
        logger.info("random seed = %d", args.random_seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    logger.info("device = %s", device)

    train_threshold = args.low_loss_threshold if args.train_loss_threshold is None else args.train_loss_threshold
    val_threshold = args.low_loss_threshold if args.val_loss_threshold is None else args.val_loss_threshold
    radii = _parse_radii(args.local_radii)
    relative_flatness_layers = _parse_layer_names(args.relative_flatness_layers)
    if args.relative_flatness_samples <= 0:
        raise ValueError("--relative_flatness_samples must be positive")
    if args.relative_flatness_examples < -1:
        raise ValueError("--relative_flatness_examples must be positive or -1/0 for the full partition")
    if train_threshold <= 0 or val_threshold <= 0:
        raise ValueError("loss thresholds must be positive")

    if args.output_folder_name is None:
        output_folder = os.path.join(
            os.curdir,
            f"misalignment_measurement_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_%f')}",
        )
    else:
        output_folder = os.path.join(os.curdir, args.output_folder_name)
    os.makedirs(output_folder, exist_ok=False)
    with open(os.path.join(output_folder, "command.txt"), "w", encoding="utf-8") as outfile:
        outfile.write(" ".join([sys.executable, *sys.argv]))

    train_dataset, val_dataset = loading_dataset_from(args.dataset_path, modulus=args.modulus)
    train_batch = _batch_for_dataset(train_dataset, device)
    val_batch = _batch_for_dataset(val_dataset, device)
    model = build_grokking_model(
        train_dataset,
        n_layers=args.m_nlayer,
        n_heads=args.m_n_heads,
        d_model=args.m_d_model,
        context_len=args.m_context_len,
        position_encoding=args.m_pos_encoding,
    ).to(device)
    initial_state = _clone_state(model)

    train_state, fit_success_epoch, _ = train_train_only(
        model,
        train_batch,
        val_batch,
        train_dataset.tokenizer,
        epochs=args.fit_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epoch,
        train_threshold=train_threshold,
        accuracy_threshold=args.fit_accuracy_threshold,
        report_interval=args.report_interval,
        log_path=os.path.join(output_folder, "train_only.log.csv"),
    )
    _load_state(model, train_state)
    train_only_train_metrics = _metrics(model, train_batch, train_dataset.tokenizer, train_threshold)
    train_only_val_metrics = _metrics(model, val_batch, train_dataset.tokenizer, val_threshold)
    save_model_state(
        os.path.join(output_folder, "train_only.model.pt"),
        model.state_dict(),
        args.model_type,
        train_dataset.name,
    )
    save_model_state(
        os.path.join(output_folder, "initial.model.pt"),
        initial_state,
        args.model_type,
        train_dataset.name,
    )

    local_results = probe_local_geometry(
        model,
        train_batch,
        val_batch,
        train_dataset.tokenizer,
        train_state,
        radii=radii,
        samples=args.local_samples,
        train_threshold=train_threshold,
        val_threshold=val_threshold,
        normalization_floor=args.normalization_floor,
        csv_path=os.path.join(output_folder, "local_probe.csv"),
    )

    relative_flatness_rows: list[dict[str, object]] = []
    relative_flatness_payload: dict[str, object] | None = None
    if not args.skip_relative_flatness:
        relative_flatness_layers = _resolve_relative_flatness_layers(model, relative_flatness_layers)
        train_only_relative_flatness = measure_checkpoint_relative_flatness(
            model,
            train_state,
            train_batch,
            val_batch,
            train_dataset.tokenizer,
            layer_names=relative_flatness_layers,
            hutchinson_samples=args.relative_flatness_samples,
            max_examples=args.relative_flatness_examples,
            seed=args.relative_flatness_seed,
        )
        relative_flatness_payload = {
            "configuration": {
                "layers": relative_flatness_layers,
                "hutchinson_samples": args.relative_flatness_samples,
                "examples_per_partition": args.relative_flatness_examples,
                "estimator": "Hutchinson estimate of Tr((W W^T ⊗ I) H)",
            },
            "train_only": {
                dataset_name: [asdict(result) for result in results]
                for dataset_name, results in train_only_relative_flatness.items()
            },
        }
        for dataset_name, results in train_only_relative_flatness.items():
            relative_flatness_rows.extend(
                {"checkpoint": "train_only", "dataset": dataset_name, **asdict(result)}
                for result in results
            )

    overlap_payload = None
    if not args.skip_overlap_search:
        overlap_state, overlap_train, overlap_val, overlap_distance, overlap_found = search_nearest_overlap(
            model,
            train_batch,
            val_batch,
            train_dataset.tokenizer,
            train_state,
            epochs=args.overlap_epoch,
            learning_rate=args.overlap_learning_rate,
            train_threshold=train_threshold,
            val_threshold=val_threshold,
            penalty=args.overlap_penalty,
            normalization_floor=args.normalization_floor,
            report_interval=args.report_interval,
            csv_path=os.path.join(output_folder, "overlap_search.log.csv"),
        )
        save_model_state(
            os.path.join(output_folder, "overlap_candidate.model.pt"),
            overlap_state,
            args.model_type,
            train_dataset.name,
        )
        overlap_payload = {
            "found_feasible_overlap": overlap_found,
            "normalized_distance_from_train_only": overlap_distance,
            "train": asdict(overlap_train),
            "val": asdict(overlap_val),
        }
        if not args.skip_relative_flatness:
            overlap_relative_flatness = measure_checkpoint_relative_flatness(
                model,
                overlap_state,
                train_batch,
                val_batch,
                train_dataset.tokenizer,
                layer_names=relative_flatness_layers,
                hutchinson_samples=args.relative_flatness_samples,
                max_examples=args.relative_flatness_examples,
                seed=args.relative_flatness_seed,
            )
            assert relative_flatness_payload is not None
            relative_flatness_payload["overlap_candidate"] = {
                dataset_name: [asdict(result) for result in results]
                for dataset_name, results in overlap_relative_flatness.items()
            }
            for dataset_name, results in overlap_relative_flatness.items():
                relative_flatness_rows.extend(
                    {"checkpoint": "overlap_candidate", "dataset": dataset_name, **asdict(result)}
                    for result in results
                )

    if relative_flatness_rows:
        relative_flatness_header = list(relative_flatness_rows[0].keys())
        _write_csv(
            os.path.join(output_folder, "relative_flatness.csv"),
            relative_flatness_header,
            ([row[key] for key in relative_flatness_header] for row in relative_flatness_rows),
        )

    summary = {
        "dataset_path": os.path.abspath(args.dataset_path),
        "dataset_name": train_dataset.name,
        "modulus": train_dataset.modulus,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "device": str(device),
        "model_type": args.model_type,
        "low_loss_thresholds": {"train": train_threshold, "val": val_threshold},
        "fit_accuracy_threshold": args.fit_accuracy_threshold,
        "fit_success_epoch": fit_success_epoch,
        "train_only": {
            "train": asdict(train_only_train_metrics),
            "val": asdict(train_only_val_metrics),
        },
        "local_probe": [asdict(result) for result in local_results],
        "relative_flatness": relative_flatness_payload,
        "overlap_search": overlap_payload,
        "interpretation": {
            "higher_normalized_distance_means_more_misalignment": True,
            "higher_conditional_train_only_fraction_means_more_misalignment": True,
            "finite_radius_sharpness_is_auxiliary_not_a_direct_alignment_measure": True,
        },
    }
    with open(os.path.join(output_folder, "summary.json"), "w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, indent=2, allow_nan=True)
    logger.info("measurement complete: %s", os.path.abspath(output_folder))


if __name__ == "__main__":
    main()
