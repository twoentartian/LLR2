"""Estimate the local RLCT (local learning coefficient, LLC) of a trained model.

The estimator follows Lau et al., "Quantifying degeneracy in singular models via
the learning coefficient" (2023), as popularised by the DSLT sequence:

    lambda_hat(w*) = n * beta * (E_{w ~ p_beta(w | w*)}[L_n(w)] - L_n(w*))

where L_n is the empirical (training) loss over n samples, beta = 1 / log(n) and
the expectation is taken over a tempered posterior localised around w* by a
Gaussian prior with strength gamma. The expectation is approximated by
stochastic-gradient Langevin dynamics (SGLD):

    w <- w - eps / 2 * (n * beta * grad L_batch(w) + gamma * (w - w*)) + N(0, eps)

The effective number of parameters is 2 * lambda_hat, which can be compared with
the raw parameter count d (a regular model has lambda = d / 2).
"""

import argparse
import io
import json
import math
import os
import re
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src import ml_setup, model_opti_save_load
from py_src.adapters import CustomStepAdapter, StandardAdapter
from py_src.engine import Device
from py_src.ml_setup.grokking import arithmetic_unknown_exp_grokking
from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_dataset import DatasetSetup, DatasetType
from py_src.ml_setup_dataset.dataset_modular import TRAIN_SPLIT_TYPES, ArithmeticDataset
from py_src.ml_setup_model import ModelType

LossFn = Callable[[torch.nn.Module, object], torch.Tensor]


# ---------------------------------------------------------------------------
# Model weights loading
# ---------------------------------------------------------------------------

def get_model_weights_from_file(path: Path, tick: Optional[int] = None) -> tuple[dict, Optional[str], Optional[str], str]:
    if path.is_file():
        model_weights, model_name, dataset_name = model_opti_save_load.load_model_state_file(str(path), map_location="cpu")
        return model_weights, model_name, dataset_name, str(path)
    if not path.is_dir():
        raise SystemExit(f"Path does not exist: {path}")
    if tick is None:
        raise SystemExit("tick has to be provided in LMDB mode")

    import lmdb

    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
    all_ticks = set()
    with env.begin() as txn:
        for key, value in txn.cursor():
            match = re.search(r"/(\d+)\.model\.pt$", key.decode())
            if match is None:
                continue
            current_tick = int(match.group(1))
            all_ticks.add(current_tick)
            if current_tick == tick:
                payload = torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)
                return (
                    payload[model_opti_save_load.stat_dict_key],
                    payload.get(model_opti_save_load.model_name_key),
                    payload.get(model_opti_save_load.dataset_name_key),
                    f"{path}_tick{tick}",
                )
    raise SystemExit(f"tick {tick} is not in the lmdb, all ticks: {sorted(all_ticks)}")


# ---------------------------------------------------------------------------
# ML setup / dataset resolution
# ---------------------------------------------------------------------------

def _normalize_expression(expression: str, modulus: int) -> str:
    expr = expression.replace(" ", "")
    if "_mod_" not in expr and ("x" in expr or "y" in expr):
        return f"{expr}_mod_{modulus}"
    return expr


def _infer_modulus(*names: Optional[str]) -> Optional[int]:
    for name in names:
        if name is None:
            continue
        match = re.search(r"modulus(\d+)", os.path.basename(os.path.normpath(name)))
        if match is not None:
            return int(match.group(1))
    return None


def _load_grokking_dataset_from_folder(dataset_path: str, modulus: int) -> DatasetSetup:
    dataset_path = os.path.abspath(dataset_path)
    tokenizer_path = os.path.join(dataset_path, "tokenizer.txt")
    train_dataset = ArithmeticDataset.load_from_file(
        os.path.join(dataset_path, "train.txt"),
        modulus,
        name=dataset_path,
        train=True,
        tokenizer_path=tokenizer_path,
    )
    val_dataset = ArithmeticDataset.load_from_file(
        os.path.join(dataset_path, "val.txt"),
        modulus,
        name=dataset_path,
        train=False,
        tokenizer_path=tokenizer_path,
    )
    return DatasetSetup(DatasetType.arithmetic_exp_unknown, train_dataset, val_dataset)


def _limit_operands(dataset: ArithmeticDataset, limit: int) -> ArithmeticDataset:
    """Keep only equations whose two operands x and y are both < ``limit``."""
    data = dataset.data
    eq_index = dataset.tokenizer.stoi["="]
    eq_positions = (data[0] == eq_index).nonzero(as_tuple=False)
    assert eq_positions.numel() == 1, "expected exactly one '=' token per equation"
    # Serialized equations have the form EOS, x, operator, y, =, rhs, EOS.
    y_position = int(eq_positions.item()) - 1
    token_values = torch.tensor(
        [int(token) if token.isdigit() else -1 for token in dataset.tokenizer.itos],
        dtype=torch.long,
    )
    x_values = token_values[data[:, 1]]
    y_values = token_values[data[:, y_position]]
    assert (x_values >= 0).all() and (y_values >= 0).all(), "operands must be integer tokens"
    mask = (x_values < limit) & (y_values < limit)
    return ArithmeticDataset(dataset.name, data[mask], dataset.modulus, dataset.train, tokenizer=dataset.tokenizer)


def _build_grokking_ml_setup(args, dataset_name: Optional[str]) -> MLSetup:
    modulus = args.modulus
    if modulus is None:
        modulus = _infer_modulus(args.grokking_dataset_path, dataset_name)
    if modulus is None:
        raise SystemExit("Could not infer modulus; pass --modulus.")

    if args.grokking_dataset_path is not None:
        print(f"[info] loading grokking dataset from {args.grokking_dataset_path} (modulus {modulus}).")
        dataset_setup = _load_grokking_dataset_from_folder(args.grokking_dataset_path, modulus)
    else:
        expression = _normalize_expression(args.grokking_dataset_exp, modulus)
        print(
            f"[info] generating grokking dataset '{expression}' "
            f"(train {args.train_pct}%, split {args.split_type}, seed {args.dataset_seed})."
        )
        train_dataset, val_dataset = ArithmeticDataset.splits(
            train_pct=args.train_pct,
            operator=expression,
            modulus=modulus,
            train_split_type=args.split_type,
            seed=args.dataset_seed,
        )
        dataset_setup = DatasetSetup(DatasetType.arithmetic_exp_unknown, train_dataset, val_dataset)

    if args.grokking_operand_limit is not None:
        limit = int(args.grokking_operand_limit)
        train_dataset = _limit_operands(dataset_setup.train_data, limit)
        val_dataset = _limit_operands(dataset_setup.valdation_data, limit)
        print(f"[info] operand limit {limit}: kept {len(train_dataset)} train / {len(val_dataset)} val equations.")
        dataset_setup = DatasetSetup(dataset_setup.dataset_type, train_dataset, val_dataset)
    return arithmetic_unknown_exp_grokking(override_dataset=dataset_setup)


def _resolve_ml_setup(args, model_name: str, dataset_name: Optional[str]) -> MLSetup:
    is_grokking = model_name == ModelType.transformer_for_grokking.name
    if is_grokking and (args.grokking_dataset_path is not None or args.grokking_dataset_exp is not None):
        return _build_grokking_ml_setup(args, dataset_name)
    if dataset_name is None or dataset_name not in DatasetType.__members__:
        hint = " Use --grokking_dataset_path or --grokking_dataset_exp." if is_grokking else " Use -d/--dataset."
        raise SystemExit(f"Dataset '{dataset_name}' is not a known dataset type.{hint}")
    return ml_setup.get_ml_setup_from_config(model_name, dataset_type=dataset_name)


# ---------------------------------------------------------------------------
# Differentiable loss functions
# ---------------------------------------------------------------------------

def _move_to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: _move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_to_device(value, device) for value in batch)
    return batch


def build_loss_fn(current_ml_setup: MLSetup) -> LossFn:
    adapter = current_ml_setup.adapter
    if current_ml_setup.model_type == ModelType.transformer_for_grokking:
        assert isinstance(adapter, CustomStepAdapter)
        eq_position = adapter.extra_ctx["eq_position"]

        def grokking_loss(model: torch.nn.Module, batch) -> torch.Tensor:
            logits, _, _ = model(x=batch["text"])
            logits = logits.transpose(-2, -1)[..., eq_position + 1:]
            target = batch["target"][..., eq_position + 1:]
            return F.cross_entropy(logits, target, reduction="mean")

        return grokking_loss

    if isinstance(adapter, StandardAdapter):
        criterion = current_ml_setup.criterion

        def standard_loss(model: torch.nn.Module, batch) -> torch.Tensor:
            data, label = batch
            return criterion(model(data), label)

        return standard_loss

    raise SystemExit(f"RLCT estimation is not supported for adapter {type(adapter).__name__}.")


# ---------------------------------------------------------------------------
# SGLD-based LLC estimation
# ---------------------------------------------------------------------------

def _build_train_dataloader(dataset: Dataset, batch_size: int, collate_fn, *, shuffle: bool, device: Device, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )


def _infinite_batches(dataloader: DataLoader) -> Iterator:
    while True:
        yield from dataloader


@torch.no_grad()
def compute_full_loss(model: torch.nn.Module, loss_fn: LossFn, dataloader: DataLoader, device: torch.device) -> float:
    total_loss = 0.0
    total_count = 0
    for batch in dataloader:
        batch = _move_to_device(batch, device)
        loss = loss_fn(model, batch)
        count = len(batch["target"]) if isinstance(batch, dict) else len(batch[1])
        total_loss += float(loss.item()) * count
        total_count += count
    return total_loss / max(total_count, 1)


def run_sgld_chain(
    model: torch.nn.Module,
    loss_fn: LossFn,
    batches: Iterator,
    center: list[torch.Tensor],
    *,
    epsilon: float,
    gamma: float,
    nbeta: float,
    num_steps: int,
    device: torch.device,
    chain_index: int,
    log_interval: int,
) -> np.ndarray:
    """Run one SGLD chain starting at ``center`` and return the mini-batch loss of every step."""
    params = [p for p in model.parameters() if p.requires_grad]
    with torch.no_grad():
        for param, center_value in zip(params, center):
            param.copy_(center_value)

    noise_std = math.sqrt(epsilon)
    losses = np.full(num_steps, np.nan, dtype=np.float64)
    for step in range(num_steps):
        batch = _move_to_device(next(batches), device)
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch)
        loss.backward()
        loss_value = float(loss.item())
        losses[step] = loss_value
        if not math.isfinite(loss_value):
            print(f"[warning] chain {chain_index}: loss diverged at step {step}, stopping this chain.")
            break

        with torch.no_grad():
            for param, center_value in zip(params, center):
                if param.grad is None:
                    continue
                drift = param.grad.mul(nbeta).add_(param - center_value, alpha=gamma)
                param.add_(drift, alpha=-0.5 * epsilon)
                param.add_(torch.randn_like(param), alpha=noise_std)

        if log_interval > 0 and (step + 1) % log_interval == 0:
            print(f"[info] chain {chain_index}: step {step + 1}/{num_steps}, loss {loss_value:.6f}")
    model.zero_grad(set_to_none=True)
    return losses


def estimate_llc(
    model: torch.nn.Module,
    loss_fn: LossFn,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    n: int,
    *,
    epsilon: float,
    gamma: float,
    nbeta: Optional[float],
    num_chains: int,
    num_draws: int,
    num_burnin: int,
    device: torch.device,
    log_interval: int,
) -> dict:
    model.eval()
    model.to(device)
    if nbeta is None:
        nbeta = n / math.log(n)

    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)
    center = [p.detach().clone() for p in params]
    original_state = OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())

    init_loss = compute_full_loss(model, loss_fn, eval_dataloader, device)
    print(f"[info] n = {n}, parameters = {num_params}, n*beta = {nbeta:.4f}, L_n(w*) = {init_loss:.6f}")

    batches = _infinite_batches(train_dataloader)
    num_steps = num_burnin + num_draws
    traces = []
    chain_llcs = []
    for chain_index in range(num_chains):
        losses = run_sgld_chain(
            model,
            loss_fn,
            batches,
            center,
            epsilon=epsilon,
            gamma=gamma,
            nbeta=nbeta,
            num_steps=num_steps,
            device=device,
            chain_index=chain_index,
            log_interval=log_interval,
        )
        traces.append(losses)
        draws = losses[num_burnin:]
        chain_llc = nbeta * (float(np.mean(draws)) - init_loss)
        chain_llcs.append(chain_llc)
        print(f"[info] chain {chain_index}: LLC = {chain_llc:.4f}")

    model.load_state_dict(original_state, strict=True)

    chain_llcs_array = np.asarray(chain_llcs, dtype=np.float64)
    finite = np.isfinite(chain_llcs_array)
    llc_mean = float(np.mean(chain_llcs_array[finite])) if finite.any() else float("nan")
    llc_std = float(np.std(chain_llcs_array[finite])) if finite.any() else float("nan")
    return {
        "llc_mean": llc_mean,
        "llc_std": llc_std,
        "llc_per_chain": [float(x) for x in chain_llcs],
        "effective_dimension": 2.0 * llc_mean,
        "num_parameters": num_params,
        "init_loss": init_loss,
        "n": n,
        "nbeta": nbeta,
        "epsilon": epsilon,
        "gamma": gamma,
        "num_chains": num_chains,
        "num_draws": num_draws,
        "num_burnin": num_burnin,
        "traces": traces,
    }


def _warn_on_bad_estimate(result: dict) -> None:
    traces = np.stack(result["traces"])
    draws = traces[:, result["num_burnin"]:]
    if not np.isfinite(draws).all():
        print("[warning] some chains diverged; decrease --epsilon or increase --gamma.")
        return
    tail = draws[:, -max(1, draws.shape[1] // 10):]
    if result["llc_mean"] < 0 or tail.mean() < result["init_loss"]:
        print(
            "[warning] SGLD reached a lower loss than L_n(w*), so w* is probably not a local minimum and the "
            "LLC is not well defined there (train the model closer to convergence, or increase --gamma)."
        )
    # A healthy chain plateaus; if the second half is still clearly drifting the estimate is unreliable.
    half = draws.shape[1] // 2
    if half > 0:
        first_half = draws[:, :half].mean()
        second_half = draws[:, half:].mean()
        spread = draws.std() + 1e-12
        if abs(second_half - first_half) > spread:
            print(
                "[warning] loss traces have not plateaued (first half mean "
                f"{first_half:.4f} vs second half {second_half:.4f}); consider more --num_burnin/--num_draws "
                "or tuning --epsilon / --gamma."
            )


def _plot_traces(result: dict, output_path: str) -> None:
    plt.figure(figsize=(10, 5))
    for index, trace in enumerate(result["traces"]):
        plt.plot(trace, linewidth=0.8, label=f"chain {index}")
    plt.axhline(result["init_loss"], color="black", linestyle="--", linewidth=1, label="L_n(w*)")
    if result["num_burnin"] > 0:
        plt.axvline(result["num_burnin"], color="gray", linestyle=":", linewidth=1, label="burn-in end")
    plt.xlabel("SGLD step")
    plt.ylabel("mini-batch loss")
    plt.title(f"SGLD loss traces, LLC = {result['llc_mean']:.2f} +/- {result['llc_std']:.2f}")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the local RLCT (local learning coefficient) of a model with SGLD.",
    )
    parser.add_argument("model_weights_path", type=str, help="a .model.pt file or a lmdb directory.")
    parser.add_argument("-t", "--tick", type=int, help="specify the model weights tick index for a lmdb file.")
    parser.add_argument("-m", "--model", type=str, default=None, help="specify the model name")
    parser.add_argument("-d", "--dataset", type=str, default=None, help="specify the dataset name")
    parser.add_argument("--cpu", action="store_true", help="force using CPU")
    parser.add_argument("-c", "--core", type=int, default=0, help="number of dataloader workers")
    parser.add_argument("-o", "--output", type=str, default=None, help="output file prefix (default: next to the model)")

    grokking_group = parser.add_argument_group("grokking dataset (transformer_for_grokking only)")
    grokking_group.add_argument("--grokking_dataset_path", type=str, default=None, help="folder with train.txt/val.txt/tokenizer.txt")
    grokking_group.add_argument("--grokking_dataset_exp", type=str, default=None, help="generate a dataset from this expression, e.g. 'x+y'")
    grokking_group.add_argument("--modulus", type=int, default=None, help="modulus (default: inferred from 'modulus<N>' in dataset name)")
    grokking_group.add_argument("--grokking_operand_limit", type=int, default=None, help="only keep equations with x < N and y < N")
    grokking_group.add_argument("-tp", "--train_pct", type=float, default=50)
    grokking_group.add_argument("-st", "--split_type", type=str, default="random", choices=TRAIN_SPLIT_TYPES)
    grokking_group.add_argument("--dataset_seed", type=int, default=None, help="seed for the generated train/val split")

    sgld_group = parser.add_argument_group("SGLD")
    sgld_group.add_argument("--epsilon", type=float, default=1e-4, help="SGLD step size")
    sgld_group.add_argument("--gamma", type=float, default=100.0, help="localization strength")
    sgld_group.add_argument("--nbeta", type=float, default=None, help="n * beta (default: n / log(n))")
    sgld_group.add_argument("--num_chains", type=int, default=4)
    sgld_group.add_argument("--num_draws", type=int, default=500)
    sgld_group.add_argument("--num_burnin", type=int, default=100)
    sgld_group.add_argument("-bs", "--batch_size", type=int, default=None, help="SGLD batch size (default: ml setup batch size)")
    sgld_group.add_argument("-s", "--seed", type=int, default=None)
    sgld_group.add_argument("--log_interval", type=int, default=50)

    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = Device.cpu() if args.cpu else Device.auto()
    model_weight_file_path = Path(args.model_weights_path).expanduser().resolve()
    model_weights, model_name_from_model, dataset_name_from_model, output_name = get_model_weights_from_file(
        model_weight_file_path,
        args.tick,
    )

    if model_name_from_model is not None and args.model is not None:
        assert model_name_from_model == args.model, f"model name mismatch {model_name_from_model} != {args.model}"
    model_name = model_name_from_model if model_name_from_model is not None else args.model
    dataset_name = args.dataset if args.dataset is not None else dataset_name_from_model
    if model_name is None:
        raise SystemExit("model must be provided either in the weight file or via -m.")

    current_ml_setup = _resolve_ml_setup(args, model_name, dataset_name)
    target_model: torch.nn.Module = deepcopy(current_ml_setup.model)
    target_model.load_state_dict(model_weights, strict=True)
    loss_fn = build_loss_fn(current_ml_setup)

    train_data = current_ml_setup.training_data
    n = len(train_data)
    batch_size = args.batch_size if args.batch_size is not None else current_ml_setup.default_batch_size
    collate_fn = current_ml_setup.default_collate_fn
    collate_fn_val = current_ml_setup.default_collate_fn_val or collate_fn
    num_workers = max(0, int(args.core))
    train_dataloader = _build_train_dataloader(train_data, batch_size, collate_fn, shuffle=True, device=device, num_workers=num_workers)
    eval_dataloader = _build_train_dataloader(train_data, batch_size, collate_fn_val, shuffle=False, device=device, num_workers=num_workers)

    result = estimate_llc(
        target_model,
        loss_fn,
        train_dataloader,
        eval_dataloader,
        n,
        epsilon=args.epsilon,
        gamma=args.gamma,
        nbeta=args.nbeta,
        num_chains=args.num_chains,
        num_draws=args.num_draws,
        num_burnin=args.num_burnin,
        device=device.device,
        log_interval=args.log_interval,
    )
    _warn_on_bad_estimate(result)

    print(
        f"[info] LLC (local RLCT) = {result['llc_mean']:.4f} +/- {result['llc_std']:.4f} "
        f"(effective dimension 2*LLC = {result['effective_dimension']:.2f}, parameters = {result['num_parameters']})"
    )

    output_prefix = args.output if args.output is not None else output_name
    traces = result.pop("traces")
    pd.DataFrame({f"chain_{i}": trace for i, trace in enumerate(traces)}).to_csv(f"{output_prefix}.rlct_trace.csv", index_label="step")
    summary = {"model_weights_path": str(model_weight_file_path), "model_name": model_name, "dataset_name": dataset_name, **result}
    with open(f"{output_prefix}.rlct.json", "w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, indent=2)
    result["traces"] = traces
    _plot_traces(result, f"{output_prefix}.rlct_trace.pdf")
    print(f"[info] results saved to '{output_prefix}.rlct.json', '.rlct_trace.csv' and '.rlct_trace.pdf'.")


if __name__ == "__main__":
    main()
