#!/usr/bin/env python3
"""Align one or more B checkpoints to reference A using neuron permutations.

Examples (from the repository root):
  python3 result_processing_tool/permute_models.py --reference A.model.pt \
      --models B.model.pt B2.model.pt --output-dir aligned
  python3 result_processing_tool/permute_models.py --reference A.model.pt \
      --models MODELS_DIRECTORY --output-dir aligned

Built-in architecture specifications: bnn, lenet4, lenet5, lenet5_large_fc.
They reuse the model classes used by ml_setup, without constructing datasets.
Other architectures require --spec JSON; see README_model_comparison.md.

Coordinate-ascent weight matching solves a Hungarian assignment for each hidden
channel group, considering both incoming and outgoing weights. The objective
maximizes the global parameter dot product (equivalently cosine / minimizes L2,
since permutations preserve norms). It does not guarantee a global optimum or
improve every individual layer. Original latent BNN weights are used unchanged.
The approach follows Git Re-Basin: https://arxiv.org/abs/2209.04836 .

Input features and output class order stay fixed in built-in specifications;
BN statistics and channel blocks at convolution-to-linear flattening boundaries
are permuted together. Each C retains B's checkpoint format/model/dataset names.
Built-ins check B/C outputs on synthetic inputs before saving. Custom specs are
structurally validated but require the caller to ensure graph-level symmetry.
Dependencies: torch, numpy, scipy; built-ins also use project model dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from py_src.model_opti_save_load import save_model_state
from result_processing_tool.model_weight_utils import (
    load_checkpoint, natural_key, select_layers, validate_states,
)

# One entry per tensor axis: None means fixed, (group_name, block_size) means
# reorder contiguous blocks on this axis with the named permutation.
Axis = tuple[str, int] | None
Spec = dict[str, list[Axis]]
BUILTINS = ("bnn", "lenet4", "lenet5", "lenet5_large_fc")


def make_model(model_type: str):
    if model_type == "bnn":
        from py_src.ml_setup_model.bnn import VGGNet7Binary
        return VGGNet7Binary(), (3, 32, 32)
    from py_src.ml_setup_model.lenet import LeNet4, LeNet5, LeNet5LargeFc
    constructors = {"lenet4": LeNet4, "lenet5": LeNet5, "lenet5_large_fc": LeNet5LargeFc}
    if model_type not in constructors:
        raise ValueError(f"No built-in model for {model_type!r}; provide --spec")
    return constructors[model_type](), (1, 28, 28)


def builtin_spec(model_type: str, state: dict) -> Spec:
    model, _ = make_model(model_type)
    expected = model.state_dict()
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
        raise ValueError(f"Checkpoint does not match the project's {model_type} architecture")
    spec = {key: [None] * value.ndim for key, value in state.items()}
    if model_type == "bnn":
        chain = [f"conv{i}" for i in range(1, 7)] + ["fc1", "fc2", "fc3"]
    else:
        chain = ["conv1", "conv2", "fc1", "fc2"]
        if model_type != "lenet4":
            chain.append("fc3")
    previous = None
    previous_size = None
    for index, name in enumerate(chain):
        weight = state[f"{name}.weight"]
        group = name if index < len(chain) - 1 else None
        spec[f"{name}.weight"][0] = (group, 1) if group else None
        if previous:
            block = weight.shape[1] // previous_size
            spec[f"{name}.weight"][1] = (previous, block)
        if f"{name}.bias" in spec:
            spec[f"{name}.bias"][0] = (group, 1) if group else None
        if model_type == "bnn":
            bn = f"bn{index + 1}"
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                key = f"{bn}.{suffix}"
                if key in spec:
                    spec[key][0] = (group, 1) if group else None
        previous, previous_size = group, weight.shape[0]
    return spec


def read_spec(path: Path) -> Spec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("axes"), dict):
        raise ValueError("Spec must be a JSON object with an 'axes' mapping")
    spec = {}
    for key, axes in raw["axes"].items():
        if not isinstance(axes, list):
            raise ValueError(f"{key}: axes must be a list")
        spec[key] = []
        for entry in axes:
            if entry is None:
                spec[key].append(None)
            elif isinstance(entry, str):
                spec[key].append((entry, 1))
            elif isinstance(entry, dict) and set(entry) <= {"group", "block_size"} and "group" in entry:
                spec[key].append((entry["group"], entry.get("block_size", 1)))
            else:
                raise ValueError(f"{key}: invalid axis entry {entry!r}")
    return spec


def validate_spec(spec: Spec, state: dict) -> dict[str, int]:
    if set(spec) != set(state):
        raise ValueError("Permutation spec must explicitly cover every state_dict key, "
                         "including unchanged tensors (null axes) and scalar buffers ([])" )
    sizes = {}
    for key, axes in spec.items():
        if len(axes) != state[key].ndim:
            raise ValueError(f"{key}: expected {state[key].ndim} axis entries")
        used = set()
        for axis, entry in enumerate(axes):
            if entry is None:
                continue
            group, block = entry
            if not isinstance(group, str) or not group or type(block) is not int or block < 1:
                raise ValueError(f"{key}: group must be nonempty and block_size a positive integer")
            if group in used:
                raise ValueError(f"{key}: same group on multiple axes is not supported by linear assignment")
            used.add(group)
            if state[key].shape[axis] == 0 or state[key].shape[axis] % block:
                raise ValueError(f"{key}: axis {axis} is not divisible by block_size={block}")
            n = state[key].shape[axis] // block
            if group in sizes and sizes[group] != n:
                raise ValueError(f"Inconsistent channel count for permutation group {group}")
            sizes[group] = n
    if not sizes:
        raise ValueError("Spec has no permutable groups")
    return sizes


def permuted_tensor(value: torch.Tensor, axes: list[Axis], permutations: dict,
                    except_axis: int | None = None) -> torch.Tensor:
    for axis, entry in enumerate(axes):
        if entry is None or axis == except_axis:
            continue
        group, block = entry
        indices = permutations[group]
        if block != 1:
            indices = (indices[:, None] * block + torch.arange(block)[None, :]).reshape(-1)
        value = value.index_select(axis, indices)
    return value


def apply_permutations(state: dict, spec: Spec, permutations: dict) -> dict:
    sizes = validate_spec(spec, state)
    if set(permutations) != set(sizes):
        raise ValueError("Permutation groups do not match specification")
    for group, n in sizes.items():
        indices = permutations[group]
        if indices.dtype != torch.long or indices.ndim != 1 or not torch.equal(
                indices.sort().values, torch.arange(n)):
            raise ValueError(f"{group}: expected a bijection of 0..{n - 1}")
    return {key: permuted_tensor(value, spec[key], permutations).clone() for key, value in state.items()}


def parameter_cosine(a: dict, b: dict, keys: list[str]) -> float | None:
    dot = aa = bb = 0.0
    for key in keys:
        left, right = a[key].double().reshape(-1), b[key].double().reshape(-1)
        dot += torch.dot(left, right).item()
        aa += torch.dot(left, left).item()
        bb += torch.dot(right, right).item()
    return dot / np.sqrt(aa) / np.sqrt(bb) if aa > 0 and bb > 0 else None


def match_weights(a: dict, b: dict, spec: Spec, *, max_iter: int = 50,
                  seed: int = 0, tolerance: float = 1e-10) -> tuple[dict, dict]:
    """Coordinate ascent of the unnormalized dot product over all parameters.

    Each Hungarian step considers all incident parameter tensors, excluding BN
    running statistics from the objective but including them when saving C.
    Only strict improvements are accepted; report whether iteration limit hit.
    """
    if max_iter < 1 or tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("max_iter must be positive and tolerance finite/nonnegative")
    validate_states(a, a, "reference")
    validate_states(a, b, "model B")
    sizes = validate_spec(spec, a)
    keys = [key for layer in select_layers(a).values() for key in layer]
    if not keys:
        raise ValueError("No floating parameters to match")
    incident = {group: [] for group in sizes}
    for key in keys:
        for axis, entry in enumerate(spec[key]):
            if entry:
                incident[entry[0]].append((key, axis))
    if any(not terms for terms in incident.values()):
        raise ValueError("Every permutation group must affect at least one floating parameter")
    permutations = {group: torch.arange(n) for group, n in sizes.items()}
    rng = np.random.default_rng(seed)
    groups = list(sizes)
    converged = False
    for iteration in range(max_iter):
        improvements = 0
        for group in rng.permutation(groups):
            n = sizes[group]
            score = torch.zeros((n, n), dtype=torch.float64)
            for key, axis in incident[group]:
                left = a[key].movedim(axis, 0).reshape(n, -1).double()
                right = permuted_tensor(b[key], spec[key], permutations, except_axis=axis)
                right = right.movedim(axis, 0).reshape(n, -1).double()
                score.addmm_(left, right.T)
            if not torch.isfinite(score).all():
                raise ValueError(f"{group}: non-finite matching scores")
            _, assignment = linear_sum_assignment(score.numpy(), maximize=True)
            candidate = torch.from_numpy(assignment).long()
            old = score[torch.arange(n), permutations[group]].sum().item()
            new = score[torch.arange(n), candidate].sum().item()
            if new > old + tolerance * max(1.0, abs(old)):
                permutations[group] = candidate
                improvements += 1
        print(f"Matching sweep {iteration + 1}/{max_iter}: {improvements} groups improved", flush=True)
        if improvements == 0:
            converged = True
            break
    result = apply_permutations(b, spec, permutations)
    report = {"iterations": iteration + 1, "converged": converged,
              "parameter_cosine_before": parameter_cosine(a, b, keys),
              "parameter_cosine_after": parameter_cosine(a, result, keys),
              "permutations": {group: indices.tolist() for group, indices in permutations.items()}}
    return result, report


@torch.no_grad()
def verify_outputs(b: dict, c: dict, model_type: str, seed: int, samples: int = 4) -> dict:
    model, shape = make_model(model_type)
    model.double().eval()
    data = torch.randn((samples, *shape), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    model.load_state_dict(b, strict=True)
    before = model(data)
    model.load_state_dict(c, strict=True)
    after = model(data)
    error = (before - after).abs().max().item()
    if not torch.allclose(before, after, rtol=1e-6, atol=1e-7):
        raise ValueError(f"B/C output verification failed: max absolute error={error}")
    return {"samples": samples, "input_shape": list(shape), "dtype": "float64",
            "max_absolute_error": error, "passed": True}


def run(args) -> list[Path]:
    reference_path = args.reference.resolve()
    a, metadata = load_checkpoint(reference_path)
    validate_states(a, a, str(reference_path))
    model_type = args.model_type or metadata[0]
    if args.model_type and metadata[0] and args.model_type != metadata[0]:
        raise ValueError("--model-type conflicts with reference checkpoint metadata")
    if args.spec:
        spec = read_spec(args.spec)
    elif model_type in BUILTINS:
        spec = builtin_spec(model_type, a)
    else:
        raise ValueError(f"No built-in permutation spec for {model_type!r}; supply --spec JSON")
    validate_spec(spec, a)
    output = args.output_dir.resolve()
    jobs = {}
    for item in args.models:
        item = item.resolve()
        if item.is_dir():
            discovered = sorted((p.resolve() for p in item.glob(args.pattern) if p.is_file()), key=natural_key)
            for path in discovered:
                if path != reference_path:
                    jobs.setdefault(path, path.relative_to(item))
        elif item.is_file():
            jobs.setdefault(item, Path(item.name))
        else:
            raise ValueError(f"B path does not exist: {item}")
    if not jobs:
        raise ValueError("No B checkpoints found (reference is excluded from directory inputs)")
    destinations = []
    for path, relative in jobs.items():
        name = relative.name.removesuffix(".model.pt").removesuffix(".pt") + ".permuted.model.pt"
        destination = output / relative.parent / name
        report_path = destination.with_suffix(".json")
        if destination in destinations:
            raise ValueError(f"Duplicate output filename for inputs: {destination}")
        if destination == reference_path or destination in jobs or destination.exists() or report_path.exists():
            raise ValueError(f"Output would overwrite an existing file: {destination}")
        destinations.append(destination)
    for (path, _), destination in zip(jobs.items(), destinations):
        print(f"Aligning {path} to {reference_path}", flush=True)
        b, b_metadata = load_checkpoint(path)
        if b_metadata != metadata:
            raise ValueError(f"{path}: model/dataset mismatch: {b_metadata} != {metadata}")
        c, report = match_weights(a, b, spec, max_iter=args.max_iter,
                                  seed=args.seed, tolerance=args.tolerance)
        report.update({"reference": str(reference_path), "source": str(path), "output": str(destination),
                       "model_type": b_metadata[0], "dataset_type": b_metadata[1], "seed": args.seed,
                       "spec": str(args.spec.resolve()) if args.spec else f"builtin:{model_type}",
                       "objective": "global parameter dot product; BN running statistics excluded",
                       "verification": None})
        if not args.spec:
            report["verification"] = verify_outputs(b, c, model_type, args.seed)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            save_model_state(str(temporary), c, *b_metadata)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        destination.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Saved {destination}; cosine {report['parameter_cosine_before']} -> "
              f"{report['parameter_cosine_after']}; converged={report['converged']}", flush=True)
    return destinations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-a", "--reference", required=True, type=Path)
    parser.add_argument("-b", "--models", required=True, nargs="+", type=Path, help="B files and/or directories")
    parser.add_argument("-o", "--output-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*.model.pt", help="Glob used for B directories")
    parser.add_argument("--model-type", choices=BUILTINS, help="Only needed if checkpoint metadata is absent")
    parser.add_argument("--spec", type=Path, help="Custom JSON permutation specification")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-10, help="Relative improvement threshold")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.max_iter, args.threads) < 1 or args.tolerance < 0 or not np.isfinite(args.tolerance):
        parser.error("max-iter/threads must be positive; tolerance must be finite and nonnegative")
    torch.set_num_threads(args.threads)
    try:
        run(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
