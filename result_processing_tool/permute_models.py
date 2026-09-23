#!/usr/bin/env python3
"""Align one or more B checkpoints to reference A using neuron permutations.

Examples (from the repository root):
  python3 result_processing_tool/permute_models.py --reference A.model.pt \
      --models B.model.pt B2.model.pt --output-dir aligned
  python3 result_processing_tool/permute_models.py --reference A.model.pt \
      --models MODELS_DIRECTORY --output-dir aligned

Built-in architecture specifications: bnn, bnn_floating, lenet4, lenet5,
lenet5_large_fc, cct_7_3x1_32, binary_attention_cct_7_3x1_32.
They reuse the model classes used by ml_setup, without constructing datasets.
Other architectures require --spec JSON; see README_model_comparison.md.

The default ``auto`` method uses Git Re-Basin-style coordinate-ascent weight
matching for CNN/MLP architectures. For Binary CCT it adds a CCT-specific
signed matcher: Q/K feature signs are coupled, V signs are compensated in the
output projection, and Q/K/V feature permutations are no longer forced to be
the same. Use ``--method`` to select ``git_rebasin`` or ``signed`` explicitly.
These are checkpoint transformations only; no retraining is performed.

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
from dataclasses import dataclass

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
BUILTINS = ("bnn", "bnn_floating", "lenet4", "lenet5", "lenet5_large_fc",
            "cct_7_3x1_32", "binary_attention_cct_7_3x1_32")


class ReshapedSpec(dict):
    """Axis specification over logical tensor views, preserving checkpoint shapes.

    Used internally for packed attention weights. Ordinary/custom specs retain
    their original meaning: axes refer directly to stored tensor dimensions.
    """

    def __init__(self, axes: Spec, shapes: dict[str, tuple[int, ...]]):
        super().__init__(axes)
        self.shapes = shapes


def tensor_view(value: torch.Tensor, spec: Spec, key: str) -> torch.Tensor:
    if isinstance(spec, ReshapedSpec) and key in spec.shapes:
        return value.reshape(spec.shapes[key])
    return value


@dataclass
class CCTSignedState:
    """Signed attention symmetries for one Binary CCT checkpoint.

    Each permutation maps target indices in C to source indices in B. Q/K use
    one shared feature permutation and sign per head; V uses an independent
    feature permutation and sign, compensated by the projection columns.
    """

    heads: dict[int, torch.Tensor]
    qk_permutations: dict[int, torch.Tensor]
    qk_signs: dict[int, torch.Tensor]
    v_permutations: dict[int, torch.Tensor]
    v_signs: dict[int, torch.Tensor]


def make_model(model_type: str):
    if model_type == "cct_7_3x1_32":
        from py_src.third_party.compact_transformers.src.cct import cct_7_3x1_32
        return cct_7_3x1_32(), (3, 32, 32)
    if model_type == "binary_attention_cct_7_3x1_32":
        from py_src.ml_setup_model.bnn.binary_cct import binary_cct_7_3x1_32
        return binary_cct_7_3x1_32(), (3, 32, 32)
    if model_type in ("bnn", "bnn_floating"):
        from py_src.ml_setup_model.bnn import VGGNet7Binary, VGGNet7Floating
        model_class = VGGNet7Binary if model_type == "bnn" else VGGNet7Floating
        return model_class(), (3, 32, 32)
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
    if model_type in ("cct_7_3x1_32", "binary_attention_cct_7_3x1_32"):
        return cct_spec(model, state)
    spec = {key: [None] * value.ndim for key, value in state.items()}
    if model_type in ("bnn", "bnn_floating"):
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
        if model_type in ("bnn", "bnn_floating"):
            bn = f"bn{index + 1}"
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                key = f"{bn}.{suffix}"
                if key in spec:
                    spec[key][0] = (group, 1) if group else None
        previous, previous_size = group, weight.shape[0]
    return spec


def cct_spec(model: torch.nn.Module, state: dict) -> ReshapedSpec:
    """CCT symmetries: residual embedding, block MLP, heads and head features.

    Head-feature permutations are shared across Q/K/V and all heads in a block.
    This is a valid subset of attention symmetries, not unrestricted independent
    Q/K/V row matching (which would change the attention function).
    Token positions, input RGB channels and output class order remain fixed.
    """
    spec = {key: [None] * value.ndim for key, value in state.items()}
    shapes = {}
    embedding = ("embedding", 1)
    tokenizer_weight = ("tokenizer.conv.weight" if "tokenizer.conv.weight" in state
                        else "tokenizer.conv_layers.0.0.weight")
    spec[tokenizer_weight][0] = embedding
    spec["classifier.positional_emb"][2] = embedding
    for index, block in enumerate(model.classifier.blocks):
        prefix = f"classifier.blocks.{index}"
        heads = (f"{prefix}.heads", 1)
        features = (f"{prefix}.head_features", 1)
        hidden = (f"{prefix}.mlp", 1)
        for norm in ("pre_norm", "norm1"):
            for suffix in ("weight", "bias"):
                spec[f"{prefix}.{norm}.{suffix}"] = [embedding]
        spec[f"{prefix}.linear1.weight"] = [hidden, embedding]
        spec[f"{prefix}.linear1.bias"] = [hidden]
        spec[f"{prefix}.linear2.weight"] = [embedding, hidden]
        spec[f"{prefix}.linear2.bias"] = [embedding]
        attention = f"{prefix}.self_attn"
        dim = block.self_attn.qkv.in_features
        n_heads = block.self_attn.num_heads
        head_dim = dim // n_heads
        # qkv rows store Q then K then V, each with contiguous heads/features.
        shapes[f"{attention}.qkv.weight"] = (3, n_heads, head_dim, dim)
        spec[f"{attention}.qkv.weight"] = [None, heads, features, embedding]
        shapes[f"{attention}.proj.weight"] = (dim, n_heads, head_dim)
        spec[f"{attention}.proj.weight"] = [embedding, heads, features]
        spec[f"{attention}.proj.bias"] = [embedding]
        attention_bias = f"{attention}.attention_bias"
        if attention_bias in state:
            spec[attention_bias] = [heads, None, None]
    for suffix in ("weight", "bias"):
        spec[f"classifier.norm.{suffix}"] = [embedding]
    spec["classifier.attention_pool.weight"] = [None, embedding]
    spec["classifier.fc.weight"] = [None, embedding]
    return ReshapedSpec(spec, shapes)


def binary_cct_spec(model: torch.nn.Module, state: dict) -> ReshapedSpec:
    """Backward-compatible name for the CCT specification helper."""
    return cct_spec(model, state)


def _new_cct_signed_state(model: torch.nn.Module) -> CCTSignedState:
    heads = {}
    qk_permutations, qk_signs = {}, {}
    v_permutations, v_signs = {}, {}
    for index, block in enumerate(model.classifier.blocks):
        head_count = block.self_attn.num_heads
        head_dim = block.self_attn.qkv.out_features // (3 * head_count)
        heads[index] = torch.arange(head_count)
        qk_permutations[index] = torch.arange(head_dim).repeat(head_count, 1)
        qk_signs[index] = torch.ones(head_count, head_dim)
        v_permutations[index] = torch.arange(head_dim).repeat(head_count, 1)
        v_signs[index] = torch.ones(head_count, head_dim)
    return CCTSignedState(heads, qk_permutations, qk_signs, v_permutations, v_signs)


def _cct_signed_state_dict(state: dict, signed: CCTSignedState) -> dict:
    """Apply CCT attention head/feature signed permutations to a checkpoint."""
    output = {key: value.clone() for key, value in state.items()}
    for index, heads in signed.heads.items():
        prefix = f"classifier.blocks.{index}.self_attn"
        qkv_key = f"{prefix}.qkv.weight"
        proj_key = f"{prefix}.proj.weight"
        bias_key = f"{prefix}.attention_bias"
        qkv = state[qkv_key].reshape(3, heads.numel(),
                                     signed.qk_permutations[index].shape[1], -1)
        proj = state[proj_key].reshape(state[proj_key].shape[0], heads.numel(), -1)
        qkv_out = torch.empty_like(qkv)
        proj_out = torch.empty_like(proj)
        bias_out = torch.empty_like(state[bias_key])
        for target_head, source_head in enumerate(heads.tolist()):
            qk_index = signed.qk_permutations[index][target_head]
            qk_sign = signed.qk_signs[index][target_head].to(qkv.device, qkv.dtype)
            v_index = signed.v_permutations[index][target_head]
            v_sign = signed.v_signs[index][target_head].to(qkv.device, qkv.dtype)
            qkv_out[0, target_head] = qkv[0, source_head].index_select(0, qk_index) * qk_sign[:, None]
            qkv_out[1, target_head] = qkv[1, source_head].index_select(0, qk_index) * qk_sign[:, None]
            qkv_out[2, target_head] = qkv[2, source_head].index_select(0, v_index) * v_sign[:, None]
            proj_out[:, target_head] = proj[:, source_head].index_select(1, v_index) * v_sign[None, :]
            bias_out[target_head] = state[bias_key][source_head]
        output[qkv_key] = qkv_out.reshape_as(state[qkv_key])
        output[proj_key] = proj_out.reshape_as(state[proj_key])
        output[bias_key] = bias_out
    return output


def _signed_assignment(score: torch.Tensor, current_permutation: torch.Tensor,
                       current_signs: torch.Tensor, tolerance: float) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Maximize absolute pair scores and choose the corresponding signs."""
    if not torch.isfinite(score).all():
        raise ValueError("Signed matching scores contain NaN or infinity")
    _, assignment_np = linear_sum_assignment(score.abs().numpy(), maximize=True)
    assignment = torch.from_numpy(assignment_np).long()
    selected = score[torch.arange(score.shape[0]), assignment]
    signs = torch.where(selected >= 0, torch.ones_like(selected), -torch.ones_like(selected))
    old = (score[torch.arange(score.shape[0]), current_permutation] * current_signs).sum().item()
    new = selected.abs().sum().item()
    improved = new > old + tolerance * max(1.0, abs(old))
    return assignment, signs, improved


def _signed_cct_refine(a: dict, b: dict, model: torch.nn.Module, *, max_iter: int,
                       seed: int, tolerance: float) -> tuple[dict, dict]:
    """Refine a CCT checkpoint with exact signed Q/K and V/projection symmetries."""
    signed = _new_cct_signed_state(model)
    rng = np.random.default_rng(seed + 1)
    converged = False
    block_indices = list(signed.heads)
    for iteration in range(max_iter):
        improvements = 0
        for index in rng.permutation(block_indices):
            prefix = f"classifier.blocks.{index}.self_attn"
            qkv = b[f"{prefix}.qkv.weight"].reshape(3, signed.heads[index].numel(),
                                                     signed.qk_permutations[index].shape[1], -1).double()
            proj = b[f"{prefix}.proj.weight"].reshape(b[f"{prefix}.proj.weight"].shape[0],
                                                       signed.heads[index].numel(), -1).double()
            bias = b[f"{prefix}.attention_bias"].double()
            target_qkv = a[f"{prefix}.qkv.weight"].reshape_as(qkv).double()
            target_proj = a[f"{prefix}.proj.weight"].reshape_as(proj).double()
            target_bias = a[f"{prefix}.attention_bias"].double()
            head_count = signed.heads[index].numel()
            head_dim = signed.qk_permutations[index].shape[1]

            # Match complete heads while respecting the current within-head maps.
            score = torch.zeros(head_count, head_count, dtype=torch.float64)
            for target_head in range(head_count):
                qk_index = signed.qk_permutations[index][target_head]
                qk_sign = signed.qk_signs[index][target_head].double()
                v_index = signed.v_permutations[index][target_head]
                v_sign = signed.v_signs[index][target_head].double()
                for source_head in range(head_count):
                    q = qkv[0, source_head].index_select(0, qk_index) * qk_sign[:, None]
                    k = qkv[1, source_head].index_select(0, qk_index) * qk_sign[:, None]
                    v = qkv[2, source_head].index_select(0, v_index) * v_sign[:, None]
                    p = proj[:, source_head].index_select(1, v_index) * v_sign[None, :]
                    score[target_head, source_head] = (
                        (target_qkv[0, target_head] * q).sum()
                        + (target_qkv[1, target_head] * k).sum()
                        + (target_qkv[2, target_head] * v).sum()
                        + (target_proj[:, target_head] * p).sum()
                        + (target_bias[target_head] * bias[source_head]).sum()
                    )
            _, head_assignment = linear_sum_assignment(score.numpy(), maximize=True)
            head_assignment = torch.from_numpy(head_assignment).long()
            old_head = score[torch.arange(head_count), signed.heads[index]].sum().item()
            new_head = score[torch.arange(head_count), head_assignment].sum().item()
            if new_head > old_head + tolerance * max(1.0, abs(old_head)):
                signed.heads[index] = head_assignment
                improvements += 1

            # With heads fixed, independently match each head's Q/K and V
            # features. Signs are selected at the same time as assignments.
            for target_head in range(head_count):
                source_head = int(signed.heads[index][target_head])
                qk_score = torch.zeros(head_dim, head_dim, dtype=torch.float64)
                v_score = torch.zeros(head_dim, head_dim, dtype=torch.float64)
                for target_feature in range(head_dim):
                    for source_feature in range(head_dim):
                        qk_score[target_feature, source_feature] = (
                            (target_qkv[0, target_head, target_feature] * qkv[0, source_head, source_feature]).sum()
                            + (target_qkv[1, target_head, target_feature] * qkv[1, source_head, source_feature]).sum()
                        )
                        v_score[target_feature, source_feature] = (
                            (target_qkv[2, target_head, target_feature] * qkv[2, source_head, source_feature]).sum()
                            + (target_proj[:, target_head, target_feature] * proj[:, source_head, source_feature]).sum()
                        )
                qk_assignment, qk_signs, qk_improved = _signed_assignment(
                    qk_score, signed.qk_permutations[index][target_head],
                    signed.qk_signs[index][target_head], tolerance,
                )
                v_assignment, v_signs, v_improved = _signed_assignment(
                    v_score, signed.v_permutations[index][target_head],
                    signed.v_signs[index][target_head], tolerance,
                )
                if qk_improved:
                    signed.qk_permutations[index][target_head] = qk_assignment
                    signed.qk_signs[index][target_head] = qk_signs
                    improvements += 1
                if v_improved:
                    signed.v_permutations[index][target_head] = v_assignment
                    signed.v_signs[index][target_head] = v_signs
                    improvements += 1
        print(f"Signed CCT sweep {iteration + 1}/{max_iter}: {improvements} groups improved", flush=True)
        if improvements == 0:
            converged = True
            break
    result = _cct_signed_state_dict(b, signed)
    keys = [key for layer in select_layers(a).values() for key in layer]
    report = {
        "iterations": iteration + 1,
        "converged": converged,
        "signed_groups": len(block_indices) * 3,
        "parameter_cosine_before": parameter_cosine(a, b, keys),
        "parameter_cosine_after": parameter_cosine(a, result, keys),
        "permutations": {str(index): signed.heads[index].tolist() for index in block_indices},
    }
    return result, report


def match_signed_cct(a: dict, b: dict, model: torch.nn.Module, spec: Spec, *,
                     max_iter: int, seed: int, tolerance: float) -> tuple[dict, dict]:
    """Run baseline Re-Basin matching, then exact signed CCT refinement."""
    base, base_report = match_weights(a, b, spec, max_iter=max_iter,
                                      seed=seed, tolerance=tolerance)
    refined, signed_report = _signed_cct_refine(a, base, model, max_iter=max_iter,
                                                seed=seed, tolerance=tolerance)
    signed_report["base_parameter_cosine_before"] = parameter_cosine(
        a, b, [key for layer in select_layers(a).values() for key in layer]
    )
    signed_report["base_parameter_cosine_after"] = base_report["parameter_cosine_after"]
    # The public before/after fields always describe the original B and final C.
    signed_report["parameter_cosine_before"] = base_report["parameter_cosine_before"]
    signed_report["base_iterations"] = base_report["iterations"]
    signed_report["method"] = "signed_cct"
    return refined, signed_report


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
        value = tensor_view(state[key], spec, key)
        if len(axes) != value.ndim:
            raise ValueError(f"{key}: expected {value.ndim} axis entries")
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
            if value.shape[axis] == 0 or value.shape[axis] % block:
                raise ValueError(f"{key}: axis {axis} is not divisible by block_size={block}")
            n = value.shape[axis] // block
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
    return {key: permuted_tensor(tensor_view(value, spec, key), spec[key], permutations)
            .reshape(value.shape).clone() for key, value in state.items()}


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
                left = tensor_view(a[key], spec, key).movedim(axis, 0).reshape(n, -1).double()
                right = permuted_tensor(tensor_view(b[key], spec, key), spec[key],
                                        permutations, except_axis=axis)
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
        b, b_metadata = load_checkpoint(path)
        if b_metadata != metadata:
            raise ValueError(f"{path}: model/dataset mismatch: {b_metadata} != {metadata}")
        method = args.method
        if method == "auto":
            method = "signed" if model_type == "binary_attention_cct_7_3x1_32" else "git_rebasin"
        display_method = "signed_matching" if method == "signed" else "git_rebasin"
        print(f"Alignment mode: {display_method}", flush=True)
        print(f"Aligning {path} to {reference_path}", flush=True)
        if method == "signed":
            if model_type != "binary_attention_cct_7_3x1_32" or args.spec:
                raise ValueError("--method signed currently requires the built-in Binary CCT7 architecture")
            model, _ = make_model(model_type)
            c, report = match_signed_cct(a, b, model, spec, max_iter=args.max_iter,
                                         seed=args.seed, tolerance=args.tolerance)
        else:
            c, report = match_weights(a, b, spec, max_iter=args.max_iter,
                                      seed=args.seed, tolerance=args.tolerance)
            report["method"] = "git_rebasin"
        report.update({"reference": str(reference_path), "source": str(path), "output": str(destination),
                       "model_type": b_metadata[0], "dataset_type": b_metadata[1], "seed": args.seed,
                       "spec": str(args.spec.resolve()) if args.spec else f"builtin:{model_type}",
                       "method": method,
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
    parser.add_argument("--method", choices=("auto", "git_rebasin", "signed"), default="auto",
                        help="Alignment method: auto selects signed for Binary CCT and Git Re-Basin otherwise")
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
