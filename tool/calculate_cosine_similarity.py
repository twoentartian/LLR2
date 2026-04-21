#!/usr/bin/env python3
"""
Compute cosine similarity between two PyTorch model checkpoints/state_dicts.

Examples:
  python cos_sim_models.py a.pt b.pt
  python cos_sim_models.py a.pt b.pt --per-key
  python cos_sim_models.py a.pt b.pt --per-layer
  python cos_sim_models.py a.pt b.pt --per-layer --layer-level 2
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import torch


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if obj and all(isinstance(k, str) for k in obj.keys()) and all(torch.is_tensor(v) for v in obj.values()):
            return obj  # raw state_dict

        for k in ("state_dict", "model_state_dict", "model", "net", "module", "model_state"):
            if k in obj and isinstance(obj[k], dict) and obj[k] and all(torch.is_tensor(v) for v in obj[k].values()):
                return obj[k]

    raise ValueError(
        "Could not extract a state_dict from the loaded file. "
        "Expected a state_dict or a checkpoint containing 'state_dict' / 'model_state_dict' / 'model'."
    )


def _normalize_key(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def _is_probably_trainable_param(key: str, t: torch.Tensor) -> bool:
    """
    Best-effort heuristic for trainable params from a state_dict (no nn.Module available).
    Count floating-point tensors, excluding common non-trainable buffers (e.g., BN running stats).
    """
    if not torch.is_floating_point(t):
        return False

    # Common buffers / non-trainable state
    buffer_suffixes = (
        "running_mean",
        "running_var",
        "num_batches_tracked",
    )
    if any(key.endswith(suf) for suf in buffer_suffixes):
        return False

    # Some frameworks store EMA/shadow weights or other non-trainable copies; ignore common patterns
    lowered = key.lower()
    if any(tok in lowered for tok in ("ema.", "shadow", "moving_average", "avg_model")):
        return False

    return True


def _collect_common_tensors(
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
    strict: bool,
    key_regex: Optional[str],
    skip_non_float: bool,
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, str]]:
    a_map = {_normalize_key(k): v for k, v in sd_a.items()}
    b_map = {_normalize_key(k): v for k, v in sd_b.items()}

    pat = re.compile(key_regex) if key_regex else None

    notes: Dict[str, str] = {}
    common: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    all_keys = sorted(set(a_map.keys()) | set(b_map.keys()))
    for k in all_keys:
        if pat and not pat.search(k):
            continue

        if k not in a_map:
            notes[k] = "missing in A"
            continue
        if k not in b_map:
            notes[k] = "missing in B"
            continue

        ta = a_map[k]
        tb = b_map[k]

        if ta.shape != tb.shape:
            msg = f"shape mismatch: {tuple(ta.shape)} vs {tuple(tb.shape)}"
            if strict:
                notes[k] = msg
                continue
            notes[k] = msg + " (skipped)"
            continue

        if skip_non_float and (not torch.is_floating_point(ta) or not torch.is_floating_point(tb)):
            notes[k] = f"non-float dtype: {ta.dtype} / {tb.dtype} (skipped)"
            continue

        common[k] = (ta, tb)

    if strict:
        bad = {k: v for k, v in notes.items() if ("missing" in v) or ("shape mismatch" in v)}
        if bad:
            msg = "\n".join([f"  {k}: {v}" for k, v in list(bad.items())[:50]])
            raise ValueError(f"Strict check failed; first issues:\n{msg}")

    return common, notes


def _layer_name_from_key(key: str, layer_level: Optional[int]) -> str:
    """
    Default "layer" grouping: drop the last token (.weight/.bias/.running_mean/etc.).
      e.g. layer1.0.conv1.weight -> layer1.0.conv1

    If layer_level is set, keep only that many prefix tokens:
      e.g. key=layer1.0.conv1.weight, layer_level=2 -> layer1.0
    """
    base = key.rsplit(".", 1)[0] if "." in key else key
    if layer_level is None:
        return base
    parts = base.split(".")
    return ".".join(parts[:layer_level]) if len(parts) > layer_level else base


def _count_trainable_params(common: Dict[str, Tuple[torch.Tensor, torch.Tensor]], group_fn) -> Dict[str, int]:
    """
    Counts (estimated) trainable parameters per group, based on state_dict keys.
    """
    out: Dict[str, int] = {}
    for k, (ta, _tb) in common.items():
        if not _is_probably_trainable_param(k, ta):
            continue
        g = group_fn(k)
        out[g] = out.get(g, 0) + ta.numel()
    return out


@torch.no_grad()
def cosine_similarity_flat(common: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[float, float, float]:
    dot = torch.tensor(0.0, dtype=torch.float64)
    na2 = torch.tensor(0.0, dtype=torch.float64)
    nb2 = torch.tensor(0.0, dtype=torch.float64)

    for _, (ta, tb) in common.items():
        a = ta.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        b = tb.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        dot += (a * b).sum()
        na2 += (a * a).sum()
        nb2 += (b * b).sum()

    norm_a = torch.sqrt(na2).item()
    norm_b = torch.sqrt(nb2).item()
    if norm_a == 0.0 or norm_b == 0.0:
        return float("nan"), norm_a, norm_b

    cos = (dot / (torch.sqrt(na2) * torch.sqrt(nb2))).item()
    return float(cos), norm_a, norm_b


@torch.no_grad()
def cosine_and_norms_per_group(
    common: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    group_fn,
) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      group -> {
        "cos": cosine similarity on concatenated tensors in group,
        "norm_a": ||A_group||,
        "norm_b": ||B_group||,
        "n": number of scalar elements
      }
    """
    dot: Dict[str, torch.Tensor] = {}
    na2: Dict[str, torch.Tensor] = {}
    nb2: Dict[str, torch.Tensor] = {}
    n_el: Dict[str, int] = {}

    for k, (ta, tb) in common.items():
        g = group_fn(k)

        a = ta.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        b = tb.detach().reshape(-1).to(dtype=torch.float64, device="cpu")

        if g not in dot:
            dot[g] = torch.tensor(0.0, dtype=torch.float64)
            na2[g] = torch.tensor(0.0, dtype=torch.float64)
            nb2[g] = torch.tensor(0.0, dtype=torch.float64)
            n_el[g] = 0

        dot[g] += (a * b).sum()
        na2[g] += (a * a).sum()
        nb2[g] += (b * b).sum()
        n_el[g] += a.numel()

    out: Dict[str, Dict[str, float]] = {}
    for g in dot.keys():
        norm_a = torch.sqrt(na2[g]).item()
        norm_b = torch.sqrt(nb2[g]).item()
        denom = (torch.sqrt(na2[g]) * torch.sqrt(nb2[g])).item()
        cos = float("nan") if denom == 0.0 else float((dot[g] / (torch.sqrt(na2[g]) * torch.sqrt(nb2[g]))).item())

        out[g] = {
            "cos": float(cos),
            "norm_a": float(norm_a),
            "norm_b": float(norm_b),
            "n": float(n_el[g]),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Cosine similarity between two PyTorch models/checkpoints.")
    ap.add_argument("model_a", type=Path, help="Path to first model checkpoint/state_dict")
    ap.add_argument("model_b", type=Path, help="Path to second model checkpoint/state_dict")

    ap.add_argument("--strict", action="store_true", help="Require same keys + shapes (after stripping 'module.')")
    ap.add_argument("--key-regex", type=str, default=None, help="Only include keys matching this regex")
    ap.add_argument(
        "--skip-non-float",
        action="store_true",
        help="Skip non-floating tensors (e.g., num_batches_tracked)",
    )

    args = ap.parse_args()

    obj_a = torch.load(args.model_a, map_location="cpu")
    obj_b = torch.load(args.model_b, map_location="cpu")

    sd_a = _extract_state_dict(obj_a)
    sd_b = _extract_state_dict(obj_b)

    common, notes = _collect_common_tensors(
        sd_a,
        sd_b,
        strict=args.strict,
        key_regex=args.key_regex,
        skip_non_float=args.skip_non_float,
    )

    if not common:
        raise SystemExit("No common tensors to compare (after filters).")

    cos, norm_a, norm_b = cosine_similarity_flat(common)

    # param counts (estimated from common tensors)
    layer_group_fn = lambda k: _layer_name_from_key(k, None)
    params_per_layer = _count_trainable_params(common, layer_group_fn)
    total_params = sum(params_per_layer.values())

    print(f"Common tensors used: {len(common)}")
    print(f"Estimated trainable params (in common tensors): {total_params:,}")
    print(f"||A|| = {norm_a:.6g}   ||B|| = {norm_b:.6g}")
    print(f"Cosine similarity (flattened all common tensors): {cos:.12f}")

    # Only show notes in non-strict mode (strict already errors)
    if not args.strict:
        skipped = [k for k, v in notes.items() if "skipped" in v or "shape mismatch" in v or "missing" in v]
        if skipped:
            print(f"\nNotes (showing up to 20):")
            for k in skipped[:20]:
                print(f"  {k}: {notes[k]}")
            if len(skipped) > 20:
                print(f"  ... and {len(skipped) - 20} more")

    per_layer = cosine_and_norms_per_group(common, layer_group_fn)
    items = sorted(per_layer.items(), key=lambda kv: kv[0])

    # --- alignment: dynamic column width for the layer name ---
    name_w = min(80, max(len(name) for name, _ in items))  # cap so it doesn't get silly
    name_w = max(name_w, len("layer"))

    header = (
        f"  {'layer':<{name_w}}  "
        f"{'cos':>14}  {'||A||':>12}  {'||B||':>12}  {'B/A':>10}  {'params':>12}"
    )
    print("\nPer-layer cosine + norms + params:")
    print(header)

    for name, stats in items:
        cos_v = stats["cos"]
        na = stats["norm_a"]
        nb = stats["norm_b"]
        ratio = float("nan") if na == 0.0 else (nb / na)
        p = params_per_layer.get(name, 0)

        print(
            f"  {name:<{name_w}}  "
            f"{cos_v:> .12f}  "
            f"{na:>12.6g}  "
            f"{nb:>12.6g}  "
            f"{ratio:>10.6g}  "
            f"{p:>12,}"
        )



if __name__ == "__main__":
    main()
