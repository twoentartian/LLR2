"""Shared checkpoint loading, parameter selection and bounded-memory caching.

The parameter heuristic matches the existing cosine tool: floating tensors,
excluding common BN/EMA buffers. Custom floating buffers require regex filters.
"""

from contextlib import contextmanager
from pathlib import Path
import re
import tempfile

import numpy as np
import torch

from py_src.model_opti_save_load import load_model_state_file
from result_processing_tool.calculate_cosine_similarity import (
    _is_probably_trainable_param, _layer_name_from_key,
)


def is_binary_weight_model(model_type: str | None) -> bool:
    """Return whether checkpoints use binary convolution/linear weights."""
    # ``bnn_floating`` has binary activations but intentionally keeps floating
    # weights. Binary CCT has binary Q/K attention activations, while its
    # checkpoint weights are also floating point. Only ``bnn`` has binary
    # convolution and linear weights in the forward pass.
    return model_type == "bnn"


def is_binary_weight_key(key: str) -> bool:
    """Return whether a normalized BNN key is a binary layer weight."""
    return bool(re.fullmatch(r"(?:conv|fc)\d+\.weight", key))


def binarize_weight(value: torch.Tensor) -> torch.Tensor:
    """Match the project's BNN binarization, including zero -> -1."""
    return value.add(1).div(2).clamp(0, 1).round().mul(2).sub(1)


def natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.as_posix())]


def load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], tuple]:
    state, model_type, dataset_type = load_model_state_file(str(path), map_location="cpu")
    normalized = {}
    for key, value in state.items():
        name = key
        while name.startswith("_orig_mod.") or name.startswith("module."):
            name = name.split(".", 1)[1]
        if name in normalized:
            raise ValueError(f"{path}: duplicate normalized key {name!r}")
        normalized[name] = value
    return normalized, (model_type, dataset_type)


def select_layers(state: dict, key_regex: str | None = None,
                  exclude_regex: str | None = None,
                  layer_level: int | None = None, exclude_bias: bool = False) -> dict[str, list[str]]:
    include = re.compile(key_regex) if key_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    layers: dict[str, list[str]] = {}
    for key, value in state.items():
        if not torch.is_tensor(value) or not _is_probably_trainable_param(key, value):
            continue
        if value.numel() == 0:
            continue
        if include and not include.search(key):
            continue
        if exclude and exclude.search(key):
            continue
        if exclude_bias and key.rsplit(".", 1)[-1].endswith("bias"):
            continue
        name = _layer_name_from_key(key, layer_level)
        layers.setdefault(name, []).append(key)
    return layers


def validate_states(reference: dict, other: dict, context: str = "checkpoint") -> None:
    if set(reference) != set(other):
        raise ValueError(f"{context}: state keys differ; missing={sorted(set(reference) - set(other))}, "
                         f"extra={sorted(set(other) - set(reference))}")
    for key, value in other.items():
        if not torch.is_tensor(value) or not torch.is_tensor(reference[key]):
            raise ValueError(f"{context}: {key} is not a tensor")
        if value.shape != reference[key].shape or value.dtype != reference[key].dtype:
            raise ValueError(f"{context}: shape/dtype mismatch for {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{context}: {key} contains NaN or infinity")


@contextmanager
def cached_weights(files: list[Path], *, key_regex=None, exclude_regex=None,
                   layer_level=None, exclude_bias=False, cache_dir=None,
                   binary_weights: bool = False):
    """Yield (N x P memmap, layers, layer slices, metadata); clean up on exit.

    Loads one checkpoint at a time; cache consumes 8*N*P bytes of temporary disk.
    Validate all selected keys, shapes, finite values and model/dataset metadata.
    """
    if not files:
        raise ValueError("No checkpoints supplied")
    selection = (key_regex, exclude_regex, layer_level, exclude_bias)
    first, metadata = load_checkpoint(files[0])
    use_binary_weights = binary_weights and is_binary_weight_model(metadata[0])
    layers = select_layers(first, *selection)
    if not layers:
        raise ValueError("No floating parameter tensors matched the selection")
    shapes = {key: first[key].shape for keys in layers.values() for key in keys}
    slices, width = {}, 0
    for name, keys in layers.items():
        size = sum(first[key].numel() for key in keys)
        slices[name] = (width, width + size)
        width += size
    del first
    print(f"{len(files)} models, {len(layers)} layers; temporary cache "
          f"{len(files) * width * 8 / 2**30:.2f} GiB", flush=True)
    with tempfile.TemporaryDirectory(prefix="model_weights_", dir=cache_dir) as temp:
        cache = np.memmap(Path(temp) / "weights.dat", mode="w+", dtype=np.float64,
                          shape=(len(files), width))
        try:
            for row, path in enumerate(files):
                state, current_metadata = load_checkpoint(path)
                if current_metadata != metadata:
                    raise ValueError(f"{path}: model/dataset mismatch: {current_metadata} != {metadata}")
                keys_now = {k for keys in select_layers(state, *selection).values() for k in keys}
                if keys_now != set(shapes):
                    raise ValueError(f"{path}: selected parameter keys differ")
                for name, keys in layers.items():
                    offset = slices[name][0]
                    for key in keys:
                        value = state[key]
                        if value.shape != shapes[key]:
                            raise ValueError(f"{path}: shape mismatch for {key}")
                        transformed = (binarize_weight(value)
                                       if use_binary_weights and is_binary_weight_key(key)
                                       else value)
                        flat = transformed.detach().reshape(-1).to(dtype=torch.float64).numpy()
                        if not np.isfinite(flat).all():
                            raise ValueError(f"{path}: {key} contains NaN or infinity")
                        cache[row, offset:offset + flat.size] = flat
                        offset += flat.size
                del state, value, flat
                print(f"Loaded {row + 1}/{len(files)}: {path.name}", flush=True)
            cache.flush()
            yield cache, layers, slices, metadata
        finally:
            cache._mmap.close()
