from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


class RunningTensorStats:
    def __init__(self, initial_value: torch.Tensor):
        value = initial_value.detach().to(device="cpu", dtype=torch.float64).contiguous()
        self.count = 0
        self.shape = tuple(value.shape)
        self.mean = torch.zeros_like(value)
        self.m2 = torch.zeros_like(value)

    def update(self, value: torch.Tensor) -> None:
        current = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
        if tuple(current.shape) != self.shape:
            raise ValueError(f"shape mismatch: expected {self.shape}, got {tuple(current.shape)}")
        self.count += 1
        delta = current - self.mean
        self.mean += delta / self.count
        delta2 = current - self.mean
        self.m2 += delta * delta2

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count <= 1:
            variance = torch.zeros_like(self.mean)
        else:
            variance = self.m2 / (self.count - 1)
        return self.mean.to(dtype=torch.float32), variance.to(dtype=torch.float32)


def _initialize_running_stats(layers: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, RunningTensorStats]:
    return OrderedDict((name, RunningTensorStats(tensor)) for name, tensor in layers.items())


def _flatten_tensor_tree(prefix: str, value: Any) -> list[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        return [(prefix, value)]
    if isinstance(value, Mapping):
        items: list[tuple[str, torch.Tensor]] = []
        for key, item in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            items.extend(_flatten_tensor_tree(child_prefix, item))
        return items
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = []
        for index, item in enumerate(value):
            child_prefix = f"{prefix}/{index}" if prefix else str(index)
            items.extend(_flatten_tensor_tree(child_prefix, item))
        return items
    raise TypeError(f"unsupported value type {type(value)!r} at {prefix!r}")


def _extract_layer_tensors(payload: Mapping[str, Any]) -> OrderedDict[str, torch.Tensor]:
    layers: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, tensor in _flatten_tensor_tree("input", payload["input"]):
        layers[name] = tensor
    for layer_name, value in payload["activations"].items():
        for name, tensor in _flatten_tensor_tree(str(layer_name), value):
            layers[name] = tensor
    return layers


def _normalize_layer_name(layer_name: str) -> str:
    normalized = layer_name
    if ":" in normalized:
        normalized = normalized.split(":", 1)[1]
    if "#" in normalized:
        normalized = normalized.split("#", 1)[0]
    return normalized


def _is_batch_norm_layer(layer_name: str) -> bool:
    if layer_name == "input":
        return False
    module_name = _normalize_layer_name(layer_name)
    if module_name == "bn1":
        return True
    if module_name.endswith("downsample.1"):
        return True
    return module_name.split(".")[-1].startswith("bn")


def _filter_layers_in_forward_order(layers: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((layer_name, tensor) for layer_name, tensor in layers.items() if not _is_batch_norm_layer(layer_name))


def _load_payload(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "input" not in payload or "activations" not in payload:
        raise ValueError(f"{path} does not look like an activation dump file")
    return payload


def _load_run_config(input_dir: Path) -> Mapping[str, Any] | None:
    run_config_path = input_dir / "run_config.json"
    if not run_config_path.exists():
        return None
    with open(run_config_path, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    return data if isinstance(data, Mapping) else None


def _collect_split_files(input_dir: Path, split: str) -> list[Path]:
    split_dir = input_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"missing split directory: {split_dir}")
    if not split_dir.is_dir():
        raise NotADirectoryError(f"split path is not a directory: {split_dir}")
    return sorted(path for path in split_dir.rglob("*.pt") if path.is_file())


def _collect_selected_files(input_dir: Path, split: str) -> tuple[list[Path], list[str]]:
    selected_splits = ["train", "val"] if split == "all" else [split]
    files: list[Path] = []
    for split_name in selected_splits:
        files.extend(_collect_split_files(input_dir, split_name))
    return files, selected_splits


def _make_layer_summary(layer_name: str, mean_tensor: torch.Tensor, variance_tensor: torch.Tensor, count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    mean_variance = float(variance_tensor.mean().item())
    summary = {
        "layer_name": layer_name,
        "shape": list(mean_tensor.shape),
        "numel": mean_tensor.numel(),
        "count": count,
        "amount_of_variation": mean_variance,
        "mean_variance": mean_variance,
        "total_variance": float(variance_tensor.sum().item()),
        "max_variance": float(variance_tensor.max().item()),
        "mean_abs_mean": float(mean_tensor.abs().mean().item()),
    }
    tensor_record = {"mean": mean_tensor, "variance": variance_tensor}
    return summary, tensor_record


def _finalize_stats_collection(
    stats: OrderedDict[str, RunningTensorStats],
    include_tensors: bool,
) -> tuple[OrderedDict[str, dict[str, Any]], OrderedDict[str, dict[str, torch.Tensor]] | None]:
    layer_summaries: OrderedDict[str, dict[str, Any]] = OrderedDict()
    layer_tensors: OrderedDict[str, dict[str, torch.Tensor]] | None = OrderedDict() if include_tensors else None
    for layer_name, layer_stats in stats.items():
        mean_tensor, variance_tensor = layer_stats.finalize()
        summary, tensor_record = _make_layer_summary(layer_name, mean_tensor, variance_tensor, layer_stats.count)
        layer_summaries[layer_name] = summary
        if layer_tensors is not None:
            layer_tensors[layer_name] = tensor_record
    return layer_summaries, layer_tensors


def _extract_label_from_metadata(payload: Mapping[str, Any], path: Path) -> int:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or "label" not in metadata:
        raise ValueError(f"{path} does not contain metadata.label, so label-aware variation cannot be computed")
    label = metadata["label"]
    if isinstance(label, bool) or not isinstance(label, int):
        raise ValueError(f"{path} has a non-integer metadata.label={label!r}")
    return int(label)


def _extract_label_from_path(path: Path, input_dir: Path) -> int | None:
    relative_path = path.relative_to(input_dir)
    if len(relative_path.parts) < 3:
        return None
    label_folder = relative_path.parts[1]
    try:
        return int(label_folder)
    except ValueError:
        if label_folder.startswith("label_"):
            try:
                return int(label_folder.removeprefix("label_"))
            except ValueError:
                return None
        return None


def _scan_candidate_entries(candidate_file_paths: list[Path], input_dir: Path) -> list[tuple[Path, int]]:
    entries: list[tuple[Path, int]] = []
    for path in candidate_file_paths:
        label = _extract_label_from_path(path, input_dir)
        if label is None:
            label = _extract_label_from_metadata(_load_payload(path), path)
        entries.append((path, label))
    return entries


def _make_label_quota(label_ids: Sequence[int], num_samples: int | None) -> dict[int, int] | None:
    if num_samples is None:
        return None
    if len(label_ids) == 0:
        raise ValueError("cannot build a balanced quota for an empty label set")
    base_quota = num_samples // len(label_ids)
    remainder = num_samples % len(label_ids)
    return {label: base_quota + (1 if index < remainder else 0) for index, label in enumerate(label_ids)}


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate per-layer variation from activation dump .pt files using streaming mean and sample variance.")
    parser.add_argument("--input-dir", required=True, help="directory produced by dump_resnet_activations.py containing train/ and val/")
    parser.add_argument("--output-dir", required=True, help="directory that will receive variation_<split>.pt and variation_<split>.json")
    parser.add_argument("--split", choices=("all", "train", "val"), default="all", help="which dumped split(s) to include in the variation calculation")
    parser.add_argument("--num-samples", "--max-samples", dest="num_samples", type=int, default=None, help="use up to N samples total, balanced as evenly as possible across the selected labels")
    parser.add_argument("--labels", nargs="+", type=int, default=None, help="optional label filter; only samples with these labels will be included")
    parser.add_argument("--log-every", type=int, default=100, help="progress print interval")
    return parser


def main(args: argparse.Namespace) -> None:
    input_dir = Path(os.path.expanduser(args.input_dir)).resolve()
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_file_paths, selected_splits = _collect_selected_files(input_dir, args.split)
    if not candidate_file_paths:
        raise ValueError(f"no activation dump files found under {input_dir} for split={args.split!r}")

    print(f"Scanning {len(candidate_file_paths)} activation files from {selected_splits}.")
    run_config = _load_run_config(input_dir)
    candidate_entries = _scan_candidate_entries(candidate_file_paths, input_dir)
    available_labels = sorted({label for _, label in candidate_entries})
    selected_labels = sorted(set(args.labels)) if args.labels is not None else available_labels
    if len(selected_labels) == 0:
        raise ValueError("no labels were selected for variation calculation")
    missing_labels = [label for label in selected_labels if label not in available_labels]
    if missing_labels:
        raise ValueError(f"requested labels are not present in the activation dump: {missing_labels}")
    label_quota = _make_label_quota(selected_labels, args.num_samples)
    target_selected = None if label_quota is None else sum(label_quota.values())
    expected_layer_names: list[str] | None = None
    overall_stats: OrderedDict[str, RunningTensorStats] | None = None
    per_label_stats: dict[int, OrderedDict[str, RunningTensorStats]] = {}
    selected_per_label = {label: 0 for label in selected_labels}
    label_counts: dict[int, int] = {}
    matched_count = 0
    scanned_count = 0

    for path, label in candidate_entries:
        if target_selected is not None and matched_count >= target_selected:
            break
        scanned_count += 1
        if label not in selected_per_label:
            if scanned_count % args.log_every == 0:
                print(f"Scanned {scanned_count}/{len(candidate_file_paths)} files, matched {matched_count}.")
            continue
        if label_quota is not None and selected_per_label[label] >= label_quota[label]:
            if scanned_count % args.log_every == 0:
                print(f"Scanned {scanned_count}/{len(candidate_file_paths)} files, matched {matched_count}.")
            continue

        payload = _load_payload(path)
        layers = _filter_layers_in_forward_order(_extract_layer_tensors(payload))
        current_layer_names = list(layers.keys())
        if expected_layer_names is None:
            if len(current_layer_names) == 0:
                raise ValueError(f"no non-BatchNorm layers were found in {path}")
            expected_layer_names = current_layer_names
            overall_stats = _initialize_running_stats(layers)
        elif current_layer_names != expected_layer_names:
            raise ValueError(f"layer mismatch in {path}: expected {expected_layer_names[:5]}..., got {current_layer_names[:5]}...")

        if label not in per_label_stats:
            per_label_stats[label] = _initialize_running_stats(layers)
        selected_per_label[label] += 1
        label_counts[label] = label_counts.get(label, 0) + 1
        for layer_name, tensor in layers.items():
            assert overall_stats is not None
            overall_stats[layer_name].update(tensor)
            per_label_stats[label][layer_name].update(tensor)
        matched_count += 1

        if matched_count % args.log_every == 0 or (target_selected is not None and matched_count == target_selected):
            print(f"Scanned {scanned_count}/{len(candidate_file_paths)} files, matched {matched_count}.")

    if overall_stats is None or expected_layer_names is None or matched_count == 0:
        raise ValueError(f"no activation dump files matched split={args.split!r} and labels={selected_labels}")

    print(f"Finished scanning {scanned_count} files and using {matched_count} samples.")

    layer_summaries, layer_tensors = _finalize_stats_collection(overall_stats, include_tensors=True)
    assert layer_tensors is not None
    per_label_layer_summaries: OrderedDict[str, OrderedDict[str, dict[str, Any]]] = OrderedDict()
    for label in selected_labels:
        if label not in per_label_stats:
            continue
        label_summary, _ = _finalize_stats_collection(per_label_stats[label], include_tensors=False)
        per_label_layer_summaries[str(label)] = label_summary

    result = {
        "metadata": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "split": args.split,
            "selected_splits": selected_splits,
            "num_candidate_files": len(candidate_file_paths),
            "num_scanned_files": scanned_count,
            "num_samples": matched_count,
            "requested_num_samples": args.num_samples,
            "max_samples": args.num_samples,
            "labels": selected_labels,
            "label_counts": {str(label): count for label, count in sorted(label_counts.items())},
            "per_label_quota": None if label_quota is None else {str(label): count for label, count in sorted(label_quota.items())},
            "ignored_layers": "all BatchNorm layers are skipped, including bn* and downsample.1",
            "report_order": "forward_order_after_batchnorm_filtering",
            "variation_definition": "amount_of_variation = mean(sample_variance_tensor) with sample variance denominator n-1; reported for all selected samples and separately within each selected label",
            "source_run_config": dict(run_config) if run_config is not None else None,
            "layer_order": list(layer_summaries.keys()),
            "per_label_report": "summary_only",
        },
        "layer_summaries": layer_summaries,
        "layers": layer_tensors,
        "per_label_layer_summaries": per_label_layer_summaries,
    }
    json_result = {"metadata": result["metadata"], "layer_summaries": layer_summaries, "per_label_layer_summaries": per_label_layer_summaries}

    pt_path = output_dir / f"variation_{args.split}.pt"
    json_path = output_dir / f"variation_{args.split}.json"
    torch.save(result, pt_path)
    with open(json_path, "w", encoding="utf-8") as file_handle:
        json.dump(json_result, file_handle, indent=2)

    print(f"Saved tensor statistics to {pt_path}.")
    print(f"Saved summary statistics to {json_path}.")
    print("Overall variation, first layers in forward order:")
    for layer_name, summary in list(layer_summaries.items())[:10]:
        print(f"  {layer_name}: {summary['amount_of_variation']:.8f}")
    if len(per_label_layer_summaries) > 0:
        first_label = next(iter(per_label_layer_summaries))
        print(f"Within-label variation example for label {first_label}, first layers:")
        for layer_name, summary in list(per_label_layer_summaries[first_label].items())[:5]:
            print(f"  {layer_name}: {summary['amount_of_variation']:.8f}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
