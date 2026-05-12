from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from py_src.ml_setup.resnet import resnet18_cifar10, resnet50_imagenet1k
from py_src.model_opti_save_load import load_model_state_file


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset) # type: ignore

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        return image, target, index


def _is_leaf_module(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


def _to_float16_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        output = value.detach().to(device="cpu", dtype=torch.float16).contiguous()
        if output.ndim > 0 and output.shape[0] == 1:
            output = output.squeeze(0)
        return output
    if isinstance(value, Mapping):
        return {str(key): _to_float16_cpu(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        converted = [_to_float16_cpu(item) for item in value]
        return tuple(converted) if isinstance(value, tuple) else converted
    raise TypeError(f"unsupported activation type {type(value)!r}")


class ActivationRecorder:
    def __init__(self, model: nn.Module):
        self._records: OrderedDict[str, Any] = OrderedDict()
        self._module_calls: defaultdict[str, int] = defaultdict(int)
        self._call_index = 0
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.leaf_module_names: list[str] = []
        for module_name, module in model.named_modules():
            if not module_name or not _is_leaf_module(module):
                continue
            self.leaf_module_names.append(module_name)
            self._handles.append(module.register_forward_hook(self._build_hook(module_name)))

    def _build_hook(self, module_name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            self._call_index += 1
            self._module_calls[module_name] += 1
            module_call_count = self._module_calls[module_name]
            record_name = f"{self._call_index:04d}:{module_name}"
            if module_call_count > 1:
                record_name = f"{record_name}#{module_call_count}"
            self._records[record_name] = _to_float16_cpu(output)

        return hook

    def reset(self) -> None:
        self._records = OrderedDict()
        self._module_calls = defaultdict(int)
        self._call_index = 0

    def snapshot(self) -> OrderedDict[str, Any]:
        return OrderedDict(self._records)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _load_state_dict_from_path(path: str) -> dict[str, Any]:
    try:
        state_dict, _, _ = load_model_state_file(path)
        return state_dict
    except Exception:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                return checkpoint["state_dict"]
            if "model" in checkpoint and isinstance(checkpoint["model"], dict):
                return checkpoint["model"]
            if checkpoint and all(isinstance(key, str) for key in checkpoint.keys()):
                if all(torch.is_tensor(value) for value in checkpoint.values()):
                    return checkpoint
        raise ValueError(f"could not extract a state_dict from {path}")


def _build_workload(workload: str) -> tuple[nn.Module, Dataset, Dataset]:
    if workload == "resnet18_cifar10":
        setup = resnet18_cifar10()
        model = setup.model
        train_dataset = setup.training_data
        val_dataset = setup.testing_data
    elif workload == "resnet50_imagenet1k":
        setup = resnet50_imagenet1k()
        model = setup.model
        train_dataset = setup.training_data
        val_dataset = setup.testing_data
    else:
        raise ValueError(f"unsupported workload {workload!r}")

    train_dataset.transform = val_dataset.transform
    return model, train_dataset, val_dataset


def _make_loader(dataset: Dataset, num_workers: int) -> DataLoader:
    return DataLoader(
        IndexedDataset(dataset),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _resolve_label_ids(dataset: Dataset) -> list[int]:
    if hasattr(dataset, "classes"):
        classes = dataset.classes  # type: ignore[attr-defined]
        if isinstance(classes, Sequence):
            return list(range(len(classes)))
    if hasattr(dataset, "targets"):
        targets = dataset.targets  # type: ignore[attr-defined]
        if torch.is_tensor(targets):
            return sorted(int(label) for label in targets.unique().tolist())
        return sorted({int(label) for label in targets})
    if hasattr(dataset, "samples"):
        samples = dataset.samples  # type: ignore[attr-defined]
        return sorted({int(sample[1]) for sample in samples})
    raise ValueError("could not infer dataset label ids for balanced sampling")


def _make_label_quota(dataset: Dataset, num_samples: int | None) -> dict[int, int] | None:
    if num_samples is None:
        return None
    label_ids = _resolve_label_ids(dataset)
    if len(label_ids) == 0:
        raise ValueError("cannot build a label quota for a dataset with no labels")
    base_quota = num_samples // len(label_ids)
    remainder = num_samples % len(label_ids)
    return {label: base_quota + (1 if index < remainder else 0) for index, label in enumerate(label_ids)}


def _resolve_source_path(dataset: Dataset, index: int) -> str | None:
    if hasattr(dataset, "samples"):
        sample = dataset.samples[index]  # type: ignore[index]
        if isinstance(sample, Sequence) and sample:
            source_path = sample[0]
            if isinstance(source_path, str):
                return os.path.abspath(source_path)
    if hasattr(dataset, "imgs"):
        sample = dataset.imgs[index]  # type: ignore[index]
        if isinstance(sample, Sequence) and sample:
            source_path = sample[0]
            if isinstance(source_path, str):
                return os.path.abspath(source_path)
    return None


def _save_sample(
    *,
    model: nn.Module,
    recorder: ActivationRecorder,
    image: torch.Tensor,
    target: torch.Tensor,
    dataset: Dataset,
    dataset_index: int,
    split_name: str,
    output_path: Path,
    device: torch.device,
    workload: str,
) -> None:
    recorder.reset()
    image = image.to(device, non_blocking=True)
    with torch.inference_mode():
        _ = model(image)

    payload = {
        "metadata": {
            "workload": workload,
            "split": split_name,
            "dataset_index": dataset_index,
            "label": int(target.item()),
            "source_path": _resolve_source_path(dataset, dataset_index),
            "saved_dtype": "float16",
            "batch_dim_removed": True,
        },
        "input": _to_float16_cpu(image),
        "activations": recorder.snapshot(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def _process_split(
    *,
    model: nn.Module,
    recorder: ActivationRecorder,
    dataset: Dataset,
    split_name: str,
    split_output_dir: Path,
    device: torch.device,
    workload: str,
    num_workers: int,
    max_samples_per_split: int | None,
    overwrite: bool,
    log_every: int,
) -> None:
    loader = _make_loader(dataset, num_workers)
    label_quota = _make_label_quota(dataset, max_samples_per_split)
    selected_per_label = {} if label_quota is None else {label: 0 for label in label_quota}
    target_selected = None if label_quota is None else sum(label_quota.values())
    scanned_samples = 0
    selected_samples = 0
    saved_samples = 0
    skipped_samples = 0

    for image, target, dataset_index_tensor in loader:
        if target_selected is not None and selected_samples >= target_selected:
            break
        scanned_samples += 1
        dataset_index = int(dataset_index_tensor.item())
        label = int(target.item())
        if label_quota is not None:
            label_target = label_quota.get(label, 0)
            label_count = selected_per_label.get(label, 0)
            if label_count >= label_target:
                if scanned_samples % log_every == 0:
                    print(f"[{split_name}] scanned={scanned_samples} selected={selected_samples}/{target_selected} saved={saved_samples} skipped={skipped_samples}")
                continue
            selected_per_label[label] = label_count + 1
        selected_samples += 1
        output_path = split_output_dir / str(label) / f"{dataset_index:08d}.pt"

        if output_path.exists() and not overwrite:
            skipped_samples += 1
        else:
            _save_sample(
                model=model,
                recorder=recorder,
                image=image,
                target=target,
                dataset=dataset,
                dataset_index=dataset_index,
                split_name=split_name,
                output_path=output_path,
                device=device,
                workload=workload,
            )
            saved_samples += 1

        if selected_samples % log_every == 0 or (target_selected is not None and selected_samples == target_selected):
            selected_total = selected_samples if target_selected is None else f"{selected_samples}/{target_selected}"
            print(f"[{split_name}] scanned={scanned_samples} selected={selected_total} saved={saved_samples} skipped={skipped_samples}")


def _write_run_metadata(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    train_dataset: Dataset,
    val_dataset: Dataset,
    recorder: ActivationRecorder,
    device: torch.device,
) -> None:
    metadata = {
        "workload": args.workload,
        "device": str(device),
        "weights_path": args.weights_path,
        "num_workers": args.num_workers,
        "num_samples": args.num_samples_per_split,
        "max_samples": args.num_samples_per_split,
        "max_samples_per_split": args.num_samples_per_split,
        "overwrite": args.overwrite,
        "train_size": len(train_dataset), # type: ignore
        "val_size": len(val_dataset), # type: ignore
        "train_transform": repr(train_dataset.transform), # type: ignore
        "val_transform": repr(val_dataset.transform), # type: ignore
        "output_layout": "split/<label>/<dataset_index>.pt",
        "sampling_strategy": "balanced_per_label" if args.num_samples_per_split is not None else "all_samples",
        "leaf_module_names": recorder.leaf_module_names,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dump float16 inputs and per-layer activations for one sample per file. "
            "Only ResNet18+CIFAR10 and ResNet50+ImageNet1k are supported."
        )
    )
    parser.add_argument("--workload", choices=("resnet18_cifar10", "resnet50_imagenet1k"), required=True, help="model + dataset pair to run")
    parser.add_argument("--output-dir", required=True, help="directory that will receive run_config.json plus train/ and val/ activation files")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="torch device for the forward pass")
    parser.add_argument("--weights-path", default=None, help="optional model checkpoint or raw state_dict to load after the architecture is built")
    parser.add_argument("--num-workers", type=int, default=0, help="number of dataloader workers")
    parser.add_argument("--num-samples", "--max-samples", "--max-samples-per-split", dest="num_samples_per_split", type=int, default=None, help="process up to N samples in each split, balanced as evenly as possible across labels")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False, help="overwrite existing sample files; by default existing files are skipped")
    parser.add_argument("--log-every", type=int, default=50, help="progress print interval")
    parser.add_argument("--strict-load", action=argparse.BooleanOptionalAction, default=True, help="use strict state_dict loading when --weights-path is provided")
    return parser


def main(args: argparse.Namespace) -> None:
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    device = torch.device(args.device)
    using_random_init = args.weights_path is None

    model, train_dataset, val_dataset = _build_workload(args.workload)
    if args.weights_path is not None:
        state_dict = _load_state_dict_from_path(os.path.expanduser(args.weights_path))
        model.load_state_dict(state_dict, strict=args.strict_load)
    elif using_random_init:
        print(f"Using randomly initialized weights for {args.workload}.")

    model.eval()
    model.to(device)
    recorder = ActivationRecorder(model)
    try:
        _write_run_metadata(
            output_dir,
            args=args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            recorder=recorder,
            device=device,
        )
        print(
            "Saving activations with validation preprocessing for both splits. "
            "This can produce very large outputs for full datasets."
        )
        _process_split(
            model=model,
            recorder=recorder,
            dataset=train_dataset,
            split_name="train",
            split_output_dir=output_dir / "train",
            device=device,
            workload=args.workload,
            num_workers=args.num_workers,
            max_samples_per_split=args.num_samples_per_split,
            overwrite=args.overwrite,
            log_every=args.log_every,
        )
        _process_split(
            model=model,
            recorder=recorder,
            dataset=val_dataset,
            split_name="val",
            split_output_dir=output_dir / "val",
            device=device,
            workload=args.workload,
            num_workers=args.num_workers,
            max_samples_per_split=args.num_samples_per_split,
            overwrite=args.overwrite,
            log_every=args.log_every,
        )
    finally:
        recorder.close()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
