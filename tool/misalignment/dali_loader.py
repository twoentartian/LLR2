"""ImageNet DALI loaders for the *repartitioned* misalignment datasets.

DALI is imported lazily, so all other datasets retain their existing PyTorch
path without a DALI dependency. Only file names and labels are materialized;
image decoding/resizing uses the same pipeline as generate_high_accuracy_model.
This module adapts the partition manifest, iteration lifecycle and pass seed.
"""

from __future__ import annotations

from bisect import bisect_right
import math
from typing import Any

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset
from py_src import distributed
from .augmentation import IMAGENET_DATASETS
from .epoch_order import ORDER_RNG, epoch_order, shard_order


def resolve_loader_backend(dataset_name: str, requested: str = "auto") -> str:
    if requested not in {"auto", "pytorch", "dali"}:
        raise ValueError(f"unsupported loader backend {requested!r}")
    backend = ("dali" if dataset_name in IMAGENET_DATASETS else "pytorch") if requested == "auto" else requested
    if backend == "dali" and dataset_name not in IMAGENET_DATASETS:
        raise ValueError("DALI loader backend is supported only for imagenet1k and imagenet21k_p")
    return backend


def _sample_at(dataset: Dataset, index: int) -> tuple[str, int]:
    """Resolve nested partition indices without calling image __getitem__."""
    if isinstance(dataset, Subset):
        return _sample_at(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, ConcatDataset):
        child = bisect_right(dataset.cumulative_sizes, index)
        start = 0 if child == 0 else dataset.cumulative_sizes[child - 1]
        return _sample_at(dataset.datasets[child], index - start)
    # AugmentedDataset is defined in the launcher; avoid a circular import.
    if hasattr(dataset, "transform") and hasattr(dataset, "stochastic") and hasattr(dataset, "dataset"):
        return _sample_at(dataset.dataset, index)
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise TypeError("DALI ImageNet loading requires ImageFolder samples (paths and labels)")
    path, label = samples[index]
    return str(path), int(label)


def image_file_manifest(dataset: Dataset) -> tuple[list[str], list[int]]:
    if len(dataset) == 0:
        raise ValueError("DALI cannot load an empty partition")
    files, labels = [], []
    for index in range(len(dataset)):
        path, label = _sample_at(dataset, index)
        files.append(path)
        labels.append(label)
    return files, labels


def _augmentation_epoch(dataset: Dataset) -> int:
    if isinstance(dataset, Subset):
        return _augmentation_epoch(dataset.dataset)
    return int(getattr(dataset, "epoch", 0))


def _import_dali():
    try:
        import nvidia.dali as dali
        from nvidia.dali import fn, types
        from nvidia.dali.pipeline import pipeline_def
        from nvidia.dali.plugin.pytorch import DALIGenericIterator
        from nvidia.dali.plugin.base_iterator import LastBatchPolicy
    except (ImportError, OSError) as exc:
        raise ImportError(
            "ImageNet's default DALI backend requires a working NVIDIA DALI installation "
            "in this Python environment. Use --loader_backend pytorch for the old loader."
        ) from exc
    return dali, fn, types, pipeline_def, DALIGenericIterator, LastBatchPolicy


class DaliImageNetLoader:
    """DataLoader-shaped GPU iterator over one half, in fixed or per-epoch order.

    With ``shuffle`` each pass reads the half in ``epoch_order`` for the
    current augmentation epoch (see misalignment/epoch_order.py); otherwise
    in the canonical partition order.
    Stochastic pipelines are rebuilt with partition+epoch seeds at each pass.
    Thus evaluation, checkpoint flatness and a second swapped orientation do
    not inherit advanced reader/RNG state from a previous traversal. Level
    zero can reuse its deterministic pipeline after a complete pass. An
    interrupted pass is discarded instead of resetting an unfinished epoch.
    """

    backend = "dali"

    def __init__(
        self, dataset: Dataset, batch_size: int, *, device: torch.device,
        num_workers: int, seed: int, prefetch_factor: int,
        augmentation_config: dict[str, Any],
        is_train: bool = True,
        distributed_loader: bool = True,
        shuffle: bool = False,
    ) -> None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("DALI ImageNet loading requires CUDA; use --loader_backend pytorch on CPU")
        if batch_size <= 0 or num_workers < 0 or prefetch_factor <= 0:
            raise ValueError("invalid DALI batch size, thread count or prefetch depth")
        if augmentation_config.get("dataset") not in IMAGENET_DATASETS:
            raise ValueError("DALI requires an ImageNet augmentation configuration")
        self.level = int(augmentation_config["level"])
        if augmentation_config["dataset"] == "imagenet21k_p" and self.level not in (0, 1):
            raise ValueError("ImageNet-21k-p DALI supports augmentation none/0 and 1 only")
        if self.level not in (0, 1, 2, 3):
            raise ValueError("DALI ImageNet augmentation supports none/0, 1, 2 and 3")
        self.crop_size = int(augmentation_config["input_size"])
        self.dataset = dataset
        self.world_size = distributed.world_size() if distributed_loader else 1
        self.rank = distributed.rank() if distributed_loader else 0
        self.batch_size = distributed.local_batch_size(batch_size) if self.world_size > 1 else int(batch_size)
        self.device_id = device.index if device.index is not None else torch.cuda.current_device()
        self.num_threads = max(1, int(num_workers))
        self.seed = int(seed) + self.rank
        # The order is global, so its seed excludes the rank.
        self.order_seed = int(seed)
        self.shuffle = bool(shuffle)
        self.prefetch_factor = int(prefetch_factor)
        self._dali_api = _import_dali()
        self._pad = is_train
        self.padded_examples = 0
        if self.world_size > 1 and is_train:
            total = math.ceil(len(dataset) / self.world_size) * self.world_size
            self.padded_examples = total - len(dataset)
        # Shard the global order, so each paired global batch contains the
        # same source examples as the PyTorch backend.
        self._all_files, self._all_labels = image_file_manifest(dataset) if len(dataset) else ([], [])
        self._order_epoch = None
        self._set_order(0)
        self._pipeline = None
        self._iterator = None
        self._active = False

    def _set_order(self, epoch: int) -> None:
        order = epoch_order(len(self._all_files), seed=self.order_seed, epoch=epoch, shuffle=self.shuffle)
        local = shard_order(order, world_size=self.world_size, rank=self.rank, pad=self._pad)
        self.files = [self._all_files[index] for index in local]
        self.labels = [self._all_labels[index] for index in local]
        self._order_epoch = epoch

    def __len__(self) -> int:
        return math.ceil(len(self.files) / self.batch_size)

    def runtime_info(self) -> dict[str, Any]:
        return {
            "backend": "dali", "dali_version": self._dali_api[0].__version__,
            "device_id": self.device_id, "num_threads": self.num_threads,
            "rank": self.rank, "world_size": self.world_size,
            "local_examples": len(self.files), "padded_training_examples_global": self.padded_examples,
            "prefetch_queue_depth": self.prefetch_factor, "decoder_device": "cpu",
            "resize_normalize_device": "gpu", "input_size": self.crop_size,
            "last_batch_policy": "partial", "shuffle": self.shuffle,
            "order_rng": ORDER_RNG if self.shuffle else "fixed partition order",
            "repeated_augmentation": False, "mixup_cutmix": False,
            "augmentation_seed": self.seed,
            "augmentation_rng": "partition_seed + 1000003 * epoch; reset for each pass",
            "augmentation_implementation": "shared high-accuracy ImageNet DALI pipeline",
            "preprocessing_source": "py_src.ml_setup_dataset.dataset_imagenet.build_dali_imagenet_pipeline",
            "torchvision_bitwise_equivalent": False,
        }

    def _build_pipeline(self) -> None:
        from py_src.ml_setup_dataset.dataset_imagenet import build_dali_imagenet_pipeline

        _, _, _, _, iterator_type, last_batch_policy = self._dali_api
        pass_seed = (self.seed + 1_000_003 * _augmentation_epoch(self.dataset)) % (2**31 - 1)
        pipeline = build_dali_imagenet_pipeline(
            reader_kwargs={"files": self.files, "labels": self.labels, "random_shuffle": False,
                           "shuffle_after_epoch": False, "pad_last_batch": True, "name": "Reader"},
            training=self.level > 0, crop_size=self.crop_size,
            resize_size=math.ceil(256 * self.crop_size / 224),
            preset_version=2 if self.level == 2 else 1,
            auto_augment_policy="imagenet" if self.level == 3 else None,
            random_erase_prob=0.1 if self.level == 2 else 0.0,
            batch_size=self.batch_size, num_threads=self.num_threads, device_id=self.device_id,
            labels_on_gpu=True,
            pipeline_kwargs={"seed": pass_seed, "prefetch_queue_depth": self.prefetch_factor},
        )
        pipeline.build()
        self._pipeline = pipeline
        self._iterator = iterator_type(
            pipeline, ["data", "label"], reader_name="Reader", auto_reset=False,
            last_batch_policy=last_batch_policy.PARTIAL, prepare_first_batch=False,
        )

    def __iter__(self):
        if not self.files:
            return
        if self._active:
            raise RuntimeError("Cannot traverse a DALI partition concurrently")
        self._active = True
        completed = False
        try:
            epoch = _augmentation_epoch(self.dataset)
            if self.shuffle and epoch != self._order_epoch:
                self.close()
                self._set_order(epoch)
            if self.level > 0:
                self.close()
            if self._iterator is None:
                self._build_pipeline()
            for batch in self._iterator:
                values = batch[0]
                # reshape, not squeeze: a one-example final batch stays 1-D.
                yield values["data"], values["label"].reshape(-1).long()
            completed = True
        finally:
            self._active = False
            if completed and self.level == 0:
                self._iterator.reset()
            else:
                self.close()

    def close(self) -> None:
        # DALI owns its buffers/threads; dropping the iterator and pipeline
        # releases them without relying on private _shutdown/reset APIs.
        self._iterator = None
        self._pipeline = None


def close_dali_loaders(*loaders) -> None:
    for loader in loaders:
        if isinstance(loader, DaliImageNetLoader):
            loader.close()


def loader_runtime_info(loader) -> dict[str, Any]:
    if isinstance(loader, DaliImageNetLoader):
        return loader.runtime_info()
    return {"backend": "pytorch"}
