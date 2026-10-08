"""ImageNet DALI loaders for the *repartitioned* misalignment datasets.

DALI is imported lazily, so all other datasets retain their existing PyTorch
path without a DALI dependency. Only file names and labels are materialized;
image decoding/resizing happens in bounded, prefetched CPU/GPU batches.
"""

from __future__ import annotations

from bisect import bisect_right
import math
from typing import Any

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset


def resolve_loader_backend(dataset_name: str, requested: str = "auto") -> str:
    if requested not in {"auto", "pytorch", "dali"}:
        raise ValueError(f"unsupported loader backend {requested!r}")
    backend = ("dali" if dataset_name == "imagenet1k" else "pytorch") if requested == "auto" else requested
    if backend == "dali" and dataset_name != "imagenet1k":
        raise ValueError("DALI loader backend is supported only for imagenet1k")
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


class _RandomEraseSource:
    """Generate tiny erase rectangles, not images, on the CPU.

    Match torchvision's p=.1, scale=.02..33, log-uniform aspect ratio .3..3.3,
    and ten rejection attempts. Erase raw RGB with zero *before* normalization,
    as in this project's ImageNet preset 2 (not normalized tensors with zero).
    A zero-sized rectangle means no erasure. This independent RNG never
    consumes the model/dropout/Hutchinson torch RNG streams.
    """

    def __init__(self, batch_size: int, crop_size: int, seed: int) -> None:
        import numpy as np

        self.np = np
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.crop_size = crop_size

    def __call__(self):
        anchors, shapes = [], []
        size = self.crop_size
        for _ in range(self.batch_size):
            anchor, shape = (0, 0), (0, 0)
            if self.rng.random() < 0.1:
                for _ in range(10):
                    area = size * size * self.rng.uniform(0.02, 0.33)
                    aspect = math.exp(self.rng.uniform(math.log(0.3), math.log(3.3)))
                    height = round(math.sqrt(area * aspect))
                    width = round(math.sqrt(area / aspect))
                    if 0 < height < size and 0 < width < size:
                        anchor = (self.rng.integers(size - height + 1), self.rng.integers(size - width + 1))
                        shape = (height, width)
                        break
            anchors.append(self.np.asarray(anchor, dtype=self.np.int32))
            shapes.append(self.np.asarray(shape, dtype=self.np.int32))
        return anchors, shapes


class DaliImageNetLoader:
    """DataLoader-shaped GPU iterator over one canonical, fixed-order half.

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
    ) -> None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("DALI ImageNet loading requires CUDA; use --loader_backend pytorch on CPU")
        if batch_size <= 0 or num_workers < 0 or prefetch_factor <= 0:
            raise ValueError("invalid DALI batch size, thread count or prefetch depth")
        if augmentation_config.get("dataset") != "imagenet1k":
            raise ValueError("DALI requires an ImageNet augmentation configuration")
        self.level = int(augmentation_config["level"])
        if self.level not in (0, 1, 2):
            raise ValueError("DALI ImageNet augmentation supports none/0, 1 and 2")
        self.crop_size = int(augmentation_config["input_size"])
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.device_id = device.index if device.index is not None else torch.cuda.current_device()
        self.num_threads = max(1, int(num_workers))
        self.seed = int(seed)
        self.prefetch_factor = int(prefetch_factor)
        self._dali_api = _import_dali()
        self.files, self.labels = image_file_manifest(dataset)
        self._pipeline = None
        self._iterator = None
        self._active = False

    def __len__(self) -> int:
        return math.ceil(len(self.files) / self.batch_size)

    def runtime_info(self) -> dict[str, Any]:
        return {
            "backend": "dali", "dali_version": self._dali_api[0].__version__,
            "device_id": self.device_id, "num_threads": self.num_threads,
            "prefetch_queue_depth": self.prefetch_factor, "decoder_device": "mixed",
            "resize_normalize_device": "gpu", "input_size": self.crop_size,
            "last_batch_policy": "partial", "shuffle": False,
            "repeated_augmentation": False, "mixup_cutmix": False,
            "augmentation_seed": self.seed,
            "augmentation_rng": "partition_seed + 1000003 * epoch; reset for each pass",
            "augmentation_implementation": "DALI-native equivalents of the configured torchvision operations",
            "trivial_augment_max_translate_pixels": 32 if self.level == 2 else None,
            "rand_augment_max_translate_fraction": 150.0 / 331.0 if self.level == 2 else None,
            "torchvision_bitwise_equivalent": False,
        }

    def _build_pipeline(self) -> None:
        _, fn, types, pipeline_def, iterator_type, last_batch_policy = self._dali_api
        pass_seed = (self.seed + 1_000_003 * _augmentation_epoch(self.dataset)) % (2**31 - 1)
        crop_size, level = self.crop_size, self.level
        erase_source = _RandomEraseSource(self.batch_size, crop_size, pass_seed + 71) if level == 2 else None

        @pipeline_def(enable_conditionals=level == 2)
        def image_pipeline():
            encoded, labels = fn.readers.file(
                files=self.files, labels=self.labels, random_shuffle=False,
                shuffle_after_epoch=False, pad_last_batch=True, name="Reader",
            )
            if level == 0:
                images = fn.decoders.image(
                    encoded, device="mixed", output_type=types.RGB,
                    adjust_orientation=False, jpeg_fancy_upsampling=True,
                )
                images = fn.resize(
                    images, device="gpu", resize_shorter=math.ceil(256 * crop_size / 224),
                    interp_type=types.INTERP_LINEAR, antialias=True,
                )
            else:
                images = fn.decoders.image_random_crop(
                    encoded, device="mixed", output_type=types.RGB,
                    # nvJPEG ROI decoding cannot reliably use fancy upsampling
                    # on all supported CUDA versions. Keep its safe default.
                    adjust_orientation=False, jpeg_fancy_upsampling=False,
                    random_area=[0.08, 1.0], random_aspect_ratio=[0.75, 4.0 / 3.0],
                    num_attempts=10,
                )
                images = fn.resize(
                    images, device="gpu", resize_x=crop_size, resize_y=crop_size,
                    interp_type=types.INTERP_LINEAR, antialias=True,
                )
                images = fn.flip(images, device="gpu", horizontal=fn.random.coin_flip(probability=0.5))
                if level == 2:
                    from nvidia.dali.auto_aug import trivial_augment, rand_augment

                    images = trivial_augment.trivial_augment_wide(
                        # Passing shape without an explicit translation limit
                        # would change DALI's default from 32 pixels to 100%
                        # of the image. Torchvision TrivialAugmentWide uses 32.
                        images, max_translate_abs=32, fill_value=0, interp_type=types.INTERP_NN,
                    )
                    images = rand_augment.rand_augment(
                        images, n=2, m=9, shape=(crop_size, crop_size), fill_value=0,
                        interp_type=types.INTERP_NN, max_translate_rel=150.0 / 331.0,
                    )
                    anchors, shapes = fn.external_source(
                        source=erase_source, num_outputs=2, batch=True,
                        dtype=[types.INT32, types.INT32], ndim=[1, 1],
                    )
                    images = fn.erase(
                        images, device="gpu", anchor=fn.cast(anchors, dtype=types.FLOAT),
                        shape=fn.cast(shapes, dtype=types.FLOAT),
                        axis_names="HW", normalized_anchor=False, normalized_shape=False, fill_value=0,
                    )
            images = fn.crop_mirror_normalize(
                images, device="gpu", dtype=types.FLOAT, output_layout="CHW",
                crop=(crop_size, crop_size), crop_pos_x=0.5, crop_pos_y=0.5,
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            )
            return images, labels.gpu()

        pipeline = image_pipeline(
            batch_size=self.batch_size, num_threads=self.num_threads,
            device_id=self.device_id, seed=pass_seed,
            prefetch_queue_depth=self.prefetch_factor,
            # The conventional executor keeps memory bounded by prefetch depth.
            exec_dynamic=False,
        )
        pipeline.build()
        self._pipeline = pipeline
        self._iterator = iterator_type(
            pipeline, ["data", "label"], reader_name="Reader", auto_reset=False,
            last_batch_policy=last_batch_policy.PARTIAL, prepare_first_batch=False,
        )

    def __iter__(self):
        if self._active:
            raise RuntimeError("Cannot traverse a DALI partition concurrently")
        self._active = True
        completed = False
        try:
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
