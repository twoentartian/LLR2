"""GPU-resident NVIDIA DALI batch loading for the simulator.

The simulator samples dataset indices per node before loading data.  This
module preserves that behavior: the complete raw image and target arrays are
cached on one CUDA device, and each finite ``BatchRequest`` is gathered from
that cache and augmented by a DALI pipeline.

DALI is imported lazily so importing the simulator does not make it a required
dependency for the normal PyTorch ``SharedDataLoader`` path.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator

import torch

from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup_model import ModelType
from py_src.shared_dataloader import BatchRequest, RoutedBatch


LOGGER = logging.getLogger("SimulatorBase")

_SUPPORTED_WORKLOADS = frozenset(
    (
        (ModelType.lenet4, DatasetType.mnist),
        (ModelType.lenet5, DatasetType.mnist),
        (ModelType.mobilenet_v2, DatasetType.cifar10),
        (ModelType.resnet18_bn, DatasetType.cifar10),
        (ModelType.resnet18_bn, DatasetType.cifar100),
        (ModelType.cct_7_3x1_32, DatasetType.cifar10),
    )
)

_CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
_CIFAR10_STD = (0.24703223, 0.24348513, 0.26158784)
_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def is_supported_dali_workload(ml_setup: Any) -> bool:
    """Return whether ``ml_setup`` is one of the explicitly supported pairs."""

    return (ml_setup.model_type, ml_setup.dataset_type) in _SUPPORTED_WORKLOADS


def ensure_supported_dali_workload(ml_setup: Any) -> None:
    """Reject DALI for model/dataset pairs outside the requested scope."""

    if is_supported_dali_workload(ml_setup):
        return
    model_name = getattr(getattr(ml_setup, "model_type", None), "name", "unknown")
    dataset_name = getattr(getattr(ml_setup, "dataset_type", None), "name", "unknown")
    supported = ", ".join(
        f"{model.name}/{dataset.name}"
        for model, dataset in sorted(
            _SUPPORTED_WORKLOADS,
            key=lambda pair: (pair[0].name, pair[1].name),
        )
    )
    raise ValueError(
        f"DALI simulator loading is not supported for {model_name}/{dataset_name}; "
        f"supported pairs are: {supported}"
    )


def _as_raw_image_tensor(dataset: Any, dataset_type: DatasetType) -> torch.Tensor:
    if not hasattr(dataset, "data"):
        raise TypeError(
            "DALI simulator loading requires a torchvision-style dataset with "
            "an untransformed .data array"
        )

    images = torch.as_tensor(getattr(dataset, "data"))
    if images.dtype != torch.uint8:
        raise TypeError(
            f"DALI raw images must use uint8 storage, got {images.dtype}"
        )

    if dataset_type == DatasetType.mnist:
        if images.ndim != 3 or tuple(images.shape[1:]) != (28, 28):
            raise ValueError(
                "MNIST DALI input must have shape [N, 28, 28], "
                f"got {tuple(images.shape)}"
            )
        images = images.unsqueeze(-1)
    elif dataset_type in (DatasetType.cifar10, DatasetType.cifar100):
        if images.ndim != 4 or tuple(images.shape[1:]) != (32, 32, 3):
            raise ValueError(
                "CIFAR DALI input must have shape [N, 32, 32, 3], "
                f"got {tuple(images.shape)}"
            )
    else:  # Guarded by ensure_supported_dali_workload, kept defensive here.
        raise ValueError(f"unsupported DALI dataset type: {dataset_type.name}")

    return images.contiguous()


def _as_target_tensor(dataset: Any, image_count: int) -> torch.Tensor:
    if not hasattr(dataset, "targets"):
        raise TypeError(
            "DALI simulator loading requires a torchvision-style dataset with "
            "a .targets array"
        )
    targets = torch.as_tensor(getattr(dataset, "targets"), dtype=torch.int64)
    targets = targets.reshape(-1).contiguous()
    if targets.numel() != image_count:
        raise ValueError(
            f"DALI image/target length mismatch: {image_count} images and "
            f"{targets.numel()} targets"
        )
    return targets


def _normalization_parameters(
    dataset_type: DatasetType,
    raw_images: torch.Tensor,
) -> tuple[list[float], list[float]]:
    if dataset_type == DatasetType.mnist:
        # Match dataset_mnist(): statistics are computed from the raw training
        # pixels after conversion to [0, 1].
        pixels = raw_images.to(dtype=torch.float32)
        mean = float(pixels.mean().item())
        std = float(pixels.std().item())
        if std <= 0:
            raise ValueError("MNIST DALI input has zero pixel standard deviation")
        return [mean], [std]
    if dataset_type == DatasetType.cifar10:
        return (
            [value * 255.0 for value in _CIFAR10_MEAN],
            [value * 255.0 for value in _CIFAR10_STD],
        )
    if dataset_type == DatasetType.cifar100:
        return (
            [value * 255.0 for value in _CIFAR100_MEAN],
            [value * 255.0 for value in _CIFAR100_STD],
        )
    raise ValueError(f"unsupported DALI dataset type: {dataset_type.name}")


class _DaliPipelineRunner:
    """One DALI graph for a fixed simulator request batch size."""

    def __init__(
        self,
        *,
        dataset_type: DatasetType,
        batch_size: int,
        device_id: int,
        num_threads: int,
        mean: list[float],
        std: list[float],
        seed: int | None,
    ) -> None:
        try:
            from nvidia.dali import fn, types
            from nvidia.dali.pipeline import pipeline_def
            from nvidia.dali.plugin.pytorch import feed_ndarray
        except ImportError as exc:
            raise ImportError(
                "DALI simulator loading requires NVIDIA DALI; install DALI or "
                "rerun without --dali"
            ) from exc

        self.batch_size = int(batch_size)
        self.device_id = int(device_id)
        self.feed_ndarray = feed_ndarray
        self.output_channels = 1 if dataset_type == DatasetType.mnist else 3
        self.output_size = 28 if dataset_type == DatasetType.mnist else 32

        dali_mean = list(mean)
        dali_std = list(std)
        dali_seed = None if seed is None else int(seed)

        def random_seed(offset: int) -> int | None:
            return None if dali_seed is None else dali_seed + offset

        def reflect_pad_cifar(images):
            # torchvision RandomCrop(..., padding_mode="reflect") excludes the
            # edge pixel.  Recreate its four-pixel border entirely with DALI
            # GPU slice/flip/cat operators.
            left = fn.slice(
                images,
                start=[0, 1, 0],
                shape=[32, 4, 3],
                axes=[0, 1, 2],
                device="gpu",
            )
            left = fn.flip(left, horizontal=True, device="gpu")
            right = fn.slice(
                images,
                start=[0, 27, 0],
                shape=[32, 4, 3],
                axes=[0, 1, 2],
                device="gpu",
            )
            right = fn.flip(right, horizontal=True, device="gpu")
            wide = fn.cat(left, images, right, axis=1, device="gpu")

            top = fn.slice(
                wide,
                start=[1, 0, 0],
                shape=[4, 40, 3],
                axes=[0, 1, 2],
                device="gpu",
            )
            top = fn.flip(top, horizontal=False, vertical=True, device="gpu")
            bottom = fn.slice(
                wide,
                start=[27, 0, 0],
                shape=[4, 40, 3],
                axes=[0, 1, 2],
                device="gpu",
            )
            bottom = fn.flip(
                bottom,
                horizontal=False,
                vertical=True,
                device="gpu",
            )
            return fn.cat(top, wide, bottom, axis=0, device="gpu")

        def random_fixed_crop(images, *, input_size: int, crop_size: int):
            max_start = input_size - crop_size
            y = fn.random.uniform(
                values=list(range(max_start + 1)),
                dtype=types.INT32,
                seed=random_seed(0),
            )
            x = fn.random.uniform(
                values=list(range(max_start + 1)),
                dtype=types.INT32,
                seed=random_seed(1),
            )
            anchor = fn.stack(y, x)
            shape = types.Constant([crop_size, crop_size], dtype=types.INT32)
            return fn.slice(
                images,
                anchor,
                shape,
                axes=[0, 1],
                normalized_anchor=False,
                normalized_shape=False,
                device="gpu",
            )

        @pipeline_def
        def mnist_pipeline():
            images = fn.external_source(
                name="images",
                device="gpu",
                layout="HWC",
                dtype=types.UINT8,
                ndim=3,
                no_copy=True,
            )
            angle = fn.random.uniform(
                range=[-5.0, 5.0],
                seed=random_seed(2),
            )
            images = fn.rotate(
                images,
                angle=angle,
                device="gpu",
                keep_size=True,
                fill_value=0,
                # torchvision RandomRotation defaults to nearest-neighbor
                # interpolation for this MNIST setup.
                interp_type=types.INTERP_NN,
            )
            # A 28x28 slice whose anchor ranges from -2 through 2 is exactly a
            # random 28x28 crop from a two-pixel zero-padded image.
            y = fn.random.uniform(
                values=[-2, -1, 0, 1, 2],
                dtype=types.INT32,
                seed=random_seed(3),
            )
            x = fn.random.uniform(
                values=[-2, -1, 0, 1, 2],
                dtype=types.INT32,
                seed=random_seed(4),
            )
            anchor = fn.stack(y, x)
            shape = types.Constant([28, 28], dtype=types.INT32)
            images = fn.slice(
                images,
                anchor,
                shape,
                axes=[0, 1],
                normalized_anchor=False,
                normalized_shape=False,
                out_of_bounds_policy="pad",
                fill_values=[0.0],
                device="gpu",
            )
            return fn.crop_mirror_normalize(
                images,
                device="gpu",
                dtype=types.FLOAT,
                output_layout="CHW",
                mean=dali_mean,
                std=dali_std,
            )

        @pipeline_def
        def cifar_pipeline():
            images = fn.external_source(
                name="images",
                device="gpu",
                layout="HWC",
                dtype=types.UINT8,
                ndim=3,
                no_copy=True,
            )
            mirror = fn.random.coin_flip(
                probability=0.5,
                seed=random_seed(2),
            )
            images = fn.flip(images, horizontal=mirror, device="gpu")
            images = reflect_pad_cifar(images)
            images = random_fixed_crop(images, input_size=40, crop_size=32)
            return fn.crop_mirror_normalize(
                images,
                device="gpu",
                dtype=types.FLOAT,
                output_layout="CHW",
                mean=dali_mean,
                std=dali_std,
            )

        pipeline_factory = (
            mnist_pipeline if dataset_type == DatasetType.mnist else cifar_pipeline
        )
        self.pipeline = pipeline_factory(
            batch_size=self.batch_size,
            num_threads=max(1, int(num_threads)),
            device_id=self.device_id,
            seed=dali_seed,
            exec_dynamic=True,
            prefetch_queue_depth=1,
        )
        self.pipeline.build()

    def run(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[0] != self.batch_size:
            raise ValueError(
                f"DALI pipeline expects batch size {self.batch_size}, "
                f"received {images.shape[0]}"
            )
        stream = torch.cuda.current_stream(self.device_id)
        self.pipeline.feed_input(
            "images",
            images,
            layout="HWC",
            cuda_stream=stream,
            use_copy_kernel=False,
        )
        dali_output = self.pipeline.run()[0]
        output = torch.empty(
            (
                self.batch_size,
                self.output_channels,
                self.output_size,
                self.output_size,
            ),
            dtype=torch.float32,
            device=images.device,
        )
        self.feed_ndarray(dali_output.as_tensor(), output, cuda_stream=stream)
        return output


class DaliSharedDataLoader:
    """Route simulator batch plans through GPU-resident DALI pipelines."""

    def __init__(
        self,
        dataset: Any,
        ml_setup: Any,
        *,
        device_id: int = 0,
        num_threads: int = 1,
        seed: int = -1,
    ) -> None:
        ensure_supported_dali_workload(ml_setup)
        try:
            import nvidia.dali  # noqa: F401
            import nvidia.dali.plugin.pytorch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "DALI simulator loading requires NVIDIA DALI; install DALI or "
                "rerun without --dali"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("DALI simulator loading requires CUDA")
        if getattr(ml_setup, "collate_fn", None) is not None:
            raise ValueError(
                "DALI simulator loading does not support a custom training "
                "collate function for these workloads"
            )

        self.dataset_type: DatasetType = ml_setup.dataset_type
        self.device_id = int(device_id)
        if self.device_id < 0 or self.device_id >= torch.cuda.device_count():
            raise ValueError(
                f"invalid DALI CUDA device id {self.device_id}; "
                f"available device count is {torch.cuda.device_count()}"
            )
        self.device = torch.device("cuda", self.device_id)
        self.num_threads = max(1, int(num_threads))
        configured_seed = int(seed)
        self.seed: int | None = configured_seed if configured_seed >= 0 else None
        self._plan: tuple[BatchRequest, ...] = ()
        self._pipelines: dict[int, _DaliPipelineRunner] = {}

        raw_images_cpu = _as_raw_image_tensor(dataset, self.dataset_type)
        raw_targets_cpu = _as_target_tensor(dataset, raw_images_cpu.shape[0])
        mean, std = _normalization_parameters(
            self.dataset_type,
            raw_images_cpu,
        )
        self.mean = mean
        self.std = std

        try:
            with torch.cuda.device(self.device):
                self.raw_images = raw_images_cpu.to(
                    self.device,
                    non_blocking=False,
                ).contiguous()
                self.raw_targets = raw_targets_cpu.to(
                    self.device,
                    non_blocking=False,
                ).contiguous()
        except torch.OutOfMemoryError as exc:
            required_mib = (
                raw_images_cpu.numel() * raw_images_cpu.element_size()
                + raw_targets_cpu.numel() * raw_targets_cpu.element_size()
            ) / (1024 * 1024)
            raise RuntimeError(
                f"could not cache the raw {self.dataset_type.name} training "
                f"dataset on {self.device}; raw storage requires about "
                f"{required_mib:.1f} MiB"
            ) from exc

        cached_mib = (
            self.raw_images.numel() * self.raw_images.element_size()
            + self.raw_targets.numel() * self.raw_targets.element_size()
        ) / (1024 * 1024)
        LOGGER.info(
            "cached %s raw training samples on %s for DALI (%.1f MiB)",
            self.raw_images.shape[0],
            self.device,
            cached_mib,
        )

    def _get_pipeline(self, batch_size: int) -> _DaliPipelineRunner:
        pipeline = self._pipelines.get(batch_size)
        if pipeline is None:
            pipeline_seed = (
                self.seed + batch_size * 17
                if self.seed is not None
                else None
            )
            pipeline = _DaliPipelineRunner(
                dataset_type=self.dataset_type,
                batch_size=batch_size,
                device_id=self.device_id,
                num_threads=self.num_threads,
                mean=self.mean,
                std=self.std,
                seed=pipeline_seed,
            )
            self._pipelines[batch_size] = pipeline
        return pipeline

    def set_plan(self, requests: Iterable[BatchRequest]) -> None:
        plan = tuple(requests)
        dataset_size = int(self.raw_images.shape[0])
        for request in plan:
            if not request.dataset_indices:
                raise ValueError("DALI shared loader requests must not be empty")
            if any(index < 0 or index >= dataset_size for index in request.dataset_indices):
                raise IndexError(
                    f"DALI request for node {request.node_name} contains an "
                    "out-of-range dataset index"
                )
        self._plan = plan

        # Building is expensive, so cache one graph for each batch size that
        # the simulation actually requests. This also supports nodes changing
        # batch size while the simulation is running.
        for batch_size in sorted({len(request.dataset_indices) for request in plan}):
            self._get_pipeline(batch_size)

    def __iter__(self) -> Iterator[RoutedBatch]:
        with torch.cuda.device(self.device):
            for request in self._plan:
                indices = torch.tensor(
                    request.dataset_indices,
                    dtype=torch.int64,
                    device=self.device,
                )
                images = self.raw_images.index_select(0, indices).contiguous()
                targets = self.raw_targets.index_select(0, indices).contiguous()
                augmented = self._get_pipeline(len(request.dataset_indices)).run(images)
                yield RoutedBatch(
                    request.node_name,
                    request.batch_index,
                    (augmented, targets),
                )

    def __len__(self) -> int:
        return len(self._plan)

    def close(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self._pipelines.clear()

    def __enter__(self) -> "DaliSharedDataLoader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
