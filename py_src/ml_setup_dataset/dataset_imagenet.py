"""ImageNet dataset setups for LLR2.

Two preprocessing presets (matching DFL_torch):
  preset_version=1 — RandomResizedCrop + RandomHorizontalFlip  (basic)
  preset_version=2 — + TrivialAugmentWide + RandAugment + RandomErasing  (PyTorch Recipe v2)

Path resolution is handled by dataset_default.py; override via a
``dataset_env.py`` file placed in the same directory.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torchvision import transforms, datasets
from torchvision.transforms.autoaugment import TrivialAugmentWide
from torchvision.transforms.v2 import RandAugment
from torch.utils.data import Dataset

from .dataset_types import DatasetSetup, DatasetType
from .dataset_default import imagenet1k_path, imagenet100_path, imagenet10_path
from .dataset_masked import MaskedImageDataset

# Standard ImageNet normalisation constants
_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_DALI_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
_DALI_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]
_DALI_RANDOM_ERASE_FILL = [
    -(0.485 / 0.229),
    -(0.456 / 0.224),
    -(0.406 / 0.225),
]


class _DaliImageNetIterator:
    def __init__(
        self,
        dali_iterator,
        sample_count: int,
        batch_size: int,
        drop_last: bool,
        repetitions: int = 1,
        batch_transform: Optional[Callable] = None,
    ):
        self._dali_iterator = dali_iterator
        self._sample_count = sample_count
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._repetitions = repetitions
        self._batch_transform = batch_transform

    def _finalize_batch(self, data, label):
        batch = (data, label)
        if self._batch_transform is not None:
            batch = self._batch_transform(batch)
        return batch

    def __iter__(self):
        consumed = 0
        pending_data = None
        pending_label = None

        try:
            for repetition in range(self._repetitions):
                for batch in self._dali_iterator:
                    if isinstance(batch, list):
                        batch = batch[0]
                    data = batch["data"]
                    label = batch["label"].squeeze().long()

                    remaining = self._sample_count - consumed
                    if remaining <= 0:
                        break
                    if data.size(0) > remaining:
                        data = data[:remaining]
                        label = label[:remaining]
                    consumed += data.size(0)

                    if pending_data is not None:
                        data = torch.cat((pending_data, data), dim=0)
                        label = torch.cat((pending_label, label), dim=0)
                        pending_data = None
                        pending_label = None

                    while data.size(0) >= self._batch_size:
                        current_data = data[:self._batch_size]
                        current_label = label[:self._batch_size]
                        yield self._finalize_batch(current_data, current_label)
                        data = data[self._batch_size:]
                        label = label[self._batch_size:]

                    if data.size(0) > 0:
                        pending_data = data
                        pending_label = label

                    if consumed >= self._sample_count:
                        break

                if consumed >= self._sample_count:
                    break
                if repetition + 1 < self._repetitions:
                    self._dali_iterator.reset()

            if pending_data is not None and not self._drop_last:
                yield self._finalize_batch(pending_data, pending_label)
        finally:
            self._dali_iterator.reset()

    def __len__(self) -> int:
        if self._drop_last:
            return self._sample_count // self._batch_size
        return math.ceil(self._sample_count / self._batch_size)


@dataclass
class DaliImageNetDataset(Dataset):
    file_root: str
    dataset_type: DatasetType
    split: str
    preset_version: int = 2
    train_crop_size: Optional[int] = None
    val_resize_size: Optional[int] = None
    val_crop_size: Optional[int] = None
    augmentation: bool = True
    device_id: int = 0
    auto_augment_policy: Optional[str] = None
    auto_augment_interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR
    random_erase_prob: float = 0.0
    repeated_augmentation_repetitions: int = 1

    def __post_init__(self) -> None:
        folder_dataset = datasets.ImageFolder(self.file_root)
        self.base_sample_count = len(folder_dataset)
        if self.split == "train" and self.augmentation:
            self.sample_count = self.base_sample_count * max(1, self.repeated_augmentation_repetitions)
        else:
            self.sample_count = self.base_sample_count
        self.classes = folder_dataset.classes

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index):
        raise TypeError("DaliImageNetDataset must be consumed through build_dataloader()")

    def build_dataloader(self, default_batch_size: int, config, is_train: bool):
        try:
            from nvidia.dali import fn, types
            from nvidia.dali import math as dali_math
            from nvidia.dali.pipeline import pipeline_def
            from nvidia.dali.plugin.pytorch import DALIGenericIterator
            from nvidia.dali.plugin.base_iterator import LastBatchPolicy
        except ImportError as exc:
            raise ImportError(
                "DALI support requires NVIDIA DALI. Install it from https://github.com/NVIDIA/DALI "
                "or rerun without --dali."
            ) from exc

        from nvidia.dali.auto_aug import auto_augment as dali_auto_augment
        from nvidia.dali.auto_aug import rand_augment as dali_rand_augment
        from nvidia.dali.auto_aug import trivial_augment as dali_trivial_augment

        dali_rgb = getattr(types, "RGB")
        dali_interp_linear = getattr(types, "INTERP_LINEAR")
        dali_interp_nearest = getattr(types, "INTERP_NN")
        dali_float = getattr(types, "FLOAT")

        batch_size = config.batch_size or default_batch_size
        num_threads = max(1, config.num_workers or 4)
        drop_last = config.drop_last
        shuffle = config.shuffle if config.shuffle is not None else is_train
        sample_count = min(len(self), config.num_samples) if config.num_samples is not None else len(self)
        repetitions = self.repeated_augmentation_repetitions if is_train and self.augmentation else 1
        use_mixup_cutmix = is_train and config.collate_fn is not None
        auto_augment_policy = _to_dali_auto_augment_policy(self.auto_augment_policy)
        auto_augment_fill_value = _to_dali_fill_value(self.auto_augment_policy)
        auto_augment_interp = _to_dali_interp_type(self.auto_augment_interpolation, types)
        use_preset_v2_augmentations = self.preset_version == 2 and auto_augment_policy is None
        preset_v2_fill_value = 0
        preset_v2_interp = dali_interp_nearest

        train_crop, val_resize, val_crop = _resolve_imagenet_sizes(
            self.preset_version,
            self.train_crop_size,
            self.val_resize_size,
            self.val_crop_size,
        )
        use_train_pipeline = is_train and self.split == "train" and self.augmentation

        def apply_random_erasing(images):
            if self.random_erase_prob <= 0.0:
                return images

            should_erase = fn.random.coin_flip(probability=self.random_erase_prob)
            if should_erase:
                erase_area_fraction = fn.random.uniform(range=[0.02, 0.33])
                erase_ratio = dali_math.exp(fn.random.uniform(range=[math.log(0.3), math.log(3.3)]))
                erase_height = dali_math.clamp(dali_math.sqrt(erase_area_fraction * erase_ratio), 0.0, 1.0)
                erase_width = dali_math.clamp(dali_math.sqrt(erase_area_fraction / erase_ratio), 0.0, 1.0)
                max_anchor_y = dali_math.max(1.0 - erase_height, 0.0)
                max_anchor_x = dali_math.max(1.0 - erase_width, 0.0)
                erase_anchor_y = fn.random.uniform(range=[0.0, 1.0]) * max_anchor_y
                erase_anchor_x = fn.random.uniform(range=[0.0, 1.0]) * max_anchor_x
                erase_anchor = fn.stack(erase_anchor_y, erase_anchor_x, axis=0)
                erase_shape = fn.stack(erase_height, erase_width, axis=0)
                images = fn.erase(
                    images,
                    anchor=erase_anchor,
                    shape=erase_shape,
                    axis_names="HW",
                    normalized_anchor=True,
                    normalized_shape=True,
                    fill_value=_DALI_RANDOM_ERASE_FILL,
                )
            return images

        @pipeline_def(enable_conditionals=True)
        def train_pipeline(file_root: str, crop_size: int, random_shuffle: bool):
            images, labels = fn.readers.file(
                file_root=file_root,
                random_shuffle=random_shuffle,
                name="Reader",
            )
            images = fn.decoders.image_random_crop(
                images,
                device="cpu",
                output_type=dali_rgb,
                random_area=[0.08, 1.0],
                random_aspect_ratio=[3.0 / 4.0, 4.0 / 3.0],
                num_attempts=100,
            )
            images = images.gpu()
            images = fn.resize(
                images,
                device="gpu",
                resize_x=crop_size,
                resize_y=crop_size,
                interp_type=dali_interp_linear,
            )
            mirror = fn.random.coin_flip(probability=0.5)
            images = fn.flip(images, horizontal=mirror)
            if auto_augment_policy is not None:
                images = dali_auto_augment.auto_augment(
                    images,
                    policy_name=auto_augment_policy,
                    shape=(crop_size, crop_size),
                    fill_value=auto_augment_fill_value,
                    interp_type=auto_augment_interp,
                )
            elif use_preset_v2_augmentations:
                images = dali_trivial_augment.trivial_augment_wide(
                    images,
                    shape=(crop_size, crop_size),
                    fill_value=preset_v2_fill_value,
                    interp_type=preset_v2_interp,
                )
                images = dali_rand_augment.rand_augment(
                    images,
                    n=2,
                    m=9,
                    shape=(crop_size, crop_size),
                    fill_value=preset_v2_fill_value,
                    interp_type=preset_v2_interp,
                )
            images = fn.crop_mirror_normalize(  # type: ignore[call-overload]
                images, # type: ignore
                dtype=dali_float,
                output_layout="CHW",
                mean=_DALI_MEAN,
                std=_DALI_STD,
            )
            images = apply_random_erasing(images)
            return images, labels

        @pipeline_def
        def eval_pipeline(file_root: str, resize_size: int, crop_size: int):
            images, labels = fn.readers.file(
                file_root=file_root,
                random_shuffle=False,
                name="Reader",
            )
            images = fn.decoders.image(images, device="cpu", output_type=dali_rgb)
            images = images.gpu()
            images = fn.resize(
                images,
                device="gpu",
                resize_shorter=resize_size,
                interp_type=dali_interp_linear,
            )
            images = fn.crop_mirror_normalize(  # type: ignore[call-overload]
                images, # type: ignore
                dtype=dali_float,
                output_layout="CHW",
                crop=(crop_size, crop_size),
                mean=_DALI_MEAN,
                std=_DALI_STD,
            )
            return images, labels

        if use_train_pipeline:
            pipeline = train_pipeline(
                batch_size=batch_size,
                num_threads=num_threads,
                device_id=self.device_id,
                file_root=self.file_root,
                crop_size=train_crop,
                random_shuffle=shuffle,
            )
        else:
            pipeline = eval_pipeline(
                batch_size=batch_size,
                num_threads=num_threads,
                device_id=self.device_id,
                file_root=self.file_root,
                resize_size=val_resize,
                crop_size=val_crop,
            )

        dali_iterator = DALIGenericIterator(
            [pipeline],
            output_map=["data", "label"],
            reader_name="Reader",
            auto_reset=False,
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )
        return _DaliImageNetIterator(
            dali_iterator,
            sample_count,
            batch_size,
            drop_last,
            repetitions=repetitions,
            batch_transform=config.collate_fn if use_mixup_cutmix else None,
        )


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def get_imagenet_preprocessing(
    version: int = 2,
    train_crop_size: Optional[int] = None,
    val_resize_size: Optional[int] = None,
    val_crop_size: Optional[int] = None,
    random_erasing: Optional[float] = None,
    augmentation: bool = True,
):
    """Return ``(train_transform, val_transform)`` for ImageNet.

    Parameters
    ----------
    version:
        1 — basic recipe; 2 — PyTorch Recipe v2 (TrivialAugmentWide + RandAugment).
    train_crop_size / val_resize_size / val_crop_size:
        Override default crop/resize sizes.
    random_erasing:
        Random erasing probability; ``None`` uses the version default
        (0.0 for v1, 0.1 for v2).
    augmentation:
        Set to ``False`` to disable all training augmentation (evaluation mode).
    """
    if not augmentation:
        crop = train_crop_size or 224
        vrs = val_resize_size or 256
        vcs = val_crop_size or 224
        train_tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            _NORMALIZE,
        ])
        val_tfm = transforms.Compose([
            transforms.Resize(vrs),
            transforms.CenterCrop(vcs),
            transforms.ToTensor(),
            _NORMALIZE,
        ])
        return train_tfm, val_tfm

    if version == 1:
        crop = train_crop_size or 224
        vrs = val_resize_size or 256
        vcs = val_crop_size or 224
        train_list: list = [
            transforms.RandomResizedCrop(crop, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            _NORMALIZE,
        ]
        if random_erasing is not None and random_erasing > 0:
            train_list.append(transforms.RandomErasing(p=random_erasing))
        train_tfm = transforms.Compose(train_list)
        val_tfm = transforms.Compose([
            transforms.Resize(vrs),
            transforms.CenterCrop(vcs),
            transforms.ToTensor(),
            _NORMALIZE,
        ])
        return train_tfm, val_tfm

    elif version == 2:
        crop = train_crop_size or 176
        vrs = val_resize_size or 232
        vcs = val_crop_size or 224
        erasing_p = 0.1 if random_erasing is None else random_erasing
        train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(crop),
            transforms.RandomHorizontalFlip(),
            TrivialAugmentWide(),
            RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.RandomErasing(p=erasing_p),
            _NORMALIZE,
        ])
        val_tfm = transforms.Compose([
            transforms.Resize(vrs),
            transforms.CenterCrop(vcs),
            transforms.ToTensor(),
            _NORMALIZE,
        ])
        return train_tfm, val_tfm

    else:
        raise ValueError(f"unknown preset_version {version!r}, expected 1 or 2")


def _resolve_imagenet_sizes(
    version: int,
    train_crop_size: Optional[int],
    val_resize_size: Optional[int],
    val_crop_size: Optional[int],
) -> tuple[int, int, int]:
    if version == 1:
        return train_crop_size or 224, val_resize_size or 256, val_crop_size or 224
    if version == 2:
        return train_crop_size or 176, val_resize_size or 232, val_crop_size or 224
    raise ValueError(f"unknown preset_version {version!r}, expected 1 or 2")


def _to_dali_auto_augment_policy(policy: Optional[str]) -> Optional[str]:
    if policy is None:
        return None
    if policy == "imagenet":
        return "image_net"
    raise ValueError(f"unsupported DALI auto augment policy: {policy!r}")


def _to_dali_fill_value(policy: Optional[str]) -> int:
    if policy is None:
        return 0
    if policy == "imagenet":
        return 0
    raise ValueError(f"unsupported DALI auto augment fill policy: {policy!r}")


def _to_dali_interp_type(interpolation: transforms.InterpolationMode, types_module):
    if interpolation == transforms.InterpolationMode.BILINEAR:
        return getattr(types_module, "INTERP_LINEAR")
    if interpolation == transforms.InterpolationMode.NEAREST:
        return getattr(types_module, "INTERP_NN")
    raise ValueError(f"unsupported DALI interpolation mode: {interpolation!r}")


def _default_dali_random_erase_prob(version: int, random_erasing: Optional[float]) -> float:
    if random_erasing is not None:
        return random_erasing
    if version == 2:
        return 0.1
    return 0.0


def _default_dali_repetitions(version: int) -> int:
    if version == 2:
        return 4
    return 1


def _make_dali_imagenet_setup(
    dataset_type: DatasetType,
    dataset_root,
    preset_version: int,
    train_crop_size: Optional[int],
    val_resize_size: Optional[int],
    val_crop_size: Optional[int],
    augmentation: bool,
    dali_device_id: int,
    auto_augment_policy: Optional[str] = None,
    auto_augment_interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
    random_erase_prob: float = 0.0,
    repeated_augmentation_repetitions: Optional[int] = None,
) -> DatasetSetup:
    root = str(dataset_root)
    if repeated_augmentation_repetitions is None:
        repeated_augmentation_repetitions = _default_dali_repetitions(preset_version)
    train_data = DaliImageNetDataset(
        file_root=os.path.join(root, "train"),
        dataset_type=dataset_type,
        split="train",
        preset_version=preset_version,
        train_crop_size=train_crop_size,
        val_resize_size=val_resize_size,
        val_crop_size=val_crop_size,
        augmentation=augmentation,
        device_id=dali_device_id,
        auto_augment_policy=auto_augment_policy,
        auto_augment_interpolation=auto_augment_interpolation,
        random_erase_prob=random_erase_prob,
        repeated_augmentation_repetitions=repeated_augmentation_repetitions,
    )
    val_data = DaliImageNetDataset(
        file_root=os.path.join(root, "val"),
        dataset_type=dataset_type,
        split="val",
        preset_version=preset_version,
        train_crop_size=train_crop_size,
        val_resize_size=val_resize_size,
        val_crop_size=val_crop_size,
        augmentation=False,
        device_id=dali_device_id,
        auto_augment_policy=None,
        auto_augment_interpolation=auto_augment_interpolation,
        random_erase_prob=0.0,
        repeated_augmentation_repetitions=1,
    )
    return DatasetSetup(dataset_type, train_data, val_data)


# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------

def dataset_imagenet1k(
    preset_version: int = 2,
    transforms_training=None,
    transforms_testing=None,
    train_crop_size: Optional[int] = None,
    val_resize_size: Optional[int] = None,
    val_crop_size: Optional[int] = None,
    random_erasing: Optional[float] = None,
    augmentation: bool = True,
    use_dali: bool = False,
    dali_device_id: int = 0,
) -> DatasetSetup:
    """Full ImageNet-1k dataset (torchvision ``ImageNet`` loader).

    Expects the standard ``train/`` and ``val/`` directory layout at
    ``~/dataset/imagenet1k`` (overridable via ``dataset_env.py``).
    """
    if use_dali:
        if transforms_training is not None or transforms_testing is not None:
            raise ValueError("DALI ImageNet setup does not accept torchvision transforms")
        return _make_dali_imagenet_setup(
            DatasetType.imagenet1k,
            imagenet1k_path,
            preset_version,
            train_crop_size,
            val_resize_size,
            val_crop_size,
            augmentation,
            dali_device_id,
            random_erase_prob=_default_dali_random_erase_prob(preset_version, random_erasing),
        )

    if transforms_training is None and transforms_testing is None:
        transforms_training, transforms_testing = get_imagenet_preprocessing(
            preset_version, train_crop_size, val_resize_size, val_crop_size,
            random_erasing, augmentation,
        )
    train_data = datasets.ImageNet(root=imagenet1k_path, split="train", transform=transforms_training)
    val_data = datasets.ImageNet(root=imagenet1k_path, split="val", transform=transforms_testing)
    return DatasetSetup(DatasetType.imagenet1k, train_data, val_data)


def dataset_imagenet100(
    preset_version: int = 2,
    transforms_training=None,
    transforms_testing=None,
    train_crop_size: Optional[int] = None,
    val_resize_size: Optional[int] = None,
    val_crop_size: Optional[int] = None,
    random_erasing: Optional[float] = None,
    augmentation: bool = True,
    use_dali: bool = False,
    dali_device_id: int = 0,
) -> DatasetSetup:
    """100-class ImageNet subset (``ImageFolder`` layout).

    Expects ``train/`` and ``val/`` under ``~/dataset/imagenet100``.
    """
    if use_dali:
        if transforms_training is not None or transforms_testing is not None:
            raise ValueError("DALI ImageNet setup does not accept torchvision transforms")
        return _make_dali_imagenet_setup(
            DatasetType.imagenet100,
            imagenet100_path,
            preset_version,
            train_crop_size,
            val_resize_size,
            val_crop_size,
            augmentation,
            dali_device_id,
            random_erase_prob=_default_dali_random_erase_prob(preset_version, random_erasing),
        )

    if transforms_training is None and transforms_testing is None:
        transforms_training, transforms_testing = get_imagenet_preprocessing(
            preset_version, train_crop_size, val_resize_size, val_crop_size,
            random_erasing, augmentation,
        )
    train_data = datasets.ImageFolder(os.path.join(imagenet100_path, "train"), transform=transforms_training)
    val_data = datasets.ImageFolder(os.path.join(imagenet100_path, "val"), transform=transforms_testing)
    return DatasetSetup(DatasetType.imagenet100, train_data, val_data)


def dataset_imagenet10(
    preset_version: int = 2,
    transforms_training=None,
    transforms_testing=None,
    train_crop_size: Optional[int] = None,
    val_resize_size: Optional[int] = None,
    val_crop_size: Optional[int] = None,
    random_erasing: Optional[float] = None,
    augmentation: bool = True,
    use_dali: bool = False,
    dali_device_id: int = 0,
) -> DatasetSetup:
    """10-class ImageNet subset (``ImageFolder`` layout).

    Expects ``train/`` and ``val/`` under ``~/dataset/imagenet10``.
    """
    if use_dali:
        if transforms_training is not None or transforms_testing is not None:
            raise ValueError("DALI ImageNet setup does not accept torchvision transforms")
        return _make_dali_imagenet_setup(
            DatasetType.imagenet10,
            imagenet10_path,
            preset_version,
            train_crop_size,
            val_resize_size,
            val_crop_size,
            augmentation,
            dali_device_id,
            random_erase_prob=_default_dali_random_erase_prob(preset_version, random_erasing),
        )

    if transforms_training is None and transforms_testing is None:
        transforms_training, transforms_testing = get_imagenet_preprocessing(
            preset_version, train_crop_size, val_resize_size, val_crop_size,
            random_erasing, augmentation,
        )
    train_data = datasets.ImageFolder(os.path.join(imagenet10_path, "train"), transform=transforms_training)
    val_data = datasets.ImageFolder(os.path.join(imagenet10_path, "val"), transform=transforms_testing)
    return DatasetSetup(DatasetType.imagenet10, train_data, val_data)

def dataset_imagenet1k_from_pytorch(train_crop_size=224, val_resize_size=256, val_crop_size=224,
                              interpolation=transforms.InterpolationMode.BILINEAR, auto_augment_policy=None,
                              random_erase_prob=0.0, ra_magnitude=9, augmix_severity=3,
                              backend='pil', use_v2=False):
    from py_src.torch_vision_train.presets import ClassificationPresetTrain
    dataset_type = DatasetType.imagenet1k
    dataset_path = f'{imagenet1k_path}/train' if imagenet1k_path is None else f"{imagenet1k_path}/train"
    dataset_train = datasets.ImageFolder(
        dataset_path,
        ClassificationPresetTrain(
            crop_size=train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
            ra_magnitude=ra_magnitude,
            augmix_severity=augmix_severity,
            backend=backend,
            use_v2=use_v2,
        ),
    )
    dataset_path = f'{imagenet1k_path}/val' if imagenet1k_path is None else f"{imagenet1k_path}/val"
    transforms_test = transforms.Compose([
        transforms.Resize(val_resize_size),
        transforms.CenterCrop(val_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_test = datasets.ImageFolder(dataset_path, transforms_test)
    return DatasetSetup(dataset_type, dataset_train, dataset_test)


def dataset_imagenet1k_from_pytorch_dali(
    train_crop_size: int = 224,
    val_resize_size: int = 256,
    val_crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BILINEAR,
    auto_augment_policy: Optional[str] = "imagenet",
    random_erase_prob: float = 0.0,
    ra_repetitions: int = 3,
    dali_device_id: int = 0,
) -> DatasetSetup:
    return _make_dali_imagenet_setup(
        DatasetType.imagenet1k,
        imagenet1k_path,
        preset_version=1,
        train_crop_size=train_crop_size,
        val_resize_size=val_resize_size,
        val_crop_size=val_crop_size,
        augmentation=True,
        dali_device_id=dali_device_id,
        auto_augment_policy=auto_augment_policy,
        auto_augment_interpolation=interpolation,
        random_erase_prob=random_erase_prob,
        repeated_augmentation_repetitions=ra_repetitions,
    )


def dataset_imagenet1k_sam_mask_random_noise(
    train_crop_size: int = 224,
    val_resize_size: int = 256,
    val_crop_size: int = 224,
) -> DatasetSetup:
    """ImageNet-1k with SAM-mask regions replaced by random noise (training only).

    Expects ``train/``, ``val/``, and ``train_sam_mask/`` under the ImageNet-1k root.
    """
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(train_crop_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _NORMALIZE,
    ])
    train_data = MaskedImageDataset(
        image_root=os.path.join(str(imagenet1k_path), "train"),
        mask_root=os.path.join(str(imagenet1k_path), "train_sam_mask"),
        transform=train_tfm,
        unmasked_area_type="random",
        use_imagenet_label=True,
    )
    val_tfm = transforms.Compose([
        transforms.Resize(val_resize_size),
        transforms.CenterCrop(val_crop_size),
        transforms.ToTensor(),
        _NORMALIZE,
    ])
    val_data = datasets.ImageNet(root=imagenet1k_path, split="val", transform=val_tfm)
    return DatasetSetup(DatasetType.imagenet1k_sam_mask_random_noise, train_data, val_data)


def dataset_imagenet1k_sam_mask_black(
    train_crop_size: int = 224,
    val_resize_size: int = 256,
    val_crop_size: int = 224,
) -> DatasetSetup:
    """ImageNet-1k with SAM-mask regions zeroed out (training only).

    Expects ``train/``, ``val/``, and ``train_sam_mask/`` under the ImageNet-1k root.
    """
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(train_crop_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _NORMALIZE,
    ])
    train_data = MaskedImageDataset(
        image_root=os.path.join(str(imagenet1k_path), "train"),
        mask_root=os.path.join(str(imagenet1k_path), "train_sam_mask"),
        transform=train_tfm,
        unmasked_area_type="zero",
        use_imagenet_label=True,
    )
    val_tfm = transforms.Compose([
        transforms.Resize(val_resize_size),
        transforms.CenterCrop(val_crop_size),
        transforms.ToTensor(),
        _NORMALIZE,
    ])
    val_data = datasets.ImageNet(root=imagenet1k_path, split="val", transform=val_tfm)
    return DatasetSetup(DatasetType.imagenet1k_sam_mask_black, train_data, val_data)
