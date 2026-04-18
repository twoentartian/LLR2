"""ImageNet dataset setups for LLR2.

Two preprocessing presets (matching DFL_torch):
  preset_version=1 — RandomResizedCrop + RandomHorizontalFlip  (basic)
  preset_version=2 — + TrivialAugmentWide + RandAugment + RandomErasing  (PyTorch Recipe v2)

Path resolution is handled by dataset_default.py; override via a
``dataset_env.py`` file placed in the same directory.
"""

from __future__ import annotations

import os
from typing import Optional

from torchvision import transforms, datasets
from torchvision.transforms.autoaugment import TrivialAugmentWide
from torchvision.transforms.v2 import RandAugment

from .dataset_types import DatasetSetup, DatasetType
from .dataset_default import imagenet1k_path, imagenet100_path, imagenet10_path
from .dataset_masked import MaskedImageDataset

# Standard ImageNet normalisation constants
_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


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
) -> DatasetSetup:
    """Full ImageNet-1k dataset (torchvision ``ImageNet`` loader).

    Expects the standard ``train/`` and ``val/`` directory layout at
    ``~/dataset/imagenet1k`` (overridable via ``dataset_env.py``).
    """
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
) -> DatasetSetup:
    """100-class ImageNet subset (``ImageFolder`` layout).

    Expects ``train/`` and ``val/`` under ``~/dataset/imagenet100``.
    """
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
) -> DatasetSetup:
    """10-class ImageNet subset (``ImageFolder`` layout).

    Expects ``train/`` and ``val/`` under ``~/dataset/imagenet10``.
    """
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
    from .dataset_imagenet_raw_pytorch import ClassificationPresetTrain
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
