from __future__ import annotations

from collections.abc import Callable
from typing import Any, List, Optional

from torchvision import datasets, transforms

from .dataset_default import flowers102_path
from .dataset_types import DatasetSetup, DatasetType

_type_mean_std = Optional[tuple[tuple[float, float, float], tuple[float, float, float]]]
_type_transform = Callable[[Any], Any]


def dataset_flowers102(
    image_size: int = 128,
    transforms_training: Optional[List[_type_transform]] = None,
    transforms_testing: Optional[List[_type_transform]] = None,
    mean_std: _type_mean_std = None,
    augmentation: bool = True,
    override_dataset_path: Optional[str] = None,
    *args,
    **kwargs,
) -> DatasetSetup:
    dataset_path = flowers102_path if override_dataset_path is None else override_dataset_path
    dataset_type = DatasetType.flowers102
    stats = mean_std or ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if augmentation:
        data_augmentation: list[_type_transform] = [transforms.RandomHorizontalFlip(p=0.5)]
    else:
        data_augmentation = []

    resize_and_crop: list[_type_transform] = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]
    train_transforms = resize_and_crop + data_augmentation + [
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
    test_transforms = resize_and_crop + [
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]

    if transforms_training is None:
        train_data = datasets.Flowers102(
            root=dataset_path,
            split="train",
            download=True,
            transform=transforms.Compose(train_transforms),
        )
    else:
        train_data = datasets.Flowers102(
            root=dataset_path,
            split="train",
            download=True,
            transform=transforms.Compose(transforms_training),
        )

    if transforms_testing is None:
        validation_data = datasets.Flowers102(
            root=dataset_path,
            split="val",
            download=True,
            transform=transforms.Compose(test_transforms),
        )
    else:
        validation_data = datasets.Flowers102(
            root=dataset_path,
            split="val",
            download=True,
            transform=transforms.Compose(transforms_testing),
        )

    return DatasetSetup(dataset_type, train_data, validation_data)
