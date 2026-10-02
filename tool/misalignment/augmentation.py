"""Independent augmentation presets for symmetric misalignment experiments.

Both random halves must receive the same transform distribution.  This
factory returns one transform that can be assigned to every source dataset
before they are concatenated and repartitioned.
"""

from __future__ import annotations

import math
from typing import Any

from torchvision import transforms

from py_src.ml_setup_dataset import get_imagenet_preprocessing

_SUPPORTED_LEVELS = {
    "mnist": (0, 1, 2, 3),
    "cifar10": (0, 1, 2, 3),
    "cifar100": (0, 1, 2, 3),
    "imagenet1k": (0, 1, 2),
    "modular": (0,),
}

_CIFAR_STATS = {
    "cifar10": (
        (0.49139968, 0.48215841, 0.44653091),
        (0.24703223, 0.24348513, 0.26158784),
    ),
    "cifar100": (
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761),
    ),
}


def parse_augmentation_level(level: str | int) -> int:
    """Normalize ``none`` or a non-negative integer augmentation level."""
    normalized = str(level).strip().lower()
    if normalized in {"none", "off", "false"}:
        return 0
    try:
        result = int(normalized)
    except ValueError as exc:
        raise ValueError("augmentation must be 'none' or a non-negative integer") from exc
    if result < 0:
        raise ValueError("augmentation must be non-negative")
    return result


def _validate(dataset_name: str, level: str | int, input_size: int | None) -> tuple[str, int]:
    dataset_name = str(dataset_name).lower()
    if dataset_name not in _SUPPORTED_LEVELS:
        raise ValueError(f"unsupported augmentation dataset {dataset_name!r}")
    parsed_level = parse_augmentation_level(level)
    if parsed_level not in _SUPPORTED_LEVELS[dataset_name]:
        supported = ["none/0", *[str(item) for item in _SUPPORTED_LEVELS[dataset_name] if item > 0]]
        raise ValueError(f"{dataset_name} supports augmentation levels {', '.join(supported)}")
    if input_size is not None:
        if dataset_name != "imagenet1k":
            raise ValueError("input_size is supported only for ImageNet augmentation")
        if not isinstance(input_size, int) or isinstance(input_size, bool) or input_size <= 0:
            raise ValueError("input_size must be a positive integer")
    return dataset_name, parsed_level


def describe_augmentation(
    dataset_name: str,
    level: str | int,
    *,
    input_size: int | None = None,
) -> dict[str, Any]:
    """Return JSON-friendly preset metadata for logs and result summaries."""
    dataset_name, level = _validate(dataset_name, level, input_size)
    operations: list[str] = []
    if dataset_name == "modular":
        image_size = None
    elif dataset_name == "mnist":
        image_size = 28
        if level >= 1:
            operations.extend(["RandomRotation(degrees=5)", "RandomCrop(size=28, padding=2)"])
        if level >= 2:
            operations.append("RandomAffine(degrees=10, translate=0.1, scale=0.9..1.1)")
        operations.extend(["ToTensor", "Normalize(original MNIST training statistics)"])
        if level >= 3:
            operations.append("RandomErasing(p=0.1)")
    elif dataset_name in _CIFAR_STATS:
        image_size = 32
        if level >= 1:
            operations.extend(["RandomHorizontalFlip(p=0.5)", "RandomCrop(size=32, padding=4, reflect)"])
        if level >= 2:
            operations.append("ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)")
        operations.extend(["ToTensor", "Normalize(dataset constants)"])
        if level >= 3:
            operations.append("RandomErasing(p=0.1)")
    else:
        image_size = input_size or (176 if level == 2 else 224)
        if level == 0:
            resize_size = 256 if input_size is None else math.ceil(256 * input_size / 224)
            operations.extend([f"Resize(size={resize_size})", f"CenterCrop(size={image_size})"])
        else:
            operations.extend([f"RandomResizedCrop(size={image_size})", "RandomHorizontalFlip(p=0.5)"])
            if level == 2:
                operations.extend(["TrivialAugmentWide", "RandAugment(num_ops=2, magnitude=9)"])
        operations.append("ToTensor")
        if level == 2:
            operations.append("RandomErasing(p=0.1)")
        operations.append("Normalize(ImageNet constants)")
    return {
        "dataset": dataset_name,
        "level": level,
        "name": "none" if level == 0 else str(level),
        "stochastic": level > 0,
        "input_size": image_size,
        "operations": operations,
        "shared_between_partitions": True,
    }


def build_augmentation(
    ml_setup,
    dataset_name: str,
    level: str | int,
    *,
    input_size: int | None = None,
):
    """Build a transform for MNIST, CIFAR, ImageNet, or ``None`` for modular.

    Level zero removes stochastic augmentation while retaining conversion,
    normalization, and deterministic ImageNet resizing.  Higher levels are
    cumulative for MNIST/CIFAR.  ImageNet levels one and two reuse the
    project's existing preprocessing recipes without adding batch Mixup or
    CutMix.  ``input_size`` optionally overrides only the ImageNet crop size.
    """
    dataset_name, level = _validate(dataset_name, level, input_size)
    if dataset_name == "modular":
        return None

    if dataset_name == "mnist":
        raw_data = ml_setup.training_data.data
        mean = float(raw_data.float().mean().item() / 255.0)
        std = float(raw_data.float().std().item() / 255.0)
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError("MNIST training data must have finite non-zero standard deviation")
        operations: list[Any] = []
        if level >= 1:
            operations.extend([transforms.RandomRotation(5, fill=0), transforms.RandomCrop(28, padding=2)])
        if level >= 2:
            operations.append(
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0)
            )
        operations.extend([transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])])
        if level >= 3:
            operations.append(transforms.RandomErasing(p=0.1))
        return transforms.Compose(operations)

    if dataset_name in _CIFAR_STATS:
        mean, std = _CIFAR_STATS[dataset_name]
        operations = []
        if level >= 1:
            operations.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            ])
        if level >= 2:
            operations.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
        operations.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        if level >= 3:
            operations.append(transforms.RandomErasing(p=0.1))
        return transforms.Compose(operations)

    if level == 0:
        resize_size = 256 if input_size is None else math.ceil(256 * input_size / 224)
        _, transform = get_imagenet_preprocessing(
            version=1,
            augmentation=False,
            val_resize_size=resize_size,
            val_crop_size=input_size,
        )
    else:
        transform, _ = get_imagenet_preprocessing(
            version=level,
            augmentation=True,
            train_crop_size=input_size,
        )
    return transform


__all__ = ["build_augmentation", "describe_augmentation", "parse_augmentation_level"]
