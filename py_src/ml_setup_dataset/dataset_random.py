import numpy as np
import os
import json
from typing import Literal, Optional
from multiprocessing import Pool
from pathlib import Path

from torchvision import transforms, datasets
from PIL import Image

from .dataset_default import (
    default_path_random_cifar10,
    default_path_random_cifar100,
    default_path_random_imagenet10,
    default_path_random_imagenet100,
    default_path_random_imagenet1k,
    default_path_random_mnist,
)
from .dataset_types import DatasetSetup
from .dataset_util import calculate_mean_std
from .dataset_types import DatasetType

def generate_images_for_label(label, num_images, img_size, channels, split: Literal["train", "test", "val"], output_path, reset_random_seed_per_sample, reset_random_seed_per_label):
    current_label_dir = os.path.join(output_path, split, str(label))
    os.makedirs(current_label_dir, exist_ok=True)
    if reset_random_seed_per_label:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(seed)
    for i in range(num_images):
        if reset_random_seed_per_sample:
            random_data = os.urandom(4)
            seed = int.from_bytes(random_data, byteorder="big")
            np.random.seed(seed)
        if channels == 1:
            noise_image = (np.random.rand(*img_size) * 255).astype(np.uint8)
            # noise_image = np.random.randint(0, 256, size=img_size, dtype=np.uint8)  # (H, W)
        else:
            noise_image = (np.random.rand(*img_size, channels) * 255).astype(np.uint8)
            # noise_image = np.random.randint(0, 256, size=(*img_size, channels), dtype=np.uint8)  # (H, W, 3)
        img = Image.fromarray(noise_image)
        img.save(os.path.join(current_label_dir, f'image_{label}_{i}.png'))

def save_random_images(num_images_per_label, dataset_type: DatasetType, output_path,
                       num_images_per_label_test=1, num_workers: Optional[int]=None, reset_random_seeds_per_label=False, reset_random_seeds_per_sample=False):
    if num_workers is not None and num_workers <= 0:
        num_workers = None

    match dataset_type.value:
        case DatasetType.random_mnist.value:
            num_classes = 10
            img_size = (28, 28)
            channels = 1
            default_worker_count = 2
        case DatasetType.random_cifar10.value:
            num_classes = 10
            img_size = (32, 32)
            channels = 3
            default_worker_count = 4
        case DatasetType.random_cifar100.value:
            num_classes = 100
            img_size = (32, 32)
            channels = 3
            default_worker_count = 4
        case DatasetType.random_imagenet10.value:
            num_classes = 10
            img_size = (224, 224)
            channels = 3
            default_worker_count = 4
        case DatasetType.random_imagenet100.value:
            num_classes = 100
            img_size = (224, 224)
            channels = 3
            default_worker_count = 8
        case DatasetType.random_imagenet1k.value:
            num_classes = 1000
            img_size = (224, 224)
            channels = 3
            default_worker_count = 8
        case _:
            raise NotImplementedError

    if num_workers is None:
        for class_label in range(num_classes):
            generate_images_for_label(class_label, num_images_per_label, img_size, channels, "train", output_path,
                                      reset_random_seed_per_sample=reset_random_seeds_per_sample, reset_random_seed_per_label=reset_random_seeds_per_label)
            generate_images_for_label(class_label, num_images_per_label_test, img_size, channels, "test", output_path,
                                      reset_random_seed_per_sample=reset_random_seeds_per_sample, reset_random_seed_per_label=reset_random_seeds_per_label)
    else:
        if reset_random_seeds_per_label:
            raise NotImplementedError(f"reset_random_seeds cannot be set in multi-process mode")
        args_list_train = [(class_label, num_images_per_label, img_size, channels, "train", output_path, reset_random_seeds_per_sample, True) for class_label in range(num_classes)]
        args_list_test = [(class_label, num_images_per_label_test, img_size, channels, "test", output_path, reset_random_seeds_per_sample, True) for class_label in range(num_classes)]
        args_list = args_list_train + args_list_test
        num_workers = default_worker_count if num_workers is None else num_workers
        print(f"use {num_workers} workers to generate dataset")
        with Pool(processes=num_workers) as pool:
            pool.starmap(generate_images_for_label, args_list)


    # calculate mean and variance
    mnist_train = datasets.ImageFolder(os.path.join(output_path, "train"), transform=transforms.Compose([transforms.ToTensor()]))
    mean, std = calculate_mean_std(mnist_train)
    mean, std = mean.tolist(), std.tolist()
    with open(os.path.join(output_path, "mean_std.json"), "w", encoding="utf-8") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)


def dataset_type_to_random(dataset_type: DatasetType) -> DatasetType:
    match dataset_type:
        case DatasetType.mnist:
            return DatasetType.random_mnist
        case DatasetType.cifar10:
            return DatasetType.random_cifar10
        case DatasetType.cifar100:
            return DatasetType.random_cifar100
        case DatasetType.imagenet10:
            return DatasetType.random_imagenet10
        case DatasetType.imagenet100:
            return DatasetType.random_imagenet100
        case DatasetType.imagenet1k:
            return DatasetType.random_imagenet1k
        case _:
            raise NotImplementedError(f"Random dataset mapping is not implemented for {dataset_type.name}")


def get_dataset_random(
    dataset_type: DatasetType,
    default_dataset_path: str | Path,
    override_dataset_path: str | Path | None,
    channel: int,
    label_count: int,
) -> DatasetSetup:
    dataset_path = default_dataset_path if override_dataset_path is None else override_dataset_path

    mean_std_file_path = os.path.join(dataset_path, "mean_std.json")
    if os.path.exists(mean_std_file_path):
        with open(mean_std_file_path, "r", encoding="utf-8") as infile:
            data = json.load(infile)
        mean = data["mean"]
        std = data["std"]
    else:
        dataset_train = datasets.ImageFolder(
            os.path.join(dataset_path, "train"),
            transform=transforms.Compose([transforms.ToTensor()]),
        )
        mean_tensor, std_tensor = calculate_mean_std(dataset_train)
        mean = mean_tensor.tolist()
        std = std_tensor.tolist()

    if isinstance(mean, (int, float)):
        mean = [float(mean)]
    if isinstance(std, (int, float)):
        std = [float(std)]

    train_transforms = []
    test_transforms = []
    if channel == 1:
        mean = [sum(mean) / len(mean)]
        std = [sum(std) / len(std)]
        train_transforms.append(transforms.Grayscale())
        test_transforms.append(transforms.Grayscale())
    elif channel != 3:
        raise NotImplementedError(f"Unsupported channel count: {channel}")

    transform_train = transforms.Compose(train_transforms + [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    transform_test = transforms.Compose(test_transforms + [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    dataset_train = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=transform_train)
    dataset_test = datasets.ImageFolder(os.path.join(dataset_path, "test"), transform=transform_test)
    if len(dataset_train.classes) != label_count or len(dataset_test.classes) != label_count:
        raise ValueError(
            f"Random dataset at {dataset_path} has unexpected class count: "
            f"train={len(dataset_train.classes)}, test={len(dataset_test.classes)}, expected={label_count}"
        )
    return DatasetSetup(dataset_type, dataset_train, dataset_test)


def dataset_random_mnist(override_dataset_path=None, *args, **kwargs):
    return get_dataset_random(DatasetType.random_mnist, default_path_random_mnist, override_dataset_path, 1, 10)


def dataset_random_cifar10(override_dataset_path=None, *args, **kwargs):
    return get_dataset_random(DatasetType.random_cifar10, default_path_random_cifar10, override_dataset_path, 3, 10)


def dataset_random_cifar100(override_dataset_path=None, *args, **kwargs):
    return get_dataset_random(DatasetType.random_cifar100, default_path_random_cifar100, override_dataset_path, 3, 100)


def dataset_random_imagenet10(override_dataset_path=None, *args, **kwargs):
    return get_dataset_random(DatasetType.random_imagenet10, default_path_random_imagenet10, override_dataset_path, 3, 10)


def dataset_random_imagenet100(override_dataset_path=None, *args, **kwargs):
    return get_dataset_random(DatasetType.random_imagenet100, default_path_random_imagenet100, override_dataset_path, 3, 100)


def dataset_random_imagenet1k(override_dataset_path=None, *args, **kwargs):
    return get_dataset_random(DatasetType.random_imagenet1k, default_path_random_imagenet1k, override_dataset_path, 3, 1000)


_RANDOM_DATASET_LOADERS = {
    DatasetType.random_mnist: dataset_random_mnist,
    DatasetType.random_cifar10: dataset_random_cifar10,
    DatasetType.random_cifar100: dataset_random_cifar100,
    DatasetType.random_imagenet10: dataset_random_imagenet10,
    DatasetType.random_imagenet100: dataset_random_imagenet100,
    DatasetType.random_imagenet1k: dataset_random_imagenet1k,
}


def get_random_dataset_setup(dataset_type: DatasetType, override_dataset_path: Optional[str] = None) -> DatasetSetup:
    try:
        dataset_loader = _RANDOM_DATASET_LOADERS[dataset_type]
    except KeyError as exc:
        raise NotImplementedError(f"Random dataset loader is not implemented for {dataset_type.name}") from exc
    return dataset_loader(override_dataset_path=override_dataset_path)

if __name__ == '__main__':
    save_random_images(5, DatasetType.random_mnist, "test_mnist_random_dataset")
    save_random_images(5, DatasetType.random_imagenet10, "test_imagenet1k_random_dataset")
