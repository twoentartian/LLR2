import numpy as np
import os
import json
from typing import Literal, Optional
from multiprocessing import Pool

from torchvision import transforms, datasets
from PIL import Image

from .dataset_util import calculate_mean_std
from .dataset_types import DatasetType

def generate_images_for_label(label, num_images, img_size, channels, split: Literal["train", "test", "val"], output_path, reset_random_seed_per_sample, reset_random_seed_per_label):
    current_label_dir = os.path.join(output_path, split, str(label))
    os.makedirs(current_label_dir)
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

if __name__ == '__main__':
    save_random_images(5, DatasetType.random_mnist, "test_mnist_random_dataset")
    save_random_images(5, DatasetType.random_imagenet10, "test_imagenet1k_random_dataset")
