from typing import Optional
from torchvision import transforms, datasets

from .dataset_types import DatasetType
from .dataset_env import default_path_mnist
from .dataset_types import DatasetSetup

""" MNIST """
def dataset_mnist(rescale_to_224=False, random_rotation=5, override_dataset_path: Optional[str]=None, augmentation=True, *args, **kwargs):
    dataset_path = default_path_mnist if override_dataset_path is None else override_dataset_path
    mnist_train = datasets.MNIST(root=dataset_path, train=True, download=True)
    mean = mnist_train.data.float().mean() / 255
    std = mnist_train.data.float().std() / 255
    if rescale_to_224:
        dataset_type = DatasetType.mnist_224
    else:
        dataset_type = DatasetType.mnist

    train_transforms = []
    test_transforms = []
    # data augmentation
    if augmentation:
        if random_rotation != 0:
            train_transforms.append(transforms.RandomRotation(random_rotation, fill=0))
    if rescale_to_224:
        train_transforms.append(transforms.Resize((224, 224)))
        test_transforms.append(transforms.Resize((224, 224)))
    else:
        train_transforms.append(transforms.RandomCrop(28, padding=2))
    train_transforms = train_transforms + [transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])]
    test_transforms = test_transforms + [transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])]
    train_data = datasets.MNIST(root=dataset_path, train=True, download=False, transform=transforms.Compose(train_transforms))
    test_data = datasets.MNIST(root=dataset_path, train=False, download=False, transform=transforms.Compose(test_transforms))
    return DatasetSetup(dataset_type, train_data, test_data)
