from collections.abc import Callable
from typing import Optional, Any, List
from torchvision import transforms, datasets

from .dataset_types import DatasetType
from .dataset_env import default_path_cifar10, default_path_cifar100
from .dataset_types import DatasetSetup

_type_mean_std = Optional[tuple[tuple[float, float, float], tuple[float, float, float]]]
_type_transform = Callable[[Any], Any]

""" CIFAR10 """
def dataset_cifar10(rescale_to_224=False, transforms_training:Optional[List[_type_transform]]=None, transforms_testing:Optional[List[_type_transform]]=None, 
                    mean_std:_type_mean_std =None, augmentation=True, *args, **kwargs):
    dataset_path = default_path_cifar10
    if rescale_to_224:
        dataset_type = DatasetType.cifar10_224
    else:
        dataset_type = DatasetType.cifar10
    dataset_name = str(dataset_type.name)
    if mean_std is not None:
        stats = mean_std
    else:
        stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))

    # data augmentation
    if augmentation:
        data_augmentation: list[_type_transform] = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomCrop(32, padding=4, padding_mode='reflect')]
    else:
        data_augmentation: list[_type_transform] = []
    if rescale_to_224:
        transform_rescale_to_224: list[_type_transform] = [transforms.Resize((224, 224))]
    else:
        transform_rescale_to_224: list[_type_transform] = []
    train_transforms = data_augmentation + transform_rescale_to_224 + [transforms.ToTensor(), transforms.Normalize(*stats)]
    test_transforms = transform_rescale_to_224 + [transforms.ToTensor(), transforms.Normalize(*stats)]

    if transforms_training is None:
        cifar10_train = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transforms.Compose(train_transforms))
    else:
        cifar10_train = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transforms.Compose(transforms_training))
    if transforms_testing is None:
        cifar10_test = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transforms.Compose(test_transforms))
    else:
        cifar10_test = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transforms.Compose(transforms_testing))

    return DatasetSetup(dataset_type, cifar10_train, cifar10_test)



""" CIFAR100 """
def dataset_cifar100(rescale_to_224=False, transforms_training:Optional[List[_type_transform]]=None, transforms_testing:Optional[List[_type_transform]]=None, 
                     mean_std:_type_mean_std=None, augmentation=True, *args, **kwargs):
    dataset_path = default_path_cifar100
    if rescale_to_224:
        dataset_type = DatasetType.cifar100_224
    else:
        dataset_type = DatasetType.cifar100
    dataset_name = str(dataset_type.name)
    if mean_std is not None:
        stats = mean_std
    else:
        stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    # data augmentation
    if augmentation:
        data_augmentation: list[_type_transform] = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomCrop(32, padding=4, padding_mode='reflect')]
    else:
        data_augmentation: list[_type_transform] = []
    if rescale_to_224:
        transform_rescale_to_224: list[_type_transform] = [transforms.Resize((224, 224))]
    else:
        transform_rescale_to_224: list[_type_transform] = []
    train_transforms = data_augmentation + transform_rescale_to_224 + [transforms.ToTensor(), transforms.Normalize(*stats)]
    test_transforms = transform_rescale_to_224 + [transforms.ToTensor(), transforms.Normalize(*stats)]

    if transforms_training is None:
        cifar100_train = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transforms.Compose(train_transforms))
    else:
        cifar100_train = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transforms.Compose(transforms_training))
    if transforms_testing is None:
        cifar100_test = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transforms.Compose(test_transforms))
    else:
        cifar100_test = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transforms.Compose(transforms_testing))
    return DatasetSetup(dataset_type, cifar100_train, cifar100_test)