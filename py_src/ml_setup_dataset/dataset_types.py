from dataclasses import dataclass
from enum import Enum, auto

from torch.utils.data import Dataset

""" Dataset Enum """
class DatasetType(Enum):
    default = auto()
    mnist = auto()
    mnist_224 = auto()
    random_mnist = auto()
    cifar10 = auto()
    cifar10_224 = auto()
    random_cifar10 = auto()
    cifar100 = auto()
    cifar100_224 = auto()
    random_cifar100 = auto()
    flowers102 = auto()
    imagenet10 = auto()
    random_imagenet10 = auto()
    imagenet100 = auto()
    random_imagenet100 = auto()
    imagenet1k = auto()
    random_imagenet1k = auto()
    imagenet1k_sam_mask_random_noise = auto()
    imagenet1k_sam_mask_black = auto()
    svhn = auto()
    flickr30k = auto()
    arithmetic_addition = auto()
    arithmetic_cubepoly = auto()
    arithmetic_cube2 = auto()
    arithmetic_exp_unknown = auto()

@dataclass
class DatasetSetup:
    dataset_type: DatasetType
    train_data: Dataset
    valdation_data: Dataset
