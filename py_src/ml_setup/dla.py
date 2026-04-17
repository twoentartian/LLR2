"""DLA model setups for LLR2."""

import torch.nn as nn

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_dataset import DatasetSetup, dataset_imagenet10, dataset_cifar10
from .shared_setup_util import make_setup
from .imagenet_preset import preset_version, imagenet_criterion


def dla_cifar10() -> MLSetup:
    from py_src.ml_setup_model.dla.dla_cifar import DLA
    model = DLA(num_classes=10)
    return make_setup(model, ModelType.densenet121, dataset_cifar10(), 256)

def dla_cifar100() -> MLSetup:
    from py_src.ml_setup_model.dla.dla_cifar import DLA
    model = DLA(num_classes=100)
    return make_setup(model, ModelType.densenet121, dataset_cifar10(), 256)

def dla46c_imagenet10(preset: int = 1) -> MLSetup:
    """DLA-46-C on ImageNet-10, batch=256.

    preset=0 → dataset v1 preprocessing + plain CrossEntropyLoss  (default)
    preset=1 → dataset v2 preprocessing + label_smoothing=0.1
    """
    from py_src.ml_setup_model.dla.dla_imagenet import dla46_c
    pv: int = preset_version(preset)
    ds: DatasetSetup = dataset_imagenet10(preset_version=pv)
    model = dla46_c(num_classes=10)
    criterion = imagenet_criterion(pv)
    return make_setup(model, ModelType.dla_46_c, ds, 256, criterion=criterion)
