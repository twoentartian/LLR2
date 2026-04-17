import torch.nn as nn

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_dataset import dataset_cifar10, dataset_cifar100
from .shared_setup_util import make_setup
from .imagenet_preset import preset_version, imagenet_criterion


# ---------------------------------------------------------------------------
# EfficientNet-B0  (CIFAR)
# ---------------------------------------------------------------------------

def efficientnet_b0_cifar10() -> MLSetup:
    from torchvision import models
    model = models.efficientnet_b0(num_classes=10)
    return make_setup(model, ModelType.efficientnet_b0, dataset_cifar10(), 128)


def efficientnet_b0_cifar100() -> MLSetup:
    from torchvision import models
    model = models.efficientnet_b0(num_classes=100)
    return make_setup(model, ModelType.efficientnet_b0, dataset_cifar100(), 128)

# ---------------------------------------------------------------------------
# EfficientNetV2-S  (ImageNet-1k)
# ---------------------------------------------------------------------------

def efficientnet_v2_s_imagenet1k(preset: int = 1) -> MLSetup:
    """EfficientNetV2-S on ImageNet-1k, batch=64.

    Uses enlarged input sizes: train_crop=300, val_resize=384, val_crop=384.
    preset=0 → preprocessing v2 + label_smoothing=0.1  (default)
    preset=1 → preprocessing v1 + plain CrossEntropyLoss
    """
    from torchvision import models
    from py_src.ml_setup_dataset import dataset_imagenet1k
    ds = dataset_imagenet1k(
        preset_version=preset_version(preset),
        train_crop_size=300,
        val_resize_size=384,
        val_crop_size=384,
    )
    model = models.efficientnet_v2_s()
    return make_setup(model, ModelType.efficientnet_v2_s, ds, 64, criterion=imagenet_criterion(preset))


# ---------------------------------------------------------------------------
# EfficientNet-B1  (ImageNet-1k)
# ---------------------------------------------------------------------------

def efficientnet_b1_imagenet1k(preset: int = 1) -> MLSetup:
    """EfficientNet-B1 on ImageNet-1k, batch=256.

    Uses tuned crop/resize sizes: train_crop=208, val_crop=240, val_resize=255.
    preset=0 → preprocessing v2 + label_smoothing=0.1  (default)
    preset=1 → preprocessing v1 + plain CrossEntropyLoss
    """
    from torchvision import models
    from py_src.ml_setup_dataset import dataset_imagenet1k
    ds = dataset_imagenet1k(
        preset_version=preset_version(preset),
        train_crop_size=208,
        val_resize_size=255,
        val_crop_size=240,
    )
    model = models.efficientnet_b1()
    return make_setup(model, ModelType.efficientnet_b1, ds, 256, criterion=imagenet_criterion(preset))

