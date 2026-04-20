from typing import Optional

from torch.utils.data.dataloader import default_collate

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_dataset import DatasetSetup, dataset_cifar10, dataset_imagenet1k
from .shared_setup_util import make_setup
from .imagenet_preset import preset_version, imagenet_criterion, imagenet_collate_fn, imagenet_sampler_fn

# ---------------------------------------------------------------------------
# VGG
# ---------------------------------------------------------------------------

def vgg11_bn_cifar10(override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    from py_src.ml_setup_model.vgg import VGGCifar
    ds = dataset_cifar10() if override_dataset is None else override_dataset
    model = VGGCifar('VGG11', num_classes=10)
    return make_setup(model, ModelType.vgg11_bn, ds, 256)


def vgg11_no_bn_cifar10(override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    from py_src.ml_setup_model.vgg import VGG11_no_bn
    ds = dataset_cifar10(rescale_to_224=True) if override_dataset is None else override_dataset
    model = VGG11_no_bn(in_channels=3, num_classes=10)
    return make_setup(model, ModelType.vgg11_no_bn, ds, 32, has_normalization=False)


def vgg11_bn_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    """VGG-11-BN on ImageNet-1k, ported from DFL_torch."""
    from torchvision import models

    pv = preset_version(preset)
    ds = override_dataset if override_dataset is not None else dataset_imagenet1k(
        preset_version=pv,
        train_crop_size=224,
        val_resize_size=256,
        val_crop_size=224,
    )
    model = models.vgg11_bn(progress=False, weights=None, num_classes=1000)
    return make_setup(
        model,
        ModelType.vgg11_bn,
        ds,
        128,
        criterion=imagenet_criterion(pv),
        default_collate_fn=imagenet_collate_fn(pv),
        default_collate_fn_val=default_collate,
        default_sampler_fn=imagenet_sampler_fn(pv),
    )
