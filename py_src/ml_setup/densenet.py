from typing import Optional

from torch.utils.data.dataloader import default_collate

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_dataset import DatasetSetup, dataset_cifar10, dataset_imagenet1k
from .shared_setup_util import make_setup
from .imagenet_preset import preset_version, imagenet_criterion, imagenet_collate_fn, imagenet_sampler_fn

# ---------------------------------------------------------------------------
# DenseNet (CIFAR)
# ---------------------------------------------------------------------------

def densenet121_cifar10(override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    from py_src.ml_setup_model.densenet import _DenseNetCifar
    ds = dataset_cifar10() if override_dataset is None else override_dataset
    model = _DenseNetCifar([6, 12, 24, 16], growth_rate=32, num_classes=10)
    return make_setup(model, ModelType.densenet121, ds, 256)


def densenet_cifar_cifar10(override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    from py_src.ml_setup_model.densenet import _DenseNetCifar
    ds = dataset_cifar10() if override_dataset is None else override_dataset
    model = _DenseNetCifar([6, 12, 24, 16], growth_rate=12, num_classes=10)
    return make_setup(model, ModelType.densenet_cifar, ds, 256)


def densenet121_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    """DenseNet-121 on ImageNet-1k, ported from DFL_torch."""
    from torchvision import models

    pv = preset_version(preset)
    ds = override_dataset if override_dataset is not None else dataset_imagenet1k(
        preset_version=pv,
        train_crop_size=224,
        val_resize_size=256,
        val_crop_size=224,
    )
    model = models.densenet121(progress=False, weights=None, num_classes=1000)
    return make_setup(
        model,
        ModelType.densenet121,
        ds,
        128,
        criterion=imagenet_criterion(pv),
        default_collate_fn=imagenet_collate_fn(pv),
        default_collate_fn_val=default_collate,
        default_sampler_fn=imagenet_sampler_fn(pv),
    )
