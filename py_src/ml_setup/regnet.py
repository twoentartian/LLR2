from typing import Optional

from torch.utils.data.dataloader import default_collate

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_dataset import DatasetSetup, dataset_cifar10, dataset_cifar100, dataset_imagenet1k
from .shared_setup_util import make_setup
from .imagenet_preset import preset_version, imagenet_criterion, imagenet_collate_fn, imagenet_sampler_fn


# ---------------------------------------------------------------------------
# RegNet-X-200MF
# ---------------------------------------------------------------------------

def regnet_x_200mf_cifar10(override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    from py_src.ml_setup_model.regnet import RegNetX_200MF
    ds = dataset_cifar10() if override_dataset is None else override_dataset
    model = RegNetX_200MF(num_classes=10)
    return make_setup(model, ModelType.regnet_x_200mf, ds, 256)


def regnet_x_200mf_cifar100(override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    from py_src.ml_setup_model.regnet import RegNetX_200MF
    ds = dataset_cifar100() if override_dataset is None else override_dataset
    model = RegNetX_200MF(num_classes=100)
    return make_setup(model, ModelType.regnet_x_200mf, ds, 256)


def regnet_y_400mf_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    """RegNet-Y-400MF on ImageNet-1k, ported from DFL_torch."""
    from torchvision import models

    pv = preset_version(preset)
    ds = override_dataset if override_dataset is not None else dataset_imagenet1k(
        preset_version=pv,
        train_crop_size=224,
        val_resize_size=256 if pv == 1 else 232,
        val_crop_size=224,
    )
    model = models.regnet_y_400mf(progress=False, weights=None, num_classes=1000)
    return make_setup(
        model,
        ModelType.regnet_y_400mf,
        ds,
        128,
        criterion=imagenet_criterion(pv),
        default_collate_fn=imagenet_collate_fn(pv),
        default_collate_fn_val=default_collate,
        default_sampler_fn=imagenet_sampler_fn(pv),
    )
