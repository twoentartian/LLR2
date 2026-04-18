from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torch.utils.data.dataloader import default_collate

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_dataset import DatasetSetup, dataset_cifar10, dataset_cifar100
from .shared_setup_util import make_setup
from .imagenet_preset import preset_version, imagenet_criterion, imagenet_collate_fn, imagenet_sampler_fn

# ---------------------------------------------------------------------------
# ResNet-18
# ---------------------------------------------------------------------------

class GroupNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=2, num_channels=num_channels, eps=1e-5, affine=True)

    def forward(self, x):
        return self.norm(x)

def _resnet18_cifar(num_classes, dataset_setup, model_type, use_gn=False) -> MLSetup:
    from torchvision import models
    norm = GroupNorm if use_gn else nn.BatchNorm2d
    model = models.resnet18(num_classes=num_classes, norm_layer=norm)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity() # type: ignore
    return make_setup(model, model_type, dataset_setup, 256)


def resnet18_cifar10(override_dataset:Optional[DatasetSetup]=None, use_gn=False) -> MLSetup:
    ds = dataset_cifar10() if override_dataset is None else override_dataset
    mt = ModelType.resnet18_bn if use_gn is False else ModelType.resnet18_gn
    return _resnet18_cifar(10, ds, mt, use_gn)


def resnet18_cifar100(override_dataset:Optional[DatasetSetup]=None, use_gn=False) -> MLSetup:
    ds = dataset_cifar100() if override_dataset is None else override_dataset
    mt = ModelType.resnet18_bn if use_gn is False else ModelType.resnet18_gn
    return _resnet18_cifar(10, ds, mt, use_gn)



# ---------------------------------------------------------------------------
# ResNet ImageNet helpers
# ---------------------------------------------------------------------------

def _resnet18_imagenet(num_classes:int, dataset_setup:DatasetSetup, pyotrch_preset:int, model_type:ModelType, use_gn=False) -> MLSetup:
    from torchvision import models
    norm = GroupNorm if use_gn else nn.BatchNorm2d
    model = models.resnet18(num_classes=num_classes, norm_layer=norm)
    criterion = imagenet_criterion(pyotrch_preset)
    collate_fn = imagenet_collate_fn(pyotrch_preset)
    sampler = imagenet_sampler_fn(pyotrch_preset)
    return make_setup(model, model_type, dataset_setup, 256, criterion=criterion, 
                      default_collate_fn=collate_fn, default_collate_fn_val=default_collate,
                      default_sampler_fn=sampler)


# ---------------------------------------------------------------------------
# ResNet-18 ImageNet-10
# ---------------------------------------------------------------------------

def resnet18_imagenet10(preset: int = 1, override_dataset:Optional[DatasetSetup]=None, use_gn=False) -> MLSetup:
    """ResNet-18 (BN) on ImageNet-10, batch=256."""
    from py_src.ml_setup_dataset import dataset_imagenet10
    pv = preset_version(preset)
    ds = override_dataset if override_dataset is not None else dataset_imagenet10(pv)
    return _resnet18_imagenet(10, ds, pv, ModelType.resnet18_bn, use_gn=use_gn,)


# ---------------------------------------------------------------------------
# ResNet-18 ImageNet-100
# ---------------------------------------------------------------------------

def resnet18_imagenet100(preset: int = 1, override_dataset:Optional[DatasetSetup]=None, use_gn=False) -> MLSetup:
    """ResNet-18 (BN) on ImageNet-100, batch=256."""
    from py_src.ml_setup_dataset import dataset_imagenet100
    pv = preset_version(preset)
    ds = override_dataset if override_dataset is not None else dataset_imagenet100(pv)
    return _resnet18_imagenet(100, ds, pv, ModelType.resnet18_bn, use_gn=use_gn)


# ---------------------------------------------------------------------------
# ResNet-18 ImageNet-1k
# ---------------------------------------------------------------------------

def resnet18_imagenet1k(preset: int = 1, override_dataset:Optional[DatasetSetup]=None, use_gn=False) -> MLSetup:
    """ResNet-18 (BN) on ImageNet-1k, batch=256."""
    from py_src.ml_setup_dataset import dataset_imagenet1k
    pv = preset_version(preset)
    ds = override_dataset if override_dataset is not None else dataset_imagenet1k(pv)
    return _resnet18_imagenet(1000, ds, pv, ModelType.resnet18_bn, use_gn=use_gn)


# ---------------------------------------------------------------------------
# ResNet-18 ImageNet-1k SAM-mask variants (fixed v1-style preprocessing)
# ---------------------------------------------------------------------------

def resnet18_bn_imagenet1k_sam_mask_random_noise() -> MLSetup:
    """ResNet-18 (BN) on ImageNet-1k with SAM-mask regions filled with random noise, batch=256."""
    from torchvision import models
    from py_src.ml_setup_dataset import dataset_imagenet1k_sam_mask_random_noise
    model = models.resnet18(num_classes=1000)
    return make_setup(model, ModelType.resnet18_bn,
                       dataset_imagenet1k_sam_mask_random_noise(), 256)


def resnet18_bn_imagenet1k_sam_mask_black() -> MLSetup:
    """ResNet-18 (BN) on ImageNet-1k with SAM-mask regions zeroed out, batch=256."""
    from torchvision import models
    from py_src.ml_setup_dataset import dataset_imagenet1k_sam_mask_black
    model = models.resnet18(num_classes=1000)
    return make_setup(model, ModelType.resnet18_bn,
                       dataset_imagenet1k_sam_mask_black(), 256)


# ---------------------------------------------------------------------------
# ResNet-34 ImageNet-1k
# ---------------------------------------------------------------------------

def resnet34_imagenet1k(preset: int = 1, override_dataset:Optional[DatasetSetup]=None) -> MLSetup:
    """ResNet-34 on ImageNet-1k, batch=256."""
    from torchvision import models
    from py_src.ml_setup_dataset import dataset_imagenet1k
    model = models.resnet34(num_classes=1000)
    pv = preset_version(preset)
    ds = override_dataset if override_dataset is not None else dataset_imagenet1k(pv)
    criterion = imagenet_criterion(pv)
    collate_fn = imagenet_collate_fn(pv)
    sampler = imagenet_sampler_fn(pv)
    return make_setup(model, ModelType.resnet34, ds, 256,
                       criterion=criterion, 
                       default_collate_fn=collate_fn,
                       default_collate_fn_val=default_collate,
                       default_sampler_fn=sampler)


# ---------------------------------------------------------------------------
# ResNet-50 ImageNet
# ---------------------------------------------------------------------------

def resnet50_imagenet1k(preset: int = 1, override_dataset:Optional[DatasetSetup]=None) -> MLSetup:
    """ResNet-50 on ImageNet-1k, batch=256."""
    from torchvision import models
    from py_src.ml_setup_dataset import dataset_imagenet1k
    model = models.resnet50(num_classes=1000)
    pv = preset_version(preset)
    ds = override_dataset if override_dataset is not None else dataset_imagenet1k(pv)
    criterion = imagenet_criterion(pv)
    collate_fn = imagenet_collate_fn(pv)
    sampler = imagenet_sampler_fn(pv)
    return make_setup(model, ModelType.resnet50, ds, 256,
                       criterion=criterion, 
                       default_collate_fn=collate_fn,
                       default_collate_fn_val=default_collate,
                       default_sampler_fn=sampler)



def resnet50_imagenet100(preset: int = 1, override_dataset:Optional[DatasetSetup]=None) -> MLSetup:
    """ResNet-50 on ImageNet-100, batch=256."""
    from torchvision import models
    from py_src.ml_setup_dataset import dataset_imagenet100
    model = models.resnet50(num_classes=100)
    pv = preset_version(preset)
    ds = override_dataset if override_dataset is not None else dataset_imagenet100(pv)
    criterion = imagenet_criterion(pv)
    collate_fn = imagenet_collate_fn(pv)
    sampler = imagenet_sampler_fn(pv)
    return make_setup(model, ModelType.resnet50, ds, 256,
                       criterion=criterion, 
                       default_collate_fn=collate_fn,
                       default_collate_fn_val=default_collate,
                       default_sampler_fn=sampler)
