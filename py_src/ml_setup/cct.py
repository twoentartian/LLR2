import os
import yaml

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.utils.data._utils.collate import default_collate

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_dataset import dataset_cifar10, dataset_cifar100, dataset_imagenet1k_from_pytorch
from .shared_setup_util import make_setup
from py_src.ml_setup_dataset.dataset_default import default_path_imagenet1k

# ---------------------------------------------------------------------------
# CCT-7 3x1 32
# ---------------------------------------------------------------------------

def cct_7_3x1_cifar10() -> MLSetup:
    import py_src.third_party.compact_transformers.src.cct as cct
    model = cct.cct_7_3x1_32()
    return make_setup(model, ModelType.cct_7_3x1_32, dataset_cifar10(), 128)


def cct_7_3x1_cifar100() -> MLSetup:
    import py_src.third_party.compact_transformers.src.cct as cct
    model = cct.cct_7_3x1_32(num_classes=100)
    return make_setup(model, ModelType.cct_7_3x1_32, dataset_cifar100(), 128)


def __mixup_collate(
    batch: list[tuple[Any, Any]],
    mixup_fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    images, targets = default_collate(batch)
    images, targets = mixup_fn(images, targets)
    return images, targets

def cct14_7x2_imagenet1k() -> MLSetup:
    import py_src.third_party.compact_transformers.src.cct as cct
    from .timm_helper import timm_build_loaders, timm_build_mixup_and_loss
    dataset = dataset_imagenet1k_from_pytorch(auto_augment_policy='imagenet', val_crop_size=224, val_resize_size=256, train_crop_size=224)

    cfg = yaml.safe_load(open(f"{os.path.dirname(os.path.abspath(__file__))}/timm_config/cct_imagenet_config.yaml", "r"))

    train_loader, val_loader = timm_build_loaders(cfg, str(default_path_imagenet1k))
    mixup_fn, criterion = timm_build_mixup_and_loss(cfg)

    output_setup: MLSetup = make_setup(cct.cct_14_7x2_224(), ModelType.cct_14_7x2_224, dataset, 128, criterion=criterion, default_collate_fn=partial(__mixup_collate, mixup_fn=mixup_fn)) # type: ignore
    output_setup.override_train_loader = train_loader
    output_setup.override_test_loader = val_loader

    return output_setup