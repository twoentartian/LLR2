from functools import partial
from typing import Optional

import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_dataset import DatasetSetup, dataset_imagenet1k_from_pytorch
from py_src.torch_vision_train import RASampler, get_mixup_cutmix
from .shared_setup_util import make_setup


def _mixup_cutmix_collate(
    batch,
    *,
    mixup_cutmix,
):
    if isinstance(batch, tuple) and len(batch) == 2:
        images, targets = batch
    else:
        images, targets = default_collate(batch)
    images, targets = mixup_cutmix(images, targets)
    if hasattr(images, "device") and hasattr(targets, "to") and targets.device != images.device:
        targets = targets.to(images.device, non_blocking=True)
    return images, targets


def _vit_b_32_torchvision_collate_fn():
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        num_classes=1000,
        use_v2=True,
    )
    return partial(_mixup_cutmix_collate, mixup_cutmix=mixup_cutmix)


def _vit_b_32_torchvision_sampler_fn(dataset):
    return RASampler(dataset, shuffle=True, repetitions=3)


def vit_b_32_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    """ViT-B/32 on ImageNet-1k.

    The only supported recipe is the Torchvision reference recipe (preset=1),
    except that the optimizer lr / weight decay come from the local DeiT-style tuning in
    ``complete_ml_setup.py``.
    """
    from torchvision import models

    if preset not in (0, 1):
        raise ValueError(f"vit_b_32 only supports the Torchvision recipe (preset=1), got preset={preset}")
    ds = override_dataset if override_dataset is not None else dataset_imagenet1k_from_pytorch(
        train_crop_size=224,
        val_resize_size=256,
        val_crop_size=224,
        interpolation=InterpolationMode.BILINEAR,
        auto_augment_policy="imagenet",
        random_erase_prob=0.0,
        use_v2=True,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.11)
    collate_fn = _vit_b_32_torchvision_collate_fn()
    sampler_fn = _vit_b_32_torchvision_sampler_fn

    model = models.vit_b_32(progress=False, weights=None, num_classes=1000)
    return make_setup(
        model,
        ModelType.vit_b_32,
        ds,
        512,
        criterion=criterion,
        clip_grad_norm=1,
        default_collate_fn=collate_fn,
        default_collate_fn_val=default_collate,
        default_sampler_fn=sampler_fn,
        default_prefetch_factor=16,
    )
