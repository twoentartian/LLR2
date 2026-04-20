from typing import Optional

from torch.utils.data.dataloader import default_collate

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_dataset import DatasetSetup, dataset_imagenet1k
from .shared_setup_util import make_setup
from .imagenet_preset import preset_version, imagenet_criterion, imagenet_collate_fn, imagenet_sampler_fn


def conveNeXt_tiny_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    """ConvNeXt-Tiny on ImageNet-1k, ported from DFL_torch.

    The function name preserves the original DFL_torch capitalization typo for compatibility.
    """
    from torchvision import models

    pv = preset_version(preset)
    ds = override_dataset if override_dataset is not None else dataset_imagenet1k(
        preset_version=pv,
        train_crop_size=176,
        val_resize_size=232,
        val_crop_size=224,
    )
    model = models.convnext_tiny(progress=False, weights=None, num_classes=1000)
    return make_setup(
        model,
        ModelType.convnext_tiny,
        ds,
        128,
        criterion=imagenet_criterion(pv),
        default_collate_fn=imagenet_collate_fn(pv),
        default_collate_fn_val=default_collate,
        default_sampler_fn=imagenet_sampler_fn(pv),
    )


def convnext_tiny_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    return conveNeXt_tiny_imagenet1k(preset, override_dataset)
