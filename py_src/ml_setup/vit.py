from typing import Optional

from torch.utils.data.dataloader import default_collate

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_dataset import DatasetSetup, dataset_imagenet1k
from .shared_setup_util import make_setup
from .imagenet_preset import preset_version, imagenet_criterion, imagenet_collate_fn, imagenet_sampler_fn


def vit_b_32_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    """ViT-B/32 on ImageNet-1k, ported from DFL_torch."""
    from torchvision import models

    pv = preset_version(preset)
    ds = override_dataset if override_dataset is not None else dataset_imagenet1k(
        preset_version=pv,
        train_crop_size=224,
        val_resize_size=256,
        val_crop_size=224,
    )
    model = models.vit_b_32(progress=False, weights=None, num_classes=1000)
    return make_setup(
        model,
        ModelType.vit_b_32,
        ds,
        512,
        criterion=imagenet_criterion(pv),
        clip_grad_norm=1,
        default_collate_fn=imagenet_collate_fn(pv, mixup_alpha=0.2, cutmix_alpha=1.0),
        default_collate_fn_val=default_collate,
        default_sampler_fn=imagenet_sampler_fn(pv),
    )
