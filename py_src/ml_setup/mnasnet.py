from typing import Optional

from torch.utils.data.dataloader import default_collate

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_dataset import DatasetSetup, dataset_imagenet1k
from .shared_setup_util import make_setup
from .imagenet_preset import preset_version, imagenet_criterion, imagenet_collate_fn, imagenet_sampler_fn


def _mnasnet_imagenet1k(
    width: float,
    model_type: ModelType,
    batch_size: int,
    preset: int = 1,
    override_dataset: Optional[DatasetSetup] = None,
) -> MLSetup:
    from torchvision import models

    pv = preset_version(preset)
    ds = override_dataset if override_dataset is not None else dataset_imagenet1k(
        preset_version=pv,
        train_crop_size=224,
        val_resize_size=256,
        val_crop_size=224,
    )
    if width == 0.5:
        model = models.mnasnet0_5(progress=False, weights=None)
    elif width == 1.0:
        model = models.mnasnet1_0(progress=False, weights=None)
    else:
        raise NotImplementedError(f"Unsupported MNASNet width {width}")
    return make_setup(
        model,
        model_type,
        ds,
        batch_size,
        criterion=imagenet_criterion(pv),
        default_collate_fn=imagenet_collate_fn(pv),
        default_collate_fn_val=default_collate,
        default_sampler_fn=imagenet_sampler_fn(pv),
    )


def mnasnet0_5_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    """MNASNet-0.5 on ImageNet-1k, ported from DFL_torch."""
    return _mnasnet_imagenet1k(0.5, ModelType.mnasnet0_5, 512, preset, override_dataset)


def mnasnet1_0_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    """MNASNet-1.0 on ImageNet-1k, ported from DFL_torch."""
    return _mnasnet_imagenet1k(1.0, ModelType.mnasnet1_0, 256, preset, override_dataset)
