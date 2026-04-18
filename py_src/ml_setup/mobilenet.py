from typing import Optional

from torch.utils.data.dataloader import default_collate

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_dataset import DatasetSetup, dataset_cifar10, dataset_cifar100
from .shared_setup_util import make_setup
from .imagenet_preset import preset_version, imagenet_criterion, imagenet_collate_fn, imagenet_sampler_fn

# ---------------------------------------------------------------------------
# MobileNet-V2 (CIFAR-optimised custom implementation)
# ---------------------------------------------------------------------------

def mobilenet_v2_cifar10(override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    from py_src.ml_setup_model.mobilenet import MobileNetV2
    ds = dataset_cifar10() if override_dataset is None else override_dataset
    model = MobileNetV2(output_size=10)
    return make_setup(model, ModelType.mobilenet_v2, ds, 128)


def mobilenet_v2_cifar100(override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    from py_src.ml_setup_model.mobilenet import MobileNetV2
    ds = dataset_cifar100() if override_dataset is None else override_dataset
    model = MobileNetV2(output_size=100)
    return make_setup(model, ModelType.mobilenet_v2, ds, 128)


# ---------------------------------------------------------------------------
# MobileNetV3-Large  (ImageNet-1k)
# ---------------------------------------------------------------------------

def mobilenet_v3_large_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    """MobileNetV3-Large on ImageNet-1k, batch=128.

    preset=0 → preprocessing v2 + label_smoothing=0.1  (default)
    preset=1 → preprocessing v1 + plain CrossEntropyLoss
    """
    from torchvision import models
    from py_src.ml_setup_dataset import dataset_imagenet1k
    pv = preset_version(preset)
    ds = override_dataset if override_dataset is not None else dataset_imagenet1k(preset_version=pv)
    model = models.mobilenet_v3_large()
    collate_fn = imagenet_collate_fn(pv)
    sampler_fn = imagenet_sampler_fn(pv)
    return make_setup(model, ModelType.mobilenet_v3_large, ds, 128,
                       criterion=imagenet_criterion(pv),
                       default_collate_fn=collate_fn,
                       default_collate_fn_val=default_collate,
                       default_sampler_fn=sampler_fn)
