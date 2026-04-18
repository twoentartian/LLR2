from typing import Optional

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_dataset import DatasetSetup, dataset_cifar10
from .shared_setup_util import make_setup

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
