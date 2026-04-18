from typing import Optional

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_dataset import DatasetSetup, dataset_cifar10
from .shared_setup_util import make_setup

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
