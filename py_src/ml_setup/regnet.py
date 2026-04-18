from typing import Optional

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_dataset import DatasetSetup, dataset_cifar10, dataset_cifar100
from .shared_setup_util import make_setup


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
