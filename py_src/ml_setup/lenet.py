from typing import Optional

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_dataset import DatasetSetup, dataset_mnist
from py_src.ml_setup_model.lenet import LeNet4, LeNet5, LeNet5LargeFc, weights_init_xavier

from .shared_setup_util import make_setup


# ---------------------------------------------------------------------------
# LeNet
# ---------------------------------------------------------------------------

def lenet5_mnist(override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    ds = dataset_mnist() if override_dataset is None else override_dataset
    model = LeNet5()
    model.apply(weights_init_xavier)
    return make_setup(model, ModelType.lenet5, ds, 64, has_normalization=False)


def lenet4_mnist(override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    ds = dataset_mnist() if override_dataset is None else override_dataset
    model = LeNet4()
    model.apply(weights_init_xavier)
    return make_setup(model, ModelType.lenet4, ds, 64, has_normalization=False)


def lenet5_large_fc_mnist(override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    ds = dataset_mnist() if override_dataset is None else override_dataset
    model = LeNet5LargeFc()
    model.apply(weights_init_xavier)
    return make_setup(model, ModelType.lenet5_large_fc, ds, 64, has_normalization=False)

