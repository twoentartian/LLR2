from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_dataset import dataset_mnist
from py_src.ml_setup_model.lenet import LeNet4, LeNet5, LeNet5LargeFc, weights_init_xavier

from .shared_setup_util import _make_setup


# ---------------------------------------------------------------------------
# LeNet
# ---------------------------------------------------------------------------

def lenet5_mnist() -> MLSetup:
    model = LeNet5()
    model.apply(weights_init_xavier)
    return _make_setup(model, ModelType.lenet5, dataset_mnist(), 64, has_normalization=False)


def lenet4_mnist() -> MLSetup:
    model = LeNet4()
    model.apply(weights_init_xavier)
    return _make_setup(model, ModelType.lenet4, dataset_mnist(), 64, has_normalization=False)


def lenet5_large_fc_mnist() -> MLSetup:
    model = LeNet5LargeFc()
    model.apply(weights_init_xavier)
    return _make_setup(model, ModelType.lenet5_large_fc, dataset_mnist(), 64, has_normalization=False)


