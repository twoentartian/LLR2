from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_dataset import dataset_cifar10, dataset_cifar100
from .shared_setup_util import make_setup


# ---------------------------------------------------------------------------
# RegNet-X-200MF
# ---------------------------------------------------------------------------

def regnet_x_200mf_cifar10() -> MLSetup:
    from py_src.ml_setup_model.regnet import RegNetX_200MF
    model = RegNetX_200MF(num_classes=10)
    return make_setup(model, ModelType.regnet_x_200mf, dataset_cifar10(), 256)


def regnet_x_200mf_cifar100() -> MLSetup:
    from py_src.ml_setup_model.regnet import RegNetX_200MF
    model = RegNetX_200MF(num_classes=100)
    return make_setup(model, ModelType.regnet_x_200mf, dataset_cifar100(), 256)

