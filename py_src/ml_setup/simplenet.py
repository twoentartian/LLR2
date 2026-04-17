from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_dataset import dataset_cifar10, dataset_cifar100

from .shared_setup_util import _make_setup

# ---------------------------------------------------------------------------
# SimpleNet
# ---------------------------------------------------------------------------

def simplenet_cifar10() -> MLSetup:
    from py_src.ml_setup_model.simplenet import simplenet_cifar_5m
    model = simplenet_cifar_5m(num_classes=10)
    return _make_setup(model, ModelType.simplenet, dataset_cifar10(), 64)


def simplenet_cifar100() -> MLSetup:
    from py_src.ml_setup_model.simplenet import simplenet_cifar_5m
    model = simplenet_cifar_5m(num_classes=100)
    return _make_setup(model, ModelType.simplenet, dataset_cifar100(), 64)


