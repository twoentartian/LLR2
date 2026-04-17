from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_dataset import dataset_cifar10
from .shared_setup_util import _make_setup

# ---------------------------------------------------------------------------
# DenseNet (CIFAR)
# ---------------------------------------------------------------------------

def densenet121_cifar10() -> MLSetup:
    from ml_setup_model.densenet import _DenseNetCifar
    model = _DenseNetCifar([6, 12, 24, 16], growth_rate=32, num_classes=10)
    return _make_setup(model, ModelType.densenet121, dataset_cifar10(), 256)


def densenet_cifar_cifar10() -> MLSetup:
    from ml_setup_model.densenet import _DenseNetCifar
    model = _DenseNetCifar([6, 12, 24, 16], growth_rate=12, num_classes=10)
    return _make_setup(model, ModelType.densenet_cifar, dataset_cifar10(), 256)