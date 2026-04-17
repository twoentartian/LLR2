from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_dataset import dataset_cifar10, dataset_cifar100
from .shared_setup_util import _make_setup

# ---------------------------------------------------------------------------
# MobileNet-V2 (CIFAR-optimised custom implementation)
# ---------------------------------------------------------------------------

def mobilenet_v2_cifar10() -> MLSetup:
    from py_src.ml_setup_model.mobilenet import MobileNetV2
    model = MobileNetV2(output_size=10)
    return _make_setup(model, ModelType.mobilenet_v2, dataset_cifar10(), 128)


def mobilenet_v2_cifar100() -> MLSetup:
    from py_src.ml_setup_model.mobilenet import MobileNetV2
    model = MobileNetV2(output_size=100)
    return _make_setup(model, ModelType.mobilenet_v2, dataset_cifar100(), 128)



