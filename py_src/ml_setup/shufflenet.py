from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_dataset import dataset_cifar10, dataset_cifar100

from .shared_setup_util import make_setup

# ---------------------------------------------------------------------------
# ShuffleNet-V2 (CIFAR-optimised custom implementation)
# ---------------------------------------------------------------------------

def shufflenet_v2_cifar10() -> MLSetup:
    from py_src.ml_setup_model.shufflenet import ShuffleNet
    model = ShuffleNet(output_size=10, g=1, scale_factor=1)
    return make_setup(model, ModelType.shufflenet_v2, dataset_cifar10(), 128)


def shufflenet_v2_cifar100() -> MLSetup:
    from py_src.ml_setup_model.shufflenet import ShuffleNet
    model = ShuffleNet(output_size=100, g=1, scale_factor=1)
    return make_setup(model, ModelType.shufflenet_v2, dataset_cifar100(), 128)



