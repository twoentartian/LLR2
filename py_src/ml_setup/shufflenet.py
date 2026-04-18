from typing import Optional

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_dataset import DatasetSetup, dataset_cifar10, dataset_cifar100

from .shared_setup_util import make_setup

# ---------------------------------------------------------------------------
# ShuffleNet-V2 (CIFAR-optimised custom implementation)
# ---------------------------------------------------------------------------

def shufflenet_v2_cifar10(override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    from py_src.ml_setup_model.shufflenet import ShuffleNet
    ds = dataset_cifar10() if override_dataset is None else override_dataset
    model = ShuffleNet(output_size=10, g=1, scale_factor=1)
    return make_setup(model, ModelType.shufflenet_v2, ds, 128)


def shufflenet_v2_cifar100(override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    from py_src.ml_setup_model.shufflenet import ShuffleNet
    ds = dataset_cifar100() if override_dataset is None else override_dataset
    model = ShuffleNet(output_size=100, g=1, scale_factor=1)
    return make_setup(model, ModelType.shufflenet_v2, ds, 128)


