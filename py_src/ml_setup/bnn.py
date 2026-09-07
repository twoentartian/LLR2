"""The upstream BinaryNet CIFAR-10 recipe."""


from py_src.ml_setup_dataset import DatasetSetup
from py_src.ml_setup_dataset.dataset_cifar import dataset_cifar10_bnn
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_model.bnn import VGGNet7Binary

from .ml_setup import MLSetup
from .shared_setup_util import make_setup


def bnn_cifar10(override_dataset: DatasetSetup | None = None) -> MLSetup:
    ds = dataset_cifar10_bnn() if override_dataset is None else override_dataset
    return make_setup(VGGNet7Binary(), ModelType.bnn, ds, 50)
