from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_dataset import dataset_cifar10, dataset_cifar100
from .shared_setup_util import make_setup


# ---------------------------------------------------------------------------
# CCT-7 3x1 32
# ---------------------------------------------------------------------------

def cct_7_3x1_cifar10() -> MLSetup:
    import py_src.third_party.compact_transformers.src.cct as cct
    model = cct.cct_7_3x1_32()
    return make_setup(model, ModelType.cct_7_3x1_32, dataset_cifar10(), 128)


def cct_7_3x1_cifar100() -> MLSetup:
    import py_src.third_party.compact_transformers.src.cct as cct
    model = cct.cct_7_3x1_32(num_classes=100)
    return make_setup(model, ModelType.cct_7_3x1_32, dataset_cifar100(), 128)