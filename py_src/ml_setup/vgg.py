from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_dataset import dataset_cifar10
from .shared_setup_util import make_setup

# ---------------------------------------------------------------------------
# VGG
# ---------------------------------------------------------------------------

def vgg11_bn_cifar10() -> MLSetup:
    from py_src.ml_setup_model.vgg import VGGCifar
    model = VGGCifar('VGG11', num_classes=10)
    return make_setup(model, ModelType.vgg11_bn, dataset_cifar10(), 256)


def vgg11_no_bn_cifar10() -> MLSetup:
    from py_src.ml_setup_model.vgg import VGG11_no_bn
    model = VGG11_no_bn(in_channels=3, num_classes=10)
    return make_setup(model, ModelType.vgg11_no_bn, dataset_cifar10(rescale_to_224=True), 32, has_normalization=False)
