from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_dataset import dataset_cifar10, dataset_cifar100
from .shared_setup_util import _make_setup


# ---------------------------------------------------------------------------
# EfficientNet-B0
# ---------------------------------------------------------------------------

def efficientnet_b0_cifar10() -> MLSetup:
    from torchvision import models
    model = models.efficientnet_b0(num_classes=10)
    return _make_setup(model, ModelType.efficientnet_b0, dataset_cifar10(), 128)


def efficientnet_b0_cifar100() -> MLSetup:
    from torchvision import models
    model = models.efficientnet_b0(num_classes=100)
    return _make_setup(model, ModelType.efficientnet_b0, dataset_cifar100(), 128)


