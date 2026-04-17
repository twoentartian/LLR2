import torch.nn as nn

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_dataset import dataset_cifar10, dataset_cifar100
from .shared_setup_util import _make_setup

# ---------------------------------------------------------------------------
# ResNet-18
# ---------------------------------------------------------------------------

class GroupNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=2, num_channels=num_channels, eps=1e-5, affine=True)

    def forward(self, x):
        return self.norm(x)

def _resnet18_cifar(num_classes, dataset_setup, model_type, use_gn=False) -> MLSetup:
    from torchvision import models
    norm = GroupNorm if use_gn else nn.BatchNorm2d
    model = models.resnet18(num_classes=num_classes, norm_layer=norm)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity() # type: ignore
    return _make_setup(model, model_type, dataset_setup, 256)


def resnet18_bn_cifar10() -> MLSetup:
    return _resnet18_cifar(10, dataset_cifar10(), ModelType.resnet18_bn, False)


def resnet18_gn_cifar10() -> MLSetup:
    return _resnet18_cifar(10, dataset_cifar10(), ModelType.resnet18_gn, True)


def resnet18_bn_cifar100() -> MLSetup:
    return _resnet18_cifar(100, dataset_cifar100(), ModelType.resnet18_bn, False)


def resnet18_gn_cifar100() -> MLSetup:
    return _resnet18_cifar(100, dataset_cifar100(), ModelType.resnet18_gn, True)


