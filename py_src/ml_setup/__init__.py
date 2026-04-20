from .factory import get_ml_setup_from_config
from .ml_setup import MLSetup, ApplicationType

from .alexnet import alexnet_imagenet1k
from .densenet import densenet121_cifar10, densenet_cifar_cifar10
from .dla import dla_cifar10, dla_cifar100, dla46c_imagenet10
from .efficientnet import efficientnet_b0_cifar10, efficientnet_b0_cifar100, efficientnet_v2_s_imagenet1k, efficientnet_b1_imagenet1k
from .lenet import lenet4_mnist, lenet5_large_fc_mnist, lenet5_mnist
from .mobilenet import mobilenet_v2_cifar10, mobilenet_v2_cifar100, mobilenet_v3_large_imagenet1k
from .regnet import regnet_x_200mf_cifar10, regnet_x_200mf_cifar100
from .resnet import resnet18_cifar10, resnet18_cifar100
from .resnet import resnet18_imagenet10, resnet18_imagenet100, resnet18_imagenet1k, resnet34_imagenet1k, resnet50_imagenet100, resnet50_imagenet1k
from .shufflenet import shufflenet_v2_cifar10, shufflenet_v2_cifar100
from .simplenet import simplenet_cifar10, simplenet_cifar100
from .vgg import vgg11_bn_cifar10, vgg11_no_bn_cifar10

from .cct import cct_7_3x1_cifar10, cct_7_3x1_cifar100, cct14_7x2_imagenet1k

from .ddpm import ddpm_cifar10
from .grokking import (
    arithmetic_addition_grokking,
    arithmetic_cubepoly_grokking,
    arithmetic_cube2_grokking,
    arithmetic_unknown_exp_grokking,
)
from .nanoclip import nanoclip_flickr30k_default

__all__ = [
    "get_ml_setup_from_config", "MLSetup", "ApplicationType",
    "alexnet_imagenet1k",
    "densenet121_cifar10", "densenet_cifar_cifar10",
    "dla_cifar10", "dla_cifar100", "dla46c_imagenet10",
    "efficientnet_b0_cifar10", "efficientnet_b0_cifar100", "efficientnet_v2_s_imagenet1k", "efficientnet_b1_imagenet1k",
    "lenet4_mnist", "lenet5_large_fc_mnist", "lenet5_mnist",
    "mobilenet_v2_cifar10", "mobilenet_v2_cifar100", "mobilenet_v3_large_imagenet1k",
    "regnet_x_200mf_cifar10", "regnet_x_200mf_cifar100",
    "resnet18_cifar10", "resnet18_cifar100",
    "resnet18_imagenet10", "resnet18_imagenet100", "resnet18_imagenet1k", "resnet34_imagenet1k", "resnet50_imagenet100", "resnet50_imagenet1k",
    "shufflenet_v2_cifar10", "shufflenet_v2_cifar100",
    "simplenet_cifar10", "simplenet_cifar100",
    "vgg11_bn_cifar10", "vgg11_no_bn_cifar10",
    "cct_7_3x1_cifar10", "cct_7_3x1_cifar100", "cct14_7x2_imagenet1k",
    "ddpm_cifar10",
    "arithmetic_addition_grokking",
    "arithmetic_cubepoly_grokking",
    "arithmetic_cube2_grokking",
    "arithmetic_unknown_exp_grokking",
    "nanoclip_flickr30k_default",
    ]
