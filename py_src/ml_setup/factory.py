"""Factory function to build an MLSetup from string-based config."""

from __future__ import annotations

import sys

from py_src.ml_setup_model import ModelType
from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup.ml_setup import MLSetup


def get_ml_setup_from_config(
    model_type: str,
    dataset_type: str = "default",
    preset: int = 0,
    device=None,
) -> MLSetup:
    try:
        mt = ModelType[model_type]
    except KeyError:
        print(f"Unknown model type: {model_type!r}. Valid options: {[e.name for e in ModelType]}")
        sys.exit(1)

    try:
        dt = DatasetType[dataset_type] if dataset_type != "default" else None
    except KeyError:
        print(f"Unknown dataset type: {dataset_type!r}. Valid options: {[e.name for e in DatasetType]}")
        sys.exit(1)

    return _build(mt, dt, preset, device)


def _build(mt: ModelType, dt, preset: int, device) -> MLSetup:
    _default = dt is None

    if mt == ModelType.lenet5:
        from .lenet import lenet5_mnist
        if _default or dt == DatasetType.mnist:
            return lenet5_mnist()
        raise _nie(mt, dt)

    elif mt == ModelType.lenet4:
        from .lenet import lenet4_mnist
        if _default or dt == DatasetType.mnist:
            return lenet4_mnist()
        raise _nie(mt, dt)

    elif mt == ModelType.lenet5_large_fc:
        from .lenet import lenet5_large_fc_mnist
        if _default or dt == DatasetType.mnist:
            return lenet5_large_fc_mnist()
        raise _nie(mt, dt)

    elif mt in (ModelType.resnet18_bn, ModelType.resnet18_gn):
        from .resnet import (
            resnet18_cifar10, resnet18_cifar100, resnet18_imagenet10,
            resnet18_imagenet100, resnet18_imagenet1k,
            resnet18_bn_imagenet1k_sam_mask_random_noise,
            resnet18_bn_imagenet1k_sam_mask_black,
        )
        use_gn = mt == ModelType.resnet18_gn
        if _default or dt == DatasetType.cifar10:
            return resnet18_cifar10(use_gn=use_gn)
        elif dt == DatasetType.cifar100:
            return resnet18_cifar100(use_gn=use_gn)
        elif dt == DatasetType.imagenet10:
            return resnet18_imagenet10(preset=preset, use_gn=use_gn)
        elif dt == DatasetType.imagenet100:
            return resnet18_imagenet100(preset=preset, use_gn=use_gn)
        elif dt == DatasetType.imagenet1k:
            return resnet18_imagenet1k(preset=preset, use_gn=use_gn)
        elif dt == DatasetType.imagenet1k_sam_mask_random_noise:
            if use_gn:
                raise _nie(mt, dt)
            return resnet18_bn_imagenet1k_sam_mask_random_noise()
        elif dt == DatasetType.imagenet1k_sam_mask_black:
            if use_gn:
                raise _nie(mt, dt)
            return resnet18_bn_imagenet1k_sam_mask_black()
        raise _nie(mt, dt)

    elif mt == ModelType.resnet34:
        from .resnet import resnet34_imagenet1k
        if _default or dt == DatasetType.imagenet1k:
            return resnet34_imagenet1k(preset)
        raise _nie(mt, dt)

    elif mt == ModelType.resnet50:
        from .resnet import resnet50_imagenet1k, resnet50_imagenet100
        if _default or dt == DatasetType.imagenet1k:
            return resnet50_imagenet1k(preset)
        elif dt == DatasetType.imagenet100:
            return resnet50_imagenet100(preset)
        raise _nie(mt, dt)

    elif mt == ModelType.mobilenet_v2:
        from .mobilenet import mobilenet_v2_cifar10, mobilenet_v2_cifar100
        if _default or dt == DatasetType.cifar10:
            return mobilenet_v2_cifar10()
        elif dt == DatasetType.cifar100:
            return mobilenet_v2_cifar100()
        raise _nie(mt, dt)

    elif mt == ModelType.mobilenet_v3_large:
        from .mobilenet import mobilenet_v3_large_imagenet1k
        if _default or dt == DatasetType.imagenet1k:
            return mobilenet_v3_large_imagenet1k(preset)
        raise _nie(mt, dt)

    elif mt == ModelType.vgg11_bn:
        from .vgg import vgg11_bn_cifar10
        if _default or dt == DatasetType.cifar10:
            return vgg11_bn_cifar10()
        raise _nie(mt, dt)

    elif mt == ModelType.vgg11_no_bn:
        from .vgg import vgg11_no_bn_cifar10
        if _default or dt == DatasetType.cifar10:
            return vgg11_no_bn_cifar10()
        raise _nie(mt, dt)

    elif mt == ModelType.simplenet:
        from .simplenet import simplenet_cifar10, simplenet_cifar100
        if _default or dt == DatasetType.cifar10:
            return simplenet_cifar10()
        elif dt == DatasetType.cifar100:
            return simplenet_cifar100()
        raise _nie(mt, dt)

    elif mt == ModelType.shufflenet_v2:
        from .shufflenet import shufflenet_v2_cifar10, shufflenet_v2_cifar100
        if _default or dt == DatasetType.cifar10:
            return shufflenet_v2_cifar10()
        elif dt == DatasetType.cifar100:
            return shufflenet_v2_cifar100()
        raise _nie(mt, dt)

    elif mt == ModelType.efficientnet_b0:
        from .efficientnet import efficientnet_b0_cifar10, efficientnet_b0_cifar100
        if _default or dt == DatasetType.cifar10:
            return efficientnet_b0_cifar10()
        elif dt == DatasetType.cifar100:
            return efficientnet_b0_cifar100()
        raise _nie(mt, dt)

    elif mt == ModelType.efficientnet_v2_s:
        from .efficientnet import efficientnet_v2_s_imagenet1k
        if _default or dt == DatasetType.imagenet1k:
            return efficientnet_v2_s_imagenet1k(preset)
        raise _nie(mt, dt)

    elif mt == ModelType.efficientnet_b1:
        from .efficientnet import efficientnet_b1_imagenet1k
        if _default or dt == DatasetType.imagenet1k:
            return efficientnet_b1_imagenet1k(preset)
        raise _nie(mt, dt)

    elif mt == ModelType.densenet121:
        from .densenet import densenet121_cifar10
        if _default or dt == DatasetType.cifar10:
            return densenet121_cifar10()
        raise _nie(mt, dt)

    elif mt == ModelType.densenet_cifar:
        from .densenet import densenet_cifar_cifar10
        if _default or dt == DatasetType.cifar10:
            return densenet_cifar_cifar10()
        raise _nie(mt, dt)

    elif mt == ModelType.regnet_x_200mf:
        from .regnet import regnet_x_200mf_cifar10, regnet_x_200mf_cifar100
        if _default or dt == DatasetType.cifar10:
            return regnet_x_200mf_cifar10()
        elif dt == DatasetType.cifar100:
            return regnet_x_200mf_cifar100()
        raise _nie(mt, dt)

    elif mt == ModelType.cct_7_3x1_32:
        from .cct import cct_7_3x1_cifar10, cct_7_3x1_cifar100
        if _default or dt == DatasetType.cifar10:
            return cct_7_3x1_cifar10()
        elif dt == DatasetType.cifar100:
            return cct_7_3x1_cifar100()
        raise _nie(mt, dt)

    elif mt == ModelType.dla_46_c:
        from .dla import dla46c_imagenet10
        if _default or dt == DatasetType.imagenet10:
            return dla46c_imagenet10(preset)
        raise _nie(mt, dt)

    elif mt == ModelType.ddpm_cifar10:
        from py_src.ml_setup.ddpm import ddpm_cifar10
        if _default or dt == DatasetType.cifar10:
            return ddpm_cifar10()
        raise _nie(mt, dt)

    elif mt == ModelType.nanoclip_default:
        from .nanoclip import nanoclip_flickr30k_default
        if _default or dt == DatasetType.flickr30k:
            return nanoclip_flickr30k_default()
        raise _nie(mt, dt)

    elif mt == ModelType.transformer_for_grokking:
        from .grokking import (
            arithmetic_addition_grokking,
            arithmetic_cubepoly_grokking,
            arithmetic_cube2_grokking,
            arithmetic_unknown_exp_grokking,
        )
        if _default or dt == DatasetType.arithmetic_addition:
            return arithmetic_addition_grokking()
        elif dt == DatasetType.arithmetic_cubepoly:
            return arithmetic_cubepoly_grokking()
        elif dt == DatasetType.arithmetic_cube2:
            return arithmetic_cube2_grokking()
        elif dt == DatasetType.arithmetic_exp_unknown:
            return arithmetic_unknown_exp_grokking()
        raise _nie(mt, dt)

    else:
        raise NotImplementedError(f"No MLSetup defined for model_type={mt.name} in LLR2 yet.")


def _nie(mt, dt):
    return NotImplementedError(f"No MLSetup for model_type={mt.name}, dataset_type={dt}")
