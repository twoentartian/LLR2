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
    use_dali: bool = False,
    dali_device_id: int = 0,
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

    return _build(mt, dt, preset, device, use_dali=use_dali, dali_device_id=dali_device_id)


def _imagenet_dataset_override(dataset_type: DatasetType, preset: int, use_dali: bool, dali_device_id: int):
    if not use_dali:
        return None

    from py_src.ml_setup.imagenet_preset import preset_version
    from py_src.ml_setup_dataset import dataset_imagenet1k, dataset_imagenet100, dataset_imagenet10

    preprocessing_version = preset_version(preset)
    if dataset_type == DatasetType.imagenet1k:
        return dataset_imagenet1k(preprocessing_version, use_dali=True, dali_device_id=dali_device_id)
    if dataset_type == DatasetType.imagenet100:
        return dataset_imagenet100(preprocessing_version, use_dali=True, dali_device_id=dali_device_id)
    if dataset_type == DatasetType.imagenet10:
        return dataset_imagenet10(preprocessing_version, use_dali=True, dali_device_id=dali_device_id)
    return None


def _build(mt: ModelType, dt, preset: int, device, *, use_dali: bool = False, dali_device_id: int = 0) -> MLSetup:
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
            override = _imagenet_dataset_override(DatasetType.imagenet10, preset, use_dali, dali_device_id)
            return resnet18_imagenet10(preset=preset, override_dataset=override, use_gn=use_gn)
        elif dt == DatasetType.imagenet100:
            override = _imagenet_dataset_override(DatasetType.imagenet100, preset, use_dali, dali_device_id)
            return resnet18_imagenet100(preset=preset, override_dataset=override, use_gn=use_gn)
        elif dt == DatasetType.imagenet1k:
            override = _imagenet_dataset_override(DatasetType.imagenet1k, preset, use_dali, dali_device_id)
            return resnet18_imagenet1k(preset=preset, override_dataset=override, use_gn=use_gn)
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
            override = _imagenet_dataset_override(DatasetType.imagenet1k, preset, use_dali, dali_device_id)
            return resnet34_imagenet1k(preset, override_dataset=override)
        raise _nie(mt, dt)

    elif mt == ModelType.resnet50:
        from .resnet import resnet50_imagenet1k, resnet50_imagenet100
        if _default or dt == DatasetType.imagenet1k:
            override = _imagenet_dataset_override(DatasetType.imagenet1k, preset, use_dali, dali_device_id)
            return resnet50_imagenet1k(preset, override_dataset=override)
        elif dt == DatasetType.imagenet100:
            override = _imagenet_dataset_override(DatasetType.imagenet100, preset, use_dali, dali_device_id)
            return resnet50_imagenet100(preset, override_dataset=override)
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
            override = _imagenet_dataset_override(DatasetType.imagenet1k, preset, use_dali, dali_device_id)
            return mobilenet_v3_large_imagenet1k(preset, override_dataset=override)
        raise _nie(mt, dt)

    elif mt == ModelType.vgg11_bn:
        from .vgg import vgg11_bn_cifar10, vgg11_bn_imagenet1k
        if _default or dt == DatasetType.cifar10:
            return vgg11_bn_cifar10()
        elif dt == DatasetType.imagenet1k:
            override = _imagenet_dataset_override(DatasetType.imagenet1k, preset, use_dali, dali_device_id)
            return vgg11_bn_imagenet1k(preset, override_dataset=override)
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
            override = _imagenet_dataset_override(DatasetType.imagenet1k, preset, use_dali, dali_device_id)
            return efficientnet_v2_s_imagenet1k(preset, override_dataset=override)
        raise _nie(mt, dt)

    elif mt == ModelType.efficientnet_b1:
        from .efficientnet import efficientnet_b1_imagenet1k
        if _default or dt == DatasetType.imagenet1k:
            override = _imagenet_dataset_override(DatasetType.imagenet1k, preset, use_dali, dali_device_id)
            return efficientnet_b1_imagenet1k(preset, override_dataset=override)
        raise _nie(mt, dt)

    elif mt == ModelType.densenet121:
        from .densenet import densenet121_cifar10, densenet121_imagenet1k
        if _default or dt == DatasetType.cifar10:
            return densenet121_cifar10()
        elif dt == DatasetType.imagenet1k:
            override = _imagenet_dataset_override(DatasetType.imagenet1k, preset, use_dali, dali_device_id)
            return densenet121_imagenet1k(preset, override_dataset=override)
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

    elif mt == ModelType.regnet_y_400mf:
        from .regnet import regnet_y_400mf_imagenet1k
        if _default or dt == DatasetType.imagenet1k:
            override = _imagenet_dataset_override(DatasetType.imagenet1k, preset, use_dali, dali_device_id)
            return regnet_y_400mf_imagenet1k(preset, override_dataset=override)
        raise _nie(mt, dt)

    elif mt == ModelType.vit_b_32:
        from .vit import vit_b_32_imagenet1k
        if _default or dt == DatasetType.imagenet1k:
            if use_dali:
                from py_src.ml_setup_dataset import dataset_imagenet1k_from_pytorch_dali

                override = dataset_imagenet1k_from_pytorch_dali(dali_device_id=dali_device_id)
            else:
                override = None
            return vit_b_32_imagenet1k(preset, override_dataset=override)
        raise _nie(mt, dt)

    elif mt == ModelType.squeezenet1_1:
        from .squeezenet import squeezenet1_1_imagenet1k
        if _default or dt == DatasetType.imagenet1k:
            override = _imagenet_dataset_override(DatasetType.imagenet1k, preset, use_dali, dali_device_id)
            return squeezenet1_1_imagenet1k(preset, override_dataset=override)
        raise _nie(mt, dt)

    elif mt == ModelType.resnext50_32x4d:
        from .resnext import resnext50_32x4d_imagenet1k
        if _default or dt == DatasetType.imagenet1k:
            override = _imagenet_dataset_override(DatasetType.imagenet1k, preset, use_dali, dali_device_id)
            return resnext50_32x4d_imagenet1k(preset, override_dataset=override)
        raise _nie(mt, dt)

    elif mt == ModelType.mnasnet0_5:
        from .mnasnet import mnasnet0_5_imagenet1k
        if _default or dt == DatasetType.imagenet1k:
            override = _imagenet_dataset_override(DatasetType.imagenet1k, preset, use_dali, dali_device_id)
            return mnasnet0_5_imagenet1k(preset, override_dataset=override)
        raise _nie(mt, dt)

    elif mt == ModelType.mnasnet1_0:
        from .mnasnet import mnasnet1_0_imagenet1k
        if _default or dt == DatasetType.imagenet1k:
            override = _imagenet_dataset_override(DatasetType.imagenet1k, preset, use_dali, dali_device_id)
            return mnasnet1_0_imagenet1k(preset, override_dataset=override)
        raise _nie(mt, dt)

    elif mt == ModelType.convnext_tiny:
        from .convnext import convnext_tiny_imagenet1k
        if _default or dt == DatasetType.imagenet1k:
            override = _imagenet_dataset_override(DatasetType.imagenet1k, preset, use_dali, dali_device_id)
            return convnext_tiny_imagenet1k(preset, override_dataset=override)
        raise _nie(mt, dt)

    elif mt == ModelType.alexnet:
        from .alexnet import alexnet_imagenet1k
        if _default or dt == DatasetType.imagenet1k:
            override = _imagenet_dataset_override(DatasetType.imagenet1k, preset, use_dali, dali_device_id)
            return alexnet_imagenet1k(preset, override_dataset=override)
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
            override = _imagenet_dataset_override(DatasetType.imagenet10, preset, use_dali, dali_device_id)
            return dla46c_imagenet10(preset, override_dataset=override)
        raise _nie(mt, dt)

    elif mt == ModelType.ddpm_cifar10:
        from py_src.ml_setup.ddpm_cifar import ddpm_cifar10
        if _default or dt == DatasetType.cifar10:
            return ddpm_cifar10()
        raise _nie(mt, dt)

    elif mt == ModelType.ddpm_flowers102:
        from py_src.ml_setup.ddpm_flowers import ddpm_flowers102
        if _default or dt == DatasetType.flowers102:
            return ddpm_flowers102()
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
