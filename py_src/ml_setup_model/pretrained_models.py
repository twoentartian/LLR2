from __future__ import annotations

import ssl
from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn
import torchvision.models as models

import py_src.third_party.compact_transformers.src.cct as cct


@dataclass(frozen=True)
class PretrainedModelExportSpec:
    file_name: str
    model_type: str
    variant: str


_PRETRAINED_BUILDERS: dict[str, dict[str, Callable[[], nn.Module]]] = {
    "resnet18_bn": {
        "imagenet1k_v1": lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
    },
    "resnet34": {
        "imagenet1k_v1": lambda: models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1),
    },
    "resnet50": {
        "imagenet1k_v1": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
        "imagenet1k_v2": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
    },
    "vgg11_bn": {
        "imagenet1k_v1": lambda: models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1),
    },
    "squeezenet1_1": {
        "imagenet1k_v1": lambda: models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1),
    },
    "shufflenet_v2_x2_0": {
        "imagenet1k_v1": lambda: models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1),
    },
    "mobilenet_v3_large": {
        "imagenet1k_v1": lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1),
        "imagenet1k_v2": lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    },
    "efficientnet_v2_s": {
        "imagenet1k_v1": lambda: models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1),
    },
    "efficientnet_b1": {
        "imagenet1k_v1": lambda: models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1),
        "imagenet1k_v2": lambda: models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V2),
    },
    "mnasnet1_0": {
        "imagenet1k_v1": lambda: models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1),
    },
    "mnasnet0_5": {
        "imagenet1k_v1": lambda: models.mnasnet0_5(weights=models.MNASNet0_5_Weights.IMAGENET1K_V1),
    },
    "densenet121": {
        "imagenet1k_v1": lambda: models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1),
    },
    "regnet_y_400mf": {
        "imagenet1k_v1": lambda: models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V1),
        "imagenet1k_v2": lambda: models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V2),
    },
    "convnext_tiny": {
        "imagenet1k_v1": lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    },
    "alexnet": {
        "imagenet1k_v1": lambda: models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1),
    },
    "resnext50_32x4d": {
        "imagenet1k_v1": lambda: models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1),
        "imagenet1k_v2": lambda: models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2),
    },
    "vit_b_32": {
        "imagenet1k_v1": lambda: models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1),
    },
    "wide_resnet50_2": {
        "imagenet1k_v1": lambda: models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1),
        "imagenet1k_v2": lambda: models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2),
    },
    "cct_14_7x2_224": {
        "imagenet1k_v1": lambda: cct.cct_14_7x2_224(pretrained=True, progress=False),
    },
}

_DEFAULT_PRETRAINED_VARIANTS: dict[str, str] = {
    "resnet18_bn": "imagenet1k_v1",
    "resnet34": "imagenet1k_v1",
    "resnet50": "imagenet1k_v2",
    "vgg11_bn": "imagenet1k_v1",
    "squeezenet1_1": "imagenet1k_v1",
    "shufflenet_v2_x2_0": "imagenet1k_v1",
    "mobilenet_v3_large": "imagenet1k_v2",
    "efficientnet_v2_s": "imagenet1k_v1",
    "efficientnet_b1": "imagenet1k_v2",
    "mnasnet1_0": "imagenet1k_v1",
    "mnasnet0_5": "imagenet1k_v1",
    "densenet121": "imagenet1k_v1",
    "regnet_y_400mf": "imagenet1k_v2",
    "convnext_tiny": "imagenet1k_v1",
    "alexnet": "imagenet1k_v1",
    "resnext50_32x4d": "imagenet1k_v2",
    "vit_b_32": "imagenet1k_v1",
    "wide_resnet50_2": "imagenet1k_v2",
    "cct_14_7x2_224": "imagenet1k_v1",
}

_PRETRAINED_EXPORT_SPECS: tuple[PretrainedModelExportSpec, ...] = (
    PretrainedModelExportSpec("resnet18_imagenet1k.model.pt", "resnet18_bn", "imagenet1k_v1"),
    PretrainedModelExportSpec("resnet34_imagenet1k.model.pt", "resnet34", "imagenet1k_v1"),
    PretrainedModelExportSpec("resnet50_imagenet1k_v1.model.pt", "resnet50", "imagenet1k_v1"),
    PretrainedModelExportSpec("resnet50_imagenet1k_v2.model.pt", "resnet50", "imagenet1k_v2"),
    PretrainedModelExportSpec("vgg11_bn_imagenet1k_v1.model.pt", "vgg11_bn", "imagenet1k_v1"),
    PretrainedModelExportSpec("squeezenet1_1.model.pt", "squeezenet1_1", "imagenet1k_v1"),
    PretrainedModelExportSpec("shufflenet_v2_x2_0.model.pt", "shufflenet_v2_x2_0", "imagenet1k_v1"),
    PretrainedModelExportSpec("mobilenet_v3_large_imagenet_v1.model.pt", "mobilenet_v3_large", "imagenet1k_v1"),
    PretrainedModelExportSpec("mobilenet_v3_large_imagenet_v2.model.pt", "mobilenet_v3_large", "imagenet1k_v2"),
    PretrainedModelExportSpec("efficientnet_v2_s.model.pt", "efficientnet_v2_s", "imagenet1k_v1"),
    PretrainedModelExportSpec("efficientnet_b1_imagenet_v1.model.pt", "efficientnet_b1", "imagenet1k_v1"),
    PretrainedModelExportSpec("efficientnet_b1_imagenet_v2.model.pt", "efficientnet_b1", "imagenet1k_v2"),
    PretrainedModelExportSpec("mnasnet1_0_imagenet_v1.model.pt", "mnasnet1_0", "imagenet1k_v1"),
    PretrainedModelExportSpec("mnasnet0_5_imagenet_v1.model.pt", "mnasnet0_5", "imagenet1k_v1"),
    PretrainedModelExportSpec("densenet121_imagenet_v1.model.pt", "densenet121", "imagenet1k_v1"),
    PretrainedModelExportSpec("regnet_y_400_mf_imagenet_v1.model.pt", "regnet_y_400mf", "imagenet1k_v1"),
    PretrainedModelExportSpec("regnet_y_400_mf_imagenet_v2.model.pt", "regnet_y_400mf", "imagenet1k_v2"),
    PretrainedModelExportSpec("convnext_tiny_imagenet_v1.model.pt", "convnext_tiny", "imagenet1k_v1"),
    PretrainedModelExportSpec("alexnet_imagenet.model.pt", "alexnet", "imagenet1k_v1"),
    PretrainedModelExportSpec("resnext50_32x4d_imagenet_v1.model.pt", "resnext50_32x4d", "imagenet1k_v1"),
    PretrainedModelExportSpec("resnext50_32x4d_imagenet_v2.model.pt", "resnext50_32x4d", "imagenet1k_v2"),
    PretrainedModelExportSpec("vit_b_32_imagenet_v1.model.pt", "vit_b_32", "imagenet1k_v1"),
    PretrainedModelExportSpec("wide_resnet50_2_imagenet_v1.model.pt", "wide_resnet50_2", "imagenet1k_v1"),
    PretrainedModelExportSpec("wide_resnet50_2_imagenet_v2.model.pt", "wide_resnet50_2", "imagenet1k_v2"),
    PretrainedModelExportSpec("cct_14_7x2_224.model.pt", "cct_14_7x2_224", "imagenet1k_v1"),
)


def create_torchvision_model(model_type: str, variant: str = "default") -> nn.Module:
    ssl._create_default_https_context = ssl._create_unverified_context
    variants = _PRETRAINED_BUILDERS.get(model_type)
    if variants is None:
        supported = ", ".join(sorted(_PRETRAINED_BUILDERS))
        raise ValueError(
            f"torchvision pretrained loading is not configured for model type {model_type!r}. "
            f"Supported values: {supported}"
        )

    resolved_variant = _DEFAULT_PRETRAINED_VARIANTS.get(model_type) if variant == "default" else variant
    if resolved_variant is None:
        supported = ", ".join(sorted(variants))
        raise ValueError(
            f"no default pretrained variant is configured for model type {model_type!r}. "
            f"Supported variants: {supported}"
        )

    builder = variants.get(resolved_variant)
    if builder is None:
        supported = ", ".join(sorted(variants))
        raise ValueError(
            f"pretrained variant {resolved_variant!r} is not configured for model type {model_type!r}. "
            f"Supported variants: {supported}"
        )
    return builder()


def get_torchvision_pretrained_export_specs() -> list[PretrainedModelExportSpec]:
    return list(_PRETRAINED_EXPORT_SPECS)


def get_supported_torchvision_pretrained_model_types() -> list[str]:
    return sorted(_PRETRAINED_BUILDERS)


def get_supported_torchvision_pretrained_variants(model_type: str) -> list[str]:
    variants = _PRETRAINED_BUILDERS.get(model_type)
    if variants is None:
        supported = ", ".join(sorted(_PRETRAINED_BUILDERS))
        raise ValueError(
            f"torchvision pretrained loading is not configured for model type {model_type!r}. "
            f"Supported values: {supported}"
        )
    return sorted(variants)
