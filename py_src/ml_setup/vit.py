from functools import partial
from typing import Optional

import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_dataset import (
    DatasetSetup, DatasetType, dataset_imagenet1k_from_pytorch, dataset_imagenet21k_p,
    get_imagenet_preprocessing,
)
from .imagenet_preset import imagenet_criterion, imagenet_collate_fn, imagenet_sampler_fn
from .shared_setup_util import make_setup
from py_src.torch_vision_train import RASampler, get_mixup_cutmix

# Version the corrected preset meanings; adding legacy preset 3 preserves 0/1/2.
VIT_PRESET_VERSION = 2
VIT_LEGACY_PRESET = 3


def _mixup_cutmix_collate(batch, *, mixup_cutmix):
    """Preserve the historical ViT collate path for preset 3."""
    if isinstance(batch, tuple) and len(batch) == 2:
        images, targets = batch
    else:
        images, targets = default_collate(batch)
    images, targets = mixup_cutmix(images, targets)
    if hasattr(images, "device") and hasattr(targets, "to") and targets.device != images.device:
        targets = targets.to(images.device, non_blocking=True)
    return images, targets


def _vit_b_32_torchvision_collate_fn(num_classes: int = 1000):
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=0.2, cutmix_alpha=1.0, num_classes=num_classes, use_v2=True,
    )
    return partial(_mixup_cutmix_collate, mixup_cutmix=mixup_cutmix)


def _vit_b_32_torchvision_sampler_fn(dataset):
    return RASampler(dataset, shuffle=True, repetitions=3)


def vit_b_32_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None,
                       *, use_dali: bool = False, dali_device_id: int = 0) -> MLSetup:
    """ViT-B/32: 0 disables augmentation; 1/2 use project v1/v2; 3 restores legacy ViT."""
    return _vit_b_32_imagenet1k(preset, override_dataset, model_type=ModelType.vit_b_32,
                              use_dali=use_dali, dali_device_id=dali_device_id)


def vit_b_32_reg4_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None,
                            *, use_dali: bool = False, dali_device_id: int = 0) -> MLSetup:
    """ViT-B/32 with four registers and the same recipe as the baseline."""
    return _vit_b_32_imagenet1k(preset, override_dataset, model_type=ModelType.vit_b_32_reg4,
                              use_dali=use_dali, dali_device_id=dali_device_id)


def vit_b_32_rope_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None,
                            *, use_dali: bool = False, dali_device_id: int = 0) -> MLSetup:
    """ViT-B/32 with 2D RoPE-Mixed and the baseline training recipe."""
    return _vit_b_32_imagenet1k(preset, override_dataset, model_type=ModelType.vit_b_32_rope,
                              use_dali=use_dali, dali_device_id=dali_device_id)


def vit_b_32_swiglu_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None,
                              *, use_dali: bool = False, dali_device_id: int = 0) -> MLSetup:
    """ViT-B/32 with SwiGLU FFNs and the baseline training recipe."""
    return _vit_b_32_imagenet1k(preset, override_dataset, model_type=ModelType.vit_b_32_swiglu,
                              use_dali=use_dali, dali_device_id=dali_device_id)


def vit_b_32_imagenet21k_p(preset: int = 1, override_dataset: Optional[DatasetSetup] = None,
                          *, use_dali: bool = False, dali_device_id: int = 0) -> MLSetup:
    """ViT-B/32 for the existing ImageNet-21k-p folder layout."""
    return _vit_b_32_imagenet21k_p(preset, override_dataset, model_type=ModelType.vit_b_32,
                                 use_dali=use_dali, dali_device_id=dali_device_id)


def vit_b_32_reg4_imagenet21k_p(preset: int = 1, override_dataset: Optional[DatasetSetup] = None,
                               *, use_dali: bool = False, dali_device_id: int = 0) -> MLSetup:
    """Four-register ViT with the same 21k-p recipe as the baseline."""
    return _vit_b_32_imagenet21k_p(preset, override_dataset, model_type=ModelType.vit_b_32_reg4,
                                 use_dali=use_dali, dali_device_id=dali_device_id)


def vit_b_32_rope_imagenet21k_p(preset: int = 1, override_dataset: Optional[DatasetSetup] = None,
                               *, use_dali: bool = False, dali_device_id: int = 0) -> MLSetup:
    """RoPE ViT with the same 21k-p recipe as the baseline."""
    return _vit_b_32_imagenet21k_p(preset, override_dataset, model_type=ModelType.vit_b_32_rope,
                                 use_dali=use_dali, dali_device_id=dali_device_id)


def vit_b_32_swiglu_imagenet21k_p(preset: int = 1, override_dataset: Optional[DatasetSetup] = None,
                                 *, use_dali: bool = False, dali_device_id: int = 0) -> MLSetup:
    """SwiGLU ViT with the same 21k-p recipe as the baseline."""
    return _vit_b_32_imagenet21k_p(preset, override_dataset, model_type=ModelType.vit_b_32_swiglu,
                                 use_dali=use_dali, dali_device_id=dali_device_id)


def _vit_b_32_imagenet21k_p(preset, override_dataset, *, model_type, use_dali, dali_device_id):
    _validate_preset(preset)
    ds = override_dataset if override_dataset is not None else _vit_dataset(
        DatasetType.imagenet21k_p, preset, use_dali, dali_device_id,
    )
    classes = getattr(ds.train_data, "classes", None)
    validation_classes = getattr(ds.valdation_data, "classes", None)
    if classes is not None and validation_classes is not None and classes != validation_classes:
        raise ValueError("ImageNet-21k train/val class mappings differ")
    num_classes = len(classes) if classes is not None else 10450
    if num_classes < 1:
        raise ValueError("ImageNet-21k requires at least one class")
    return _vit_b_32_setup(ds, preset=preset, model_type=model_type, num_classes=num_classes)


def _vit_b_32_imagenet1k(preset, override_dataset, *, model_type, use_dali, dali_device_id) -> MLSetup:
    _validate_preset(preset)
    ds = override_dataset if override_dataset is not None else _vit_dataset(
        DatasetType.imagenet1k, preset, use_dali, dali_device_id,
    )
    return _vit_b_32_setup(ds, preset=preset, model_type=model_type, num_classes=1000)


def _validate_preset(preset):
    if preset not in (0, 1, 2, 3):
        raise ValueError(f"ViT preset must be 0 (no augmentation), 1 (ImageNet v1), "
                         f"2 (ImageNet v2), or 3 (legacy ViT), got {preset}")


def _vit_dataset(dataset_type, preset, use_dali, dali_device_id):
    if preset == VIT_LEGACY_PRESET:
        return _vit_v3_dataset(dataset_type, use_dali, dali_device_id)
    version = max(1, preset)
    sizes = dict(train_crop_size=224, val_resize_size=256, val_crop_size=224)
    if dataset_type == DatasetType.imagenet21k_p:
        return dataset_imagenet21k_p(
            preset_version=version, augmentation=preset != 0, **sizes,
            use_dali=use_dali, dali_device_id=dali_device_id,
        )
    if use_dali:
        from py_src.ml_setup_dataset import dataset_imagenet1k_from_pytorch_dali

        return dataset_imagenet1k_from_pytorch_dali(
            preset_version=version, augmentation=preset != 0, **sizes,
            auto_augment_policy=None, random_erase_prob=0.1 if preset == 2 else 0.0,
            ra_repetitions=4 if preset == 2 else 1, dali_device_id=dali_device_id,
        )
    train_transform, val_transform = get_imagenet_preprocessing(version, **sizes, augmentation=preset != 0)
    return dataset_imagenet1k_from_pytorch(
        transforms_training=train_transform, transforms_testing=val_transform,
    )


def _vit_v3_dataset(dataset_type, use_dali, dali_device_id):
    """The pre-fix ViT transforms, including its original transforms API choices."""
    sizes = dict(train_crop_size=224, val_resize_size=256, val_crop_size=224)
    if use_dali:
        if dataset_type == DatasetType.imagenet21k_p:
            return dataset_imagenet21k_p(
                preset_version=1, **sizes, use_dali=True, dali_device_id=dali_device_id,
                augmentation=True, random_erasing=0.0,
                auto_augment_policy="imagenet", repeated_augmentation_repetitions=3,
            )
        from py_src.ml_setup_dataset import dataset_imagenet1k_from_pytorch_dali

        return dataset_imagenet1k_from_pytorch_dali(
            **sizes, interpolation=InterpolationMode.BILINEAR, auto_augment_policy="imagenet",
            random_erase_prob=0.0, ra_repetitions=3, dali_device_id=dali_device_id,
            preset_version=1, augmentation=True,
        )
    if dataset_type == DatasetType.imagenet21k_p:
        from py_src.torch_vision_train.presets import ClassificationPresetTrain, ClassificationPresetEval

        return dataset_imagenet21k_p(
            transforms_training=ClassificationPresetTrain(
                crop_size=224, interpolation=InterpolationMode.BILINEAR,
                auto_augment_policy="imagenet", random_erase_prob=0.0, use_v2=True,
            ),
            transforms_testing=ClassificationPresetEval(crop_size=224, resize_size=256),
        )
    return dataset_imagenet1k_from_pytorch(
        **sizes, interpolation=InterpolationMode.BILINEAR,
        auto_augment_policy="imagenet", random_erase_prob=0.0, use_v2=True,
    )


def _vit_b_32_setup(ds: DatasetSetup, *, preset: int, model_type: ModelType, num_classes: int) -> MLSetup:
    from torchvision import models

    if preset == VIT_LEGACY_PRESET:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.11)
        collate_fn = _vit_b_32_torchvision_collate_fn(num_classes)
        sampler_fn = _vit_b_32_torchvision_sampler_fn
    else:
        criterion = imagenet_criterion(preset) if preset else nn.CrossEntropyLoss()
        collate_fn = imagenet_collate_fn(preset, num_classes=num_classes) if preset else None
        sampler_fn = imagenet_sampler_fn(preset) if preset else None

    if model_type == ModelType.vit_b_32_reg4:
        from py_src.ml_setup_model.vit_registers import vit_b_32_reg4

        model = vit_b_32_reg4(num_classes=num_classes)
    elif model_type == ModelType.vit_b_32_rope:
        from py_src.ml_setup_model.vit_rope import vit_b_32_rope

        model = vit_b_32_rope(num_classes=num_classes)
    elif model_type == ModelType.vit_b_32_swiglu:
        from py_src.ml_setup_model.vit_swiglu import vit_b_32_swiglu

        model = vit_b_32_swiglu(num_classes=num_classes)
    else:
        model = models.vit_b_32(progress=False, weights=None, num_classes=num_classes)
    return make_setup(
        model,
        model_type,
        ds,
        512,
        criterion=criterion,
        clip_grad_norm=1,
        default_collate_fn=collate_fn,
        default_collate_fn_val=default_collate,
        default_sampler_fn=sampler_fn,
        default_prefetch_factor=16,
    )
