from functools import partial
from typing import Optional

import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_dataset import DatasetSetup, dataset_imagenet1k_from_pytorch, dataset_imagenet21k_p
from py_src.torch_vision_train import RASampler, get_mixup_cutmix
from .shared_setup_util import make_setup


def _mixup_cutmix_collate(
    batch,
    *,
    mixup_cutmix,
):
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
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        num_classes=num_classes,
        use_v2=True,
    )
    return partial(_mixup_cutmix_collate, mixup_cutmix=mixup_cutmix)


def _vit_b_32_torchvision_sampler_fn(dataset):
    return RASampler(dataset, shuffle=True, repetitions=3)


def vit_b_32_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    """ViT-B/32 on ImageNet-1k.

    The only supported recipe is the Torchvision reference recipe (preset=1),
    except that the optimizer lr / weight decay come from the local DeiT-style tuning in
    ``complete_ml_setup.py``.
    """
    return _vit_b_32_imagenet1k(preset, override_dataset, model_type=ModelType.vit_b_32)


def vit_b_32_reg4_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    """ViT-B/32 with four registers and the same recipe as the baseline."""
    return _vit_b_32_imagenet1k(preset, override_dataset, model_type=ModelType.vit_b_32_reg4)


def vit_b_32_rope_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    """ViT-B/32 with 2D RoPE-Mixed and the baseline training recipe."""
    return _vit_b_32_imagenet1k(preset, override_dataset, model_type=ModelType.vit_b_32_rope)


def vit_b_32_swiglu_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    """ViT-B/32 with SwiGLU FFNs and the baseline training recipe."""
    return _vit_b_32_imagenet1k(preset, override_dataset, model_type=ModelType.vit_b_32_swiglu)


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
    """Use the existing ViT recipe with a classifier matching the 21k-p folders."""
    if preset not in (0, 1):
        raise ValueError(f"vit_b_32 only supports the Torchvision recipe (preset=1), got preset={preset}")
    if override_dataset is not None:
        ds = override_dataset
    elif use_dali:
        ds = dataset_imagenet21k_p(
            preset_version=1, train_crop_size=224, val_resize_size=256, val_crop_size=224,
            use_dali=True, dali_device_id=dali_device_id,
            auto_augment_policy="imagenet", repeated_augmentation_repetitions=3,
        )
    else:
        from py_src.torch_vision_train.presets import ClassificationPresetTrain, ClassificationPresetEval

        ds = dataset_imagenet21k_p(
            transforms_training=ClassificationPresetTrain(
                crop_size=224, interpolation=InterpolationMode.BILINEAR,
                auto_augment_policy="imagenet", random_erase_prob=0.0, use_v2=True,
            ),
            transforms_testing=ClassificationPresetEval(crop_size=224, resize_size=256),
        )
    classes = getattr(ds.train_data, "classes", None)
    validation_classes = getattr(ds.valdation_data, "classes", None)
    if classes is not None and validation_classes is not None and classes != validation_classes:
        raise ValueError("ImageNet-21k train/val class mappings differ")
    num_classes = len(classes) if classes is not None else 10450
    if num_classes < 1:
        raise ValueError("ImageNet-21k requires at least one class")
    return _vit_b_32_setup(ds, model_type=model_type, num_classes=num_classes)


def _vit_b_32_imagenet1k(preset, override_dataset, *, model_type: ModelType) -> MLSetup:
    if preset not in (0, 1):
        raise ValueError(f"vit_b_32 only supports the Torchvision recipe (preset=1), got preset={preset}")
    ds = override_dataset if override_dataset is not None else dataset_imagenet1k_from_pytorch(
        train_crop_size=224,
        val_resize_size=256,
        val_crop_size=224,
        interpolation=InterpolationMode.BILINEAR,
        auto_augment_policy="imagenet",
        random_erase_prob=0.0,
        use_v2=True,
    )
    return _vit_b_32_setup(ds, model_type=model_type, num_classes=1000)


def _vit_b_32_setup(ds: DatasetSetup, *, model_type: ModelType, num_classes: int) -> MLSetup:
    from torchvision import models

    criterion = nn.CrossEntropyLoss(label_smoothing=0.11)
    collate_fn = _vit_b_32_torchvision_collate_fn(num_classes)
    sampler_fn = _vit_b_32_torchvision_sampler_fn

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
