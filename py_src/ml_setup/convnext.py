"""ConvNeXt model setups; the 21k-p preset 4 follows the original 22k recipe.

Source: https://arxiv.org/html/2201.03545v2#A1.SS1 (appendix A.1, table 5),
https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/main.py
Preset 3 is LLR2's historical ViT control, not a ConvNeXt paper recipe.
See ../ml_setup_dataset/README_imagenet_augmentation.md sections 1.1/1.2 and 6
for every augmentation's source and CPU/DALI adaptations.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_dataset import DatasetSetup, DatasetType, dataset_imagenet1k, dataset_imagenet21k_p
from .shared_setup_util import make_setup
from .imagenet_preset import preset_version, imagenet_criterion, imagenet_collate_fn, imagenet_sampler_fn


CONVNEXT_RECIPE_VERSION = 1
CONVNEXT_OFFICIAL_PRESET = 4


def initialize_convnext_weights(model: nn.Module) -> None:
    """Keep ConvNeXt's truncated-normal initialization during worker re-init."""
    from torchvision.models.convnext import CNBlock

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, CNBlock):
                module.layer_scale.fill_(1e-6)


class ConvNeXtMixupCollate:
    """ConvNeXt reference coefficients, timm batch pairing and target smoothing.

    Implementation: https://github.com/huggingface/pytorch-image-models/blob/v1.0.29/timm/data/mixup.py
    Method papers: Mixup arxiv:1710.09412, CutMix arxiv:1905.04899,
    label smoothing arxiv:1512.00567. The odd-tail handling below is an LLR2
    adaptation to timm's even-batch requirement, not part of the paper recipe.
    """

    def __init__(self, num_classes: int):
        from timm.data import Mixup

        self.num_classes = num_classes
        self.mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0,
                           switch_prob=0.5, mode="batch", label_smoothing=0.1,
                           num_classes=num_classes)

    def __call__(self, batch):
        images, targets = batch if isinstance(batch, tuple) and len(batch) == 2 else default_collate(batch)
        targets = targets.to(images.device)
        if len(images) % 2 == 0:
            return self.mixup(images, targets)
        # timm requires even batches. Keep every tail sample: mix the even
        # prefix and smooth the unpaired final target without mixing its image.
        from timm.data.mixup import mixup_target

        final_target = mixup_target(targets[-1:], self.num_classes, lam=1.0, smoothing=0.1)
        if len(images) == 1:
            return images, final_target
        mixed_images, mixed_targets = self.mixup(images[:-1], targets[:-1])
        return torch.cat((mixed_images, images[-1:])), torch.cat((mixed_targets, final_target))


def convnext_tiny_imagenet21k_p(
    preset: int = CONVNEXT_OFFICIAL_PRESET, override_dataset: Optional[DatasetSetup] = None,
    *, use_dali: bool = False, dali_device_id: int = 0,
) -> MLSetup:
    """Random-init ConvNeXt-Tiny: 0/1/2 project recipes, 3 legacy ViT, 4 ConvNeXt."""
    from torchvision import models
    from .vit import (_vit_dataset, _vit_b_32_torchvision_collate_fn,
                      _vit_b_32_torchvision_sampler_fn)

    if preset not in (0, 1, 2, 3, 4):
        raise ValueError(f"ConvNeXt/ImageNet-21k-p preset must be 0, 1, 2, 3 or 4, got {preset}")
    if override_dataset is not None:
        ds = override_dataset
    elif preset == CONVNEXT_OFFICIAL_PRESET:
        from py_src.ml_setup_dataset.convnext_augmentation import convnext_transforms

        if use_dali:
            # The official ConvNeXt loader has no repeated-augmentation sampler;
            # one dataset pass per epoch, unlike LLR2's legacy DALI v3 RA3.
            ds = dataset_imagenet21k_p(
                use_dali=True, dali_device_id=dali_device_id, preset_version=1,
                train_crop_size=224, val_resize_size=256, val_crop_size=224,
                random_erasing=0.25, repeated_augmentation_repetitions=1,
                augmentation_recipe="convnext_v1",
            )
        else:
            train_transform, val_transform = convnext_transforms()
            ds = dataset_imagenet21k_p(transforms_training=train_transform,
                                       transforms_testing=val_transform)
    else:
        # Reuse the exact existing image pipeline, collate and sampler for v3.
        ds = _vit_dataset(DatasetType.imagenet21k_p, preset, use_dali, dali_device_id)
    classes = getattr(ds.train_data, "classes", None)
    validation_classes = getattr(ds.valdation_data, "classes", None)
    if classes is not None and validation_classes is not None and classes != validation_classes:
        raise ValueError("ImageNet-21k train/val class mappings differ")
    num_classes = len(classes) if classes is not None else 10450
    if num_classes < 1:
        raise ValueError("ImageNet-21k requires at least one class")
    model = models.convnext_tiny(weights=None, progress=False, num_classes=num_classes,
                                 stochastic_depth_prob=0.0)
    if preset == CONVNEXT_OFFICIAL_PRESET:
        # timm's Mixup includes smoothing; hard-label validation uses plain CE.
        criterion = nn.CrossEntropyLoss()
        collate_fn, sampler_fn = ConvNeXtMixupCollate(num_classes), None
    elif preset == 3:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.11)
        collate_fn = _vit_b_32_torchvision_collate_fn(num_classes)
        sampler_fn = _vit_b_32_torchvision_sampler_fn
    else:
        criterion = imagenet_criterion(preset) if preset else nn.CrossEntropyLoss()
        collate_fn = imagenet_collate_fn(preset, num_classes=num_classes) if preset else None
        sampler_fn = imagenet_sampler_fn(preset) if preset else None
    return make_setup(
        model, ModelType.convnext_tiny, ds, 512, criterion=criterion,
        default_collate_fn=collate_fn, default_collate_fn_val=default_collate,
        default_sampler_fn=sampler_fn, gradient_accumulate_every=8,
        default_prefetch_factor=2, reinitialize_model_fn=initialize_convnext_weights,
    )


def conveNeXt_tiny_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    """ConvNeXt-Tiny on ImageNet-1k, ported from DFL_torch.

    The function name preserves the original DFL_torch capitalization typo for compatibility.
    """
    from torchvision import models

    pv = preset_version(preset)
    ds = override_dataset if override_dataset is not None else dataset_imagenet1k(
        preset_version=pv,
        train_crop_size=176,
        val_resize_size=232,
        val_crop_size=224,
    )
    model = models.convnext_tiny(progress=False, weights=None, num_classes=1000)
    return make_setup(
        model,
        ModelType.convnext_tiny,
        ds,
        128,
        criterion=imagenet_criterion(pv),
        default_collate_fn=imagenet_collate_fn(pv),
        default_collate_fn_val=default_collate,
        default_sampler_fn=imagenet_sampler_fn(pv),
    )


def convnext_tiny_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    return conveNeXt_tiny_imagenet1k(preset, override_dataset)
