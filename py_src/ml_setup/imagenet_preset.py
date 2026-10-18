"""Shared ImageNet criterion, batch augmentation and sampler presets.

v1 uses torchvision's basic classification recipe; v2 takes Mixup/CutMix
0.2/1.0, label smoothing 0.1 and RA4 from its improved training recipe:
https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
LLR2's v2 image pipeline additionally chains TA Wide and RandAugment; it is not
an exact copy of that recipe. See ../ml_setup_dataset/README_imagenet_augmentation.md
sections 1.1/1.2 for per-operation sources, v3/v4, and backend epoch budgets.
"""

from typing import Callable, Optional

import torch.nn as nn
from torch.utils.data.dataloader import default_collate

from py_src.torch_vision_train import RASampler, get_mixup_cutmix

def preset_version(preset: int) -> int:
    """Map LLR2 preset index to imagenet preprocessing version.

    preset=1 → preprocessing v1 (basic)
    preset!=1 → preprocessing v2 (advanced, +TrivialAugmentWide+RandAugment)
    """
    return 1 if preset == 1 else 2


def imagenet_criterion(preset: int) -> nn.Module:
    # Improved torchvision recipe: smoothing=0.1; basic recipe uses hard CE.
    return nn.CrossEntropyLoss() if preset == 1 else nn.CrossEntropyLoss(label_smoothing=0.1)



class collate_fn_inst():
    def __init__(self, target):
        self.target = target

    def __call__(self, batch):
        if isinstance(batch, tuple) and len(batch) == 2:
            images, targets = batch
        else:
            images, targets = default_collate(batch)
        images, targets = self.target(images, targets)
        if hasattr(images, "device") and hasattr(targets, "to") and targets.device != images.device:
            targets = targets.to(images.device, non_blocking=True)
        return images, targets


def imagenet_collate_fn(preset: int, mixup_alpha=0.2, cutmix_alpha=1.0, num_classes=1000) -> Optional[Callable]:
    assert preset in [1,2], f"preset has to be 1 or 2, get {preset}"
    if preset == 1:
        output_collate_fn = None
    else:
        mixup_cutmix = get_mixup_cutmix(mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, num_classes=num_classes, use_v2=True)
        output_collate_fn = collate_fn_inst(mixup_cutmix)

    return output_collate_fn


def imagenet_sampler_fn(preset: int, ) -> Optional[Callable]:
    assert preset in [1,2], f"preset has to be 1 or 2, get {preset}"
    if preset == 1:
        output_sampler_fn = None
    else:
        # RA4 is a torchvision improved-recipe parameter. The local sampler's
        # non-DDP full-repeat fallback is LLR2-specific; DDP truncates to ~N.
        def sampler_fn(dataset):
            return RASampler(dataset, shuffle=True, repetitions=4)
        output_sampler_fn = sampler_fn
    return output_sampler_fn
