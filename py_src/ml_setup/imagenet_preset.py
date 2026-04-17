from typing import Callable, Optional

import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Sampler
from torchvision.transforms.v2 import RandomChoice

from .transforms import RandomMixUp, RandomCutMix
from .sampler import RASampler

def preset_version(preset: int) -> int:
    """Map LLR2 preset index to imagenet preprocessing version.

    preset=0 → preprocessing v1 (basic, DFL_torch default for DLA)
    preset=1 → preprocessing v2 (advanced, +TrivialAugmentWide+RandAugment)
    """
    return 1 if preset == 1 else 2


def imagenet_criterion(preset: int) -> nn.Module:
    return nn.CrossEntropyLoss() if preset == 1 else nn.CrossEntropyLoss(label_smoothing=0.1)



class collate_fn_inst():
    def __init__(self, target):
        self.target = target

    def __call__(self, batch):
        return self.target(*default_collate(batch))

def _get_mixup_cutmix(*, mixup_alpha, cutmix_alpha, num_classes):
    mixup_cutmix = []
    if mixup_alpha > 0:
        mixup_cutmix.append(RandomMixUp(alpha=mixup_alpha, num_classes=num_classes))
    if cutmix_alpha > 0:
        mixup_cutmix.append(RandomCutMix(alpha=cutmix_alpha, num_classes=num_classes))
    if not mixup_cutmix:
        return None
    return RandomChoice(mixup_cutmix)

def imagenet_collate_fn(preset: int, mixup_alpha=0.2, cutmix_alpha=1.0, num_classes=1000) -> Optional[Callable]:
    assert preset in [1,2], f"preset has to be 1 or 2, get {preset}"
    if preset == 1:
        output_collate_fn = None
    else:
        mixup_cutmix = _get_mixup_cutmix(mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, num_classes=num_classes)
        output_collate_fn = collate_fn_inst(mixup_cutmix)

    return output_collate_fn


def imagenet_sampler_fn(preset: int, ) -> Optional[Callable]:
    assert preset in [1,2], f"preset has to be 1 or 2, get {preset}"
    if preset == 1:
        output_sampler_fn = None
    else:
        def sampler_fn(dataset):
            return RASampler(dataset, shuffle=True, repetitions=4)
        output_sampler_fn = sampler_fn
    return output_sampler_fn