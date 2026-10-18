"""ConvNeXt 22k augmentation, adapted to LLR2 21k-p as preset 4.

Recipe: A ConvNet for the 2020s, appendix A.1/table 5 (CVPR 2022):
https://arxiv.org/html/2201.03545v2#A1.SS1
Official parameter defaults and transform construction:
https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/main.py
https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/datasets.py
Implementation snapshots used to check this port (not runtime version pins):
https://github.com/huggingface/pytorch-image-models/blob/v1.0.29/timm/data/transforms_factory.py
https://github.com/huggingface/pytorch-image-models/blob/v1.0.29/timm/data/auto_augment.py
https://github.com/huggingface/pytorch-image-models/blob/v1.0.29/timm/data/random_erasing.py

CPU calls installed timm; DALI below is an LLR2 GPU port of its increasing
RandAugment and pixel erasing. DALI geometry uses bilinear/scalar fill 128,
and magnitude is quantized to 0.01; this is not a PIL-exact reproduction.
Original ConvNeXt used timm 0.3.2. Per-operation papers and further differences
are in README_imagenet_augmentation.md sections 1.1/1.2 and 6.
Keep this separate from historical ImageNet presets 1/2/3.
"""

import math

import numpy as np


def convnext_transforms():
    from timm.data import create_transform
    from torchvision import transforms

    # ConvNeXt main.py defaults, passed through datasets.py to timm. With this
    # RA policy timm disables standalone ColorJitter unless explicitly forced;
    # color_jitter=0.4 does not add another transform to the actual pipeline.
    train = create_transform(
        input_size=224, is_training=True, color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1", interpolation="bicubic",
        re_prob=0.25, re_mode="pixel", re_count=1,
    )
    validation = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return train, validation


def _posterize_mask(level):
    # timm PosterizeIncreasing: keep 4-int(level/10*4) bits, including zero.
    bits = 4 - int(level / 10 * 4)
    return np.array((255 << (8 - bits)) & 255, dtype=np.uint8)


def _solarize_threshold(level):
    return np.array(256 - int(level / 10 * 256), dtype=np.int32)


def _solarize_add_shift(level):
    return np.array(min(128, int(level / 10 * 110)), dtype=np.int32)


def _solarize_add(data, shift):
    from nvidia.dali import fn, types, math as dm

    pixels = fn.cast(data, dtype=types.INT32)
    return fn.cast((pixels < 128) * dm.clamp(pixels + shift, 0, 255)
                   + (pixels >= 128) * pixels, dtype=types.UINT8)


def _sharpness(data, kernel):
    from nvidia.dali import fn

    filter_op = getattr(fn, "filter", None)
    if filter_op is None:
        filter_op = fn.experimental.filter
    return filter_op(data, kernel)


def _randaugment_suite():
    from nvidia.dali.auto_aug import augmentations as a
    from nvidia.dali.auto_aug.core import augmentation
    from .dataset_imagenet import _dali_equalize

    # Same order/pool as timm _RAND_INCREASING_TRANSFORMS / LEVEL_TO_ARG in the
    # pinned auto_augment.py above. Gaussian magnitude and "increasing" mappings
    # are timm choices beyond the base RandAugment paper (arxiv:1909.13719).
    return [
        a.auto_contrast, augmentation(_dali_equalize, name="equalize"), a.invert,
        a.rotate.augmentation((0, 30), True),
        a.posterize.augmentation((0, 10), False, _posterize_mask),
        a.solarize.augmentation((0, 10), False, _solarize_threshold),
        augmentation(_solarize_add, mag_range=(0, 10), mag_to_param=_solarize_add_shift),
        a.color.augmentation((0, 0.9), True, a.shift_enhance_range),
        a.contrast.augmentation((0, 0.9), True, a.shift_enhance_range),
        a.brightness.augmentation((0, 0.9), True, a.shift_enhance_range),
        augmentation(_sharpness, mag_range=(0, 0.9), randomly_negate=True,
                     mag_to_param=a.sharpness_kernel, param_device="auto"),
        a.shear_x.augmentation((0, 0.3), True), a.shear_y.augmentation((0, 0.3), True),
        a.translate_x.augmentation((0, 0.45), True), a.translate_y.augmentation((0, 0.45), True),
    ]


def dali_convnext_randaugment(images, crop_size):
    """LLR2 DALI port of timm rand-m9-mstd0.5-inc1; see module sources."""
    from nvidia.dali import fn, types, math as dm
    from nvidia.dali.auto_aug.core import select, signed_bin

    suite = _randaugment_suite()
    for _ in range(2):
        # timm chooses two ops with replacement; each applies with p=0.5.
        selected = fn.random.uniform(values=list(range(len(suite))), dtype=types.INT32)
        if fn.random.coin_flip(probability=0.5):
            # Public augmentation API accepts a DataNode magnitude bin. 0.01
            # magnitude resolution approximates timm's clipped Gaussian M=9±0.5.
            magnitude = fn.cast(dm.clamp(fn.random.normal(mean=9.0, stddev=0.5), 0, 10) * 100,
                                dtype=types.INT32)
            magnitude = signed_bin(magnitude)
            images = select(
                suite, selected, images, magnitude_bin=magnitude, num_magnitude_bins=1001,
                # DALI geometric operators accept a scalar border fill.
                # DALI warp/rotate support linear or nearest interpolation;
                # the initial crop/resize and validation resize remain bicubic.
                shape=(crop_size, crop_size), fill_value=128, interp_type=types.INTERP_LINEAR,
            )
    return images


def dali_convnext_erasing(images, crop_size, probability):
    """Port timm RandomErasing(mode='pixel', max_count=1) to DALI.

    Method: https://arxiv.org/abs/1708.04896. Retry limits, aspect/area ranges
    and per-pixel normal fill follow the linked timm implementation; DALI RNG
    and mask arithmetic are project adaptations, not exact timm pixel replay.
    """
    from nvidia.dali import fn, types, math as dm

    if fn.random.coin_flip(probability=probability):
        found = False
        # Same area/aspect limits, ten attempts, and N(0,1) pixel fill as timm.
        for _ in range(10):
            area = fn.random.uniform(range=[0.02, 1 / 3])
            ratio = dm.exp(fn.random.uniform(range=[math.log(0.3), math.log(1 / 0.3)]))
            height = fn.cast(dm.sqrt(area * ratio) * crop_size + 0.5, dtype=types.INT32)
            width = fn.cast(dm.sqrt(area / ratio) * crop_size + 0.5, dtype=types.INT32)
            valid = (height > 0) & (height < crop_size) & (width > 0) & (width < crop_size)
            if (found == False) & valid:
                anchor_y = fn.cast(fn.random.uniform() * (crop_size - height + 1), dtype=types.INT32)
                anchor_x = fn.cast(fn.random.uniform() * (crop_size - width + 1), dtype=types.INT32)
                mask = fn.constant(fdata=0.0, shape=(1, crop_size, crop_size),
                                   dtype=types.FLOAT, device="gpu", layout="CHW")
                mask = fn.erase(mask, anchor=fn.cast(fn.stack(anchor_y, anchor_x), dtype=types.FLOAT),
                                shape=fn.cast(fn.stack(height, width), dtype=types.FLOAT),
                                axis_names="HW", fill_value=1.0)
                noise = fn.random.normal(device="gpu", shape=(3, crop_size, crop_size), dtype=types.FLOAT)
                images = images * (1 - mask) + noise * mask
            found = found | valid
    return images
