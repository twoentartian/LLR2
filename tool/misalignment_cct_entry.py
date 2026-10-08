#!/usr/bin/env python3
"""CCT-only launcher; keep misalignment_measurement_2.py unchanged.

ImageNet augmentation 2 normally uses a 176 crop. CCT-14 has fixed positional
embeddings for a 224 crop, so all ImageNet levels use 224 here. The optimizer
defaults and warmup/cosine endpoints follow the CCT recipes in
py_src/complete_ml_setup.py, not the generic CIFAR/ImageNet SGD presets.
The measurement algorithm, two orientations, checkpoint selection and
flatness implementation are delegated to the existing measurement script.
"""

from __future__ import annotations

import math

import torch

import misalignment_measurement_2 as measurement


_build_augmentation = measurement.build_augmentation
_describe_augmentation = measurement.describe_augmentation
_build_optimizer = measurement.build_optimizer_and_scheduler


def build_augmentation(ml_setup, dataset_name, level, **kwargs):
    if dataset_name == "imagenet1k":
        kwargs["input_size"] = 224
    return _build_augmentation(ml_setup, dataset_name, level, **kwargs)


def describe_augmentation(dataset_name, level, **kwargs):
    if dataset_name == "imagenet1k":
        kwargs["input_size"] = 224
    return _describe_augmentation(dataset_name, level, **kwargs)


def build_optimizer_and_scheduler(ml_setup, model, training_dataset, **kwargs):
    model_name = getattr(ml_setup.model_type, "name", str(ml_setup.model_type))
    if model_name not in {"cct_7_3x1_32", "cct_14_7x2_224"}:
        return _build_optimizer(ml_setup, model, training_dataset, **kwargs)

    preset = int(kwargs.get("preset", 0))
    if model_name == "cct_7_3x1_32":
        default_lr, default_wd = (5.5e-4, 6e-2) if preset == 0 else (1e-3, 1e-2)
        default_warmup, warmup_lr = 10, 1e-5
    else:
        default_lr, default_wd = 5e-4, 5e-2
        default_warmup, warmup_lr = 25, 1e-6
    options = dict(kwargs)
    if options.get("optimizer", "auto") == "auto":
        options["optimizer"] = "adamw"
    for key, value in (("learning_rate", default_lr), ("weight_decay", default_wd),
                       ("epochs", 300), ("warmup_epochs", default_warmup)):
        if options.get(key) is None:
            options[key] = value
    mode = options.get("scheduler", "cosine")
    if mode in (None, "auto"):
        mode = "cosine"
    if mode not in {"fixed", "none", "cosine"}:
        return _build_optimizer(ml_setup, model, training_dataset, **options)

    # Resolve the optimizer through the existing factory, but construct CCT's
    # standard scheduler separately (including its non-zero starting/end LR).
    warmup_epochs = int(options["warmup_epochs"])
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    optimizer, _, epochs = _build_optimizer(
        ml_setup, model, training_dataset,
        **{**options, "scheduler": "fixed", "warmup_epochs": 0},
    )
    updates_per_epoch = math.ceil(len(training_dataset) / int(options["batch_size"]))
    warmup_epochs = min(warmup_epochs, epochs - 1)
    warmup_steps = warmup_epochs * updates_per_epoch
    cosine_steps = max(1, (epochs - warmup_epochs) * updates_per_epoch)
    initial_lr = float(options["learning_rate"])
    warmup_lr = min(warmup_lr, initial_lr)
    min_lr = min(1e-5, initial_lr)

    def lr_factor(step):
        if step < warmup_steps:
            lr = warmup_lr + (initial_lr - warmup_lr) * step / warmup_steps
        elif mode == "cosine":
            elapsed = min(max(step - warmup_steps, 0), cosine_steps)
            lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * elapsed / cosine_steps))
        else:
            lr = initial_lr
        return lr / initial_lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor) if warmup_steps or mode == "cosine" else None
    optimizer.misalignment_config.update({
        "scheduler": mode,
        "warmup_epochs": warmup_epochs,
        "warmup_learning_rate": warmup_lr,
        "minimum_learning_rate": min_lr if mode == "cosine" else initial_lr,
        "recipe_source": "py_src/complete_ml_setup.py:CCT",
    })
    return optimizer, scheduler, epochs


def main():
    # Runtime-only adapters: no changes to shared py_src or measurement code.
    measurement.build_augmentation = build_augmentation
    measurement.describe_augmentation = describe_augmentation
    measurement.build_optimizer_and_scheduler = build_optimizer_and_scheduler
    measurement.main()


if __name__ == "__main__":
    main()
