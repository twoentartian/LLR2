from __future__ import annotations
from collections.abc import Callable, Sequence
from typing import Any

import os
import sys
import torch
from PIL import Image
import lightning as L
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from py_src.ml_setup import ApplicationType, MLSetup
from py_src.complete_ml_setup import FastTrainingSetup
from py_src.engine import Device, TrainResult, ValResult, train, val
from py_src.ml_setup_dataset import DatasetSetup, DatasetType
from py_src.ml_setup.dataloader_util import DataloaderConfig

# ---------------------------------------------------------------------------
# Single-batch train and interface (for unit tests and smoke-checks)
# ---------------------------------------------------------------------------

def run_single_batch(
    ml_setup: MLSetup,
    *,
    use_cpu: bool = True,
    amp: bool = False,
    preset: int = 0,
    run_val: bool = True,
    batch_size: int = 8,
) -> "tuple[TrainResult, ValResult | None]":
    """Train for one batch and (optionally) evaluate for one batch.

    Intended for unit testing and quick smoke-checks.  The model and adapter
    stored in *ml_setup* are used directly (no deepcopy).

    Parameters
    ----------
    ml_setup:
        Fully configured :class:`MLSetup`.
    use_cpu:
        Force CPU execution (default ``True`` so tests run anywhere).
    amp:
        Enable automatic mixed precision.
    preset:
        Hyperparameter preset index forwarded to
        :func:`~py_src.complete_ml_setup.FastTrainingSetup.get_optimizer_lr_scheduler_epoch`.
    run_val:
        Whether to run the validation pass.  Always skipped for diffusion
        models regardless of this flag.

    Returns
    -------
    tuple[TrainResult, ValResult | None]
        ``(train_result, val_result)`` where *val_result* is ``None`` when
        validation was skipped.
    """
    device = Device.cpu() if use_cpu else Device.auto()

    one_batch_cfg = DataloaderConfig(num_samples=batch_size, num_workers=0, pin_memory=False)

    train_loader = ml_setup.train_dataloader(one_batch_cfg, ignore_override=True)

    model = ml_setup.model
    adapter = ml_setup.adapter
    model.to(device.device)

    # Build optimizer / scheduler — mirrors the logic in training_model()
    if isinstance(model, L.LightningModule):
        optimizer_lit, lr_scheduler_lit = model.configure_optimizers()  # type: ignore[call-arg]
        optimizer_cfg, lr_scheduler_cfg, _ = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(
            ml_setup,
            model,
            preset,
            override_steps_per_epoch=len(train_loader), # type: ignore
        )
        optimizer = optimizer_cfg if optimizer_cfg is not None else optimizer_lit
        lr_scheduler = lr_scheduler_cfg if lr_scheduler_cfg is not None else lr_scheduler_lit
    else:
        optimizer, lr_scheduler, _ = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(
            ml_setup,
            model,
            preset,
            override_steps_per_epoch=len(train_loader), # type: ignore
        )

    scaler = device.make_scaler() if amp else None

    train_result = train(
        adapter, train_loader, optimizer, lr_scheduler, # type: ignore
        device=device, scaler=scaler, max_rounds=1,
    )

    val_result = None
    if run_val and ml_setup.application_type == ApplicationType.classifier:
        val_loader = ml_setup.val_dataloader(one_batch_cfg, ignore_override=True)
        val_result = val(adapter, val_loader, device=device)

    return train_result, val_result


class DummyImageClassificationDataset(Dataset[tuple[Any, int]]):
    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        image_size: tuple[int, int],
        channels: int = 3,
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[int], Any] | None = None,
        return_pil: bool = True,
        seed: int = 0,
    ) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.channels = channels
        self.transform = transform
        self.target_transform = target_transform
        self.return_pil = return_pil
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        generator = torch.Generator().manual_seed(self.seed + index)

        h, w = self.image_size
        image_tensor = torch.randint(
            0,
            256,
            (self.channels, h, w),
            dtype=torch.uint8,
            generator=generator,
        )
        target = index % self.num_classes

        if self.return_pil:
            if self.channels == 1:
                image = Image.fromarray(image_tensor.squeeze(0).numpy(), mode="L")
            elif self.channels == 3:
                image = Image.fromarray(image_tensor.permute(1, 2, 0).numpy(), mode="RGB")
            else:
                raise ValueError(f"Unsupported channels for PIL output: {self.channels}")
        else:
            image = image_tensor.float() / 255.0

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


_type_transform = Callable[[Any], Any]
_type_transform_config = _type_transform | Sequence[_type_transform]


def _resolve_transform(
    split_transforms: _type_transform_config | None,
    fallback_transform: _type_transform | None = None,
) -> _type_transform | None:
    if split_transforms is None:
        return fallback_transform
    if callable(split_transforms):
        return split_transforms
    return transforms.Compose(list(split_transforms))


def _make_dummy_image_classification_setup(
    *,
    dataset_type: DatasetType,
    num_samples: int,
    num_classes: int,
    image_size: tuple[int, int],
    channels: int,
    transforms_training: _type_transform_config | None = None,
    transforms_testing: _type_transform_config | None = None,
    transform: _type_transform | None = None,
    target_transform: Callable[[int], Any] | None = None,
    return_pil: bool = True,
    seed: int,
) -> DatasetSetup:
    train_data = DummyImageClassificationDataset(
        num_samples=num_samples,
        num_classes=num_classes,
        image_size=image_size,
        channels=channels,
        transform=_resolve_transform(transforms_training, transform),
        target_transform=target_transform,
        return_pil=return_pil,
        seed=seed,
    )
    test_data = DummyImageClassificationDataset(
        num_samples=num_samples,
        num_classes=num_classes,
        image_size=image_size,
        channels=channels,
        transform=_resolve_transform(transforms_testing, transform),
        target_transform=target_transform,
        return_pil=return_pil,
        seed=seed + 10_000,
    )
    return DatasetSetup(dataset_type, train_data, test_data)


def make_dummy_imagenet1k(
    num_samples: int = 32,
    transforms_training: _type_transform_config | None = None,
    transforms_testing: _type_transform_config | None = None,
    transform: _type_transform | None = None,
    target_transform: Callable[[int], Any] | None = None,
    return_pil: bool = True,
    image_size: tuple[int, int] = (224, 224),
) -> DatasetSetup:
    return _make_dummy_image_classification_setup(
        dataset_type=DatasetType.imagenet1k,
        num_samples=num_samples,
        num_classes=1000,
        image_size=image_size,
        channels=3,
        transforms_training=transforms_training,
        transforms_testing=transforms_testing,
        transform=transform,
        target_transform=target_transform,
        return_pil=return_pil,
        seed=1000,
    )


def make_dummy_imagenet100(
    num_samples: int = 32,
    transforms_training: _type_transform_config | None = None,
    transforms_testing: _type_transform_config | None = None,
    transform: _type_transform | None = None,
    target_transform: Callable[[int], Any] | None = None,
    return_pil: bool = True,
    image_size: tuple[int, int] = (224, 224),
) -> DatasetSetup:
    return _make_dummy_image_classification_setup(
        dataset_type=DatasetType.imagenet100,
        num_samples=num_samples,
        num_classes=100,
        image_size=image_size,
        channels=3,
        transforms_training=transforms_training,
        transforms_testing=transforms_testing,
        transform=transform,
        target_transform=target_transform,
        return_pil=return_pil,
        seed=100,
    )


def make_dummy_imagenet10(
    num_samples: int = 32,
    transforms_training: _type_transform_config | None = None,
    transforms_testing: _type_transform_config | None = None,
    transform: _type_transform | None = None,
    target_transform: Callable[[int], Any] | None = None,
    return_pil: bool = True,
    image_size: tuple[int, int] = (224, 224),
) -> DatasetSetup:
    return _make_dummy_image_classification_setup(
        dataset_type=DatasetType.imagenet10,
        num_samples=num_samples,
        num_classes=10,
        image_size=image_size,
        channels=3,
        transforms_training=transforms_training,
        transforms_testing=transforms_testing,
        transform=transform,
        target_transform=target_transform,
        return_pil=return_pil,
        seed=10,
    )


def make_dummy_cifar10(
    num_samples: int = 32,
    transforms_training: _type_transform_config | None = None,
    transforms_testing: _type_transform_config | None = None,
    transform: _type_transform | None = None,
    target_transform: Callable[[int], Any] | None = None,
    return_pil: bool = True,
) -> DatasetSetup:
    return _make_dummy_image_classification_setup(
        dataset_type=DatasetType.cifar10,
        num_samples=num_samples,
        num_classes=10,
        image_size=(32, 32),
        channels=3,
        transforms_training=transforms_training,
        transforms_testing=transforms_testing,
        transform=transform,
        target_transform=target_transform,
        return_pil=return_pil,
        seed=2010,
    )


def make_dummy_cifar100(
    num_samples: int = 32,
    transforms_training: _type_transform_config | None = None,
    transforms_testing: _type_transform_config | None = None,
    transform: _type_transform | None = None,
    target_transform: Callable[[int], Any] | None = None,
    return_pil: bool = True,
) -> DatasetSetup:
    return _make_dummy_image_classification_setup(
        dataset_type=DatasetType.cifar100,
        num_samples=num_samples,
        num_classes=100,
        image_size=(32, 32),
        channels=3,
        transforms_training=transforms_training,
        transforms_testing=transforms_testing,
        transform=transform,
        target_transform=target_transform,
        return_pil=return_pil,
        seed=2100,
    )


def make_dummy_mnist(
    num_samples: int = 32,
    transforms_training: _type_transform_config | None = None,
    transforms_testing: _type_transform_config | None = None,
    transform: _type_transform | None = None,
    target_transform: Callable[[int], Any] | None = None,
    return_pil: bool = True,
    image_size: tuple[int, int] = (28, 28),
) -> DatasetSetup:
    return _make_dummy_image_classification_setup(
        dataset_type=DatasetType.mnist,
        num_samples=num_samples,
        num_classes=10,
        image_size=image_size,
        channels=1,
        transforms_training=transforms_training,
        transforms_testing=transforms_testing,
        transform=transform,
        target_transform=target_transform,
        return_pil=return_pil,
        seed=1905,
    )


class DummyDatasetNanoCLIP(torch.utils.data.Dataset):
    """Random (image, caption_tokens, attention_mask) triples."""

    def __init__(
        self,
        n: int = 16,
        img_size: int = 8,
        vocab_size: int = 100,
        seq_len: int = 8,
    ):
        self.images = torch.randn(n, 3, img_size, img_size)
        self.captions = torch.randint(0, vocab_size, (n, seq_len), dtype=torch.long)
        self.masks = torch.ones(n, seq_len, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.captions[idx], self.masks[idx]
