"""CIFAR DDPM setup for LLR2."""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Literal, Optional

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from py_src.adapters import DiffusionAdapter
from py_src.ml_setup.ml_setup import ApplicationType, MLSetup
from py_src.ml_setup_dataset import DatasetSetup, dataset_cifar10
from py_src.ml_setup_model import ModelType


class RescaleChannels:
    """Scale pixel values from [0, 1] to [-1, 1]."""

    def __call__(self, sample):
        return 2 * sample - 1


def _build_ddpm_cifar_model(
    img_channels: int = 3,
    img_size: int = 32,
    num_classes: int = 10,
    use_labels: bool = False,
    base_channels: int = 128,
    channel_mults=(1, 2, 2, 2),
    time_emb_dim: int = 512,
    norm: str = "gn",
    dropout: float = 0.1,
    activation: str = "silu",
    attention_resolutions=(1,),
    schedule: Literal["cosine", "linear"] = "linear",
    num_timesteps: int = 1000,
    schedule_low: float = 1e-4,
    schedule_high: float = 2e-2,
    ema_decay: float = 0.9999,
    ema_update_rate: int = 1,
    loss_type: str = "l2",
):
    from py_src.third_party.ddpm.ddpm.unet import UNet
    from py_src.third_party.ddpm.ddpm.diffusion import (
        GaussianDiffusion,
        generate_cosine_schedule,
        generate_linear_schedule,
    )

    activations = {"relu": F.relu, "mish": F.mish, "silu": F.silu}

    unet = UNet(
        img_channels=img_channels,
        base_channels=base_channels,
        channel_mults=channel_mults,
        time_emb_dim=time_emb_dim,
        norm=norm,
        dropout=dropout,
        activation=activations[activation],
        attention_resolutions=attention_resolutions,
        num_classes=None if not use_labels else num_classes,
        initial_pad=0,
    )

    if schedule == "cosine":
        betas = generate_cosine_schedule(num_timesteps)
    else:
        betas = generate_linear_schedule(
            num_timesteps,
            schedule_low * 1000 / num_timesteps,
            schedule_high * 1000 / num_timesteps,
        )

    return GaussianDiffusion(
        unet,
        (img_size, img_size),
        img_channels,
        num_classes,
        betas,
        ema_decay=ema_decay,
        ema_update_rate=ema_update_rate,
        ema_start=2000,
        loss_type=loss_type,
    )


def _save_samples(samples: torch.Tensor, output_folder: str, current_epoch: int) -> None:
    samples_ndarray = torch.clamp(samples, 0, 1).permute(0, 2, 3, 1).cpu().numpy()
    for i, sample in enumerate(samples_ndarray):
        Image.fromarray((sample * 255).astype(np.uint8)).save(
            os.path.join(output_folder, f"epoch{current_epoch}_{i}.png")
        )


@contextmanager
def _fixed_sampling_seed(device: torch.device, seed: Optional[int]):
    if seed is None:
        yield
        return

    devices: list[int] = []
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        devices = [device_index]

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def _generate_sample_from_neg_one_to_one(
    model: torch.nn.Module,
    output_folder: str,
    current_epoch: int,
    device: torch.device,
    count: int,
    seed: Optional[int] = None,
) -> None:
    old_mode = model.training
    model.eval()
    try:
        with _fixed_sampling_seed(device, seed):
            samples = model.sample(count, device)  # type: ignore
        _save_samples((samples + 1) / 2, output_folder, current_epoch)
    finally:
        model.train(old_mode)


def ddpm_cifar10(override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    transform = [torchvision.transforms.ToTensor(), RescaleChannels()]
    dataset = override_dataset if override_dataset is not None else dataset_cifar10(
        transforms_training=transform,
        transforms_testing=transform,
    )
    model = _build_ddpm_cifar_model()

    output_ml_setup = MLSetup(
        model=model,
        adapter=DiffusionAdapter(model),
        model_type=ModelType.ddpm_cifar10,
        training_data=dataset.train_data,
        testing_data=dataset.valdation_data,
        dataset_type=dataset.dataset_type,
        default_batch_size=128,
        has_normalization_layer=True,
        application_type=ApplicationType.diffusion,
    )
    output_ml_setup.difussion_generate_sample = _generate_sample_from_neg_one_to_one

    return output_ml_setup


__all__ = ["ddpm_cifar10", "RescaleChannels"]
