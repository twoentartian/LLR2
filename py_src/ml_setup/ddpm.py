"""DDPM (Denoising Diffusion Probabilistic Model) setup.

This wraps the existing GaussianDiffusion model with a
:class:`DiffusionAdapter` so the execution engine can drive it uniformly.
"""

from __future__ import annotations

import copy
import importlib
import os
from pathlib import Path
import sys
from typing import Literal, Optional, Callable, Any

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from py_src.ml_setup_dataset import DatasetSetup, DatasetType, dataset_cifar10, dataset_flowers102
from py_src.ml_setup_model import ModelType
from py_src.adapters import DiffusionAdapter
from py_src.ml_setup.ml_setup import ApplicationType, MLSetup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RescaleChannels:
    """Scale pixel values from [0, 1] to [-1, 1]."""
    def __call__(self, sample):
        return 2 * sample - 1


class _LucidrainsDiffusionWithEMA(torch.nn.Module):
    """LLR2 wrapper that adds EMA behavior around lucidrains diffusion models.

    LLR2 trains through ``engine.py`` and ``DiffusionAdapter`` instead of the
    upstream lucidrains ``Trainer``. This wrapper restores the two main pieces
    of trainer behavior that the engine expects:

    * ``forward(...)`` uses the live training diffusion model.
    * ``sample(...)`` uses an EMA copy for generation.
    * ``update_ema()`` is called from ``DiffusionAdapter.post_train_step()``.
    """

    def __init__(
        self,
        diffusion_model: torch.nn.Module,
        *,
        ema_decay: float = 0.995,
        ema_update_every: int = 10,
    ) -> None:
        super().__init__()
        self.train_diffusion = diffusion_model
        self.ema_diffusion = copy.deepcopy(diffusion_model)
        self.ema_diffusion.requires_grad_(False)

        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        self.register_buffer("_ema_step", torch.zeros((), dtype=torch.long))

        self._copy_parameters_from_train_model()
        self._copy_buffers_from_train_model()

    @property
    def channels(self):
        return self.train_diffusion.channels

    @property
    def image_size(self):
        return self.train_diffusion.image_size

    @property
    def num_timesteps(self):
        return self.train_diffusion.num_timesteps

    @property
    def is_ddim_sampling(self):
        return self.train_diffusion.is_ddim_sampling

    def parameters(self, recurse: bool = True):
        return self.train_diffusion.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        return self.train_diffusion.named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def forward(self, *args, **kwargs):
        return self.train_diffusion(*args, **kwargs)

    @torch.no_grad()
    def update_ema(self) -> None:
        self._ema_step.add_(1) # type: ignore
        step = int(self._ema_step.item()) # type: ignore
        if step % self.ema_update_every != 0:
            return

        for ema_param, train_param in zip(
            self.ema_diffusion.parameters(),
            self.train_diffusion.parameters(),
        ):
            ema_param.lerp_(train_param.detach(), 1 - self.ema_decay)

        self._copy_buffers_from_train_model()

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        old_mode = self.ema_diffusion.training
        self.ema_diffusion.eval()
        try:
            return self.ema_diffusion.sample(*args, **kwargs) # type: ignore
        finally:
            self.ema_diffusion.train(old_mode)

    @torch.no_grad()
    def _copy_buffers_from_train_model(self) -> None:
        for ema_buffer, train_buffer in zip(
            self.ema_diffusion.buffers(),
            self.train_diffusion.buffers(),
        ):
            ema_buffer.copy_(train_buffer.detach())

    @torch.no_grad()
    def _copy_parameters_from_train_model(self) -> None:
        for ema_param, train_param in zip(
            self.ema_diffusion.parameters(),
            self.train_diffusion.parameters(),
        ):
            ema_param.copy_(train_param.detach())


def _load_vendored_denoising_diffusion_pytorch():
    module_name = "denoising_diffusion_pytorch"
    vendor_root = Path(__file__).resolve().parent.parent / "third_party" / module_name
    vendor_package_dir = vendor_root / module_name

    existing_module = sys.modules.get(module_name)
    if existing_module is not None:
        existing_file = getattr(existing_module, "__file__", None)
        if existing_file is not None and vendor_package_dir in Path(existing_file).resolve().parents:
            return existing_module
        for loaded_name in list(sys.modules):
            if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
                del sys.modules[loaded_name]

    vendor_root_str = str(vendor_root)
    if vendor_root_str not in sys.path:
        sys.path.insert(0, vendor_root_str)

    importlib.invalidate_caches()
    return importlib.import_module(module_name)


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


def _build_ddpm_flowers102_model(
    image_size: int = 128,
    timesteps: int = 1000,
    sampling_timesteps: int = 250,
    ema_decay: float = 0.995,
    ema_update_every: int = 10,
    flash_attn: bool = False,
):
    denoising_diffusion_pytorch = _load_vendored_denoising_diffusion_pytorch()
    unet = denoising_diffusion_pytorch.Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        # LLR2 commonly trains through float32 paths unless callers opt into
        # ``--amp``. Disabling flash attention by default avoids hard failures
        # on GPUs where lucidrains' attention wrapper requests a flash-only
        # kernel for float32 inputs.
        flash_attn=flash_attn,
    )
    diffusion = denoising_diffusion_pytorch.GaussianDiffusion(
        unet,
        image_size=image_size,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
        objective="pred_v",
        beta_schedule="sigmoid",
    )
    return _LucidrainsDiffusionWithEMA(
        diffusion,
        ema_decay=ema_decay,
        ema_update_every=ema_update_every,
    )


# ---------------------------------------------------------------------------
# Generate samples
# ---------------------------------------------------------------------------

def _save_samples(samples: torch.Tensor, output_folder: str, current_epoch: int) -> None:
    samples_ndarray = torch.clamp(samples, 0, 1).permute(0, 2, 3, 1).cpu().numpy()
    for i, sample in enumerate(samples_ndarray):
        Image.fromarray((sample * 255).astype(np.uint8)).save(
            os.path.join(output_folder, f"epoch{current_epoch}_{i}.png")
        )

def _generate_sample_from_neg_one_to_one(
    model: torch.nn.Module,
    output_folder: str,
    current_epoch: int,
    device: torch.device,
) -> None:
    model.eval()
    samples = model.sample(10, device) # type: ignore
    _save_samples((samples + 1) / 2, output_folder, current_epoch)

def _generate_sample_from_zero_to_one(
    model: torch.nn.Module,
    output_folder: str,
    current_epoch: int,
    device: torch.device,
) -> None:
    del device
    model.eval()
    samples = model.sample(batch_size=10) # type: ignore
    _save_samples(samples, output_folder, current_epoch)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def ddpm_cifar10(override_dataset:Optional[DatasetSetup]=None) -> MLSetup:
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




def _lucidrains_ddpm_transform(image_size: int, augmentation: bool = False):
    transforms_list: list[Callable[[Any], Any]] = [torchvision.transforms.Resize(image_size)]
    if augmentation:
        transforms_list.append(torchvision.transforms.RandomHorizontalFlip())
    transforms_list.extend(
        [
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.ToTensor(),
        ]
    )
    return transforms_list

def ddpm_flowers102(override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    image_size = 128
    dataset = override_dataset if override_dataset is not None else dataset_flowers102(
        image_size=image_size,
        transforms_training=_lucidrains_ddpm_transform(image_size, augmentation=True),
        transforms_testing=_lucidrains_ddpm_transform(image_size, augmentation=False),
    )
    model = _build_ddpm_flowers102_model(image_size=image_size)

    output_ml_setup = MLSetup(
        model=model,
        adapter=DiffusionAdapter(model),
        model_type=ModelType.ddpm_flowers102,
        training_data=dataset.train_data,
        testing_data=dataset.valdation_data,
        dataset_type=dataset.dataset_type,
        default_batch_size=32,
        has_normalization_layer=True,
        application_type=ApplicationType.diffusion,
    )
    output_ml_setup.difussion_generate_sample = _generate_sample_from_zero_to_one

    return output_ml_setup
