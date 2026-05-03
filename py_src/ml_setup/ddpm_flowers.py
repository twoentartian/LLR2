"""Flowers102 DDPM setup for LLR2."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys
from typing import Any, Callable, Optional

from PIL import Image
import numpy as np
import torch
import torchvision

from py_src.adapters import DiffusionAdapter
from py_src.ml_setup.ml_setup import ApplicationType, MLSetup
from py_src.ml_setup_dataset import DatasetSetup, dataset_flowers102
from py_src.ml_setup_model import ModelType


class _LucidrainsDiffusionWithEMA(torch.nn.Module):
    """LLR2 wrapper that adds EMA behavior around lucidrains diffusion models."""

    def __init__(
        self,
        diffusion_model: torch.nn.Module,
        *,
        ema_decay: float = 0.995,
        ema_update_every: int = 10,
        ema_update_after_step: int = 100,
    ) -> None:
        super().__init__()
        from ema_pytorch import EMA

        self.train_diffusion = diffusion_model
        self.ema = EMA(
            diffusion_model,
            beta=ema_decay,
            update_every=ema_update_every,
            update_after_step=ema_update_after_step,
            include_online_model=False,
        )
        self.ema_diffusion.requires_grad_(False) # type: ignore

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
    def ema_diffusion(self):
        return self.ema.ema_model

    @property
    def is_ddim_sampling(self):
        return self.train_diffusion.is_ddim_sampling

    def parameters(self, recurse: bool = True):
        return self.train_diffusion.parameters(recurse=recurse)

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        return self.train_diffusion.named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def forward(self, *args, **kwargs):
        return self.train_diffusion(*args, **kwargs)

    @torch.no_grad()
    def update_ema(self) -> None:
        self.ema.update()

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        old_mode = self.ema_diffusion.training # type: ignore
        self.ema_diffusion.eval() # type: ignore
        try:
            with self._disable_sampling_progress_bar():
                return self.ema_diffusion.sample(*args, **kwargs)  # type: ignore
        finally:
            self.ema_diffusion.train(old_mode) # type: ignore

    @contextmanager
    def _disable_sampling_progress_bar(self):
        diffusion_module = importlib.import_module(self.ema_diffusion.__class__.__module__)
        original_tqdm = getattr(diffusion_module, "tqdm", None)
        if original_tqdm is None:
            yield
            return

        def quiet_tqdm(*args, **kwargs):
            kwargs.setdefault("disable", True)
            return original_tqdm(*args, **kwargs)

        setattr(diffusion_module, "tqdm", quiet_tqdm)
        try:
            yield
        finally:
            setattr(diffusion_module, "tqdm", original_tqdm)

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


def _build_ddpm_flowers102_model(
    image_size: int = 128,
    timesteps: int = 1000,
    sampling_timesteps: int = 250,
    ema_decay: float = 0.995,
    ema_update_every: int = 10,
    ema_update_after_step: int = 100,
    flash_attn: bool = False,
):
    denoising_diffusion_pytorch = _load_vendored_denoising_diffusion_pytorch()
    unet = denoising_diffusion_pytorch.Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=3,
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
        ema_update_after_step=ema_update_after_step,
    )


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


def _save_samples(samples: torch.Tensor, output_folder: str, current_epoch: int) -> None:
    samples_ndarray = torch.clamp(samples, 0, 1).permute(0, 2, 3, 1).cpu().numpy()
    for i, sample in enumerate(samples_ndarray):
        Image.fromarray((sample * 255).astype(np.uint8)).save(
            os.path.join(output_folder, f"epoch{current_epoch}_{i}.png")
        )


def _generate_sample_from_zero_to_one(
    model: torch.nn.Module,
    output_folder: str,
    current_epoch: int,
    device: torch.device,
    count: int,
) -> None:
    del device
    model.eval()
    samples = model.sample(batch_size=count)  # type: ignore
    _save_samples(samples, output_folder, current_epoch)


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


__all__ = ["ddpm_flowers102"]
