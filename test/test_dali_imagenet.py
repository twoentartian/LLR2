"""Optional CUDA/DALI smoke test.

This file is intentionally not included in GitHub Actions because hosted
runners do not provide CUDA GPUs. Run it manually on a CUDA machine with DALI:

    python -m unittest test.test_dali_imagenet -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from py_src.complete_ml_setup import FastTrainingSetup
from py_src.engine import Device, train, val
from py_src.ml_setup.dataloader_util import DataloaderConfig
from py_src.ml_setup.factory import get_ml_setup_from_config
import py_src.ml_setup_dataset.dataset_imagenet as dataset_imagenet_module


def _require_cuda_and_dali() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required for the DALI ImageNet test")
    try:
        import nvidia.dali  # noqa: F401
        import nvidia.dali.plugin.pytorch  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("NVIDIA DALI is not installed") from exc


def _write_tiny_imagenet10(root: Path, image_size: int = 256) -> None:
    for split in ("train", "val"):
        for label in range(10):
            class_dir = root / split / f"class_{label:02d}"
            class_dir.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                color = (
                    (label * 23 + index * 5) % 255,
                    (label * 47 + index * 11) % 255,
                    (label * 71 + index * 17) % 255,
                )
                image = Image.new("RGB", (image_size, image_size), color=color)
                image.save(class_dir / f"{index}.jpg", quality=95)


class TestDaliImageNetTraining(unittest.TestCase):
    def test_resnet18_imagenet10_dali_train_and_val_one_batch(self) -> None:
        _require_cuda_and_dali()

        old_imagenet10_path = dataset_imagenet_module.imagenet10_path
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            _write_tiny_imagenet10(dataset_root)
            dataset_imagenet_module.imagenet10_path = dataset_root
            try:
                setup = get_ml_setup_from_config(
                    "resnet18_bn",
                    dataset_type="imagenet10",
                    preset=1,
                    use_dali=True,
                    dali_device_id=0,
                )
            finally:
                dataset_imagenet_module.imagenet10_path = old_imagenet10_path

            device = Device.cuda(0)
            setup.model.to(device.device)
            optimizer, lr_scheduler, _ = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(
                setup,
                setup.model,
                preset=1,
                override_dataset=setup.training_data,
                override_batch_size=4,
            )
            loader_config = DataloaderConfig(batch_size=4, num_workers=2, num_samples=4)
            train_loader = setup.train_dataloader(loader_config)
            val_loader = setup.val_dataloader(loader_config)

            train_result = train(
                setup.adapter,
                train_loader,
                optimizer,
                lr_scheduler,
                device=device,
                max_rounds=1,
            )
            val_result = val(setup.adapter, val_loader, device=device)

        self.assertEqual(train_result.iterations, 1)
        self.assertGreater(train_result.total_count, 0)
        self.assertIsNotNone(train_result.accuracy)
        self.assertGreater(val_result.total_count, 0)
        self.assertIsNotNone(val_result.accuracy)


if __name__ == "__main__":
    unittest.main()
