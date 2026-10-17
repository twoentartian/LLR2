"""Optional CUDA/DALI smoke test.

This file is intentionally not included in GitHub Actions because hosted
runners do not provide CUDA GPUs. Run it manually on a CUDA machine with DALI:

    python -m unittest test.test_dali_imagenet -v
"""

from __future__ import annotations

import ast
import linecache
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
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
    def test_equalize_uses_regular_operator_in_autoaugment_and_v2(self) -> None:
        _require_cuda_and_dali()
        from nvidia.dali import fn

        if not hasattr(fn, "equalize"):
            self.skipTest("Regular equalize requires a newer DALI release")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_pattern_images(root)
            for version, policy in ((1, "imagenet"), (2, None)):
                with self.subTest(version=version, policy=policy), \
                     warnings.catch_warnings(record=True) as caught, \
                     patch.object(fn.experimental, "equalize", create=True,
                                  side_effect=AssertionError("Deprecated equalize alias was called")):
                    warnings.simplefilter("always", DeprecationWarning)
                    pipeline = self._augmentation_pipeline(root, version, policy)
                    images, _ = pipeline.run()
                    self.assertEqual(images.as_cpu().as_array().shape, (4, 3, 32, 32))
                    self.assertFalse(any("equalize" in str(item.message) for item in caught))
                    del pipeline

    def test_seeded_pixels_match_pipeline_before_equalize_fix(self) -> None:
        _require_cuda_and_dali()
        # Load the actual pre-fix builder, including its upstream convenience
        # wrappers, so a change to policy ordering or RNG consumption is detected.
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "show", "8ec69b1:py_src/ml_setup_dataset/dataset_imagenet.py"],
            cwd=repository, capture_output=True, text=True,
        )
        if result.returncode:
            self.skipTest("Pre-fix commit 8ec69b1 is unavailable")
        filename = "git:8ec69b1:dataset_imagenet.py"
        linecache.cache[filename] = (len(result.stdout), None, result.stdout.splitlines(True), filename)
        node = next(item for item in ast.parse(result.stdout).body
                    if isinstance(item, ast.FunctionDef) and item.name == "build_dali_imagenet_pipeline")
        namespace = dict(vars(dataset_imagenet_module))
        exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
        baseline_builder = namespace["build_dali_imagenet_pipeline"]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_pattern_images(root)
            for version, policy in ((1, "imagenet"), (2, None)):
                with self.subTest(version=version, policy=policy), warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    baseline = self._augmentation_pipeline(root, version, policy, baseline_builder)
                    current = self._augmentation_pipeline(root, version, policy)
                    for _ in range(3):
                        old_images, old_labels = baseline.run()
                        new_images, new_labels = current.run()
                        np.testing.assert_array_equal(new_images.as_cpu().as_array(),
                                                      old_images.as_cpu().as_array())
                        np.testing.assert_array_equal(new_labels.as_array(), old_labels.as_array())
                    del baseline, current

    @staticmethod
    def _write_pattern_images(root):
        pattern = (np.arange(128 * 140 * 3).reshape(128, 140, 3) % 251).astype(np.uint8)
        for label in range(2):
            folder = root / f"class{label}"
            folder.mkdir()
            for index in range(4):
                Image.fromarray(np.roll(pattern, label * 17 + index * 5, axis=1)).save(folder / f"{index}.jpg")

    @staticmethod
    def _augmentation_pipeline(root, version, policy, builder=None):
        builder = builder or dataset_imagenet_module.build_dali_imagenet_pipeline
        pipeline = builder(
            reader_kwargs=dict(file_root=str(root), random_shuffle=True, seed=71, name="Reader"),
            training=True, crop_size=32, batch_size=4, num_threads=1, device_id=0,
            preset_version=version, auto_augment_policy=policy,
            random_erase_prob=0.1 if version == 2 else 0.0,
            pipeline_kwargs=dict(seed=2026),
        )
        pipeline.build()
        return pipeline

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
