"""ImageNet-21k-p layout and CCT14 training integration checks."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch.utils.data import TensorDataset

from py_src.complete_ml_setup import FastTrainingSetup
from py_src.engine import Device, train, val
from py_src.ml_setup.cct import cct14_7x2_imagenet21k_p
from py_src.ml_setup.dataloader_util import DataloaderConfig
from py_src.ml_setup.factory import get_ml_setup_from_config
from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_dataset import DatasetSetup, DatasetType
from py_src.ml_setup_dataset.dataset_imagenet import dataset_imagenet21k_p
from py_src.ml_setup_model import ModelType


def _write_images(root, val_classes=("n00000001", "n00000002")):
    for split, classes in (
        ("imagenet21k_train", ("n00000001", "n00000002")),
        ("imagenet21k_val", val_classes),
    ):
        for label, name in enumerate(classes):
            class_dir = root / split / name
            class_dir.mkdir(parents=True)
            Image.new("RGB", (256, 256), (label * 100, 50, 100)).save(class_dir / "0.JPEG")


class TestImageNet21kP(unittest.TestCase):
    def test_cct14_factory_train_and_val(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_images(root)
            with patch("py_src.ml_setup_dataset.dataset_imagenet.imagenet21k_p_path", root):
                setup = get_ml_setup_from_config("cct_14_7x2_224", "imagenet21k_p")

            self.assertEqual(setup.dataset_type, DatasetType.imagenet21k_p)
            self.assertEqual(setup.model.classifier.fc.out_features, 2)
            self.assertEqual(setup.training_data.class_to_idx, setup.testing_data.class_to_idx)
            config = DataloaderConfig(batch_size=2, pin_memory=False)
            train_loader = setup.train_dataloader(config)
            val_loader = setup.val_dataloader(config)
            images, targets = next(iter(val_loader))
            self.assertEqual(tuple(images.shape), (2, 3, 224, 224))
            self.assertEqual(targets.tolist(), [0, 1])
            optimizer, scheduler, epochs = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(
                setup, setup.model, override_steps_per_epoch=len(train_loader)
            )
            self.assertEqual(epochs, 90)
            result = train(setup.adapter, train_loader, optimizer, scheduler, device=Device.cpu())
            evaluation = val(setup.adapter, val_loader, device=Device.cpu())
            self.assertEqual(result.total_count, 2)
            self.assertEqual(evaluation.total_count, 2)
            self.assertTrue(math.isfinite(result.avg_loss))
            self.assertTrue(math.isfinite(evaluation.avg_loss))

    def test_cct14_learning_rate_schedule_for_each_dataset(self):
        steps_per_epoch = 4
        for dataset_type, expected_epochs, warmup_epochs in (
            (DatasetType.imagenet21k_p, 90, 5),
            (DatasetType.imagenet1k, 300, 25),
        ):
            with self.subTest(dataset_type=dataset_type):
                model = torch.nn.Linear(2, 2)
                setup = MLSetup(model_type=ModelType.cct_14_7x2_224, dataset_type=dataset_type)
                optimizer, scheduler, epochs = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(
                    setup, model, override_steps_per_epoch=steps_per_epoch
                )
                self.assertEqual(epochs, expected_epochs)
                self.assertIsInstance(optimizer, torch.optim.AdamW)
                self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.05)
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-6)

                warmup_steps = warmup_epochs * steps_per_epoch
                total_steps = epochs * steps_per_epoch
                previous_lr = optimizer.param_groups[0]["lr"]
                for step in range(1, total_steps + 1):
                    optimizer.step()
                    scheduler.step()
                    lr = optimizer.param_groups[0]["lr"]
                    if step <= warmup_steps:
                        self.assertGreater(lr, previous_lr)
                    else:
                        self.assertLess(lr, previous_lr)
                    if step == warmup_steps:
                        self.assertAlmostEqual(lr, 5e-4)
                    if step == warmup_steps + (total_steps - warmup_steps) // 2:
                        self.assertAlmostEqual(lr, (5e-4 + 1e-5) / 2)
                    previous_lr = lr
                self.assertAlmostEqual(previous_lr, 1e-5)

    def test_full_classification_head_without_class_metadata(self):
        data = TensorDataset(torch.zeros(1, 3, 224, 224), torch.tensor([10449]))
        setup = cct14_7x2_imagenet21k_p(DatasetSetup(DatasetType.imagenet21k_p, data, data))
        self.assertEqual(setup.model.classifier.fc.out_features, 10450)

    def test_cli_spellings_and_dali_forwarding(self):
        for name in ("imagenet21k_p", "imagenet21k-p", "imagenet21kp"):
            with self.subTest(name=name), patch("py_src.ml_setup.cct.cct14_7x2_imagenet21k_p") as builder:
                get_ml_setup_from_config("cct_14_7x2_224", name, use_dali=True, dali_device_id=2)
                builder.assert_called_once_with(use_dali=True, dali_device_id=2)

    def test_dali_split_names_and_matching_classes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_images(root)
            with patch("py_src.ml_setup_dataset.dataset_imagenet.imagenet21k_p_path", root):
                setup = dataset_imagenet21k_p(use_dali=True)
            self.assertEqual(Path(setup.train_data.file_root), root / "imagenet21k_train")
            self.assertEqual(Path(setup.valdation_data.file_root), root / "imagenet21k_val")
            self.assertEqual(setup.train_data.classes, setup.valdation_data.classes)

    def test_mismatched_class_directories_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_images(root, val_classes=("n00000001", "n00000003"))
            with patch("py_src.ml_setup_dataset.dataset_imagenet.imagenet21k_p_path", root):
                for use_dali in (False, True):
                    with self.subTest(use_dali=use_dali), self.assertRaisesRegex(ValueError, "class mappings differ"):
                        dataset_imagenet21k_p(use_dali=use_dali)


if __name__ == "__main__":
    unittest.main()
