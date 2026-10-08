from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import misalignment_cct_entry as entry


class TestMisalignmentCctEntry(unittest.TestCase):
    def setup_optimizer(self, model_name="cct_14_7x2_224", dataset_name="imagenet1k", **options):
        setup = SimpleNamespace(
            model_type=SimpleNamespace(name=model_name),
            dataset_type=SimpleNamespace(name=dataset_name),
        )
        model = nn.Linear(2, 2)
        dataset = TensorDataset(torch.zeros(10, 2), torch.zeros(10, dtype=torch.long))
        return entry.build_optimizer_and_scheduler(
            setup, model, dataset, batch_size=4, **options,
        )

    @staticmethod
    def advance(optimizer, scheduler, steps):
        for _ in range(steps):
            optimizer.step()
            scheduler.step()

    def test_imagenet_augmentation_and_metadata_always_use_224(self):
        image = Image.new("RGB", (280, 260), (120, 90, 60))
        for level in ("none", "1", "2"):
            with self.subTest(level=level):
                transform = entry.build_augmentation(SimpleNamespace(), "imagenet1k", level)
                self.assertEqual(tuple(transform(image).shape), (3, 224, 224))
                metadata = entry.describe_augmentation("imagenet1k", level)
                self.assertEqual(metadata["input_size"], 224)
                self.assertTrue(metadata["shared_between_partitions"])

    def test_cifar_augmentation_is_unchanged(self):
        image = Image.new("RGB", (32, 32))
        transform = entry.build_augmentation(SimpleNamespace(), "cifar10", 2)
        self.assertEqual(tuple(transform(image).shape), (3, 32, 32))
        self.assertEqual(entry.describe_augmentation("cifar10", 2)["input_size"], 32)

    def test_imagenet_optimizer_defaults_follow_cct_recipe(self):
        optimizer, scheduler, epochs = self.setup_optimizer()
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
        self.assertEqual(epochs, 300)
        self.assertEqual(optimizer.defaults["lr"], 5e-4)
        self.assertEqual(optimizer.defaults["weight_decay"], 5e-2)
        self.assertEqual(optimizer.misalignment_config["warmup_epochs"], 25)
        self.assertEqual(optimizer.misalignment_config["updates_per_epoch"], 3)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-6)

    def test_cifar10_cct_presets_are_preserved(self):
        for preset, lr, wd in ((0, 5.5e-4, 6e-2), (1, 1e-3, 1e-2)):
            with self.subTest(preset=preset):
                optimizer, _, epochs = self.setup_optimizer("cct_7_3x1_32", "cifar10", preset=preset)
                self.assertEqual(epochs, 300)
                self.assertEqual(optimizer.defaults["lr"], lr)
                self.assertEqual(optimizer.defaults["weight_decay"], wd)
                self.assertEqual(optimizer.misalignment_config["warmup_epochs"], 10)

    def test_cosine_uses_minibatch_steps_and_cct_endpoints(self):
        optimizer, scheduler, epochs = self.setup_optimizer(epochs=6, warmup_epochs=2, scheduler="cosine")
        self.assertEqual(epochs, 6)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-6)
        self.advance(optimizer, scheduler, 6)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-4)
        self.advance(optimizer, scheduler, 6)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], (5e-4 + 1e-5) / 2)
        self.advance(optimizer, scheduler, 6)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)
        self.advance(optimizer, scheduler, 3)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)

    def test_explicit_launcher_options_override_defaults(self):
        optimizer, _, epochs = self.setup_optimizer(
            optimizer="adamw", learning_rate=5e-4, weight_decay=5e-2,
            epochs=100, warmup_epochs=5, scheduler="cosine",
        )
        self.assertEqual(epochs, 100)
        self.assertEqual(optimizer.misalignment_config["warmup_epochs"], 5)
        self.assertEqual(optimizer.defaults["lr"], 5e-4)
        self.assertEqual(optimizer.defaults["weight_decay"], 5e-2)

    def test_fixed_schedule_and_zero_warmup(self):
        optimizer, scheduler, _ = self.setup_optimizer(
            learning_rate=2e-4, weight_decay=0.02, epochs=4, warmup_epochs=1, scheduler="fixed",
        )
        self.advance(optimizer, scheduler, 3)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2e-4)
        self.advance(optimizer, scheduler, 12)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2e-4)
        self.assertEqual(optimizer.misalignment_config["minimum_learning_rate"], 2e-4)
        optimizer, scheduler, _ = self.setup_optimizer(warmup_epochs=0, scheduler="fixed")
        self.assertIsNone(scheduler)
        self.assertEqual(optimizer.param_groups[0]["lr"], 5e-4)

    def test_other_models_delegate_to_existing_factory(self):
        optimizer, _, epochs = self.setup_optimizer("resnet18_bn", "cifar10")
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(epochs, 70)
        self.assertEqual(optimizer.defaults["lr"], 0.1)
        self.assertNotIn("recipe_source", optimizer.misalignment_config)

    def test_cli_installs_adapters_and_delegates_to_measurement(self):
        measurement = entry.measurement
        with patch.object(measurement, "build_augmentation"), \
             patch.object(measurement, "describe_augmentation"), \
             patch.object(measurement, "build_optimizer_and_scheduler"), \
             patch.object(measurement, "main") as run:
            entry.main()
            self.assertIs(measurement.build_augmentation, entry.build_augmentation)
            self.assertIs(measurement.describe_augmentation, entry.describe_augmentation)
            self.assertIs(measurement.build_optimizer_and_scheduler, entry.build_optimizer_and_scheduler)
            run.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
