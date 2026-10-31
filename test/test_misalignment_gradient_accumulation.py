from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import misalignment_measurement_2 as measurement


def run(objective: str, batch_size: int, accumulate: int, directory: str):
    torch.manual_seed(0)
    model = nn.Linear(3, 4)
    generator = torch.Generator().manual_seed(1)
    images = torch.randn(10, 3, generator=generator)
    loader_a = DataLoader(TensorDataset(images, torch.randint(0, 4, (10,), generator=generator)), batch_size=batch_size)
    loader_b = DataLoader(TensorDataset(images.flip(0), torch.randint(0, 4, (10,), generator=generator)),
                          batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (1 + step))
    options = dict(
        device=torch.device("cpu"), modular=False, eq_position=None, optimizer=optimizer,
        scheduler=scheduler, epochs=2, report_interval=1, csv_path=str(Path(directory) / f"{objective}.csv"),
        progress_enabled=False, gradient_accumulate_every=accumulate,
    )
    if objective == "normalized_two_sided":
        rows = measurement.train_normalized_two_sided(
            model, loader_a, loader_b, nn.CrossEntropyLoss(),
            gradient_geometry_csv_path=str(Path(directory) / "geometry.csv"),
            train_gradient_weight_function="0.7", val_gradient_weight_function="0.3", **options,
        )
    else:
        rows = measurement.train_symmetric_objective(
            model, loader_a, loader_b, nn.CrossEntropyLoss(), objective_mode=objective, **options,
        )
    return model, rows, scheduler


class TestMisalignmentGradientAccumulation(unittest.TestCase):
    def test_accumulated_microbatches_match_one_large_batch(self):
        # 10 examples: batches 4+4+2 versus microbatches 2+2 | 2+2 | 2, so
        # the short final group is applied as its own update in both runs.
        for objective in ("normalized_two_sided", "difference", "mean_relative_gap"):
            with self.subTest(objective=objective), tempfile.TemporaryDirectory() as directory:
                reference, reference_rows, reference_scheduler = run(objective, 4, 1, directory)
                accumulated, rows, scheduler = run(objective, 2, 2, directory)
                for expected, observed in zip(reference.parameters(), accumulated.parameters(), strict=True):
                    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
                self.assertEqual([row["optimizer_updates"] for row in rows], [3., 3.])
                self.assertEqual([row["batches"] for row in rows], [5., 5.])
                self.assertEqual([row["batches"] for row in reference_rows], [3., 3.])
                self.assertEqual(scheduler.last_epoch, reference_scheduler.last_epoch)
                for row, reference_row in zip(rows, reference_rows, strict=True):
                    self.assertAlmostEqual(row["loss_a"], reference_row["loss_a"], places=5)
                    self.assertAlmostEqual(row["running_train_loss"], reference_row["running_train_loss"], places=6)

    def test_scheduler_counts_updates_after_accumulation(self):
        from misalignment.optimizer import build_optimizer_and_scheduler
        from types import SimpleNamespace

        setup = SimpleNamespace(model_type="convnext_tiny", dataset_type="imagenet21k_p")
        model = nn.Linear(2, 2)
        dataset = list(range(100))
        optimizer, _, epochs = build_optimizer_and_scheduler(
            setup, model, dataset, batch_size=4, gradient_accumulate_every=8, scheduler="cosine",
        )
        config = optimizer.misalignment_config
        self.assertEqual((config["updates_per_epoch"], epochs), (4, 90))  # ceil(ceil(100/4)/8)
        self.assertEqual(config["learning_rate"], 4e-3)
        self.assertAlmostEqual(config["minimum_learning_rate"], 1e-6)
        self.assertEqual([group["weight_decay"] for group in optimizer.param_groups], [0.05, 0.0])

    def test_convnext_imagenet1k_uses_paper_recipe(self):
        from misalignment.optimizer import build_optimizer_and_scheduler
        from types import SimpleNamespace

        setup = SimpleNamespace(model_type="convnext_tiny", dataset_type="imagenet1k")
        optimizer, _, epochs = build_optimizer_and_scheduler(
            setup, nn.Linear(2, 2), list(range(100)), batch_size=4, gradient_accumulate_every=8,
            scheduler="cosine",
        )
        config = optimizer.misalignment_config
        self.assertEqual((config["optimizer"], epochs, config["warmup_epochs"]), ("adamw", 300, 20))
        self.assertEqual((config["learning_rate"], config["weight_decay"]), (4e-3, 0.05))
        self.assertAlmostEqual(config["minimum_learning_rate"], 1e-6)
        self.assertEqual([group["weight_decay"] for group in optimizer.param_groups], [0.05, 0.0])

        resnet = SimpleNamespace(model_type="resnet18_bn", dataset_type="imagenet1k")
        optimizer, _, _ = build_optimizer_and_scheduler(resnet, nn.Linear(2, 2), list(range(100)), batch_size=4)
        self.assertEqual(optimizer.misalignment_config["optimizer"], "sgd")


class TestMisalignmentLabelSmoothing(unittest.TestCase):
    def test_explicit_smoothing_overrides_preset_criterion(self):
        preset = nn.CrossEntropyLoss(label_smoothing=0.11)
        self.assertIs(measurement._resolve_criterion(preset, None), preset)
        self.assertEqual(measurement._resolve_criterion(None, None).label_smoothing, 0.0)
        self.assertEqual(measurement._resolve_criterion(nn.CrossEntropyLoss(), 0.1).label_smoothing, 0.1)
        self.assertEqual(measurement._resolve_criterion(preset, 0.0).label_smoothing, 0.0)
        with self.assertRaises(ValueError):
            measurement._resolve_criterion(nn.NLLLoss(), 0.1)


class TestConvNeXtStochasticDepth(unittest.TestCase):
    def test_override_matches_torchvision_construction(self):
        from torchvision.models import convnext_tiny

        for probability in (0.0, 0.3):
            model = convnext_tiny(num_classes=10, stochastic_depth_prob=0.1)
            measurement.set_convnext_stochastic_depth(model, probability)
            reference = convnext_tiny(num_classes=10, stochastic_depth_prob=probability)
            rates = [module.p for module in model.modules() if hasattr(module, "p")]
            reference_rates = [module.p for module in reference.modules() if hasattr(module, "p")]
            self.assertEqual(len(rates), 18)
            for rate, reference_rate in zip(rates, reference_rates, strict=True):
                self.assertAlmostEqual(rate, reference_rate)
            self.assertAlmostEqual(measurement._stochastic_depth_prob(model), probability)

    def test_rejects_models_without_convnext_blocks(self):
        with self.assertRaises(ValueError):
            measurement.set_convnext_stochastic_depth(nn.Linear(2, 2), 0.0)
        self.assertIsNone(measurement._stochastic_depth_prob(nn.Linear(2, 2)))


if __name__ == "__main__":
    unittest.main()
