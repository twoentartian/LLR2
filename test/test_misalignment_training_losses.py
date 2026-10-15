from __future__ import annotations

import csv
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import misalignment_measurement_2 as measurement


def loss_fixture():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2., -1.], [-1., 2.]]))
    images = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., 0.], [0., 10.]])
    dataset_a = TensorDataset(images, torch.tensor([0, 1, 0, 0, 0]))
    dataset_b = TensorDataset(images, torch.tensor([1, 0, 1, 1, 1]))
    return model, DataLoader(dataset_a, batch_size=2), DataLoader(dataset_b, batch_size=2)


class TestMisalignmentTrainingLosses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_partial_batches_are_sample_weighted_and_logged_when_eval_is_skipped(self):
        for objective in ("normalized_two_sided", "difference", "mean_relative_gap"):
            with self.subTest(objective=objective), tempfile.TemporaryDirectory() as directory:
                model, loader_a, loader_b = loss_fixture()
                criterion = nn.CrossEntropyLoss()
                expected = [float(criterion(model(loader.dataset.tensors[0]), loader.dataset.tensors[1]).detach())
                            for loader in (loader_a, loader_b)]
                unweighted = [sum(float(criterion(model(x), y).detach()) for x, y in loader) / len(loader)
                              for loader in (loader_a, loader_b)]
                self.assertFalse(math.isclose(expected[0], unweighted[0], rel_tol=0.01))
                options = dict(
                    device=torch.device("cpu"), modular=False, eq_position=None,
                    optimizer=torch.optim.SGD(model.parameters(), lr=0.), scheduler=None,
                    epochs=2, report_interval=1, csv_path=str(Path(directory) / "optimization.csv"),
                    progress_enabled=False, eval_last_10_percent=True,
                )
                with self.assertLogs("misalignment_measurement_2", level="INFO") as logs, \
                     patch.object(measurement, "_loss_and_accuracy", wraps=measurement._loss_and_accuracy) as forwards:
                    if objective == "normalized_two_sided":
                        rows = measurement.train_normalized_two_sided(
                            model, loader_a, loader_b, criterion,
                            gradient_geometry_csv_path=str(Path(directory) / "geometry.csv"), **options,
                        )
                    else:
                        rows = measurement.train_symmetric_objective(
                            model, loader_a, loader_b, criterion, objective_mode=objective, **options,
                        )
                # Logging reuses the loss from each existing training forward.
                self.assertEqual(forwards.call_count, 18 if objective == "normalized_two_sided" else 30)
                self.assertEqual([row["evaluated"] for row in rows], [0., 1.])
                self.assertTrue(math.isnan(rows[0]["loss_a"]))
                for row in rows:
                    for field, value in zip(measurement.TRAINING_LOSS_FIELDS, expected, strict=True):
                        self.assertAlmostEqual(row[field], value, places=6)
                epoch_logs = [line for line in logs.output if "epoch " in line]
                self.assertEqual(len(epoch_logs), 2)
                for line in epoch_logs:
                    self.assertIn("running_train_loss=", line)
                    self.assertIn("running_val_loss=", line)
                self.assertIn("eval skipped", epoch_logs[0])
                self.assertIn("eval_train_loss=", epoch_logs[-1])
                self.assertIn("eval_val_loss=", epoch_logs[-1])
                with Path(options["csv_path"]).open() as infile:
                    csv_rows = list(csv.DictReader(infile))
                for row in csv_rows:
                    for field, value in zip(measurement.TRAINING_LOSS_FIELDS, expected, strict=True):
                        self.assertTrue(math.isclose(float(row[field]), value, rel_tol=1e-4))

    def test_zero_val_weight_keeps_validation_training_forward_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            model, loader_a, loader_b = loss_fixture()
            with patch.object(measurement, "_loss_and_accuracy", wraps=measurement._loss_and_accuracy) as forwards:
                rows = measurement.train_normalized_two_sided(
                    model, loader_a, loader_b, nn.CrossEntropyLoss(), device=torch.device("cpu"),
                    modular=False, eq_position=None, optimizer=torch.optim.SGD(model.parameters(), lr=0.),
                    scheduler=None, epochs=2, report_interval=1,
                    csv_path=str(Path(directory) / "optimization.csv"),
                    gradient_geometry_csv_path=str(Path(directory) / "geometry.csv"),
                    progress_enabled=False, eval_last_10_percent=True,
                    train_gradient_weight_function="1", val_gradient_weight_function="0",
                )
            self.assertEqual(forwards.call_count, 12)  # Six train + six final eval forwards.
            for row in rows:
                self.assertTrue(math.isfinite(row["running_train_loss"]))
                self.assertTrue(math.isnan(row["running_val_loss"]))
            self.assertTrue(math.isfinite(rows[-1]["loss_b"]))

    def test_loss_scaling_does_not_scale_reported_losses_or_change_gradients(self):
        model, loader, _ = loss_fixture()
        batch = next(iter(loader))
        criterion = nn.CrossEntropyLoss()
        raw_loss = float(criterion(model(batch[0]), batch[1]).detach())
        options = dict(
            device=torch.device("cpu"), modular=False, eq_position=None,
            parameters=list(model.parameters()), scaler=SimpleNamespace(scale=lambda loss: loss * 128),
        )
        reference = measurement._compute_batch_gradients(model, batch, criterion, **options)
        total = torch.zeros((), dtype=torch.float64)
        observed = measurement._compute_batch_gradients(model, batch, criterion, loss_total=total, **options)
        self.assertEqual(float(total), raw_loss * len(batch[0]))
        self.assertFalse(total.requires_grad)
        for old, new in zip(reference, observed, strict=True):
            torch.testing.assert_close(old, new, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
