from __future__ import annotations

import copy
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

from test.test_misalignment_token_norms import make_loader
from test.test_vit_registers import tiny_vit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import misalignment_measurement_2 as measurement


class TestMisalignmentEvalSchedule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_final_tenth_rounds_up_and_always_includes_final_epoch(self):
        for epochs, start in ((1, 0), (9, 8), (10, 9), (11, 9), (60, 54), (90, 81), (101, 90)):
            with self.subTest(epochs=epochs):
                self.assertEqual(measurement._epoch_evaluation_start(epochs, True), start)
                self.assertEqual(measurement._epoch_evaluation_start(epochs, False), 0)

    def test_cli_is_opt_in(self):
        with patch.object(sys, "argv", ["misalignment_measurement_2.py", "--dataset", "mnist"]):
            self.assertFalse(measurement.parse_args().eval_last_10_percent)
        for option in ("--eval_last_10_percent", "--eval-last-10-percent"):
            with patch.object(sys, "argv", ["misalignment_measurement_2.py", "--dataset", "mnist", option]):
                self.assertTrue(measurement.parse_args().eval_last_10_percent)

    def test_skipped_evaluations_preserve_all_training_updates(self):
        torch.manual_seed(42)
        dataset = TensorDataset(torch.randn(8, 4), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]))
        initial_model = nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6), nn.Dropout(0.2), nn.Linear(6, 3))
        states = []
        for late_only in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(123)
                model = copy.deepcopy(initial_model)
                loader = DataLoader(dataset, batch_size=4, generator=torch.Generator().manual_seed(7))
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=22)
                final_metrics = []
                rows = measurement.train_normalized_two_sided(
                    model, loader, loader, nn.CrossEntropyLoss(), device=torch.device("cpu"),
                    modular=False, eq_position=None, optimizer=optimizer, scheduler=scheduler,
                    epochs=11, report_interval=20, csv_path=str(Path(directory) / "optimization.csv"),
                    gradient_geometry_csv_path=str(Path(directory) / "geometry.csv"),
                    progress_enabled=False, eval_last_10_percent=late_only,
                    final_metrics_callback=lambda a, b: final_metrics.append((a, b)),
                )
                self.assertEqual(scheduler.last_epoch, 22)
                self.assertEqual(len(rows), 11)
                self.assertEqual(len(final_metrics), 1)
                self.assertEqual(final_metrics[0][0].examples, 8)
                self.assertEqual(final_metrics[0][0].loss, rows[-1]["loss_a"])
                states.append(copy.deepcopy(model.state_dict()))
        for name in states[0]:
            torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)

    def test_tail_evaluations_norms_checkpoints_and_final_summary(self):
        for objective in ("normalized_two_sided", "difference", "mean_relative_gap"):
            for direction, train_name, val_name in (
                ("a_as_train_b_as_val", "a", "b"), ("b_as_train_a_as_val", "b", "a"),
            ):
                with self.subTest(objective=objective, direction=direction), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    model, loader = tiny_vit(), make_loader()
                    bundle = measurement.DatasetBundle(
                        partition_a=loader.dataset, partition_b=loader.dataset,
                        ml_setup=SimpleNamespace(model_type="toy", dataset_type="mnist"),
                        dataset_name="toy", dataset_type_name="mnist", source_examples=10,
                        permutation=torch.arange(10), excluded_indices=torch.empty(0, dtype=torch.long), modular=False,
                    )
                    args = SimpleNamespace(
                        random_seed=123, split_seed=456, optimizer_preset=0, learning_rate=0.001,
                        weight_decay=0.0, optimizer="sgd", scheduler="fixed", warmup_epochs=0,
                        torch_compile=False, amp=False, report_interval=20, objective=objective,
                        train_gradient_weight_function="0.8", val_gradient_weight_function="0.2",
                        relative_flatness_layers="last_matrix_weight", relative_flatness_samples=1,
                        relative_flatness_batches=1, relative_flatness_seed=2718, augmentation="none", dataset="mnist",
                        token_norms=True, token_norm_threshold=8.0, eval_last_10_percent=True, progress=False,
                    )
                    with patch.object(measurement, "evaluate_partition", wraps=measurement.evaluate_partition) as evaluate:
                        result = measurement._run_direction(
                            direction_name=direction, train_partition_name=train_name, val_partition_name=val_name,
                            train_loader=loader, val_loader=loader, flatness_train_loader=loader, flatness_val_loader=loader,
                            bundle=bundle, model=model, initial_state=copy.deepcopy(model.state_dict()),
                            criterion=nn.CrossEntropyLoss(), device=torch.device("cpu"), batch_size=2,
                            epochs=11, args=args, output_folder=root, model_type_name="toy", dataset_type_name="mnist",
                        )
                    self.assertEqual(evaluate.call_count, 4 if objective == "normalized_two_sided" else 26)
                    post_update_calls = [call for call in evaluate.call_args_list
                                         if "baseline" not in call.kwargs["progress_label"]]
                    self.assertEqual([call.kwargs["progress_label"] for call in post_update_calls], [
                        f"{direction} epoch {epoch}/11 eval {role}"
                        for epoch in (10, 11) for role in ("train", "val")
                    ])
                    with (root / result["files"]["optimization"]).open() as infile:
                        rows = list(csv.DictReader(infile))
                    self.assertEqual(len(rows), 11)
                    self.assertEqual([int(row["evaluated"]) for row in rows], [0] * 9 + [1] * 2)
                    for row in rows[:9]:
                        for field in ("loss_a", "loss_b", "train_accuracy", "val_accuracy", "abs_accuracy_gap",
                                      "abs_loss_gap", "mean_loss", "relative_gap", "objective"):
                            self.assertTrue(math.isnan(float(row[field])), (field, row))
                        self.assertEqual(float(row["eval_train_seconds"]), 0.0)
                        self.assertEqual(float(row["eval_val_seconds"]), 0.0)
                    self.assertEqual(result["epoch_evaluation"]["policy"], "last_10_percent")
                    self.assertEqual(result["epoch_evaluation"]["first_epoch"], 9)
                    self.assertEqual(result["final_metrics_source"], "last_epoch_full_partition_evaluation")
                    self.assertEqual(result["final"]["train"]["loss"], result["optimization_last_row"]["loss_a"])
                    self.assertEqual(result["final"]["val"]["loss"], result["optimization_last_row"]["loss_b"])
                    self.assertEqual({point["epoch"] for point in result["gap_checkpoints"]["union"]}, {9, 10})
                    with (root / result["files"]["token_norms"]).open() as infile:
                        norm_rows = list(csv.DictReader(infile))
                    self.assertEqual({int(row["epoch"]) for row in norm_rows}, {9, 10})
                    self.assertEqual(len(norm_rows), 2 * 2 * 6)
                    with (root / result["files"]["weight_variance"]).open() as infile:
                        self.assertEqual([int(row["epoch"]) for row in csv.DictReader(infile)], list(range(11)))


if __name__ == "__main__":
    unittest.main()
