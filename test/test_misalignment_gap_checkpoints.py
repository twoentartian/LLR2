from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# The measurement script imports its local helpers as `misalignment.*`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import misalignment_measurement_2 as measurement
from misalignment.checkpoints import GapCheckpointTracker
from py_src.model_opti_save_load import load_model_state_file, save_model_state


def metric_row(epoch, accuracy_gap, loss_gap):
    return {
        "epoch": float(epoch),
        "loss_a": 2.0 + max(loss_gap, 0.0),
        "loss_b": 2.0 + max(-loss_gap, 0.0),
        "train_accuracy": 0.5 + accuracy_gap / 2,
        "val_accuracy": 0.5 - accuracy_gap / 2,
        "learning_rate": 0.0123456789,
    }


def fake_flatness_result(value):
    rows = [
        {"role": role, "partition": partition, "layer": "weight", "signed_estimate": value}
        for role, partition in (("train", "a"), ("val", "b"))
    ]
    report = {
        "layers": ["weight"], "samples": 8, "batch_size": 4, "batches": -1,
        "symmetric_by_layer": {
            "weight": {"mean_signed_estimate": value, "mean_positive_part": max(0, value)},
        },
    }
    return report, {"overall_mean": value}, rows


class TestGapCheckpointTracker(unittest.TestCase):
    def test_top_five_rankings_union_and_immutable_parameters_and_buffers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = nn.Linear(2, 2)
            model.register_buffer("epoch_buffer", torch.tensor(-1))
            tracker = GapCheckpointTracker(model, root, "forward", "toy", "mnist")
            rows = []
            for epoch in range(22):
                # The two top-5 sets are different; signed differences alternate.
                row = metric_row(epoch, (-1) ** epoch * epoch / 25, (-1) ** epoch * (22 - epoch))
                rows.append(row)
                with torch.no_grad():
                    model.weight.fill_(epoch)
                    model.bias.fill_(-epoch)
                    model.epoch_buffer.fill_(epoch + 100)
                tracker.observe(row)
                self.assertLessEqual(len(list(tracker.checkpoint_folder.glob("*.model.pt"))), 10)

            expected_accuracy = sorted(
                rows, key=lambda row: (-abs(row["train_accuracy"] - row["val_accuracy"]), row["epoch"])
            )[:5]
            expected_loss = sorted(
                rows, key=lambda row: (-abs(row["loss_a"] - row["loss_b"]), row["epoch"])
            )[:5]
            selection = tracker.selection()
            self.assertEqual([point["epoch"] for point in selection["top_accuracy_gap"]],
                             [int(row["epoch"]) for row in expected_accuracy])
            self.assertEqual([point["epoch"] for point in selection["top_loss_gap"]],
                             [int(row["epoch"]) for row in expected_loss])
            self.assertEqual(len(selection["union"]), 10)
            self.assertEqual(json.loads(tracker.manifest_path.read_text()), selection)
            with torch.no_grad():
                model.weight.fill_(999)
                model.epoch_buffer.fill_(999)
            for point in selection["union"]:
                state, model_name, dataset_name = load_model_state_file(str(root / point["checkpoint"]), map_location="cpu")
                self.assertEqual((model_name, dataset_name), ("toy", "mnist"))
                self.assertTrue(torch.equal(state["weight"], torch.full_like(model.weight, point["epoch"])))
                self.assertEqual(state["epoch_buffer"].item(), point["epoch"] + 100)
                self.assertFalse(any(name.startswith("_orig_mod.") for name in state))

    def test_ties_keep_earlier_epochs_and_overlap_has_only_one_file(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = GapCheckpointTracker(nn.Linear(1, 1), Path(directory), "forward", "toy", "mnist")
            for epoch in range(12):
                tracker.observe(metric_row(epoch, 0.2, -3.0))
            selection = tracker.selection()
            self.assertEqual([point["epoch"] for point in selection["union"]], list(range(5)))
            self.assertEqual(len(list(tracker.checkpoint_folder.glob("*.model.pt"))), 5)
            for point in selection["union"]:
                self.assertEqual(point["selected_by"], ["accuracy_gap", "loss_gap"])

    def test_checkpoint_removed_only_after_leaving_both_rankings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracker = GapCheckpointTracker(nn.Linear(1, 1), root, "forward", "toy", "mnist", top_k=1)
            tracker.observe(metric_row(0, 0.9, 1))
            path0 = root / tracker.selection()["union"][0]["checkpoint"]
            tracker.observe(metric_row(1, 0.1, 10))
            self.assertTrue(path0.exists())  # Still accuracy top-1.
            tracker.observe(metric_row(2, 0.95, 2))
            self.assertFalse(path0.exists())
            self.assertEqual([point["epoch"] for point in tracker.selection()["union"]], [1, 2])
            tracker.observe(metric_row(3, 0.2, 20))
            self.assertEqual([point["epoch"] for point in tracker.selection()["union"]], [2, 3])
            self.assertEqual(len(list(tracker.checkpoint_folder.glob("*.model.pt"))), 2)

    def test_nonfinite_metrics_are_skipped_and_directions_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = nn.Linear(1, 1)
            first = GapCheckpointTracker(model, root, "forward", "toy", "mnist")
            second = GapCheckpointTracker(model, root, "reverse", "toy", "mnist")
            first.observe(metric_row(0, 0.1, 2))
            invalid = metric_row(1, 0.2, 3)
            invalid["loss_b"] = math.inf
            first.observe(invalid)
            second.observe(metric_row(0, -0.2, -3))
            self.assertEqual(len(first.selection()["union"]), 1)
            self.assertNotEqual(first.manifest_path, second.manifest_path)
            self.assertTrue(second.manifest_path.exists())

    def test_union_flatness_is_measured_once_and_final_state_is_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = nn.Linear(1, 1)
            tracker = GapCheckpointTracker(model, root, "forward", "toy", "mnist")
            for epoch in range(6):
                with torch.no_grad():
                    model.weight.fill_(epoch)
                tracker.observe(metric_row(epoch, epoch / 10, 6 - epoch))
            final_path = root / "final.model.pt"
            save_model_state(str(final_path), model.state_dict(), "toy", "mnist")
            measured = []

            def measure_model():
                value = float(model.weight.item())
                measured.append(value)
                return fake_flatness_result(value)

            result = measurement._measure_gap_checkpoint_union(
                tracker=tracker, model=model, direction_name="forward", final_epoch=5,
                final_checkpoint_path=final_path, final_result=fake_flatness_result(5.0),
                measure_model=measure_model,
            )
            self.assertEqual(measured, [0., 1., 2., 3., 4.])  # Final epoch reused.
            self.assertEqual(len(result["union"]), 6)
            self.assertEqual(model.weight.item(), 5.)
            saved = json.loads((root / "selected_checkpoints_forward.json").read_text())
            self.assertEqual(saved, result)
            with (root / "selected_relative_flatness_forward.csv").open() as infile:
                rows = list(csv.DictReader(infile))
            self.assertEqual(len(rows), 12)  # Two roles, not duplicate selections.
            self.assertEqual(rows[0]["learning_rate"], "0.012346")  # Five significant digits.

    def test_flatness_failure_restores_final_state_and_keeps_completed_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = nn.Linear(1, 1)
            tracker = GapCheckpointTracker(model, root, "forward", "toy", "mnist")
            for epoch in range(2):
                with torch.no_grad():
                    model.weight.fill_(epoch)
                tracker.observe(metric_row(epoch, epoch / 10, 2 - epoch))
            with torch.no_grad():
                model.weight.fill_(9)
            final_path = root / "final.model.pt"
            save_model_state(str(final_path), model.state_dict(), "toy", "mnist")

            def measure_model():
                if model.weight.item() == 1:
                    raise RuntimeError("simulated Hessian failure")
                return fake_flatness_result(0.)

            with self.assertRaisesRegex(RuntimeError, "simulated Hessian"):
                measurement._measure_gap_checkpoint_union(
                    tracker=tracker, model=model, direction_name="forward", final_epoch=9,
                    final_checkpoint_path=final_path, final_result=fake_flatness_result(9.),
                    measure_model=measure_model,
                )
            self.assertEqual(model.weight.item(), 9.)
            saved = json.loads((root / "selected_checkpoints_forward.json").read_text())
            self.assertEqual([point["epoch"] for point in saved["union"]], [0])


class TestMisalignmentCheckpointIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_real_training_and_hessian_measurement_for_all_objectives_and_directions(self):
        torch.manual_seed(123)
        dataset_a = TensorDataset(torch.randn(8, 4), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]))
        dataset_b = TensorDataset(torch.randn(8, 4), torch.tensor([2, 1, 0, 2, 1, 0, 2, 1]))
        loader_a = DataLoader(dataset_a, batch_size=4)
        loader_b = DataLoader(dataset_b, batch_size=4)
        model = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 3))
        initial_state = {name: value.clone() for name, value in model.state_dict().items()}
        bundle = measurement.DatasetBundle(
            partition_a=dataset_a, partition_b=dataset_b,
            ml_setup=SimpleNamespace(model_type="toy", dataset_type="mnist"),
            dataset_name="toy_mnist", dataset_type_name="mnist", source_examples=16,
            permutation=torch.arange(16), excluded_indices=torch.empty(0, dtype=torch.long), modular=False,
        )
        args = SimpleNamespace(
            random_seed=123, split_seed=456, optimizer_preset=0, learning_rate=0.01,
            weight_decay=0., optimizer="sgd", scheduler="fixed", warmup_epochs=0,
            torch_compile=False, amp=False, report_interval=1,
            train_gradient_weight_function="0.5", val_gradient_weight_function="0.5",
            relative_flatness_layers="last_matrix_weight", relative_flatness_samples=2,
            relative_flatness_batches=-1, relative_flatness_seed=2718,
            augmentation="none", dataset="mnist",
        )
        for objective in ("normalized_two_sided", "difference", "mean_relative_gap"):
            with self.subTest(objective=objective), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args.objective = objective
                for direction, train_name, val_name, train_loader, val_loader in (
                    ("a_as_train_b_as_val", "a", "b", loader_a, loader_b),
                    ("b_as_train_a_as_val", "b", "a", loader_b, loader_a),
                ):
                    result = measurement._run_direction(
                        direction_name=direction, train_partition_name=train_name, val_partition_name=val_name,
                        train_loader=train_loader, val_loader=val_loader,
                        flatness_train_loader=train_loader, flatness_val_loader=val_loader,
                        bundle=bundle, model=model, initial_state=initial_state, criterion=nn.CrossEntropyLoss(),
                        device=torch.device("cpu"), batch_size=4, epochs=6, args=args, output_folder=root,
                        model_type_name="toy", dataset_type_name="mnist",
                    )
                    selected = result["gap_checkpoints"]
                    self.assertEqual(len(selected["top_accuracy_gap"]), 5)
                    self.assertEqual(len(selected["top_loss_gap"]), 5)
                    self.assertLessEqual(len(selected["union"]), 10)
                    self.assertEqual(len({point["epoch"] for point in selected["union"]}), len(selected["union"]))
                    for point in selected["union"]:
                        self.assertTrue((root / point["checkpoint"]).is_file())
                        report = point["relative_flatness"]
                        self.assertEqual(report["partitions"]["train"], report["partitions"][train_name])
                        self.assertAlmostEqual(report["partitions"]["train"][0]["loss"], point["train_loss"], places=6)
                        self.assertAlmostEqual(report["partitions"]["val"][0]["loss"], point["val_loss"], places=6)
                        self.assertEqual(report["partitions"]["train"][0]["examples"], 8)
                        self.assertTrue(math.isfinite(point["misalignment_measure"]["overall_mean"]))
                    for key in ("initial_model", "final_model", "relative_flatness", "weight_variance",
                                "gap_checkpoint_selection", "selected_relative_flatness", "selected_checkpoint_summary"):
                        self.assertTrue((root / result["files"][key]).is_file())
                    final_state, _, _ = load_model_state_file(str(root / result["files"]["final_model"]), map_location="cpu")
                    for name, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, final_state[name]))
                    with (root / result["files"]["optimization"]).open() as infile:
                        rows = list(csv.DictReader(infile))
                    self.assertEqual(len(rows), 6)
                    self.assertIn("abs_accuracy_gap", rows[0])
                    self.assertAlmostEqual(float(rows[-1]["loss_a"]), result["final"]["train"]["loss"], places=4)
                    with (root / result["files"]["weight_variance"]).open() as infile:
                        variance_rows = list(csv.DictReader(infile))
                    self.assertEqual([int(row["epoch"]) for row in variance_rows], list(range(6)))
                    weight_names = [name for name in final_state if "weight" in name]
                    self.assertEqual(list(variance_rows[0]), ["epoch", *weight_names])
                    self.assertEqual(result["weight_variance"]["layers"], weight_names)
                    # Check every epoch against its captured model, not just
                    # the final model (also catches callback timing mistakes).
                    for point in selected["union"]:
                        state, _, _ = load_model_state_file(str(root / point["checkpoint"]), map_location="cpu")
                        variance_row = variance_rows[point["epoch"]]
                        for name in weight_names:
                            expected = f"{torch.var(state[name]).item():.3E}"
                            self.assertEqual(variance_row[name], expected)


if __name__ == "__main__":
    unittest.main()
