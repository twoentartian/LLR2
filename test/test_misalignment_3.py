from __future__ import annotations

import copy
import csv
from datetime import timedelta
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import misalignment_3 as measurement
import misalignment_measurement_2 as original


def make_args(*extra):
    with patch.object(sys, "argv", ["misalignment_3.py", "--dataset", "mnist", "--device", "cpu",
                                  "--num_workers", "0", "--no-compile", "--no-amp", "--no-progress",
                                  "--random_seed", "123", "--split_seed", "456", "--epochs", "3",
                                  "--angle_batch_size", "3", "--skip_relative_flatness", *extra]):
        return measurement.parse_args()


def fixture():
    generator = torch.Generator().manual_seed(17)
    inputs = torch.randn(10, 2, generator=generator)
    a = TensorDataset(inputs, torch.arange(10) % 2)
    b = TensorDataset(inputs * 1.5, 1 - torch.arange(10) % 2)
    model = nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4), nn.Dropout(.2), nn.Linear(4, 2))
    bundle = measurement.DatasetBundle(
        partition_a=a, partition_b=b,
        ml_setup=SimpleNamespace(model=model, criterion=nn.CrossEntropyLoss(), model_type=SimpleNamespace(name="toy")),
        dataset_name="mnist", dataset_type_name="mnist", source_examples=20,
        permutation=torch.arange(20), excluded_indices=torch.empty(0, dtype=torch.long), modular=False,
    )
    return model, bundle


def optimizer_builder(setup, model, partition, **kwargs):
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.9)
    return optimizer, scheduler, kwargs.get("epochs") or 3


def uneven_shard_worker(rank, init_method):
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2,
                                         timeout=timedelta(seconds=45))
    try:
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1., -2.], [.5, 1.]]))
        model.eval()
        # Also exercise a rank with no local examples; collectives must still
        # execute and retain the exact global partition mean.
        for size in (5, 1):
            x = torch.arange(size * 2, dtype=torch.float32).reshape(size, 2) / 10
            y = torch.arange(size) % 2
            dataset = TensorDataset(x, y)
            criterion = nn.CrossEntropyLoss()
            expected = torch.autograd.grad(criterion(model(x), y), list(model.parameters()))
            loader = DataLoader(dataset, batch_size=2, sampler=measurement.distributed.EvaluationSampler(dataset))
            actual, metrics = measurement._probe_partition_gradient(
                model, loader, criterion, parameters=list(model.parameters()), device=torch.device("cpu"),
                modular=False, eq_position=None, progress_enabled=False,
            )
            assert metrics.examples == size
            for a, b in zip(actual, expected, strict=True):
                torch.testing.assert_close(a, b)
    finally:
        torch.distributed.destroy_process_group()


class TestMisalignment3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_geometry_uses_raw_gradients_and_global_concatenated_angle(self):
        geometry = measurement._partition_gradient_geometry(
            ["small", "large"], [torch.tensor([1.]), torch.tensor([10.])],
            [torch.tensor([-1.]), torch.tensor([10.])],
        )
        self.assertAlmostEqual(geometry[0]["raw_gradient_angle_deg"], math.degrees(math.acos(99 / 101)))
        self.assertAlmostEqual(geometry[1]["raw_gradient_angle_deg"], 180.)
        self.assertAlmostEqual(geometry[2]["raw_gradient_angle_deg"], 0.)
        self.assertAlmostEqual(geometry[1]["descent_ascent_angle_deg"], 0.)
        for b, expected in (([-1., 0.], 180.), ([0., 1.], 90.), ([1., 0.], 0.)):
            row = measurement._partition_gradient_geometry(["p"], [torch.tensor([1., 0.])], [torch.tensor(b)])[0]
            self.assertAlmostEqual(row["raw_gradient_angle_deg"], expected)
        for values in ([0., 0.], [float("inf"), 1.]):
            row = measurement._partition_gradient_geometry(["p"], [torch.tensor(values)], [torch.ones(2)])[0]
            self.assertFalse(row["valid_angle"])
            self.assertTrue(math.isnan(row["raw_gradient_angle_deg"]))
        large = torch.full((5,), 1e20)
        row = measurement._partition_gradient_geometry(["p"], [large], [-large])[0]
        self.assertTrue(row["valid_angle"])
        self.assertAlmostEqual(row["raw_gradient_angle_deg"], 180.)

    def test_partition_gradient_matches_full_batch_with_partial_microbatch(self):
        model, bundle = fixture()
        model.eval()
        parameters = list(model.parameters())
        for p in parameters:
            p.grad = torch.full_like(p, 7.)
        x, y = bundle.partition_a.tensors
        criterion = nn.CrossEntropyLoss()
        expected = torch.autograd.grad(criterion(model(x), y), parameters)
        actual, metrics = measurement._probe_partition_gradient(
            model, DataLoader(bundle.partition_a, batch_size=3), criterion, parameters=parameters,
            device=torch.device("cpu"), modular=False, eq_position=None, progress_enabled=False,
        )
        self.assertEqual(metrics.examples, 10)
        self.assertAlmostEqual(metrics.loss, float(criterion(model(x), y).detach()), places=6)
        for a, b, p in zip(actual, expected, parameters, strict=True):
            torch.testing.assert_close(a, b)
            self.assertTrue(torch.equal(p.grad, torch.full_like(p, 7.)))

    def test_distributed_partition_gradient_uses_exact_uneven_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            init_method = (Path(directory) / "rendezvous").as_uri()
            torch.multiprocessing.spawn(uneven_shard_worker, args=(init_method,), nprocs=2, join=True)

    def test_stability_uses_raw_180_and_resets_after_zero_gradient(self):
        model, bundle = fixture()
        args = make_args("--stability_patience", "2")
        with tempfile.TemporaryDirectory() as directory:
            recorder = measurement.PartitionGradientAngleRecorder(
                model, bundle, nn.CrossEntropyLoss(), torch.device("cpu"), args, Path(directory), "stability",
            )
            templates = [[torch.ones_like(p) for p in model.parameters()],
                         [-torch.ones_like(p) for p in model.parameters()]]
            metrics = measurement.LossMetrics(loss=1., accuracy=.5, examples=10)

            def probe(*unused, **kwargs):
                index = 0 if unused[1] is recorder.loaders[0] else 1
                return templates[index], metrics

            with patch.object(measurement, "_probe_partition_gradient", side_effect=probe):
                self.assertFalse(recorder.measure()["angle_stable"])
                self.assertFalse(recorder.measure(epoch=0, phase="training")["angle_stable"])
                stable = recorder.measure(epoch=1, phase="training")
                self.assertTrue(stable["angle_stable"])
                self.assertEqual(stable["raw_gradient_angle_deg"], 180.)
                templates[1] = [torch.zeros_like(p) for p in model.parameters()]
                invalid = recorder.measure(epoch=2, phase="training")
                self.assertFalse(invalid["angle_stable"])
                self.assertEqual(invalid["stable_streak"], 0)
            with recorder.path.open() as infile:
                rows = list(csv.DictReader(infile))
            self.assertEqual(len([row for row in rows if row["parameter"] == "__all__"]), 4)
            recorder.close()

    def test_random_points_are_independent_reproducible_and_restore_training_state(self):
        model, bundle = fixture()
        model.train()
        model[-1].eval()
        modes = [module.training for module in model.modules()]
        state = copy.deepcopy(model.state_dict())
        args = make_args("--random_parameter_points", "3", "--random_parameter_radius", ".2", "--angle_samples", "7")
        initial_vector = torch.cat([p.detach().flatten() for p in model.parameters()])
        points = []
        real_measure = measurement.PartitionGradientAngleRecorder.measure

        def observe(recorder, **kwargs):
            points.append(torch.cat([p.detach().flatten() for p in model.parameters()]).clone())
            return real_measure(recorder, **kwargs)

        for p in model.parameters():
            p.grad = torch.full_like(p, 2.)
        expected_torch = torch.random.get_rng_state().clone()
        expected_python, expected_numpy = random.getstate(), np.random.get_state()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(measurement.PartitionGradientAngleRecorder, "measure", observe):
                result = measurement._run_initial_angle_probes(model, bundle, nn.CrossEntropyLoss(), torch.device("cpu"), args, root)
            repeated = measurement._run_initial_angle_probes(model, bundle, nn.CrossEntropyLoss(), torch.device("cpu"), args, root)
            self.assertEqual(result["samples_per_partition"], [7, 7])
            self.assertEqual(len(result["measurements"]), 4)
            for left, right in zip(result["measurements"], repeated["measurements"], strict=True):
                self.assertEqual(left["raw_gradient_angle_deg"], right["raw_gradient_angle_deg"])
            for point in points[1:]:
                self.assertAlmostEqual(float((point - initial_vector).norm() / initial_vector.norm()), .2, places=6)
            self.assertFalse(torch.equal(points[1], points[2]))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        for p in model.parameters():
            self.assertTrue(torch.equal(p.grad, torch.full_like(p, 2.)))
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assertTrue(torch.equal(expected_torch, torch.random.get_rng_state()))
        self.assertEqual(expected_python, random.getstate())
        np.testing.assert_equal(expected_numpy, np.random.get_state())

    def test_angle_callbacks_preserve_original_training_for_all_objectives(self):
        for objective in ("normalized_two_sided", "difference", "mean_relative_gap"):
            with self.subTest(objective=objective), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = make_args("--objective", objective, "--angle_interval", "2", "--eval_last_10_percent")
                model, bundle = fixture()
                initial = copy.deepcopy(model.state_dict())
                results, final_states = [], []
                for module in (original, measurement):
                    folder = root / module.__name__
                    folder.mkdir()
                    with patch.object(module, "build_optimizer_and_scheduler", side_effect=optimizer_builder):
                        result = module._run_direction(
                            direction_name="a_as_train_b_as_val", train_partition_name="a", val_partition_name="b",
                            train_loader=DataLoader(bundle.partition_a, batch_size=4, generator=torch.Generator().manual_seed(5)),
                            val_loader=DataLoader(bundle.partition_b, batch_size=4, generator=torch.Generator().manual_seed(6)),
                            flatness_train_loader=None, flatness_val_loader=None,
                            bundle=bundle, model=model, initial_state=initial, criterion=nn.CrossEntropyLoss(),
                            device=torch.device("cpu"), batch_size=4, epochs=3, args=args, output_folder=folder,
                            model_type_name="toy", dataset_type_name="mnist",
                        )
                    results.append(result)
                    final_states.append(copy.deepcopy(model.state_dict()))
                for name in final_states[0]:
                    torch.testing.assert_close(final_states[0][name], final_states[1][name], rtol=0, atol=0)
                self.assertEqual(results[0]["final"], results[1]["final"])
                angles = results[1]["partition_gradient_angles"]["measurements"]
                self.assertEqual([row["epoch"] for row in angles], [-1, 1, 2])
                self.assertEqual(angles[-1]["examples_a"], 10)

    def test_probe_only_main_never_builds_optimizer_and_writes_summary(self):
        model, bundle = fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "probe"
            args = make_args("--probe_only", "--random_parameter_points", "2", "-o", str(output))
            with patch.object(measurement, "parse_args", return_value=args), \
                 patch.object(measurement, "load_dataset_bundle", return_value=bundle), \
                 patch.object(measurement, "build_optimizer_and_scheduler") as optimizer:
                measurement.main()
                optimizer.assert_not_called()
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["mode"], "probe_only")
            self.assertEqual(len(summary["initial_parameter_probes"]["measurements"]), 3)
            self.assertTrue((output / "initial.model.pt").is_file())
            self.assertFalse(list(output.glob("optimization_*.csv")))

    def test_training_main_keeps_both_orientations_after_random_parameter_probes(self):
        model, bundle = fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "training"
            args = make_args("--epochs", "2", "--batch_size", "4", "--random_parameter_points", "2",
                             "--angle_interval", "2", "-o", str(output))
            with patch.object(measurement, "parse_args", return_value=args), \
                 patch.object(measurement, "load_dataset_bundle", return_value=bundle), \
                 patch.object(measurement, "build_optimizer_and_scheduler", side_effect=optimizer_builder):
                measurement.main()
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["mode"], "training")
            self.assertEqual(summary["run_order"], ["a_as_train_b_as_val", "b_as_train_a_as_val"])
            self.assertEqual(len(summary["initial_parameter_probes"]["measurements"]), 3)
            initial_angles = []
            for run in summary["runs"].values():
                angles = run["partition_gradient_angles"]["measurements"]
                self.assertEqual([row["epoch"] for row in angles], [-1, 1])
                initial_angles.append(angles[0]["raw_gradient_angle_deg"])
                self.assertTrue((output / run["files"]["final_model"]).is_file())
            self.assertEqual(initial_angles[0], initial_angles[1])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_probe_uses_full_precision_inside_amp_and_preserves_cuda_rng(self):
        model, bundle = fixture()
        device = torch.device("cuda")
        model.to(device)
        args = make_args()
        expected_state = copy.deepcopy(model.state_dict())
        model.eval()
        gradients = []
        for dataset in (bundle.partition_a, bundle.partition_b):
            x, y = [tensor.to(device) for tensor in dataset.tensors]
            gradients.append(torch.autograd.grad(nn.CrossEntropyLoss()(model(x), y), list(model.parameters())))
        expected = measurement._partition_gradient_geometry(
            [name for name, p in model.named_parameters()], *gradients,
        )[0]
        model.train()
        expected_rng = torch.cuda.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            recorder = measurement.PartitionGradientAngleRecorder(
                model, bundle, nn.CrossEntropyLoss(), device, args, Path(directory), "cuda",
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                actual = recorder.measure()
            recorder.close()
        self.assertAlmostEqual(actual["raw_gradient_angle_deg"], expected["raw_gradient_angle_deg"], places=4)
        self.assertTrue(torch.equal(expected_rng, torch.cuda.get_rng_state()))
        self.assertTrue(model.training)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
