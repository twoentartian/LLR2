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
from misalignment.gradient_geometry import GRADIENT_GEOMETRY_FIELDS, GradientGeometryAccumulator
from misalignment.tokenizer_stats import TokenizerOutputMonitor


def legacy_combination(parameters, train_gradients, val_gradients, train_weight, val_weight):
    """The pre-optimization update/statistics, used only as a reference."""
    combined, rows = [], []
    for parameter, train, val in zip(parameters, train_gradients, val_gradients, strict=True):
        train_norm, val_norm = float(train.norm().item()), float(val.norm().item())
        parameter_norm = float(parameter.detach().norm().item())
        if train_norm == 0.:
            gradient, cosine = torch.zeros_like(train), float("nan")
        elif val_norm == 0.:
            gradient, cosine = train.clone(), float("nan")
        else:
            direction = train_weight * (train / train_norm) - val_weight * (val / val_norm)
            direction_norm = float(direction.norm().item())
            gradient = direction * (train_norm / direction_norm) if direction_norm else torch.zeros_like(train)
            cosine = float((train * val).sum().item()) / (train_norm * val_norm)
        combined_norm = float(gradient.norm().item())
        combined.append(gradient)
        rows.append([
            train_norm, val_norm, combined_norm, parameter_norm, cosine,
            train_norm / max(parameter_norm, 1e-12), combined_norm / max(parameter_norm, 1e-12),
        ])
    return combined, torch.tensor(rows, dtype=torch.float64, device=parameters[0].device)


class RecordingLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.output_dtypes = []

    def forward(self, inputs):
        result = self.linear(inputs)
        self.output_dtypes.append(result.dtype)
        return result


class TestMisalignmentPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def devices(self):
        return [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])

    def test_combination_matches_reference_including_zero_cancellation_and_overflow(self):
        for device in self.devices():
            torch.manual_seed(31)
            model = nn.ParameterList([nn.Parameter(torch.randn(4, 3, device=device)) for _ in range(7)])
            model.append(nn.Parameter(torch.ones(1, device=device), requires_grad=False))
            parameters = [p for p in model.parameters() if p.requires_grad]
            train = [torch.randn_like(p) for p in parameters]
            val = [torch.randn_like(p) for p in parameters]
            train[1].zero_()
            val[2].zero_()
            val[3] = train[3].clone()
            val[4] = -train[4]
            train[5].zero_()
            val[5].zero_()
            # Finite elements whose FP32 norm overflows; exact-zero signed
            # directions must still yield zero rather than 0 * inf -> NaN.
            train[6].fill_(1e20)
            val[6].copy_(train[6])
            with torch.no_grad():
                parameters[0].zero_()
            for train_weight, val_weight in ((.5, .5), (.8, .2), (1., 0.), (0., 1.), (0., 0.)):
                with self.subTest(device=device, train_weight=train_weight, val_weight=val_weight):
                    expected, expected_stats = legacy_combination(parameters, train, val, train_weight, val_weight)
                    actual, stats = measurement._combine_normalized_train_val_gradients(
                        model, train, val, parameters=parameters,
                        train_weight=train_weight, val_weight=val_weight,
                    )
                    for result, reference in zip(actual, expected, strict=True):
                        torch.testing.assert_close(result, reference, rtol=2e-6, atol=2e-7, equal_nan=True)
                        self.assertFalse(result.requires_grad)
                    torch.testing.assert_close(stats, expected_stats, rtol=2e-6, atol=2e-7, equal_nan=True)
                    self.assertEqual(stats.device, parameters[0].device)
                    reduced, cosines = measurement._combine_normalized_train_val_gradients(
                        model, train, val, parameters=parameters,
                        train_weight=train_weight, val_weight=val_weight, collect_geometry=False,
                    )
                    for result, reference in zip(reduced, actual, strict=True):
                        self.assertTrue(torch.equal(result, reference))
                    torch.testing.assert_close(cosines[:, 0], stats[:, 4], equal_nan=True)

    def test_hot_loop_has_no_scalar_reads_or_host_transfers(self):
        for device in self.devices():
            with self.subTest(device=device):
                model = nn.Linear(4, 3).to(device)
                parameters = list(model.parameters())
                train = [torch.randn_like(p) for p in parameters]
                val = [torch.randn_like(p) for p in parameters]
                names = [name for name, _ in model.named_parameters()]
                accumulator = GradientGeometryAccumulator(names, GRADIENT_GEOMETRY_FIELDS, device)
                # Fail even on CPU, so GPU synchronization cannot be hidden
                # in a Python float/bool conversion instead of .item().
                with patch.object(torch.Tensor, "item", side_effect=AssertionError("scalar read")), \
                     patch.object(torch.Tensor, "__float__", side_effect=AssertionError("float read")), \
                     patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("bool read")), \
                     patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host copy")), \
                     patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host list")):
                    for _ in range(3):
                        _, stats = measurement._combine_normalized_train_val_gradients(
                            model, train, val, parameters=parameters, train_weight=.8, val_weight=.2,
                        )
                        accumulator.update(stats)
                rows = accumulator.averages()
                self.assertEqual(len(rows), len(parameters))
                self.assertTrue(math.isfinite(rows[0]["train_val_gradient_cosine"]))

    def test_accumulator_preserves_finite_counts_and_epoch_reset(self):
        for device in self.devices():
            with self.subTest(device=device):
                fields = ("norm", "cosine")
                accumulator = GradientGeometryAccumulator(["weight", "bias"], fields, device)
                accumulator.update(torch.tensor([[2., float("nan")], [float("inf"), .5]], device=device))
                accumulator.update(torch.tensor([[4., .2], [float("nan"), -.5]], device=device))
                rows = accumulator.averages()
                self.assertEqual(rows[0]["norm"], 3.)
                self.assertAlmostEqual(rows[0]["cosine"], .2)
                self.assertTrue(math.isnan(rows[1]["norm"]))
                self.assertEqual(rows[1]["cosine"], 0.)
                empty = GradientGeometryAccumulator(["weight"], fields, device).averages()[0]
                self.assertTrue(math.isnan(empty["norm"]))
                self.assertTrue(math.isnan(empty["cosine"]))

    def test_train_only_adam_updates_remain_identical(self):
        for device in self.devices():
            with self.subTest(device=device):
                torch.manual_seed(12)
                reference = nn.Linear(4, 3).to(device)
                model = nn.Linear(4, 3).to(device)
                model.load_state_dict(reference.state_dict())
                optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.03)
                reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.01, weight_decay=.03)
                parameters = list(model.parameters())
                criterion = nn.CrossEntropyLoss()
                for _ in range(4):
                    inputs = torch.randn(5, 4, device=device)
                    targets = torch.randint(0, 3, (5,), device=device)
                    gradients = measurement._compute_batch_gradients(
                        model, (inputs, targets), criterion, device=device,
                        modular=False, eq_position=None, parameters=parameters,
                    )
                    combined, _ = measurement._combine_normalized_train_val_gradients(
                        model, gradients, [torch.zeros_like(g) for g in gradients],
                        parameters=parameters, train_weight=1., val_weight=0.,
                    )
                    measurement._apply_gradient_update(optimizer, None, parameters, combined)
                    reference_optimizer.zero_grad(set_to_none=True)
                    criterion(reference(inputs), targets).backward()
                    reference_optimizer.step()
                    for actual, expected in zip(parameters, reference.parameters(), strict=True):
                        self.assertTrue(torch.equal(actual, expected))

    def test_evaluation_weighting_and_no_per_batch_scalar_reads(self):
        torch.manual_seed(45)
        model = RecordingLinear()
        loader = DataLoader(TensorDataset(torch.randn(5, 4), torch.tensor([0, 1, 2, 1, 0])), batch_size=2)
        criterion = nn.CrossEntropyLoss()
        expected_loss, expected_accuracy = 0., 0.
        with torch.no_grad():
            for inputs, targets in loader:
                logits = model(inputs)
                expected_loss += criterion(logits, targets).item() * len(targets)
                expected_accuracy += (logits.argmax(1) == targets).float().mean().item() * len(targets)
        batches = list(loader)  # DataLoader initialization itself reads a CPU RNG seed.
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("batch scalar read")):
            result = measurement.evaluate_partition(
                model, batches, criterion, device=torch.device("cpu"), modular=False, eq_position=None,
                amp_enabled=True, amp_dtype=torch.bfloat16,
            )
        self.assertEqual(result.examples, 5)
        self.assertEqual(result.loss, expected_loss / 5)
        self.assertEqual(result.accuracy, expected_accuracy / 5)
        self.assertEqual(set(model.output_dtypes), {torch.float32})  # AMP stays ignored on CPU.
        self.assertFalse(model.training)
        with self.assertRaisesRegex(ValueError, "empty partition"):
            measurement.evaluate_partition(
                model, [], criterion, device=torch.device("cpu"), modular=False, eq_position=None,
            )

    def test_geometry_sampling_keeps_every_epoch_cosine_and_existing_csv_schema(self):
        torch.manual_seed(18)
        dataset = TensorDataset(torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1]))
        loader = DataLoader(dataset, batch_size=2)
        model = nn.Linear(4, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        calls, expected_epochs = [], {0, 2}
        combine = measurement._combine_normalized_train_val_gradients

        def record_combination(*args, **kwargs):
            calls.append(kwargs["collect_geometry"])
            return combine(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            optimization_path, geometry_path = Path(directory) / "optimization.csv", Path(directory) / "geometry.csv"
            with patch.object(measurement, "_gradient_geometry_sample_epochs", return_value=expected_epochs), \
                 patch.object(measurement, "_combine_normalized_train_val_gradients", side_effect=record_combination), \
                 patch.object(measurement, "evaluate_partition", wraps=measurement.evaluate_partition) as evaluate:
                rows = measurement.train_normalized_two_sided(
                    model, loader, loader, nn.CrossEntropyLoss(), device=torch.device("cpu"),
                    modular=False, eq_position=None, optimizer=optimizer, scheduler=None,
                    epochs=3, report_interval=3, csv_path=str(optimization_path),
                    gradient_geometry_csv_path=str(geometry_path),
                    train_gradient_weight_function="1-0.5*f", val_gradient_weight_function="0.5*f",
                    amp_enabled=True, amp_dtype=torch.bfloat16,
                )
            self.assertEqual(calls, [True] * 3 + [False] * 3 + [True] * 3)
            self.assertEqual(len(evaluate.call_args_list), 6)
            self.assertTrue(all(call.kwargs["amp_enabled"] for call in evaluate.call_args_list))
            self.assertTrue(math.isnan(rows[0]["train_val_gradient_cosine"]))
            self.assertTrue(math.isfinite(rows[1]["train_val_gradient_cosine"]))
            self.assertEqual([row["train_gradient_weight"] for row in rows], [1., .75, .5])
            with geometry_path.open() as infile:
                geometry_rows = list(csv.DictReader(infile))
            self.assertEqual({int(row["epoch"]) for row in geometry_rows}, expected_epochs)
            self.assertEqual(list(geometry_rows[0]), [
                "epoch", "parameter", "train_gradient_norm", "val_gradient_norm", "combined_gradient_norm",
                "parameter_norm", "train_val_gradient_cosine", "train_gradient_weight", "val_gradient_weight",
                "relative_train_gradient", "relative_combined_gradient",
            ])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_amp_evaluation_and_full_precision_flatness_in_both_training_modes(self):
        device = torch.device("cuda")
        torch.manual_seed(7)
        dataset = TensorDataset(torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1]))
        loader = DataLoader(dataset, batch_size=2)
        for objective in ("normalized_two_sided", "difference", "mean_relative_gap"):
            with self.subTest(objective=objective), tempfile.TemporaryDirectory() as directory:
                model = RecordingLinear().to(device)
                initial_state = {name: value.clone() for name, value in model.state_dict().items()}
                bundle = measurement.DatasetBundle(
                    dataset, dataset, SimpleNamespace(model_type="toy", dataset_type="mnist"),
                    "toy", "mnist", 10, torch.arange(10), torch.empty(0, dtype=torch.long), False,
                )
                args = SimpleNamespace(
                    random_seed=123, split_seed=456, optimizer_preset=0, learning_rate=.001,
                    weight_decay=0., optimizer="adamw", scheduler="fixed", warmup_epochs=0,
                    torch_compile=False, amp=True, report_interval=1, objective=objective,
                    train_gradient_weight_function="0.8", val_gradient_weight_function="0.2",
                    relative_flatness_layers="last_matrix_weight", relative_flatness_samples=1,
                    relative_flatness_batches=1, relative_flatness_seed=2718, augmentation="none", dataset="mnist",
                )
                with patch.object(measurement, "evaluate_partition", wraps=measurement.evaluate_partition) as evaluate:
                    result = measurement._run_direction(
                        direction_name="a_as_train_b_as_val", train_partition_name="a", val_partition_name="b",
                        train_loader=loader, val_loader=loader, flatness_train_loader=loader, flatness_val_loader=loader,
                        bundle=bundle, model=model, initial_state=initial_state, criterion=nn.CrossEntropyLoss(),
                        device=device, batch_size=2, epochs=2, args=args, output_folder=Path(directory),
                        model_type_name="toy", dataset_type_name="mnist",
                    )
                self.assertTrue(all(call.kwargs["amp_enabled"] for call in evaluate.call_args_list))
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                self.assertIn(dtype, model.output_dtypes)
                self.assertEqual(model.output_dtypes[-1], torch.float32)
                self.assertTrue(result["runtime"]["evaluation_amp"])
                self.assertFalse(result["runtime"]["relative_flatness_amp"])
                self.assertTrue(math.isfinite(result["misalignment_measure"]["overall_mean"]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_amp_can_be_disabled_for_evaluation(self):
        model = RecordingLinear().cuda()
        loader = DataLoader(TensorDataset(torch.randn(5, 4), torch.tensor([0, 1, 2, 1, 0])), batch_size=2)
        criterion = nn.CrossEntropyLoss()
        measurement.evaluate_partition(
            model, loader, criterion, device=torch.device("cuda"), modular=False, eq_position=None,
            amp_enabled=True, amp_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        self.assertNotEqual(set(model.output_dtypes), {torch.float32})
        model.output_dtypes.clear()
        measurement.evaluate_partition(
            model, loader, criterion, device=torch.device("cuda"), modular=False, eq_position=None,
            amp_enabled=False, amp_dtype=torch.bfloat16,
        )
        self.assertEqual(set(model.output_dtypes), {torch.float32})

    @unittest.skipUnless(torch.cuda.is_available() and hasattr(torch, "compile"), "CUDA/torch.compile unavailable")
    def test_compiled_cuda_cct_amp_keeps_tokenizer_counts_and_short_batch_weighting(self):
        from py_src.third_party.compact_transformers.src.cct import CCT

        torch.manual_seed(19)
        model = CCT(
            img_size=32, embedding_dim=32, n_conv_layers=2, kernel_size=3, stride=1,
            padding=1, num_layers=1, num_heads=2, num_classes=3,
            attention_dropout=0., stochastic_depth=0.,
        ).cuda()
        images = torch.cat((torch.zeros(1, 3, 32, 32), torch.randn(2, 3, 32, 32)))
        loader = DataLoader(TensorDataset(images, torch.tensor([0, 1, 2])), batch_size=2)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs = dict(device=torch.device("cuda"), modular=False, eq_position=None,
                      amp_enabled=True, amp_dtype=dtype)
        criterion = nn.CrossEntropyLoss()
        monitor = TokenizerOutputMonitor(model)
        try:
            baseline = measurement.evaluate_partition(model, loader, criterion, tokenizer_monitor=monitor, **kwargs)
        finally:
            monitor.close()
        graphs = []

        def backend(graph, example_inputs):
            graphs.append(graph)
            return graph.forward

        compiled = torch.compile(model, backend=backend)
        monitor = TokenizerOutputMonitor(model)
        try:
            for _ in range(3):
                actual = measurement.evaluate_partition(compiled, loader, criterion, tokenizer_monitor=monitor, **kwargs)
                self.assertEqual(actual.tokenizer_nonzero_ratio, baseline.tokenizer_nonzero_ratio)
                self.assertAlmostEqual(actual.loss, baseline.loss, places=6)
                self.assertEqual(actual.accuracy, baseline.accuracy)
                self.assertEqual(actual.examples, 3)
            graph_count = len(graphs)
            measurement.evaluate_partition(compiled, loader, criterion, tokenizer_monitor=monitor, **kwargs)
            self.assertEqual(len(graphs), graph_count)
        finally:
            monitor.close()


if __name__ == "__main__":
    unittest.main()
