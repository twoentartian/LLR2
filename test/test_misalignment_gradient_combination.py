from __future__ import annotations

from pathlib import Path
import csv
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import misalignment_measurement_2 as measurement
from misalignment.gradient_combination import NormalizedGradientCombiner, combine_normalized_gradient_tensors


class TestNormalizedGradientCombiner(unittest.TestCase):
    def setUp(self):
        # Models/dtypes differ between tests; do not consume the production
        # per-code-object graph budget with unrelated unit-test signatures.
        if hasattr(torch, "compiler"):
            torch.compiler.reset()

    def test_cli_follows_model_compile_by_default_and_allows_independent_override(self):
        for extra, model_enabled, combine_enabled in (
            ([], True, None),
            (["--no-compile"], False, None),
            (["--no-compile-gradient-combination"], True, False),
            (["--no-compile", "--compile_gradient_combination"], False, True),
        ):
            with self.subTest(extra=extra), patch.object(sys, "argv", ["measurement", "--dataset", "cifar10", *extra]):
                args = measurement.parse_args()
                self.assertEqual(args.torch_compile, model_enabled)
                self.assertEqual(args.compile_gradient_combination, combine_enabled)

    def test_cpu_uses_eager_and_preserves_inputs(self):
        parameters = [nn.Parameter(torch.randn(3, 4)), nn.Parameter(torch.tensor(2.))]
        train, val = [torch.randn_like(p) for p in parameters], [torch.randn_like(p) for p in parameters]
        originals = [tensor.detach().clone() for tensor in parameters + train + val]
        with patch.object(torch, "compile") as compile_function:
            combiner = NormalizedGradientCombiner(parameters)
            actual, stats = combiner(train, val, train_weight=.8, val_weight=.2)
            compile_function.assert_not_called()
        expected, expected_stats = combine_normalized_gradient_tensors(parameters, train, val, (.8, .2))
        for a, b in zip(actual, expected, strict=True):
            self.assertTrue(torch.equal(a, b))
        torch.testing.assert_close(stats, expected_stats, rtol=0, atol=0)
        for actual_input, original in zip(parameters + train + val, originals, strict=True):
            self.assertTrue(torch.equal(actual_input, original))
        self.assertFalse(combiner.runtime_info()["enabled"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_explicit_low_precision_parameters_keep_eager_weight_rounding(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                parameter = nn.Parameter(torch.randn(100, device="cuda", dtype=dtype))
                train, val = [torch.randn_like(parameter)], [torch.randn_like(parameter)]
                with patch.object(torch, "compile") as compile_function:
                    combiner = NormalizedGradientCombiner([parameter])
                    actual, stats = combiner(train, val, train_weight=.8, val_weight=.2)
                    compile_function.assert_not_called()
                expected, expected_stats = combine_normalized_gradient_tensors([parameter], train, val, (.8, .2))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                torch.testing.assert_close(stats, expected_stats, rtol=0, atol=0, equal_nan=True)

    def test_invalid_weights_and_list_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            NormalizedGradientCombiner([])
        p = nn.Parameter(torch.ones(2))
        combiner = NormalizedGradientCombiner([p], enabled=False)
        for weight in (-1., float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "finite and non-negative"):
                combiner([p], [p], train_weight=weight, val_weight=0.)
        with self.assertRaisesRegex(ValueError, "lengths must match"):
            combiner([], [p], train_weight=1., val_weight=0.)

    @unittest.skipUnless(torch.cuda.is_available() and hasattr(torch, "compile"), "CUDA/torch.compile unavailable")
    def test_mixed_full_precision_noncontiguous_parameters(self):
        parameters = [
            nn.Parameter(torch.randn(3, 5, device="cuda").t()),
            nn.Parameter(torch.randn(7, device="cuda", dtype=torch.float64)),
            nn.Parameter(torch.tensor(2., device="cuda", dtype=torch.float64)),
        ]
        train, val = [torch.randn_like(p) for p in parameters], [torch.randn_like(p) for p in parameters]
        combiner = NormalizedGradientCombiner(parameters)
        actual, stats = combiner(train, val, train_weight=.83, val_weight=.17)
        self.assertTrue(combiner.runtime_info()["enabled"], combiner.fallback_reason)
        expected, expected_stats = combine_normalized_gradient_tensors(parameters, train, val, (.83, .17))
        torch.testing.assert_close(actual, expected, rtol=5e-6, atol=5e-7)
        torch.testing.assert_close(stats, expected_stats, rtol=5e-6, atol=5e-7)

    @unittest.skipUnless(torch.cuda.is_available() and hasattr(torch, "compile"), "CUDA/torch.compile unavailable")
    def test_inductor_matches_eager_for_zeros_cancellation_overflow_and_changing_weights(self):
        torch.manual_seed(32)
        parameters = [nn.Parameter(torch.randn(4, 3, device="cuda")) for _ in range(8)]
        parameters.append(nn.Parameter(torch.tensor(2., device="cuda")))
        train, val = [torch.randn_like(p) for p in parameters], [torch.randn_like(p) for p in parameters]
        train[1].zero_()
        val[2].zero_()
        val[3] = train[3].clone()
        val[4] = -train[4]
        train[5].zero_()
        val[5].zero_()
        train[6].fill_(1e20)
        val[6].copy_(train[6])
        train[7].fill_(float("inf"))
        with torch.no_grad():
            parameters[0].zero_()
        combiner = NormalizedGradientCombiner(parameters)
        for weights in ((.5, .5), (.8, .2), (1., 0.), (0., 1.), (0., 0.), (.2, .2)):
            for collect in (True, False):
                with self.subTest(weights=weights, collect=collect):
                    expected, expected_stats = combine_normalized_gradient_tensors(parameters, train, val, weights, collect)
                    actual, stats = combiner(train, val, train_weight=weights[0], val_weight=weights[1], collect_geometry=collect)
                    self.assertTrue(combiner.runtime_info()["enabled"], combiner.fallback_reason)
                    for a, b in zip(actual, expected, strict=True):
                        torch.testing.assert_close(a, b, rtol=5e-6, atol=5e-7, equal_nan=True)
                        self.assertFalse(a.requires_grad)
                    torch.testing.assert_close(stats, expected_stats, rtol=5e-6, atol=5e-7, equal_nan=True)
                    self.assertEqual(stats.shape, (len(parameters), 7 if collect else 1))
                    self.assertTrue(torch.equal(actual[2], train[2]))  # Zero-val fallback stays exact.
                    self.assertEqual(torch.count_nonzero(actual[1]).item(), 0)
                    if weights[0] == weights[1]:
                        self.assertEqual(torch.count_nonzero(actual[3]).item(), 0)

    @unittest.skipUnless(torch.cuda.is_available() and hasattr(torch, "compile"), "CUDA/torch.compile unavailable")
    def test_tensor_weights_do_not_recompile_and_parameters_are_read_afresh(self):
        graphs = []
        real_compile = torch.compile

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        def compile_with_count(fn, **kwargs):
            kwargs.pop("options")
            return real_compile(fn, backend=backend, **kwargs)

        parameters = [nn.Parameter(torch.randn(3, 4, device="cuda"))]
        train, val = [torch.randn_like(parameters[0])], [torch.randn_like(parameters[0])]
        with patch.object(torch, "compile", side_effect=compile_with_count):
            combiner = NormalizedGradientCombiner(parameters)
        for collect in (True, False):
            for index, weights in enumerate(((1., 0.), (.95, .05), (.8, .2), (.5, .5))):
                with torch.no_grad():
                    parameters[0].mul_(2.)
                _, stats = combiner(train, val, train_weight=weights[0], val_weight=weights[1], collect_geometry=collect)
                if collect:
                    torch.testing.assert_close(stats[0, 3], parameters[0].norm().double())
                self.assertEqual(len(graphs), 2 if collect else 4)

    @unittest.skipUnless(torch.cuda.is_available() and hasattr(torch, "compile"), "CUDA/torch.compile unavailable")
    def test_large_equal_and_power_of_two_proportional_gradients_cancel_exactly(self):
        torch.manual_seed(5)
        parameters = [nn.Parameter(torch.randn(*shape, device="cuda")) for shape in ((1000003,), (384, 384), (512,))]
        train = [torch.randn_like(p) for p in parameters]
        combiner = NormalizedGradientCombiner(parameters)
        for factor in (1., .5, 2.):
            val = [g * factor for g in train]
            for collect in (True, False):
                actual, _ = combiner(train, val, train_weight=.2, val_weight=.2, collect_geometry=collect)
                self.assertTrue(combiner.runtime_info()["enabled"], combiner.fallback_reason)
                expected, _ = combine_normalized_gradient_tensors(parameters, train, val, (.2, .2), collect)
                for a, b in zip(actual, expected, strict=True):
                    self.assertEqual(torch.count_nonzero(b).item(), 0)
                    self.assertEqual(torch.count_nonzero(a).item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_lazy_compilation_failure_falls_back_once_without_mutations(self):
        parameters = [nn.Parameter(torch.randn(3, 4, device="cuda"))]
        train, val = [torch.randn_like(parameters[0])], [torch.randn_like(parameters[0])]
        original = parameters[0].detach().clone()
        failure = unittest.mock.Mock(side_effect=RuntimeError("simulated compiler failure"))
        with patch.object(torch, "compile", return_value=failure):
            combiner = NormalizedGradientCombiner(parameters)
        with self.assertLogs("misalignment_measurement_2", level="WARNING") as logs:
            for _ in range(3):
                actual, stats = combiner(train, val, train_weight=.8, val_weight=.2)
                expected, expected_stats = combine_normalized_gradient_tensors(parameters, train, val, (.8, .2))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(stats, expected_stats, rtol=0, atol=0)
        self.assertEqual(failure.call_count, 1)
        self.assertEqual(len(logs.output), 1)
        self.assertFalse(combiner.runtime_info()["enabled"])
        self.assertIn("simulated compiler failure", combiner.runtime_info()["fallback_reason"])
        self.assertTrue(torch.equal(parameters[0], original))

    @unittest.skipUnless(torch.cuda.is_available() and hasattr(torch, "compile"), "CUDA/torch.compile unavailable")
    def test_fp16_grad_scaler_overflow_still_skips_optimizer_step(self):
        torch.manual_seed(44)
        model = nn.Linear(4, 3).cuda()
        parameters = list(model.parameters())
        original = [p.detach().clone() for p in parameters]
        optimizer = torch.optim.AdamW(parameters, lr=.01)
        scaler = torch.amp.GradScaler("cuda", init_scale=1e12)
        batch = (torch.randn(5, 4, device="cuda"), torch.randint(0, 3, (5,), device="cuda"))
        gradients = measurement._compute_batch_gradients(
            model, batch, nn.CrossEntropyLoss(), device=torch.device("cuda"),
            modular=False, eq_position=None, parameters=parameters,
            amp_enabled=True, amp_dtype=torch.float16, scaler=scaler,
        )
        scale_before_update = scaler.get_scale()
        combiner = NormalizedGradientCombiner(parameters)
        combined, _ = combiner(gradients, [torch.zeros_like(g) for g in gradients], train_weight=1., val_weight=0.)
        self.assertTrue(combiner.runtime_info()["enabled"], combiner.fallback_reason)
        self.assertTrue(any(not torch.isfinite(g).all().item() for g in combined))
        measurement._apply_gradient_update(optimizer, None, parameters, combined, scaler=scaler)
        for a, b in zip(parameters, original, strict=True):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(scaler.get_scale(), scale_before_update * .5)
        self.assertEqual(len(optimizer.state), 0)

    @unittest.skipUnless(torch.cuda.is_available() and hasattr(torch, "compile"), "CUDA/torch.compile unavailable")
    def test_training_loop_uses_fused_updates_and_preserves_epoch_csvs(self):
        torch.manual_seed(4)
        dataset_a = TensorDataset(torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1]))
        dataset_b = TensorDataset(torch.randn(5, 4), torch.tensor([2, 1, 0, 1, 2]))
        model = nn.Linear(4, 3).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        combiner = NormalizedGradientCombiner(list(model.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            optimization_path, geometry_path = Path(directory) / "optimization.csv", Path(directory) / "geometry.csv"
            with patch.object(measurement, "_gradient_geometry_sample_epochs", return_value={0, 2}):
                rows = measurement.train_normalized_two_sided(
                    model, DataLoader(dataset_a, batch_size=2), DataLoader(dataset_b, batch_size=2),
                    nn.CrossEntropyLoss(), device=torch.device("cuda"), modular=False, eq_position=None,
                    optimizer=optimizer, scheduler=None, epochs=3, report_interval=3,
                    csv_path=str(optimization_path), gradient_geometry_csv_path=str(geometry_path),
                    train_gradient_weight_function="1-0.5*f", val_gradient_weight_function="0.5*f",
                    amp_enabled=True, amp_dtype=torch.float16, scaler=torch.amp.GradScaler("cuda", init_scale=8.),
                    gradient_combiner=combiner,
                )
            self.assertTrue(combiner.runtime_info()["enabled"], combiner.fallback_reason)
            self.assertEqual([row["batches"] for row in rows], [3.] * 3)
            self.assertEqual([row["train_gradient_weight"] for row in rows], [1., .75, .5])
            self.assertTrue(math.isnan(rows[0]["train_val_gradient_cosine"]))
            self.assertTrue(math.isfinite(rows[1]["train_val_gradient_cosine"]))
            with optimization_path.open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)
            with geometry_path.open() as stream:
                self.assertEqual({int(row["epoch"]) for row in csv.DictReader(stream)}, {0, 2})

    @unittest.skipUnless(torch.cuda.is_available() and hasattr(torch, "compile"), "CUDA/torch.compile unavailable")
    def test_compiled_train_only_adamw_remains_bitwise_identical(self):
        torch.manual_seed(9)
        reference = nn.Linear(4, 3).cuda()
        model = nn.Linear(4, 3).cuda()
        model.load_state_dict(reference.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.03)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.01, weight_decay=.03)
        parameters = list(model.parameters())
        combiner = NormalizedGradientCombiner(parameters)
        criterion = nn.CrossEntropyLoss()
        for _ in range(4):
            inputs, targets = torch.randn(5, 4, device="cuda"), torch.randint(0, 3, (5,), device="cuda")
            loss = criterion(model(inputs), targets)
            gradients = list(torch.autograd.grad(loss, parameters))
            combined, _ = combiner(gradients, [torch.zeros_like(g) for g in gradients], train_weight=1., val_weight=0.)
            self.assertTrue(combiner.runtime_info()["enabled"], combiner.fallback_reason)
            measurement._apply_gradient_update(optimizer, None, parameters, combined)
            reference_optimizer.zero_grad(set_to_none=True)
            criterion(reference(inputs), targets).backward()
            reference_optimizer.step()
            for a, b in zip(parameters, reference.parameters(), strict=True):
                self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
