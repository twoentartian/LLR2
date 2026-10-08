from __future__ import annotations

import csv
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
from misalignment.tokenizer_stats import TokenizerOutputMonitor


class ToyCCT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = nn.Sequential(nn.Conv2d(1, 2, 1, bias=False), nn.ReLU())
        self.classifier = nn.Linear(8, 2)
        with torch.no_grad():
            self.tokenizer[0].weight.copy_(torch.tensor([[[[1.]]], [[[-1.]]]]))

    def forward(self, x):
        return self.classifier(self.tokenizer(x).flatten(1))


def make_loader():
    images = torch.tensor([
        [[[1., 0.], [0., 0.]]],
        [[[1., 1.], [1., 1.]]],
        [[[0., 0.], [0., 0.]]],
        [[[-1., -1.], [-1., -1.]]],
        [[[1., 1.], [0., 0.]]],
    ])
    return DataLoader(TensorDataset(images, torch.tensor([0, 1, 0, 1, 0])), batch_size=2)


class TestTokenizerOutputMonitor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def evaluate(self, model, loader, monitor=None):
        return measurement.evaluate_partition(
            model, loader, nn.CrossEntropyLoss(), device=torch.device("cpu"),
            modular=False, eq_position=None, tokenizer_monitor=monitor,
        )

    def test_full_partition_counts_weight_short_last_batch_and_no_extra_forwards(self):
        model = ToyCCT()
        loader = make_loader()
        baseline = self.evaluate(model, loader)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        monitor = TokenizerOutputMonitor(model)
        try:
            with patch.object(model, "forward", wraps=model.forward) as forward:
                result = self.evaluate(model, loader, monitor)
                self.assertEqual(forward.call_count, len(loader))
            # 11 nonzeros / 40 elements, not an unweighted batch-ratio mean.
            self.assertEqual(result.tokenizer_nonzero_ratio, 11 / 40)
            self.assertEqual(result.loss, baseline.loss)
            self.assertEqual(result.accuracy, baseline.accuracy)
            self.assertEqual(result.examples, 5)
            self.assertFalse(monitor.active)
            self.assertEqual(monitor._counts.dtype, torch.int64)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, state[name]))
        finally:
            monitor.close()
        self.assertFalse(model.tokenizer._forward_hooks)

    def test_all_zero_output_and_counts_reset_for_each_partition(self):
        model = ToyCCT()
        monitor = TokenizerOutputMonitor(model)
        try:
            alive = self.evaluate(model, make_loader(), monitor)
            zero_loader = DataLoader(TensorDataset(torch.zeros(3, 1, 2, 2), torch.zeros(3, dtype=torch.long)), batch_size=2)
            dead = self.evaluate(model, zero_loader, monitor)
            self.assertEqual(alive.tokenizer_nonzero_ratio, 0.275)
            self.assertEqual(dead.tokenizer_nonzero_ratio, 0.)
            self.assertEqual(monitor._counts.tolist(), [0, 24])
            again = self.evaluate(model, make_loader(), monitor)
            self.assertEqual(again.tokenizer_nonzero_ratio, alive.tokenizer_nonzero_ratio)
        finally:
            monitor.close()

    def test_training_is_not_counted_or_changed(self):
        model = ToyCCT()
        reference = ToyCCT()
        reference.load_state_dict(model.state_dict())
        monitor = TokenizerOutputMonitor(model)
        try:
            inputs, target = next(iter(make_loader()))
            nn.CrossEntropyLoss()(model(inputs), target).backward()
            nn.CrossEntropyLoss()(reference(inputs), target).backward()
            self.assertIsNone(monitor.nonzero_ratio)
            self.assertTrue(model.training)
            for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
                self.assertTrue(torch.equal(actual.grad, expected.grad))
        finally:
            monitor.close()

    def test_models_without_tokenizer_remain_supported(self):
        model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
        monitor = TokenizerOutputMonitor(model)
        try:
            self.assertFalse(monitor.available)
            self.assertIsNone(self.evaluate(model, make_loader(), monitor).tokenizer_nonzero_ratio)
        finally:
            monitor.close()

    def test_capture_deactivates_after_evaluation_failure(self):
        model = ToyCCT()
        monitor = TokenizerOutputMonitor(model)
        try:
            with patch.object(model, "forward", side_effect=RuntimeError("broken forward")):
                with self.assertRaisesRegex(RuntimeError, "broken forward"):
                    self.evaluate(model, make_loader(), monitor)
            self.assertFalse(monitor.active)
        finally:
            monitor.close()
        self.assertFalse(model.tokenizer._forward_hooks)

    @unittest.skipUnless(hasattr(torch, "compile"), "torch.compile unavailable")
    def test_compiled_model_counts_correctly_without_per_epoch_retracing(self):
        model = ToyCCT()
        graphs = []

        def backend(graph, example_inputs):
            graphs.append(graph)
            return graph.forward

        compiled = torch.compile(model, backend=backend)
        # Match the launcher: wrapping is lazy, then the hook is installed
        # before the first compiled forward.
        monitor = TokenizerOutputMonitor(model)
        try:
            # Warm the compiled evaluation paths (including the short batch).
            self.assertEqual(self.evaluate(compiled, make_loader(), monitor).tokenizer_nonzero_ratio, 0.275)
            graph_count = len(graphs)
            for _ in range(3):
                result = self.evaluate(compiled, make_loader(), monitor)
                self.assertEqual(result.tokenizer_nonzero_ratio, 0.275)
            self.assertEqual(len(graphs), graph_count)
        finally:
            monitor.close()

    def test_real_cct_counts_tokenizer_before_positional_embeddings(self):
        from py_src.third_party.compact_transformers.src.cct import CCT

        model = CCT(
            img_size=32, embedding_dim=32, n_conv_layers=2, kernel_size=3, stride=1,
            padding=1, num_layers=1, num_heads=2, num_classes=2,
            attention_dropout=0., stochastic_depth=0.,
        )
        images = torch.zeros(3, 3, 32, 32)
        loader = DataLoader(TensorDataset(images, torch.zeros(3, dtype=torch.long)), batch_size=2)
        monitor = TokenizerOutputMonitor(model)
        try:
            result = self.evaluate(model, loader, monitor)
            self.assertEqual(result.tokenizer_nonzero_ratio, 0.)
            # Positional embeddings still produce nonzero logits, but must not
            # make the diagnostic mistakenly report a live image tokenizer.
            with torch.no_grad():
                self.assertGreater(torch.count_nonzero(model(images)).item(), 0)
        finally:
            monitor.close()

    def test_epoch_logs_csv_and_monitor_cleanup_for_each_objective(self):
        for objective in ("normalized_two_sided", "difference", "mean_relative_gap"):
            with self.subTest(objective=objective), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = ToyCCT()
                loader = make_loader()
                dataset = loader.dataset
                bundle = measurement.DatasetBundle(
                    partition_a=dataset, partition_b=dataset,
                    ml_setup=SimpleNamespace(model_type="toy", dataset_type="mnist"),
                    dataset_name="toy", dataset_type_name="mnist", source_examples=10,
                    permutation=torch.arange(10), excluded_indices=torch.empty(0, dtype=torch.long), modular=False,
                )
                args = SimpleNamespace(
                    random_seed=123, split_seed=456, optimizer_preset=0, learning_rate=0.001,
                    weight_decay=0., optimizer="sgd", scheduler="fixed", warmup_epochs=0,
                    torch_compile=False, amp=False, report_interval=1, objective=objective,
                    train_gradient_weight_function="0.8", val_gradient_weight_function="0.2",
                    relative_flatness_layers="last_matrix_weight", relative_flatness_samples=1,
                    relative_flatness_batches=1, relative_flatness_seed=2718, augmentation="none", dataset="mnist",
                )
                with self.assertLogs("misalignment_measurement_2", level="INFO") as log:
                    result = measurement._run_direction(
                        direction_name="a_as_train_b_as_val", train_partition_name="a", val_partition_name="b",
                        train_loader=loader, val_loader=loader, flatness_train_loader=loader, flatness_val_loader=loader,
                        bundle=bundle, model=model,
                        initial_state={name: value.clone() for name, value in model.state_dict().items()},
                        criterion=nn.CrossEntropyLoss(), device=torch.device("cpu"), batch_size=2,
                        epochs=2, args=args, output_folder=root, model_type_name="toy", dataset_type_name="mnist",
                    )
                epoch_logs = [message for message in log.output if "epoch " in message and "/2:" in message]
                self.assertEqual(len(epoch_logs), 2)
                for message in epoch_logs:
                    self.assertIn("train_tokenizer_nonzero_ratio=", message)
                    self.assertIn("val_tokenizer_nonzero_ratio=", message)
                with (root / result["files"]["optimization"]).open() as infile:
                    rows = list(csv.DictReader(infile))
                self.assertEqual(len(rows), 2)
                for row in rows:
                    for name in ("train_tokenizer_nonzero_ratio", "val_tokenizer_nonzero_ratio"):
                        self.assertRegex(row[name], r"^\d\.\d{3}E[+-]\d+$")
                        self.assertGreaterEqual(float(row[name]), 0.)
                        self.assertLessEqual(float(row[name]), 1.)
                self.assertTrue(result["tokenizer_diagnostics"]["enabled"])
                self.assertIsNotNone(result["final"]["train"]["tokenizer_nonzero_ratio"])
                self.assertFalse(model.tokenizer._forward_hooks)


if __name__ == "__main__":
    unittest.main()
