from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from test.test_vit_registers import tiny_vit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import misalignment_measurement_2 as measurement
from misalignment.token_norms import TokenNormMonitor


class FixedTokens(nn.Module):
    def __init__(self, registers):
        super().__init__()
        norms = [5.0, *([2.0, 4.0, 6.0, 8.0] if registers else []), 0.0, 10.0, 5.0, 13.0]
        tokens = torch.zeros(len(norms), 8)
        tokens[:, 0] = torch.tensor(norms)
        self.register_buffer("tokens", tokens)

    def forward(self, x):
        return self.tokens.unsqueeze(0) * (1 + x[:, -1:, :1])


def fixed_model(registers):
    model = tiny_vit(registers=registers)
    model.encoder.layers = FixedTokens(registers)
    with torch.no_grad():
        model.encoder.pos_embedding.zero_()
        model.conv_proj.weight.zero_()
        model.conv_proj.weight[0, 0, 0, 0] = 1.0
        model.conv_proj.bias.zero_()
    return model


def make_loader():
    images = torch.arange(5.0).reshape(5, 1, 1, 1).expand(5, 3, 8, 8).clone()
    return DataLoader(TensorDataset(images, torch.tensor([0, 1, 2, 0, 1])), batch_size=2)


def evaluate(model, loader, monitor=None):
    return measurement.evaluate_partition(
        model, loader, nn.CrossEntropyLoss(), device=torch.device("cpu"),
        modular=False, eq_position=None, token_norm_monitor=monitor,
    )


class TestTokenNorms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_exact_pre_norm_statistics_and_histograms_exclude_special_tokens(self):
        expected = (torch.arange(1.0, 6.0)[:, None] * torch.tensor([0.0, 10.0, 5.0, 13.0])).flatten()
        for registers in (False, True):
            with self.subTest(registers=registers):
                model = fixed_model(registers)
                monitor = TokenNormMonitor(model, threshold=8.0)
                try:
                    with patch.object(model, "forward", wraps=model.forward) as forward:
                        result = evaluate(model, make_loader(), monitor)
                        self.assertEqual(forward.call_count, 3)
                    stats = result.token_norms
                    self.assertEqual(stats["patch"]["count"], 20)
                    self.assertAlmostEqual(stats["patch"]["mean"], expected.double().mean().item())
                    self.assertAlmostEqual(stats["patch"]["std"], expected.double().std(correction=0).item())
                    self.assertEqual(stats["patch"]["max"], 65.0)
                    self.assertAlmostEqual(stats["patch"]["high_norm_ratio"], (expected > 8).double().mean().item())
                    self.assertEqual(sum(stats["patch"]["histogram_counts"]), 20)
                    self.assertEqual(stats["cls"]["mean"], 15.0)
                    self.assertEqual(len(stats), 6 if registers else 2)
                    if registers:
                        self.assertEqual(stats["register_0"]["mean"], 6.0)
                        self.assertEqual(stats["register_3"]["max"], 40.0)
                    # The next capture replaces, rather than appends to, counts.
                    self.assertEqual(evaluate(model, make_loader(), monitor).token_norms, stats)
                finally:
                    monitor.close()
                self.assertFalse(model.encoder.ln._forward_pre_hooks)

    def test_monitor_does_not_change_logits_or_training_gradients(self):
        model = tiny_vit()
        nn.init.normal_(model.heads.head.weight)
        reference = copy.deepcopy(model)
        monitor = TokenNormMonitor(model)
        try:
            images, target = next(iter(make_loader()))
            nn.CrossEntropyLoss()(model(images), target).backward()
            nn.CrossEntropyLoss()(reference(images), target).backward()
            self.assertIsNone(monitor.statistics)
            for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
                self.assertTrue(torch.equal(actual.grad, expected.grad))
            self.assertEqual(evaluate(model, make_loader(), monitor).loss, evaluate(reference, make_loader()).loss)
        finally:
            monitor.close()

    def test_disabled_and_non_vit_models_and_failure_cleanup(self):
        self.assertFalse(TokenNormMonitor(nn.Linear(2, 2)).available)
        model = tiny_vit()
        disabled = TokenNormMonitor(model, enabled=False)
        self.assertIsNone(evaluate(model, make_loader(), disabled).token_norms)
        monitor = TokenNormMonitor(model)
        try:
            with patch.object(model, "forward", side_effect=RuntimeError("broken")):
                with self.assertRaisesRegex(RuntimeError, "broken"):
                    evaluate(model, make_loader(), monitor)
            self.assertFalse(monitor.active)
        finally:
            monitor.close()

    def test_compile_reuses_capture_graphs(self):
        model = fixed_model(True)
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        compiled = torch.compile(model, backend=backend)
        monitor = TokenNormMonitor(model, threshold=8.0)
        try:
            expected = evaluate(compiled, make_loader(), monitor).token_norms
            graph_count = len(graphs)
            for _ in range(2):
                self.assertEqual(evaluate(compiled, make_loader(), monitor).token_norms, expected)
            self.assertEqual(len(graphs), graph_count)
        finally:
            monitor.close()

    def test_all_objectives_write_norm_csv_and_summary_and_restore_checkpoints(self):
        for objective in ("normalized_two_sided", "difference", "mean_relative_gap"):
            with self.subTest(objective=objective), tempfile.TemporaryDirectory() as directory:
                model = tiny_vit()
                loader = make_loader()
                bundle = measurement.DatasetBundle(
                    partition_a=loader.dataset, partition_b=loader.dataset,
                    ml_setup=SimpleNamespace(model_type="toy", dataset_type="mnist"),
                    dataset_name="toy", dataset_type_name="mnist", source_examples=10,
                    permutation=torch.arange(10), excluded_indices=torch.empty(0, dtype=torch.long), modular=False,
                )
                args = SimpleNamespace(
                    random_seed=123, split_seed=456, optimizer_preset=0, learning_rate=0.001,
                    weight_decay=0.0, optimizer="sgd", scheduler="fixed", warmup_epochs=0,
                    torch_compile=False, amp=False, report_interval=1, objective=objective,
                    train_gradient_weight_function="0.8", val_gradient_weight_function="0.2",
                    relative_flatness_layers="last_matrix_weight", relative_flatness_samples=1,
                    relative_flatness_batches=1, relative_flatness_seed=2718, augmentation="none", dataset="mnist",
                    token_norms=True, token_norm_threshold=8.0,
                )
                with self.assertLogs("misalignment_measurement_2", level="INFO"):
                    result = measurement._run_direction(
                        direction_name="a_as_train_b_as_val", train_partition_name="a", val_partition_name="b",
                        train_loader=loader, val_loader=loader, flatness_train_loader=loader, flatness_val_loader=loader,
                        bundle=bundle, model=model,
                        initial_state={name: value.clone() for name, value in model.state_dict().items()},
                        criterion=nn.CrossEntropyLoss(), device=torch.device("cpu"), batch_size=2,
                        epochs=2, args=args, output_folder=Path(directory), model_type_name="toy", dataset_type_name="mnist",
                    )
                with (Path(directory) / result["files"]["token_norms"]).open() as infile:
                    rows = list(csv.DictReader(infile))
                self.assertEqual(len(rows), 2 * 2 * 6)
                self.assertEqual({row["role"] for row in rows}, {"train", "val"})
                self.assertEqual({row["token_group"] for row in rows}, set(result["final"]["train"]["token_norms"]))
                for row in rows:
                    self.assertEqual(sum(json.loads(row["histogram_counts"])), int(row["count"]))
                self.assertTrue(result["token_norm_diagnostics"]["enabled"])
                self.assertEqual(result["token_norm_diagnostics"]["num_register_tokens"], 4)
                self.assertFalse(model.encoder.ln._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
