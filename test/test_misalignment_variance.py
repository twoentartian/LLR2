from __future__ import annotations

import csv
from pathlib import Path
import re
import sys
import tempfile
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
from misalignment.variance import EpochWeightVarianceRecorder


class TestEpochWeightVarianceRecorder(unittest.TestCase):
    def test_weight_selection_sample_variance_and_four_significant_digits(self):
        model = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2), nn.Embedding(4, 3))
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[0.1111, 0.2222, 0.3333], [0.4444, 0.5555, 0.6666]]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "variance.csv"
            recorder = EpochWeightVarianceRecorder(model, path)
            recorder.record(0)
            with path.open() as infile:
                rows = list(csv.DictReader(infile))
            self.assertEqual(list(rows[0]), ["epoch", "0.weight", "1.weight", "2.weight"])
            state = model.state_dict()
            for name in recorder.layer_names:
                expected = f"{torch.var(state[name], correction=1).item():.3E}"
                self.assertEqual(rows[0][name], expected)
                self.assertRegex(rows[0][name], r"^\d\.\d{3}E[+-]\d+$")
            self.assertEqual(rows[0]["1.weight"], "0.000E+00")
            self.assertNotEqual(rows[0]["0.weight"], f"{torch.var(state['0.weight'], correction=0).item():.3E}")

    def test_each_epoch_is_immediately_persisted_without_changing_model_or_gradients(self):
        model = nn.Linear(3, 2)
        model.train()
        with torch.no_grad():
            model.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        model.weight.grad = torch.ones_like(model.weight)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "variance.csv"
            recorder = EpochWeightVarianceRecorder(model, path)
            for epoch in range(3):
                expected_weight = model.weight.detach().clone()
                expected_gradient = model.weight.grad.clone()
                expected_variance = f"{torch.var(model.weight).item():.3E}"
                recorder.record(epoch)
                self.assertTrue(model.training)
                self.assertTrue(torch.equal(model.weight, expected_weight))
                self.assertTrue(torch.equal(model.weight.grad, expected_gradient))
                with path.open() as infile:
                    rows = list(csv.DictReader(infile))
                self.assertEqual(len(rows), epoch + 1)
                self.assertEqual(rows[-1]["epoch"], str(epoch))
                self.assertEqual(rows[-1]["weight"], expected_variance)
                with torch.no_grad():
                    model.weight.mul_(2)

    def test_single_element_weight_has_undefined_sample_variance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "variance.csv"
            recorder = EpochWeightVarianceRecorder(nn.Linear(1, 1), path)
            recorder.record(0)
            with path.open() as infile:
                row = next(csv.DictReader(infile))
            self.assertEqual(row["weight"], "NAN")

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "variance.csv"
            recorder = EpochWeightVarianceRecorder(nn.Linear(2, 2), path)
            recorder.record(0)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                EpochWeightVarianceRecorder(nn.Linear(2, 2), path)
            self.assertEqual(path.read_bytes(), before)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_gpu_reductions_and_scientific_format(self):
        model = nn.Sequential(nn.Linear(3, 2), nn.LayerNorm(2)).cuda()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "variance.csv"
            recorder = EpochWeightVarianceRecorder(model, path)
            recorder.record(4)
            with path.open() as infile:
                row = next(csv.DictReader(infile))
            self.assertEqual(row["epoch"], "4")
            for name in recorder.layer_names:
                self.assertTrue(re.fullmatch(r"\d\.\d{3}E[+-]\d+", row[name]))
                self.assertEqual(row[name], f"{torch.var(model.state_dict()[name]).item():.3E}")


if __name__ == "__main__":
    unittest.main()
