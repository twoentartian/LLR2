from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import unittest
import unittest.mock

from torch.utils.data import Subset, TensorDataset
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import remeasure_misalignment_flatness as remeasure


class TestRemeasureHelpers(unittest.TestCase):
    def setUp(self):
        self.partition = TensorDataset(torch.arange(100))

    def test_random_subset_is_seeded_sorted_and_distinct(self):
        first = remeasure._subset(self.partition, 10, order="random", seed=3)
        again = remeasure._subset(self.partition, 10, order="random", seed=3)
        other = remeasure._subset(self.partition, 10, order="random", seed=4)
        self.assertIsInstance(first, Subset)
        self.assertEqual(first.indices, again.indices)
        self.assertNotEqual(first.indices, other.indices)
        self.assertEqual(first.indices, sorted(set(first.indices)))
        self.assertNotEqual(first.indices, list(range(10)))

    def test_first_order_all_and_oversized_requests(self):
        self.assertEqual(remeasure._subset(self.partition, 5, order="first", seed=0).indices, list(range(5)))
        self.assertIs(remeasure._subset(self.partition, None, order="random", seed=0), self.partition)
        self.assertIs(remeasure._subset(self.partition, 1000, order="random", seed=0), self.partition)

    def test_sample_count_and_measure_parsing(self):
        self.assertIsNone(remeasure._sample_count("all"))
        self.assertEqual(remeasure._sample_count("512"), 512)
        with self.assertRaises(argparse.ArgumentTypeError):
            remeasure._sample_count("0")
        self.assertEqual(remeasure._measurements("flatness, accuracy"), ("flatness", "accuracy"))
        with self.assertRaises(argparse.ArgumentTypeError):
            remeasure._measurements("accuracy,hessian")

    def test_flatness_samples_default_to_samples_unless_given(self):
        argv = ["remeasure", "folder", "--samples", "100"]
        with unittest.mock.patch.object(sys, "argv", argv):
            self.assertEqual(remeasure.parse_args().flatness_samples, 100)
        with unittest.mock.patch.object(sys, "argv", [*argv, "--flatness_samples", "all"]):
            self.assertIsNone(remeasure.parse_args().flatness_samples)

    def test_final_epoch_checkpoint_is_not_measured_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "gap_checkpoints_a_as_train_b_as_val").mkdir()
            for name in ("final_a_as_train_b_as_val.model.pt", "gap_checkpoints_a_as_train_b_as_val/epoch_000007.model.pt",
                         "gap_checkpoints_a_as_train_b_as_val/epoch_000009.model.pt"):
                (folder / name).touch()
            labels = [label for label, _, _ in remeasure._checkpoints(folder, "a_as_train_b_as_val", "all", 9)]
            self.assertEqual(labels, ["final", "epoch_7"])
            self.assertEqual([label for label, _, _ in remeasure._checkpoints(folder, "a_as_train_b_as_val", "final", 9)],
                             ["final"])


if __name__ == "__main__":
    unittest.main()
