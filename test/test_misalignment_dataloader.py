from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import misalignment_measurement_2 as measurement


class TestMisalignmentDataLoader(unittest.TestCase):
    def setUp(self):
        self.dataset = TensorDataset(torch.arange(8))

    def test_default_prefetch_factor_is_four_per_worker(self):
        loader = measurement._build_loader(
            self.dataset, 2, device=torch.device("cpu"), num_workers=2, seed=1,
        )
        self.assertEqual(loader.prefetch_factor, 4)
        self.assertTrue(loader.persistent_workers)
        self.assertEqual(loader.batch_size, 2)
        self.assertEqual(len(loader), 4)

    def test_prefetch_factor_is_configurable(self):
        loader = measurement._build_loader(
            self.dataset, 2, device=torch.device("cpu"), num_workers=1, seed=1, prefetch_factor=8,
        )
        self.assertEqual(loader.prefetch_factor, 8)

    def test_zero_workers_disables_prefetch_and_preserves_batch_order(self):
        loader = measurement._build_loader(
            self.dataset, 2, device=torch.device("cpu"), num_workers=0, seed=1, prefetch_factor=4,
        )
        self.assertIsNone(loader.prefetch_factor)
        self.assertFalse(loader.persistent_workers)
        batches = [batch[0].tolist() for batch in loader]
        self.assertEqual(batches, [[0, 1], [2, 3], [4, 5], [6, 7]])

    def test_nonpositive_prefetch_factor_is_rejected(self):
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be positive"):
                measurement._build_loader(
                    self.dataset, 2, device=torch.device("cpu"), num_workers=1, seed=1,
                    prefetch_factor=value,
                )

    def test_cli_default_and_override(self):
        with patch.object(sys, "argv", ["misalignment_measurement_2.py", "--dataset", "cifar10"]):
            self.assertEqual(measurement.parse_args().prefetch_factor, 4)
        with patch.object(sys, "argv", ["misalignment_measurement_2.py", "--dataset", "cifar10",
                                       "--prefetch_factor", "8"]):
            self.assertEqual(measurement.parse_args().prefetch_factor, 8)

    def test_main_rejects_invalid_factor_before_loading_dataset(self):
        with patch.object(sys, "argv", ["misalignment_measurement_2.py", "--dataset", "cifar10",
                                       "--prefetch_factor", "0"]), \
             patch.object(measurement, "load_dataset_bundle") as load_dataset:
            with self.assertRaisesRegex(ValueError, "--prefetch_factor must be positive"):
                measurement.main()
            load_dataset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
