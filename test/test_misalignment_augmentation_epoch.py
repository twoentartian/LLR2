from __future__ import annotations

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import misalignment_measurement_2 as measurement


class _IndexDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return index, 0


def _epoch_transform(_):
    # The augmentation RNG is seeded from the epoch, so the first draw
    # identifies which epoch the worker observed.
    return torch.rand(())


class TestAugmentationEpoch(unittest.TestCase):
    def test_persistent_workers_observe_epoch_updates(self):
        dataset = measurement.AugmentedDataset(_IndexDataset(), _epoch_transform, seed=3, stochastic=True)
        loader = measurement._build_loader(
            Subset(dataset, list(range(8))), 4, device=torch.device("cpu"), num_workers=2, seed=1,
        )
        self.assertTrue(loader.persistent_workers)
        views = []
        for epoch in range(3):
            measurement._set_augmentation_epoch(loader.dataset, epoch)
            views.append(torch.cat([batch[0] for batch in loader]))
        expected = []
        for epoch in range(3):
            dataset.epoch = epoch
            expected.append(torch.stack([dataset[index][0] for index in range(8)]))
        for epoch in range(3):
            torch.testing.assert_close(views[epoch], expected[epoch])
        self.assertFalse(torch.equal(views[0], views[1]))

    def test_flatness_partitions_use_unaugmented_transform_on_same_examples(self):
        source = _IndexDataset()
        augmented = measurement.AugmentedDataset(source, _epoch_transform, seed=3, stochastic=True)
        partition_a, partition_b = Subset(augmented, [5, 1, 2, 7]), Subset(augmented, [0, 3, 4, 6])
        calls = []

        def fake_build(ml_setup, dataset_name, level):
            calls.append(level)
            return lambda image: float(image)

        original = measurement.build_augmentation
        measurement.build_augmentation = fake_build
        try:
            clean_a, clean_b = measurement._flatness_partitions(
                SimpleNamespace(), "cifar10", 2, source, partition_a, partition_b, seed=3,
            )
            same_a, same_b = measurement._flatness_partitions(
                SimpleNamespace(), "cifar10", 0, source, partition_a, partition_b, seed=3,
            )
        finally:
            measurement.build_augmentation = original
        self.assertEqual(calls, [0])
        self.assertEqual([clean_a[i][0] for i in range(4)], [5.0, 1.0, 2.0, 7.0])
        self.assertEqual([clean_b[i][0] for i in range(4)], [0.0, 3.0, 4.0, 6.0])
        self.assertIs(same_a, partition_a)
        self.assertIs(same_b, partition_b)


if __name__ == "__main__":
    unittest.main()
