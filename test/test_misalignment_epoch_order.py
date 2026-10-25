from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DistributedSampler, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import misalignment_measurement_2 as measurement
from misalignment.epoch_order import EpochOrderSampler, epoch_order, shard_order


def read_order(loader):
    return torch.cat([batch[0] for batch in loader]).tolist()


class TestEpochOrder(unittest.TestCase):
    def test_fixed_order_is_identity_and_matches_distributed_sampler(self):
        self.assertEqual(epoch_order(7, seed=3, epoch=5, shuffle=False), list(range(7)))
        dataset = TensorDataset(torch.arange(10))
        for world_size in (2, 3, 4):
            for rank in range(world_size):
                with self.subTest(world_size=world_size, rank=rank):
                    reference = list(DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False))
                    sampler = EpochOrderSampler(10, seed=3, shuffle=False, world_size=world_size, rank=rank)
                    self.assertEqual(list(sampler), reference)
                    self.assertEqual(len(sampler), len(reference))

    def test_shuffled_order_is_seeded_per_epoch_and_sharded_consistently(self):
        first = epoch_order(50, seed=7, epoch=0, shuffle=True)
        self.assertEqual(sorted(first), list(range(50)))
        self.assertEqual(first, epoch_order(50, seed=7, epoch=0, shuffle=True))
        self.assertNotEqual(first, epoch_order(50, seed=7, epoch=1, shuffle=True))
        self.assertNotEqual(first, epoch_order(50, seed=8, epoch=0, shuffle=True))
        shards = [shard_order(first, world_size=4, rank=rank, pad=True) for rank in range(4)]
        padded = first + first[:2]
        self.assertEqual([index for step in zip(*shards) for index in step], padded)
        self.assertEqual(shard_order(first, world_size=1, rank=0, pad=True), first)

    def test_pytorch_training_loader_reorders_each_epoch_and_evaluation_stays_fixed(self):
        dataset = TensorDataset(torch.arange(40))
        for workers in (0, 2):
            with self.subTest(num_workers=workers):
                train = measurement._build_loader(dataset, 8, device=torch.device("cpu"), num_workers=workers,
                                                  seed=11, shuffle=True)
                fixed = measurement._build_loader(dataset, 8, device=torch.device("cpu"), num_workers=workers,
                                                  seed=11, is_train=False)
                orders = []
                for epoch in range(3):
                    measurement._set_loader_epoch(train, epoch)
                    orders.append(read_order(train))
                    self.assertEqual(orders[-1], epoch_order(40, seed=11, epoch=epoch, shuffle=True))
                    self.assertEqual(read_order(fixed), list(range(40)))
                self.assertEqual(len({tuple(order) for order in orders}), 3)
                measurement._set_loader_epoch(train, 1)
                self.assertEqual(read_order(train), orders[1])

    def test_each_half_keeps_its_order_when_train_and_val_roles_swap(self):
        loaders = {name: measurement._build_loader(TensorDataset(torch.arange(offset, offset + 24)), 6,
                                                   device=torch.device("cpu"), num_workers=0, seed=seed,
                                                   shuffle=True)
                   for name, offset, seed in (("a", 0, 101), ("b", 100, 202))}
        for loader in loaders.values():
            measurement._set_loader_epoch(loader, 4)
        a_then_b = [(a[0].tolist(), b[0].tolist()) for a, b in zip(loaders["a"], loaders["b"])]
        b_then_a = [(a[0].tolist(), b[0].tolist()) for b, a in zip(loaders["b"], loaders["a"])]
        self.assertEqual(a_then_b, b_then_a)

    def test_device_cached_loader_regroups_examples_per_epoch(self):
        class DictLoader:
            dataset = None
            batch_size = 4

            def __iter__(self):
                for start in range(0, 10, 4):
                    yield {"text": torch.arange(start, min(start + 4, 10)), "target": -torch.arange(start, min(start + 4, 10))}

        fixed = measurement._DeviceCachedLoader(DictLoader(), torch.device("cpu"))
        self.assertEqual([batch["text"].tolist() for batch in fixed], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])
        shuffled = measurement._DeviceCachedLoader(DictLoader(), torch.device("cpu"), shuffle=True, seed=5)
        for epoch in (0, 1):
            measurement._set_loader_epoch(shuffled, epoch)
            batches = list(shuffled)
            self.assertEqual([len(batch["text"]) for batch in batches], [4, 4, 2])
            text = torch.cat([batch["text"] for batch in batches])
            self.assertEqual(text.tolist(), epoch_order(10, seed=5, epoch=epoch, shuffle=True))
            self.assertTrue(torch.equal(torch.cat([batch["target"] for batch in batches]), -text))

    def test_cli_shuffles_by_default(self):
        base = ["misalignment_measurement_2.py", "--dataset", "cifar10"]
        with patch.object(sys, "argv", base):
            self.assertTrue(measurement.parse_args().shuffle)
        with patch.object(sys, "argv", [*base, "--no-shuffle"]):
            self.assertFalse(measurement.parse_args().shuffle)


if __name__ == "__main__":
    unittest.main()
