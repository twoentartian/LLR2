from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from py_src.ml_setup_dataset.dataset_intermediate_layer import (
    DatasetWithFastLabelSelection,
)
from py_src.shared_dataloader import BatchRequest, SharedDataLoader


class _PidDataset(Dataset):
    def __init__(self):
        self.targets = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.tensor(index), self.targets[index], os.getpid()


def _minimal_ml_setup():
    return SimpleNamespace(
        default_sampler_fn=None,
        override_train_loader=None,
        default_collate_fn=None,
    )


class SharedDataLoaderTest(unittest.TestCase):
    def test_routes_batches_to_the_requesting_nodes(self):
        loader = SharedDataLoader(
            _PidDataset(),
            num_workers=0,
            pin_memory=False,
        )
        loader.set_plan(
            [
                BatchRequest.from_indices(3, 0, [0, 2]),
                BatchRequest.from_indices(9, 0, [5, 7]),
            ]
        )

        batches = list(loader)

        self.assertEqual(
            [(batch.node_name, batch.batch_index) for batch in batches],
            [(3, 0), (9, 0)],
        )
        self.assertEqual(batches[0].batch[0].tolist(), [0, 2])
        self.assertEqual(batches[1].batch[0].tolist(), [5, 7])

    def test_persistent_worker_pool_is_reused_between_plans(self):
        loader = SharedDataLoader(
            _PidDataset(),
            num_workers=2,
            pin_memory=False,
            prefetch_factor=2,
        )
        try:
            loader.set_plan(
                BatchRequest.from_indices(node_name, 0, [node_name % 8])
                for node_name in range(6)
            )
            first_worker_pids = {
                int(batch.batch[2].item())
                for batch in loader
            }

            loader.set_plan(
                [
                    BatchRequest.from_indices(10, 0, [0]),
                    BatchRequest.from_indices(11, 0, [1]),
                ]
            )
            second_worker_pids = {
                int(batch.batch[2].item())
                for batch in loader
            }
        finally:
            loader.close()

        self.assertEqual(len(first_worker_pids), 2)
        self.assertNotIn(os.getpid(), first_worker_pids)
        self.assertEqual(second_worker_pids, first_worker_pids)

    def test_label_selector_generates_node_specific_indices(self):
        dataset = _PidDataset()
        selector = DatasetWithFastLabelSelection(dataset, _minimal_ml_setup())

        indices = selector.sample_indices(
            np.array([0.0, 1.0]),
            64,
            rng=np.random.default_rng(7),
        )

        self.assertEqual(len(indices), 64)
        self.assertTrue(all(int(dataset.targets[index]) == 1 for index in indices))

    def test_default_selector_uses_each_index_once_before_repeating(self):
        selector = DatasetWithFastLabelSelection(
            _PidDataset(),
            _minimal_ml_setup(),
        )

        indices = selector.sample_indices(
            None,
            selector.dataset_size,
            rng=np.random.default_rng(11),
        )

        self.assertEqual(sorted(indices), list(range(selector.dataset_size)))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    unittest.main()
