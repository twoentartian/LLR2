"""Dataset wrappers used to construct per-node simulator data loaders."""

from __future__ import annotations

from typing import Any, Iterable, Optional, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from py_src.ml_setup.dataloader_util import DataloaderConfig

if TYPE_CHECKING:
    from py_src.ml_setup import MLSetup


class LabelProbabilitySampler(Sampler[int]):
    """Sample dataset indices according to a probability for each label."""

    def __init__(
        self,
        label_probabilities: np.ndarray,
        indices_by_label: dict[int, np.ndarray],
        num_samples: int,
    ):
        self.label_probabilities = np.asarray(label_probabilities, dtype=np.float64)
        self.indices_by_label = indices_by_label
        self.num_samples = int(num_samples)
        self.labels_in_order = list(indices_by_label.keys())

    def __iter__(self):
        for _ in range(self.num_samples):
            label_index = np.random.choice(
                len(self.labels_in_order),
                p=self.label_probabilities,
            )
            label = self.labels_in_order[label_index]
            yield int(np.random.choice(self.indices_by_label[label]))

    def __len__(self) -> int:
        return self.num_samples


def _resolve_sampler(dataset: Any, sampler_or_factory: Any) -> Optional[Sampler]:
    if sampler_or_factory is None:
        return None
    if isinstance(sampler_or_factory, Sampler):
        return sampler_or_factory
    if callable(sampler_or_factory):
        return sampler_or_factory(dataset)
    return sampler_or_factory


def _infer_dataset_labels_for_indices(dataset: Any) -> list[int]:
    if dataset is None:
        return []

    if isinstance(dataset, Subset):
        base = dataset.dataset
        if hasattr(base, "targets"):
            targets = np.asarray(base.targets)
            return [int(targets[index]) for index in dataset.indices]

    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        if torch.is_tensor(targets):
            return [int(value) for value in targets.detach().cpu().view(-1).tolist()]
        if isinstance(targets, np.ndarray):
            return [int(value) for value in targets.reshape(-1).tolist()]
        return [int(value) for value in targets]

    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        labels: list[int] = []
        for index in range(len(dataset)):
            sample = dataset[index]
            if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                label = sample[1]
                if torch.is_tensor(label):
                    labels.append(int(label.item()))
                else:
                    labels.append(int(label))
        return labels

    return []


def _build_dataloader_kwargs(
    *,
    batch_size: int,
    num_workers: Optional[int],
    collate_fn: Any,
    sampler: Optional[Sampler],
    shuffle: bool,
) -> dict[str, Any]:
    worker_count = 0 if num_workers is None else int(num_workers)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "num_workers": worker_count,
        "pin_memory": True,
        "collate_fn": collate_fn,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    if worker_count > 0:
        kwargs["prefetch_factor"] = 4
        kwargs["persistent_workers"] = True
    return kwargs


class DatasetWithFastLabelSelection:
    """Index a map-style dataset once and build label-weighted data loaders."""

    def __init__(self, dataset: Any, ml_setup: MLSetup):
        self.raw_dataset = dataset
        self.ml_setup = ml_setup
        self.labels = np.asarray(_infer_dataset_labels_for_indices(dataset))
        if self.labels.size == 0:
            raise ValueError(
                "Dataset label selection requires a map-style dataset "
                "with accessible labels"
            )

        indices = np.arange(len(self.labels))
        self.indices_by_label: dict[int, np.ndarray] = {}
        for label in sorted(set(int(value) for value in self.labels.tolist())):
            self.indices_by_label[label] = indices[self.labels == label]

    def get_train_loader_by_label_prob(
        self,
        label_prob: np.ndarray,
        batch_size: int,
        worker: Optional[int] = None,
    ) -> Iterable:
        if hasattr(self.raw_dataset, "build_dataloader"):
            raise NotImplementedError(
                "label-distribution sampling is not supported for datasets "
                "with custom build_dataloader() backends"
            )

        sampler = LabelProbabilitySampler(
            label_prob,
            self.indices_by_label,
            batch_size,
        )
        loader_kwargs = _build_dataloader_kwargs(
            batch_size=batch_size,
            num_workers=worker,
            collate_fn=self.ml_setup.collate_fn,
            sampler=sampler,
            shuffle=False,
        )
        return DataLoader(self.raw_dataset, **loader_kwargs)

    def get_train_loader_default(
        self,
        batch_size: int,
        worker: Optional[int] = None,
    ) -> Iterable:
        if self.ml_setup.override_training_dataset_loader is not None:
            return self.ml_setup.override_training_dataset_loader

        if hasattr(self.raw_dataset, "build_dataloader"):
            config = DataloaderConfig(
                batch_size=batch_size,
                num_workers=worker or 0,
                shuffle=True,
                pin_memory=True,
                prefetch_factor=4 if (worker or 0) > 0 else None,
                persistent_workers=(worker or 0) > 0,
            )
            return self.raw_dataset.build_dataloader(
                default_batch_size=batch_size,
                config=config,
                is_train=True,
            )

        sampler = _resolve_sampler(self.raw_dataset, self.ml_setup.sampler_fn)
        loader_kwargs = _build_dataloader_kwargs(
            batch_size=batch_size,
            num_workers=worker,
            collate_fn=self.ml_setup.collate_fn,
            sampler=sampler,
            shuffle=sampler is None,
        )
        return DataLoader(self.raw_dataset, **loader_kwargs)
