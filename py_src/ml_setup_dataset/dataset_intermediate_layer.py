"""Dataset indexing and sampling helpers used by the simulator."""

from __future__ import annotations

import itertools
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
        label_indices = np.random.choice(
            len(self.labels_in_order),
            size=self.num_samples,
            p=self.label_probabilities,
        )
        sampled_indices = np.empty(self.num_samples, dtype=np.int64)
        for label_index, label in enumerate(self.labels_in_order):
            positions = np.flatnonzero(label_indices == label_index)
            if positions.size == 0:
                continue
            sampled_indices[positions] = np.random.choice(
                self.indices_by_label[label],
                size=positions.size,
            )
        yield from (int(index) for index in sampled_indices)

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
    """Index a map-style dataset once and sample node-specific indices."""

    def __init__(self, dataset: Any, ml_setup: MLSetup):
        self.raw_dataset = dataset
        self.ml_setup = ml_setup
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise ValueError("shared loading requires a map-style dataset")
        self.dataset_size = len(dataset)
        if self.dataset_size <= 0:
            raise ValueError("training dataset must not be empty")

        self.labels = np.asarray(_infer_dataset_labels_for_indices(dataset))
        self.indices_by_label: dict[int, np.ndarray] = {}
        if self.labels.size:
            if len(self.labels) != self.dataset_size:
                raise ValueError("inferred dataset labels do not match dataset length")
            indices = np.arange(len(self.labels))
            for label in sorted(set(int(value) for value in self.labels.tolist())):
                self.indices_by_label[label] = indices[self.labels == label]

    @property
    def supports_shared_loading(self) -> bool:
        """Whether this dataset can be indexed by the shared worker pool."""

        return (
            not hasattr(self.raw_dataset, "build_dataloader")
            and getattr(
                self.ml_setup,
                "override_training_dataset_loader",
                getattr(self.ml_setup, "override_train_loader", None),
            )
            is None
        )

    def sample_indices(
        self,
        label_probabilities: Optional[np.ndarray],
        num_samples: int,
        *,
        rng: Any = None,
    ) -> list[int]:
        """Generate one or more batches of indices for a node.

        Sampling remains in the simulator process.  DataLoader workers only
        fetch and transform the resulting indices, which allows one worker
        pool to serve different node distributions.
        """

        sample_count = int(num_samples)
        if sample_count < 0:
            raise ValueError("num_samples must be non-negative")
        if sample_count == 0:
            return []

        random_source = np.random if rng is None else rng
        if label_probabilities is None:
            sampler = _resolve_sampler(
                self.raw_dataset,
                getattr(
                    self.ml_setup,
                    "sampler_fn",
                    getattr(self.ml_setup, "default_sampler_fn", None),
                ),
            )
            if sampler is not None:
                sampled = list(itertools.islice(iter(sampler), sample_count))
                if len(sampled) != sample_count:
                    raise ValueError(
                        "configured sampler produced fewer indices than requested"
                    )
                return [int(index) for index in sampled]

            sampled: list[int] = []
            while len(sampled) < sample_count:
                permutation = random_source.permutation(self.dataset_size)
                remaining = sample_count - len(sampled)
                sampled.extend(int(index) for index in permutation[:remaining])
            return sampled

        if not self.indices_by_label:
            raise ValueError(
                "label-distribution sampling requires accessible dataset labels"
            )

        probabilities = np.asarray(label_probabilities, dtype=np.float64)
        labels_in_order = list(self.indices_by_label)
        if probabilities.shape != (len(labels_in_order),):
            raise ValueError(
                "label probability count does not match the dataset label count"
            )
        if np.any(probabilities < 0) or not np.isfinite(probabilities).all():
            raise ValueError("label probabilities must be finite and non-negative")
        probability_sum = probabilities.sum()
        if probability_sum <= 0:
            raise ValueError("label probabilities must have a positive sum")
        probabilities = probabilities / probability_sum

        sampled_label_indices = random_source.choice(
            len(labels_in_order),
            size=sample_count,
            p=probabilities,
        )
        sampled_indices = np.empty(sample_count, dtype=np.int64)
        for label_index, label in enumerate(labels_in_order):
            positions = np.flatnonzero(sampled_label_indices == label_index)
            if positions.size == 0:
                continue
            sampled_indices[positions] = random_source.choice(
                self.indices_by_label[label],
                size=positions.size,
            )
        return [int(index) for index in sampled_indices]

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
