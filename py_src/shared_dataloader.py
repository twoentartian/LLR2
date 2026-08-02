"""A bounded DataLoader worker pool shared by simulator nodes.

The normal PyTorch ``DataLoader`` owns its worker processes.  Constructing one
loader per simulator node therefore creates one worker pool per node.  This
module routes batches for several nodes through a single DataLoader instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, NamedTuple, Optional, Sequence

from torch.utils.data import DataLoader, Dataset, Sampler, default_collate


@dataclass(frozen=True)
class BatchRequest:
    """One dataset batch requested on behalf of a simulator node."""

    node_name: int
    batch_index: int
    dataset_indices: tuple[int, ...]

    @classmethod
    def from_indices(
        cls,
        node_name: int,
        batch_index: int,
        dataset_indices: Sequence[int],
    ) -> "BatchRequest":
        return cls(
            node_name=node_name,
            batch_index=int(batch_index),
            dataset_indices=tuple(int(index) for index in dataset_indices),
        )


@dataclass(frozen=True)
class RoutedIndex:
    """Dataset index together with the node batch it belongs to."""

    node_name: int
    batch_index: int
    dataset_index: int


class RoutedBatch(NamedTuple):
    """A collated batch routed back to its requesting node."""

    node_name: int
    batch_index: int
    batch: Any


class PlannedBatchSampler(Sampler[list[RoutedIndex]]):
    """Emit a finite batch plan which can be replaced between iterations."""

    def __init__(self) -> None:
        self._plan: tuple[BatchRequest, ...] = ()

    def set_plan(self, requests: Iterable[BatchRequest]) -> None:
        plan = tuple(requests)
        if any(not request.dataset_indices for request in plan):
            raise ValueError("shared DataLoader requests must not be empty")
        self._plan = plan

    def __iter__(self) -> Iterator[list[RoutedIndex]]:
        # Capture the plan now.  Persistent DataLoader workers call __iter__
        # again when a new simulator tick resets the loader iterator.
        plan = self._plan
        return iter(
            [
                RoutedIndex(
                    node_name=request.node_name,
                    batch_index=request.batch_index,
                    dataset_index=dataset_index,
                )
                for dataset_index in request.dataset_indices
            ]
            for request in plan
        )

    def __len__(self) -> int:
        return len(self._plan)


class RoutedDataset(Dataset):
    """Resolve routed indices against a common map-style dataset."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, routed_index: RoutedIndex):
        sample = self.dataset[routed_index.dataset_index]
        return routed_index.node_name, routed_index.batch_index, sample


class RoutedCollate:
    """Apply the pipeline collator and retain the node routing token."""

    def __init__(self, collate_fn: Optional[Callable] = None):
        self.collate_fn = default_collate if collate_fn is None else collate_fn

    def __call__(self, routed_samples) -> RoutedBatch:
        if not routed_samples:
            raise ValueError("cannot collate an empty routed batch")

        node_name = routed_samples[0][0]
        batch_index = routed_samples[0][1]
        if any(
            sample_node_name != node_name or sample_batch_index != batch_index
            for sample_node_name, sample_batch_index, _ in routed_samples
        ):
            raise ValueError("one routed batch contains samples for multiple requests")

        batch = self.collate_fn([sample for _, _, sample in routed_samples])
        return RoutedBatch(node_name, batch_index, batch)


class SharedDataLoader:
    """Route finite node batch plans through one persistent worker pool."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
    ):
        worker_count = int(num_workers)
        if worker_count < 0:
            raise ValueError("num_workers must be non-negative")
        if prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive")

        self.num_workers = worker_count
        self.batch_sampler = PlannedBatchSampler()
        loader_kwargs: dict[str, Any] = {
            "dataset": RoutedDataset(dataset),
            "batch_sampler": self.batch_sampler,
            "num_workers": worker_count,
            "pin_memory": pin_memory,
            "collate_fn": RoutedCollate(collate_fn),
        }
        if worker_count > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        self._loader = DataLoader(**loader_kwargs)

    def set_plan(self, requests: Iterable[BatchRequest]) -> None:
        self.batch_sampler.set_plan(requests)

    def __iter__(self) -> Iterator[RoutedBatch]:
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def close(self) -> None:
        """Stop persistent workers immediately when the simulator exits."""

        iterator = getattr(self._loader, "_iterator", None)
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()
        if hasattr(self._loader, "_iterator"):
            setattr(self._loader, "_iterator", None)

    def __enter__(self) -> "SharedDataLoader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
