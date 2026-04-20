from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler, Subset


# ---------------------------------------------------------------------------
# Dataloader configuration
# ---------------------------------------------------------------------------

@dataclass
class DataloaderConfig:
    """User-tweakable knobs for building a DataLoader.

    Any field left as ``None`` falls back to the default stored in
    :class:`ModelSetup`.
    """
    batch_size: Optional[int] = None
    num_workers: int = 0
    num_samples: Optional[int] = None  # subsample the dataset
    shuffle: Optional[bool] = None     # None → True for train, False for val
    pin_memory: bool = True
    drop_last: bool = False
    collate_fn: Optional[Callable] = None
    sampler: Optional[Any] = None
    prefetch_factor: Optional[int] = None
    persistent_workers: Optional[bool] = None


# ---------------------------------------------------------------------------
# Helper: build a DataLoader from a dataset + config
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset: Any,
    default_batch_size: int,
    config: Optional[DataloaderConfig] = None,
    is_train: bool = True,
    default_collate_fn: Optional[Callable] = None,
    default_sampler_fn: Optional[Callable] = None,
) -> Iterable:
    """Build a DataLoader (or return an IterableDataset directly).

    Parameters
    ----------
    dataset:
        A map-style ``Dataset``, an ``IterableDataset``, or an existing
        ``DataLoader``/iterable.  If it is already an iterable that is *not*
        a plain ``Dataset`` we return it as-is (respecting ``num_samples``
        is not supported in that case).
    default_batch_size:
        Batch size from the ``ModelSetup``.
    config:
        Optional overrides provided by the caller.
    is_train:
        Whether this dataloader is for training (affects default shuffle).
    default_collate_fn:
        Collate function from ``ModelSetup`` (used if config doesn't override).
    """
    cfg = config or DataloaderConfig()

    # --- already an iterable (IterableDataset, existing DataLoader, etc.) ---
    if isinstance(dataset, (IterableDataset, DataLoader)):
        return dataset

    # --- map-style Dataset: we can build a proper DataLoader ----------------
    actual_dataset: Dataset = dataset

    # subsample
    if cfg.num_samples is not None and cfg.num_samples < len(actual_dataset):  # type: ignore[arg-type]
        indices = torch.randperm(len(actual_dataset))[:cfg.num_samples].tolist()  # type: ignore[arg-type]
        actual_dataset = Subset(actual_dataset, indices)

    batch_size = cfg.batch_size or default_batch_size
    shuffle = cfg.shuffle if cfg.shuffle is not None else is_train
    collate_fn = cfg.collate_fn or default_collate_fn
    assert default_sampler_fn is None or cfg.sampler is None, "default_sampler_fn and cfg.sampler cannot be set at the same time"
    sampler = cfg.sampler if cfg.sampler is not None else default_sampler_fn
    if sampler is not None and not isinstance(sampler, Sampler) and callable(sampler):
        sampler = sampler(actual_dataset)

    loader_kwargs: dict = dict(
        dataset=actual_dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        collate_fn=collate_fn,
    )
    if cfg.num_workers > 0:
        prefetch_factor = cfg.prefetch_factor if cfg.prefetch_factor is not None else 4
        persistent_workers = (
            cfg.persistent_workers
            if cfg.persistent_workers is not None
            else True
        )
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
    if sampler is not None:
        loader_kwargs["sampler"] = sampler

    return DataLoader(**loader_kwargs)
