"""Per-epoch example order shared by every misalignment training loader.

With shuffling, each partition gets ``randperm(size)`` seeded by its loader
seed and the epoch. The seed belongs to the partition (A or B), not to its
train/val role, so both orientations of a run see every half in the same
order and the A/B exchange stays symmetric. Without shuffling the order is
the fixed partition order. A fixed order lets models fit the examples by
their position in the epoch (observed for ViT-B/32 on ImageNet), which is
why shuffling is the default for training loaders.

Multi-GPU loaders pad and shard the global order exactly like
``DistributedSampler(drop_last=False)``, so every paired global batch holds
the same examples as the single-device order.
"""

from __future__ import annotations

import math

import torch

ORDER_RNG = "torch.randperm(size) seeded with partition_loader_seed + 1000003 * epoch"


def epoch_order(size: int, *, seed: int, epoch: int, shuffle: bool) -> list[int]:
    """Global example order of one partition for one epoch."""
    if not shuffle:
        return list(range(size))
    generator = torch.Generator(device="cpu").manual_seed((int(seed) + 1_000_003 * int(epoch)) % (2**63 - 1))
    return torch.randperm(size, generator=generator).tolist()


def shard_order(order: list[int], *, world_size: int, rank: int, pad: bool) -> list[int]:
    """This rank's share of a global order; ``pad`` repeats leading examples."""
    if world_size <= 1:
        return list(order)
    if pad:
        total = math.ceil(len(order) / world_size) * world_size
        order = order + order[:total - len(order)]
    return order[rank::world_size]


class EpochOrderSampler(torch.utils.data.Sampler[int]):
    """PyTorch sampler yielding :func:`epoch_order`, sharded for multi-GPU."""

    def __init__(self, size: int, *, seed: int, shuffle: bool, world_size: int = 1, rank: int = 0,
                 pad: bool = True) -> None:
        self.size = int(size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.pad = pad
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        order = epoch_order(self.size, seed=self.seed, epoch=self.epoch, shuffle=self.shuffle)
        return iter(shard_order(order, world_size=self.world_size, rank=self.rank, pad=self.pad))

    def __len__(self) -> int:
        if self.world_size <= 1:
            return self.size
        if self.pad:
            return math.ceil(self.size / self.world_size)
        return len(range(self.rank, self.size, self.world_size))
