"""Single-node distributed runtime shared by the two training entry points.

Batch sizes exposed by these entry points are global (before accumulation).
No process group is created on import, so ordinary single-device callers keep
their existing behavior. CPU/Gloo also permits numerical regression tests.
"""

from __future__ import annotations

import os
import secrets
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def enabled() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def rank() -> int:
    return dist.get_rank() if enabled() else 0


def world_size() -> int:
    return dist.get_world_size() if enabled() else 1


def is_main() -> bool:
    return rank() == 0


def initialize(*, use_cpu: bool = False) -> torch.device | None:
    """Initialize a 2/4-worker torchrun job, rejecting multi-node launches."""
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if size == 1:
        return None
    local_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(size)))
    if size not in (2, 4) or local_size != size:
        raise ValueError("Distributed training supports one node with 2 or 4 processes only")
    local_rank = int(os.environ["LOCAL_RANK"])
    if use_cpu:
        device = torch.device("cpu")
        backend = "gloo"
    else:
        if not torch.cuda.is_available() or torch.cuda.device_count() < size:
            raise RuntimeError(f"torchrun requested {size} GPUs, but fewer CUDA devices are visible")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return device


def shutdown() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def device() -> torch.device:
    if enabled() and dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def broadcast_object(value):
    if enabled():
        objects = [value if is_main() else None]
        dist.broadcast_object_list(objects, src=0)
        return objects[0]
    return value


def shared_seed(seed: int | None) -> int:
    return int(broadcast_object(seed if seed is not None else secrets.randbelow(2**31)))


def barrier() -> None:
    if enabled():
        dist.barrier()


def local_batch_size(global_batch_size: int) -> int:
    size = world_size()
    if global_batch_size < size or global_batch_size % size:
        raise ValueError(f"Global batch size {global_batch_size} must be a positive multiple of {size}")
    return global_batch_size // size


class EvaluationSampler(Sampler[int]):
    """An exact evaluation shard: no padding and no dropped examples."""

    def __init__(self, dataset) -> None:
        self.count = len(dataset)
        self.rank = rank()
        self.size = world_size()

    def __iter__(self):
        return iter(range(self.rank, self.count, self.size))

    def __len__(self) -> int:
        return len(range(self.rank, self.count, self.size))


@torch.no_grad()
def broadcast_model(model: torch.nn.Module, *, buffers_only: bool = False) -> None:
    if not enabled():
        return
    tensors = list(model.buffers())
    if not buffers_only:
        tensors = list(model.parameters()) + tensors
    for tensor in tensors:
        dist.broadcast(tensor, src=0)


def sum_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if enabled():
        dist.all_reduce(tensor)
    return tensor


@torch.no_grad()
def reduce_gradients(gradients: list[torch.Tensor], *, average: bool = True,
                     sample_count: int | None = None, bucket_bytes: int = 25 * 1024 * 1024) -> None:
    """Reduce explicitly constructed gradients before nonlinear combination.

    Weight batch-mean gradients by examples when sample_count is supplied.
    Bucketing avoids one collective per parameter on models such as ViT.
    """
    if not enabled():
        return
    divisor = float(world_size()) if average else 1.0
    weight = 1.0
    if sample_count is not None:
        count = torch.tensor(float(sample_count), dtype=torch.float64, device=device())
        sum_tensor(count)
        divisor = float(count.item())
        if divisor <= 0:
            raise ValueError("Cannot average gradients from zero examples")
        weight = float(sample_count)
    groups = defaultdict(list)
    for gradient in gradients:
        groups[(gradient.device, gradient.dtype)].append(gradient)

    def reduce_bucket(bucket):
        flat = torch.cat([gradient.reshape(-1) for gradient in bucket])
        flat.mul_(weight)
        dist.all_reduce(flat)
        flat.div_(divisor)
        offset = 0
        for gradient in bucket:
            length = gradient.numel()
            gradient.copy_(flat[offset:offset + length].view_as(gradient))
            offset += length

    for group in groups.values():
        bucket, length = [], 0
        for gradient in group:
            bucket.append(gradient)
            length += gradient.numel() * gradient.element_size()
            if length >= bucket_bytes:
                reduce_bucket(bucket)
                bucket, length = [], 0
        if bucket:
            reduce_bucket(bucket)
