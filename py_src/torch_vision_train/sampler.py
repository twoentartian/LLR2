import math
import random

import torch
import torch.distributed as dist


class RASampler(torch.utils.data.Sampler):
    """Repeated augmentation sampler with a local fallback.

    When distributed training is initialized, this matches the torchvision /
    DeiT repeated-augmentation behaviour. Outside distributed runs it behaves
    like the old LLR2 sampler and simply repeats the local indices.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3):
        if repetitions < 1:
            raise ValueError(f"repetitions must be >= 1, got {repetitions}")

        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions
        self.epoch = 0

        dist_ready = dist.is_available() and dist.is_initialized()
        self.distributed = num_replicas is not None or rank is not None or dist_ready

        if not self.distributed:
            self.num_replicas = 1
            self.rank = 0
            self.num_samples = len(self.dataset) * self.repetitions
            self.total_size = self.num_samples
            self.num_selected_samples = self.num_samples
            return

        if num_replicas is None:
            if not dist_ready:
                raise RuntimeError("Distributed RASampler requires an initialized process group.")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist_ready:
                raise RuntimeError("Distributed RASampler requires an initialized process group.")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * float(self.repetitions) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        rounded_selected = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        fallback_selected = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = rounded_selected if rounded_selected > 0 else fallback_selected

    def __iter__(self):
        if self.shuffle:
            if self.distributed:
                generator = torch.Generator()
                generator.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=generator).tolist()
            else:
                rng = random.Random(self.seed + self.epoch)
                indices = list(range(len(self.dataset)))
                rng.shuffle(indices)
        else:
            indices = list(range(len(self.dataset)))

        indices = [index for index in indices for _ in range(self.repetitions)]

        if not self.distributed:
            return iter(indices)

        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
