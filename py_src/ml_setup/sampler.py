from torch.utils.data import Sampler
import random

class RASampler(Sampler):
    def __init__(self, dataset, repetitions=4, shuffle=True):
        self.dataset = dataset
        self.repetitions = repetitions
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # get base indices
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(indices)
        # repeat each index num_repeats times
        repeated = [idx for idx in indices for _ in range(self.repetitions)]
        return iter(repeated)

    def __len__(self):
        return len(self.dataset) * self.repetitions
