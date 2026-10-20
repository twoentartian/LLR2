"""Read the document-isolated A/B corpus produced by tool/prepare_pile.py."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class PileTokenDataset(Dataset):
    """Packed causal examples; stride L predicts every target exactly once.

    Windows overlap by one *input* token, not by prediction targets. A/B streams
    are never concatenated together, so context cannot cross their boundary.
    A final short window is padded and its padding labels are ignored (-100).
    This follows GPT-style packed text with EOS-delimited documents and causal
    attention across documents within a partition, as in GPT-NeoX/Pythia.
    """

    def __init__(self, directory, *, partitions="AB", sequence_length=2048):
        if partitions not in ("A", "B", "AB") or sequence_length <= 0:
            raise ValueError("Use A, B or AB and a positive sequence length")
        self.directory = Path(directory)
        content = (self.directory / "manifest.json").read_bytes()
        self.fingerprint = hashlib.sha256(content).hexdigest()
        self.manifest = json.loads(content)
        if (
            self.manifest.get("format_version") != 1
            or self.manifest.get("dtype") != "uint16"
        ):
            raise ValueError("Unsupported prepared Pile format")
        if self.manifest.get("byte_order") != "little":
            raise ValueError("Expected little-endian token data")
        self.partitions = partitions
        self.sequence_length = sequence_length
        self.segments = []
        self.ends = []
        self.target_tokens = 0
        self._maps = {}
        examples = 0
        for name in partitions:
            details = self.manifest["partitions"][name]
            path = self.directory / details["data"]
            if path.parent.resolve() != self.directory.resolve():
                raise ValueError("Data path escapes the dataset directory")
            tokens = int(details["tokens"])
            if path.stat().st_size != 2 * tokens or tokens < 2:
                raise ValueError(f"Invalid or empty token stream {path}")
            targets = tokens - 1
            count = math.ceil(targets / sequence_length)
            self.segments.append((path, tokens, examples, count))
            examples += count
            self.ends.append(examples)
            self.target_tokens += targets

    def __len__(self):
        return self.ends[-1]

    def _location(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        segment = self.segments[bisect.bisect_right(self.ends, index)]
        path, tokens, base, _ = segment
        start = (index - base) * self.sequence_length
        return path, tokens, start

    def valid_tokens(self, index):
        _, tokens, start = self._location(index)
        return min(self.sequence_length, tokens - 1 - start)

    def __getitem__(self, key):
        index, limit = (
            key if isinstance(key, tuple) else (int(key), self.sequence_length)
        )
        inputs = torch.zeros(self.sequence_length, dtype=torch.long)
        labels = torch.full((self.sequence_length,), -100, dtype=torch.long)
        if index >= 0 and limit > 0:
            path, tokens, start = self._location(index)
            if path not in self._maps:
                self._maps[path] = np.memmap(
                    path, dtype="<u2", mode="r", shape=(tokens,)
                )
            count = min(self.sequence_length, tokens - 1 - start, limit)
            window = self._maps[path][start : start + count + 1].astype(np.int64)
            inputs[:count] = torch.from_numpy(window[:-1])
            labels[:count] = torch.from_numpy(window[1:])
        return {"input_ids": inputs, "labels": labels}

    def __getstate__(self):
        return {**self.__dict__, "_maps": {}}


class ExactTokenBatchSampler(Sampler):
    """Identical global permutation on all ranks, no distributed padding/repeats.

    Empty final rank shards receive one all-ignored dummy example, maintaining
    equal DDP collective counts. Gradients must be normalized by global valid
    target tokens, not by local batch size. Optional token limits mask the last
    target exactly instead of overshooting by a global batch.
    """

    def __init__(
        self,
        dataset,
        *,
        micro_batch_size,
        rank=0,
        world_size=1,
        seed=42,
        epoch=0,
        start_micro=0,
        token_limit=None,
        shuffle=True,
    ):
        if micro_batch_size <= 0 or not 0 <= rank < world_size or start_micro < 0:
            raise ValueError("Invalid batch/shard settings")
        self.dataset = dataset
        self.micro_batch_size = micro_batch_size
        self.rank, self.world_size = rank, world_size
        self.seed, self.epoch = seed, epoch
        self.start_micro = start_micro
        self.token_limit = token_limit
        self.shuffle = shuffle
        self.global_micro_batch = micro_batch_size * world_size
        self.total_micro_steps = math.ceil(len(dataset) / self.global_micro_batch)

    def __len__(self):
        return max(0, self.total_micro_steps - self.start_micro)

    def __iter__(self):
        # ~0.8 GB/rank for the complete 207B-token corpus at L=2048; host RAM,
        # never GPU memory. Reconstructed from seed+epoch after checkpoint resume.
        order = (
            np.random.Generator(np.random.PCG64(self.seed + self.epoch)).permutation(
                len(self.dataset)
            )
            if self.shuffle
            else np.arange(len(self.dataset))
        )
        remaining = self.token_limit
        for step in range(self.start_micro, self.total_micro_steps):
            if remaining is not None and remaining <= 0:
                break
            indices = order[
                step * self.global_micro_batch : (step + 1) * self.global_micro_batch
            ]
            planned = []
            for index in indices:
                count = self.dataset.valid_tokens(int(index))
                if remaining is not None:
                    count = min(count, remaining)
                    remaining -= count
                planned.append((int(index), count))
            begin = self.rank * self.micro_batch_size
            local = planned[begin : begin + self.micro_batch_size]
            yield local or [(-1, 0)]
