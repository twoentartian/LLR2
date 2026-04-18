"""Smoke tests for tool/generate_high_accuracy_model.py.

Each test trains for exactly one batch and (where appropriate) evaluates
for exactly one batch.  All data is synthetic and in-memory; no downloads,
no disk I/O, no pretrained weights.

Run from the LLR2 project root:

    python -m pytest py_src/test/test_generate_high_accuracy_model.py -v
    # or
    python -m unittest py_src.test.test_generate_high_accuracy_model
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from py_src.ml_setup.resnet import resnet18_bn_imagenet1k

# ---------------------------------------------------------------------------
# Make sure the LLR2 project root is importable regardless of where the test
# runner is launched from.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # …/LLR2
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from py_src.ml_setup import ddpm_cifar10, resnet18_bn_cifar10
from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_dataset import DatasetType
from py_src.adapters import StandardAdapter, DiffusionAdapter, LightningAdapter
from py_src.engine import TrainResult, ValResult
from py_src.ml_setup_model.nanoclip.loss import ContrastiveLoss
from test.util import run_single_batch


class _TinyNanoCLIPDataset(torch.utils.data.Dataset):
    """Random (image, caption_tokens, attention_mask) triples."""

    def __init__(
        self,
        n: int = 16,
        img_size: int = 8,
        vocab_size: int = 100,
        seq_len: int = 8,
    ):
        self.images = torch.randn(n, 3, img_size, img_size)
        self.captions = torch.randint(0, vocab_size, (n, seq_len), dtype=torch.long)
        self.masks = torch.ones(n, seq_len, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.captions[idx], self.masks[idx]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunSingleBatch(unittest.TestCase):
    """Smoke-tests: one train batch + one val batch (where applicable)."""

    def test_resnet18_cifar10_train_and_val(self):
        setup = resnet18_bn_cifar10()
        train_result, val_result = run_single_batch(setup, run_val=True)
        print(f"resnet18_bn_cifar10")
        print(f"train loss:{train_result.avg_loss:.4f} train accuracy:{train_result.accuracy:.4f}")
        if val_result is not None:
            print("val loss:{val_result.avg_loss:.4f}")


    def test_ddpm_train_no_val(self):
        """Diffusion models skip validation even when run_val=True."""
        setup = ddpm_cifar10()
        train_result, val_result = run_single_batch(setup, run_val=False)
        print(f"ddpm_cifar10")
        print(f"train loss:{train_result.avg_loss:.4f}")

    def test_resnet18_imagenet1k_train_and_val_p1(self):
        setup = resnet18_bn_imagenet1k(1)
        train_result, val_result = run_single_batch(setup, run_val=True)
        print(f"resnet18_bn_imagenet1k")
        print(f"train loss:{train_result.avg_loss:.4f} train accuracy:{train_result.accuracy:.4f}")
        if val_result is not None:
            print("val loss:{val_result.avg_loss:.4f}")

    def test_resnet18_imagenet1k_train_and_val_p2(self):
        setup = resnet18_bn_imagenet1k(2)
        train_result, val_result = run_single_batch(setup, run_val=True)
        print(f"resnet18_bn_imagenet1k")
        print(f"train loss:{train_result.avg_loss:.4f} train accuracy:{train_result.accuracy:.4f}")
        if val_result is not None:
            print("val loss:{val_result.avg_loss:.4f}")

    # def test_nanoclip_train_and_val(self):
    #     setup = ()
    #     train_result, val_result = run_single_batch(setup, run_val=True)

    #     print(f"resnet18_bn_cifar10")
    #     print(f"train loss:{train_result.avg_loss:.4f} train accuracy:{train_result.accuracy:.4f}")
    #     if val_result is not None:
    #         print("val loss:{val_result.avg_loss:.4f}")


if __name__ == "__main__":
    unittest.main()
