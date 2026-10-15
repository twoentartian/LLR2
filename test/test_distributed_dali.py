"""Exercise real DALI shard readers on one GPU, independently of NCCL."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from py_src import distributed
from py_src.ml_setup.dataloader_util import DataloaderConfig
from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup_dataset.dataset_imagenet import DaliImageNetDataset


@pytest.mark.parametrize("size", [2, 4])
def test_real_dali_shards_have_equal_training_steps_and_exact_evaluation(tmp_path, size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("nvidia.dali")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
    from misalignment.augmentation import describe_augmentation
    from misalignment.dali_loader import DaliImageNetLoader

    for label in range(11):
        folder = tmp_path / f"class_{label:02d}"
        folder.mkdir()
        Image.new("RGB", (64, 64), (label * 19, 30, 60)).save(folder / "image.jpg")
    dataset = DaliImageNetDataset(
        file_root=str(tmp_path), dataset_type=DatasetType.imagenet1k, split="train",
        augmentation=False, val_crop_size=32, val_resize_size=36, device_id=0,
    )
    partition = ImageFolder(tmp_path)
    for backend in ("high_accuracy", "misalignment"):
        for is_train in (True, False):
            all_labels, steps = [], []
            for rank in range(size):
                with patch.object(distributed, "enabled", return_value=True), \
                     patch.object(distributed, "world_size", return_value=size), \
                     patch.object(distributed, "rank", return_value=rank):
                    if backend == "high_accuracy":
                        loader = dataset.build_dataloader(
                            8, DataloaderConfig(num_workers=1, shuffle=False), is_train=is_train,
                        )
                    else:
                        loader = DaliImageNetLoader(
                            partition, 8, device=torch.device("cuda", 0), num_workers=1,
                            seed=0, prefetch_factor=1, is_train=is_train,
                            augmentation_config=describe_augmentation("imagenet1k", 0, input_size=32),
                        )
                    labels = [batch[1].cpu().tolist() for batch in loader]
                    assert len(labels) == len(loader)
                    steps.append(len(labels))
                    all_labels.extend(label for batch in labels for label in batch)
                    if hasattr(loader, "close"):
                        loader.close()
                    del loader
            if is_train:
                assert len(set(steps)) == 1
                assert len(all_labels) == 12  # one padded training example
                assert set(all_labels) == set(range(11))
            else:
                assert sorted(all_labels) == list(range(11))
