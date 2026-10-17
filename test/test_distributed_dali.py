"""Exercise real DALI shard readers on one GPU, independently of NCCL."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import copy
import ast
import linecache
import logging
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from py_src import distributed
from py_src.ml_setup.dataloader_util import DataloaderConfig
from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup_dataset.dataset_imagenet import DaliImageNetDataset
from py_src.ml_setup_dataset import dataset_imagenet as imagenet


def _pattern_images(root, count=9):
    pattern = (np.arange(128 * 140 * 3).reshape(128, 140, 3) % 251).astype(np.uint8)
    for index in range(count):
        folder = root / f"class{index % 3}"
        folder.mkdir(exist_ok=True)
        Image.fromarray(np.roll(pattern, index * 7, axis=1)).save(folder / f"{index}.jpg")


def _collect(loader):
    return [(images.cpu().clone(), labels.cpu().clone()) for images, labels in loader]


@pytest.mark.parametrize("size,repetitions", [(1, 3), (2, 3), (4, 3), (2, 1), (4, 1)])
def test_pipeline_reused_and_rank_stream_resumes_exactly(tmp_path, size, repetitions):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("nvidia.dali")
    from py_src.ml_setup.vit import _vit_b_32_torchvision_collate_fn
    from tool.generate_high_accuracy_model import _capture_rng_state, _restore_rng_state

    _pattern_images(tmp_path)
    dataset = DaliImageNetDataset(
        str(tmp_path), DatasetType.imagenet21k_p, "train", preset_version=1,
        train_crop_size=32, auto_augment_policy="imagenet", device_id=0,
        repeated_augmentation_repetitions=repetitions,
    )
    config = DataloaderConfig(batch_size=4, num_workers=1, seed=17,
                              collate_fn=_vit_b_32_torchvision_collate_fn(3))
    for rank in range(size):
        with patch.object(distributed, "enabled", return_value=size > 1), \
             patch.object(distributed, "world_size", return_value=size), \
             patch.object(distributed, "rank", return_value=rank), \
             patch.object(torch.distributed, "get_backend", return_value="nccl"), \
             patch.object(imagenet, "build_dali_imagenet_pipeline", wraps=imagenet.build_dali_imagenet_pipeline) as build:
            loader = dataset.build_dataloader(4, config, is_train=True)
            assert build.call_count == 0  # no unused pipeline before restore
            loader.set_epoch(0)
            first = _collect(loader)
            iterator = loader._dali_iterator
            state = copy.deepcopy(loader.state_dict())
            rng = _capture_rng_state()
            loader.set_epoch(1)
            expected = _collect(loader)
            loader.set_epoch(2)
            _collect(loader)
            assert loader._dali_iterator is iterator and build.call_count == 1
            assert len(first) == len(expected) == len(loader)
            expected_samples = (9 * repetitions + size - 1) // size
            assert sum(len(images) for images, _ in expected) == expected_samples
            assert state["checkpoint"] and state["next_epoch"] == 1
            assert not state["epoch_seeded"]

            restored = dataset.build_dataloader(4, config, is_train=True)
            invalid = copy.deepcopy(state)
            invalid["signature"]["batch_size"] = 999
            with pytest.raises(ValueError, match="Incompatible DALI"):
                restored.load_state_dict(invalid)
            restored.load_state_dict(state)
            _restore_rng_state(rng, logging.getLogger(__name__))
            restored.set_epoch(1)
            actual = _collect(restored)
            assert build.call_count == 2
            for old, new in zip(expected, actual, strict=True):
                for a, b in zip(old, new, strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert restored.state_dict()["next_epoch"] == 2
            with pytest.raises(ValueError, match="restore the loader checkpoint"):
                restored.set_epoch(7)
            del restored, loader, iterator


def test_interrupted_pass_restarts_without_resetting_unfinished_reader(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("nvidia.dali")
    _pattern_images(tmp_path)
    dataset = DaliImageNetDataset(str(tmp_path), DatasetType.imagenet1k, "train",
                                  preset_version=1, train_crop_size=32)
    config = DataloaderConfig(batch_size=4, num_workers=1, seed=13)
    baseline = _collect(dataset.build_dataloader(4, config, is_train=True))
    loader = dataset.build_dataloader(4, config, is_train=True)
    iterator = iter(loader)
    next(iterator)
    with pytest.raises(RuntimeError, match="complete epoch"):
        loader.state_dict()
    iterator.close()
    with pytest.raises(RuntimeError, match="complete epoch"):
        loader.state_dict()
    replayed = _collect(loader)
    for expected, actual in zip(baseline, replayed, strict=True):
        for a, b in zip(expected, actual, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_reader_checkpoint_allows_new_node_local_dataset_root(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("nvidia.dali")
    roots = [tmp_path / name for name in ("source", "next_allocation")]
    for root in roots:
        root.mkdir()
        _pattern_images(root)
    config = DataloaderConfig(batch_size=4, num_workers=1, seed=13)
    def dataset(root):
        return DaliImageNetDataset(str(root), DatasetType.imagenet1k, "train",
                                   preset_version=1, train_crop_size=32)
    loader = dataset(roots[0]).build_dataloader(4, config, is_train=True)
    _collect(loader)
    state = loader.state_dict()
    expected = _collect(loader)
    del loader
    # Simulate Slurm removing the previous allocation's scratch path.
    roots[0].rename(tmp_path / "old_allocation")
    restored = dataset(roots[1]).build_dataloader(4, config, is_train=True)
    restored.load_state_dict(state)
    actual = _collect(restored)
    for old, new in zip(expected, actual, strict=True):
        for a, b in zip(old, new, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("size", [2, 4])
def test_old_epoch_seeded_checkpoint_preserves_historical_stream(tmp_path, size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("nvidia.dali")
    from tool.generate_high_accuracy_model import _capture_rng_state, _restore_rng_state, restore_dataloader_states
    from py_src.ml_setup.vit import _vit_b_32_torchvision_collate_fn
    source = subprocess.run(
        ["git", "show", "8ec69b1:py_src/ml_setup_dataset/dataset_imagenet.py"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    if source.returncode:
        pytest.skip("Historical dataset builder 8ec69b1 unavailable")
    filename = "git:8ec69b1:dataset_classes.py"
    linecache.cache[filename] = (len(source.stdout), None, source.stdout.splitlines(True), filename)
    nodes = [item for item in ast.parse(source.stdout).body if isinstance(item, ast.ClassDef)
             and item.name in {"_DaliImageNetIterator", "DaliImageNetDataset"}]
    namespace = dict(vars(imagenet))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    _pattern_images(tmp_path)
    arguments = dict(file_root=str(tmp_path), dataset_type=DatasetType.imagenet1k, split="train",
                     preset_version=1, train_crop_size=32, auto_augment_policy="imagenet",
                     repeated_augmentation_repetitions=3)
    config = DataloaderConfig(batch_size=4, num_workers=1, seed=17,
                              collate_fn=_vit_b_32_torchvision_collate_fn(3))
    with patch.object(distributed, "enabled", return_value=True), \
         patch.object(distributed, "world_size", return_value=size), \
         patch.object(distributed, "rank", return_value=size-1), \
         patch.object(torch.distributed, "get_backend", return_value="nccl"):
        historical = namespace["DaliImageNetDataset"](**arguments).build_dataloader(4, config, True)
        current = DaliImageNetDataset(**arguments).build_dataloader(4, config, True)
        restore_dataloader_states({"next_epoch": 3}, current, None, logging.getLogger(__name__))
        for epoch in (3, 4):
            historical.set_epoch(epoch)
            current.set_epoch(epoch)
            rng = _capture_rng_state()
            expected = _collect(historical)
            _restore_rng_state(rng, logging.getLogger(__name__))
            actual = _collect(current)
            for old, new in zip(expected, actual, strict=True):
                for a, b in zip(old, new, strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            state = current.state_dict()
            assert state["epoch_seeded"]
            # A newly saved checkpoint must retain the legacy lifecycle too.
            current = DaliImageNetDataset(**arguments).build_dataloader(4, config, True)
            current.load_state_dict(state)


def test_generator_dali_checkpoint_matches_uninterrupted_training(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("nvidia.dali")
    from tool import generate_high_accuracy_model as generate
    from test.test_vit_imagenet21k import tiny_builder
    from torchvision.models.vision_transformer import VisionTransformer

    for split in ("imagenet21k_train", "imagenet21k_val"):
        folder = tmp_path / split
        folder.mkdir()
        _pattern_images(folder, 6)
    def optimizer_factory(setup, model, *args, **kwargs):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        return optimizer, torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95), 3
    def run(root, epochs, resume=False):
        checkpoint = root / "0" / "training_checkpoint.pt"
        if resume:
            args = ["generate", "--load_checkpoint", str(checkpoint), "--epoch", str(epochs), "--core", "1"]
        else:
            args = ["generate", "-m", "vit_b_32", "-d", "imagenet21k_p", "--dali", "--preset", "3",
                    "--random_seed", "7", "--batch_size", "2", "--core", "1", "--epoch", str(epochs),
                    "--no-compile", "--save_format", "none", "--enable_eval", "--eval_precision", "fp32",
                    "-o", str(root)]
        # Simulate identical fresh process RNGs before the factory constructs
        # parameters (e.g. ViT positional embeddings) outside worker re-init.
        generate.set_seed(123)
        with patch.object(sys, "argv", args):
            generate.main()
        return generate.load_training_checkpoint(str(checkpoint))

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with patch.object(imagenet, "imagenet21k_p_path", tmp_path), \
             patch("torchvision.models.vit_b_32", side_effect=tiny_builder(VisionTransformer)), \
             patch.object(generate.FastTrainingSetup, "get_optimizer_lr_scheduler_epoch", side_effect=optimizer_factory):
            expected = run(tmp_path / "continuous", 3)
            first = run(tmp_path / "resumed", 1)
            assert first["dataloader_state"]["train"]["next_epoch"] == 1
            assert first["dataloader_state"]["validation"]["next_epoch"] == 1
            actual = run(tmp_path / "resumed", 3, resume=True)
        for field in ("state_dict", "optimizer_state_dict", "lr_scheduler_state_dict"):
            torch.testing.assert_close(actual[field], expected[field], rtol=0, atol=0)
        assert actual["dataloader_state"]["train"]["next_epoch"] == 3
        assert actual["run_config"]["dali_loader"]["pipeline_lifecycle"] == "persistent"
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("dataset_name", ["imagenet1k", "imagenet21k_p"])
def test_real_dali_shards_have_equal_training_steps_and_exact_evaluation(tmp_path, size, dataset_name):
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
        file_root=str(tmp_path), dataset_type=DatasetType[dataset_name], split="train",
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
                            augmentation_config=describe_augmentation(dataset_name, 0, input_size=32),
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
