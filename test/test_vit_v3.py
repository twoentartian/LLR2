"""Compare preset 3 against the actual pre-fix ViT source, not a second recipe copy."""

from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys
from unittest.mock import patch

import pytest
import torch
from PIL import Image
import numpy as np

from py_src.ml_setup import get_ml_setup_from_config
from py_src.ml_setup.dataloader_util import DataloaderConfig
from py_src.ml_setup_dataset import dataset_imagenet as imagenet
from py_src.util import set_seed
from test.test_vit_imagenet21k import VARIANTS, tiny_builder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import misalignment_measurement_2 as measurement


@pytest.fixture(scope="module")
def historical_vit():
    root = Path(__file__).resolve().parents[1]
    source = subprocess.run(["git", "show", "7484fdc:py_src/ml_setup/vit.py"], cwd=root,
                            capture_output=True, text=True)
    if source.returncode:
        pytest.skip("Historical commit 7484fdc unavailable in this checkout")
    namespace = {"__name__": "legacy_vit_7484fdc", "__package__": "py_src.ml_setup"}
    exec(compile(source.stdout, "git:7484fdc:py_src/ml_setup/vit.py", "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("model_name", VARIANTS)
@pytest.mark.parametrize("dataset_name", ["imagenet1k", "imagenet21k_p"])
@pytest.mark.parametrize("dali", [False, True])
def test_v3_matches_historical_transforms_collate_sampler_and_weights(
    tmp_path, historical_vit, model_name, dataset_name, dali,
):
    if dali:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        pytest.importorskip("nvidia.dali")
    splits = ("train", "val") if dataset_name == "imagenet1k" else ("imagenet21k_train", "imagenet21k_val")
    # Nonuniform images exercise crop, interpolation and spatial/color policies.
    pattern = (np.arange(256 * 280 * 3, dtype=np.uint32).reshape(256, 280, 3) % 251).astype(np.uint8)
    for split in splits:
        for label in range(2):
            folder = tmp_path / split / f"n{label:08}"
            folder.mkdir(parents=True)
            Image.fromarray(np.roll(pattern, label * 13, axis=1)).save(folder / "sample.jpg")
    target, constructor = VARIANTS[model_name]
    pipeline_builder = imagenet.build_dali_imagenet_pipeline

    def fixed_pipeline(**kwargs):
        # DALI's default -1 seed is random. Set the same reader/pipeline seeds
        # for the two executions of the identical backend to compare pixels.
        kwargs["reader_kwargs"] = {**kwargs["reader_kwargs"], "seed": 3201}
        kwargs["pipeline_kwargs"] = {**(kwargs.get("pipeline_kwargs") or {}), "seed": 7201}
        return pipeline_builder(**kwargs)

    with patch.object(imagenet, "imagenet1k_path", tmp_path), \
         patch.object(imagenet, "imagenet21k_p_path", tmp_path), \
         patch(target, side_effect=tiny_builder(constructor)), \
         patch.object(imagenet, "build_dali_imagenet_pipeline", side_effect=fixed_pipeline):
        set_seed(71)
        legacy_builder = getattr(historical_vit, f"{model_name}_{dataset_name}")
        if dataset_name == "imagenet1k":
            override = imagenet.dataset_imagenet1k_from_pytorch_dali() if dali else None
            legacy = legacy_builder(1, override_dataset=override)
        else:
            legacy = legacy_builder(1, use_dali=dali)
        set_seed(71)
        current = get_ml_setup_from_config(model_name, dataset_name, preset=3, use_dali=dali)
        for name, value in legacy.model.state_dict().items():
            torch.testing.assert_close(value, current.model.state_dict()[name], rtol=0, atol=0)
        assert legacy.criterion.label_smoothing == current.criterion.label_smoothing == 0.11
        if not dali:
            assert list(legacy.default_sampler_fn(legacy.training_data)) == list(
                current.default_sampler_fn(current.training_data))
        config = DataloaderConfig(batch_size=2, num_workers=1 if dali else 0, seed=71)

        def collect(setup, train):
            set_seed(2026)
            loader = setup.train_dataloader(config) if train else setup.val_dataloader(config)
            result = [(images.cpu().clone(), labels.cpu().clone()) for images, labels in loader]
            del loader
            return result

        for train in (True, False):
            old_batches, new_batches = collect(legacy, train), collect(current, train)
            assert len(old_batches) == len(new_batches) == (3 if train else 1)
            for old, new in zip(old_batches, new_batches, strict=True):
                for expected, actual in zip(old, new, strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("preset, smoothing", [(None, 0.11), (1, 0.0), (3, 0.11)])
def test_misalignment_default_legacy_criterion_and_explicit_override(tmp_path, preset, smoothing):
    from test.test_imagenet21k_p import _write_images

    _write_images(tmp_path)
    args = SimpleNamespace(dataset="imagenet21k_p", model_type="vit_b_32", model_preset=preset,
                           augmentation=0, random_seed=7, split_seed=11, odd_size_policy="drop")
    target, constructor = VARIANTS["vit_b_32"]
    with patch.object(imagenet, "imagenet21k_p_path", tmp_path), \
         patch(target, side_effect=tiny_builder(constructor)):
        bundle = measurement.load_dataset_bundle(args)
    assert bundle.model_preset == (3 if preset is None else preset)
    assert bundle.ml_setup.criterion.label_smoothing == smoothing
    loader = measurement._build_loader(bundle.partition_a, 2, device=torch.device("cpu"),
                                       num_workers=0, seed=7,
                                       augmentation_config=measurement.describe_augmentation("imagenet21k_p", 0))
    images, labels = next(iter(loader))
    assert images.shape == (2, 3, 224, 224)
    assert labels.shape == (2,) and labels.dtype == torch.long
