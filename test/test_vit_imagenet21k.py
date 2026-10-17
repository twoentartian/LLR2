"""ViT/ImageNet-21k-p factory, real loaders and experiment entry points."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import TensorDataset
from torchvision.models.vision_transformer import VisionTransformer

from py_src.complete_ml_setup import FastTrainingSetup
from py_src.engine import Device, train, val
from py_src.ml_setup import get_ml_setup_from_config, vit_b_32_imagenet21k_p
from py_src.ml_setup.dataloader_util import DataloaderConfig
from py_src.ml_setup_dataset import DatasetSetup, DatasetType
from py_src.ml_setup_model.vit_registers import VisionTransformerWithRegisters
from py_src.ml_setup_model.vit_rope import VisionTransformerWithRoPE
from py_src.ml_setup_model.vit_swiglu import VisionTransformerWithSwiGLU
from py_src.model_opti_save_load import load_model_state_file, save_model_state
from test.test_imagenet21k_p import _write_images

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import misalignment_measurement_2 as measurement
from misalignment.augmentation import describe_augmentation
from misalignment.dali_loader import close_dali_loaders, resolve_loader_backend
from misalignment.optimizer import MisalignmentOptimizationSetup


VARIANTS = {
    "vit_b_32": ("torchvision.models.vit_b_32", VisionTransformer),
    "vit_b_32_reg4": ("py_src.ml_setup_model.vit_registers.vit_b_32_reg4", VisionTransformerWithRegisters),
    "vit_b_32_rope": ("py_src.ml_setup_model.vit_rope.vit_b_32_rope", VisionTransformerWithRoPE),
    "vit_b_32_swiglu": ("py_src.ml_setup_model.vit_swiglu.vit_b_32_swiglu", VisionTransformerWithSwiGLU),
}


def tiny_builder(constructor):
    def build(*, num_classes, **kwargs):
        return constructor(image_size=224, patch_size=32, num_layers=1, num_heads=2,
                           hidden_dim=16, mlp_dim=32, num_classes=num_classes)
    return build


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("model_name", VARIANTS)
@pytest.mark.parametrize("dali", [False, True])
def test_factory_train_val_and_checkpoint(tmp_path, model_name, dali):
    if dali:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        pytest.importorskip("nvidia.dali")
    _write_images(tmp_path)
    target, constructor = VARIANTS[model_name]
    with patch("py_src.ml_setup_dataset.dataset_imagenet.imagenet21k_p_path", tmp_path), \
         patch(target, side_effect=tiny_builder(constructor)):
        setup = get_ml_setup_from_config(model_name, "imagenet21k_p", preset=1, use_dali=dali)
        assert setup.model.heads.head.out_features == 2
        assert setup.dataset_type == DatasetType.imagenet21k_p
        config = DataloaderConfig(batch_size=2, num_workers=1 if dali else 0, pin_memory=dali)
        train_loader, val_loader = setup.train_dataloader(config), setup.val_dataloader(config)
        # Dynamic soft-label width must match the classifier, including DALI's
        # GPU-resident batches and the mixed/cut-mixed targets.
        images, targets = next(iter(train_loader))
        assert images.shape == (2, 3, 224, 224)
        assert targets.shape == (2, 2)
        torch.testing.assert_close(targets.sum(dim=1), torch.ones(2, device=targets.device))
        device = Device.cuda() if dali else Device.cpu()
        setup.model.to(device.device)
        optimizer, scheduler, epochs = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(
            setup, setup.model, override_steps_per_epoch=len(train_loader),
        )
        assert epochs == 90
        assert optimizer.param_groups[0]["initial_lr"] == pytest.approx(5e-4)
        assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.05)
        assert scheduler._schedulers[0].total_iters == 5 * len(train_loader)
        result = train(setup.adapter, train_loader, optimizer, scheduler, device=device)
        evaluated = val(setup.adapter, val_loader, device=device)
        assert result.total_count == 6  # three augmentations of each source image
        assert evaluated.total_count == 2
        assert math.isfinite(result.avg_loss) and math.isfinite(evaluated.avg_loss)
        path = str(tmp_path / "model.pt")
        save_model_state(path, setup.model.state_dict(), model_name, setup.dataset_type.name)
        state, saved_model, saved_dataset = load_model_state_file(path, map_location="cpu")
        restored = get_ml_setup_from_config(saved_model, saved_dataset, preset=1)
        restored.model.load_state_dict(state, strict=True)
        inputs = torch.randn(2, 3, 224, 224)
        restored.model.eval()
        setup.model.cpu().eval()
        with torch.no_grad():
            torch.testing.assert_close(restored.model(inputs), setup.model(inputs))
        del train_loader, val_loader


def test_full_class_head_without_class_metadata():
    dataset = TensorDataset(torch.zeros(2, 3, 224, 224), torch.tensor([0, 10449]))
    with patch("torchvision.models.vit_b_32", side_effect=tiny_builder(VisionTransformer)):
        setup = vit_b_32_imagenet21k_p(override_dataset=DatasetSetup(DatasetType.imagenet21k_p, dataset, dataset))
    assert setup.model.heads.head.out_features == 10450
    _, targets = next(iter(setup.train_dataloader(DataloaderConfig(batch_size=2))))
    assert targets.shape == (2, 10450)


@pytest.mark.parametrize("dataset_name", ["imagenet21k_p", "imagenet21k-p", "imagenet21kp", "imagenet21k"])
def test_cli_aliases_and_dali_forwarding(dataset_name):
    with patch("py_src.ml_setup.vit.vit_b_32_imagenet21k_p") as builder:
        get_ml_setup_from_config("vit_b_32", dataset_name, preset=1, use_dali=True, dali_device_id=2)
        builder.assert_called_once_with(1, use_dali=True, dali_device_id=2)
    with patch.object(sys, "argv", ["measurement", "--dataset", dataset_name]):
        assert measurement.parse_args().dataset == "imagenet21k_p"


@pytest.mark.parametrize("level", [0, 1])
def test_misalignment_bundle_and_dali_pipeline(tmp_path, level):
    _write_images(tmp_path)
    args = SimpleNamespace(dataset="imagenet21k_p", model_type=None, augmentation=level,
                           random_seed=7, split_seed=11, odd_size_policy="drop")
    with patch("py_src.ml_setup_dataset.dataset_imagenet.imagenet21k_p_path", tmp_path), \
         patch("torchvision.models.vit_b_32", side_effect=tiny_builder(VisionTransformer)):
        bundle = measurement.load_dataset_bundle(args)
    assert len(bundle.partition_a) == len(bundle.partition_b) == 2
    assert bundle.ml_setup.model.heads.head.out_features == 2
    config = describe_augmentation("imagenet21k_p", level)
    assert config["input_size"] == 224
    assert resolve_loader_backend("imagenet21k_p") == "dali"
    loader = measurement._build_loader(bundle.partition_a, 2, device=torch.device("cpu"),
                                       num_workers=0, seed=7, augmentation_config=config)
    images, targets = next(iter(loader))
    assert images.shape == (2, 3, 224, 224)
    assert bundle.ml_setup.model(images).shape == (2, 2)
    if torch.cuda.is_available():
        pytest.importorskip("nvidia.dali")
        loader = measurement._build_loader(bundle.partition_a, 2, device=torch.device("cuda"),
                                           num_workers=1, seed=7, loader_backend="dali",
                                           augmentation_config=config)
        try:
            images, dali_targets = next(iter(loader))
            assert images.shape == (2, 3, 224, 224)
            torch.testing.assert_close(dali_targets.cpu(), targets)
        finally:
            close_dali_loaders(loader)


def test_level_two_is_rejected_before_dataset_loading():
    with patch.object(sys, "argv", ["measurement", "--dataset", "imagenet21k_p", "--augmentation", "2",
                                   "--device", "cpu", "--loader_backend", "pytorch"]), \
         patch.object(measurement, "load_dataset_bundle") as load:
        with pytest.raises(ValueError, match="supports augmentation levels none/0, 1"):
            measurement.main()
        load.assert_not_called()


def test_misalignment_entry_point_saves_models_with_flatness_deferred(tmp_path):
    _write_images(tmp_path)
    output = tmp_path / "results"
    argv = ["measurement", "--dataset", "imagenet21k", "--device", "cpu", "--batch_size", "2",
            "--epochs", "1", "--warmup_epochs", "0", "--num_workers", "0", "--random_seed", "7",
            "--loader_backend", "pytorch", "--no-compile", "--no-amp", "--skip_relative_flatness",
            "--output_folder_name", str(output)]
    with patch.object(sys, "argv", argv), \
         patch("py_src.ml_setup_dataset.dataset_imagenet.imagenet21k_p_path", tmp_path), \
         patch("torchvision.models.vit_b_32", side_effect=tiny_builder(VisionTransformer)):
        measurement.main()
    summary = json.loads((output / "summary.json").read_text())
    for name, run in summary["runs"].items():
        assert run["dataset"] == "imagenet21k_p"
        assert run["model_type"] == "vit_b_32"
        assert run["optimizer"]["optimizer"] == "adamw"
        assert run["relative_flatness_status"] == "deferred"
        assert (output / f"final_{name}.model.pt").is_file()


def test_misalignment_preset_and_explicit_overrides():
    setup = SimpleNamespace(model_type="vit_b_32", dataset_type=DatasetType.imagenet21k_p)
    config = MisalignmentOptimizationSetup.resolve_config(setup)
    assert (config.epochs, config.warmup_epochs) == (90, 5)
    assert (config.optimizer, config.learning_rate, config.weight_decay) == ("adamw", 5e-4, 0.05)
    overridden = MisalignmentOptimizationSetup.resolve_config(
        setup, optimizer="sgd", learning_rate=0.01, weight_decay=0.0, epochs=2, warmup_epochs=0,
    )
    assert (overridden.optimizer, overridden.learning_rate, overridden.epochs) == ("sgd", 0.01, 2)
