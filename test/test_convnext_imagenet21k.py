"""ConvNeXt 90-epoch recipe, real CPU/CUDA loaders and resumable training."""

import copy
import math
from pathlib import Path
import sys
import warnings
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torchvision.models.convnext import ConvNeXt, CNBlockConfig

from py_src.complete_ml_setup import FastTrainingSetup
from py_src.engine import Device, train, val
from py_src.ml_setup import get_ml_setup_from_config
from py_src.ml_setup.convnext import ConvNeXtMixupCollate, initialize_convnext_weights
from py_src.ml_setup.dataloader_util import DataloaderConfig
from py_src.ml_setup_dataset import dataset_imagenet as imagenet
from py_src.util import set_seed
from test.test_imagenet21k_p import _write_images


def tiny_convnext(*, num_classes, **kwargs):
    assert kwargs.get("weights") is None
    assert kwargs.get("stochastic_depth_prob") == 0.0
    return ConvNeXt([CNBlockConfig(8, 16, 1), CNBlockConfig(16, 24, 1),
                     CNBlockConfig(24, 32, 1), CNBlockConfig(32, None, 1)],
                    num_classes=num_classes, stochastic_depth_prob=0.0)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("dali", [False, True])
@pytest.mark.parametrize("preset", [0, 1, 2, 3, 4])
def test_factory_loaders_train_and_validation(tmp_path, dali, preset):
    if dali:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        pytest.importorskip("nvidia.dali")
    _write_images(tmp_path)
    with patch.object(imagenet, "imagenet21k_p_path", tmp_path), \
         patch("torchvision.models.convnext_tiny", side_effect=tiny_convnext):
        setup = get_ml_setup_from_config("convnext_tiny", "imagenet21k_p", preset, use_dali=dali)
    assert setup.model.classifier[-1].out_features == 2
    assert setup.gradient_accumulate_every == 8
    assert setup.reinitialize_model_fn is initialize_convnext_weights
    config = DataloaderConfig(batch_size=2, num_workers=1 if dali else 0, seed=17)
    loader, evaluation = setup.train_dataloader(config), setup.val_dataloader(config)
    device = Device.cuda() if dali else Device.cpu()
    setup.model.to(device.device)
    optimizer, scheduler, epochs = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(
        setup, setup.model, override_batch_size=2,
        override_steps_per_epoch=math.ceil(len(loader) / 8),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        trained = train(setup.adapter, loader, optimizer, scheduler, device=device,
                        gradient_accumulate_every=8, scaler=device.make_scaler() if dali else None)
        evaluated = val(setup.adapter, evaluation, device=device)
    assert trained.total_count == {0: 2, 1: 2, 2: 8, 3: 6, 4: 2}[preset]
    assert evaluated.total_count == 2
    assert math.isfinite(trained.avg_loss) and math.isfinite(evaluated.avg_loss)
    assert epochs == 90
    assert scheduler.last_epoch == 1
    if preset == 4:
        assert setup.criterion.label_smoothing == 0  # smoothing is in timm's mixed targets
        assert setup.default_sampler_fn is None
        if dali:
            assert setup.training_data.augmentation_recipe == "convnext_v1"
            assert setup.training_data.repeated_augmentation_repetitions == 1
            assert loader.runtime_info()["pipeline_lifecycle"] == "persistent"


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_mixup_tail_targets_are_smoothed_once(size):
    collate = ConvNeXtMixupCollate(3)
    images = torch.ones(size, 3, 8, 8)
    targets = torch.arange(size) % 3
    result, labels = collate((images, targets))
    assert result.shape == images.shape
    assert labels.shape == (size, 3)
    torch.testing.assert_close(labels.sum(1), torch.ones(size))
    assert labels.min() >= 0.1 / 3 - 1e-7
    if size % 2:
        assert labels[-1, targets[-1]].item() == pytest.approx(0.9 + 0.1 / 3)


def test_actual_10450_class_model_initialization_fp16_train_and_bf16_eval():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from torch.utils.data import TensorDataset
    from torchvision.models.convnext import CNBlock
    from py_src.ml_setup import convnext_tiny_imagenet21k_p
    from py_src.ml_setup_dataset import DatasetSetup, DatasetType
    from py_src.precision import inference_dtype

    data = TensorDataset(torch.rand(2, 3, 224, 224), torch.tensor([0, 10449]))
    setup = convnext_tiny_imagenet21k_p(override_dataset=DatasetSetup(DatasetType.imagenet21k_p, data, data))
    assert setup.model.classifier[-1].out_features == 10450
    initialize_convnext_weights(setup.model)
    assert setup.model.features[0][0].weight.std().item() == pytest.approx(0.02, abs=0.002)
    for module in setup.model.modules():
        if isinstance(module, CNBlock):
            torch.testing.assert_close(module.layer_scale, torch.full_like(module.layer_scale, 1e-6))
        if isinstance(module, nn.LayerNorm):
            torch.testing.assert_close(module.weight, torch.ones_like(module.weight))
            torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))
    device = Device.cuda()
    setup.model.to(device.device)
    outputs = []
    setup.model.register_forward_hook(lambda m, inputs, output: outputs.append(
        (torch.is_grad_enabled(), output.dtype)))
    config = DataloaderConfig(batch_size=2, num_workers=0)
    optimizer, scheduler, _ = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(setup, setup.model,
                                                                               override_steps_per_epoch=1)
    result = train(setup.adapter, setup.train_dataloader(config), optimizer, scheduler, device=device,
                   scaler=device.make_scaler(), gradient_accumulate_every=8)
    evaluation = val(setup.adapter, setup.val_dataloader(config), device=device, precision="bf16")
    assert math.isfinite(result.avg_loss) and math.isfinite(evaluation.avg_loss)
    assert (True, torch.float16) in outputs
    assert (False, inference_dtype(device.device, "bf16") or torch.float32) in outputs


def test_schedule_and_decay_groups_follow_official_recipe(tmp_path):
    _write_images(tmp_path)
    with patch.object(imagenet, "imagenet21k_p_path", tmp_path), \
         patch("torchvision.models.convnext_tiny", side_effect=tiny_convnext):
        setup = get_ml_setup_from_config("convnext_tiny", "imagenet21k_p", 4)
    optimizer, scheduler, epochs = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(
        setup, setup.model, override_steps_per_epoch=3,
    )
    assert epochs == 90
    assert optimizer.defaults.get("fused") is not True
    no_decay = {id(p) for group in optimizer.param_groups if group["weight_decay"] == 0 for p in group["params"]}
    for name, parameter in setup.model.named_parameters():
        assert (id(parameter) in no_decay) == (parameter.ndim == 1 or name.endswith((".bias", ".layer_scale")))
    assert optimizer.param_groups[0]["lr"] == 0
    lrs = []
    for _ in range(270):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert max(lrs) == pytest.approx(4e-3)
    assert lrs[13] == pytest.approx(4e-3)
    assert lrs[14] == pytest.approx(4e-3)
    assert lrs[-1] == pytest.approx(1e-6)


def test_short_accumulation_group_matches_full_batch_update():
    set_seed(8)
    model = nn.Linear(4, 3)
    expected = copy.deepcopy(model)
    data = torch.randn(10, 4)
    targets = torch.arange(10) % 3
    from py_src.adapters import StandardAdapter
    from torch.utils.data import DataLoader, TensorDataset

    actual_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    expected_optimizer = torch.optim.SGD(expected.parameters(), lr=0.1)
    for start in (0, 6):
        expected_optimizer.zero_grad()
        nn.functional.cross_entropy(expected(data[start:start + 6]), targets[start:start + 6]).backward()
        expected_optimizer.step()
    result = train(StandardAdapter(model, nn.CrossEntropyLoss()),
                   DataLoader(TensorDataset(data, targets), batch_size=2), actual_optimizer,
                   device=Device.cpu(), gradient_accumulate_every=3)
    assert result.total_count == 10
    torch.testing.assert_close(model.state_dict(), expected.state_dict(), rtol=1e-6, atol=1e-7)


def _distributed_accumulation_worker(rank, size, directory):
    import torch.distributed as dist
    from py_src import distributed
    from py_src.adapters import StandardAdapter
    from py_src.ml_setup.dataloader_util import build_dataloader
    from torch.utils.data import TensorDataset

    torch.set_num_threads(1)
    set_seed(38)
    original = nn.Linear(3, 2)
    expected = copy.deepcopy(original)
    data, labels = torch.randn(20, 3), torch.arange(20) % 2
    optimizer = torch.optim.SGD(expected.parameters(), lr=0.1)
    for start in (0, 12):
        optimizer.zero_grad()
        nn.functional.cross_entropy(expected(data[start:start + 12]), labels[start:start + 12]).backward()
        optimizer.step()
    dist.init_process_group("gloo", rank=rank, world_size=size, init_method=f"file://{directory}/group")
    try:
        adapter = StandardAdapter(Device.cpu().wrap_ddp(original), nn.CrossEntropyLoss())
        loader = build_dataloader(TensorDataset(data, labels), 4,
                                  config=DataloaderConfig(shuffle=False, pin_memory=False))
        optimizer = torch.optim.SGD(original.parameters(), lr=0.1)
        train(adapter, loader, optimizer, device=Device.cpu(), gradient_accumulate_every=3)
        torch.testing.assert_close(original.state_dict(), expected.state_dict(), rtol=1e-6, atol=1e-7)
    finally:
        distributed.shutdown()


@pytest.mark.parametrize("size", [2, 4])
def test_distributed_short_accumulation_matches_global_batch(tmp_path, size):
    import torch.multiprocessing as mp

    mp.spawn(_distributed_accumulation_worker, args=(size, str(tmp_path)), nprocs=size, join=True)


def test_all_dali_convnext_operations_execute_without_deprecated_aliases(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("nvidia.dali")
    import numpy as np
    from PIL import Image
    from nvidia.dali import fn, types
    from nvidia.dali.pipeline import pipeline_def
    from nvidia.dali.auto_aug.core import signed_bin
    from py_src.ml_setup_dataset.convnext_augmentation import _randaugment_suite, dali_convnext_erasing

    folder = tmp_path / "class"
    folder.mkdir()
    pattern = (np.arange(32 * 32 * 3).reshape(32, 32, 3) % 256).astype(np.uint8)
    Image.fromarray(pattern).save(folder / "pattern.png")

    @pipeline_def(enable_conditionals=True)
    def pipeline(operation):
        encoded, _ = fn.readers.file(file_root=str(tmp_path))
        image = fn.decoders.image(encoded, output_type=types.RGB).gpu()
        image = operation(image, magnitude_bin=signed_bin(900), num_magnitude_bins=1001,
                          shape=(32, 32), fill_value=128, interp_type=types.INTERP_LINEAR)
        return image

    @pipeline_def(enable_conditionals=True)
    def erasing_pipeline():
        image = fn.constant(fdata=0.0, shape=(3, 32, 32), dtype=types.FLOAT, device="gpu", layout="CHW")
        return dali_convnext_erasing(image, 32, probability=1.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        for operation in _randaugment_suite():
            graph = pipeline(operation, batch_size=2, num_threads=1, device_id=0, seed=42)
            output = graph.run()[0].as_cpu().as_array()
            assert output.shape == (2, 32, 32, 3), operation.name
            assert np.isfinite(output).all(), operation.name
            del graph
        graph = erasing_pipeline(batch_size=4, num_threads=1, device_id=0, seed=42)
        output = graph.run()[0].as_cpu().as_array()
        assert output.shape == (4, 3, 32, 32)
        assert (output != 0).any() and (output == 0).any()
        assert output.min() < 0 and output.max() > 0
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


@pytest.mark.parametrize("dali", [False, True])
def test_entry_point_accumulation_and_resume(tmp_path, dali):
    if dali:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        pytest.importorskip("nvidia.dali")
    from tool import generate_high_accuracy_model as generate

    _write_images(tmp_path)
    # Five batches per epoch, including a short final accumulation group.
    for split in ("imagenet21k_train", "imagenet21k_val"):
        for folder in (tmp_path / split).iterdir():
            for index in range(1, 5):
                (folder / f"{index}.JPEG").write_bytes((folder / "0.JPEG").read_bytes())

    def run(name, epochs, resume=False):
        output = tmp_path / name
        checkpoint = output / "0" / "training_checkpoint.pt"
        if resume:
            argv = ["generate", "--load_checkpoint", str(checkpoint), "--epoch", str(epochs), "--core", "1"]
        else:
            argv = ["generate", "-m", "convnext_tiny", "-d", "imagenet21k_p", "--preset", "4",
                    "--batch_size", "2", "--gradient_accumulate_every", "3", "--epoch", str(epochs),
                    "--random_seed", "42", "--core", "1", "--enable_eval", "--no-compile",
                    "--eval_precision", "fp32", "-o", str(output)]
            argv.append("--dali" if dali else "--cpu")
        # A fresh process need not have the same RNG state before main().
        set_seed(epochs + (100 if resume else 200))
        with patch.object(sys, "argv", argv):
            generate.main()
        return generate.load_training_checkpoint(str(checkpoint))

    with patch.object(imagenet, "imagenet21k_p_path", tmp_path), \
         patch("torchvision.models.convnext_tiny", side_effect=tiny_convnext), \
         patch.object(generate, "MAX_THREAD_PER_PROCESS", 1 if dali else 0), \
         patch.object(torch, "set_num_threads"), \
         torch.backends.cudnn.flags(benchmark=False, deterministic=True):
        # Disable PyTorch augmentation workers to make RNG checkpoint replay exact.
        expected = run("continuous", 2)
        first = run("resumed", 1)
        actual = run("resumed", 2, resume=True)
    assert first["lr_scheduler_state_dict"]["last_epoch"] == 2
    assert actual["lr_scheduler_state_dict"]["last_epoch"] == 4
    assert actual["run_config"]["gradient_accumulate_every"] == 3
    assert actual["run_config"]["convnext_recipe_version"] == 1
    for field in ("state_dict", "optimizer_state_dict", "lr_scheduler_state_dict"):
        torch.testing.assert_close(actual[field], expected[field], rtol=0, atol=0)
