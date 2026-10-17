"""Check actual forward dtypes and metric/storage preservation during evaluation."""

from pathlib import Path
import json
import sys
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from py_src.adapters import StandardAdapter
from py_src.engine import Device, val
from py_src.precision import inference_autocast, inference_dtype

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import misalignment_measurement_2 as measurement


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class RecordingLinear(nn.Linear):
    def forward(self, inputs):
        outputs = super().forward(inputs)
        self.output_dtypes.append(outputs.dtype)
        return outputs


@pytest.mark.parametrize("entrypoint", ["engine", "misalignment"])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_evaluation_dtype_metrics_and_parameters(entrypoint, device_name, precision):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device(device_name)
    torch.manual_seed(71)
    model = RecordingLinear(4, 3).to(device)
    model.output_dtypes = []
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    dataset = TensorDataset(torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1]))
    loader = DataLoader(dataset, batch_size=2)
    criterion = nn.CrossEntropyLoss()
    expected_loss = 0.0
    with torch.no_grad(), inference_autocast(device, precision):
        for inputs, target in loader:
            expected_loss += criterion(model(inputs.to(device)), target.to(device)).item() * len(target)
    model.output_dtypes.clear()
    if entrypoint == "engine":
        result = val(StandardAdapter(model, criterion), loader, device=Device(device), precision=precision)
        actual_loss = result.avg_loss
        count = result.total_count
    else:
        result = measurement.evaluate_partition(model, loader, criterion, device=device,
                                                modular=False, eq_position=None, precision=precision)
        actual_loss = result.loss
        count = result.examples
    expected_dtype = inference_dtype(device, precision) or torch.float32
    assert set(model.output_dtypes) == {expected_dtype}
    assert actual_loss == pytest.approx(expected_loss / 5)
    assert count == 5
    assert not torch.is_autocast_enabled(device_name)
    assert all(parameter.grad is None and parameter.dtype == torch.float32 for parameter in model.parameters())
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)


def test_unsupported_cuda_falls_back_without_emulation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    with patch("torch.cuda.is_bf16_supported", return_value=False) as supported:
        assert inference_dtype(torch.device("cuda")) is None
    supported.assert_called_once_with(including_emulation=False)


def test_invalid_precision_is_rejected():
    with pytest.raises(ValueError, match="evaluation precision"):
        inference_dtype(torch.device("cpu"), "fp8")


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("precision", [None, "fp32"])
def test_high_accuracy_cli_default_and_checkpoint_resume(tmp_path, device_name, precision):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import generate_high_accuracy_model as generate
    from py_src.ml_setup.shared_setup_util import make_setup
    from py_src.ml_setup_dataset import DatasetSetup, DatasetType
    from py_src.ml_setup_model import ModelType

    dataset = TensorDataset(torch.randn(4, 4), torch.tensor([0, 1, 2, 1]))
    setup = make_setup(nn.Linear(4, 3), ModelType.vit_b_32,
                       DatasetSetup(DatasetType.imagenet21k_p, dataset, dataset), 2)
    outputs = []
    setup.model.register_forward_hook(lambda module, args, result: outputs.append(
        (torch.is_grad_enabled(), result.dtype)))
    root = tmp_path / "run"
    argv = ["generate", "-m", "vit_b_32", "-d", "imagenet21k_p", "--core", "1", "--epoch", "1",
            "--enable_eval", "--no-compile", "--preset", "0", "-o", str(root)]
    if device_name == "cpu":
        argv.append("--cpu")
    if precision is not None:
        argv += ["--eval_precision", precision]

    def optimizer_factory(ml_setup, model, *args, **kwargs):
        return torch.optim.SGD(model.parameters(), lr=0.001), None, 2

    with patch.object(generate, "get_ml_setup_from_config", return_value=setup), \
         patch.object(generate.FastTrainingSetup, "get_optimizer_lr_scheduler_epoch", side_effect=optimizer_factory), \
         patch.object(sys, "argv", argv):
        generate.main()
    expected_precision = precision or "bf16"
    expected_dtype = inference_dtype(torch.device(device_name), expected_precision) or torch.float32
    assert {dtype for grad_enabled, dtype in outputs if grad_enabled} == {torch.float32}
    assert {dtype for grad_enabled, dtype in outputs if not grad_enabled} == {expected_dtype}
    assert json.loads((root / "info.json").read_text())["eval_precision"] == expected_precision
    checkpoint_path = root / "0" / "training_checkpoint.pt"
    checkpoint = generate.load_training_checkpoint(str(checkpoint_path))
    assert checkpoint["run_config"]["eval_precision"] == expected_precision
    assert checkpoint["run_config"]["vit_preset_version"] == 2
    outputs.clear()
    with patch.object(generate, "get_ml_setup_from_config", return_value=setup), \
         patch.object(generate.FastTrainingSetup, "get_optimizer_lr_scheduler_epoch", side_effect=optimizer_factory), \
         patch.object(sys, "argv", ["generate", "--load_checkpoint", str(checkpoint_path), "--core", "1", "--epoch", "2"]):
        generate.main()
    assert {dtype for grad_enabled, dtype in outputs if not grad_enabled} == {expected_dtype}
    checkpoint = generate.load_training_checkpoint(str(checkpoint_path))
    assert checkpoint["completed_epoch"] == 1
    # A pre-fix preset 0/1 checkpoint maps explicitly to equivalent legacy v3.
    del checkpoint["run_config"]["vit_preset_version"]
    del checkpoint["run_config"]["eval_precision"]
    torch.save(checkpoint, checkpoint_path)
    with patch.object(generate, "get_ml_setup_from_config", return_value=setup) as factory, \
         patch.object(generate.FastTrainingSetup, "get_optimizer_lr_scheduler_epoch", side_effect=optimizer_factory), \
         patch.object(sys, "argv", ["generate", "--load_checkpoint", str(checkpoint_path), "--core", "1", "--epoch", "3"]):
        generate.main()
        assert factory.call_args.args[2] == 3
    migrated = generate.load_training_checkpoint(str(checkpoint_path))
    assert migrated["run_config"]["preset"] == 3
    assert migrated["run_config"]["vit_legacy_preset"] == 0
    assert migrated["run_config"]["vit_preset_version"] == 2
    assert migrated["run_config"]["eval_precision"] == "fp32"


@pytest.mark.parametrize("trainer", ["normalized", "symmetric"])
def test_misalignment_bf16_evaluation_with_fp32_training(tmp_path, trainer):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("Native CUDA BF16 required")
    model = RecordingLinear(4, 3).cuda()
    model.output_dtypes = []
    dataset = TensorDataset(torch.randn(4, 4), torch.tensor([0, 1, 2, 1]))
    loader = DataLoader(dataset, batch_size=2)
    kwargs = dict(device=torch.device("cuda"), modular=False, eq_position=None,
                  optimizer=torch.optim.SGD(model.parameters(), lr=.001), scheduler=None,
                  epochs=1, report_interval=1, csv_path=str(tmp_path / "losses.csv"), amp_enabled=False)
    if trainer == "normalized":
        measurement.train_normalized_two_sided(model, loader, loader, nn.CrossEntropyLoss(),
                                              gradient_geometry_csv_path=str(tmp_path / "geometry.csv"), **kwargs)
    else:
        measurement.train_symmetric_objective(model, loader, loader, nn.CrossEntropyLoss(),
                                             objective_mode="difference", **kwargs)
    assert set(model.output_dtypes) == {torch.float32, torch.bfloat16}
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


@pytest.mark.parametrize("model_name", ["vit_b_32", "vit_b_32_reg4", "vit_b_32_rope", "vit_b_32_swiglu"])
@pytest.mark.parametrize("dali", [False, True])
@pytest.mark.parametrize("preset", [0, 1, 2, 3])
def test_imagenet1k_presets_use_the_same_recipe(tmp_path, model_name, dali, preset):
    if dali and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if dali:
        pytest.importorskip("nvidia.dali")
    from test.test_imagenet21k_p import _write_images
    from test.test_vit_imagenet21k import VARIANTS, tiny_builder
    from py_src.ml_setup import get_ml_setup_from_config
    from py_src.ml_setup.dataloader_util import DataloaderConfig

    _write_images(tmp_path)
    (tmp_path / "imagenet21k_train").rename(tmp_path / "train")
    (tmp_path / "imagenet21k_val").rename(tmp_path / "val")
    target, constructor = VARIANTS[model_name]
    with patch("py_src.ml_setup_dataset.dataset_imagenet.imagenet1k_path", tmp_path), \
         patch(target, side_effect=tiny_builder(constructor)):
        setup = get_ml_setup_from_config(model_name, "imagenet1k", preset=preset, use_dali=dali)
    assert setup.criterion.label_smoothing == {0: 0.0, 1: 0.0, 2: 0.1, 3: 0.11}[preset]
    assert (setup.default_collate_fn is not None) == (preset in (2, 3))
    assert (setup.default_sampler_fn is not None) == (preset in (2, 3))
    if dali:
        assert setup.training_data.augmentation == (preset != 0)
        assert setup.training_data.preset_version == (2 if preset == 2 else 1)
        assert setup.training_data.repeated_augmentation_repetitions == {0: 1, 1: 1, 2: 4, 3: 3}[preset]
        assert setup.training_data.auto_augment_policy == ("imagenet" if preset == 3 else None)
    else:
        operations = setup.training_data.transform.transforms
        operations = operations if isinstance(operations, list) else operations.transforms
        names = [type(transform).__name__ for transform in operations]
        assert ("RandomResizedCrop" in names) == (preset != 0)
        assert ("RandomHorizontalFlip" in names) == (preset != 0)
        assert ("TrivialAugmentWide" in names) == (preset == 2)
        assert ("RandAugment" in names) == (preset == 2)
        assert ("RandomErasing" in names) == (preset == 2)
        assert ("AutoAugment" in names) == (preset == 3)
    loader = setup.train_dataloader(DataloaderConfig(batch_size=2, num_workers=1 if dali else 0))
    inputs, labels = next(iter(loader))
    assert inputs.shape == (2, 3, 224, 224)
    assert labels.shape == ((2, 1000) if preset in (2, 3) else (2,))
    del loader
