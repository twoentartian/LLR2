from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from py_src.adapters import DiffusionAdapter, StandardAdapter
from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup_model import ModelType
from tool.find_high_accuracy_path.functions import _try_get_criterion, rebuild_norm_layer_function
import tool.find_high_accuracy_path as find_high_accuracy_path_pkg


def _clone_state_dict(state_dict):
    output = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            output[key] = value.detach().clone()
        else:
            output[key] = value
    return output


class _ToyDiffusionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(4)
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.proj(x)
        return x.pow(2).mean()


class _ToyTrainDiffusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.register_buffer("betas", torch.tensor([0.1, 0.2]))


class _ToyDiffusionWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.train_diffusion = _ToyTrainDiffusion()
        self.ema = nn.Linear(4, 4)
        self.register_buffer("ema_step", torch.tensor(0))

    def parameters(self, recurse: bool = True):
        return self.train_diffusion.parameters(recurse=recurse)

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        return self.train_diffusion.named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )


class TestFindHighAccuracyPathHelpers(unittest.TestCase):
    def test_build_compensate_destination_only_uses_requested_floating_layers(self) -> None:
        impl = find_high_accuracy_path_pkg._load_impl_module()
        source_state = {
            "proj.weight": torch.tensor([2.0, 3.0]),
            "proj.mask": torch.tensor([True, False]),
        }
        end_state = {
            "proj.weight": torch.tensor([0.5, 1.5]),
            "proj.mask": torch.tensor([False, True]),
        }

        destination, skipped_layers = impl._build_compensate_destination(
            source_state,
            end_state,
            ["proj.weight"],
        )

        self.assertEqual(skipped_layers, [])
        self.assertEqual(sorted(destination.keys()), ["proj.weight"])
        self.assertTrue(torch.equal(destination["proj.weight"], torch.tensor([3.5, 4.5])))

        destination, skipped_layers = impl._build_compensate_destination(
            source_state,
            end_state,
            ["proj.weight", "proj.mask"],
        )

        self.assertEqual(skipped_layers, ["proj.mask"])
        self.assertEqual(sorted(destination.keys()), ["proj.weight"])

    def test_apply_ml_setup_compatibility_backfills_standard_adapter_criterion(self) -> None:
        model = nn.Linear(4, 2)
        criterion = nn.CrossEntropyLoss()
        ml_setup = MLSetup(
            model=model,
            adapter=StandardAdapter(model, criterion),
            model_type=ModelType.lenet5,
            training_data=TensorDataset(torch.randn(2, 4), torch.tensor([0, 1])),
            testing_data=TensorDataset(torch.randn(2, 4), torch.tensor([0, 1])),
            dataset_type=DatasetType.cifar10,
            default_batch_size=2,
        )

        impl = find_high_accuracy_path_pkg._load_impl_module()
        impl._apply_ml_setup_compatibility(ml_setup)

        self.assertIs(ml_setup.criterion, criterion)

    def test_try_get_criterion_returns_none_for_diffusion_adapter(self) -> None:
        model = _ToyDiffusionModel()
        dataset = TensorDataset(torch.randn(4, 4), torch.zeros(4, dtype=torch.long))
        ml_setup = MLSetup(
            model=model,
            adapter=DiffusionAdapter(model),
            model_type=ModelType.ddpm_cifar10,
            training_data=dataset,
            testing_data=dataset,
            dataset_type=DatasetType.cifar10,
            default_batch_size=2,
        )

        self.assertIsNone(_try_get_criterion(ml_setup))

    def test_trainable_state_key_detection_ignores_ema_and_buffers(self) -> None:
        model = _ToyDiffusionWrapper()
        model_state = model.state_dict()
        impl = find_high_accuracy_path_pkg._load_impl_module()

        trainable_keys = impl._get_trainable_state_keys(model, model_state)
        non_trainable_keys = impl._get_non_trainable_state_keys(model, model_state)

        self.assertEqual(
            trainable_keys,
            [
                "train_diffusion.proj.bias",
                "train_diffusion.proj.weight",
            ],
        )
        self.assertIn("train_diffusion.betas", non_trainable_keys)
        self.assertIn("ema.weight", non_trainable_keys)
        self.assertIn("ema.bias", non_trainable_keys)
        self.assertIn("ema_step", non_trainable_keys)
        self.assertNotIn("train_diffusion.proj.weight", non_trainable_keys)

    def test_state_dict_equality_on_trainable_keys_ignores_ema_only_differences(self) -> None:
        model = _ToyDiffusionWrapper()
        impl = find_high_accuracy_path_pkg._load_impl_module()
        state_a = _clone_state_dict(model.state_dict())
        state_b = _clone_state_dict(model.state_dict())
        state_b["ema.weight"] = state_b["ema.weight"] + 1.0
        state_b["train_diffusion.betas"] = state_b["train_diffusion.betas"] + 1.0

        trainable_keys = impl._get_trainable_state_keys(model, state_a)

        self.assertFalse(impl._state_dicts_equal(state_a, state_b))
        self.assertTrue(impl._state_dicts_equal_on_keys(state_a, state_b, trainable_keys))

    def test_rebuild_norm_layer_function_uses_diffusion_adapter_training_step(self) -> None:
        torch.manual_seed(0)
        model = _ToyDiffusionModel()
        dataset = TensorDataset(
            torch.arange(32, dtype=torch.float32).view(8, 4),
            torch.zeros(8, dtype=torch.long),
        )
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        ml_setup = MLSetup(
            model=model,
            adapter=DiffusionAdapter(model),
            model_type=ModelType.ddpm_cifar10,
            training_data=dataset,
            testing_data=dataset,
            dataset_type=DatasetType.cifar10,
            default_batch_size=4,
        )
        rebuild_optimizer = torch.optim.SGD(model.norm.parameters(), lr=0.01)
        initial_state = _clone_state_dict(model.state_dict())

        rebuild_norm_layer_function(
            model,
            initial_state,
            initial_state,
            rebuild_optimizer,
            rebuild_optimizer.state_dict(),
            [],
            ml_setup,
            dataloader,
            SimpleNamespace(
                rebuild_norm_for_max_rounds=1,
                rebuild_norm_for_min_rounds=1,
                rebuild_norm_until_loss=0.0,
                rebuild_norm_use_initial_norm_weights=False,
                rebuild_norm_use_start_model_norm_weights=False,
            ),
            SimpleNamespace(use_amp=False, verbose=False, current_tick=0),
            torch.device("cpu"),
        )

        self.assertFalse(torch.equal(model.norm.running_mean, initial_state["norm.running_mean"]))


if __name__ == "__main__":
    unittest.main()
