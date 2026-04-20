from __future__ import annotations

import tempfile
import unittest

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from py_src.adapters import StandardAdapter
from py_src.engine import Device
from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup_model import ModelType
from py_src.service.record_consecutive_linear_interpolation import (
    ServiceConsecutiveLinearInterpolationRecorder,
)
from py_src.simulation_runtime_parameters import SimulationPhase


class _ModelWithNonPersistentBuffer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.register_buffer(
            "position_ids",
            torch.arange(4, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestConsecutiveLinearInterpolationRecorder(unittest.TestCase):
    def test_ignores_non_persistent_buffers_missing_from_state_dict(self) -> None:
        model = _ModelWithNonPersistentBuffer()
        criterion = nn.CrossEntropyLoss()
        adapter = StandardAdapter(model, criterion)
        dataset = TensorDataset(torch.randn(4, 4), torch.tensor([0, 1, 0, 1]))
        ml_setup = MLSetup(
            model=model,
            adapter=adapter,
            model_type=ModelType.lenet5,
            training_data=dataset,
            testing_data=dataset,
            dataset_type=DatasetType.cifar10,
            default_batch_size=2,
            criterion=criterion,
        )

        with tempfile.TemporaryDirectory(prefix="consec_interp_test_") as output_dir:
            service = ServiceConsecutiveLinearInterpolationRecorder(
                interval=1,
                batch_size=2,
                dataset_size=4,
                points_size=3,
                recorded_node_name=0,
                training_mode=False,
            )
            service.initialize_without_runtime_parameters(
                output_path=output_dir,
                model=model,
                criterion=criterion,
                train_dataset=dataset,
                ml_setup=ml_setup,
                device=Device.cpu(),
            )

            start_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            end_state = {k: v.detach().clone() for k, v in start_state.items()}
            end_state["linear.weight"] = end_state["linear.weight"] + 0.1
            end_state["linear.bias"] = end_state["linear.bias"] - 0.05

            service.trigger_without_runtime_parameters(0, SimulationPhase.START_OF_TICK, start_state)
            service.trigger_without_runtime_parameters(0, SimulationPhase.END_OF_TICK, end_state)

            restored_state = model.state_dict()
            for key, expected_value in end_state.items():
                self.assertTrue(torch.equal(restored_state[key], expected_value), key)


if __name__ == "__main__":
    unittest.main()
