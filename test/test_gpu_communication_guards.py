import unittest
from types import SimpleNamespace

import torch

from py_src.model_average.variance import (
    VarianceCorrectionType,
    VarianceCorrector,
)
from py_src.service.record_test_accuracy_loss import ServiceTestAccuracyLossRecorder
from py_src.simulation_runtime_parameters import SimulationPhase


class _CountingNode:
    def __init__(self):
        self.get_model_stat_calls = 0

    def get_model_stat(self):
        self.get_model_stat_calls += 1
        return {"weight": torch.tensor([1.0])}


class GPUCommunicationGuardTest(unittest.TestCase):
    def test_accuracy_service_does_not_clone_models_outside_its_interval(self):
        recorder = ServiceTestAccuracyLossRecorder(
            interval=10,
            test_batch_size=10,
            model_name="resnet18",
            dataset_name="cifar10",
        )
        node = _CountingNode()
        recorder.node_order = [0]
        parameters = SimpleNamespace(
            phase=SimulationPhase.END_OF_TICK,
            current_tick=9,
            node_container={0: node},
        )

        recorder.trigger(parameters)

        self.assertEqual(node.get_model_stat_calls, 0)

    def test_variance_values_remain_tensors_on_the_model_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_stat = {
            "weight": torch.tensor([-3.0, 3.0], device=device),
            "scalar": torch.tensor(1.0, device=device),
        }
        corrector = VarianceCorrector(VarianceCorrectionType.FollowOthers)
        tensor_device = model_stat["weight"].device

        variance = corrector.calculate_variance_for_tensor(model_stat["weight"])
        self.assertIsInstance(variance, torch.Tensor)
        self.assertEqual(variance.device, tensor_device)

        corrector.add_variance(model_stat)
        self.assertIsNotNone(corrector.variance_record)
        assert corrector.variance_record is not None
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in corrector.variance_record.values()))
        self.assertTrue(all(value.device == tensor_device for value in corrector.variance_record.values()))

        target = corrector.get_variance()
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in target.values()))
        self.assertTrue(all(value.device == tensor_device for value in target.values()))
        torch.testing.assert_close(target["weight"], torch.var(model_stat["weight"]))
        torch.testing.assert_close(target["scalar"], torch.zeros((), device=tensor_device))


if __name__ == "__main__":
    unittest.main()
