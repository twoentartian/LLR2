from __future__ import annotations

import unittest

import torch
from torch import nn

from py_src.model_average import (
    DecentralizedFedProxAverager,
    VarianceCorrectionType,
    VarianceCorrector,
)


class _ScalarModel(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[value]]))


class _VectorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([-1.0, 1.0]))


class DecentralizedFedProxTest(unittest.TestCase):
    def test_first_local_step_moves_toward_buffered_peer_average(self):
        model = _ScalarModel(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        averager = DecentralizedFedProxAverager(mu=0.5).attach(model, optimizer)
        averager.add_model({"weight": torch.tensor([[3.0]])})
        averager.add_model({"weight": torch.tensor([[5.0]])})

        mixed_model = averager.get_model(self_model=model.state_dict())
        model.load_state_dict(mixed_model)

        reference = averager.get_reference_model_stat()
        self.assertIsNotNone(reference)
        torch.testing.assert_close(reference["weight"], torch.tensor([[4.0]]))
        torch.testing.assert_close(model.weight, torch.tensor([[3.0]]))

        optimizer.zero_grad(set_to_none=True)
        optimizer.step()

        torch.testing.assert_close(model.weight, torch.tensor([[3.05]]))
        self.assertAlmostEqual(averager.last_proximal_loss, 0.25)

    def test_variance_correction_is_optional(self):
        model = _VectorModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        corrector = VarianceCorrector(VarianceCorrectionType.FollowOthers)
        averager = DecentralizedFedProxAverager(
            mu=0.1,
            variance_corrector=corrector,
        ).attach(model, optimizer)
        averager.add_model({"weight": torch.tensor([-3.0, 3.0])})
        averager.add_model({"weight": torch.tensor([-5.0, 5.0])})

        mixed_model = averager.get_model(self_model=model.state_dict())

        torch.testing.assert_close(
            torch.var(mixed_model["weight"]),
            torch.tensor(34.0),
            atol=1e-4,
            rtol=1e-4,
        )
        reference = averager.get_reference_model_stat()
        self.assertIsNotNone(reference)
        torch.testing.assert_close(reference["weight"], torch.tensor([-4.0, 4.0]))

    def test_mu_zero_does_not_modify_task_gradient(self):
        model = _ScalarModel(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        averager = DecentralizedFedProxAverager(mu=0.0).attach(model, optimizer)
        averager.add_model({"weight": torch.tensor([[3.0]])})
        mixed_model = averager.get_model(self_model=model.state_dict())
        model.load_state_dict(mixed_model)

        model.weight.grad = torch.tensor([[2.0]])
        optimizer.step()

        torch.testing.assert_close(model.weight, torch.tensor([[1.8]]))
        self.assertEqual(averager.last_proximal_loss, 0.0)

    def test_attach_is_required_before_averaging(self):
        averager = DecentralizedFedProxAverager(mu=0.1)
        averager.add_model({"weight": torch.tensor([[3.0]])})

        with self.assertRaisesRegex(RuntimeError, "attach"):
            averager.get_model(self_model={"weight": torch.tensor([[1.0]])})

    def test_negative_mu_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            DecentralizedFedProxAverager(mu=-0.1)


if __name__ == "__main__":
    unittest.main()
