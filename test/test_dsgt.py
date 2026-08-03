from __future__ import annotations

import unittest

import torch
from torch import nn

from py_src.model_average import (
    DSGTModelAverager,
    VarianceCorrectionType,
    VarianceCorrector,
)


class _ScalarModel(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[value]]))


class _VectorModel(nn.Module):
    def __init__(self, values: list[float]):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(values))


def _make_dsgt_node(model: nn.Module, *, variance_corrector=None):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    averager = DSGTModelAverager(
        variance_corrector=variance_corrector,
    ).attach(model, optimizer)
    return optimizer, averager


def _step_with_gradient(model: nn.Module, optimizer, gradient) -> None:
    model.weight.grad = torch.as_tensor(gradient).reshape_as(model.weight)
    optimizer.step()


def _communicate_pair(models, averagers) -> None:
    payloads = [
        averager.get_communication_payload(model.state_dict())
        for model, averager in zip(models, averagers)
    ]
    averagers[0].add_model(payloads[1])
    averagers[1].add_model(payloads[0])
    outputs = [
        averager.get_model(self_model=model.state_dict())
        for model, averager in zip(models, averagers)
    ]
    for model, output in zip(models, outputs):
        model.load_state_dict(output)


class DSGTTest(unittest.TestCase):
    def test_equations_and_independent_initialization(self):
        models = [_ScalarModel(1.0), _ScalarModel(3.0)]
        node_state = [_make_dsgt_node(model) for model in models]
        optimizers = [state[0] for state in node_state]
        averagers = [state[1] for state in node_state]

        # The first step is suppressed; communication computes X[1] = W X[0].
        _step_with_gradient(models[0], optimizers[0], [[99.0]])
        _step_with_gradient(models[1], optimizers[1], [[99.0]])
        _communicate_pair(models, averagers)
        torch.testing.assert_close(models[0].weight, torch.tensor([[2.0]]))
        torch.testing.assert_close(models[1].weight, torch.tensor([[2.0]]))

        # Y[1] = G[1], q_i = X[1] - eta Y_i[1], X[2] = W q.
        _step_with_gradient(models[0], optimizers[0], [[1.0]])
        _step_with_gradient(models[1], optimizers[1], [[2.0]])
        torch.testing.assert_close(models[0].weight, torch.tensor([[1.9]]))
        torch.testing.assert_close(models[1].weight, torch.tensor([[1.8]]))
        _communicate_pair(models, averagers)
        torch.testing.assert_close(models[0].weight, torch.tensor([[1.85]]))
        torch.testing.assert_close(models[1].weight, torch.tensor([[1.85]]))
        for averager in averagers:
            mixed_tracker = averager.get_mixed_tracker_stat()
            torch.testing.assert_close(
                mixed_tracker["weight"],
                torch.tensor([[1.5]]),
            )

        # Y_i[2] = 1.5 + G_i[2] - G_i[1].
        _step_with_gradient(models[0], optimizers[0], [[1.2]])
        _step_with_gradient(models[1], optimizers[1], [[2.3]])
        torch.testing.assert_close(models[0].weight, torch.tensor([[1.68]]))
        torch.testing.assert_close(models[1].weight, torch.tensor([[1.67]]))
        torch.testing.assert_close(
            averagers[0].get_tracker_stat()["weight"],
            torch.tensor([[1.7]]),
        )
        torch.testing.assert_close(
            averagers[1].get_tracker_stat()["weight"],
            torch.tensor([[1.8]]),
        )

    def test_weighted_vc_changes_model_but_not_tracker(self):
        models = [_VectorModel([-1.0, 1.0]), _VectorModel([-3.0, 3.0])]
        correctors = [
            VarianceCorrector(VarianceCorrectionType.FollowConservative),
            VarianceCorrector(VarianceCorrectionType.FollowConservative),
        ]
        node_state = [
            _make_dsgt_node(model, variance_corrector=corrector)
            for model, corrector in zip(models, correctors)
        ]
        optimizers = [state[0] for state in node_state]
        averagers = [state[1] for state in node_state]

        for model, optimizer in zip(models, optimizers):
            _step_with_gradient(model, optimizer, [7.0, 9.0])
        _communicate_pair(models, averagers)

        for model, averager in zip(models, averagers):
            torch.testing.assert_close(
                torch.var(model.weight),
                torch.tensor(10.0),
                atol=1e-4,
                rtol=1e-4,
            )
            torch.testing.assert_close(
                averager.get_mixed_tracker_stat()["weight"],
                torch.zeros(2),
            )

    def test_payload_contains_model_and_tracker(self):
        model = _ScalarModel(1.0)
        _, averager = _make_dsgt_node(model)

        payload = averager.get_communication_payload(model.state_dict())

        self.assertEqual(payload["algorithm"], "DSGT")
        self.assertIn("model", payload)
        self.assertIn("gradient_tracker", payload)
        torch.testing.assert_close(
            payload["gradient_tracker"]["weight"],
            torch.zeros(1, 1),
        )

    def test_non_plain_sgd_optimizer_is_rejected(self):
        model = _ScalarModel(1.0)
        with self.assertRaisesRegex(TypeError, "torch.optim.SGD"):
            DSGTModelAverager().attach(
                model,
                torch.optim.AdamW(model.parameters()),
            )

        with self.assertRaisesRegex(ValueError, "zero momentum"):
            DSGTModelAverager().attach(
                model,
                torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
            )

    def test_alternate_optimizer_can_be_enabled_explicitly(self):
        model = _ScalarModel(1.0)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.1,
            weight_decay=0.5,
        )
        DSGTModelAverager(enforce_plain_sgd=False).attach(model, optimizer)

        # Initial exchange suppression must also suppress AdamW weight decay.
        _step_with_gradient(model, optimizer, [[99.0]])
        torch.testing.assert_close(model.weight, torch.tensor([[1.0]]))

    def test_multiple_local_steps_before_communication_are_rejected(self):
        model = _ScalarModel(1.0)
        optimizer, _ = _make_dsgt_node(model)
        _step_with_gradient(model, optimizer, [[1.0]])

        with self.assertRaisesRegex(RuntimeError, "one optimizer step"):
            _step_with_gradient(model, optimizer, [[1.0]])


if __name__ == "__main__":
    unittest.main()
