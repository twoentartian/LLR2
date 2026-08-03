from __future__ import annotations

import unittest

import torch
from torch import nn

import simulator_config_cct_agc as cct_agc_config
from py_src.adaptive_gradient_clipping import (
    AdaptiveGradientClipper,
    adaptive_clip_grad_,
)
from py_src.model_average import DFedAvgMAverager
from py_src.simulation_runtime_parameters import RuntimeParameters


class _ModelWithNormalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.normalization = nn.LayerNorm(2)


class AdaptiveGradientClippingTest(unittest.TestCase):
    def test_function_clips_each_output_unit(self):
        parameter = nn.Parameter(torch.tensor([[3.0, 4.0], [0.0, 10.0]]))
        parameter.grad = torch.tensor([[30.0, 40.0], [0.0, 100.0]])

        adaptive_clip_grad_(parameter, clip_factor=0.1, eps=1e-6)

        torch.testing.assert_close(
            parameter.grad,
            torch.tensor([[0.3, 0.4], [0.0, 1.0]]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_hook_excludes_biases_and_normalization_parameters(self):
        model = _ModelWithNormalization()
        with torch.no_grad():
            model.linear.weight.fill_(1.0)
            model.linear.bias.fill_(1.0)
            model.normalization.weight.fill_(1.0)
            model.normalization.bias.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        clipper = AdaptiveGradientClipper(clip_factor=0.1, eps=1e-6).attach(
            model,
            optimizer,
        )

        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 10.0)
        optimizer.step()

        torch.testing.assert_close(
            model.linear.weight,
            torch.full_like(model.linear.weight, 0.9),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(model.linear.bias, torch.full((2,), -9.0))
        torch.testing.assert_close(
            model.normalization.weight,
            torch.full((2,), -9.0),
        )
        torch.testing.assert_close(
            model.normalization.bias,
            torch.full((2,), -9.0),
        )
        self.assertEqual(clipper.parameter_names, ("linear.weight",))
        self.assertEqual(
            set(clipper.excluded_parameter_names),
            {
                "linear.bias",
                "normalization.weight",
                "normalization.bias",
            },
        )

    def test_invalid_hyperparameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "clip_factor"):
            AdaptiveGradientClipper(clip_factor=0.0)
        with self.assertRaisesRegex(ValueError, "eps"):
            AdaptiveGradientClipper(eps=0.0)

    def test_cct_config_uses_agc_with_dfedavgm(self):
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

        class Target:
            pass

        target = Target()
        target.model = model
        target.optimizer = optimizer
        averager = cct_agc_config.get_average_algorithm(
            target,
            RuntimeParameters(),
        )

        self.assertIsInstance(averager, DFedAvgMAverager)
        self.assertIsNotNone(averager.variance_corrector)
        self.assertEqual(
            averager.variance_corrector.variance_correction_type,
            cct_agc_config.preset_variance_correction,
        )
        model.weight.grad = torch.full_like(model.weight, 10.0)
        optimizer.step()
        torch.testing.assert_close(
            model.weight,
            torch.full_like(model.weight, 0.99),
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
