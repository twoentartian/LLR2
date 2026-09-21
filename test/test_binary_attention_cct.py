"""Regression tests for the CIFAR-10 binary-attention CCT recipe."""

import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from py_src.complete_ml_setup import FastTrainingSetup
from py_src.ml_setup import (
    binary_attention_cct_7_3x1_cifar10,
    cct_7_3x1_cifar10,
    get_ml_setup_from_config,
)
from py_src.ml_setup_dataset import DatasetSetup, DatasetType
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_model.bnn.binary_cct import BinaryAttention


class TestBinaryAttentionCCT(unittest.TestCase):
    @staticmethod
    def make_dataset():
        images = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([0, 1])
        return DatasetSetup(
            DatasetType.cifar10,
            TensorDataset(images, labels),
            TensorDataset(images, labels),
        )

    def test_binary_attention_uses_ste_sign_and_has_bias(self):
        values = torch.tensor([-2.0, -0.25, 0.0, 0.25, 2.0], requires_grad=True)
        signed = BinaryAttention._ste_sign(values)
        torch.testing.assert_close(signed.detach(), torch.tensor([-1., -1., 1., 1., 1.]))
        signed.sum().backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values))

        setup = binary_attention_cct_7_3x1_cifar10(self.make_dataset())
        attention = setup.model.classifier.blocks[0].self_attn
        self.assertIsInstance(attention, BinaryAttention)
        self.assertEqual(attention.attention_bias.shape, (4, 256, 256))

    def test_cifar10_forward_backward_and_cct_recipe(self):
        binary_setup = binary_attention_cct_7_3x1_cifar10(self.make_dataset())
        regular_setup = cct_7_3x1_cifar10(self.make_dataset())
        images, _ = binary_setup.training_data[:]
        output = binary_setup.model(images)
        self.assertEqual(output.shape, (2, 10))
        output.sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in binary_setup.model.parameters()))
        self.assertEqual(binary_setup.model_type, ModelType.binary_attention_cct_7_3x1_32)

        for preset in (0, 1):
            binary_optimizer, _, binary_epochs = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(
                binary_setup, binary_setup.model, preset, override_steps_per_epoch=2,
            )
            regular_optimizer, _, regular_epochs = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(
                regular_setup, regular_setup.model, preset, override_steps_per_epoch=2,
            )
            self.assertIsInstance(binary_optimizer, torch.optim.AdamW)
            self.assertEqual(binary_epochs, regular_epochs)
            self.assertEqual(binary_epochs, 300)
            self.assertEqual(binary_optimizer.defaults["lr"], regular_optimizer.defaults["lr"])
            self.assertEqual(binary_optimizer.defaults["weight_decay"], regular_optimizer.defaults["weight_decay"])

    def test_generator_factory_name(self):
        with patch("py_src.ml_setup.cct.dataset_cifar10", return_value=self.make_dataset()):
            setup = get_ml_setup_from_config("binary_attention_cct_7_3x1_32", "cifar10")
        self.assertEqual(setup.model_type, ModelType.binary_attention_cct_7_3x1_32)


if __name__ == "__main__":
    unittest.main()
