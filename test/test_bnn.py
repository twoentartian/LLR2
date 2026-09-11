"""BinaryNet recipe, STE, optimizer and generator checkpoint regressions."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from py_src.complete_ml_setup import FastTrainingSetup
from py_src.ml_setup import bnn_cifar10, bnn_floating_cifar10, get_ml_setup_from_config
from py_src.ml_setup.dataloader_util import DataloaderConfig
from py_src.ml_setup_dataset import DatasetSetup, DatasetType
from py_src.ml_setup_dataset.dataset_cifar import dataset_cifar10_bnn
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_model.bnn import (
    BinaryAdam,
    BinaryConv2d,
    BinaryLinear,
    FloatingConv2d,
    FloatingLinear,
    VGGNet7Binary,
    VGGNet7Floating,
    binarize,
)
from py_src.util import re_initialize_model
from tool.generate_high_accuracy_model import load_training_checkpoint, training_model


class TestBNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make_setup(self):
        data = TensorDataset(torch.randn(4, 3, 32, 32), torch.arange(4))
        return bnn_cifar10(DatasetSetup(DatasetType.cifar10, data, data))

    def test_binary_forward_and_gradient(self):
        x = torch.tensor([-2., -1., -0.2, 0., 0.2, 1., 2.], requires_grad=True)
        y = binarize(F.hardtanh(x))
        torch.testing.assert_close(y, torch.tensor([-1., -1., -1., -1., 1., 1., 1.]))
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([0., 0., 1., 1., 1., 0., 0.]))

        layer = BinaryLinear(3, 2, bias=False)
        original = layer.weight.detach().clone()
        inputs = torch.randn(4, 3, requires_grad=True)
        output = layer(inputs)
        expected_weight = original.add(1).div(2).clamp(0, 1).round().mul(2).sub(1)
        torch.testing.assert_close(output, F.linear(inputs, expected_weight))
        output.sum().backward()
        torch.testing.assert_close(layer.weight.grad, inputs.detach().sum(0).expand(2, -1))
        torch.testing.assert_close(inputs.grad, expected_weight.sum(0).expand(4, -1))
        torch.testing.assert_close(layer.weight, original)

    def test_adam_clips_only_binary_weights(self):
        model = nn.Sequential(BinaryLinear(2, 2, bias=False), nn.BatchNorm1d(2))
        optimizer = BinaryAdam(model, lr=0.001)
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(2.)
                param.grad = torch.ones_like(param)
        optimizer.step()
        self.assertEqual(model[0].weight.max().item(), 1.)
        self.assertGreater(model[1].weight.min().item(), 1.)
        self.assertGreater(model[1].bias.min().item(), 1.)

    def test_factory_and_recipe(self):
        setup = self.make_setup()
        with patch("py_src.ml_setup.bnn.bnn_cifar10", return_value=setup):
            self.assertIs(get_ml_setup_from_config("bnn"), setup)
            self.assertIs(get_ml_setup_from_config("bnn", "cifar10"), setup)
            with self.assertRaises(NotImplementedError):
                get_ml_setup_from_config("bnn", "mnist")
        self.assertEqual(setup.default_batch_size, 50)
        self.assertEqual(setup.val_dataloader().batch_size, setup.default_batch_size)
        self.assertEqual(setup.val_dataloader(DataloaderConfig(batch_size=2)).batch_size, 2)
        self.assertIsInstance(setup.criterion, nn.CrossEntropyLoss)
        for preset in (0, 1):
            opt, scheduler, epochs = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(
                setup, BinaryLinear(2, 2, bias=False), preset, override_steps_per_epoch=2,
            )
            self.assertEqual(epochs, 500)
            self.assertIsInstance(opt, BinaryAdam)
            self.assertEqual(opt.defaults["betas"], (0.9, 0.999))
            self.assertEqual(opt.defaults["weight_decay"], 0)
            self.assertEqual(opt.defaults["eps"], 1e-8)
            # Check the LR used for EVERY batch, especially milestone boundaries.
            for epoch in range(1, 501):
                expected = 0.001 * 0.1 ** sum(epoch >= m for m in (100, 200, 300, 400))
                for _ in range(2):
                    self.assertAlmostEqual(opt.param_groups[0]["lr"], expected)
                    opt.step()
                    scheduler.step()
        with self.assertRaises(NotImplementedError):
            FastTrainingSetup.get_optimizer_lr_scheduler_epoch(setup, setup.model, 2)
        _, scheduler, _ = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(
            setup, BinaryLinear(2, 2, bias=False), override_dataset=range(50000),
        )
        self.assertEqual(sorted(scheduler.milestones), [99000, 199000, 299000, 399000])

    def test_floating_model_factory_and_weights(self):
        data = TensorDataset(torch.randn(4, 3, 32, 32), torch.arange(4))
        setup = bnn_floating_cifar10(DatasetSetup(DatasetType.cifar10, data, data))
        self.assertIsInstance(setup.model, VGGNet7Floating)
        self.assertEqual(setup.model_type, ModelType.bnn_floating)
        self.assertEqual(setup.default_batch_size, 50)
        self.assertEqual(len(setup.training_data), 4)
        self.assertTrue(all(isinstance(module, FloatingConv2d | FloatingLinear)
                            for module in setup.model.modules()
                            if isinstance(module, (FloatingConv2d, FloatingLinear))))
        self.assertFalse(any(isinstance(module, (BinaryConv2d, BinaryLinear))
                             for module in setup.model.modules()))
        self.assertTrue(all(not torch.all((parameter == -1) | (parameter == 1))
                            for name, parameter in setup.model.named_parameters()
                            if name.startswith(("conv", "fc"))))
        optimizer, scheduler, epochs = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(
            setup, setup.model, override_steps_per_epoch=2,
        )
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertNotIsInstance(optimizer, BinaryAdam)
        self.assertEqual(epochs, 500)
        self.assertEqual(sorted(scheduler.milestones), [198, 398, 598, 798])

    def test_dataset_uses_full_training_set_and_upstream_augmentation(self):
        train_data, test_data = range(50000), range(10000)
        with patch("py_src.ml_setup_dataset.dataset_cifar.datasets.CIFAR10",
                   side_effect=[train_data, test_data]) as cifar:
            setup = dataset_cifar10_bnn()
        self.assertIs(setup.train_data, train_data)
        self.assertIs(setup.valdation_data, test_data)
        self.assertEqual(len(setup.train_data), 50000)
        self.assertEqual(len(setup.valdation_data), 10000)
        train_call, test_call = cifar.call_args_list
        self.assertTrue(train_call.kwargs["train"])
        self.assertFalse(test_call.kwargs["train"])
        transforms = train_call.kwargs["transform"].transforms
        self.assertEqual([type(t).__name__ for t in transforms],
                         ["RandomHorizontalFlip", "RandomCrop", "ToTensor", "Normalize"])
        self.assertEqual(transforms[1].padding_mode, "edge")
        self.assertEqual(transforms[1].padding, 4)
        self.assertEqual(transforms[3].mean, (0.5, 0.5, 0.5))
        self.assertEqual(transforms[3].std, (0.5, 0.5, 0.5))
        test_transforms = test_call.kwargs["transform"].transforms
        self.assertEqual([type(t).__name__ for t in test_transforms], ["ToTensor", "Normalize"])
        self.assertEqual(test_transforms[1].mean, (0.5, 0.5, 0.5))
        self.assertEqual(test_transforms[1].std, (0.5, 0.5, 0.5))

    def test_reinitialization_retains_xavier_and_topology(self):
        model = VGGNet7Binary()
        with torch.no_grad():
            model.fc1.weight.fill_(1.)
        re_initialize_model(model)
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(model.fc1.weight)
        self.assertAlmostEqual(model.fc1.weight.std().item(), (2 / (fan_in + fan_out)) ** 0.5, places=4)
        self.assertEqual(sum(isinstance(m, BinaryConv2d) for m in model.modules()), 6)
        self.assertEqual(sum(isinstance(m, BinaryLinear) for m in model.modules()), 3)
        self.assertFalse(model.bn9.affine)
        self.assertTrue(all(m.bias is None for m in model.modules()
                            if isinstance(m, (BinaryConv2d, BinaryLinear))))

    def test_generator_training_validation_and_exact_resume(self):
        torch.manual_seed(1)
        setup = self.make_setup()
        config = {"model_type": "bnn", "dataset_type": "cifar10"}
        with tempfile.TemporaryDirectory() as folder:
            kwargs = {
                "output_folder": folder, "index": 0, "number_of_models": 1, "arg_ml_setup": setup,
                "arg_use_cpu": True, "random_seed": 1, "arg_worker_count": 1, "arg_total_cpu_count": 2,
                "arg_save_format": "none", "arg_save_interval": 1, "arg_amp": False, "arg_compile": False,
                "arg_preset": 1, "arg_epoch_override": 1, "transfer_learn_model_path": None,
                "init_model_path": None, "opposite_init_model_path": None, "disable_reinit": False,
                "enable_validation": True, "run_config": config, "load_checkpoint_path": None,
            }
            training_model(**kwargs)
            path = str(Path(folder) / "0/training_checkpoint.pt")
            first = load_training_checkpoint(path)
            self.assertEqual(first["next_epoch"], 1)
            self.assertTrue((Path(folder) / "0.model.pt").is_file())
            self.assertTrue((Path(folder) / "0.optimizer.pt").is_file())
            self.assertFalse(list(Path(folder).glob("*.best_training_*.model.pt")))
            latent = first["state_dict"]["conv1.weight"]
            self.assertTrue(((latent != -1) & (latent != 1)).any())

            # Eval must not replace the latent parameters or invalidate later training.
            setup.model.load_state_dict(first["state_dict"])
            setup.model.eval()
            inputs, _labels = setup.training_data.tensors
            with torch.inference_mode():
                output = setup.model(inputs)
            restored = copy.deepcopy(setup.model)
            with torch.inference_mode():
                torch.testing.assert_close(output, restored(inputs), rtol=0, atol=0)
            torch.testing.assert_close(setup.model.conv1.weight, latent, rtol=0, atol=0)

            # Continue twice from the same on-disk checkpoint, including RNG and Adam.
            kwargs.update(arg_epoch_override=2, load_checkpoint_path=path, run_config=first["run_config"])
            training_model(**kwargs)
            second = load_training_checkpoint(path)
            torch.save(first, path)
            training_model(**kwargs)
            repeated = load_training_checkpoint(path)
            self.assertEqual(second["next_epoch"], 2)
            self.assertEqual(second["lr_scheduler_state_dict"]["last_epoch"], 2)
            for key, value in second["state_dict"].items():
                torch.testing.assert_close(value, repeated["state_dict"][key], rtol=0, atol=0)
            self.assertNotEqual(first["state_dict"]["conv1.weight"].tolist(),
                                second["state_dict"]["conv1.weight"].tolist())


if __name__ == "__main__":
    unittest.main()
