"""Smoke tests for model setup factories with dummy datasets.

Each test trains for exactly one batch and, for classifiers, evaluates one
validation batch. All data is synthetic and in-memory.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from py_src.ml_setup import (
    convnext_tiny_imagenet1k,
    arithmetic_addition_grokking,
    cct14_7x2_imagenet1k,
    cct_7_3x1_cifar10,
    ddpm_cifar10,
    ddpm_flowers102,
    densenet121_cifar10,
    densenet121_imagenet1k,
    densenet_cifar_cifar10,
    dla46c_imagenet10,
    dla_cifar10,
    dla_cifar100,
    efficientnet_b0_cifar10,
    efficientnet_b0_cifar100,
    efficientnet_b1_imagenet1k,
    efficientnet_v2_s_imagenet1k,
    lenet4_mnist,
    lenet5_large_fc_mnist,
    lenet5_mnist,
    mnasnet0_5_imagenet1k,
    mnasnet1_0_imagenet1k,
    mobilenet_v2_cifar10,
    mobilenet_v2_cifar100,
    mobilenet_v3_large_imagenet1k,
    nanoclip_flickr30k_default,
    regnet_x_200mf_cifar10,
    regnet_x_200mf_cifar100,
    regnet_y_400mf_imagenet1k,
    resnet18_cifar10,
    resnet18_imagenet1k,
    resnet50_imagenet1k,
    resnext50_32x4d_imagenet1k,
    shufflenet_v2_cifar10,
    shufflenet_v2_cifar100,
    simplenet_cifar10,
    simplenet_cifar100,
    squeezenet1_1_imagenet1k,
    vit_b_32_imagenet1k,
    vgg11_bn_cifar10,
    vgg11_bn_imagenet1k,
    vgg11_no_bn_cifar10,
)

# ---------------------------------------------------------------------------
# Make sure the LLR2 project root is importable regardless of where the test
# runner is launched from.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from py_src.ml_setup.ml_setup import MLSetup
from py_src.complete_ml_setup import FastTrainingSetup
from py_src.ml_setup.dataloader_util import DataloaderConfig
from py_src.ml_setup_dataset import DatasetSetup, DatasetType
from tool.generate_high_accuracy_model import (
    _capture_rng_state,
    _prepare_log_csv_for_resume,
    _restore_rng_state,
    load_training_checkpoint,
    save_training_checkpoint,
)
from test.util import (
    DummyDatasetNanoCLIP,
    make_dummy_cifar10,
    make_dummy_cifar100,
    make_dummy_flowers102,
    make_dummy_imagenet10,
    make_dummy_imagenet1k,
    make_dummy_mnist,
    run_single_batch,
)


class _DummyNanoCLIPModel(L.LightningModule):
    def __init__(self, embed_dim: int = 16, image_size: int = 8, vocab_size: int = 100):
        super().__init__()
        self.img_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * image_size * image_size, embed_dim),
        )
        self.txt_encoder = nn.Embedding(vocab_size, embed_dim)
        self.latest_loss = 0.0
        self.latest_accuracy = 0.0

    def configure_optimizers(self) -> Any:
        return None, None

    def forward(self, images: torch.Tensor, captions: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_embedding = F.normalize(self.img_encoder(images), p=2, dim=-1)
        token_embeddings = self.txt_encoder(captions)
        masked_embeddings = token_embeddings * masks.unsqueeze(-1)
        text_embedding = masked_embeddings.sum(dim=1) / masks.sum(dim=1, keepdim=True).clamp_min(1)
        text_embedding = F.normalize(text_embedding, p=2, dim=-1)
        return image_embedding, text_embedding

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        images, captions, masks = batch
        image_embedding, text_embedding = self(images, captions, masks)
        logits = image_embedding @ text_embedding.T
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        return loss, accuracy

    def on_validation_epoch_start(self):
        self.latest_loss = 0.0
        self.latest_accuracy = 0.0

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        images, captions, masks = batch
        image_embedding, text_embedding = self(images, captions, masks)
        logits = image_embedding @ text_embedding.T
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        self.latest_loss = float(loss.item())
        self.latest_accuracy = float(accuracy.item())

    def on_validation_epoch_end(self):
        return None

    def get_validation_result(self):
        return self.latest_loss, int(self.latest_accuracy)


class _RandomSamplingModel(nn.Module):
    def sample(self, *, batch_size: int) -> torch.Tensor:
        return torch.rand(batch_size, 3, 8, 8)


class TestRunSingleBatch(unittest.TestCase):
    """Smoke-tests: one train batch + one val batch (where applicable)."""

    def _assert_classifier_single_batch(
        self,
        name: str,
        setup: MLSetup,
        *,
        batch_size: int = 2,
        use_cpu: bool = True,
    ) -> None:
        train_result, val_result = run_single_batch(
            setup,
            run_val=True,
            batch_size=batch_size,
            use_cpu=use_cpu,
        )
        self.assertGreater(train_result.iterations, 0, name)
        self.assertGreater(train_result.total_count, 0, name)
        self.assertIsNotNone(train_result.accuracy, name)
        self.assertIsNotNone(val_result, name)
        assert val_result is not None
        self.assertGreater(val_result.total_count, 0, name)
        print(name)
        print(f"train loss:{train_result.avg_loss:.4f} train accuracy:{train_result.accuracy:.4f}")
        print(f"val loss:{val_result.avg_loss:.4f}")

    def test_ddpm_cifar10_train_no_val(self):
        setup = ddpm_cifar10(override_dataset=make_dummy_cifar10(return_pil=False))
        train_result, val_result = run_single_batch(setup, run_val=False, batch_size=2)
        self.assertGreater(train_result.iterations, 0)
        self.assertIsNone(val_result)
        print("ddpm_cifar10")
        print(f"train loss:{train_result.avg_loss:.4f}")

    def test_ddpm_flowers102_disables_flash_attention_and_exposes_ema_hook(self):
        setup = ddpm_flowers102(
            override_dataset=make_dummy_flowers102(num_samples=1, return_pil=False)
        )

        self.assertTrue(hasattr(setup.model, "update_ema"))
        self.assertTrue(hasattr(setup.model, "train_diffusion"))
        self.assertTrue(hasattr(setup.model, "ema_diffusion"))
        self.assertEqual(setup.gradient_accumulate_every, 2)
        self.assertEqual(setup.max_grad_norm, 1.0)

        attention_modules = [
            module for module in setup.model.train_diffusion.model.modules()
            if hasattr(module, "attend") and hasattr(module.attend, "flash")
        ]
        self.assertGreater(len(attention_modules), 0)
        self.assertTrue(all(module.attend.flash is False for module in attention_modules))

    def test_ddpm_flowers102_train_no_val(self):
        setup = ddpm_flowers102(override_dataset=make_dummy_flowers102(num_samples=1, return_pil=False))
        train_result, val_result = run_single_batch(
            setup,
            run_val=False,
            batch_size=1,
            use_cpu=False,
        )
        self.assertGreater(train_result.iterations, 0)
        self.assertGreater(train_result.total_count, 0)
        self.assertIsNone(val_result)
        self.assertTrue(hasattr(setup.model, "ema"))
        self.assertEqual(int(setup.model.ema.step.item()), 1)
        self.assertTrue(bool(setup.model.ema.initted.item()))
        print("ddpm_flowers102")
        print(f"train loss:{train_result.avg_loss:.4f}")

    def test_ddpm_flowers102_sample_hook_uses_fixed_seed_and_saves_ten_images(self):
        from py_src.ml_setup.ddpm_flowers import _generate_sample_from_zero_to_one

        model = _RandomSamplingModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            _generate_sample_from_zero_to_one(model, tmpdir, 0, torch.device("cpu"), 10)
            _generate_sample_from_zero_to_one(model, tmpdir, 10, torch.device("cpu"), 10)

            for epoch in (0, 10):
                for image_index in range(10):
                    image_path = os.path.join(tmpdir, f"epoch{epoch}_{image_index}.png")
                    self.assertTrue(os.path.exists(image_path))

            for image_index in range(10):
                first_path = os.path.join(tmpdir, f"epoch0_{image_index}.png")
                second_path = os.path.join(tmpdir, f"epoch10_{image_index}.png")
                with open(first_path, "rb") as first_file, open(second_path, "rb") as second_file:
                    self.assertEqual(first_file.read(), second_file.read())

    def test_ddpm_flowers102_sample_hook_allows_seed_override(self):
        from py_src.ml_setup.ddpm_flowers import _generate_sample_from_zero_to_one

        model = _RandomSamplingModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            _generate_sample_from_zero_to_one(model, tmpdir, 0, torch.device("cpu"), 1, 11)
            _generate_sample_from_zero_to_one(model, tmpdir, 1, torch.device("cpu"), 1, 11)
            _generate_sample_from_zero_to_one(model, tmpdir, 2, torch.device("cpu"), 1, 12)

            with open(os.path.join(tmpdir, "epoch0_0.png"), "rb") as first_file:
                first_bytes = first_file.read()
            with open(os.path.join(tmpdir, "epoch1_0.png"), "rb") as second_file:
                second_bytes = second_file.read()
            with open(os.path.join(tmpdir, "epoch2_0.png"), "rb") as third_file:
                third_bytes = third_file.read()

            self.assertEqual(first_bytes, second_bytes)
            self.assertNotEqual(first_bytes, third_bytes)

    def test_arithmetic_addition_grokking_train_and_val(self):
        self._assert_classifier_single_batch(
            "arithmetic_addition_grokking",
            arithmetic_addition_grokking(train_percentage=50, modulus=13),
            batch_size=8,
        )

    def test_resnet18_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "resnet18_cifar10",
            resnet18_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=4,
        )

    def test_resnet18_imagenet1k_train_and_val_p1(self):
        self._assert_classifier_single_batch(
            "resnet18_imagenet1k_p1",
            resnet18_imagenet1k(1, override_dataset=make_dummy_imagenet1k(return_pil=False)),
            batch_size=2,
            use_cpu=False,
        )

    def test_resnet18_imagenet1k_train_and_val_p2(self):
        self._assert_classifier_single_batch(
            "resnet18_imagenet1k_p2",
            resnet18_imagenet1k(2, override_dataset=make_dummy_imagenet1k(return_pil=False)),
            batch_size=2,
            use_cpu=False,
        )

    def test_resnet50_imagenet1k_train_and_val_p2(self):
        self._assert_classifier_single_batch(
            "resnet50_imagenet1k_p2",
            resnet50_imagenet1k(2, override_dataset=make_dummy_imagenet1k(return_pil=False)),
            batch_size=1,
            use_cpu=False,
        )

    def test_resnet50_imagenet1k_p2_scheduler_uses_actual_dataloader_length(self):
        setup = resnet50_imagenet1k(
            2,
            override_dataset=make_dummy_imagenet1k(num_samples=300, return_pil=False),
        )
        train_loader = setup.train_dataloader(
            DataloaderConfig(num_workers=0),
            ignore_override=True,
        )
        actual_steps_per_epoch = len(train_loader) # type: ignore
        nominal_steps_per_epoch = len(setup.training_data) // setup.default_batch_size + 1

        self.assertGreater(actual_steps_per_epoch, nominal_steps_per_epoch)

        _, scheduler, epochs = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(
            setup,
            setup.model,
            preset=2,
            override_steps_per_epoch=actual_steps_per_epoch,
        )

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler is not None
        self.assertEqual(scheduler.T_max, epochs * actual_steps_per_epoch)

    def test_cct_7_3x1_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "cct_7_3x1_cifar10",
            cct_7_3x1_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
            use_cpu=False,
        )

    def test_cct14_7x2_imagenet1k_train_and_val(self):
        self._assert_classifier_single_batch(
            "cct14_7x2_imagenet1k",
            cct14_7x2_imagenet1k(override_dataset=make_dummy_imagenet1k(return_pil=False)),
            batch_size=2,
            use_cpu=False,
        )

    def test_densenet121_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "densenet121_cifar10",
            densenet121_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_densenet_cifar_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "densenet_cifar_cifar10",
            densenet_cifar_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_dla_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "dla_cifar10",
            dla_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_dla_cifar100_train_and_val(self):
        self._assert_classifier_single_batch(
            "dla_cifar100",
            dla_cifar100(override_dataset=make_dummy_cifar100(return_pil=False)),
            batch_size=2,
        )

    def test_dla46c_imagenet10_train_and_val(self):
        self._assert_classifier_single_batch(
            "dla46c_imagenet10",
            dla46c_imagenet10(override_dataset=make_dummy_imagenet10(return_pil=False)),
            batch_size=2,
            use_cpu=False,
        )

    def test_efficientnet_b0_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "efficientnet_b0_cifar10",
            efficientnet_b0_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_efficientnet_b0_cifar100_train_and_val(self):
        self._assert_classifier_single_batch(
            "efficientnet_b0_cifar100",
            efficientnet_b0_cifar100(override_dataset=make_dummy_cifar100(return_pil=False)),
            batch_size=2,
        )

    def test_efficientnet_v2_s_imagenet1k_train_and_val(self):
        self._assert_classifier_single_batch(
            "efficientnet_v2_s_imagenet1k",
            efficientnet_v2_s_imagenet1k(override_dataset=make_dummy_imagenet1k(return_pil=False)),
            batch_size=1,
            use_cpu=False,
        )

    def test_efficientnet_b1_imagenet1k_train_and_val(self):
        self._assert_classifier_single_batch(
            "efficientnet_b1_imagenet1k",
            efficientnet_b1_imagenet1k(override_dataset=make_dummy_imagenet1k(return_pil=False)),
            batch_size=1,
            use_cpu=False,
        )

    def test_lenet5_mnist_train_and_val(self):
        self._assert_classifier_single_batch(
            "lenet5_mnist",
            lenet5_mnist(override_dataset=make_dummy_mnist(return_pil=False)),
            batch_size=4,
        )

    def test_lenet4_mnist_train_and_val(self):
        self._assert_classifier_single_batch(
            "lenet4_mnist",
            lenet4_mnist(override_dataset=make_dummy_mnist(return_pil=False)),
            batch_size=4,
        )

    def test_lenet5_large_fc_mnist_train_and_val(self):
        self._assert_classifier_single_batch(
            "lenet5_large_fc_mnist",
            lenet5_large_fc_mnist(override_dataset=make_dummy_mnist(return_pil=False)),
            batch_size=4,
        )

    def test_mobilenet_v2_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "mobilenet_v2_cifar10",
            mobilenet_v2_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_mobilenet_v2_cifar100_train_and_val(self):
        self._assert_classifier_single_batch(
            "mobilenet_v2_cifar100",
            mobilenet_v2_cifar100(override_dataset=make_dummy_cifar100(return_pil=False)),
            batch_size=2,
        )

    def test_mobilenet_v3_large_imagenet1k_train_and_val(self):
        self._assert_classifier_single_batch(
            "mobilenet_v3_large_imagenet1k",
            mobilenet_v3_large_imagenet1k(2, override_dataset=make_dummy_imagenet1k(return_pil=False)),
            batch_size=2,
            use_cpu=False,
        )

    def test_regnet_x_200mf_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "regnet_x_200mf_cifar10",
            regnet_x_200mf_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_regnet_x_200mf_cifar100_train_and_val(self):
        self._assert_classifier_single_batch(
            "regnet_x_200mf_cifar100",
            regnet_x_200mf_cifar100(override_dataset=make_dummy_cifar100(return_pil=False)),
            batch_size=2,
        )

    def test_shufflenet_v2_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "shufflenet_v2_cifar10",
            shufflenet_v2_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_shufflenet_v2_cifar100_train_and_val(self):
        self._assert_classifier_single_batch(
            "shufflenet_v2_cifar100",
            shufflenet_v2_cifar100(override_dataset=make_dummy_cifar100(return_pil=False)),
            batch_size=2,
        )

    def test_simplenet_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "simplenet_cifar10",
            simplenet_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_simplenet_cifar100_train_and_val(self):
        self._assert_classifier_single_batch(
            "simplenet_cifar100",
            simplenet_cifar100(override_dataset=make_dummy_cifar100(return_pil=False)),
            batch_size=2,
        )

    def test_vgg11_bn_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "vgg11_bn_cifar10",
            vgg11_bn_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_vgg11_no_bn_cifar10_train_and_val(self):
        resize_224 = [transforms.Resize((224, 224))]
        self._assert_classifier_single_batch(
            "vgg11_no_bn_cifar10",
            vgg11_no_bn_cifar10(
                override_dataset=make_dummy_cifar10(
                    return_pil=False,
                    transforms_training=resize_224,
                    transforms_testing=resize_224,
                )
            ),
            batch_size=1,
        )

    def test_ported_imagenet_setups_build(self):
        dummy_imagenet = make_dummy_imagenet1k(return_pil=False)
        builders = [
            ("vgg11_bn_imagenet1k", lambda: vgg11_bn_imagenet1k(override_dataset=dummy_imagenet)),
            ("densenet121_imagenet1k", lambda: densenet121_imagenet1k(override_dataset=dummy_imagenet)),
            ("regnet_y_400mf_imagenet1k", lambda: regnet_y_400mf_imagenet1k(override_dataset=dummy_imagenet)),
            ("vit_b_32_imagenet1k", lambda: vit_b_32_imagenet1k(override_dataset=dummy_imagenet)),
            ("squeezenet1_1_imagenet1k", lambda: squeezenet1_1_imagenet1k(override_dataset=dummy_imagenet)),
            ("resnext50_32x4d_imagenet1k", lambda: resnext50_32x4d_imagenet1k(override_dataset=dummy_imagenet)),
            ("mnasnet0_5_imagenet1k", lambda: mnasnet0_5_imagenet1k(override_dataset=dummy_imagenet)),
            ("mnasnet1_0_imagenet1k", lambda: mnasnet1_0_imagenet1k(override_dataset=dummy_imagenet)),
            ("convnext_tiny_imagenet1k", lambda: convnext_tiny_imagenet1k(override_dataset=dummy_imagenet)),
        ]

        for name, build in builders:
            with self.subTest(name=name):
                setup = build()
                self.assertEqual(setup.dataset_type, DatasetType.imagenet1k)
                self.assertIsNotNone(setup.model)
                self.assertIsNotNone(setup.adapter)
                self.assertIsNotNone(setup.criterion)

    @patch("py_src.ml_setup.nanoclip.AutoTokenizer.from_pretrained", return_value=object())
    @patch("py_src.ml_setup.nanoclip.NanoCLIP", side_effect=lambda *args, **kwargs: _DummyNanoCLIPModel())
    def test_nanoclip_flickr30k_train_no_val(self, _mock_model_ctor, _mock_tokenizer):
        dummy_dataset = DatasetSetup(
            DatasetType.flickr30k,
            DummyDatasetNanoCLIP(n=8, img_size=8, seq_len=8),
            DummyDatasetNanoCLIP(n=8, img_size=8, seq_len=8),
        )
        setup = nanoclip_flickr30k_default(override_dataset=dummy_dataset)
        setup.default_collate_fn = None
        setup.default_collate_fn_val = None

        train_result, val_result = run_single_batch(setup, run_val=True, batch_size=4)
        self.assertGreater(train_result.iterations, 0)
        self.assertGreater(train_result.total_count, 0)
        self.assertIsNotNone(train_result.accuracy)
        self.assertIsNone(val_result)
        print("nanoclip_flickr30k_default")
        print(f"train loss:{train_result.avg_loss:.4f} train accuracy:{train_result.accuracy:.4f}")


class TestGenerateHighAccuracyModelCheckpointing(unittest.TestCase):
    def test_training_checkpoint_round_trip_preserves_training_state(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        batch = torch.randn(3, 4)
        loss = model(batch).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        run_config = {
            "model_type": "dummy_model",
            "dataset_type": "dummy_dataset",
            "cpu": True,
            "dali": False,
            "dali_device_id": 0,
            "save_format": "file",
            "save_interval": 1,
            "amp": False,
            "compile": False,
            "random_seed": 123,
            "preset": 1,
            "epoch_override": 4,
            "transfer_learn": None,
            "initial_model": None,
            "opposite_init_model": None,
            "disable_reinit": False,
            "enable_eval": False,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "training_checkpoint_epoch0.pt")
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                scaler=None,
                completed_epoch=0,
                total_epochs=4,
                index=3,
                number_of_models=10,
                output_folder=tmpdir,
                run_config=run_config,
            )

            checkpoint = load_training_checkpoint(checkpoint_path)

        self.assertEqual(checkpoint["checkpoint_type"], "generate_high_accuracy_model")
        self.assertEqual(checkpoint["completed_epoch"], 0)
        self.assertEqual(checkpoint["next_epoch"], 1)
        self.assertEqual(checkpoint["total_epochs"], 4)
        self.assertEqual(checkpoint["index"], 3)
        self.assertEqual(checkpoint["number_of_models"], 10)
        self.assertEqual(checkpoint["run_config"]["model_type"], "dummy_model")
        self.assertIsNotNone(checkpoint["optimizer_state_dict"])
        self.assertIsNotNone(checkpoint["lr_scheduler_state_dict"])
        self.assertIsNone(checkpoint["scaler_state_dict"])

        restored_model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        restored_model.load_state_dict(checkpoint["state_dict"])
        for expected, actual in zip(model.parameters(), restored_model.parameters()):
            self.assertTrue(torch.allclose(expected, actual))

    def test_prepare_log_csv_for_resume_truncates_rows_after_checkpoint_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_csv_path = os.path.join(tmpdir, "00.log.csv")
            with open(log_csv_path, "w", encoding="utf-8") as log_file:
                log_file.write("epoch,training_loss,training_accuracy,validation_loss,validation_accuracy,lrs\n")
                log_file.write("0,1.0,0.1,nan,nan,[0.1]\n")
                log_file.write("1,0.8,0.2,nan,nan,[0.1]\n")
                log_file.write("2,0.6,0.3,nan,nan,[0.1]\n")

            _prepare_log_csv_for_resume(log_csv_path, completed_epoch=1)

            with open(log_csv_path, "r", encoding="utf-8") as log_file:
                lines = log_file.readlines()

        self.assertEqual(
            lines,
            [
                "epoch,training_loss,training_accuracy,validation_loss,validation_accuracy,lrs\n",
                "0,1.0,0.1,nan,nan,[0.1]\n",
                "1,0.8,0.2,nan,nan,[0.1]\n",
            ],
        )

    def test_rng_state_round_trip_replays_same_samples(self):
        logger = logging.getLogger("test_generate_high_accuracy_model_checkpointing")
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)

        rng_state = _capture_rng_state()

        python_first = random.random()
        numpy_first = np.random.rand(3)
        torch_first = torch.rand(3)

        random.random()
        np.random.rand(3)
        torch.rand(3)

        _restore_rng_state(rng_state, logger)

        python_second = random.random()
        numpy_second = np.random.rand(3)
        torch_second = torch.rand(3)

        self.assertEqual(python_first, python_second)
        self.assertTrue(np.allclose(numpy_first, numpy_second))
        self.assertTrue(torch.allclose(torch_first, torch_second))


if __name__ == "__main__":
    unittest.main()
