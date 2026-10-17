from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset
from torchvision.models import vit_b_32
from torchvision.models.vision_transformer import VisionTransformer

from py_src.ml_setup import get_ml_setup_from_config
from py_src.ml_setup_dataset import DatasetSetup, DatasetType
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_model.vit_registers import VisionTransformerWithRegisters, vit_b_32_reg4
from py_src.model_opti_save_load import load_model_state_file, save_model_state


def tiny_vit(*, registers=True, num_classes=3):
    constructor = VisionTransformerWithRegisters if registers else VisionTransformer
    return constructor(
        image_size=8, patch_size=4, num_layers=1, num_heads=2,
        hidden_dim=8, mlp_dim=16, num_classes=num_classes,
    )


class TestViTRegisters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_real_b32_common_initialization_is_identical_and_only_registers_are_added(self):
        torch.manual_seed(123)
        baseline = vit_b_32(weights=None)
        torch.manual_seed(123)
        registered = vit_b_32_reg4()
        original = baseline.state_dict()
        actual = registered.state_dict()
        self.assertEqual(set(actual) - set(original), {"register_tokens"})
        for name, value in original.items():
            self.assertTrue(torch.equal(value, actual[name]), name)
        self.assertEqual(registered.register_tokens.shape, (1, 4, 768))
        self.assertEqual(actual["encoder.pos_embedding"].shape, (1, 50, 768))
        self.assertEqual(
            sum(p.numel() for p in registered.parameters()) - sum(p.numel() for p in baseline.parameters()),
            4 * 768,
        )

    def test_registers_have_no_position_embedding_and_classification_uses_cls(self):
        model = tiny_vit()
        model.encoder.layers = nn.Identity()
        model.encoder.ln = nn.Identity()
        with torch.no_grad():
            model.encoder.pos_embedding.fill_(2.0)
            model.class_token.fill_(3.0)
            model.register_tokens.fill_(7.0)
            model.heads.head.weight.normal_()
        features = model.forward_features(torch.zeros(2, 3, 8, 8))
        self.assertEqual(features["x_prenorm"].shape, (2, 9, 8))
        self.assertTrue(torch.equal(features["x_norm_regtokens"], torch.full((2, 4, 8), 7.0)))
        expected = model.heads(torch.full((2, 8), 5.0))
        self.assertTrue(torch.equal(model(torch.zeros(2, 3, 8, 8)), expected))
        self.assertEqual(features["x_norm_patchtokens"].shape, (2, 4, 8))

    def test_registers_receive_gradients_through_attention(self):
        model = tiny_vit()
        nn.init.normal_(model.heads.head.weight)
        loss = nn.CrossEntropyLoss()(model(torch.randn(2, 3, 8, 8)), torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.register_tokens.grad).all())
        self.assertGreater(model.register_tokens.grad.norm().item(), 0.0)

    def test_factory_and_checkpoint_roundtrip(self):
        dataset = TensorDataset(torch.zeros(2, 3, 8, 8), torch.tensor([0, 1]))
        setup = DatasetSetup(DatasetType.imagenet1k, dataset, dataset)
        with patch("py_src.ml_setup.vit.dataset_imagenet1k_from_pytorch", return_value=setup), \
             patch("py_src.ml_setup_model.vit_registers.vit_b_32_reg4", side_effect=lambda **kw: tiny_vit(**kw)):
            built = get_ml_setup_from_config("vit_b_32_reg4", "imagenet1k", preset=1)
            self.assertEqual(built.model_type, ModelType.vit_b_32_reg4)
            self.assertEqual(built.criterion.label_smoothing, 0.0)
            with tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory) / "registered.model.pt")
                save_model_state(path, built.model.state_dict(), built.model_type.name, "imagenet1k")
                state, model_name, dataset_name = load_model_state_file(path, map_location="cpu")
                restored = get_ml_setup_from_config(model_name, dataset_name, preset=1)
                restored.model.load_state_dict(state, strict=True)
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, restored.model.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
