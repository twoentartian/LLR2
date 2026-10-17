from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import vit_b_32
from torchvision.models.vision_transformer import VisionTransformer

from py_src.complete_ml_setup import FastTrainingSetup
from py_src.ml_setup import get_ml_setup_from_config
from py_src.ml_setup_dataset import DatasetSetup, DatasetType
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_model.vit_rope import RotarySelfAttention, VisionTransformerWithRoPE, vit_b_32_rope
from py_src.ml_setup_model.vit_swiglu import SwiGLUBlock, VisionTransformerWithSwiGLU, vit_b_32_swiglu
from py_src.model_opti_save_load import load_model_state_file, save_model_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import misalignment_measurement_2 as measurement


VARIANTS = (
    ("vit_b_32_rope", VisionTransformerWithRoPE, vit_b_32_rope),
    ("vit_b_32_swiglu", VisionTransformerWithSwiGLU, vit_b_32_swiglu),
)


def tiny_variant(constructor, *, num_classes=3):
    return constructor(
        image_size=8, patch_size=4, num_layers=2, num_heads=2,
        hidden_dim=8, mlp_dim=24, num_classes=num_classes,
    )


def complex_rope_reference(module, tokens):
    """Independent complex-number formulation used by the upstream RoPE code."""
    batch, length, width = tokens.shape
    q, k, v = F.linear(tokens, module.in_proj_weight, module.in_proj_bias).reshape(
        batch, length, 3, module.num_heads, module.head_dim,
    ).permute(2, 0, 3, 1, 4).unbind(0)
    x, y = module.rope_coordinates.unbind(-1)
    fx, fy = module.rope_frequencies.unbind(0)
    phase = x[None, :, None] * fx[:, None, :] + y[None, :, None] * fy[:, None, :]
    rotation = torch.polar(torch.ones_like(phase), phase)

    def rotate(tensor):
        pairs = torch.view_as_complex(tensor[:, :, 1:].float().reshape(batch, module.num_heads, length - 1, -1, 2))
        rotated = torch.view_as_real(pairs * rotation).flatten(-2).to(tensor.dtype)
        return torch.cat((tensor[:, :, :1], rotated), dim=2)

    # Explicit softmax also checks SDPA's scale and head/token layout.
    probabilities = (rotate(q) @ rotate(k).transpose(-1, -2) / math.sqrt(module.head_dim)).softmax(-1)
    attended = (probabilities @ v).transpose(1, 2).reshape(batch, length, width)
    return module.out_proj(attended)


class TestViTVariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_b32_architectures_preserve_common_initialization_and_match_parameter_budget(self):
        torch.manual_seed(123)
        baseline = vit_b_32(weights=None)
        original = baseline.state_dict()
        count = sum(p.numel() for p in baseline.parameters())
        for name, _, builder in VARIANTS:
            with self.subTest(model=name):
                torch.manual_seed(123)
                model = builder()
                actual = model.state_dict()
                for key in original.keys() & actual.keys():
                    self.assertTrue(torch.equal(original[key], actual[key]), key)
                self.assertEqual(model.seq_length, 50)
                self.assertEqual(model.heads.head.weight.shape, (1000, 768))
                self.assertEqual(measurement._resolve_flatness_layers(model, "last_matrix_weight"), ["heads.head.weight"])
                if name.endswith("rope"):
                    self.assertEqual(original.keys() - actual.keys(), {"encoder.pos_embedding"})
                    self.assertEqual(len(actual.keys() - original.keys()), 12)
                    self.assertEqual(sum(p.numel() for p in model.parameters()) - count, 12 * 2 * 12 * 32 - 50 * 768)
                    for block in model.encoder.layers:
                        self.assertEqual(block.self_attention.rope_frequencies.shape, (2, 12, 32))
                else:
                    self.assertEqual(model.swiglu_hidden_dim, 2048)
                    self.assertEqual(sum(p.numel() for p in model.parameters()) - count, 12 * 1024)
                del model, actual

    def test_rope_matches_complex_reference_outputs_and_frequency_gradients(self):
        torch.manual_seed(14)
        attention = RotarySelfAttention(nn.MultiheadAttention(16, 2, batch_first=True), grid_size=2, theta=10.)
        tokens = torch.randn(3, 5, 16, requires_grad=True)
        actual, _ = attention(tokens, tokens, tokens)
        expected = complex_rope_reference(attention, tokens)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        inputs = (tokens, attention.rope_frequencies, attention.in_proj_weight)
        actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
        expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=2e-5)
        self.assertGreater(actual_grads[1].norm().item(), 0.)

    def test_zero_frequencies_recover_vanilla_attention_and_cls_is_not_rotated(self):
        torch.manual_seed(15)
        baseline = nn.MultiheadAttention(16, 2, batch_first=True).eval()
        attention = RotarySelfAttention(baseline, grid_size=2, theta=10.).eval()
        tokens = torch.randn(2, 5, 16)
        with torch.no_grad():
            attention.rope_frequencies.zero_()
        actual, _ = attention(tokens, tokens, tokens)
        expected, _ = baseline(tokens, tokens, tokens, need_weights=False)
        torch.testing.assert_close(actual, expected)
        # Check CLS preservation and norm preservation independently of attention.
        from py_src.ml_setup_model.vit_rope import _rotate_patch_tokens

        projected = torch.randn(2, 2, 5, 8)
        phase = torch.randn(1, 2, 4, 4)
        rotated = _rotate_patch_tokens(projected, phase.cos(), phase.sin())
        self.assertTrue(torch.equal(rotated[:, :, 0], projected[:, :, 0]))
        torch.testing.assert_close(rotated.norm(dim=-1), projected.norm(dim=-1))

    def test_swiglu_uses_silu_gate_and_both_projections_receive_gradients(self):
        block = SwiGLUBlock(8, 16)
        tokens = torch.randn(2, 5, 8, requires_grad=True)
        gate_weight, value_weight = block.w12.weight.chunk(2)
        gate_bias, value_bias = block.w12.bias.chunk(2)
        gate = F.linear(tokens, gate_weight, gate_bias)
        value = F.linear(tokens, value_weight, value_bias)
        expected = F.linear((gate * gate.sigmoid()) * value, block.w3.weight, block.w3.bias)
        torch.testing.assert_close(block(tokens), expected)
        block(tokens).square().sum().backward()
        for parameter in block.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(), 0.)

    def test_factory_recipe_dali_override_and_checkpoint_roundtrip(self):
        dataset = TensorDataset(torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 2, 0]))
        setup = DatasetSetup(DatasetType.imagenet1k, dataset, dataset)
        for name, constructor, _ in VARIANTS:
            for dali in (False, True):
                with self.subTest(model=name, dali=dali), \
                     patch("py_src.ml_setup.vit.dataset_imagenet1k_from_pytorch", return_value=setup), \
                     patch("py_src.ml_setup_dataset.dataset_imagenet1k_from_pytorch_dali", return_value=setup) as dali_builder, \
                     patch(f"py_src.ml_setup_model.{'vit_rope' if name.endswith('rope') else 'vit_swiglu'}.{name}",
                           side_effect=lambda _constructor=constructor, **kw: tiny_variant(_constructor, **kw)):
                    built = get_ml_setup_from_config(name, "imagenet1k", preset=1, use_dali=dali)
                    self.assertEqual(built.model_type, ModelType[name])
                    self.assertEqual(built.criterion.label_smoothing, 0.0)
                    self.assertEqual(built.default_batch_size, 512)
                    self.assertEqual(dali_builder.call_count, int(dali))
                    optimizer, scheduler, epochs = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(built, built.model)
                    self.assertIsInstance(optimizer, torch.optim.AdamW)
                    self.assertIsInstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
                    self.assertEqual(epochs, 300)
                    nn.init.normal_(built.model.heads.head.weight, std=0.1)
                    built.model.eval()
                    with tempfile.TemporaryDirectory() as directory:
                        path = str(Path(directory) / "variant.model.pt")
                        save_model_state(path, built.model.state_dict(), name, "imagenet1k")
                        state, model_name, dataset_name = load_model_state_file(path, map_location="cpu")
                        restored = get_ml_setup_from_config(model_name, dataset_name, preset=1)
                        restored.model.load_state_dict(state, strict=True)
                        restored.model.eval()
                        torch.testing.assert_close(restored.model(dataset.tensors[0]), built.model(dataset.tensors[0]))

    def test_training_norm_capture_and_hessian_measurement_use_existing_pipeline(self):
        for name, constructor, _ in VARIANTS:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory:
                model = tiny_variant(constructor)
                dataset = TensorDataset(torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 2, 0]))
                loader = DataLoader(dataset, batch_size=2)
                bundle = measurement.DatasetBundle(
                    partition_a=dataset, partition_b=dataset,
                    ml_setup=SimpleNamespace(model_type=ModelType[name], dataset_type=DatasetType.imagenet1k),
                    dataset_name="imagenet1k", dataset_type_name="imagenet1k", source_examples=8,
                    permutation=torch.arange(8), excluded_indices=torch.empty(0, dtype=torch.long), modular=False,
                )
                args = SimpleNamespace(
                    random_seed=123, split_seed=456, optimizer_preset=0, learning_rate=0.001,
                    weight_decay=0.0, optimizer="adamw", scheduler="fixed", warmup_epochs=0,
                    torch_compile=False, amp=False, report_interval=1, objective="normalized_two_sided",
                    train_gradient_weight_function="0.8", val_gradient_weight_function="0.2",
                    relative_flatness_layers="last_matrix_weight", relative_flatness_samples=2,
                    relative_flatness_batches=1, relative_flatness_seed=2718, augmentation="none", dataset="imagenet1k",
                    token_norms=True, token_norm_threshold=8.0, progress=False,
                )
                with self.assertLogs("misalignment_measurement_2", level="INFO"):
                    result = measurement._run_direction(
                        direction_name="a_as_train_b_as_val", train_partition_name="a", val_partition_name="b",
                        train_loader=loader, val_loader=loader, flatness_train_loader=loader, flatness_val_loader=loader,
                        bundle=bundle, model=model, initial_state=copy.deepcopy(model.state_dict()),
                        criterion=nn.CrossEntropyLoss(), device=torch.device("cpu"), batch_size=2,
                        epochs=2, args=args, output_folder=Path(directory), model_type_name=name, dataset_type_name="imagenet1k",
                    )
                self.assertTrue(math.isfinite(result["misalignment_measure"]["overall_mean"]))
                self.assertTrue(result["token_norm_diagnostics"]["enabled"])
                self.assertEqual(result["token_norm_diagnostics"]["num_register_tokens"], 0)
                self.assertEqual(set(result["final"]["train"]["token_norms"]), {"cls", "patch"})
                self.assertEqual(result["final"]["train"]["token_norms"]["patch"]["count"], 16)
                self.assertFalse(model.encoder.ln._forward_pre_hooks)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for AMP and Inductor integration")
    def test_cuda_compile_amp_training_and_eval(self):
        for name, constructor, _ in VARIANTS:
            with self.subTest(model=name):
                model = tiny_variant(constructor).cuda()
                nn.init.normal_(model.heads.head.weight, std=0.1)
                compiled = torch.compile(model, fullgraph=True)
                images = torch.randn(2, 3, 8, 8, device="cuda")
                targets = torch.tensor([0, 1], device="cuda")
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(compiled(images), targets)
                loss.backward()
                for key, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, key)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), key)
                optimizer.step()
                compiled.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    actual = compiled(images)
                    expected = model(images)
                torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.03)


if __name__ == "__main__":
    unittest.main()
