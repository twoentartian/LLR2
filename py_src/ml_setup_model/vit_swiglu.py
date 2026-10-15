"""Torchvision ViT with parameter-matched SwiGLU feed-forward blocks.

Uses the gate convention and 2/3 hidden-width adjustment of DINOv2:
https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py
No xFormers dependency or fused custom autograd operation is required.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.vision_transformer import VisionTransformer


class SwiGLUBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w12 = nn.Linear(in_dim, 2 * hidden_dim)
        self.w3 = nn.Linear(hidden_dim, in_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        # Keep the existing torchvision MLP initialization scheme.
        for layer in (self.w12, self.w3):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.bias, std=1e-6)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        gate, value = self.w12(tokens).chunk(2, dim=-1)
        hidden = self.dropout_1(F.silu(gate) * value)
        return self.dropout_2(self.w3(hidden))


class VisionTransformerWithSwiGLU(VisionTransformer):
    def __init__(self, *args, **kwargs) -> None:
        # Initialize all common weights before replacing the FFNs, so shared
        # attention, patch embeddings and positions match the baseline seed.
        super().__init__(*args, **kwargs)
        self.swiglu_hidden_dim = ((self.mlp_dim * 2 // 3 + 7) // 8) * 8
        for block in self.encoder.layers:
            block.mlp = SwiGLUBlock(self.hidden_dim, self.swiglu_hidden_dim, self.dropout)


def vit_b_32_swiglu(*, num_classes: int = 1000) -> VisionTransformerWithSwiGLU:
    """Random ViT-B/32 at 224px, SwiGLU width 2048 instead of GELU 3072."""
    return VisionTransformerWithSwiGLU(
        image_size=224, patch_size=32, num_layers=12, num_heads=12,
        hidden_dim=768, mlp_dim=3072, num_classes=num_classes,
    )
