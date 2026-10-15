"""Torchvision ViT with the 2D RoPE-Mixed positional encoding.

Frequency initialization and patch-only rotation follow RoPE-ViT (ECCV 2024):
https://github.com/naver-ai/rope-vit/blob/main/models/vit_rope.py
The rest of the architecture and common initial weights stay torchvision's.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.vision_transformer import Encoder, VisionTransformer


def _mixed_frequencies(head_dim: int, num_heads: int, theta: float) -> torch.Tensor:
    """Initialize two orthogonal, randomly oriented frequency axes per head."""
    magnitude = theta ** (-torch.arange(0, head_dim, 4, dtype=torch.float32) / head_dim)
    angle = torch.rand(num_heads, 1) * (2 * math.pi)
    frequency_x = torch.cat((magnitude * angle.cos(), magnitude * (angle + math.pi / 2).cos()), dim=-1)
    frequency_y = torch.cat((magnitude * angle.sin(), magnitude * (angle + math.pi / 2).sin()), dim=-1)
    return torch.stack((frequency_x, frequency_y))


def _rotate_patch_tokens(tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent channel pairs; leave the leading CLS token untouched."""
    pairs = tokens[:, :, 1:].float().unflatten(-1, (-1, 2))
    real, imaginary = pairs.unbind(dim=-1)
    rotated = torch.stack((real * cos - imaginary * sin, real * sin + imaginary * cos), dim=-1)
    return torch.cat((tokens[:, :, :1], rotated.flatten(-2).to(tokens.dtype)), dim=2)


class RotarySelfAttention(nn.Module):
    """Self-attention retaining torchvision's projection names and weights."""

    def __init__(self, attention: nn.MultiheadAttention, *, grid_size: int, theta: float) -> None:
        super().__init__()
        head_dim = attention.embed_dim // attention.num_heads
        if head_dim % 4:
            raise ValueError("2D RoPE requires the attention head dimension to be divisible by 4")
        if not math.isfinite(theta) or theta <= 0:
            raise ValueError("RoPE theta must be finite and positive")
        self.embed_dim = attention.embed_dim
        self.num_heads = attention.num_heads
        self.head_dim = head_dim
        self.dropout = attention.dropout
        # Reuse existing parameters: replacing attention must not reinitialize
        # Q/K/V or the output projection for the paired architecture experiment.
        self.in_proj_weight = attention.in_proj_weight
        self.in_proj_bias = attention.in_proj_bias
        self.out_proj = attention.out_proj
        self.rope_frequencies = nn.Parameter(_mixed_frequencies(head_dim, self.num_heads, theta))
        positions = torch.arange(grid_size * grid_size, dtype=torch.float32)
        coordinates = torch.stack((positions % grid_size, torch.div(positions, grid_size, rounding_mode="floor")), dim=-1)
        # Coordinates are derived from the architecture, not checkpoint state.
        self.register_buffer("rope_coordinates", coordinates, persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, need_weights: bool = False,
    ) -> tuple[torch.Tensor, None]:
        if query is not key or query is not value or need_weights:
            raise ValueError("RotarySelfAttention only supports self-attention with need_weights=False")
        batch, length, _ = query.shape
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.reshape(batch, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        # Phase computation and rotations stay in FP32 under BF16/FP16 AMP.
        # Real-valued rotations also avoid complex operations in torch.compile.
        with torch.autocast(device_type=query.device.type, enabled=False):
            phase = self.rope_coordinates.float() @ self.rope_frequencies.float().flatten(1)
            phase = phase.reshape(-1, self.num_heads, self.head_dim // 2).permute(1, 0, 2)
            cos, sin = phase.cos().unsqueeze(0), phase.sin().unsqueeze(0)
            q = _rotate_patch_tokens(q, cos, sin)
            k = _rotate_patch_tokens(k, cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        attended = attended.transpose(1, 2).reshape(batch, length, self.embed_dim)
        return self.out_proj(attended), None


class RotaryEncoder(nn.Module):
    """Reuse the baseline encoder, replacing absolute positions with RoPE."""

    def __init__(self, encoder: Encoder, *, grid_size: int, theta: float) -> None:
        super().__init__()
        self.dropout = encoder.dropout
        self.layers = encoder.layers
        self.ln = encoder.ln
        for block in self.layers:
            block.self_attention = RotarySelfAttention(block.self_attention, grid_size=grid_size, theta=theta)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.ln(self.layers(self.dropout(tokens)))


class VisionTransformerWithRoPE(VisionTransformer):
    def __init__(self, *args, rope_theta: float = 10.0, **kwargs) -> None:
        # Construct the entire baseline before initializing new frequencies,
        # preserving every common parameter at equal random seeds.
        super().__init__(*args, **kwargs)
        self.rope_theta = rope_theta
        self.rope_variant = "mixed_2d"
        self.encoder = RotaryEncoder(self.encoder, grid_size=self.image_size // self.patch_size, theta=rope_theta)


def vit_b_32_rope(*, num_classes: int = 1000) -> VisionTransformerWithRoPE:
    """Random ViT-B/32 at 224px, 2D RoPE-Mixed, theta=10, without APE."""
    return VisionTransformerWithRoPE(
        image_size=224, patch_size=32, num_layers=12, num_heads=12,
        hidden_dim=768, mlp_dim=3072, num_classes=num_classes,
    )
