"""Standalone binary-attention CCT-7/3x1 for CIFAR-10.

This is intentionally self-contained so the vendored compact-transformers
repository does not need to be modified.  The architecture follows CCT-7/3x1:
one 3x3 tokenizer layer, 256-dimensional tokens, seven transformer blocks,
four heads, and a two-times expansion MLP.  Only Q/K attention activations are
binary; the tokenizer, projections, values, MLPs, normalization, and
classifier remain floating point.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return value
        keep_prob = 1.0 - self.drop_prob
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=value.dtype, device=value.device)
        random_tensor.floor_()
        return value.div(keep_prob) * random_tensor


class _Tokenizer(nn.Module):
    def __init__(self, n_input_channels: int, n_output_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            n_input_channels, n_output_channels, kernel_size=3, stride=1,
            padding=1, bias=False,
        )
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten(2, 3)
        self.apply(self._init_weight)

    def sequence_length(self, height: int, width: int) -> int:
        with torch.no_grad():
            return self(torch.zeros(1, 3, height, width)).shape[1]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.pool(self.activation(self.conv(value)))
        return self.flatten(value).transpose(-2, -1)

    @staticmethod
    def _init_weight(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)


class BinaryAttention(nn.Module):
    """Scaled sign Q/K attention with a straight-through estimator."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sequence_length: int,
        attention_dropout: float,
        projection_dropout: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)
        self.attention_bias = nn.Parameter(torch.zeros(num_heads, sequence_length, sequence_length))

    @staticmethod
    def _ste_sign(value: torch.Tensor) -> torch.Tensor:
        hard = torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))
        return value + (hard - value).detach()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = value.shape
        qkv = self.qkv(value).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads,
        ).permute(2, 0, 3, 1, 4)
        query, key, values = qkv[0], qkv[1], qkv[2]

        query_scale = query.abs().mean(dim=-1, keepdim=True)
        key_scale = key.abs().mean(dim=-1, keepdim=True)
        query = self._ste_sign(query) * query_scale
        key = self._ste_sign(key) * key_scale

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention + self.attention_bias[:, :tokens, :tokens].unsqueeze(0)
        attention = self.attn_drop(attention.softmax(dim=-1))

        output = (attention @ values).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(output))


class _TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        attention_dropout: float,
        drop_path_rate: float,
        sequence_length: int,
    ):
        super().__init__()
        self.pre_norm = nn.LayerNorm(embedding_dim)
        self.self_attn = BinaryAttention(
            embedding_dim, num_heads, sequence_length, attention_dropout, dropout,
        )
        self.linear1 = nn.Linear(embedding_dim, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.linear2 = nn.Linear(dim_feedforward, embedding_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = _DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.drop_path(self.self_attn(self.pre_norm(value)))
        value = self.norm1(value)
        feedforward = self.linear2(self.dropout1(F.gelu(self.linear1(value))))
        return value + self.drop_path(self.dropout2(feedforward))


class _TransformerClassifier(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        embedding_dim: int = 256,
        num_layers: int = 7,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        num_classes: int = 10,
        dropout: float = 0.0,
        attention_dropout: float = 0.1,
        stochastic_depth: float = 0.1,
        positional_embedding: str = "learnable",
    ):
        super().__init__()
        self.sequence_length = sequence_length
        if positional_embedding == "learnable":
            self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim))
            nn.init.trunc_normal_(self.positional_emb, std=0.2)
        elif positional_embedding == "sine":
            self.register_buffer("positional_emb", self._sinusoidal_embedding(sequence_length, embedding_dim))
        elif positional_embedding == "none":
            self.positional_emb = None
        else:
            raise ValueError(f"unsupported positional_embedding={positional_embedding!r}")
        self.dropout = nn.Dropout(dropout)
        drop_path_rates = torch.linspace(0, stochastic_depth, num_layers).tolist()
        self.blocks = nn.ModuleList([
            _TransformerEncoderLayer(
                embedding_dim, num_heads, int(embedding_dim * mlp_ratio), dropout,
                attention_dropout, drop_path_rates[index], sequence_length,
            )
            for index in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)
        self.attention_pool = nn.Linear(embedding_dim, 1)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.apply(self._init_weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.positional_emb is not None:
            value = value + self.positional_emb
        value = self.dropout(value)
        for block in self.blocks:
            value = block(value)
        value = self.norm(value)
        weights = F.softmax(self.attention_pool(value), dim=1).transpose(-1, -2)
        return self.fc(torch.matmul(weights, value).squeeze(-2))

    @staticmethod
    def _init_weight(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @staticmethod
    def _sinusoidal_embedding(sequence_length: int, embedding_dim: int) -> torch.Tensor:
        embedding = torch.zeros(sequence_length, embedding_dim)
        positions = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embedding_dim)
        )
        embedding[:, 0::2] = torch.sin(positions * frequencies)
        embedding[:, 1::2] = torch.cos(positions * frequencies)
        return embedding.unsqueeze(0)


class BinaryCCT7_3x1(nn.Module):
    """CCT-7/3x1 with binary Q/K attention for 32x32 images."""

    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        positional_embedding: str = "learnable",
    ):
        super().__init__()
        self.tokenizer = _Tokenizer(3, 256)
        sequence_length = self.tokenizer.sequence_length(img_size, img_size)
        self.classifier = _TransformerClassifier(
            sequence_length,
            num_classes=num_classes,
            positional_embedding=positional_embedding,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.tokenizer(value))


def binary_cct_7_3x1_32(
    pretrained: bool = False,
    progress: bool = False,
    img_size: int = 32,
    positional_embedding: str = "learnable",
    num_classes: int = 10,
    **kwargs,
) -> BinaryCCT7_3x1:
    """Build the standalone binary CCT-7/3x1 model.

    Pretrained compact-transformers checkpoints are intentionally unsupported:
    the binary attention parameters do not exist in those checkpoints.
    """
    del progress, kwargs
    if pretrained:
        raise RuntimeError("Binary CCT has no pretrained checkpoint; train it from scratch.")
    return BinaryCCT7_3x1(
        num_classes=num_classes,
        img_size=img_size,
        positional_embedding=positional_embedding,
    )


__all__ = ["BinaryAttention", "BinaryCCT7_3x1", "binary_cct_7_3x1_32"]
