from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise")
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight_noise > 0 and self.training:
            bias = self.bias if self.bias is None else self.bias + torch.randn_like(self.bias) * self.weight_noise
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
        else:
            bias = self.bias
            weight = self.weight
        return F.linear(input, weight, bias)


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise")
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight_noise > 0 and self.training:
            bias = self.bias if self.bias is None else self.bias + torch.randn_like(self.bias) * self.weight_noise
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
        else:
            bias = self.bias
            weight = self.weight
        return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)


class Embedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise")
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight_noise > 0 and self.training:
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
        else:
            weight = self.weight
        return F.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_key: int, weight_noise: float) -> None:
        super().__init__()
        self.d_key = d_key
        self.Wq = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)
        self.Wk = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)
        self.Wv = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
        save_activations: bool = False,
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        queries = self.Wq(queries)
        keys = self.Wk(keys)
        values = self.Wv(values)

        attn = torch.matmul(queries, torch.transpose(keys, -2, -1))
        attn = attn / math.sqrt(self.d_key)
        if mask is not None:
            attn.masked_fill_(mask == 0, float("-inf"))
        attn = self.softmax(attn)

        result = torch.matmul(attn, values)
        if save_activations:
            return result, attn.detach().clone(), values.detach().clone()
        return result, None, None


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, weight_noise: float = 0.0) -> None:
        super().__init__()
        d_key = int(d_model / heads)
        self.attn_heads = nn.ModuleList(
            [AttentionHead(d_model, d_key, weight_noise=weight_noise) for _ in range(heads)]
        )
        self.Wo = Linear(d_model, d_model, bias=False, weight_noise=weight_noise)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
        save_activations: bool = False,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        head_outputs = [
            head(
                queries=queries,
                keys=keys,
                values=values,
                mask=mask,
                save_activations=save_activations,
            )
            for head in self.attn_heads
        ]
        head_results = [output[0] for output in head_outputs]
        layer_attns = [output[1] for output in head_outputs if output[1] is not None]
        layer_values = [output[2] for output in head_outputs if output[2] is not None]
        multihead_result = torch.cat(head_results, dim=-1)
        multihead_result = self.Wo(multihead_result)
        return multihead_result, layer_attns, layer_values


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        multiplier: int = 4,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()
        d_ff = int(multiplier * d_model)
        non_linearities = {"relu": nn.ReLU, "gelu": nn.GELU}
        self.ffn = nn.Sequential(
            Linear(d_model, d_ff, bias=False, weight_noise=weight_noise),
            non_linearities[non_linearity](),
            Linear(d_ff, d_model, bias=False, weight_noise=weight_noise),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        dropout: float,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, weight_noise=weight_noise)
        self.self_attn_norm = LayerNorm(d_model, weight_noise=weight_noise)
        self.ffn = FFN(d_model, non_linearity=non_linearity, weight_noise=weight_noise)
        self.ffn_drop = nn.Dropout(p=dropout)
        self.ffn_norm = LayerNorm(d_model, weight_noise=weight_noise)

    def forward(
        self,
        x: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        save_activations: bool = False,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        a1, layer_attns, layer_values = self.self_attn(
            x, x, x, self_attn_mask, save_activations
        )
        a1 = self.self_attn_norm(x + a1)
        a2 = self.ffn(a1)
        a2 = self.ffn_drop(a2)
        a2 = self.ffn_norm(a1 + a2)
        return a2, layer_attns, layer_values


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        num_blocks: int,
        dropout: float,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    heads,
                    dropout,
                    non_linearity,
                    weight_noise=weight_noise,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        x: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        save_activations: bool = False,
    ) -> tuple[Tensor, list[list[Tensor]], list[list[Tensor]]]:
        activations = x
        attentions: list[list[Tensor]] = []
        values: list[list[Tensor]] = []
        for block in self.blocks:
            activations, layer_attentions, layer_values = block(
                activations,
                self_attn_mask,
                save_activations=save_activations,
            )
            if save_activations:
                attentions.append(layer_attentions)
                values.append(layer_values)
        return activations, attentions, values


class TransformerForGrokking(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        n_heads: int = 4,
        d_model: int = 256,
        dropout: float = 0.1,
        max_context_len: int = 1024,
        vocab_len: int = 2000,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
        trainable_position_encoding: bool = False,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.max_context_len = max_context_len
        self.non_linearity = non_linearity
        self.vocab_len = vocab_len

        self.embedding = Embedding(vocab_len, d_model, weight_noise=weight_noise)
        position_encoding = self._position_encoding(max_context_len, d_model)
        if trainable_position_encoding:
            self.position_encoding = nn.Parameter(position_encoding)
        else:
            self.register_buffer("position_encoding", position_encoding)
        self.register_buffer("self_attn_mask", self.make_mask(max_context_len))

        self.decoder = Decoder(
            d_model,
            n_heads,
            n_layers,
            dropout,
            self.non_linearity,
            weight_noise=weight_noise,
        )
        self.linear = Linear(d_model, vocab_len, bias=False, weight_noise=weight_noise)

    @staticmethod
    def make_mask(context_len: int) -> Tensor:
        return torch.ones([context_len, context_len]).tril()

    @staticmethod
    def _position_encoding(context_len: int, d_model: int) -> Tensor:
        rows = []
        for pos in range(context_len):
            row = []
            for index in range(d_model):
                if index % 2 == 0:
                    row.append(math.sin(pos / (10000 ** (index / d_model))))
                else:
                    row.append(math.cos(pos / (10000 ** ((index - 1) / d_model))))
            rows.append(torch.tensor(row, dtype=torch.float32))
        stack = torch.stack(rows, dim=1)
        return stack.T

    def embed(self, indices: Tensor) -> Tensor:
        context_len = indices.shape[-1]
        position_encoding = self.position_encoding[:context_len, :]
        embedded = self.embedding(indices)
        return position_encoding + embedded

    def forward(
        self,
        x: Tensor,
        pos: Optional[int] = None,
        save_activations: bool = False,
    ) -> tuple[Tensor, list[list[Tensor]], list[list[Tensor]]]:
        x = x.to(self.embedding.weight.device)
        context_len = x.shape[-1]
        self_attn_mask = self.self_attn_mask[:context_len, :context_len]
        embedded = self.embed(x)
        decoded, attentions, values = self.decoder(
            embedded,
            self_attn_mask,
            save_activations=save_activations,
        )
        if pos is not None:
            decoded = decoded[:, pos, :]
        y_hat = self.linear(decoded)
        return y_hat, attentions, values


Transformer = TransformerForGrokking

__all__ = ["TransformerForGrokking", "Transformer"]
