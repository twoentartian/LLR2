"""Randomly initialized Pythia-14M; architecture and initialization provenance.

Architecture/optimizer recipe:
https://github.com/EleutherAI/pythia/blob/main/models/14M/pythia-14m.yml
https://huggingface.co/EleutherAI/pythia-14m/blob/main/config.json
NeoX initialization definitions:
https://github.com/EleutherAI/gpt-neox/blob/v1.0/megatron/model/init_functions.py
No pretrained parameter loader is used, including when a config is downloaded.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def pythia_14m_config():
    from transformers import GPTNeoXConfig

    return GPTNeoXConfig(
        vocab_size=50304,
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=6,
        num_attention_heads=4,
        max_position_embeddings=2048,
        rotary_pct=0.25,
        rotary_emb_base=10000,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_parallel_residual=True,
        tie_word_embeddings=False,
        bos_token_id=0,
        eos_token_id=0,
        use_cache=False,
    )


def build_pythia(config=None, *, attention="sdpa"):
    from transformers import GPTNeoXForCausalLM

    config = config or pythia_14m_config()
    config.use_cache = False
    config._attn_implementation = attention
    model = GPTNeoXForCausalLM(config)
    # NeoX uses small_init for input/QKV/MLP expansion and final unembedding;
    # wang_init for attention/MLP residual output projections. This is distinct
    # from Transformers' default fixed initializer_range=0.02.
    small_std = math.sqrt(2.0 / (5 * config.hidden_size))
    residual_std = 2.0 / (config.num_hidden_layers * math.sqrt(config.hidden_size))
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                residual = name.endswith(("attention.dense", "mlp.dense_4h_to_h"))
                nn.init.normal_(
                    module.weight, std=residual_std if residual else small_std
                )
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    return CausalLanguageModel(model)


class CausalLanguageModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, labels):
        # Labels are already shifted by the dataset; do not invoke HF's
        # built-in causal loss, which would shift them a second time.
        logits = self.model(input_ids=input_ids, use_cache=False).logits
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        correct = ((logits.detach().argmax(dim=-1) == labels) & (labels != -100)).sum()
        return loss_sum, correct
