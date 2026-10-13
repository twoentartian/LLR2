"""Torchvision-compatible ViT with learnable register tokens.

Common checkpoint keys and initialization match torchvision's ViT. Registers
are initialized after the backbone and receive no positional embeddings.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer


class VisionTransformerWithRegisters(VisionTransformer):
    def __init__(self, *args, num_register_tokens: int = 4, **kwargs) -> None:
        if not isinstance(num_register_tokens, int) or num_register_tokens <= 0:
            raise ValueError("num_register_tokens must be a positive integer")
        # Do not consume RNG before super(): equal seeds must produce the same
        # patch embedding, position embedding, transformer and classifier.
        super().__init__(*args, **kwargs)
        self.num_register_tokens = num_register_tokens
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, self.hidden_dim))
        nn.init.normal_(self.register_tokens, std=1e-6)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._process_input(x)
        batch_size = x.shape[0]
        x = torch.cat((self.class_token.expand(batch_size, -1, -1), x), dim=1)
        x = x + self.encoder.pos_embedding
        x = torch.cat(
            (x[:, :1], self.register_tokens.expand(batch_size, -1, -1), x[:, 1:]), dim=1,
        )
        x = self.encoder.layers(self.encoder.dropout(x))
        x_norm = self.encoder.ln(x)
        return {
            "x_prenorm": x,
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : 1 + self.num_register_tokens],
            "x_norm_patchtokens": x_norm[:, 1 + self.num_register_tokens :],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.heads(self.forward_features(x)["x_norm_clstoken"])


def vit_b_32_reg4(*, num_classes: int = 1000) -> VisionTransformerWithRegisters:
    """Randomly initialized ViT-B/32, 224px input and four registers."""
    return VisionTransformerWithRegisters(
        image_size=224, patch_size=32, num_layers=12, num_heads=12,
        hidden_dim=768, mlp_dim=3072, num_classes=num_classes, num_register_tokens=4,
    )
