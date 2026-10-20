"""A small randomly initialised ViT used as a stand-in backbone in synthetic tests.

It is not a pretrained model and must not be used to report results. It mirrors
the interface the backbone adapter relies on: a ``blocks`` ModuleList, an
``embed_dim`` attribute, a final ``norm`` and a forward pass that returns the CLS token.
"""
from __future__ import annotations

import torch
from torch import nn


class TinyBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 32,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        if img_size % patch_size:
            raise ValueError("img_size must be divisible by patch_size")
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.blocks = nn.ModuleList(TinyBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]
