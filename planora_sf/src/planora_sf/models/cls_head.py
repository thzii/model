"""Classification head for L_cls (FX-P Eq. 8).

FourEx never describes this head (U-17). The scaffold default is a single linear
layer on the backbone CLS token with one output per known class.
"""
from __future__ import annotations

import torch
from torch import nn


class LinearClsHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
