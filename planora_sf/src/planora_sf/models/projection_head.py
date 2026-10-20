"""DINOHead projection head, as in SelEx/GCD (SX-C models/vision_transformer.py:264-298)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class DINOHead(nn.Module):
    """MLP (in -> hidden -> ... -> bottleneck), ℓ2-normalise, weight-normalised linear to ``out_dim``.

    As with ``norm_last_layer=True`` in SelEx, the weight-norm gain is fixed at 1
    and frozen, so only the direction parameters of the last layer train.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        num_layers: int = 3,
        use_bn: bool = False,
    ):
        super().__init__()
        num_layers = max(num_layers, 1)
        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        gain = self.last_layer.parametrizations.weight.original0
        with torch.no_grad():
            gain.fill_(1.0)
        gain.requires_grad_(False)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(self.mlp(x), dim=-1, p=2)
        return self.last_layer(x)
