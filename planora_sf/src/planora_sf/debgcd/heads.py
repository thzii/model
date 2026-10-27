"""DebGCD heads: semantic distribution detector and auxiliary debiased classifier.

DebGCD (Liu et al., ICLR 2025, arXiv 2504.04804) §4.2-4.3. Layer layout follows the
official code (Visual-AI/DebGCD @ 9ec30fe, MIT licence, Copyright (c) 2025 Visual AI Lab):
``model.py:9-91`` (``DebGCDHead``) and ``train_DebGCD.py:133-177`` (temperatures).
SimGCD's own prototype classifier is not ported: SelEx has none (plan §2.1, C1).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


def _cosine_layer(in_dim: int, out_dim: int) -> nn.Module:
    """Weight-normalised linear layer with the gain fixed at 1 (upstream ``norm_last_layer=True``)."""
    layer = weight_norm(nn.Linear(in_dim, out_dim, bias=False))
    gain = layer.parametrizations.weight.original0
    with torch.no_grad():
        gain.fill_(1.0)
    gain.requires_grad_(False)
    return layer


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Module:
    """``mlp_ood`` of upstream model.py:37-66: Linear-GELU stack, trunc-normal(0.02) weights, zero bias."""
    if num_layers == 1:
        layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
    else:
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
    for m in layers:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
    return nn.Sequential(*layers)


class SemanticDistributionDetector(nn.Module):
    """ρ_s -> ℓ2 -> M one-vs-all binary heads (DebGCD §4.2, App. K: 5 layers).

    Returns logits ``[B, 2, M]``: channel 0 is o⁻ ("not class j"), channel 1 is o⁺
    ("class j"), as upstream ``ova_loss`` reads them (train_DebGCD.py:38-45).
    Input is the **un-normalised** backbone CLS, as upstream ``mlp_ood(x)``.
    """

    def __init__(self, in_dim: int, num_known: int, *, num_layers: int = 5, hidden_dim: int = 2048,
                 bottleneck_dim: int = 256, temperature: float = 0.1):
        super().__init__()
        if num_known < 1 or num_layers < 1:
            raise ValueError("num_known and num_layers must be >= 1")
        self.num_known = num_known
        self.temperature = temperature
        self.mlp = _mlp(in_dim, hidden_dim, bottleneck_dim, num_layers)
        self.last_layer = _cosine_layer(bottleneck_dim, 2 * num_known)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.last_layer(F.normalize(self.mlp(x), dim=-1)) / self.temperature
        return logits.view(x.shape[0], 2, self.num_known)


class DebiasedPrototypeHead(nn.Module):
    """ψ_s: K cosine prototypes on the ℓ2 CLS, logits / τ_a (DebGCD Eq. 8; upstream ``last_layer_deb``)."""

    def __init__(self, in_dim: int, num_classes: int, *, tau_a: float = 0.1):
        super().__init__()
        self.tau_a = tau_a
        self.last_layer = _cosine_layer(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.last_layer(F.normalize(x, dim=-1)) / self.tau_a


class DebGCDHeads(nn.Module):
    """Both DebGCD heads on the backbone CLS of the two standard views only (plan C4)."""

    def __init__(self, in_dim: int, num_known: int, num_classes: int, cfg):
        super().__init__()
        self.detector = SemanticDistributionDetector(
            in_dim, num_known, num_layers=cfg.detector_layers, hidden_dim=cfg.detector_hidden_dim,
            bottleneck_dim=cfg.detector_bottleneck_dim, temperature=cfg.tau_ova,
        )
        self.debiased = DebiasedPrototypeHead(in_dim, num_classes, tau_a=cfg.tau_a)

    def forward(self, cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``cls [N, F]`` -> (detector logits ``[N, 2, M]``, debiased logits ``[N, K]``)."""
        return self.detector(cls), self.debiased(cls)
