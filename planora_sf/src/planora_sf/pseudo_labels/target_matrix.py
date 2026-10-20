"""USE target matrices (SX-P Eq. 10-11, SX-S Alg. 6, SelEx ct.py:287-347; spec §5.2, U-21).

All builders take per-level pseudo-labels for the ``N = 2B`` stacked views
(code level i = paper level k = i + 1) and return an ``[N, N]`` matrix. The
USE loss then rearranges and smooths it.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from planora_sf.pseudo_labels.hssk import PseudoLabelBatch


def _normalise_rows(y: torch.Tensor) -> torch.Tensor:
    sums = y.sum(dim=1, keepdim=True)
    return torch.where(sums > 0, y / sums.clamp_min(torch.finfo(y.dtype).tiny), y)


def use_targets_paper(level_labels: Sequence[torch.Tensor]) -> torch.Tensor:
    """y_ab = Σ_k 1(c_a^k != c_b^k) / 2^k, rows normalised (SX-P Eq. 10). All-zero rows stay zero."""
    if not level_labels:
        raise ValueError("need at least one pseudo-label level")
    n = level_labels[0].numel()
    y = torch.zeros(n, n, device=level_labels[0].device)
    for i, labels in enumerate(level_labels):
        y = y + (labels[:, None] != labels[None, :]).float() / 2 ** (i + 1)
    return _normalise_rows(y)


def use_targets_radius(
    backbone_feats: torch.Tensor, level_labels: Sequence[torch.Tensor], radii: Sequence[torch.Tensor]
) -> torch.Tensor:
    """y_ab = Σ_k 1(d_ab > r^k_{c_a}) / 2^k, y_aa = 1, rows normalised (SX-S Alg. 6).

    Alg. 6 does not say whose radius is used; this uses the anchor's cluster.
    """
    dist = torch.cdist(backbone_feats, backbone_feats)
    y = torch.zeros_like(dist)
    for i, (labels, r) in enumerate(zip(level_labels, radii)):
        y = y + (dist > r[labels][:, None]).float() / 2 ** (i + 1)
    y.fill_diagonal_(1.0)
    return _normalise_rows(y)


def use_targets_compat(
    backbone_feats: torch.Tensor, level_labels: Sequence[torch.Tensor], radii: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Literal SelEx ``zero_one`` target (ct.py:288-347).

    - The column sample's radius is used: entry [a, b] compares d_ab with 2·r_{c_b}.
    - Pass **un-normalised** CLS features for parity: the code mixes them with radii
      computed from normalised prototypes (D-03).
    - The final ``cf / cf.sum(dim=1)`` broadcast divides column b by row-sum b (D-04).

    The only deviation is a tiny clamp on that divisor, so an all-zero matrix gives
    zeros instead of NaN.
    """
    dist = torch.cdist(backbone_feats, backbone_feats)
    cf = torch.zeros_like(dist)
    for i, (labels, r) in enumerate(zip(level_labels, radii)):
        cf = cf + (dist > 2 * r[labels]).float() / 2**i
    cf = (cf - cf.min()) / (cf.max() - cf.min() + 1e-7)
    return cf / cf.sum(dim=1).clamp_min(torch.finfo(cf.dtype).tiny)


def build_use_targets(
    mode: str, pseudo: PseudoLabelBatch, *, backbone_feats: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Dispatch on ``mode`` ("paper", "radius", "compat"); ``pseudo`` must already cover all 2B views."""
    if mode == "paper":
        return use_targets_paper(pseudo.level_labels)
    if backbone_feats is None:
        raise ValueError(f"target mode {mode!r} needs backbone features")
    feats = backbone_feats.detach().float()
    if mode == "radius":
        return use_targets_radius(feats, pseudo.level_labels, pseudo.radii)
    if mode == "compat":
        return use_targets_compat(feats, pseudo.level_labels, pseudo.radii)
    raise ValueError(f"unknown target mode {mode!r}")
