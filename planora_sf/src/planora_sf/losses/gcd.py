"""GCD base loss (FX-P Eq. 3), the L_base of the FourGCD variant (FX-A §8)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from planora_sf.losses.contrastive import ContrastiveSettings, nt_xent_logits, supcon_loss


def gcd_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    gt_labels: torch.Tensor,
    labelled: torch.Tensor,
    *,
    lam: float,
    settings: ContrastiveSettings,
) -> torch.Tensor:
    """L^g = (1 - λ) L^u + λ L^s, with sums realised as means."""
    logits, _ = nt_xent_logits(torch.cat([z_a, z_b]), settings.tau_unsup, settings.similarity)
    unsup = F.cross_entropy(logits, torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device))
    labelled = labelled.bool()
    sup = z_a.sum() * 0.0
    if labelled.any():
        views = torch.stack([z_a, z_b], dim=1)[labelled]
        sup = supcon_loss(views, gt_labels[labelled], settings.tau_sup, settings.similarity)
    return (1.0 - lam) * unsup + lam * sup
