"""FourEx Fourier self-supervision losses (FX-P Eq. 6-8, FX-A Eq. 12; spec §5.3-5.4)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from planora_sf.latent.partition import prefix_mask, suffix_mask
from planora_sf.losses.contrastive import ContrastiveSettings, nt_xent_logits, supcon_loss
from planora_sf.losses.selex import sse_loss, use_loss


def low_loss(
    z_x: torch.Tensor,
    z_l: torch.Tensor,
    d_low: torch.Tensor,
    gt_labels: torch.Tensor,
    labelled: torch.Tensor,
    level_labels: Sequence[torch.Tensor],
    use_targets: torch.Tensor,
    *,
    lam: float,
    alpha: float,
    settings: ContrastiveSettings,
    loss_type: str = "soft_ce",
) -> torch.Tensor:
    """L_low^SX = (1 - λ) L_USE|[0, d_l) + λ L_LSSE (spec §5.4, COMPOSED; U-16).

    ``z_x``/``z_l`` are [B, D] embeddings of the anchor view and its low-pass view;
    ``d_low [B]`` the prefix width per sample.
    """
    dim = z_x.shape[1]
    d_low = d_low.to(z_x.device)
    mask = prefix_mask(d_low.repeat(2), dim)
    use = use_loss(torch.cat([z_x, z_l]), use_targets, alpha, settings, dim_mask=mask, loss_type=loss_type)
    lsse = sse_loss(torch.stack([z_x, z_l], dim=1), gt_labels, labelled, level_labels, settings, cap_widths=d_low)
    return (1.0 - lam) * use + lam * lsse


def high_loss(
    z_x: torch.Tensor,
    z_h: torch.Tensor,
    d_high: torch.Tensor,
    gt_labels: torch.Tensor,
    labelled: torch.Tensor,
    *,
    lam: float,
    settings: ContrastiveSettings,
) -> torch.Tensor:
    """L_high = (1 - λ) L^u|[D-d_h, D) + λ L^s|[D-d_h, D) (FX-P Eq. 7, GCD form; U-15).

    The sums in Eq. 7 are realised as means, as in the SelEx code.
    """
    dim = z_x.shape[1]
    d_high = d_high.to(z_x.device)
    logits, _ = nt_xent_logits(
        torch.cat([z_x, z_h]),
        settings.tau_unsup,
        settings.similarity,
        suffix_mask(d_high.repeat(2), dim),
        settings.renormalize_slices,
    )
    unsup = F.cross_entropy(logits, torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device))
    labelled = labelled.bool()
    sup = z_x.sum() * 0.0
    if labelled.any():
        views = torch.stack([z_x, z_h], dim=1)[labelled]
        mask = suffix_mask(d_high[labelled].repeat(2), dim)
        sup = supcon_loss(
            views, gt_labels[labelled], settings.tau_sup, settings.similarity,
            dim_mask=mask, renormalize=settings.renormalize_slices,
        )
    return (1.0 - lam) * unsup + lam * sup


def cls_loss(
    logits: Sequence[torch.Tensor],
    gt_labels: torch.Tensor,
    labelled: torch.Tensor,
    *,
    kind: str = "sigmoid_bce",
) -> torch.Tensor:
    """L_cls = Σ over (h, l, x) of BCE(p, y) on labelled samples (FX-P Eq. 8; head and loss form: U-17)."""
    labelled = labelled.bool()
    total = logits[0].sum() * 0.0
    if not labelled.any():
        return total
    targets = gt_labels[labelled]
    for view_logits in logits:
        selected = view_logits[labelled]
        if kind == "sigmoid_bce":
            one_hot = F.one_hot(targets, selected.shape[1]).to(selected.dtype)
            total = total + F.binary_cross_entropy_with_logits(selected, one_hot)
        elif kind == "softmax_ce":
            total = total + F.cross_entropy(selected, targets)
        else:
            raise ValueError(f"unknown classification loss {kind!r}")
    return total
