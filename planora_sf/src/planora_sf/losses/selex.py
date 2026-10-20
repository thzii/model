"""SelEx self-expertise losses (SX-P Eq. 12-14; code form in spec §5.2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from planora_sf.latent.partition import prefix_mask, selex_prefix_width
from planora_sf.losses.contrastive import ContrastiveSettings, nt_xent_logits, soft_target_cross_entropy, supcon_loss


def use_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    settings: ContrastiveSettings,
    *,
    dim_mask: Optional[torch.Tensor] = None,
    loss_type: str = "soft_ce",
) -> torch.Tensor:
    """Unsupervised self-expertise loss on two stacked views ``features [2B, D]``.

    ``targets [2B, 2B]`` is the USE target matrix (pseudo_labels.target_matrix).
    The smoothed target is (1 - alpha)·e_pos + alpha·y (SX-11). ``"soft_ce"`` is
    the code's LabelSmoothingLoss (D-01); ``"bce"`` is the literal SX-12 reading.
    """
    logits, columns = nt_xent_logits(
        features, settings.tau_unsup, settings.similarity, dim_mask, settings.renormalize_slices
    )
    one_hot = torch.zeros_like(logits)
    one_hot[:, 0] = 1.0
    soft = (1.0 - alpha) * one_hot + alpha * targets.to(logits.dtype).gather(1, columns)
    if loss_type == "soft_ce":
        return soft_target_cross_entropy(logits, soft)
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(logits, soft.clamp(0.0, 1.0))
    raise ValueError(f"unknown USE loss type {loss_type!r}")


def _prefix_supcon(
    views: torch.Tensor,
    labels: torch.Tensor,
    width: int,
    settings: ContrastiveSettings,
    cap_widths: Optional[torch.Tensor],
) -> torch.Tensor:
    """SupCon on the prefix [0, width); with ``cap_widths`` each sample uses [0, min(width, cap))."""
    if cap_widths is None:
        sliced = views[..., :width]
        if settings.renormalize_slices:
            sliced = F.normalize(sliced, dim=-1)
        return supcon_loss(sliced, labels, settings.tau_sup, settings.similarity)
    widths = cap_widths.clamp(max=width).repeat(views.shape[1])
    mask = prefix_mask(widths.to(views.device), views.shape[-1])
    return supcon_loss(
        views, labels, settings.tau_sup, settings.similarity, dim_mask=mask, renormalize=settings.renormalize_slices
    )


def sse_loss(
    views: torch.Tensor,
    gt_labels: torch.Tensor,
    labelled: torch.Tensor,
    level_labels: Sequence[torch.Tensor],
    settings: ContrastiveSettings,
    *,
    cap_widths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L_SSE = ½ [SupCon(GT, full D) + Σ_{k≥1} SupCon(level k-1 pseudo-labels, [0, D/2^k)) / 2^k].

    ``views [B, 2, D]``. ``level_labels[i]`` holds the code's level-i pseudo-labels,
    i.e. the paper's level k = i + 1 (SX-13, spec §5.2). With ``cap_widths`` (d_l per
    sample) every prefix becomes [0, min(D/2^k, d_l)): this is L_LSSE (FX-A Eq. 12).
    """
    dim = views.shape[-1]
    labelled = labelled.bool()
    total = views.sum() * 0.0
    if labelled.any():
        caps = cap_widths[labelled] if cap_widths is not None else None
        total = total + _prefix_supcon(views[labelled], gt_labels[labelled], dim, settings, caps)
    for i, labels in enumerate(level_labels):
        k = i + 1
        total = total + _prefix_supcon(views, labels, selex_prefix_width(dim, k), settings, cap_widths) / 2**k
    return 0.5 * total


@dataclass
class SelExLossParts:
    total: torch.Tensor
    use: torch.Tensor
    sse: torch.Tensor


def selex_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    gt_labels: torch.Tensor,
    labelled: torch.Tensor,
    level_labels: Sequence[torch.Tensor],
    use_targets: torch.Tensor,
    *,
    lam: float,
    alpha: float,
    settings: ContrastiveSettings,
    loss_type: str = "soft_ce",
) -> SelExLossParts:
    """L_SE = (1 - λ) L_USE + λ L_SSE (SX-14) for the pair of views (z_a, z_b), each [B, D]."""
    use = use_loss(torch.cat([z_a, z_b]), use_targets, alpha, settings, loss_type=loss_type)
    sse = sse_loss(torch.stack([z_a, z_b], dim=1), gt_labels, labelled, level_labels, settings)
    return SelExLossParts(total=(1.0 - lam) * use + lam * sse, use=use, sse=sse)
