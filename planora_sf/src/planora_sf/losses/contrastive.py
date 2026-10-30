"""Contrastive primitives (FX-P Eq. 1-2) as realised in the SelEx code (spec §5.1).

Similarity is ``-||a - b||_2`` by default (ct.py:118,174, D-06). HypCD adds ``-d_c(a, b)``
on the Poincaré ball (``NegPoincare``) and cosine similarity (plan Phase 4). Every function
accepts an optional per-anchor boolean dimension mask ``[N, D]``, which restricts
anchor ``a``'s similarities to the dimensions selected in row ``a``. This is
how latent slices of different widths per sample are computed (spec §3.3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F

from planora_sf.config import PlanoraSFConfig
from planora_sf.hyperbolic.poincare import distance_from_squares

_EPS = 1e-12


@dataclass(frozen=True)
class NegPoincare:
    """Similarity -d_c(a, b) on the Poincaré ball of curvature -c (HypCD S_d = -D_H)."""

    curvature: float

    def __post_init__(self) -> None:
        if self.curvature <= 0:
            raise ValueError(f"NegPoincare needs curvature > 0, got {self.curvature}")


Similarity = Union[str, NegPoincare]  # "neg_euclidean" | "dot" | "cosine" | NegPoincare(c)


@dataclass(frozen=True)
class ContrastiveSettings:
    similarity: Similarity = "neg_euclidean"
    tau_unsup: float = 1.0
    tau_sup: float = 0.07
    renormalize_slices: bool = False

    @classmethod
    def from_config(cls, cfg: PlanoraSFConfig) -> "ContrastiveSettings":
        return cls(
            similarity=cfg.selex.similarity,
            tau_unsup=cfg.selex.tau_unsup,
            tau_sup=cfg.selex.tau_sup,
            renormalize_slices=cfg.latent.renormalize_slices,
        )


def pairwise_similarity(
    anchor: torch.Tensor,
    contrast: torch.Tensor,
    kind: Similarity = "neg_euclidean",
    anchor_dim_mask: Optional[torch.Tensor] = None,
    renormalize: bool = False,
) -> torch.Tensor:
    """``[N, M]`` similarities between ``anchor [N, D]`` and ``contrast [M, D]``.

    With ``anchor_dim_mask`` row ``a`` only uses the dimensions selected for anchor
    ``a`` on both sides. With ``renormalize`` the (masked) vectors are ℓ2-normalised
    before comparison (U-14). A masked or sliced ball point stays in the lower-dimensional
    ball, so ``NegPoincare`` on it is the distance there (upstream slices ball points too,
    train_HypSelEx.py:356-357).
    """
    if isinstance(kind, NegPoincare):
        if renormalize:
            raise ValueError("NegPoincare compares ball points; renormalising them is undefined")
        return -_poincare_distances(anchor, contrast, kind.curvature, anchor_dim_mask)
    if kind == "cosine":
        return pairwise_similarity(anchor, contrast, "dot", anchor_dim_mask, renormalize=True)
    if kind not in ("neg_euclidean", "dot"):
        raise ValueError(f"unknown similarity {kind!r}")
    if anchor_dim_mask is None:
        a, c = anchor, contrast
        if renormalize:
            a, c = F.normalize(a, dim=-1), F.normalize(c, dim=-1)
        cross = a @ c.T
        if kind == "dot":
            return cross
        sq_dist = (a * a).sum(1, keepdim=True) + (c * c).sum(1)[None, :] - 2 * cross
    else:
        if anchor_dim_mask.shape != anchor.shape:
            raise ValueError("anchor_dim_mask must have the same shape as anchor")
        m = anchor_dim_mask.to(anchor.dtype)
        a_masked = anchor * m
        cross = a_masked @ contrast.T
        contrast_sq_under_mask = m @ (contrast * contrast).T
        if renormalize:
            a_norm = a_masked.norm(dim=1, keepdim=True).clamp_min(_EPS)
            c_norm = contrast_sq_under_mask.clamp_min(_EPS).sqrt()
            cosine = cross / (a_norm * c_norm)
            if kind == "dot":
                return cosine
            sq_dist = 2.0 - 2.0 * cosine
        else:
            if kind == "dot":
                return cross
            sq_dist = (a_masked * a_masked).sum(1, keepdim=True) + contrast_sq_under_mask - 2 * cross
    # Clamping keeps the gradient finite (zero) at zero distance, e.g. on the diagonal.
    return -torch.sqrt(sq_dist.clamp_min(_EPS))


def _poincare_distances(
    anchor: torch.Tensor, contrast: torch.Tensor, c: float, anchor_dim_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """``[N, M]`` d_c from squared norms and distances, masked like the Euclidean path."""
    if anchor_dim_mask is None:
        a_sq = (anchor * anchor).sum(1, keepdim=True)
        c_sq = (contrast * contrast).sum(1)[None, :]
        sq_dist = a_sq + c_sq - 2 * anchor @ contrast.T
    else:
        if anchor_dim_mask.shape != anchor.shape:
            raise ValueError("anchor_dim_mask must have the same shape as anchor")
        m = anchor_dim_mask.to(anchor.dtype)
        a_masked = anchor * m
        a_sq = (a_masked * a_masked).sum(1, keepdim=True)
        c_sq = m @ (contrast * contrast).T
        sq_dist = a_sq + c_sq - 2 * a_masked @ contrast.T
    return distance_from_squares(sq_dist, a_sq, c_sq, c)


def nt_xent_logits(
    features: torch.Tensor,
    temperature: float = 1.0,
    kind: Similarity = "neg_euclidean",
    dim_mask: Optional[torch.Tensor] = None,
    renormalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NT-Xent logits for two stacked views ``[x(1); x(2)]`` of shape ``[2B, D]`` (ct.py:162-204).

    Returns ``(logits [2B, 2B-1], columns [2B, 2B-1])``. Column 0 of every row is
    the positive (the other view of the same image); the rest are the negatives in
    increasing index order, as in the SelEx code. ``columns`` maps each logit back
    to its index in the ``2B x 2B`` similarity matrix, so target matrices can be
    rearranged identically with ``matrix.gather(1, columns)``.
    """
    n = features.shape[0]
    if n % 2:
        raise ValueError("NT-Xent expects two stacked views (an even number of rows)")
    half = n // 2
    sim = pairwise_similarity(features, features, kind, dim_mask, renormalize)
    idx = torch.arange(n, device=features.device)
    positive = (idx + half) % n
    excluded = torch.eye(n, dtype=torch.bool, device=features.device)
    excluded[idx, positive] = True
    negatives = (~excluded).nonzero()[:, 1].view(n, n - 2)
    columns = torch.cat([positive[:, None], negatives], dim=1)
    return sim.gather(1, columns) / temperature, columns


def soft_target_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean over rows of ``-sum_j t_j log softmax(logits)_j`` (torch CrossEntropyLoss with probability targets)."""
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def supcon_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    kind: Similarity = "neg_euclidean",
    dim_mask: Optional[torch.Tensor] = None,
    renormalize: bool = False,
) -> torch.Tensor:
    """Supervised contrastive loss over ``features [N, V, D]`` (SelEx SupConLoss, ct.py:55-148).

    Rows are flattened view-major (all samples of view 0, then view 1, ...), so a
    per-anchor ``dim_mask`` must be ``[V*N, D]`` in that order. Anchors without a
    positive are skipped; an empty input gives 0.
    """
    n, n_views, _ = features.shape
    if n == 0:
        return features.sum() * 0.0
    contrast = torch.cat(torch.unbind(features, dim=1), dim=0)
    logits = pairwise_similarity(contrast, contrast, kind, dim_mask, renormalize) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    tiled = labels.reshape(-1).repeat(n_views)
    not_self = 1.0 - torch.eye(contrast.shape[0], device=features.device, dtype=logits.dtype)
    positives = (tiled[:, None] == tiled[None, :]).to(logits.dtype) * not_self
    log_prob = logits - torch.log((torch.exp(logits) * not_self).sum(dim=1, keepdim=True))
    count = positives.sum(dim=1)
    valid = count > 0
    if not valid.any():
        return features.sum() * 0.0
    mean_log_prob_pos = (positives * log_prob).sum(dim=1)[valid] / count[valid]
    return -mean_log_prob_pos.mean()
