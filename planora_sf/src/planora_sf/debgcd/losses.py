"""DebGCD losses and certainty weighting (arXiv 2504.04804 §4.2-4.3, Eq. 4-13).

Detector tensors are softmax-over-channel probabilities of ``[N, 2, M]`` logits
(channel 0 = o⁻, channel 1 = o⁺). The ``1e-8`` inside every log is upstream's
(train_DebGCD.py:44-53). Open items DG-xx are listed in docs/debgcd_port.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

_EPS = 1e-8


def _zero(like: torch.Tensor) -> torch.Tensor:
    """A graph-connected zero for an empty labelled or unlabelled subset."""
    return like.sum() * 0.0


def sdl_labelled(ova_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Eq. 4: mean_i [ -log o⁺_{i,y} - min_{k≠y} log o⁻_{i,k} ] (hard negative, DG-06)."""
    if ova_logits.shape[0] == 0:
        return _zero(ova_logits)
    probs = ova_logits.softmax(dim=1)
    pos = -torch.log(probs[:, 1].gather(1, targets[:, None]).squeeze(1) + _EPS)
    neg_nll = -torch.log(probs[:, 0] + _EPS)
    not_target = torch.ones_like(neg_nll).scatter_(1, targets[:, None], 0.0)
    hard_neg = (neg_nll * not_target).max(dim=1).values
    return (pos + hard_neg).mean()


def sdl_unlabelled(ova_logits: torch.Tensor) -> torch.Tensor:
    """Eq. 5: binary entropy of every OVA head, mean over heads and samples (DG-02)."""
    if ova_logits.shape[0] == 0:
        return _zero(ova_logits)
    probs = ova_logits.softmax(dim=1)
    return -(probs * torch.log(probs + _EPS)).sum(dim=1).mean()


def distribution_score(ova_logits: torch.Tensor) -> torch.Tensor:
    """Eq. 7: s_i = o⁻_{i, ŷ} with ŷ = argmax_j o⁺_{i,j}; s > 0.5 reads "novel"."""
    probs = ova_logits.softmax(dim=1)
    closest = probs[:, 1].argmax(dim=1, keepdim=True)
    return probs[:, 0].gather(1, closest).squeeze(1)


def certainty(score: torch.Tensor) -> torch.Tensor:
    """Eq. 10: d_i = |2 s_i - 1|."""
    return (2 * score - 1).abs()


def consistency(pred_is_novel: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """Eq. 9 (formula reading, DG-03): 1 iff the classifier and the detector agree on known vs novel."""
    return ((pred_is_novel & (score > 0.5)) | (~pred_is_novel & (score < 0.5))).float()


def adl_labelled(deb_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Eq. 12: cross-entropy of ψ_s with ground-truth labels."""
    if deb_logits.shape[0] == 0:
        return _zero(deb_logits)
    return F.cross_entropy(deb_logits, targets)


def adl_unlabelled(deb_logits: torch.Tensor, pseudo: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Eq. 11: (1/|B_u|) Σ_i w_i · CE(ψ_s(x_i), ŷ_i), with w_i = 1[max p_i > τ] · F_i · d_i."""
    if deb_logits.shape[0] == 0:
        return _zero(deb_logits)
    return (F.cross_entropy(deb_logits, pseudo, reduction="none") * weight).mean()


@dataclass
class DebGCDTerms:
    sdl: torch.Tensor  # Eq. 6: L_sdl^s + L_sdl^u
    adl: torch.Tensor  # Eq. 13: L_adl^s + L_adl^u
    utilisation: float  # share of unlabelled rows with a non-zero Eq. 11 weight (plan C6)


def debgcd_terms(
    ova_logits: torch.Tensor,
    deb_logits: torch.Tensor,
    posterior: torch.Tensor,
    targets: torch.Tensor,
    labelled: torch.Tensor,
    num_known_clusters: int,
    *,
    threshold: float,
    consistency_filter: bool = True,
) -> DebGCDTerms:
    """L_sdl and L_adl on N stacked rows (both views).

    ``posterior [N, K]`` is the class posterior p_i used for pseudo-labels (no gradient).
    ``targets`` is read only where ``labelled`` is true. Clusters ``>= num_known_clusters``
    are novel. Weights are computed without gradient, as upstream detaches the OVA scores.
    """
    rows = {t.shape[0] for t in (ova_logits, deb_logits, posterior, targets, labelled)}
    if len(rows) != 1:
        raise ValueError(f"DebGCD inputs disagree on the number of rows: {sorted(rows)}")
    labelled = labelled.bool()
    unl = ~labelled
    y = targets[labelled]
    sdl = sdl_labelled(ova_logits[labelled], y) + sdl_unlabelled(ova_logits[unl])

    with torch.no_grad():
        max_prob, pseudo = posterior[unl].max(dim=1)
        score = distribution_score(ova_logits[unl])
        weight = (max_prob > threshold).float() * certainty(score)
        if consistency_filter:
            weight = weight * consistency(pseudo >= num_known_clusters, score)
    adl = adl_labelled(deb_logits[labelled], y) + adl_unlabelled(deb_logits[unl], pseudo, weight)
    utilisation = float((weight > 0).float().mean()) if weight.numel() else 0.0
    return DebGCDTerms(sdl=sdl, adl=adl, utilisation=utilisation)

