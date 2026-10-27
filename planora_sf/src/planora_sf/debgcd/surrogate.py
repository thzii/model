"""Adapters that let DebGCD run on SelEx's non-parametric pseudo-labels (plan §2.1-2.3; COMPOSED, in no paper).

DebGCD takes p_i, ŷ_i and the known/novel status from SimGCD's parametric classifier.
SelEx has none, so p_i is a softmax over the epoch's level-0 BSSK centroids (C1), and
novel cluster ids are aligned across epochs so that ψ_s sees stable class ids (C2).
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from planora_sf.pseudo_labels.hssk import HierarchyLevel


@torch.no_grad()
def centroid_posterior(feats: torch.Tensor, centroids: torch.Tensor, tau_p: float) -> torch.Tensor:
    """p_i = softmax_k(cos(f_i, μ_k) / τ_p) over ``centroids [K, F]``; no gradient (DG-01)."""
    return (F.normalize(feats.float(), dim=-1) @ F.normalize(centroids.float(), dim=-1).T / tau_p).softmax(dim=1)


def align_novel_clusters(prev_centers: torch.Tensor, new_centers: torch.Tensor, num_known: int) -> torch.Tensor:
    """``perm [K]`` mapping each new cluster id to an aligned id.

    Known clusters keep their ids (they are anchored to known classes). Novel clusters are
    matched to the previous epoch's novel clusters by Hungarian on cosine similarity.
    """
    if prev_centers.shape != new_centers.shape:
        raise ValueError(f"centre shapes differ: {tuple(prev_centers.shape)} vs {tuple(new_centers.shape)}")
    k = new_centers.shape[0]
    perm = torch.arange(k)
    if k - num_known < 2:
        return perm
    sim = F.normalize(prev_centers[num_known:].float(), dim=1) @ F.normalize(new_centers[num_known:].float(), dim=1).T
    old_ids, new_ids = linear_sum_assignment(sim.cpu().numpy(), maximize=True)
    perm[torch.as_tensor(new_ids) + num_known] = torch.as_tensor(old_ids) + num_known
    return perm


def permute_level(level: HierarchyLevel, perm: torch.Tensor) -> HierarchyLevel:
    """Relabel cluster c as ``perm[c]``, moving its centre and radius with it.

    SelEx's losses read labels only through equality and ``radii[labels]``, so a
    consistent relabelling leaves them unchanged (checked in tests/test_debgcd.py).
    """
    perm = perm.to(level.labels.device)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel(), device=perm.device)
    return dataclasses.replace(
        level, labels=perm[level.labels], centers=level.centers[inverse], radii=level.radii[inverse]
    )
