"""(Balanced) semi-supervised k-means: BSSK (SX-P §4, SX-S Alg. 2-4, SelEx fmk.py:174-303).

Known-class centres start at the labelled class means; novel centres are seeded
with k-means++ on the unlabelled samples. Phase 1 keeps the known centres fixed
and, when ``cluster_size`` is set, balances cluster sizes (SX-S Alg. 3). Phase 2
is plain semi-supervised k-means over all centres. Labelled samples never move.

Unlike the SelEx code, labels are always returned in **input order** (D-08).
The exclusion step of SX-S Alg. 1 is not implemented, matching the code's
effective behaviour (D-09).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch


@dataclass
class KMeansResult:
    labels: torch.Tensor  # [N] cluster index per input sample, in input order
    centers: torch.Tensor  # [K, F]
    inertia: float
    known_class_ids: torch.Tensor  # [n_known]: cluster j < n_known <-> original class id known_class_ids[j]

    @property
    def num_known_clusters(self) -> int:
        return int(self.known_class_ids.numel())


def kmeans_plusplus(
    x: torch.Tensor,
    n_centers: int,
    *,
    pre_centers: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """k-means++ seeding of ``n_centers`` new centres from ``x``, given optional fixed ``pre_centers``."""
    if n_centers <= 0:
        return x.new_empty(0, x.shape[1])
    if x.shape[0] < n_centers:
        raise ValueError(f"need at least {n_centers} samples to seed {n_centers} centres, got {x.shape[0]}")
    chosen: list[torch.Tensor] = []
    current = pre_centers if pre_centers is not None and pre_centers.shape[0] > 0 else None
    if current is None:
        first = int(torch.randint(x.shape[0], (1,), generator=generator))
        chosen.append(x[first])
        current = x[first : first + 1]
    while len(chosen) < n_centers:
        sq_dist = torch.cdist(x, current).pow(2).min(dim=1).values.to(torch.float64)
        total = sq_dist.sum()
        probs = sq_dist / total if total > 0 else torch.full_like(sq_dist, 1.0 / sq_dist.numel())
        index = int(torch.multinomial(probs.cpu(), 1, generator=generator))
        chosen.append(x[index])
        current = torch.cat([current, x[index : index + 1]])
    return torch.stack(chosen)


def balance_assignments(
    sq_dist: torch.Tensor, assign: torch.Tensor, labelled_counts: torch.Tensor, cluster_size: int
) -> torch.Tensor:
    """Stable-matching balancing of unlabelled assignments (SX-S Alg. 3).

    Each cluster may take ``cluster_size - labelled_count`` unlabelled samples.
    Over-full clusters keep their nearest samples and free the rest; freed samples
    move to their nearest sparse cluster (one with spare capacity).
    """
    assign = assign.clone()
    capacity = (cluster_size - labelled_counts).clamp_min(0)
    free = torch.zeros_like(assign, dtype=torch.bool)
    sparse = torch.zeros(sq_dist.shape[1], dtype=torch.bool, device=assign.device)
    for k in range(sq_dist.shape[1]):
        members = torch.nonzero(assign == k).flatten()
        cap = int(capacity[k])
        if members.numel() > cap:
            order = sq_dist[members, k].argsort()
            free[members[order[cap:]]] = True
        elif members.numel() < cap:
            sparse[k] = True
    if free.any() and sparse.any():
        sparse_ids = torch.nonzero(sparse).flatten()
        nearest = sq_dist[free][:, sparse_ids].argmin(dim=1)
        assign[free] = sparse_ids[nearest]
    return assign


def _update_centers(feats: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor, cluster_ids: Iterable[int]) -> None:
    for k in cluster_ids:
        members = labels == k
        if members.any():  # empty clusters keep their previous centre
            centers[k] = feats[members].mean(dim=0)


def _fit_once(
    feats: torch.Tensor,
    labelled: torch.Tensor,
    labelled_assign: torch.Tensor,
    n_known: int,
    n_clusters: int,
    cluster_size: Optional[int],
    max_iter: int,
    tol: float,
    generator: Optional[torch.Generator],
) -> tuple[torch.Tensor, torch.Tensor, float]:
    l_idx = torch.nonzero(labelled).flatten()
    u_idx = torch.nonzero(~labelled).flatten()
    u_feats = feats[u_idx]
    centers = feats.new_zeros(n_clusters, feats.shape[1])
    for j in range(n_known):
        centers[j] = feats[l_idx[labelled_assign == j]].mean(dim=0)
    centers[n_known:] = kmeans_plusplus(
        u_feats, n_clusters - n_known, pre_centers=centers[:n_known] if n_known else None, generator=generator
    )
    labels = torch.empty(feats.shape[0], dtype=torch.long, device=feats.device)
    labels[l_idx] = labelled_assign
    labelled_counts = torch.bincount(labelled_assign, minlength=n_clusters)

    # Phase 1: known centres fixed, optional balancing (SX-S Alg. 2-3).
    for _ in range(max_iter):
        previous = centers.clone()
        if u_idx.numel():
            sq_dist = torch.cdist(u_feats, centers).pow(2)
            u_assign = sq_dist.argmin(dim=1)
            if cluster_size is not None:
                u_assign = balance_assignments(sq_dist, u_assign, labelled_counts, cluster_size)
            labels[u_idx] = u_assign
        _update_centers(feats, labels, centers, range(n_known, n_clusters))
        if (centers - previous).norm(dim=1).sum().pow(2) < tol:
            break

    # Phase 2: unconstrained semi-supervised k-means over all centres (SX-S Alg. 4).
    for _ in range(max_iter):
        previous = centers.clone()
        if u_idx.numel():
            labels[u_idx] = torch.cdist(u_feats, centers).argmin(dim=1)
        _update_centers(feats, labels, centers, range(n_clusters))
        if (centers - previous).norm(dim=1).sum().pow(2) < tol:
            break

    inertia = float((feats - centers[labels]).pow(2).sum())
    return labels, centers, inertia


def semi_supervised_kmeans(
    feats: torch.Tensor,
    labelled_mask: torch.Tensor,
    targets: torch.Tensor,
    n_clusters: int,
    *,
    cluster_size: Optional[int] = None,
    max_iter: int = 10,
    n_init: int = 1,
    tol: float = 1e-4,
    generator: Optional[torch.Generator] = None,
) -> KMeansResult:
    """Cluster ``feats [N, F]`` into ``n_clusters`` with labelled samples fixed to their class.

    ``targets`` is only read where ``labelled_mask`` is true. The distinct labelled
    classes become clusters ``0 .. n_known - 1`` in sorted class-id order.
    ``cluster_size=None`` disables balancing (unbalanced mode, U-30).
    """
    feats = feats.to(torch.float32)
    labelled = labelled_mask.to(torch.bool)
    known_ids = torch.unique(targets[labelled])
    n_known = int(known_ids.numel())
    if n_known > n_clusters:
        raise ValueError(f"{n_known} labelled classes do not fit into {n_clusters} clusters")
    labelled_assign = torch.searchsorted(known_ids, targets[labelled])
    best: Optional[tuple[torch.Tensor, torch.Tensor, float]] = None
    for _ in range(max(n_init, 1)):
        result = _fit_once(
            feats, labelled, labelled_assign, n_known, n_clusters, cluster_size, max_iter, tol, generator
        )
        if best is None or result[2] < best[2]:
            best = result
    labels, centers, inertia = best
    return KMeansResult(labels=labels, centers=centers, inertia=inertia, known_class_ids=known_ids)
