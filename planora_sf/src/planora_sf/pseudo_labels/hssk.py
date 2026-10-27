"""Hierarchical semi-supervised k-means: HSSK (SX-P §4, SX-S Alg. 5, SelEx ct.py:502-592).

Level 0 is BSSK with K = n_known + n_novel. Each further level halves the
known and novel counts, groups the level-0 known prototypes into hyper-classes
(as the SelEx code does, D-07), treats every sample currently assigned to a
known cluster as labelled with its hyper-class, and reruns BSSK. That gives
L = 1 + floor(log2 n_known) levels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from planora_sf.pseudo_labels.bssk import semi_supervised_kmeans


@dataclass
class HierarchyLevel:
    labels: torch.Tensor  # [N] pseudo-labels in input order (unless the D-08 bug is reproduced)
    centers: torch.Tensor  # [K, F]
    radii: torch.Tensor  # [K] ½ · min distance to another centre (ct.py:262)
    num_known_clusters: int

    @property
    def num_clusters(self) -> int:
        return int(self.centers.shape[0])


@dataclass
class PseudoLabelBatch:
    level_labels: list[torch.Tensor]  # one [B] tensor per level
    radii: list[torch.Tensor]  # one [K_i] tensor per level
    centers: Optional[torch.Tensor] = None  # [K, F] level-0 centres (DebGCD centroid posterior)
    num_known_clusters: Optional[int] = None  # level-0 clusters anchored to known classes

    def repeat_views(self, n_views: int) -> "PseudoLabelBatch":
        """Tile labels for ``n_views`` stacked views (rows ordered view-major)."""
        return PseudoLabelBatch(
            [labels.repeat(n_views) for labels in self.level_labels], list(self.radii), self.centers,
            self.num_known_clusters,
        )

    def to(self, device) -> "PseudoLabelBatch":
        return PseudoLabelBatch(
            [labels.to(device) for labels in self.level_labels], [r.to(device) for r in self.radii],
            None if self.centers is None else self.centers.to(device), self.num_known_clusters,
        )


@dataclass
class PseudoLabelState:
    levels: list[HierarchyLevel]

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def batch(self, indices: torch.Tensor) -> PseudoLabelBatch:
        base = self.levels[0]
        return PseudoLabelBatch(
            [lvl.labels[indices] for lvl in self.levels], [lvl.radii for lvl in self.levels], base.centers,
            base.num_known_clusters,
        )


def cluster_radii(centers: torch.Tensor) -> torch.Tensor:
    """r_c = ½ min_{c' != c} ||μ_c - μ_c'|| (ct.py:256-262); 0 for a single cluster."""
    if centers.shape[0] < 2:
        return centers.new_zeros(centers.shape[0])
    dist = torch.cdist(centers, centers)
    dist.fill_diagonal_(float("inf"))
    return dist.min(dim=1).values / 2


def build_hierarchy(
    feats: torch.Tensor,
    labelled_mask: torch.Tensor,
    targets: torch.Tensor,
    n_known: int,
    n_novel: int,
    *,
    balanced: bool = True,
    max_iter: int = 10,
    n_init: int = 1,
    reproduce_label_order_bug: bool = False,
    generator: Optional[torch.Generator] = None,
) -> list[HierarchyLevel]:
    """Pseudo-labels for every hierarchy level.

    ``reproduce_label_order_bug=True`` reorders levels ≥ 1 the way the official
    SelEx code consumes them (D-08, U-22); the default keeps every level aligned
    with the input order.
    """
    n = feats.shape[0]
    labelled_mask = labelled_mask.to(torch.bool)

    def size_cap(k: int) -> Optional[int]:
        return math.ceil(n / k) if balanced else None

    base = semi_supervised_kmeans(
        feats, labelled_mask, targets, n_known + n_novel,
        cluster_size=size_cap(n_known + n_novel), max_iter=max_iter, n_init=n_init, generator=generator,
    )
    n_known_eff = base.num_known_clusters
    levels = [HierarchyLevel(base.labels, base.centers, cluster_radii(base.centers), n_known_eff)]

    known_protos = base.centers[:n_known_eff]
    assigned_known = base.labels < n_known_eff
    no_labels = torch.zeros(n_known_eff, dtype=torch.bool, device=feats.device)
    dummy_targets = torch.zeros(n_known_eff, dtype=torch.long, device=feats.device)
    n_s, n_n = n_known_eff, n_novel
    while n_s > 1:
        n_s, n_n = max(n_s // 2, 1), max(n_n // 2, 1)
        groups = semi_supervised_kmeans(
            known_protos, no_labels, dummy_targets, n_s, max_iter=max_iter, n_init=n_init, generator=generator
        ).labels
        hyper_targets = torch.full((n,), -1, dtype=torch.long, device=feats.device)
        hyper_targets[assigned_known] = groups[base.labels[assigned_known]]
        level = semi_supervised_kmeans(
            feats, assigned_known, hyper_targets, n_s + n_n,
            cluster_size=size_cap(n_s + n_n), max_iter=max_iter, n_init=n_init, generator=generator,
        )
        labels = level.labels
        if reproduce_label_order_bug:
            labels = torch.cat([labels[assigned_known], labels[~assigned_known]])
        levels.append(HierarchyLevel(labels, level.centers, cluster_radii(level.centers), level.num_known_clusters))
    return levels
