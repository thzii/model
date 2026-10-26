"""Evaluator diagnostics: where Known/Novel accuracy is gained or lost (plan T1, §3.4 B6-B8).

Pure functions on 1-D integer arrays that the caller has already restricted to the
scored (unlabelled) samples. Targets follow the SelEx transform: classes
``0 .. num_known - 1`` are known. Clusters ``0 .. num_known_clusters - 1`` are anchored
to known classes by semi-supervised k-means. Return values are plain Python types that
pass ``json.dumps(..., allow_nan=False)``: a quantity with an empty denominator is ``None``.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np

from planora_sf.eval.metrics import hungarian_mapping, split_cluster_acc_v1, split_cluster_acc_v2


def _as_int(a) -> np.ndarray:
    return np.asarray(a, dtype=np.int64)


def _share(num: int, den: int) -> Optional[float]:
    return float(num / den) if den else None


def _finite_or_none(x: float) -> Optional[float]:
    """v1/v2 report NaN for an empty subset; diagnostics report None instead."""
    return float(x) if math.isfinite(x) else None


def routing_matrix(y_true, cluster_ids, num_known: int, num_known_clusters: int) -> dict:
    """Counts of {GT known, GT novel} x {known-anchored cluster, novel cluster}.

    ``absorption`` is the share of GT-novel samples in known-anchored clusters;
    ``leakage`` the share of GT-known samples in novel clusters.
    """
    y, c = _as_int(y_true), _as_int(cluster_ids)
    gt_known, in_known = y < num_known, c < num_known_clusters
    counts = {
        "known_in_known_clusters": int((gt_known & in_known).sum()),
        "known_in_novel_clusters": int((gt_known & ~in_known).sum()),
        "novel_in_known_clusters": int((~gt_known & in_known).sum()),
        "novel_in_novel_clusters": int((~gt_known & ~in_known).sum()),
    }
    return {
        **counts,
        "absorption": _share(counts["novel_in_known_clusters"], int((~gt_known).sum())),
        "leakage": _share(counts["known_in_novel_clusters"], int(gt_known.sum())),
    }


def per_class_accuracy_v2(y_true, y_pred) -> dict[int, float]:
    """Accuracy of each GT class present, under the single global matching of v2."""
    y, p = _as_int(y_true), _as_int(y_pred)
    if y.size == 0:
        return {}
    mapping = hungarian_mapping(y, p)
    return {int(c): float((p[y == c] == mapping[int(c)]).mean()) for c in np.unique(y)}


def _size_summary(sizes: Sequence[int]) -> dict:
    sizes = sorted(int(s) for s in sizes)
    values, counts = np.unique(sizes, return_counts=True) if sizes else ([], [])
    return {
        "num_clusters": len(sizes),
        "min": sizes[0] if sizes else None,
        "median": float(np.median(sizes)) if sizes else None,
        "max": sizes[-1] if sizes else None,
        "histogram": {str(int(v)): int(n) for v, n in zip(values, counts)},  # cluster size -> number of clusters
    }


def cluster_structure(y_true, y_pred, num_known: int, num_known_clusters: int) -> dict:
    """Cluster sizes, purity, unclaimed novel classes and the v1 - v2 deltas.

    A cluster's majority class is its most frequent GT class (ties go to the smaller
    class id). ``unclaimed_novel_classes`` counts GT-novel classes that are the majority
    class of no cluster. A ``v1_minus_v2`` entry is ``None`` when either metric is undefined
    (an empty Known or Novel subset).
    """
    y, p = _as_int(y_true), _as_int(y_pred)
    clusters, sizes = np.unique(p, return_counts=True)
    majority, hits = [], 0
    for k in clusters:
        classes, counts = np.unique(y[p == k], return_counts=True)
        majority.append(int(classes[counts.argmax()]))
        hits += int(counts.max())
    novel_present = {int(c) for c in np.unique(y[y >= num_known])}
    known_mask = y < num_known
    v1, v2 = split_cluster_acc_v1(y, p, known_mask), split_cluster_acc_v2(y, p, known_mask)
    anchored = clusters < num_known_clusters
    return {
        "sizes_all": _size_summary(sizes),
        "sizes_known_anchored": _size_summary(sizes[anchored]),
        "sizes_novel": _size_summary(sizes[~anchored]),
        "purity": _share(hits, int(y.size)),  # size-weighted mean majority-class share
        "novel_classes_present": len(novel_present),
        "unclaimed_novel_classes": len(novel_present - set(majority)),
        "v1_minus_v2": {
            name: _finite_or_none(getattr(v1, name) - getattr(v2, name)) for name in ("all", "known", "novel")
        },
    }


def novel_accuracy_by_tier(y_true, y_pred, tiers: Mapping[str, Sequence[int]]) -> dict:
    """Sample-weighted and class-mean accuracy per tier (name -> target ids) under the v2 matching."""
    y, p = _as_int(y_true), _as_int(y_pred)
    mapping = hungarian_mapping(y, p) if y.size else {}
    out = {}
    for name, class_ids in tiers.items():
        present = [int(c) for c in class_ids if (y == int(c)).any()]
        counts = [int((y == c).sum()) for c in present]
        hits = [int((p[y == c] == mapping[c]).sum()) for c in present]
        out[name] = {
            "num_classes": len(present),
            "num_samples": sum(counts),
            "sample_weighted": _share(sum(hits), sum(counts)),
            "class_mean": float(np.mean([h / n for h, n in zip(hits, counts)])) if present else None,
        }
    return out
