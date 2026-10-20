"""Hungarian clustering accuracy (SelEx project_utils/cluster_and_log_utils.py, spec §7)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class AccuracyReport:
    all: float
    known: float
    novel: float


def _contingency(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    size = int(max(y_pred.max(), y_true.max())) + 1
    w = np.zeros((size, size), dtype=np.int64)
    np.add.at(w, (y_pred, y_true), 1)
    return w


def cluster_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Best one-to-one cluster-to-class matching accuracy."""
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return float("nan")
    w = _contingency(y_true, y_pred)
    rows, cols = linear_sum_assignment(w.max() - w)
    return float(w[rows, cols].sum() / y_pred.size)


def split_cluster_acc_v1(y_true, y_pred, known_mask) -> AccuracyReport:
    """Separate Hungarian per subset; All is the subset-size-weighted mean (logged, not reported)."""
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    known_mask = np.asarray(known_mask, dtype=bool)
    weight = known_mask.mean()
    known = cluster_acc(y_true[known_mask], y_pred[known_mask])
    novel = cluster_acc(y_true[~known_mask], y_pred[~known_mask])
    return AccuracyReport(all=float(weight * known + (1 - weight) * novel), known=known, novel=novel)


def split_cluster_acc_v2(y_true, y_pred, known_mask) -> AccuracyReport:
    """One Hungarian assignment over all samples, then Known/Novel with that same mapping.

    This is the reported metric (``Train ACC Unlabelled_v2``).
    """
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    known_mask = np.asarray(known_mask, dtype=bool)
    w = _contingency(y_true, y_pred)
    rows, cols = linear_sum_assignment(w.max() - w)
    class_to_cluster = {int(c): int(r) for r, c in zip(rows, cols)}
    total = float(w[rows, cols].sum() / y_pred.size)

    def subset_acc(classes) -> float:
        hits = sum(w[class_to_cluster[c], c] for c in classes)
        count = sum(w[:, c].sum() for c in classes)
        return float(hits / count) if count else float("nan")

    known_classes = {int(c) for c in np.unique(y_true[known_mask])}
    novel_classes = {int(c) for c in np.unique(y_true[~known_mask])}
    return AccuracyReport(all=total, known=subset_acc(known_classes), novel=subset_acc(novel_classes))
