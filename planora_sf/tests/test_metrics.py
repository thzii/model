"""Hungarian accuracy v1/v2 (spec §7 steps 4-5)."""
import numpy as np
import pytest

from planora_sf.eval.metrics import cluster_acc, split_cluster_acc_v1, split_cluster_acc_v2


def test_perfect_clustering_under_permutation():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([2, 2, 0, 0, 1, 1])
    for fn in (split_cluster_acc_v1, split_cluster_acc_v2):
        report = fn(y, pred, y < 1)
        assert (report.all, report.known, report.novel) == (1.0, 1.0, 1.0)


def test_v2_uses_one_global_mapping_v1_does_not():
    y = np.array([0, 0, 0, 1, 1])
    pred = np.zeros(5, dtype=int)  # everything in one cluster
    v2 = split_cluster_acc_v2(y, pred, y < 1)
    assert (v2.all, v2.known, v2.novel) == pytest.approx((0.6, 1.0, 0.0))
    v1 = split_cluster_acc_v1(y, pred, y < 1)
    assert (v1.all, v1.known, v1.novel) == pytest.approx((1.0, 1.0, 1.0))


def test_cluster_acc_partial_match():
    assert cluster_acc(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])) == pytest.approx(0.75)
