"""Evaluator diagnostics on hand-built cases with known answers (plan T1)."""
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from helpers import make_blobs
from planora_sf.config import EvalConfig
from planora_sf.data.splits import cub_ssb_classes, cub_ssb_novel_tiers
from planora_sf.eval import GCDEvaluator
from planora_sf.eval.diagnostics import cluster_structure, novel_accuracy_by_tier, per_class_accuracy_v2, routing_matrix
from planora_sf.eval.metrics import split_cluster_acc_v1, split_cluster_acc_v2

# Classes 0, 1 known; 2, 3 novel. Clusters 0, 1 are known-anchored.
Y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
C = np.array([0, 0, 1, 2, 0, 2, 3, 3])


def test_routing_matrix():
    r = routing_matrix(Y, C, num_known=2, num_known_clusters=2)
    assert (r["known_in_known_clusters"], r["known_in_novel_clusters"]) == (3, 1)
    assert (r["novel_in_known_clusters"], r["novel_in_novel_clusters"]) == (1, 3)
    assert r["absorption"] == pytest.approx(0.25) and r["leakage"] == pytest.approx(0.25)


def test_routing_matrix_empty_subset_gives_none():
    r = routing_matrix(np.array([0, 1]), np.array([0, 1]), num_known=2, num_known_clusters=2)
    assert r["absorption"] is None and r["leakage"] == 0.0


def test_per_class_accuracy_uses_the_global_matching():
    # The unique optimum maps class 0 -> cluster 1 and class 1 -> cluster 0.
    acc = per_class_accuracy_v2(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 1]))
    assert acc == {0: 1.0, 1: 0.5}


def test_cluster_structure():
    s = cluster_structure(Y, C, num_known=2, num_known_clusters=2)
    # Cluster majorities: 0 -> class 0 (2 of 3), 1 -> 1, 2 -> tie {1, 2} -> 1, 3 -> class 3 (2 of 2).
    assert s["purity"] == pytest.approx(6 / 8)
    assert s["novel_classes_present"] == 2 and s["unclaimed_novel_classes"] == 1  # class 2
    assert s["sizes_all"]["histogram"] == {"1": 1, "2": 2, "3": 1}
    assert s["sizes_known_anchored"]["num_clusters"] == 2 and s["sizes_known_anchored"]["max"] == 3
    assert s["sizes_novel"]["histogram"] == {"2": 2}
    known = Y < 2
    v1, v2 = split_cluster_acc_v1(Y, C, known), split_cluster_acc_v2(Y, C, known)
    assert s["v1_minus_v2"] == pytest.approx({k: getattr(v1, k) - getattr(v2, k) for k in ("all", "known", "novel")})
    assert all(d >= 0 for d in s["v1_minus_v2"].values())


def test_novel_accuracy_by_tier():
    y = np.array([2, 2, 2, 3, 4, 4])
    pred = np.array([2, 2, 0, 3, 4, 4])  # class 2 is 2/3 right, classes 3 and 4 fully right
    t = novel_accuracy_by_tier(y, pred, {"A": [2, 3], "B": [4], "C": [9]})
    assert t["A"] == {"num_classes": 2, "num_samples": 4, "sample_weighted": 0.75,
                      "class_mean": pytest.approx((2 / 3 + 1) / 2)}
    assert t["B"]["sample_weighted"] == 1.0 and t["B"]["class_mean"] == 1.0
    assert t["C"] == {"num_classes": 0, "num_samples": 0, "sample_weighted": None, "class_mean": None}


def test_cub_ssb_novel_tiers_partition_the_novel_classes():
    tiers = cub_ssb_novel_tiers()
    assert list(tiers) == ["Hard", "Medium", "Easy"]
    ids = [c for tier in tiers.values() for c in tier]
    assert len(ids) == len(set(ids))
    _, novel = cub_ssb_classes()
    assert ids == novel


def test_score_detailed_extends_score_without_changing_it(generator):
    feats, blobs = make_blobs([12, 12, 12, 12])  # blobs 0, 1 known; 2, 3 novel
    labelled = (blobs < 2) & (torch.arange(48) % 2 == 0)
    targets = torch.where(labelled, blobs, torch.full_like(blobs, -1))
    evaluator = GCDEvaluator(2, 4, EvalConfig(n_init=3))
    pred = evaluator.predict(feats, labelled, targets, generator=generator)
    detailed = evaluator.score_detailed(pred, blobs, ~labelled, tiers={"novel": [2, 3]})
    assert set(detailed) == {"v1", "v2", "routing", "per_class_v2", "structure", "tiers"}
    assert {k: detailed[k] for k in ("v1", "v2")} == {k: asdict(v) for k, v in evaluator.score(pred, blobs, ~labelled).items()}
    assert detailed["routing"]["absorption"] == 0.0 and detailed["tiers"]["novel"]["sample_weighted"] == 1.0
    json.dumps(detailed)  # JSON-serialisable
    assert evaluator.score_detailed(pred, blobs, ~labelled)["tiers"] is None


def test_empty_subset_gives_null_not_nan_and_strict_json():
    y, pred = np.array([2, 2, 3]), np.array([0, 1, 1])  # no GT-known sample
    out = {
        "routing": routing_matrix(y, pred, 2, 2),
        "per_class": per_class_accuracy_v2(y, pred),
        "structure": cluster_structure(y, pred, 2, 2),
        "tiers": novel_accuracy_by_tier(y, pred, {"t": [2, 3], "absent": [9]}),
    }
    assert out["structure"]["v1_minus_v2"]["known"] is None and out["routing"]["leakage"] is None
    json.dumps(out, allow_nan=False)  # raises on NaN/inf


def test_unlabelled_ground_truth_cannot_change_predictions(generator):
    """predict() sees GT only through labelled entries: rewriting unlabelled targets changes nothing."""
    feats, blobs = make_blobs([10, 10, 10, 10])
    labelled = (blobs < 2) & (torch.arange(40) % 2 == 0)
    hidden = torch.where(labelled, blobs, torch.full_like(blobs, -1))
    leaked = torch.where(labelled, blobs, blobs.flip(0))  # true-looking labels on unlabelled samples
    evaluator = GCDEvaluator(2, 4, EvalConfig(n_init=3))
    a = evaluator.predict(feats, labelled, hidden, generator=torch.Generator().manual_seed(5))
    b = evaluator.predict(feats, labelled, leaked, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a.cluster_ids, b.cluster_ids)


def test_eval_seed_rejects_negatives():
    assert EvalConfig().seed == 0
    with pytest.raises(ValueError):
        EvalConfig(seed=-1)
