"""Hungarian v1/v2 against a brute-force oracle, plus their algebraic properties (plan T1, §3.2)."""
import itertools

import numpy as np
import pytest

from planora_sf.eval.metrics import cluster_acc, hungarian_mapping, split_cluster_acc_v1, split_cluster_acc_v2

CASES = 300


def _brute_force_acc(y: np.ndarray, pred: np.ndarray) -> float:
    """Best accuracy over every one-to-one cluster -> class map on the square label range."""
    size = int(max(y.max(), pred.max())) + 1
    return max(float((np.asarray(perm)[pred] == y).mean()) for perm in itertools.permutations(range(size)))


def _random_cases(seed: int = 0):
    rng = np.random.default_rng(seed)
    for _ in range(CASES):
        n_classes, n_clusters, n = int(rng.integers(2, 7)), int(rng.integers(1, 7)), int(rng.integers(2, 41))
        y = rng.integers(0, n_classes, n)
        pred = rng.integers(0, n_clusters, n)
        num_known = int(rng.integers(1, n_classes))
        yield y, pred, y < num_known


def test_cluster_acc_and_v2_all_match_brute_force():
    for y, pred, known in _random_cases():
        oracle = _brute_force_acc(y, pred)
        assert cluster_acc(y, pred) == pytest.approx(oracle)
        assert split_cluster_acc_v2(y, pred, known).all == pytest.approx(oracle)


def test_v1_dominates_v2_and_v2_all_is_the_weighted_split():
    checked = 0
    for y, pred, known in _random_cases(seed=1):
        if known.all() or not known.any():
            continue
        v1, v2 = split_cluster_acc_v1(y, pred, known), split_cluster_acc_v2(y, pred, known)
        assert v1.known >= v2.known - 1e-12 and v1.novel >= v2.novel - 1e-12
        w = known.mean()
        assert v2.all == pytest.approx(w * v2.known + (1 - w) * v2.novel)
        checked += 1
    assert checked > CASES // 2


def _v2_pre_refactor(y_true, y_pred, known_mask):
    """split_cluster_acc_v2 exactly as it was before hungarian_mapping() was extracted (commit 6e8b3c4)."""
    from scipy.optimize import linear_sum_assignment

    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    known_mask = np.asarray(known_mask, dtype=bool)
    size = int(max(y_pred.max(), y_true.max())) + 1
    w = np.zeros((size, size), dtype=np.int64)
    np.add.at(w, (y_pred, y_true), 1)
    rows, cols = linear_sum_assignment(w.max() - w)
    class_to_cluster = {int(c): int(r) for r, c in zip(rows, cols)}
    total = float(w[rows, cols].sum() / y_pred.size)

    def subset_acc(classes) -> float:
        hits = sum(w[class_to_cluster[c], c] for c in classes)
        count = sum(w[:, c].sum() for c in classes)
        return float(hits / count) if count else float("nan")

    known_classes = {int(c) for c in np.unique(y_true[known_mask])}
    novel_classes = {int(c) for c in np.unique(y_true[~known_mask])}
    return total, subset_acc(known_classes), subset_acc(novel_classes)


EDGE_CASES = [
    ([2, 2, 3], [0, 1, 1], [False] * 3),  # empty Known subset -> NaN
    ([0, 1, 1], [0, 0, 1], [True] * 3),  # empty Novel subset -> NaN
    ([0], [5], [True]),  # single sample
    ([0, 0, 1, 1], [7, 3, 3, 9], [True, True, False, False]),  # more clusters than classes
    ([0, 1, 2, 3], [0, 0, 0, 0], [True, True, False, False]),  # one cluster for everything
    ([0, 50, 50], [1, 1, 2], [True, False, False]),  # sparse label ids
]


@pytest.mark.parametrize("y, pred, known", EDGE_CASES + [tuple(map(list, c)) for c in list(_random_cases(seed=3))[:50]])
def test_v2_identical_to_pre_refactor_implementation(y, pred, known):
    new = split_cluster_acc_v2(np.array(y), np.array(pred), np.array(known))
    np.testing.assert_array_equal([new.all, new.known, new.novel], _v2_pre_refactor(y, pred, known))  # NaN == NaN here


def test_empty_input_still_raises_for_v2():
    empty = np.array([], dtype=np.int64)
    with pytest.raises(ValueError):
        split_cluster_acc_v2(empty, empty, empty.astype(bool))


def test_hungarian_mapping_is_one_to_one_and_achieves_v2_all():
    for y, pred, known in _random_cases(seed=2):
        mapping = hungarian_mapping(y, pred)
        assert len(set(mapping.values())) == len(mapping)
        hits = sum(int(((y == c) & (pred == k)).sum()) for c, k in mapping.items())
        assert hits / y.size == pytest.approx(split_cluster_acc_v2(y, pred, known).all)
