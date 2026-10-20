"""GCD split protocol: exact counts, partition properties, and RNG parity with SelEx/GCD."""
import numpy as np
import pytest

from planora_sf.data.splits import cub_ssb_classes, make_gcd_split

COUNTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # 10 classes, 95 images, uneven on purpose
TARGETS = np.repeat(np.arange(10), COUNTS)


def test_global_split_exact_counts_and_partition():
    split = make_gcd_split(TARGETS)
    assert split.known_classes == (0, 1, 2, 3, 4) and split.novel_classes == (5, 6, 7, 8, 9)
    n_known_images = sum(COUNTS[:5])  # 35
    assert split.labelled_indices.size == int(0.5 * n_known_images) == 17
    assert np.isin(TARGETS[split.labelled_indices], split.known_classes).all()
    assert np.intersect1d(split.labelled_indices, split.unlabelled_indices).size == 0
    assert np.array_equal(np.union1d(split.labelled_indices, split.unlabelled_indices), np.arange(95))
    assert split.summary(TARGETS) == {
        "known_classes": 5,
        "novel_classes": 5,
        "labelled_images": 17,
        "unlabelled_images": 78,
        "unlabelled_known_images": 35 - 17,
        "unlabelled_novel_images": sum(COUNTS[5:]),  # every novel image is unlabelled
    }


def test_labelled_draw_matches_selex_subsample_instances():
    # SelEx/GCD data_utils.subsample_instances: np.random.seed(0); np.random.choice(range(n), replace=False, size=int(0.5 n))
    known_idx = np.nonzero(np.isin(TARGETS, range(5)))[0]
    np.random.seed(0)
    picks = np.random.choice(range(len(known_idx)), replace=False, size=int(0.5 * len(known_idx)))
    assert np.array_equal(make_gcd_split(TARGETS).labelled_indices, np.sort(known_idx[picks]))


def test_stratified_split_counts_per_class():
    split = make_gcd_split(TARGETS, stratified=True)
    per_class = np.bincount(TARGETS[split.labelled_indices], minlength=10)
    assert per_class.tolist() == [int(0.5 * c) for c in COUNTS[:5]] + [0] * 5  # 2, 3, 3, 4, 4


def test_custom_known_classes_define_target_order():
    split = make_gcd_split(TARGETS, known_classes=[7, 2, 9])
    assert split.class_to_target == {7: 0, 2: 1, 9: 2, 0: 3, 1: 4, 3: 5, 4: 6, 5: 7, 6: 8, 8: 9}
    mapped = split.map_targets(TARGETS[split.labelled_indices])
    assert set(mapped.tolist()) <= {0, 1, 2}  # labelled samples carry known targets 0..K_s-1


def test_seed_controls_the_draw():
    a, b = make_gcd_split(TARGETS, seed=0), make_gcd_split(TARGETS, seed=0)
    assert np.array_equal(a.labelled_indices, b.labelled_indices)
    assert not np.array_equal(a.labelled_indices, make_gcd_split(TARGETS, seed=1).labelled_indices)


def test_cub_ssb_class_lists():
    known, novel = cub_ssb_classes()
    assert len(known) == len(novel) == 100 and set(known).isdisjoint(novel)
    assert sorted(known + novel) == list(range(200))
    assert known[:3] == [150, 70, 34]  # first entries of the SelEx/SSB pickle


def test_cub_scale_split_matches_selex_statistics():
    targets = np.repeat(np.arange(200), 30)  # CUB has ~30 training images per class (5994 in total)
    known, novel = cub_ssb_classes()
    split = make_gcd_split(targets, known_classes=known, novel_classes=novel)
    assert split.labelled_indices.size == 1500 and split.unlabelled_indices.size == 4500  # SX-S Table: 1.5K / 4.5K
    assert split.num_known == 100 and split.num_novel == 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"known_classes": [42]},
        {"known_classes": [1, 1]},
        {"labelled_fraction": 0.0},
        {"known_class_fraction": 1.5},
        {"known_classes": [0, 1], "novel_classes": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
    ],
)
def test_invalid_split_arguments(kwargs):
    with pytest.raises(ValueError):
        make_gcd_split(TARGETS, **kwargs)
