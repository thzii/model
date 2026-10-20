"""BSSK / HSSK pseudo-labelling on synthetic blobs (spec §6 part A, §8 D-07..D-09, U-22)."""
import pytest
import torch

from helpers import make_blobs
from planora_sf.pseudo_labels import (
    PseudoLabelState,
    balance_assignments,
    build_hierarchy,
    cluster_radii,
    kmeans_plusplus,
    semi_supervised_kmeans,
)


def _blob_consistent(labels, blob_ids):
    return all(labels[blob_ids == b].unique().numel() == 1 for b in blob_ids.unique())


def test_kmeans_plusplus_picks_distinct_input_points(generator):
    x = torch.randn(20, 4)
    centres = kmeans_plusplus(x, 5, generator=generator)
    assert centres.shape == (5, 4)
    assert all((x == c).all(dim=1).any() for c in centres)
    assert torch.unique(centres, dim=0).shape[0] == 5
    assert kmeans_plusplus(x, 3, pre_centers=x[:2], generator=generator).shape == (3, 4)
    with pytest.raises(ValueError):
        kmeans_plusplus(x[:2], 3)


def test_balance_assignments_stable_matching_step():
    sq_dist = torch.tensor(
        [[0.0, 5, 9], [1, 5, 9], [2, 6, 1], [3, 1, 9], [4, 9, 2], [5, 3, 8]]
    )
    everyone_in_zero = torch.zeros(6, dtype=torch.long)
    out = balance_assignments(sq_dist, everyone_in_zero, torch.zeros(3, dtype=torch.long), cluster_size=2)
    assert out.tolist() == [0, 0, 2, 1, 2, 1]  # keep 2 nearest, free the rest to the nearest sparse cluster
    out = balance_assignments(sq_dist, everyone_in_zero, torch.tensor([1, 0, 0]), cluster_size=2)
    assert out.tolist()[0] == 0 and (out[1:] != 0).all()  # a labelled member uses up capacity


def test_ss_kmeans_recovers_blobs_and_keeps_labelled_fixed(generator):
    feats, blobs = make_blobs([15, 15, 15, 15])
    labelled = (blobs < 2) & (torch.arange(60) % 3 == 0)
    class_ids = torch.tensor([3, 7])  # non-contiguous known class ids
    targets = torch.where(labelled, class_ids[blobs.clamp(max=1)], torch.full_like(blobs, -1))
    res = semi_supervised_kmeans(feats, labelled, targets, 4, cluster_size=15, n_init=3, generator=generator)
    assert res.known_class_ids.tolist() == [3, 7] and res.num_known_clusters == 2
    assert torch.equal(res.labels[labelled], (targets[labelled] == 7).long())
    assert _blob_consistent(res.labels, blobs)
    assert res.labels[blobs == 0].unique().item() == 0 and res.labels[blobs == 1].unique().item() == 1
    assert set(res.labels[blobs >= 2].unique().tolist()) == {2, 3}
    assert torch.isfinite(res.centers).all() and res.inertia >= 0


def test_ss_kmeans_returns_labels_in_input_order(generator):
    feats, blobs = make_blobs([10, 10, 10, 10])
    labelled = (blobs < 2) & (torch.arange(40) % 2 == 0)
    targets = torch.where(labelled, blobs, torch.full_like(blobs, -1))
    perm = torch.randperm(40)
    res = semi_supervised_kmeans(feats[perm], labelled[perm], targets[perm], 4, generator=generator)
    assert _blob_consistent(res.labels, blobs[perm])
    assert torch.equal(res.labels[labelled[perm]], targets[perm][labelled[perm]])


def test_hierarchy_levels_cluster_counts_and_alignment(generator):
    known, novel = [10] * 8, [10] * 8
    feats, blobs = make_blobs(known + novel, dim=16)
    labelled = (blobs < 8) & (torch.arange(160) % 10 < 4)
    targets = torch.where(labelled, blobs, torch.full_like(blobs, -1))
    levels = build_hierarchy(feats, labelled, targets, 8, 8, generator=generator)
    assert len(levels) == 4  # 1 + floor(log2 8)
    assert [lvl.num_clusters for lvl in levels] == [16, 8, 4, 2]
    assert [lvl.num_known_clusters for lvl in levels] == [8, 4, 2, 1]
    for lvl in levels:
        assert lvl.labels.shape == (160,) and lvl.radii.shape == (lvl.num_clusters,)
        torch.testing.assert_close(lvl.radii, cluster_radii(lvl.centers))
        assert _blob_consistent(lvl.labels, blobs)  # aligned with the input order at every level
    assert (levels[-1].labels[blobs < 8] == 0).all() and (levels[-1].labels[blobs >= 8] == 1).all()  # seen vs unseen


def test_label_order_bug_reproduction_matches_official_ordering():
    feats_l, blobs_l = make_blobs([5, 5])  # labelled samples of known blobs A, B come first
    feats_u, blobs_u = make_blobs([10, 10, 15, 15])
    # Interleave the unlabelled blobs as A, C, B, D, A, C, ... (keys 4i, 4i+1, 4i+2, 4i+3).
    keys = torch.cat([torch.arange(10) * 4, torch.arange(10) * 4 + 2, torch.arange(15) * 4 + 1, torch.arange(15) * 4 + 3])
    order = torch.argsort(keys)
    feats = torch.cat([feats_l, feats_u[order]])
    blobs = torch.cat([blobs_l, blobs_u[order]])  # unlabelled part interleaves known and novel blobs
    labelled = torch.arange(60) < 10
    targets = torch.where(labelled, blobs, torch.full_like(blobs, -1))

    aligned = build_hierarchy(feats, labelled, targets, 2, 2, generator=torch.Generator().manual_seed(1))
    buggy = build_hierarchy(feats, labelled, targets, 2, 2, reproduce_label_order_bug=True,
                            generator=torch.Generator().manual_seed(1))
    assigned_known = aligned[0].labels < 2
    expected = torch.cat([aligned[1].labels[assigned_known], aligned[1].labels[~assigned_known]])
    assert torch.equal(buggy[1].labels, expected)
    assert torch.equal(buggy[0].labels, aligned[0].labels)  # level 0 is unaffected
    assert _blob_consistent(aligned[1].labels, blobs)
    assert not _blob_consistent(buggy[1].labels, blobs)  # D-08: unlabelled samples get other samples' labels


def test_cluster_radii_and_state_batching():
    radii = cluster_radii(torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]))
    torch.testing.assert_close(radii, torch.tensor([1.5, 1.5, 2.0]))
    assert cluster_radii(torch.zeros(1, 2)).tolist() == [0.0]
    feats, blobs = make_blobs([6, 6, 6, 6])
    labelled = blobs < 2
    state = PseudoLabelState(build_hierarchy(feats, labelled, torch.where(labelled, blobs, -1), 2, 2))
    batch = state.batch(torch.tensor([0, 7, 13]))
    assert state.num_levels == 2 and len(batch.level_labels) == 2 and batch.level_labels[0].shape == (3,)
    assert len(batch.repeat_views(2).level_labels[1]) == 6
