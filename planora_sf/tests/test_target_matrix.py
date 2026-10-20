"""USE target matrices (spec §5.2, §8 D-02..D-04, U-21)."""
import pytest
import torch

from planora_sf.pseudo_labels.hssk import PseudoLabelBatch
from planora_sf.pseudo_labels.target_matrix import (
    build_use_targets,
    use_targets_compat,
    use_targets_paper,
    use_targets_radius,
)


def reference_compat(feats, level_labels, radii):
    """Literal port of the SelEx code (ct.py:288-347, strategy='zero_one', distance='euclidean')."""
    pair_dist = torch.cdist(feats, feats)
    confusion_factor = 0
    for i in range(len(level_labels)):
        confusion_factor += (pair_dist > 2 * radii[i][level_labels[i]]).float() / 2**i
    confusion_factor = (confusion_factor - confusion_factor.min()) / (
        confusion_factor.max() - confusion_factor.min() + 0.0000001
    )
    return confusion_factor / confusion_factor.sum(dim=1)


def test_paper_targets_hand_values():
    y = use_targets_paper([torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 0, 1])])
    # raw row 0: [0, 0, 1/2, 1/2 + 1/4] -> normalised by 1.25
    torch.testing.assert_close(y[0], torch.tensor([0.0, 0.0, 0.4, 0.6]))
    torch.testing.assert_close(y.sum(1), torch.ones(4))
    assert y[0, 1] == 0 and y[2, 3] > 0  # same cluster at every level -> strict negative (target 0)


def test_paper_targets_zero_rows_stay_zero():
    assert torch.equal(use_targets_paper([torch.zeros(5, dtype=torch.long)]), torch.zeros(5, 5))
    with pytest.raises(ValueError):
        use_targets_paper([])


def test_compat_matches_literal_selex_code_including_broadcast_quirk():
    feats = torch.randn(8, 5) * 3
    labels = [torch.randint(0, 3, (8,)), torch.randint(0, 2, (8,))]
    radii = [torch.rand(3) + 0.5, torch.rand(2) + 0.5]
    torch.testing.assert_close(use_targets_compat(feats, labels, radii), reference_compat(feats, labels, radii))

    # Points on a line with radius 1: the indicator d > 2 differs per row, so the quirk is visible.
    line = torch.tensor([0.0, 0.1, 0.2, 5.0, 5.1, 10.0, 20.0, 30.0])[:, None]
    single = [torch.zeros(8, dtype=torch.long)], [torch.ones(1)]
    y = use_targets_compat(line, *single)
    torch.testing.assert_close(y, reference_compat(line, *single))
    indicator = (torch.cdist(line, line) > 2).float()
    torch.testing.assert_close(y, indicator / indicator.sum(dim=1)[None, :])  # D-04: column b / row-sum b
    assert not torch.allclose(y.sum(dim=1), torch.ones(8))


def test_radius_targets_rows_are_distributions():
    feats = torch.randn(6, 4)
    y = use_targets_radius(feats, [torch.tensor([0, 0, 1, 1, 2, 2])], [torch.full((3,), 0.5)])
    torch.testing.assert_close(y.sum(1), torch.ones(6))
    assert (torch.diagonal(y) > 0).all()


def test_dispatch_and_view_tiling():
    pseudo = PseudoLabelBatch([torch.tensor([0, 1, 1])], [torch.ones(2)]).repeat_views(2)
    assert pseudo.level_labels[0].tolist() == [0, 1, 1, 0, 1, 1]
    paper = build_use_targets("paper", pseudo)
    assert paper.shape == (6, 6) and paper[0, 3] == 0  # the two views of one image are never negatives
    assert build_use_targets("compat", pseudo, backbone_feats=torch.randn(6, 4)).shape == (6, 6)
    with pytest.raises(ValueError):
        build_use_targets("compat", pseudo)
    with pytest.raises(ValueError):
        build_use_targets("bogus", pseudo, backbone_feats=torch.randn(6, 4))
