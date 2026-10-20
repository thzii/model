"""GCD evaluator API: KNOWN / NOVEL_CLUSTER assignment and scoring on synthetic blobs (spec §7)."""
import pytest
import torch

from helpers import make_blobs
from planora_sf.config import EvalConfig
from planora_sf.eval import AssignmentKind, GCDEvaluator, extract_features
from planora_sf.models import BackboneAdapter, TinyViT


@pytest.mark.parametrize("clustering", ["bssk", "ss_kmeans"])
def test_predict_assigns_known_and_novel_clusters(clustering, generator):
    feats, blobs = make_blobs([12, 12, 12, 12])  # blobs 0, 1 known; 2, 3 novel
    labelled = (blobs < 2) & (torch.arange(48) % 2 == 0)
    targets = torch.where(labelled, blobs, torch.full_like(blobs, -1))
    evaluator = GCDEvaluator(2, 4, EvalConfig(clustering=clustering, n_init=3))
    pred = evaluator.predict(feats, labelled, targets, generator=generator)

    assert pred.cluster_ids.shape == (48,) and pred.num_known_clusters == 2
    kinds = pred.kinds
    assert all(kinds[i] is AssignmentKind.KNOWN for i in torch.nonzero(blobs < 2).flatten().tolist())
    assert all(kinds[i] is AssignmentKind.NOVEL_CLUSTER for i in torch.nonzero(blobs >= 2).flatten().tolist())
    assert torch.equal(pred.known_class_ids[blobs < 2], blobs[blobs < 2])
    assert (pred.known_class_ids[blobs >= 2] == -1).all()

    scores = evaluator.score(pred, blobs, ~labelled)
    for report in scores.values():  # separable synthetic blobs: a mechanics check, not a performance claim
        assert report.all == report.known == report.novel == 1.0


def test_evaluator_rejects_inconsistent_class_counts():
    with pytest.raises(ValueError):
        GCDEvaluator(5, 3)


def test_extract_features_returns_unit_norm_cls():
    adapter = BackboneAdapter(TinyViT(embed_dim=16, depth=2))
    adapter.train()
    feats = extract_features(adapter, [torch.randn(3, 3, 32, 32), torch.randn(2, 3, 32, 32)])
    assert feats.shape == (5, 16) and adapter.training
    torch.testing.assert_close(feats.norm(dim=1), torch.ones(5))
