"""Loss mechanics (spec §5): similarity, NT-Xent layout, SupCon parity with SelEx, SSE/LSSE
composition, and latent confinement of the gradients of L_low / L_high."""
import pytest
import torch
import torch.nn.functional as F

from planora_sf.config import LossWeights
from planora_sf.latent.partition import prefix_mask
from planora_sf.losses import (
    ContrastiveSettings,
    cls_loss,
    combine_losses,
    gcd_loss,
    high_loss,
    low_loss,
    nt_xent_logits,
    pairwise_similarity,
    sse_loss,
    supcon_loss,
    use_loss,
)

S = ContrastiveSettings()


def unit(*shape):
    return F.normalize(torch.randn(*shape), dim=-1)


def reference_supcon(features, labels, temperature=0.07):
    """Line-by-line port of SelEx SupConLoss (ct.py:55-148) with its -cdist similarity."""
    batch = features.shape[0]
    mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
    contrast = torch.cat(torch.unbind(features, dim=1), dim=0)
    logits = -torch.cdist(contrast, contrast) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    mask = mask.repeat(2, 2)
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(2 * batch).view(-1, 1), 0)
    mask = mask * logits_mask
    log_prob = logits - torch.log((torch.exp(logits) * logits_mask).sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    return (-mean_log_prob_pos).view(2, batch).mean()


@pytest.fixture
def labelled_batch():
    gt = torch.tensor([0, 1, 0, -1, -1, -1])
    levels = [torch.tensor([0, 1, 0, 2, 2, 1]), torch.tensor([0, 0, 0, 1, 1, 0])]
    return gt, gt >= 0, levels


def test_neg_euclidean_and_dot_match_reference():
    a, c = unit(6, 16), unit(5, 16)
    torch.testing.assert_close(pairwise_similarity(a, c), -torch.cdist(a, c), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(pairwise_similarity(a, c, "dot"), a @ c.T)


def test_masked_similarity_equals_slicing_per_row():
    a, c = unit(6, 16), unit(5, 16)
    widths = torch.tensor([1, 3, 5, 7, 9, 16])
    sim = pairwise_similarity(a, c, anchor_dim_mask=prefix_mask(widths, 16))
    renorm = pairwise_similarity(a, c, anchor_dim_mask=prefix_mask(widths, 16), renormalize=True)
    for i, w in enumerate(widths.tolist()):
        torch.testing.assert_close(sim[i], -torch.cdist(a[i : i + 1, :w], c[:, :w])[0], atol=1e-5, rtol=1e-5)
        expected = -torch.cdist(F.normalize(a[i : i + 1, :w], dim=1), F.normalize(c[:, :w], dim=1))[0]
        torch.testing.assert_close(renorm[i], expected, atol=1e-4, rtol=1e-4)


def test_similarity_gradient_is_finite_at_zero_distance():
    for mask in (None, prefix_mask(torch.tensor([2, 4, 6, 8]), 8)):
        a = unit(4, 8).requires_grad_()
        pairwise_similarity(a, a, anchor_dim_mask=mask).sum().backward()
        assert torch.isfinite(a.grad).all()


def test_nt_xent_layout_matches_selex():
    b = 4
    feats = unit(2 * b, 8)
    logits, cols = nt_xent_logits(feats, temperature=0.5)
    assert logits.shape == cols.shape == (2 * b, 2 * b - 1)
    sim = -torch.cdist(feats, feats)
    for a in range(2 * b):
        partner = (a + b) % (2 * b)
        assert cols[a, 0] == partner and a not in cols[a].tolist()
        assert torch.all(cols[a, 1:][1:] > cols[a, 1:][:-1])  # negatives in increasing order
        torch.testing.assert_close(logits[a, 0], sim[a, partner] / 0.5, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError):
        nt_xent_logits(unit(3, 8))


def test_supcon_matches_selex_reference():
    feats = unit(10, 2, 16)
    labels = torch.randint(0, 3, (10,))
    torch.testing.assert_close(supcon_loss(feats, labels), reference_supcon(feats, labels), atol=1e-4, rtol=1e-4)


def test_supcon_edge_cases():
    empty = supcon_loss(torch.zeros(0, 2, 8), torch.zeros(0, dtype=torch.long))
    assert empty.item() == 0.0
    single_view_distinct = supcon_loss(unit(4, 1, 8), torch.arange(4))  # no anchor has a positive
    assert single_view_distinct.item() == 0.0


def test_use_loss_smoothing_endpoints():
    feats = unit(8, 16)
    targets = torch.rand(8, 8)
    logits, cols = nt_xent_logits(feats, 1.0)
    torch.testing.assert_close(use_loss(feats, targets, 0.0, S), F.cross_entropy(logits, torch.zeros(8, dtype=torch.long)))
    expected = -(targets.gather(1, cols) * F.log_softmax(logits, dim=1)).sum(1).mean()
    torch.testing.assert_close(use_loss(feats, targets, 1.0, S), expected)
    assert torch.isfinite(use_loss(feats, targets, 0.5, S, loss_type="bce"))
    with pytest.raises(ValueError):
        use_loss(feats, targets, 0.5, S, loss_type="hinge")


def test_sse_composition_matches_spec(labelled_batch):
    gt, labelled, levels = labelled_batch
    views = unit(6, 2, 64)
    expected = 0.5 * (
        supcon_loss(views[labelled], gt[labelled])
        + supcon_loss(views[..., :32], levels[0]) / 2
        + supcon_loss(views[..., :16], levels[1]) / 4
    )
    torch.testing.assert_close(sse_loss(views, gt, labelled, levels, S), expected)


def test_lsse_min_rule_with_uniform_cap(labelled_batch):
    gt, labelled, levels = labelled_batch
    views = unit(6, 2, 64)
    cap = torch.full((6,), 20)
    expected = 0.5 * (
        supcon_loss(views[labelled][..., :20], gt[labelled])  # k = 0: min(64, 20)
        + supcon_loss(views[..., :20], levels[0]) / 2  # k = 1: min(32, 20)
        + supcon_loss(views[..., :16], levels[1]) / 4  # k = 2: min(16, 20)
    )
    torch.testing.assert_close(sse_loss(views, gt, labelled, levels, S, cap_widths=cap), expected, atol=1e-4, rtol=1e-4)


def test_sse_without_labelled_samples_is_finite(labelled_batch):
    _, _, levels = labelled_batch
    loss = sse_loss(unit(6, 2, 64), torch.full((6,), -1), torch.zeros(6, dtype=torch.bool), levels, S)
    assert torch.isfinite(loss) and loss > 0


def test_low_loss_gradient_stays_in_low_prefix(labelled_batch):
    gt, labelled, levels = labelled_batch
    z_x, z_l = unit(6, 64).requires_grad_(), unit(6, 64).requires_grad_()
    d_low = torch.tensor([3, 10, 20, 7, 32, 15])
    targets = torch.softmax(torch.randn(12, 12), dim=1)
    loss = low_loss(z_x, z_l, d_low, gt, labelled, levels, targets, lam=0.35, alpha=0.5, settings=S)
    loss.backward()
    assert torch.isfinite(loss)
    for z in (z_x, z_l):
        assert torch.isfinite(z.grad).all()
        assert z.grad[:, 32:].abs().max() == 0  # right half untouched by L_low
        assert z.grad[:, :32].abs().sum() > 0


def test_high_loss_gradient_stays_in_high_suffix(labelled_batch):
    gt, labelled, _ = labelled_batch
    z_x, z_h = unit(6, 64).requires_grad_(), unit(6, 64).requires_grad_()
    d_high = torch.tensor([1, 5, 32, 12, 30, 8])
    loss = high_loss(z_x, z_h, d_high, gt, labelled, lam=0.35, settings=S)
    loss.backward()
    assert torch.isfinite(loss)
    for z in (z_x, z_h):
        assert z.grad[:, :32].abs().max() == 0  # left half untouched by L_high
        assert z.grad[:, 32:].abs().sum() > 0


def test_cls_loss_uses_labelled_rows_only(labelled_batch):
    gt, labelled, _ = labelled_batch
    logits = [torch.randn(6, 2, requires_grad=True) for _ in range(3)]
    loss = cls_loss(logits, gt, labelled)
    expected = sum(
        F.binary_cross_entropy_with_logits(lg[labelled], F.one_hot(gt[labelled], 2).float()) for lg in logits
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for lg in logits:
        assert lg.grad[~labelled].abs().max() == 0 and lg.grad[labelled].abs().sum() > 0
    assert torch.isfinite(cls_loss(logits, gt, labelled, kind="softmax_ce"))
    assert cls_loss(logits, gt, torch.zeros(6, dtype=torch.bool)).item() == 0.0


def test_gcd_loss_and_weighted_total(labelled_batch):
    gt, labelled, _ = labelled_batch
    base = gcd_loss(unit(6, 32), unit(6, 32), gt, labelled, lam=0.35, settings=S)
    assert torch.isfinite(base)
    parts = [torch.tensor(v) for v in (2.0, 3.0, 5.0, 7.0)]
    total = combine_losses(*parts, LossWeights()).total
    torch.testing.assert_close(total, torch.tensor(2.0 + 0.1 * (3.0 + 5.0 + 7.0)))
