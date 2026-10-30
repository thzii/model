"""HypCD losses (plan Phase 4): the Poincaré and cosine similarities, the α_d hybrid of every
contrastive term, gradient flow, and the invariances E0-E3 rely on. Synthetic tensors only."""
import math

import pytest
import torch
import torch.nn.functional as F

from helpers import make_tiny_config
from planora_sf.config import DebGCDConfig, HypCDConfig
from planora_sf.debgcd import permute_level
from planora_sf.engine import TrainBatch, compute_losses, refresh_pseudo_labels
from planora_sf.fourier import FourierViewGenerator
from planora_sf.hyperbolic import ToPoincare, pairwise_poincare_distance, poincare_distance
from planora_sf.losses.contrastive import NegPoincare, pairwise_similarity
from planora_sf.models import build_model
from planora_sf.pseudo_labels.hssk import PseudoLabelState

N, B, NUM_KNOWN, NUM_NOVEL = 40, 8, 2, 2
D64 = torch.float64


def _ball(n, dim, c=0.05):
    return ToPoincare(c, 2.3)(torch.randn(n, dim, dtype=D64) * 2)


# ---------------------------------------------------------------- similarities

def test_neg_poincare_similarity_matches_the_distance():
    x, y = _ball(12, 16), _ball(9, 16)
    torch.testing.assert_close(pairwise_similarity(x, y, NegPoincare(0.05)), -pairwise_poincare_distance(x, y, 0.05))


def test_masked_similarities_equal_explicit_per_anchor_slices():
    """Masked prefixes (FourEx L_LSSE widths) = Poincaré / cosine on sliced ball points."""
    x, y = _ball(10, 32), _ball(7, 32)
    widths = torch.randint(1, 33, (10,))
    mask = torch.arange(32)[None, :] < widths[:, None]
    masked_d = pairwise_similarity(x, y, NegPoincare(0.05), anchor_dim_mask=mask)
    masked_cos = pairwise_similarity(x, y, "cosine", anchor_dim_mask=mask)
    for a, w in enumerate(widths.tolist()):
        xa, ys = x[a : a + 1, :w], y[:, :w]
        torch.testing.assert_close(masked_d[a], -poincare_distance(xa, ys, 0.05), rtol=1e-8, atol=1e-8)
        torch.testing.assert_close(masked_cos[a], (F.normalize(xa, dim=1) @ F.normalize(ys, dim=1).T)[0])


def test_cosine_similarity_and_small_curvature_limit():
    x, y = torch.randn(6, 8, dtype=D64), torch.randn(5, 8, dtype=D64)
    torch.testing.assert_close(pairwise_similarity(x, y, "cosine"), F.normalize(x, dim=1) @ F.normalize(y, dim=1).T)
    torch.testing.assert_close(pairwise_similarity(x, y, NegPoincare(1e-10)), 2 * pairwise_similarity(x, y), rtol=1e-5,
                               atol=1e-6)


def test_similarity_validation():
    x = _ball(4, 8)
    with pytest.raises(ValueError):
        NegPoincare(0.0)
    with pytest.raises(ValueError, match="renormalising"):
        pairwise_similarity(x, x, NegPoincare(0.05), renormalize=True)
    with pytest.raises(ValueError):
        pairwise_similarity(x, x, "manhattan")


# ---------------------------------------------------------------- loss step

def _setup(cfg, generator):
    model = build_model(cfg, num_known_classes=NUM_KNOWN, num_classes=NUM_KNOWN + NUM_NOVEL)
    classes = torch.arange(N) % 4
    labelled = (classes < NUM_KNOWN) & (torch.arange(N) < N // 2)
    targets = torch.where(labelled, classes, torch.full_like(classes, -1))
    feats = F.normalize(torch.randn(N, model.backbone.feat_dim), dim=1)
    state = refresh_pseudo_labels(feats, labelled, targets, NUM_KNOWN, NUM_NOVEL, cfg, generator=generator)
    idx = torch.arange(B)
    batch = TrainBatch(torch.randn(B, 3, 32, 32), torch.randn(B, 3, 32, 32), targets[idx], labelled[idx])
    return model, FourierViewGenerator(cfg.fourier), batch, state


def _hyp_cfg(fourex=False, debgcd=False, **hypcd):
    cfg = make_tiny_config().set_components(fourex=fourex)
    cfg.hypcd = HypCDConfig(enabled=True, **hypcd)
    if debgcd:
        cfg.debgcd = DebGCDConfig(enabled=True, detector_hidden_dim=32, detector_bottleneck_dim=16, threshold=0.0)
    return cfg


def _step(model, fourier, batch, state, cfg, alpha, seed=3, s=None):
    torch.manual_seed(seed)
    pseudo = (s or state).batch(torch.arange(B))
    return compute_losses(model, fourier, batch, pseudo, cfg, generator=torch.Generator().manual_seed(seed),
                          hyp_alpha=alpha)[0]


@pytest.mark.parametrize("fourex", [False, True])
def test_alpha_mixes_every_contrastive_term_linearly(fourex, generator):
    cfg = _hyp_cfg(fourex=fourex)
    model, fourier, batch, state = _setup(cfg, generator)
    ang, dist, mid = (_step(model, fourier, batch, state, cfg, a) for a in (0.0, 1.0, 0.3))
    assert ang.as_floats()["base"] == pytest.approx(ang.stats["hyp_base_ang"])
    assert dist.as_floats()["base"] == pytest.approx(dist.stats["hyp_base_dist"])
    for term in ("base", "low", "high"):
        expected = 0.3 * getattr(dist, term) + 0.7 * getattr(ang, term)
        torch.testing.assert_close(getattr(mid, term), expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ang.cls, dist.cls)  # L_cls is on the Euclidean CLS: unaffected
    assert ang.stats["hyp_base_dist"] == pytest.approx(dist.stats["hyp_base_dist"])  # same forward, same branches


@pytest.mark.parametrize("fourex, debgcd", [(False, False), (True, False), (False, True)])  # E4, +FourEx, +DebGCD
def test_hypcd_step_is_finite_and_gradients_flow(fourex, debgcd, generator):
    cfg = _hyp_cfg(fourex=fourex, debgcd=debgcd)
    model, fourier, batch, state = _setup(cfg, generator)
    losses = _step(model, fourier, batch, state, cfg, 0.5)
    values = losses.as_floats()
    hyp_keys = {"hyp_alpha", "hyp_clip_fraction", "hyp_tangent_norm", "hyp_base_dist", "hyp_base_ang"}
    assert hyp_keys <= set(values) and all(math.isfinite(v) for v in values.values())
    assert ("adl_utilisation" in values) == debgcd  # DebGCD keeps the HypCD stats
    assert 0.0 <= values["hyp_clip_fraction"] <= 1.0 and values["hyp_alpha"] == 0.5
    losses.total.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
        else:
            assert param.grad is None, name
    assert any(p.grad.abs().sum() > 0 for p in model.head.parameters() if p.requires_grad)
    assert any(p.grad.abs().sum() > 0 for p in model.backbone.blocks[2:].parameters())


def test_cosine_branch_ignores_curvature_and_clip_but_distance_branch_does_not(generator):
    a = _hyp_cfg(curvature=0.05, clip_r=2.3)
    b = _hyp_cfg(curvature=0.1, clip_r=2.0)
    torch.manual_seed(0)
    model_a, fourier, batch, state = _setup(a, generator)
    model_b = build_model(b, num_known_classes=NUM_KNOWN, num_classes=NUM_KNOWN + NUM_NOVEL)
    model_b.load_state_dict(model_a.state_dict())
    la = _step(model_a, fourier, batch, state, a, 0.5)
    lb = _step(model_b, fourier, batch, state, b, 0.5)
    assert la.stats["hyp_base_ang"] == pytest.approx(lb.stats["hyp_base_ang"], rel=1e-5)
    assert la.stats["hyp_base_dist"] != pytest.approx(lb.stats["hyp_base_dist"], rel=1e-3)
    up = _hyp_cfg(curvature=0.05, clip_r=2.3, angle_similarity="neg_euclidean")
    lu = _step(model_a, fourier, batch, state, up, 0.5)  # upstream angle branch: Euclidean distance on ball points
    assert lu.stats["hyp_base_ang"] != pytest.approx(la.stats["hyp_base_ang"], rel=1e-3)
    assert lu.stats["hyp_base_dist"] == pytest.approx(la.stats["hyp_base_dist"], rel=1e-6)


def test_clip_fraction_tracks_the_radius(generator):
    for r, expected in ((1e6, 0.0), (1e-6, 1.0)):
        cfg = _hyp_cfg(clip_r=r)
        model, fourier, batch, state = _setup(cfg, generator)
        assert _step(model, fourier, batch, state, cfg, 0.5).stats["hyp_clip_fraction"] == expected


def test_alpha_is_required_and_validated(generator):
    cfg = _hyp_cfg()
    model, fourier, batch, state = _setup(cfg, generator)
    for bad in (None, -0.1, 1.5):
        with pytest.raises(ValueError, match="hyp_alpha"):
            _step(model, fourier, batch, state, cfg, bad)


def test_relabelling_novel_clusters_leaves_hypcd_losses_unchanged(generator):
    """The DebGCD cluster alignment (plan C2) stays a no-op under HypCD."""
    cfg = _hyp_cfg(fourex=True)
    model, fourier, batch, state = _setup(cfg, generator)
    permuted = PseudoLabelState([permute_level(state.levels[0], torch.tensor([0, 1, 3, 2])), *state.levels[1:]])
    a = _step(model, fourier, batch, state, cfg, 0.4).as_floats()
    b = _step(model, fourier, batch, state, cfg, 0.4, s=permuted).as_floats()
    assert a == pytest.approx(b, abs=1e-6)


def test_unlabelled_targets_cannot_change_hypcd_losses(generator):
    cfg = _hyp_cfg(fourex=True)
    model, fourier, batch, state = _setup(cfg, generator)
    leaked = TrainBatch(batch.view1, batch.view2, torch.where(batch.labelled, batch.targets, torch.arange(B) % 4),
                        batch.labelled)
    a = _step(model, fourier, batch, state, cfg, 0.4).as_floats()
    b = _step(model, fourier, leaked, state, cfg, 0.4).as_floats()
    assert a == b


def test_euclidean_rows_ignore_hyp_alpha_and_report_no_hyp_stats(generator):
    cfg = make_tiny_config()
    model, fourier, batch, state = _setup(cfg, generator)
    a = _step(model, fourier, batch, state, cfg, None).as_floats()
    b = _step(model, fourier, batch, state, cfg, 0.7).as_floats()
    assert a == b and not any(k.startswith("hyp_") for k in a)
