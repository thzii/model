"""Training-time diagnostics (master plan Task #2b): PL-acc L0 per epoch and HypCD radius monitors.

They read unlabelled ground truth, so they must not change training: same losses and parameters
with diagnostics on or off. Synthetic data only.
"""
import math

import pytest
import torch

from helpers import make_tiny_config
from planora_sf.data import DummyGCDDataset, build_gcd_datasets, make_gcd_split
from planora_sf.engine import fit
from planora_sf.engine.diagnostics import hyperbolic_radius_stats, pseudo_label_quality
from planora_sf.hyperbolic.layers import ToPoincare
from planora_sf.models import build_model
from planora_sf.pseudo_labels.hssk import HierarchyLevel, PseudoLabelState


def _datasets():
    train_src = DummyGCDDataset(num_classes=8, images_per_class=12, image_size=40)
    test_src = DummyGCDDataset(num_classes=8, images_per_class=3, image_size=40, seed=1)
    split = make_gcd_split(train_src.targets)
    return build_gcd_datasets(train_src, test_src, split, image_size=32)


def _run(diagnostics: bool):
    torch.manual_seed(0)
    cfg = make_tiny_config()
    cfg.optim.batch_size, cfg.optim.epochs, cfg.loop.num_workers, cfg.eval.n_init = 16, 3, 0, 2
    cfg.loop.max_steps_per_epoch, cfg.loop.diagnostics = 2, diagnostics
    datasets = _datasets()
    model = build_model(cfg, num_known_classes=datasets.split.num_known)
    result = fit(model, datasets, cfg, epochs=2, generator=torch.Generator().manual_seed(0))
    return result, {n: p.detach().clone() for n, p in model.named_parameters()}


def test_diagnostics_do_not_change_training():
    off, params_off = _run(False)
    on, params_on = _run(True)
    assert [h.losses for h in on.history] == [h.losses for h in off.history]
    assert all(torch.equal(params_on[n], params_off[n]) for n in params_off)
    assert all(h.diagnostics == {} for h in off.history)
    for h in on.history:
        pl = h.diagnostics["pseudo_labels"]
        assert all(0.0 <= pl["v2"][k] <= 1.0 for k in ("all", "known", "novel"))
        assert pl["routing"]["absorption"] is not None and pl["num_levels"] == h.num_levels


def test_pseudo_label_quality_on_known_labels():
    train = _datasets().train
    num_known = train.split.num_known
    perfect = HierarchyLevel(train.targets.clone(), torch.zeros(8, 4), torch.zeros(8), num_known)
    pl = pseudo_label_quality(PseudoLabelState([perfect]), train)
    assert pl["v2"] == {"all": 1.0, "known": 1.0, "novel": 1.0} and pl["purity"] == 1.0
    assert pl["routing"]["absorption"] == 0.0 and pl["routing"]["leakage"] == 0.0

    novel_to_known = train.targets.clone()
    novel_to_known[novel_to_known >= num_known] = 0  # every novel sample lands in known cluster 0
    pl = pseudo_label_quality(PseudoLabelState([HierarchyLevel(novel_to_known, torch.zeros(8, 4), torch.zeros(8),
                                                               num_known)]), train)
    # Cluster 0 now holds 12 unlabelled images of each novel class, so its majority is novel class 4
    # (ties go to the smaller id); the other three novel classes own no cluster.
    assert pl["routing"]["absorption"] == 1.0 and pl["unclaimed_novel_classes"] == 8 - num_known - 1


def test_hyperbolic_radius_stats_match_the_closed_form():
    c, r = 0.1, 2.0
    norms = torch.tensor([0.5, 1.0, 1.5, 3.0, 4.0])
    v = norms[:, None] * torch.nn.functional.normalize(torch.randn(5, 6), dim=1)
    z = ToPoincare(c, r)(v)
    stats = hyperbolic_radius_stats(v, z, c, r)
    assert stats["hyp_clip_fraction"] == pytest.approx(0.4)
    assert stats["hyp_tangent_norm_p50"] == pytest.approx(1.5, rel=1e-5)
    assert stats["hyp_tangent_norm_p5"] < stats["hyp_tangent_norm_p50"] < stats["hyp_tangent_norm_p95"]
    expected = 2 * norms.clamp(max=r)  # d_c(0, exp_0(C_r(v))) = 2 min(|v|, r) away from the projection
    assert stats["hyp_radius_mean"] == pytest.approx(float(expected.mean()), rel=1e-4)
    assert stats["hyp_radius_std"] == pytest.approx(float(expected.std(unbiased=False)), rel=1e-4)
    assert all(math.isfinite(x) for x in stats.values())
