"""HypCD model wiring (plan Phase 3): ToPoincare replaces the ℓ2 normalisation of z and
nothing else changes. Synthetic tensors only."""
import math

import pytest
import torch
import torch.nn.functional as F

from helpers import make_tiny_config
from planora_sf.config import HypCDConfig, PlanoraSFConfig
from planora_sf.engine import TrainBatch, compute_losses, refresh_pseudo_labels
from planora_sf.fourier import FourierViewGenerator
from planora_sf.hyperbolic import ToPoincare, clip_features, expmap0, pairwise_poincare_distance, project
from planora_sf.models import build_model


def _hyp_cfg(**hypcd):
    cfg = make_tiny_config()
    cfg.hypcd = HypCDConfig(enabled=True, **hypcd)
    return cfg


def test_to_poincare_is_parameter_free_and_composes_clip_exp_project():
    layer = ToPoincare(0.05, 2.3)
    assert list(layer.parameters()) == [] and list(layer.buffers()) == [] and layer.state_dict() == {}
    v = torch.randn(10, 32) * 5
    torch.testing.assert_close(layer(v), project(expmap0(clip_features(v, 2.3), 0.05), 0.05))
    assert (layer(v).norm(dim=1) < 1 / math.sqrt(0.05)).all()
    assert "c=0.05, clip_r=2.3" in repr(layer)
    with pytest.raises(ValueError):
        ToPoincare(0.0, 2.3)


def test_hypcd_model_maps_head_output_to_the_ball():
    torch.manual_seed(0)
    model = build_model(_hyp_cfg(), num_known_classes=3)
    out = model(torch.randn(6, 3, 32, 32))
    torch.testing.assert_close(out.z_tangent, model.head(out.cls))
    torch.testing.assert_close(out.z, ToPoincare(0.05, 2.3)(out.z_tangent))
    assert (out.z.norm(dim=1) < 1 / math.sqrt(0.05)).all()
    assert not torch.allclose(out.z.norm(dim=1), torch.ones(6))  # no ℓ2 normalisation under HypCD


def test_euclidean_model_is_unchanged_and_exposes_the_tangent():
    torch.manual_seed(0)
    model = build_model(make_tiny_config(), num_known_classes=3)
    out = model(torch.randn(6, 3, 32, 32))
    assert model.to_ball is None
    torch.testing.assert_close(out.z, F.normalize(out.z_tangent, dim=-1))


def test_hypcd_changes_no_parameters_initialisation_or_state_dict():
    torch.manual_seed(0)
    euclid = build_model(make_tiny_config(), num_known_classes=3)
    torch.manual_seed(0)
    hyp = build_model(_hyp_cfg(), num_known_classes=3)
    a, b = euclid.state_dict(), hyp.state_dict()
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    hyp.load_state_dict(a)  # checkpoints load in both directions (cli.evaluate)
    torch.manual_seed(1)
    x = torch.randn(4, 3, 32, 32)
    torch.testing.assert_close(euclid(x).cls, hyp(x).cls)  # backbone features: what BSSK and the evaluator use


def test_gradients_flow_through_the_ball_to_head_and_trainable_blocks():
    torch.manual_seed(0)
    model = build_model(_hyp_cfg(clip_r=100.0), num_known_classes=3)  # unclipped, so radial gradients exist too
    out = model(torch.randn(8, 3, 32, 32))
    pairwise_poincare_distance(out.z, out.z, 0.05).sum().backward()
    for name, param in model.named_parameters():
        if param.requires_grad and not name.startswith("cls_head"):
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
        elif not param.requires_grad:
            assert param.grad is None, name
    assert any(p.grad.abs().sum() > 0 for p in model.head.parameters() if p.requires_grad)
    assert any(p.grad.abs().sum() > 0 for p in model.backbone.blocks[2:].parameters())


def test_build_model_validation():
    gcd = _hyp_cfg()
    gcd.base = "gcd"
    with pytest.raises(NotImplementedError, match="SelEx base only"):
        build_model(gcd, num_known_classes=3)
    renorm = _hyp_cfg()
    renorm.latent.renormalize_slices = True
    with pytest.raises(ValueError, match="renormalize_slices"):
        build_model(renorm, num_known_classes=3)


def test_losses_refuse_to_run_without_alpha(generator):
    cfg = _hyp_cfg()
    model = build_model(cfg, num_known_classes=2)
    labelled = torch.arange(40) < 10
    targets = torch.where(labelled, torch.arange(40) % 2, torch.full((40,), -1))
    state = refresh_pseudo_labels(F.normalize(torch.randn(40, model.backbone.feat_dim), dim=1), labelled, targets,
                                  2, 2, cfg, generator=generator)
    batch = TrainBatch(torch.randn(8, 3, 32, 32), torch.randn(8, 3, 32, 32), targets[:8], labelled[:8])
    with pytest.raises(ValueError, match="hyp_alpha"):
        compute_losses(model, FourierViewGenerator(cfg.fourier), batch, state.batch(torch.arange(8)), cfg,
                       generator=generator)


def test_config_validation_and_defaults():
    assert PlanoraSFConfig().hypcd == HypCDConfig() and not HypCDConfig().enabled
    d = HypCDConfig()
    assert (d.curvature, d.clip_r, d.alpha_max, d.alpha_epoch_offset, d.angle_similarity) == (0.05, 2.3, 1.0, 0, "cosine")
    for bad in ({"curvature": 0.0}, {"clip_r": -1.0}, {"alpha_max": 1.5}, {"alpha_epoch_offset": -1},
                {"angle_similarity": "dot"}, {"project_eps": 0.0}):
        with pytest.raises(ValueError):
            HypCDConfig(**bad)
    cfg = PlanoraSFConfig.for_dataset("cub")
    cfg.hypcd.enabled = True
    assert cfg.components()["row"] is None  # HypCD with FourEx on is composed: no row (plan D3)
