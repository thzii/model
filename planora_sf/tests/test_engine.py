"""End-to-end loss step on synthetic tensors: shapes, finiteness and gradient flow through the
4-view batch [x(1); x(2); l; h] (spec §6 part B). No training loop is run."""
import math

import pytest
import torch
import torch.nn.functional as F

from helpers import make_tiny_config
from planora_sf.engine import TrainBatch, build_optimizer, build_scheduler, compute_losses, refresh_pseudo_labels
from planora_sf.eval.estimate_k import estimate_num_classes
from planora_sf.fourier import FourierViewGenerator
from planora_sf.models import build_model

N, B, NUM_KNOWN, NUM_NOVEL = 40, 8, 2, 2


def _setup(cfg, generator):
    model = build_model(cfg, num_known_classes=NUM_KNOWN)
    fourier = FourierViewGenerator(cfg.fourier)
    classes = torch.arange(N) % 4
    labelled = (classes < NUM_KNOWN) & (torch.arange(N) < N // 2)
    targets = torch.where(labelled, classes, torch.full_like(classes, -1))
    feats = F.normalize(torch.randn(N, model.backbone.feat_dim), dim=1)  # stand-in for view-0 CLS features
    state = refresh_pseudo_labels(feats, labelled, targets, NUM_KNOWN, NUM_NOVEL, cfg, generator=generator)
    idx = torch.arange(B)
    batch = TrainBatch(
        view1=torch.randn(B, 3, 32, 32), view2=torch.randn(B, 3, 32, 32), targets=targets[idx], labelled=labelled[idx]
    )
    return model, fourier, batch, state


@pytest.mark.parametrize("selex_compat", [True, False])
def test_full_step_is_finite_and_gradients_flow(selex_compat, generator):
    cfg = make_tiny_config(selex_compat=selex_compat)
    model, fourier, batch, state = _setup(cfg, generator)
    assert state.num_levels == 1 + int(math.log2(NUM_KNOWN))

    losses, views = compute_losses(model, fourier, batch, state.batch(torch.arange(B)), cfg, generator=generator)
    assert views.low.shape == views.high.shape == (B, 3, 32, 32)
    values = losses.as_floats()
    assert all(math.isfinite(v) for v in values.values())
    assert all(values[k] > 0 for k in ("base", "low", "high", "cls"))
    torch.testing.assert_close(losses.total, losses.base + 0.1 * (losses.low + losses.high + losses.cls))

    losses.total.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
        else:
            assert param.grad is None, name  # frozen blocks 0-1, embeddings, final norm, weight-norm gain
    trainable_blocks = model.backbone.blocks[2:]
    assert any(p.grad.abs().sum() > 0 for p in trainable_blocks.parameters())
    assert all(p.grad is None for p in model.backbone.blocks[0].parameters())


def test_gcd_base_without_cls_head(generator):
    cfg = make_tiny_config(base="gcd")
    cfg.cls_head.enabled = False
    model, fourier, batch, state = _setup(cfg, generator)
    losses, _ = compute_losses(model, fourier, batch, state.batch(torch.arange(B)), cfg, generator=generator)
    assert losses.cls.item() == 0.0 and math.isfinite(losses.total.item())
    losses.total.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)


def test_batch_without_labelled_samples(generator):
    cfg = make_tiny_config()
    model, fourier, batch, state = _setup(cfg, generator)
    unlabelled = TrainBatch(batch.view1, batch.view2, torch.full((B,), -1), torch.zeros(B, dtype=torch.bool))
    losses, _ = compute_losses(model, fourier, unlabelled, state.batch(torch.arange(B)), cfg, generator=generator)
    assert losses.cls.item() == 0.0 and math.isfinite(losses.total.item())


def test_batch_validation():
    with pytest.raises(ValueError):
        TrainBatch(torch.randn(2, 3, 8, 8), torch.randn(3, 3, 8, 8), torch.zeros(2, dtype=torch.long),
                   torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError):
        TrainBatch(torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8), torch.tensor([0, -1]),
                   torch.ones(2, dtype=torch.bool))


def test_unscaffolded_paths_raise(generator):
    cfg = make_tiny_config()
    cfg.fourier.extra_views = False
    model, fourier, batch, state = _setup(cfg, generator)
    with pytest.raises(NotImplementedError):
        compute_losses(model, fourier, batch, state.batch(torch.arange(B)), cfg)
    with pytest.raises(NotImplementedError):
        estimate_num_classes(torch.randn(4, 2), torch.ones(4, dtype=torch.bool), torch.zeros(4, dtype=torch.long), 10)


def test_optimizer_covers_trainable_params_and_cosine_floor(tiny_cfg):
    model = build_model(tiny_cfg, num_known_classes=NUM_KNOWN)
    tiny_cfg.optim.epochs = 10
    optimizer = build_optimizer(model, tiny_cfg.optim)
    params = optimizer.param_groups[0]["params"]
    assert len(params) == sum(1 for p in model.parameters() if p.requires_grad)
    scheduler = build_scheduler(optimizer, tiny_cfg.optim)
    for _ in range(tiny_cfg.optim.epochs):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 1e-3)
