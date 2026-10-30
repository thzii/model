"""Multi-loss gradient monitor (master plan Task #5): linearity of the loss parts, the statistics,
and that monitoring leaves training bit-identical. Synthetic tensors and data only."""
import json
import math

import pytest
import torch
import torch.nn.functional as F

from helpers import make_tiny_config
from planora_sf.cli import train
from planora_sf.config import DebGCDConfig, HypCDConfig
from planora_sf.data import DummyGCDDataset, build_gcd_datasets, make_gcd_split
from planora_sf.engine import TrainBatch, compute_losses, fit, refresh_pseudo_labels
from planora_sf.engine.grad_monitor import GradientMonitor, gradient_stats, shared_parameters
from planora_sf.fourier import FourierViewGenerator
from planora_sf.models import build_model

N, B, NUM_KNOWN, NUM_NOVEL = 40, 8, 2, 2


def _losses(cfg, hyp_alpha=None):
    torch.manual_seed(0)
    model = build_model(cfg, num_known_classes=NUM_KNOWN, num_classes=NUM_KNOWN + NUM_NOVEL)
    classes = torch.arange(N) % 4
    labelled = (classes < NUM_KNOWN) & (torch.arange(N) < N // 2)
    targets = torch.where(labelled, classes, torch.full_like(classes, -1))
    feats = F.normalize(torch.randn(N, model.backbone.feat_dim), dim=1)
    state = refresh_pseudo_labels(feats, labelled, targets, NUM_KNOWN, NUM_NOVEL, cfg,
                                  generator=torch.Generator().manual_seed(0))
    idx = torch.arange(B)
    batch = TrainBatch(torch.randn(B, 3, 32, 32), torch.randn(B, 3, 32, 32), targets[idx], labelled[idx])
    losses, _ = compute_losses(model, FourierViewGenerator(cfg.fourier), batch, state.batch(idx), cfg,
                               generator=torch.Generator().manual_seed(1), hyp_alpha=hyp_alpha)
    return model, losses


def _grad(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)])


def _debgcd():
    return DebGCDConfig(enabled=True, detector_hidden_dim=32, detector_bottleneck_dim=16, threshold=0.0)


def test_parts_add_up_to_the_total_gradient():
    cfg = make_tiny_config().set_components(fourex=True)
    cfg.debgcd = _debgcd()
    model, losses = _losses(cfg)
    assert set(losses.parts) == {"base", "fourier", "debias"}
    params = [p for group in shared_parameters(model.backbone).values() for p in group]
    total = _grad(losses.total, params)
    summed = sum(_grad(losses.parts[name], params) for name in ("base", "fourier", "debias"))
    torch.testing.assert_close(summed, total, rtol=1e-4, atol=1e-5 * float(total.abs().max()))
    parts_sum = losses.parts["base"] + losses.parts["fourier"] + losses.parts["debias"]
    assert float(parts_sum.detach()) == pytest.approx(float(losses.total.detach()), rel=1e-6)


def test_hypcd_branches_mix_to_the_base_gradient():
    cfg = make_tiny_config().set_components(fourex=False)
    cfg.hypcd = HypCDConfig(enabled=True)
    alpha = 0.3
    model, losses = _losses(cfg, hyp_alpha=alpha)
    assert set(losses.parts) == {"base", "hyp_dist", "hyp_ang"}  # E4: FourEx weights are 0
    params = [p for group in shared_parameters(model.backbone).values() for p in group]
    mixed = alpha * _grad(losses.parts["hyp_dist"], params) + (1 - alpha) * _grad(losses.parts["hyp_ang"], params)
    base = _grad(losses.parts["base"], params)
    torch.testing.assert_close(mixed, base, rtol=1e-4, atol=1e-5 * float(base.abs().max()))  # float32 sums


def test_gradient_stats_keys_and_cosines():
    cfg = make_tiny_config().set_components(fourex=True)
    model, losses = _losses(cfg)
    groups = shared_parameters(model.backbone)
    assert list(groups) == ["block2", "block3"]  # tiny config: grad_from_block 2 of 4
    parts = {"base": losses.parts["base"], "fourier": losses.parts["fourier"], "copy": 2.0 * losses.parts["base"]}
    stats = gradient_stats(parts, groups)
    assert stats["cos/base~copy"] == pytest.approx(1.0) and stats["block3/cos/base~copy"] == pytest.approx(1.0)
    assert stats["norm/copy"] == pytest.approx(2 * stats["norm/base"])
    assert stats["norm/base"] ** 2 == pytest.approx(stats["block2/norm/base"] ** 2 + stats["block3/norm/base"] ** 2)
    assert -1.0 <= stats["cos/base~fourier"] <= 1.0
    zero = {"base": losses.parts["base"], "flat": losses.parts["base"] * 0.0}
    assert "cos/base~flat" not in gradient_stats(zero, groups)  # undefined for a zero gradient


def _fit(every):
    torch.manual_seed(0)
    cfg = make_tiny_config()
    cfg.debgcd = _debgcd()
    cfg.optim.batch_size, cfg.optim.epochs, cfg.loop.num_workers, cfg.eval.n_init = 16, 3, 0, 2
    cfg.loop.max_steps_per_epoch, cfg.loop.grad_monitor_every = 3, every
    train_src = DummyGCDDataset(num_classes=8, images_per_class=12, image_size=40)
    split = make_gcd_split(train_src.targets)
    datasets = build_gcd_datasets(train_src, DummyGCDDataset(num_classes=8, images_per_class=3, seed=1), split,
                                  image_size=32)
    model = build_model(cfg, num_known_classes=split.num_known, num_classes=split.num_classes)
    result = fit(model, datasets, cfg, epochs=2, generator=torch.Generator().manual_seed(0))
    return result, {n: p.detach().clone() for n, p in model.named_parameters()}


def test_monitoring_leaves_training_bit_identical():
    off, params_off = _fit(0)
    on, params_on = _fit(2)
    assert [h.losses for h in on.history] == [h.losses for h in off.history]
    assert all(torch.equal(params_on[n], params_off[n]) for n in params_off)
    assert all("grad" not in h.diagnostics for h in off.history)
    for h in on.history:
        grad = h.diagnostics["grad"]
        assert grad["steps"] == 2  # steps 0 and 2 of 3
        assert {"norm/base", "norm/fourier", "norm/debias", "cos/base~fourier", "cos/fourier~debias"} <= set(grad["mean"])
        assert all(0.0 <= v <= 1.0 for v in grad["negative_fraction"].values())
        assert all(math.isfinite(v) for v in grad["mean"].values())


def test_monitor_validation_and_cli(tmp_path, capsys):
    model = build_model(make_tiny_config(), num_known_classes=2)
    with pytest.raises(ValueError):
        GradientMonitor(model.backbone, 0)
    assert train.main(["--dataset", "dummy", "--epochs", "1", "--max-steps", "2", "--batch-size", "16",
                       "--device", "cpu", "--num-workers", "0", "--grad-monitor-every", "1",
                       "--checkpoint-dir", str(tmp_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["history"][0]["diagnostics"]["grad"]["steps"] == 2
    assert "cos/base~fourier" in summary["history"][0]["diagnostics"]["grad"]["mean"]
