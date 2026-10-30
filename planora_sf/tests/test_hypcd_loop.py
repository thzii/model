"""HypCD in the training loop (plan Phase 5): the α_d schedule, per-epoch monitors, rows and
the CLI round trip. Synthetic data only."""
import json
import math

import pytest
import torch

import planora_sf.engine.train as train_mod
from helpers import make_tiny_config
from planora_sf.cli import evaluate, train
from planora_sf.config import HypCDConfig, PlanoraSFConfig
from planora_sf.data import DummyGCDDataset, build_gcd_datasets, make_gcd_split
from planora_sf.engine import fit
from planora_sf.models import build_model

HYP_KEYS = {"hyp_alpha", "hyp_clip_fraction", "hyp_tangent_norm", "hyp_base_dist", "hyp_base_ang"}


def test_alpha_schedule_paper_and_upstream():
    paper, upstream = HypCDConfig(), HypCDConfig(alpha_epoch_offset=1)
    assert [paper.alpha(e, 200) for e in (0, 1, 100, 199)] == [0.0, 0.005, 0.5, 0.995]
    assert [upstream.alpha(e, 200) for e in (0, 99, 199)] == [0.005, 0.5, 1.0]  # (e + 1)/200, train_HypSelEx.py:361
    half = HypCDConfig(alpha_max=0.5)
    assert half.alpha(199, 200) == pytest.approx(0.4975) and half.alpha(500, 200) == 0.5  # clamped
    with pytest.raises(ValueError):
        paper.alpha(-1, 200)


def _setup(**hypcd):
    cfg = make_tiny_config().set_components(fourex=False, hypcd=True)
    cfg.hypcd = HypCDConfig(enabled=True, **hypcd)
    cfg.optim.batch_size, cfg.optim.epochs, cfg.loop.num_workers, cfg.eval.n_init = 16, 4, 0, 2
    cfg.loop.max_steps_per_epoch = 2
    train_src = DummyGCDDataset(num_classes=8, images_per_class=12, image_size=40)
    test_src = DummyGCDDataset(num_classes=8, images_per_class=3, image_size=40, seed=1)
    split = make_gcd_split(train_src.targets)
    datasets = build_gcd_datasets(train_src, test_src, split, image_size=32)
    return cfg, datasets, build_model(cfg, num_known_classes=split.num_known, num_classes=split.num_classes)


def test_fit_passes_the_epoch_schedule_and_logs_monitors(monkeypatch, generator):
    cfg, datasets, model = _setup()
    seen = []
    original = train_mod.compute_losses

    def spy(*args, **kwargs):
        seen.append(kwargs["hyp_alpha"])
        return original(*args, **kwargs)

    monkeypatch.setattr(train_mod, "compute_losses", spy)
    result = fit(model, datasets, cfg, epochs=3, generator=generator)
    assert seen == [0.0, 0.0, 0.25, 0.25, 0.5, 0.5]  # e/E with E = optim.epochs = 4, two steps per epoch
    for h in result.history:
        assert HYP_KEYS <= set(h.losses) and all(math.isfinite(v) for v in h.losses.values())
        assert 0.0 <= h.losses["hyp_clip_fraction"] <= 1.0
    assert [h.losses["hyp_alpha"] for h in result.history] == [0.0, 0.25, 0.5]
    scores = result.evaluations[-1][1]  # the evaluator is unchanged: v1 and v2 on backbone CLS
    assert set(scores) == {"v1", "v2"} and all(0 <= getattr(scores["v2"], k) <= 1 for k in ("all", "known", "novel"))


def test_train_one_epoch_requires_the_epoch_under_hypcd(generator):
    cfg, datasets, model = _setup()
    loader = torch.utils.data.DataLoader(datasets.train, batch_size=16, drop_last=True)
    state = train_mod.refresh_from_dataset(model, datasets.train, cfg, batch_size=16, generator=generator)
    optimizer = train_mod.build_optimizer(model, cfg.optim)
    with pytest.raises(ValueError, match="epoch index"):
        train_mod.train_one_epoch(model, train_mod.FourierViewGenerator(cfg.fourier), loader, state, optimizer, cfg,
                                  max_steps=1, generator=generator)


def test_components_name_e4_and_leave_combinations_unnamed():
    cfg = PlanoraSFConfig.for_dataset("cub").set_components(fourex=False, hypcd=True)
    assert cfg.components() == {"fourex": False, "debgcd": False, "hypcd": True, "row": "E4"}
    assert cfg.set_components(fourex=True).components()["row"] is None
    assert cfg.set_components(fourex=False, debgcd=True).components()["row"] is None


def test_train_cli_e4_then_rescore(tmp_path, capsys):
    rc = train.main(["--dataset", "dummy", "--epochs", "2", "--max-steps", "2", "--batch-size", "16", "--device", "cpu",
                     "--num-workers", "0", "--checkpoint-dir", str(tmp_path), "--no-use-fourex", "--use_hypcd"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["components"]["row"] == "E4"
    assert all(HYP_KEYS <= set(h["losses"]) for h in summary["history"])
    assert set(summary["evaluations"][-1]) == {"epoch", "v1", "v2"}
    assert evaluate.main(["--checkpoint", str(tmp_path / "last.pt"), "--dataset", "dummy", "--device", "cpu",
                          "--num-workers", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["config"]["hypcd"]["enabled"] is True
