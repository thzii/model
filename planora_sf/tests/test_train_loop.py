"""Training engine dry runs on DummyGCDDataset + TinyViT (spec §6). Synthetic data only: these
tests check orchestration, finiteness and parameter updates, not accuracy."""
import math

import pytest
import torch
from torch.utils.data import DataLoader

import planora_sf.engine.train as train_mod
from helpers import make_tiny_config
from planora_sf.data import DummyGCDDataset, build_gcd_datasets, build_gcd_sampler, make_gcd_split
from planora_sf.engine import build_optimizer, fit, refresh_from_dataset, train_one_epoch
from planora_sf.eval import extract_dataset_features
from planora_sf.fourier import FourierViewGenerator
from planora_sf.models import build_model

BATCH = 16


def _setup(**loop_overrides):
    cfg = make_tiny_config()
    cfg.optim.batch_size = BATCH
    cfg.optim.epochs = 3
    cfg.loop.num_workers = 0
    cfg.eval.n_init = 2
    for key, value in loop_overrides.items():
        setattr(cfg.loop, key, value)
    train_src = DummyGCDDataset(num_classes=8, images_per_class=12, image_size=40)
    test_src = DummyGCDDataset(num_classes=8, images_per_class=3, image_size=40, seed=1)
    split = make_gcd_split(train_src.targets)
    datasets = build_gcd_datasets(train_src, test_src, split, image_size=32)
    return cfg, datasets, build_model(cfg, num_known_classes=split.num_known)


def _loader(datasets, generator):
    train = datasets.train
    sampler = build_gcd_sampler(train.num_labelled, train.num_unlabelled, generator=generator)
    return DataLoader(train, batch_size=BATCH, sampler=sampler, drop_last=True)


def test_train_one_epoch_dry_run(generator):
    cfg, datasets, model = _setup()
    state = refresh_from_dataset(model, datasets.train, cfg, batch_size=BATCH, generator=generator)
    assert state.num_levels == 1 + int(math.log2(4)) and state.levels[0].labels.shape == (96,)

    trainable = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = build_optimizer(model, cfg.optim)
    steps, losses = train_one_epoch(
        model, FourierViewGenerator(cfg.fourier), _loader(datasets, generator), state, optimizer, cfg,
        max_steps=3, generator=generator,
    )
    assert steps == 3
    assert set(losses) == {"total", "base", "low", "high", "cls"} and all(math.isfinite(v) for v in losses.values())
    params = dict(model.named_parameters())
    assert any(not torch.equal(params[n], v) for n, v in trainable.items())
    assert all(torch.equal(params[n], v) for n, v in frozen.items())  # frozen blocks/embeddings untouched


def test_uncapped_epoch_covers_dataset_once(generator):
    cfg, datasets, model = _setup()
    state = refresh_from_dataset(model, datasets.train, cfg, batch_size=BATCH, generator=generator)
    steps, _ = train_one_epoch(
        model, FourierViewGenerator(cfg.fourier), _loader(datasets, generator), state,
        build_optimizer(model, cfg.optim), cfg, generator=generator,
    )
    assert steps == len(datasets.train) // BATCH == 6


def test_training_sees_only_labelled_targets(monkeypatch, generator):
    cfg, datasets, model = _setup(max_steps_per_epoch=4)
    seen = []
    original = train_mod.compute_losses

    def spy(model_, fourier, batch, pseudo, cfg_, **kwargs):
        seen.append((batch.targets.clone(), batch.labelled.clone()))
        return original(model_, fourier, batch, pseudo, cfg_, **kwargs)

    monkeypatch.setattr(train_mod, "compute_losses", spy)
    fit(model, datasets, cfg, epochs=1, generator=generator)
    assert len(seen) == 4
    for targets, labelled in seen:
        assert (targets[~labelled] == -1).all()
        assert ((targets[labelled] >= 0) & (targets[labelled] < datasets.split.num_known)).all()


def test_fit_alternates_refresh_and_training(monkeypatch, generator, tmp_path):
    cfg, datasets, model = _setup(max_steps_per_epoch=2)
    calls = []
    for name in ("refresh_from_dataset", "train_one_epoch"):
        original = getattr(train_mod, name)

        def wrapper(*args, _original=original, _name=name, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(train_mod, name, wrapper)

    result = fit(model, datasets, cfg, epochs=2, checkpoint_dir=tmp_path, generator=generator)
    assert calls == ["refresh_from_dataset", "train_one_epoch"] * 2  # A -> B, every epoch
    assert [h.epoch for h in result.history] == [0, 1] and all(h.steps == 2 for h in result.history)
    assert result.history[1].lr < result.history[0].lr  # cosine schedule stepped once per epoch
    assert result.state is not None and result.state.num_levels == 3

    assert [epoch for epoch, _ in result.evaluations] == [1]  # eval_every=0: only after the last epoch
    for report in result.evaluations[0][1].values():
        assert all(0.0 <= getattr(report, k) <= 1.0 for k in ("all", "known", "novel"))

    checkpoint = torch.load(tmp_path / "last.pt", weights_only=False)
    assert checkpoint["epoch"] == 1 and set(checkpoint) == {"epoch", "model", "optimizer", "scheduler", "config", "provenance"}


def test_eval_every_and_test_transform_prototypes(generator):
    cfg, datasets, model = _setup(max_steps_per_epoch=1, eval_every=1, prototype_features="test_transform")
    result = fit(model, datasets, cfg, epochs=2, generator=generator)
    assert [epoch for epoch, _ in result.evaluations] == [0, 1]


def test_feature_extraction_follows_dataset_index_order():
    cfg, datasets, model = _setup()
    feats = extract_dataset_features(model.backbone, datasets.train_eval, batch_size=BATCH)
    assert feats.shape == (96, model.backbone.feat_dim)
    model.backbone.eval()
    with torch.no_grad():
        for index in (0, 23, 24, 95):  # labelled block ends at 23, unlabelled starts at 24
            single = model.backbone(datasets.train_eval[index][0][None])
            torch.testing.assert_close(feats[index], torch.nn.functional.normalize(single, dim=-1)[0])


def test_fit_rejects_batch_larger_than_dataset():
    cfg, datasets, model = _setup()
    cfg.optim.batch_size = 1000
    with pytest.raises(ValueError, match="smaller than one batch"):
        fit(model, datasets, cfg, epochs=1)
