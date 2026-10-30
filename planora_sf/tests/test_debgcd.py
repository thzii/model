"""DebGCD port (plan §2.2, E2/E3): paper equations on hand-computed cases, the SelEx adapters,
integration into the loss step and training loop, and proof that DebGCD-off changes nothing."""
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import planora_sf.engine.train as train_mod
from helpers import make_tiny_config
from planora_sf.cli import evaluate, train
from planora_sf.config import DebGCDConfig, PlanoraSFConfig
from planora_sf.data import DummyGCDDataset, build_gcd_datasets, make_gcd_split
from planora_sf.debgcd import (
    DebGCDHeads,
    adl_labelled,
    adl_unlabelled,
    align_novel_clusters,
    centroid_posterior,
    certainty,
    consistency,
    debgcd_terms,
    distribution_score,
    permute_level,
    sdl_labelled,
    sdl_unlabelled,
)
from planora_sf.engine import TrainBatch, compute_losses, fit, refresh_pseudo_labels
from planora_sf.fourier import FourierViewGenerator
from planora_sf.models import build_model
from planora_sf.pseudo_labels.hssk import PseudoLabelState

GOLDEN = Path(__file__).parent / "data" / "debgcd_off_golden.json"
N, B, NUM_KNOWN, NUM_NOVEL = 40, 8, 2, 2


def _ova(pos) -> torch.Tensor:
    """Detector logits [N, 2, M] whose channel softmax is exactly (1 - pos, pos)."""
    pos = torch.as_tensor(pos, dtype=torch.float64)
    return torch.stack([torch.log(1 - pos), torch.log(pos)], dim=1)


# ---------------------------------------------------------------- equations (hand-computed)

def test_eq4_labelled_detector_loss_uses_the_hardest_negative():
    logits = _ova([[0.2, 0.7, 0.4]])  # y = 1; negatives o- = 0.8 (k=0), 0.6 (k=2)
    expected = -math.log(0.7) - math.log(0.6)
    assert sdl_labelled(logits, torch.tensor([1])).item() == pytest.approx(expected, abs=1e-6)


def test_eq5_unlabelled_detector_loss_is_mean_binary_entropy():
    pos = torch.tensor([[0.2, 0.5], [0.9, 0.1]], dtype=torch.float64)
    h = -(pos * pos.log() + (1 - pos) * (1 - pos).log())
    assert sdl_unlabelled(_ova(pos)).item() == pytest.approx(h.mean().item(), abs=1e-6)


def test_eq7_eq10_score_and_certainty():
    s = distribution_score(_ova([[0.2, 0.7, 0.4], [0.9, 0.1, 0.3]]))
    torch.testing.assert_close(s, torch.tensor([0.3, 0.1], dtype=torch.float64))  # o- of the argmax-o+ head
    torch.testing.assert_close(certainty(s), torch.tensor([0.4, 0.8], dtype=torch.float64))


def test_eq9_consistency_truth_table():
    novel = torch.tensor([True, True, False, False])
    score = torch.tensor([0.9, 0.1, 0.9, 0.1])
    assert consistency(novel, score).tolist() == [1.0, 0.0, 0.0, 1.0]


def test_eq11_eq12_debiased_classifier_losses():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    ce = -logits.log_softmax(dim=1)
    assert adl_labelled(logits, torch.tensor([0, 1])).item() == pytest.approx(((ce[0, 0] + ce[1, 1]) / 2).item())
    weight = torch.tensor([0.5, 0.0])  # 1/|B_u| includes the zero-weight row
    got = adl_unlabelled(logits, torch.tensor([2, 1]), weight).item()
    assert got == pytest.approx((0.5 * ce[0, 2] / 2).item())


def test_empty_subsets_give_graph_connected_zero():
    empty = torch.zeros(0, 2, 3, requires_grad=True)
    for value in (sdl_labelled(empty, torch.zeros(0, dtype=torch.long)), sdl_unlabelled(empty)):
        assert value.item() == 0.0 and value.requires_grad


def test_combined_terms_threshold_filter_and_certainty():
    # rows: 2 labelled (classes 0, 1), 3 unlabelled. Two known clusters (0, 1), one novel (2).
    pos = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.9, 0.2], [0.1, 0.2], [0.3, 0.2]])
    ova = _ova(pos).float()
    deb = torch.zeros(5, 3)
    posterior = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0.95, 0.05, 0], [0.05, 0, 0.95], [0.5, 0, 0.5]])
    targets = torch.tensor([0, 1, -1, -1, -1])
    labelled = torch.tensor([True, True, False, False, False])
    # Unlabelled scores s: row 2 -> 0.1 (known), row 3 -> 0.8 (novel), row 4 -> 0.7 (novel).
    # Row 2: pseudo 0 (known) & s < .5 -> consistent; row 3: pseudo 2 (novel) & s > .5 -> consistent;
    # row 4: max p = 0.5 < τ -> masked.
    terms = debgcd_terms(ova, deb, posterior, targets, labelled, 2, threshold=0.85)
    assert terms.utilisation == pytest.approx(2 / 3)
    ce = math.log(3)  # uniform debiased logits
    expected_adl = ce + (0.8 * ce + 0.6 * ce + 0.0) / 3  # d = |2s-1| = 0.8, 0.6
    assert terms.adl.item() == pytest.approx(expected_adl, rel=1e-5)
    # Flip row 3 to look known to the detector: the Eq. 9 filter drops it, the upstream variant keeps it.
    ova_flip = _ova(torch.cat([pos[:3], torch.tensor([[0.9, 0.2]]), pos[4:]])).float()
    with_f = debgcd_terms(ova_flip, deb, posterior, targets, labelled, 2, threshold=0.85)
    without_f = debgcd_terms(ova_flip, deb, posterior, targets, labelled, 2, threshold=0.85, consistency_filter=False)
    assert with_f.utilisation == pytest.approx(1 / 3) and without_f.utilisation == pytest.approx(2 / 3)


def test_unlabelled_targets_are_never_read():
    ova, deb = torch.randn(6, 2, 2), torch.randn(6, 4)
    posterior = torch.softmax(torch.randn(6, 4) * 5, dim=1)
    labelled = torch.tensor([True, True, True, False, False, False])
    a = debgcd_terms(ova, deb, posterior, torch.tensor([0, 1, 0, -1, -1, -1]), labelled, 2, threshold=0.0)
    b = debgcd_terms(ova, deb, posterior, torch.tensor([0, 1, 0, 3, 2, 1]), labelled, 2, threshold=0.0)
    assert torch.equal(a.sdl, b.sdl) and torch.equal(a.adl, b.adl)


# ---------------------------------------------------------------- heads

def test_heads_shapes_and_frozen_gains():
    heads = DebGCDHeads(in_dim=16, num_known=3, num_classes=7, cfg=DebGCDConfig(detector_hidden_dim=32,
                                                                              detector_bottleneck_dim=8))
    ova, deb = heads(torch.randn(5, 16))
    assert ova.shape == (5, 2, 3) and deb.shape == (5, 7)
    assert sum(isinstance(m, torch.nn.Linear) for m in heads.detector.mlp) == 5  # App. K: 5 layers
    assert deb.abs().max() <= 1 / 0.1 + 1e-4  # cosine / τ_a
    frozen = [n for n, p in heads.named_parameters() if not p.requires_grad]
    assert frozen == ["detector.last_layer.parametrizations.weight.original0",
                      "debiased.last_layer.parametrizations.weight.original0"]


# ---------------------------------------------------------------- SelEx adapters (COMPOSED)

def test_centroid_posterior_is_a_detached_cosine_softmax():
    feats = torch.randn(4, 6, requires_grad=True)
    centres = torch.randn(3, 6)
    p = centroid_posterior(feats, centres, 0.1)
    expected = (F.normalize(feats, dim=1) @ F.normalize(centres, dim=1).T / 0.1).softmax(dim=1)
    torch.testing.assert_close(p, expected.detach())
    assert not p.requires_grad


def test_alignment_recovers_a_planted_permutation_and_keeps_known_ids():
    prev = torch.randn(7, 16)
    planted = torch.tensor([0, 1, 5, 2, 6, 4, 3])  # new cluster j was old cluster planted[j]
    new = prev[planted] + 0.01 * torch.randn(7, 16)
    perm = align_novel_clusters(prev, new, num_known=2)
    assert torch.equal(perm, planted)
    assert torch.equal(align_novel_clusters(prev, prev, 2), torch.arange(7))


def _engine_setup(cfg, generator):
    model = build_model(cfg, num_known_classes=NUM_KNOWN, num_classes=NUM_KNOWN + NUM_NOVEL)
    classes = torch.arange(N) % 4
    labelled = (classes < NUM_KNOWN) & (torch.arange(N) < N // 2)
    targets = torch.where(labelled, classes, torch.full_like(classes, -1))
    feats = F.normalize(torch.randn(N, model.backbone.feat_dim), dim=1)
    state = refresh_pseudo_labels(feats, labelled, targets, NUM_KNOWN, NUM_NOVEL, cfg, generator=generator)
    idx = torch.arange(B)
    batch = TrainBatch(torch.randn(B, 3, 32, 32), torch.randn(B, 3, 32, 32), targets[idx], labelled[idx])
    return model, FourierViewGenerator(cfg.fourier), batch, state


@pytest.mark.parametrize("mode", ["compat", "paper", "radius"])
def test_permuting_level0_novel_ids_leaves_every_selex_and_fourex_loss_unchanged(mode, generator):
    """Plan C2: aligning cluster ids cannot change E0/E1 (base, low, high and cls losses)."""
    cfg = make_tiny_config()
    cfg.selex.target_mode = mode
    model, fourier, batch, state = _engine_setup(cfg, generator)
    perm = torch.tensor([0, 1, 3, 2])
    permuted = PseudoLabelState([permute_level(state.levels[0], perm), *state.levels[1:]])
    assert not torch.equal(permuted.levels[0].labels, state.levels[0].labels)
    results = []
    for s in (state, permuted):
        torch.manual_seed(3)
        results.append(compute_losses(model, fourier, batch, s.batch(torch.arange(B)), cfg,
                                      generator=torch.Generator().manual_seed(3))[0].as_floats())
    assert results[0] == pytest.approx(results[1], abs=1e-6)


# ---------------------------------------------------------------- loss step

def _debgcd_cfg(**debgcd):
    cfg = make_tiny_config()
    cfg.debgcd = DebGCDConfig(enabled=True, detector_hidden_dim=32, detector_bottleneck_dim=16, **debgcd)
    return cfg


@pytest.mark.parametrize("fourex", [False, True])  # E2, E3
def test_step_with_debgcd_is_finite_and_gradients_flow(fourex, generator):
    cfg = _debgcd_cfg(threshold=0.0, consistency_filter=False).set_components(fourex=fourex)
    model, fourier, batch, state = _engine_setup(cfg, generator)
    seen = []
    model.debgcd.register_forward_hook(lambda m, inp, out: seen.append(inp[0].shape))
    losses, _ = compute_losses(model, fourier, batch, state.batch(torch.arange(B)), cfg, generator=generator)

    assert seen == [(2 * B, model.backbone.feat_dim)]  # x(1), x(2) only: no Fourier views (C4)
    values = losses.as_floats()
    assert set(values) == {"total", "base", "low", "high", "cls", "sdl", "adl", "adl_utilisation"}
    assert all(math.isfinite(v) for v in values.values()) and values["adl_utilisation"] > 0
    w = 0.1 if fourex else 0.0
    torch.testing.assert_close(
        losses.total, losses.base + w * (losses.low + losses.high + losses.cls) + 0.01 * losses.sdl + 1.0 * losses.adl
    )

    losses.total.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
        else:
            assert param.grad is None, name
    for module in (model.debgcd.detector, model.debgcd.debiased, model.backbone.blocks[2:]):
        assert any(p.grad.abs().sum() > 0 for p in module.parameters() if p.requires_grad)


def test_debgcd_gradient_reaches_the_backbone_on_its_own(generator):
    cfg = _debgcd_cfg(threshold=0.0)
    model, fourier, batch, state = _engine_setup(cfg, generator)
    losses, _ = compute_losses(model, fourier, batch, state.batch(torch.arange(B)), cfg, generator=generator)
    (losses.sdl + losses.adl).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.blocks[2:].parameters())
    assert all(p.grad is None for p in model.head.parameters())  # DebGCD never touches z (plan §2.3)


def test_enabled_config_needs_heads(generator):
    cfg = _debgcd_cfg()
    model, fourier, batch, state = _engine_setup(make_tiny_config(), generator)
    with pytest.raises(ValueError, match="no DebGCD heads"):
        compute_losses(model, fourier, batch, state.batch(torch.arange(B)), cfg, generator=generator)
    with pytest.raises(ValueError, match="num_classes"):
        build_model(cfg, num_known_classes=NUM_KNOWN)


# ---------------------------------------------------------------- training loop

def _loop_setup(cfg):
    cfg.optim.batch_size, cfg.optim.epochs, cfg.loop.num_workers, cfg.eval.n_init = 16, 3, 0, 2
    cfg.loop.max_steps_per_epoch = 2
    train_src = DummyGCDDataset(num_classes=8, images_per_class=12, image_size=40)
    test_src = DummyGCDDataset(num_classes=8, images_per_class=3, image_size=40, seed=1)
    split = make_gcd_split(train_src.targets)
    return build_gcd_datasets(train_src, test_src, split, image_size=32), split


def _fit(cfg, datasets, split, epochs=2):
    torch.manual_seed(0)
    model = build_model(cfg, num_known_classes=split.num_known, num_classes=split.num_classes)
    torch.manual_seed(0)
    return model, fit(model, datasets, cfg, epochs=epochs, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize("row", ["E0", "E1"])
def test_debgcd_off_reproduces_pre_debgcd_golden_values(row):
    golden = json.loads(GOLDEN.read_text())[row]
    torch.manual_seed(0)  # the golden run seeded once, before building the datasets and the model
    cfg = make_tiny_config().set_components(fourex=row == "E1")
    datasets, split = _loop_setup(cfg)
    model = build_model(cfg, num_known_classes=split.num_known, num_classes=split.num_classes)
    result = fit(model, datasets, cfg, epochs=2, generator=torch.Generator().manual_seed(0))
    for got, want in zip([h.losses for h in result.history], golden["losses"]):
        assert got == pytest.approx(want, rel=1e-5)
    scores = result.evaluations[-1][1]
    assert {k: [v.all, v.known, v.novel] for k, v in scores.items()} == pytest.approx(golden["evals"])


def test_zero_weight_debgcd_matches_disabled_including_cluster_alignment():
    """λ_sdl = λ_adl = 0: same SelEx/FourEx losses and backbone over 3 epochs, so alignment is a no-op."""
    off = make_tiny_config()
    datasets, split = _loop_setup(off)
    on = _debgcd_cfg(lam_sdl=0.0, lam_adl=0.0)
    _loop_setup(on)
    model_off, res_off = _fit(off, datasets, split, epochs=3)
    model_on, res_on = _fit(on, datasets, split, epochs=3)
    keys = ("total", "base", "low", "high", "cls")
    for a, b in zip(res_off.history, res_on.history):
        assert {k: a.losses[k] for k in keys} == pytest.approx({k: b.losses[k] for k in keys}, abs=1e-6)
    on_params = dict(model_on.named_parameters())
    for name, p in model_off.named_parameters():
        torch.testing.assert_close(p, on_params[name], atol=1e-6, rtol=0)


def test_fit_with_debgcd_aligns_clusters_and_hides_unlabelled_labels(monkeypatch):
    cfg = _debgcd_cfg()
    datasets, split = _loop_setup(cfg)
    aligned = []
    original = train_mod.align_to_previous

    def spy(state, previous, num_known):
        out = original(state, previous, num_known)
        aligned.append(previous is not None)
        return out

    monkeypatch.setattr(train_mod, "align_to_previous", spy)
    _, clean = _fit(cfg, datasets, split)
    assert aligned == [False, True]  # no previous epoch to align to at epoch 0
    for h in clean.history:
        assert {"sdl", "adl", "adl_utilisation"} <= set(h.losses) and all(map(math.isfinite, h.losses.values()))

    unl = ~datasets.train.is_labelled  # rewrite unlabelled ground truth: training must not notice
    datasets.train.targets[unl] = torch.randint(0, split.num_classes, (int(unl.sum()),))
    _, leaked = _fit(cfg, datasets, split)
    assert [h.losses for h in clean.history] == [h.losses for h in leaked.history]


def test_dg08_guard_rejects_unanchored_known_classes(generator):
    cfg = make_tiny_config()
    feats = F.normalize(torch.randn(20, 8), dim=1)
    labelled = torch.zeros(20, dtype=torch.bool)
    labelled[:4] = True
    targets = torch.where(labelled, torch.zeros(20, dtype=torch.long), torch.full((20,), -1))  # class 1 unlabelled
    state = refresh_pseudo_labels(feats, labelled, targets, 2, 2, cfg, generator=generator)
    with pytest.raises(ValueError, match="DG-08"):
        train_mod.align_to_previous(state, None, num_known=2)


# ---------------------------------------------------------------- config and CLI

def test_components_and_rows():
    cfg = PlanoraSFConfig.for_dataset("cub")
    assert cfg.components() == {"fourex": True, "debgcd": False, "hypcd": False, "row": "E1"}
    assert cfg.set_components(fourex=False).components()["row"] == "E0"
    assert cfg.weights.cls == 0.0 and not cfg.cls_head.enabled
    assert cfg.set_components(debgcd=True).components()["row"] == "E2"
    assert cfg.set_components(fourex=True).components()["row"] == "E3"
    cfg.weights.low = 0.2
    assert cfg.components()["row"] is None  # not a pure matrix row
    for bad in ({"threshold": 1.0}, {"lam_sdl": -1.0}, {"tau_p": 0.0}, {"detector_layers": 0}):
        with pytest.raises(ValueError):
            DebGCDConfig(**bad)


def test_train_cli_e2_then_rescore(tmp_path, capsys):
    rc = train.main(["--dataset", "dummy", "--epochs", "2", "--max-steps", "2", "--batch-size", "16",
                     "--device", "cpu", "--num-workers", "0", "--checkpoint-dir", str(tmp_path),
                     "--no-use-fourex", "--use_debgcd"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["components"] == {"fourex": False, "debgcd": True, "hypcd": False, "row": "E2"}
    last = summary["history"][-1]["losses"]
    assert last["cls"] == 0.0 and {"sdl", "adl", "adl_utilisation"} <= set(last)
    assert set(summary["evaluations"][-1]) == {"epoch", "v1", "v2"}

    ckpt = torch.load(tmp_path / "last.pt", weights_only=False)
    assert any(k.startswith("debgcd.") for k in ckpt["model"]) and ckpt["config"]["debgcd"]["enabled"]
    assert evaluate.main(["--checkpoint", str(tmp_path / "last.pt"), "--dataset", "dummy", "--device", "cpu",
                          "--num-workers", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["config"]["debgcd"]["enabled"] is True
