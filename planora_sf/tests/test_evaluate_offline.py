"""cli.evaluate offline diagnostics (master plan Task #2a): PL-acc L0, the log <-> checkpoint link,
and re-scoring foreign backbone weights. Synthetic data only."""
import json

import pytest
import torch

from planora_sf.cli import evaluate, train
from planora_sf.models.backbone import foreign_backbone_state

_EVAL = ["--dataset", "dummy", "--device", "cpu", "--num-workers", "0"]


@pytest.fixture
def checkpoint(tmp_path, capsys):
    assert train.main(["--dataset", "dummy", "--epochs", "1", "--max-steps", "2", "--batch-size", "16",
                       "--device", "cpu", "--num-workers", "0", "--checkpoint-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    return tmp_path / "last.pt"


def _evaluate(capsys, *args) -> dict:
    assert evaluate.main([str(a) for a in args]) == 0
    return json.loads(capsys.readouterr().out)


def test_pseudo_label_accuracy_is_reported_per_seed_and_reproducible(checkpoint, capsys):
    args = ["--checkpoint", checkpoint, *_EVAL, "--eval-seeds", "0,1", "--pseudo-labels"]
    first = _evaluate(capsys, *args)
    pl = first["pseudo_labels"]
    assert pl["features"] == "train_view0" and pl["bssk"]["n_init"] == 1 and pl["bssk"]["max_iter"] == 10
    assert [r["seed"] for r in pl["per_seed"]] == [0, 1]
    assert all(0.0 <= r["v2"]["novel"] <= 1.0 for r in pl["per_seed"])
    assert pl["novel_minus_evaluator_novel"] == pytest.approx(
        pl["summary"]["v2.novel"]["mean"] - first["summary"]["v2.novel"]["mean"])
    assert isinstance(pl["flag_novel_below_evaluator"], bool)
    assert _evaluate(capsys, *args)["pseudo_labels"]["per_seed"] == pl["per_seed"]  # seeded crops and k-means
    assert first["split_matches_training"] is True and first["provenance"]["seed"] == 1


def test_link_check_against_logged_numbers(checkpoint, capsys):
    base = _evaluate(capsys, "--checkpoint", checkpoint, *_EVAL, "--eval-seeds", "0,1,2")
    mean = {k: 100 * base["summary"][f"v2.{k}"]["mean"] for k in ("all", "known", "novel")}
    logged = ",".join(f"{mean[k]:.2f}" for k in ("all", "known", "novel"))
    same = _evaluate(capsys, "--checkpoint", checkpoint, *_EVAL, "--eval-seeds", "0,1,2", "--logged-v2", logged)
    link = same["logged"]
    assert link["linked"] is True and abs(link["delta_all_pct"]) <= 0.006
    assert link["tolerance_pct"] == pytest.approx(2 * link["sigma_eval_pct"] + 0.5)
    shifted = mean["all"] + 30 if mean["all"] < 70 else mean["all"] - 30
    far = f"{shifted:.2f},{mean['known']:.2f},{mean['novel']:.2f}"
    assert _evaluate(capsys, "--checkpoint", checkpoint, *_EVAL, "--logged-v2", far)["logged"]["linked"] is False
    with pytest.raises(SystemExit):
        evaluate.main(["--checkpoint", str(checkpoint), *_EVAL, "--logged-v2", "60,70"])


def test_foreign_backbone_rescore_matches_the_native_checkpoint(checkpoint, tmp_path, capsys):
    ckpt = torch.load(checkpoint, weights_only=False)
    vit = {k[len("backbone.vit."):]: v for k, v in ckpt["model"].items() if k.startswith("backbone.vit.")}
    # Official-code shape: DDP + nn.Sequential(backbone, head), wrapped in a dict, plus a head tensor.
    foreign = {"state_dict": {**{f"module.0.{k}": v for k, v in vit.items()}, "module.1.last_layer.weight": torch.zeros(3)}}
    torch.save(foreign, tmp_path / "foreign.pt")
    seeds = ["--eval-seeds", "0,1"]
    native = _evaluate(capsys, "--checkpoint", checkpoint, *_EVAL, *seeds)
    rescored = _evaluate(capsys, "--foreign-backbone", tmp_path / "foreign.pt", *_EVAL, *seeds)
    mapping = rescored["checkpoint"]["foreign_mapping"]
    assert mapping["loaded"] == len(vit) and mapping["num_ignored"] == 1 and mapping["missing"] == []
    assert rescored["checkpoint"]["foreign_backbone"] is True and rescored["provenance"] is None
    assert rescored["per_seed"] == native["per_seed"]  # the evaluator only reads the backbone

    torch.save({k: v for k, v in list(foreign["state_dict"].items())[1:]}, tmp_path / "partial.pt")
    with pytest.raises(ValueError, match="missing"):
        evaluate.main(["--foreign-backbone", str(tmp_path / "partial.pt"), *_EVAL])
    with pytest.raises(SystemExit):
        evaluate.main(["--checkpoint", str(checkpoint), "--unsafe-load", *_EVAL])


def test_foreign_state_mapping_rules():
    target = {"blocks.0.w", "norm.weight"}
    state, report = foreign_backbone_state({"teacher": {"backbone.blocks.0.w": torch.ones(1),
                                                        "backbone.norm.weight": torch.ones(1),
                                                        "head.last_layer.weight": torch.ones(1)}}, target)
    assert set(state) == target and report["ignored_examples"] == ["head.last_layer.weight"]
    with pytest.raises(ValueError, match="two checkpoint keys"):
        foreign_backbone_state({"0.norm.weight": torch.ones(1), "norm.weight": torch.ones(1)}, target)
    with pytest.raises(ValueError, match="no state dict"):
        foreign_backbone_state({"epoch": 3, "args": {}}, target)
