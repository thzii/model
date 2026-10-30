"""Train -> re-score round trip on synthetic data (plan T1, R0)."""
import json
import math

import numpy as np
import pytest
import torch

from planora_sf.cli import evaluate, train


def _strict_loads(text: str):
    def reject(token):
        raise ValueError(f"non-standard JSON constant {token}")

    return json.loads(text, parse_constant=reject)


@pytest.fixture
def checkpoint(tmp_path, capsys):
    rc = train.main(["--dataset", "dummy", "--epochs", "1", "--max-steps", "2", "--batch-size", "16",
                     "--device", "cpu", "--num-workers", "0", "--checkpoint-dir", str(tmp_path)])
    assert rc == 0
    capsys.readouterr()  # drop the training summary
    return tmp_path / "last.pt"


def _evaluate(capsys, *args) -> dict:
    assert evaluate.main(list(args)) == 0
    return _strict_loads(capsys.readouterr().out)  # also proves the output is strict JSON


def test_evaluate_checkpoint_round_trip(checkpoint, capsys, tmp_path):
    out_file = tmp_path / "r0" / "eval.json"
    args = ["--checkpoint", str(checkpoint), "--dataset", "dummy", "--device", "cpu", "--num-workers", "0",
            "--eval-seeds", "0,1"]
    first = _evaluate(capsys, *args, "--out", str(out_file))
    assert set(first) == {"checkpoint", "dataset", "backbone", "evaluator", "crop", "config",
                          "config_diff_vs_cub_reference", "split", "split_sha256", "provenance",
                          "split_matches_training", "code", "per_seed", "summary", "logged", "pseudo_labels"}
    assert first["checkpoint"]["epoch"] == 0 and first["checkpoint"]["frozen_backbone"] is False
    assert [r["seed"] for r in first["per_seed"]] == [0, 1]
    assert set(first["per_seed"][0]) == {"seed", "v1", "v2", "routing", "per_class_v2", "structure", "tiers"}
    assert first["per_seed"][0]["tiers"] is None  # tiers are CUB-only
    assert "backbone.name" in first["config_diff_vs_cub_reference"]  # tiny ViT vs DINOv1 reference
    assert set(first["summary"]["v2.novel"]) == {"mean", "std"}
    assert json.loads(out_file.read_text()) == first

    second = _evaluate(capsys, *args)
    assert second["per_seed"] == first["per_seed"] and second["summary"] == first["summary"]


def test_evaluator_overrides_and_argument_checks(checkpoint, capsys):
    result = _evaluate(capsys, "--checkpoint", str(checkpoint), "--dataset", "dummy", "--device", "cpu",
                       "--num-workers", "0", "--clustering", "ss_kmeans", "--n-init", "2", "--max-iter", "3")
    assert (result["evaluator"]["clustering"], result["evaluator"]["n_init"], result["evaluator"]["max_iter"]) == (
        "ss_kmeans", 2, 3)
    assert result["summary"]["v2.all"]["std"] is None  # a single eval seed
    with pytest.raises(SystemExit):
        evaluate.main(["--dataset", "dummy"])  # neither --checkpoint nor --frozen-backbone
    with pytest.raises(SystemExit):
        evaluate.main(["--frozen-backbone", "--dataset", "dummy"])  # frozen backbone is CUB-only
    with pytest.raises(SystemExit):
        evaluate.main(["--checkpoint", str(checkpoint), "--frozen-backbone"])  # mutually exclusive


def test_checkpoint_dataset_must_match(checkpoint, tmp_path):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt["config"]["dataset"] = "cub"
    tagged = tmp_path / "cub_tagged.pt"
    torch.save(ckpt, tagged)
    with pytest.raises(SystemExit):
        evaluate.main(["--checkpoint", str(tagged), "--dataset", "dummy", "--device", "cpu"])


def test_root_defaults_to_the_dataset_env(monkeypatch):
    monkeypatch.setenv("CUB_ROOT", "/data/cub")
    monkeypatch.setenv("SCARS_ROOT", "/data/scars")
    parse = evaluate.build_parser().parse_args
    assert evaluate.resolve_root(parse(["--frozen-backbone"])) == "/data/cub"
    for name in ("scars", "stanford_cars"):
        args = parse(["--frozen-backbone", "--dataset", name])
        args.dataset = evaluate.canonical_dataset(args.dataset)
        assert evaluate.resolve_root(args) == "/data/scars"
    assert evaluate.resolve_root(parse(["--frozen-backbone", "--root", "/x"])) == "/x"
    monkeypatch.delenv("CUB_ROOT")
    assert evaluate.resolve_root(parse(["--frozen-backbone"])) is None


def test_json_safe_maps_nan_and_numpy_to_plain_json():
    payload = {"a": float("nan"), "b": [np.float32(0.5), np.int64(3), float("inf")], 7: (1, None)}
    safe = evaluate.json_safe(payload)
    assert safe == {"a": None, "b": [0.5, 3, None], "7": [1, None]}
    assert type(safe["b"][1]) is int and not any(isinstance(v, float) and not math.isfinite(v) for v in safe["b"])
    json.dumps(safe, allow_nan=False)
