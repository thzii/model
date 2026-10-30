"""Checkpoint and logged-metric audit (master plan Task #1). Synthetic checkpoints only."""
import json

import pytest
import torch

from helpers import make_fake_scars
from planora_sf.cli import audit_checkpoint as audit
from planora_sf.cli import train


@pytest.fixture
def native_e1(tmp_path, capsys):
    """A real cli.train checkpoint (dummy data, FourEx on, so it has the L_cls head)."""
    assert train.main(["--dataset", "dummy", "--epochs", "1", "--max-steps", "1", "--batch-size", "16",
                       "--device", "cpu", "--num-workers", "0", "--checkpoint-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    return tmp_path / "last.pt"


def _run(capsys, *args) -> dict:
    rc = audit.main([str(a) for a in args])
    out = capsys.readouterr().out
    return rc, json.loads(out)


def test_native_checkpoint_loads_safely_and_reports_everything(native_e1, capsys):
    rc, report = _run(capsys, native_e1)
    c = report["checkpoints"][0]
    assert rc == 0 and c["format"] == "native" and c["epoch"] == 0
    assert c["dataset"] == "dummy" and c["components"]["row"] == "E1" and c["components"]["latent_dim"] == 128
    assert c["heads"]["cls_head.fc"] == 4 and c["heads"]["head.last_layer"] == 128  # DummyGCDDataset: 8 classes, 4 known
    assert c["backbone"]["family"] == "tiny/other" and c["backbone"]["num_blocks"] == 4
    assert c["rescore"].startswith("cli.evaluate")


def _native_like(path, *, num_known, num_classes, dataset=None, cutoff=None, debgcd=True, extra=None):
    """A native-format checkpoint with real-size heads (no backbone needed for the vote)."""
    sd = {
        "cls_head.fc.weight": torch.zeros(num_known, 768),
        "head.last_layer.parametrizations.weight.original1": torch.zeros(65536, 256),
        "backbone.vit.cls_token": torch.zeros(1, 1, 768),
        "backbone.vit.patch_embed.proj.weight": torch.zeros(768, 3, 16, 16),
        **{f"backbone.vit.blocks.{i}.norm1.weight": torch.zeros(768) for i in range(12)},
    }
    if debgcd:
        sd["debgcd.debiased.last_layer.parametrizations.weight.original1"] = torch.zeros(num_classes, 768)
        sd["debgcd.detector.last_layer.parametrizations.weight.original1"] = torch.zeros(2 * num_known, 256)
    sd.update(extra or {})
    config = {"dataset": dataset, "fourier": {"cutoff": cutoff}, "backbone": {"name": "dinov1_vitb16"}}
    torch.save({"epoch": 199, "model": sd, "config": config}, path)
    return path


@pytest.mark.parametrize("dataset, known, classes, cutoff", [("cub", 100, 200, 23), ("scars", 98, 196, 27)])
def test_shape_votes_identify_the_dataset_even_without_config_dataset(tmp_path, capsys, dataset, known, classes,
                                                                     cutoff):
    path = _native_like(tmp_path / "x.pt", num_known=known, num_classes=classes, dataset=None)
    _, report = _run(capsys, path)
    c = report["checkpoints"][0]
    assert c["dataset"] == dataset and len(c["dataset_votes"]) == 3  # L_cls head, ψ_s, OVA detector
    assert c["backbone"]["family"] == "dinov1_vitb16"
    assert c["components"]["row"] == "E1"  # a partial config is completed with defaults
    path = _native_like(tmp_path / "y.pt", num_known=known, num_classes=classes, dataset=dataset, cutoff=cutoff)
    _, report = _run(capsys, path)
    assert report["checkpoints"][0]["dataset"] == dataset and len(report["checkpoints"][0]["dataset_votes"]) == 5


def test_conflicting_evidence_is_reported_not_resolved(tmp_path, capsys):
    path = _native_like(tmp_path / "x.pt", num_known=100, num_classes=200, dataset="scars")
    _, report = _run(capsys, path)
    assert report["checkpoints"][0]["dataset"] == "conflict: cub vs scars"


@pytest.mark.parametrize("k, dataset", [(200, "cub"), (196, "scars")])
def test_foreign_official_debgcd_checkpoint(tmp_path, capsys, k, dataset):
    """Official DebGCD saves nn.Sequential(backbone, DebGCDHead).state_dict() with old-style weight norm."""
    m = k // 2
    sd = {
        "0.cls_token": torch.zeros(1, 1, 768),
        "0.patch_embed.proj.weight": torch.zeros(768, 3, 16, 16),
        **{f"0.blocks.{i}.norm1.weight": torch.zeros(768) for i in range(12)},
        "1.last_layer.weight_g": torch.ones(k, 1), "1.last_layer.weight_v": torch.zeros(k, 768),
        "1.last_layer_deb.weight_v": torch.zeros(k, 768),
        "1.last_layer_ood.weight_v": torch.zeros(2 * m, 256),
    }
    torch.save(sd, tmp_path / "model.pt")
    _, report = _run(capsys, tmp_path / "model.pt")
    c = report["checkpoints"][0]
    assert c["format"] == "foreign (state_dict)" and c["dataset"] == dataset
    assert c["backbone"]["family"] == "dinov1_vitb16" and "key mapping" in c["rescore"]


def test_foreign_hypcd_projection_head_has_no_class_count(tmp_path, capsys):
    torch.save({"0.last_layer.weight_g": torch.ones(8192, 1), "0.last_layer.weight_v": torch.zeros(8192, 256)},
               tmp_path / "model_proj_head.pt")
    _, report = _run(capsys, tmp_path / "model_proj_head.pt")
    c = report["checkpoints"][0]
    assert c["heads"]["0.last_layer"] == 8192 and c["dataset"] == "unknown"


def test_dinov2_register_backbone_fingerprint(tmp_path, capsys):
    sd = {"cls_token": torch.zeros(1, 1, 768), "register_tokens": torch.zeros(1, 4, 768),
          "patch_embed.proj.weight": torch.zeros(768, 3, 14, 14),
          **{f"blocks.{i}.norm1.weight": torch.zeros(768) for i in range(12)}}
    torch.save(sd, tmp_path / "v2.pt")
    _, report = _run(capsys, tmp_path / "v2.pt")
    assert report["checkpoints"][0]["backbone"]["family"] == "dinov2_vitb14_reg"


def test_unsafe_objects_are_refused_without_the_flag(tmp_path, capsys):
    import collections

    torch.save({"model": {"w": torch.zeros(2)}, "config": {}, "odd": collections.OrderedDict(a=Exception("x"))},
               tmp_path / "odd.pt")
    rc, report = _run(capsys, tmp_path / "odd.pt")
    assert rc == 1 and "--unsafe-load" in report["checkpoints"][0]["error"]
    rc, report = _run(capsys, tmp_path / "odd.pt", "--unsafe-load")
    assert rc == 0 and report["checkpoints"][0]["format"] == "native"


# ---------------------------------------------------------------- w from logs

def test_w_interval_on_the_reported_runs():
    fourex = audit.w_interval("62.39", "77.72", "54.72")
    assert fourex["w"] == pytest.approx(0.33348, abs=1e-5)
    assert fourex["w_min"] < 0.3335 < fourex["w_max"] and fourex["w_max"] - fourex["w_min"] < 0.001
    baseline = audit.w_interval("60.29", "77.31", "52.06")
    assert baseline["w"] == pytest.approx(0.32594, abs=1e-5) and baseline["w_max"] < 0.3329  # below CUB's range
    coarse = audit.w_interval("60", "77", "52")  # one fewer decimal: a wider interval
    assert coarse["w_max"] - coarse["w_min"] > 0.03
    assert not audit.w_interval("90", "77", "52")["consistent_ordering"]
    with pytest.raises(ValueError):
        audit.w_interval("60", "52", "52")


def test_logged_rows_against_references_and_eval_set_consistency(capsys):
    rc, report = _run(capsys, "--logged", "v1=62.39,77.72,54.72", "--logged", "v2=60.10,74.05,53.12")
    rows = report["logged"]["rows"]
    assert rc == 0 and [r["label"] for r in rows] == ["v1", "v2"]
    assert all("cub" in r["compatible_datasets"] for r in rows)
    assert report["logged"]["all_rows_share_one_eval_set"] is True
    _, report = _run(capsys, "--logged", "debias=64.66,81.45,56.26", "--logged", "baseline=60.29,77.31,52.06")
    assert report["logged"]["all_rows_share_one_eval_set"] is False
    assert "cub" not in report["logged"]["rows"][1]["compatible_datasets"]


def test_exact_scars_w_from_data_and_checkpoint_match(tmp_path, capsys):
    make_fake_scars(tmp_path / "scars", num_classes=196, train_per_class=2, test_per_class=1, size=8)
    path = _native_like(tmp_path / "x.pt", num_known=98, num_classes=196, dataset="scars", cutoff=27)
    # Fake data: every class has 2 images, so 98 known x 2 = 196 known images, 98 labelled; w = 98 / (392 - 98).
    _, report = _run(capsys, path, "--scars-root", tmp_path / "scars", "--logged", "v2=50.00,60.00,45.00")
    ref = report["references"]["scars"]
    assert ref["w_exact"] == pytest.approx(98 / 294, abs=1e-5) and ref["split"]["labelled_images"] == 98
    # Equal class sizes give w = 1/3 exactly, inside CUB's range too: w alone cannot separate them (shape votes can).
    assert report["logged"]["rows"][0]["compatible_datasets"] == ["cub", "scars"]
    assert report["logged"]["matches_checkpoint_dataset"] is True


def test_needs_some_input():
    with pytest.raises(SystemExit):
        audit.main([])
    with pytest.raises(SystemExit):
        audit.main(["--logged", "1,2"])
