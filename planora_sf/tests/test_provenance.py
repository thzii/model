"""Run provenance, summary.json and periodic checkpoints (master plan Task #2b). Synthetic data only."""
import json

import pytest
import torch

from planora_sf.cli import audit_checkpoint, train
from planora_sf.config import PlanoraSFConfig
from planora_sf.data import DummyGCDDataset, make_gcd_split
from planora_sf.engine import provenance

_TRAIN = ["--dataset", "dummy", "--epochs", "2", "--max-steps", "1", "--batch-size", "16", "--device", "cpu",
          "--num-workers", "0"]


def _train(capsys, tmp_path, *extra) -> dict:
    assert train.main([*_TRAIN, "--checkpoint-dir", str(tmp_path), *extra]) == 0
    return json.loads(capsys.readouterr().out)


def test_summary_and_checkpoint_carry_the_same_provenance(tmp_path, capsys):
    summary = _train(capsys, tmp_path / "a", "--seed", "7", "--label", "S2-E0")
    on_disk = json.loads((tmp_path / "a" / "summary.json").read_text())
    assert on_disk == summary
    assert summary["label"] == "S2-E0" and summary["dataset"] == "dummy" and summary["backbone"] == "tiny"
    assert summary["final"] == summary["evaluations"][-1] and summary["epochs_run"] == 2
    prov = summary["provenance"]
    assert prov["seed"] == 7 and prov["hub"] is None and prov["torch"] == torch.__version__
    assert len(prov["config_sha256"]) == len(prov["split_sha256"]) == 64
    assert prov["commit"] is None or len(prov["commit"]) == 40

    ckpt = torch.load(tmp_path / "a" / "last.pt", weights_only=False)
    assert ckpt["provenance"] == prov
    cfg = PlanoraSFConfig.from_dict(ckpt["config"])
    assert provenance.config_sha256(cfg) == prov["config_sha256"]  # the hash is of the saved config

    report = audit_checkpoint.audit_checkpoint(tmp_path / "a" / "last.pt", unsafe=True)
    assert report["provenance"]["seed"] == 7 and report["provenance"]["split_sha256"] == prov["split_sha256"]


def test_hashes_are_stable_and_track_config_not_seed(tmp_path, capsys):
    a = _train(capsys, tmp_path / "a", "--seed", "1")["provenance"]
    b = _train(capsys, tmp_path / "b", "--seed", "2")["provenance"]
    c = _train(capsys, tmp_path / "c", "--seed", "1", "--no-use-fourex")
    assert a["config_sha256"] == b["config_sha256"] and a["split_sha256"] == b["split_sha256"]
    assert c["provenance"]["config_sha256"] != a["config_sha256"]
    assert c["label"] == "E0"  # default label: the experiment row


def test_split_hash_depends_on_the_labelled_set():
    targets = DummyGCDDataset().targets
    base = provenance.split_sha256(make_gcd_split(targets))
    assert provenance.split_sha256(make_gcd_split(targets)) == base
    assert provenance.split_sha256(make_gcd_split(targets, seed=1)) != base


def test_git_state_never_fails(monkeypatch):
    monkeypatch.setattr(provenance, "_git", lambda *args: None)
    assert provenance.git_state() == {"commit": None, "commit_dirty": None}
    outputs = iter(["a" * 40, " M src/x.py"])
    monkeypatch.setattr(provenance, "_git", lambda *args: next(outputs))
    assert provenance.git_state() == {"commit": "a" * 40, "commit_dirty": True}


def test_keep_checkpoint_every_saves_epoch_copies(tmp_path, capsys):
    _train(capsys, tmp_path, "--keep-checkpoint-every", "1")
    assert sorted(p.name for p in tmp_path.glob("*.pt")) == ["epoch_000.pt", "epoch_001.pt", "last.pt"]
    first = torch.load(tmp_path / "epoch_000.pt", weights_only=False)
    assert first["epoch"] == 0 and first["provenance"]["seed"] == 1


def test_keep_checkpoint_every_rejects_negative():
    with pytest.raises(ValueError):
        PlanoraSFConfig.from_dict({"loop": {"keep_checkpoint_every": -1}})
