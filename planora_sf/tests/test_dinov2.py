"""DINOv2 integration (master plan Task #4): pinned hub source, structure checks, the --backbone flag
and the reference config. A stand-in replaces torch.hub; the real download test is opt-in."""
import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from helpers import make_fake_scars
from planora_sf.cli import evaluate, train
from planora_sf.cli.evaluate import config_diff
from planora_sf.config import BackboneConfig, PlanoraSFConfig
from planora_sf.models import backbone as backbone_mod

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "dinov2" / "e0_scars.yaml"


class FakeHubViT(nn.Module):
    """The attributes BackboneAdapter and check_hub_structure read, with a cheap forward."""

    def __init__(self, embed_dim=768, patch=14, registers=4, depth=12, out_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_register_tokens = registers
        self.patch_embed = nn.Module()
        self.patch_embed.patch_size = (patch, patch)
        self.proj = nn.Linear(3, out_dim or embed_dim)
        self.blocks = nn.ModuleList(nn.Linear(2, 2) for _ in range(depth))
        self.norm = nn.LayerNorm(2)

    def forward(self, x):
        return self.proj(x.mean(dim=(2, 3)))


@pytest.fixture
def fake_hub(monkeypatch):
    calls, model = [], {"factory": FakeHubViT}

    def load(repo, entry, **kwargs):
        calls.append((repo, entry, kwargs))
        return model["factory"]()

    monkeypatch.setattr(backbone_mod.torch.hub, "load", load)
    return calls, model


def test_hub_sources_are_pinned_commits():
    for name, (repo, _) in backbone_mod.HUB_ENTRYPOINTS.items():
        ref = repo.split(":")[1]
        assert len(ref) == 40 and all(c in "0123456789abcdef" for c in ref), name
    assert backbone_mod.hub_source("dinov2_vitb14_reg").startswith("facebookresearch/dinov2:7764ea0")
    assert backbone_mod.hub_source("tiny") is None


def test_build_dinov2_loads_the_pinned_entry_and_freezes(fake_hub):
    calls, _ = fake_hub
    adapter = backbone_mod.build_backbone(BackboneConfig(name="dinov2_vitb14_reg"))
    repo, entry, kwargs = calls[0]
    assert (repo, entry) == backbone_mod.HUB_ENTRYPOINTS["dinov2_vitb14_reg"]
    assert kwargs == {"pretrained": True, "trust_repo": True, "skip_validation": True}
    assert adapter.feat_dim == 768 and adapter.trainable_block_indices() == [9, 10, 11]
    assert adapter(torch.zeros(2, 3, 224, 224)).shape == (2, 768)


@pytest.mark.parametrize("kwargs, message", [
    ({"registers": 0}, "num_register_tokens"),
    ({"depth": 6}, "num_blocks"),
    ({"patch": 16}, "patch_size"),
    ({"out_dim": 384}, "CLS output"),
])
def test_structure_mismatches_are_refused(fake_hub, kwargs, message):
    _, model = fake_hub
    model["factory"] = lambda: FakeHubViT(**kwargs)
    with pytest.raises(ValueError, match=message):
        backbone_mod.build_backbone(BackboneConfig(name="dinov2_vitb14_reg"))


def test_image_size_must_fit_the_patch(fake_hub):
    with pytest.raises(ValueError, match="multiple of the patch size"):
        backbone_mod.build_backbone(BackboneConfig(name="dinov2_vitb14_reg", image_size=200))


def test_reference_config_differs_only_in_backbone_and_e0_toggles():
    cfg = PlanoraSFConfig.from_yaml(CONFIG)
    diff = config_diff(cfg.to_dict(), PlanoraSFConfig.for_dataset("scars").to_dict())
    assert {k: v[0] for k, v in diff.items()} == {
        "backbone.name": "dinov2_vitb14_reg", "weights.low": 0.0, "weights.high": 0.0, "weights.cls": 0.0,
        "cls_head.enabled": False,
    }
    assert cfg.components()["row"] == "E0" and cfg.dataset == "scars" and cfg.fourier.cutoff == 27


def test_frozen_dinov2_floor_through_the_cli(fake_hub, tmp_path, capsys):
    make_fake_scars(tmp_path, num_classes=196, train_per_class=2, test_per_class=1, size=8)
    assert evaluate.main(["--frozen-backbone", "--backbone", "dinov2_vitb14_reg", "--dataset", "scars",
                          "--root", str(tmp_path), "--device", "cpu", "--num-workers", "0"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["backbone"] == "dinov2_vitb14_reg" and result["checkpoint"]["frozen_backbone"] is True
    assert result["config_diff_vs_cub_reference"]["backbone.name"] == ["dinov2_vitb14_reg", "dinov1_vitb16"]


def test_backbone_flag_checks(tmp_path, capsys):
    with pytest.raises(SystemExit):
        train.main(["--dataset", "dummy", "--backbone", "dinov2_vitb14_reg"])
    with pytest.raises(SystemExit):
        evaluate.main(["--frozen-backbone", "--dataset", "dummy", "--backbone", "dinov2_vitb14_reg"])
    assert train.main(["--dataset", "dummy", "--epochs", "1", "--max-steps", "1", "--batch-size", "16",
                       "--device", "cpu", "--num-workers", "0", "--checkpoint-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit):  # the checkpoint names its own backbone (tiny)
        evaluate.main(["--checkpoint", str(tmp_path / "last.pt"), "--dataset", "dummy",
                       "--backbone", "dinov2_vitb14_reg"])


@pytest.mark.skipif(os.environ.get("PLANORA_SF_ALLOW_DOWNLOAD") != "1",
                    reason="downloads DINOv2 weights via torch.hub; set PLANORA_SF_ALLOW_DOWNLOAD=1 to run")
def test_real_dinov2_backbone_loads():
    adapter = backbone_mod.build_backbone(BackboneConfig(name="dinov2_vitb14_reg"))
    assert adapter.feat_dim == 768 and adapter.num_blocks == 12
    assert adapter.trainable_block_indices() == [9, 10, 11]
    adapter.eval()
    with torch.no_grad():
        assert adapter(torch.zeros(2, 3, 224, 224)).shape == (2, 768)
