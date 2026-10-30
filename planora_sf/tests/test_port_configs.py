"""The E4-port configs differ from the reference config in exactly the upstream keys (docs/hypcd_port.md)."""
from pathlib import Path

import pytest

from planora_sf.cli.evaluate import config_diff
from planora_sf.config import PlanoraSFConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "hypcd"
UPSTREAM = {  # train_HypSelEx.cmd + train_HypSelEx.py, H-xx ids in docs/hypcd_port.md
    "backbone.grad_from_block": 10,  # H-05
    "latent.dim": 8192,  # H-04
    "selex.reproduce_label_order_bug": True,  # H-10
    "hypcd.enabled": True,
    "hypcd.curvature": 0.1,  # H-02
    "hypcd.clip_r": 2.0,  # H-03
    "hypcd.alpha_epoch_offset": 1,  # H-06
    "hypcd.angle_similarity": "neg_euclidean",  # H-01
    "weights.low": 0.0,  # FourEx off, as E0
    "weights.high": 0.0,
    "weights.cls": 0.0,
    "cls_head.enabled": False,
}


@pytest.mark.parametrize("dataset, smoothing", [("cub", 1.0), ("scars", 0.5)])
def test_port_config_differs_from_reference_only_in_upstream_keys(dataset, smoothing):
    cfg = PlanoraSFConfig.from_yaml(CONFIGS / f"e4_port_{dataset}.yaml")
    diff = config_diff(cfg.to_dict(), PlanoraSFConfig.for_dataset(dataset).to_dict())
    expected = dict(UPSTREAM)
    if dataset == "scars":
        expected["selex.smoothing_alpha"] = smoothing  # H-07 (reference: SelEx README 1.0)
    assert {k: v[0] for k, v in diff.items()} == expected
    assert cfg.components() == {"fourex": False, "debgcd": False, "hypcd": True, "row": "E4"}
    assert cfg.dataset == dataset and cfg.selex.smoothing_alpha == smoothing and cfg.hypcd.alpha_max == 1.0
