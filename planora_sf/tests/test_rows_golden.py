"""E0-E3 must stay exactly as they were before HypCD (plan Phase 6).

Golden values were recorded on commit 2909cc6, before any HypCD code existed:
tiny config, DummyGCDDataset, fit(epochs=2, max_steps=2), generator seed 0, torch.manual_seed(0).
"""
import json
from pathlib import Path

import pytest
import torch

from helpers import make_tiny_config
from planora_sf.config import DebGCDConfig
from planora_sf.data import DummyGCDDataset, build_gcd_datasets, make_gcd_split
from planora_sf.engine import fit
from planora_sf.models import build_model

GOLDEN = Path(__file__).parent / "data" / "rows_golden.json"
ROWS = {"E0": (False, False), "E1": (True, False), "E2": (False, True), "E3": (True, True)}


def run_row(row: str) -> dict:
    fourex, debgcd = ROWS[row]
    torch.manual_seed(0)
    cfg = make_tiny_config().set_components(fourex=fourex)
    if debgcd:
        cfg.debgcd = DebGCDConfig(enabled=True, detector_hidden_dim=32, detector_bottleneck_dim=16)
    cfg.optim.batch_size, cfg.optim.epochs, cfg.loop.num_workers, cfg.eval.n_init = 16, 3, 0, 2
    cfg.loop.max_steps_per_epoch = 2
    train_src = DummyGCDDataset(num_classes=8, images_per_class=12, image_size=40)
    test_src = DummyGCDDataset(num_classes=8, images_per_class=3, image_size=40, seed=1)
    split = make_gcd_split(train_src.targets)
    datasets = build_gcd_datasets(train_src, test_src, split, image_size=32)
    model = build_model(cfg, num_known_classes=split.num_known, num_classes=split.num_classes)
    result = fit(model, datasets, cfg, epochs=2, generator=torch.Generator().manual_seed(0))
    return {
        "losses": [h.losses for h in result.history],
        "evals": {k: [v.all, v.known, v.novel] for k, v in result.evaluations[-1][1].items()},
        "param_sums": {n: float(p.detach().double().sum()) for n, p in model.named_parameters()},
    }


@pytest.mark.parametrize("row", list(ROWS))
def test_row_matches_pre_hypcd_golden(row):
    golden = json.loads(GOLDEN.read_text())[row]
    got = run_row(row)
    assert len(got["losses"]) == len(golden["losses"])
    for g, w in zip(got["losses"], golden["losses"]):
        assert g == pytest.approx(w, rel=1e-5)
    assert got["evals"] == pytest.approx(golden["evals"])
    assert set(got["param_sums"]) == set(golden["param_sums"])
    for name, value in golden["param_sums"].items():
        assert got["param_sums"][name] == pytest.approx(value, rel=1e-4, abs=1e-4), name


if __name__ == "__main__":  # regenerate only on a commit whose behaviour is the reference
    GOLDEN.write_text(json.dumps({"recorded_on": "2909cc6", **{r: run_row(r) for r in ROWS}}, indent=1))
