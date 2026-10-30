"""Bounding-box crop evaluation (master plan Task #3): crop geometry, dataset boxes, the evaluate
flags and the paired comparison. Synthetic images only."""
import json

import numpy as np
import pytest
import yaml
from PIL import Image

from helpers import make_fake_cub, make_fake_scars
from planora_sf.cli import compare_evals, evaluate, train
from planora_sf.cli.common import smoke_config
from planora_sf.data import factory
from planora_sf.data.bbox import BBoxCropSource, CropSpec, degenerate_boxes, expand_box
from planora_sf.data.scars import StanfordCars
from planora_sf.data.sources import CUB200


def test_tight_and_square_boxes():
    assert expand_box((10, 20, 30, 30), CropSpec(0.0, "tight"), (100, 100)) == (10, 20, 30, 30)
    assert expand_box((10, 20, 30, 30), CropSpec(0.1, "tight"), (100, 100)) == (8, 19, 32, 31)
    # 20 x 10 box -> 20 x 20 square around the same centre (20, 25)
    assert expand_box((10, 20, 30, 30), CropSpec(0.0, "square"), (100, 100)) == (10, 15, 30, 35)
    # A square that would leave the image is shifted inside it, not shrunk
    assert expand_box((0, 0, 20, 10), CropSpec(0.0, "square"), (100, 100)) == (0, 0, 20, 20)
    assert expand_box((90, 0, 100, 40), CropSpec(0.0, "square"), (100, 100)) == (60, 0, 100, 40)
    # Larger than the image in one direction: clipped to the image there
    assert expand_box((0, 40, 100, 60), CropSpec(0.0, "square"), (100, 50)) == (0, 0, 100, 50)
    # Boxes are clipped to the image first; no area left -> None (the caller keeps the full image)
    assert expand_box((120, 0, 130, 10), CropSpec(), (100, 100)) is None
    assert degenerate_boxes(np.array([[0, 0, 10, 10], [5, 5, 5, 9]])) == 1
    with pytest.raises(ValueError):
        CropSpec(shape="circle")


def test_scars_and_cub_boxes_are_read_as_left_top_right_bottom(tmp_path):
    make_fake_scars(tmp_path / "cars", size=40)
    cars = StanfordCars(tmp_path / "cars")
    assert cars.bboxes.shape == (len(cars), 4) and (cars.bboxes == [0, 0, 20, 20]).all()  # devkit 1..20, 1-based
    rows = make_fake_cub(tmp_path / "cub", size=48)
    cub = CUB200(tmp_path / "cub")
    train_ids = [i for i, _, t in rows if t]
    expected = [[i % 5, 2, i % 5 + 24, 2 + 16] for i in train_ids]
    assert cub.bboxes.tolist() == expected


def test_crop_source_keeps_order_and_targets(tmp_path):
    make_fake_scars(tmp_path, size=40)
    train_src, test_src = factory.build_sources("scars", tmp_path, crop=CropSpec(0.1, "square"))
    assert isinstance(train_src, BBoxCropSource) and isinstance(test_src, BBoxCropSource)
    raw = StanfordCars(tmp_path)
    assert (train_src.targets == raw.targets).all() and len(train_src) == len(raw)
    assert train_src.load(0).size == (24, 24) and raw.load(0).size == (40, 40)  # (0,0,20,20) + 2 px, squared
    with pytest.raises(ValueError, match="bounding boxes"):
        factory.build_sources("dummy", crop=CropSpec())
    img = Image.new("RGB", (40, 40))

    class _NoBoxes:
        targets = np.zeros(1, dtype=np.int64)
        bboxes = np.array([[50.0, 50.0, 60.0, 60.0]])  # outside the image

        def __len__(self):
            return 1

        def load(self, index):
            return img

    assert BBoxCropSource(_NoBoxes(), CropSpec()).load(0).size == (40, 40)


@pytest.fixture
def cars_checkpoint(tmp_path, capsys):
    """Tiny-ViT run on a fake Stanford Cars tree (4 classes, no SSB lists), so crops have boxes."""
    root = tmp_path / "cars"
    make_fake_scars(root, num_classes=4, train_per_class=6, size=40)
    cfg = smoke_config().to_dict()
    cfg["dataset"], cfg["data"]["use_ssb_splits"] = "scars", False
    (tmp_path / "tiny_scars.yaml").write_text(yaml.safe_dump(cfg))
    assert train.main(["--dataset", "scars", "--root", str(root), "--config", str(tmp_path / "tiny_scars.yaml"),
                       "--epochs", "1", "--max-steps", "1", "--batch-size", "8", "--device", "cpu",
                       "--num-workers", "0", "--checkpoint-dir", str(tmp_path / "run")]) == 0
    capsys.readouterr()
    return root, tmp_path / "run" / "last.pt"


def test_evaluate_crop_then_compare(cars_checkpoint, tmp_path, capsys):
    root, ckpt = cars_checkpoint
    common = ["--checkpoint", str(ckpt), "--dataset", "scars", "--root", str(root), "--device", "cpu",
              "--num-workers", "0", "--eval-seeds", "0,1,2"]
    assert evaluate.main([*common, "--out", str(tmp_path / "full.json")]) == 0
    assert evaluate.main([*common, "--crop-bbox", "--out", str(tmp_path / "crop.json")]) == 0
    capsys.readouterr()
    full, crop = (json.loads((tmp_path / f"{n}.json").read_text()) for n in ("full", "crop"))
    assert full["crop"] is None and crop["crop"] == {"margin": 0.1, "shape": "square", "degenerate_boxes": 0}
    assert full["split_sha256"] == crop["split_sha256"] and full["split_matches_training"] is True

    assert compare_evals.main([str(tmp_path / "full.json"), str(tmp_path / "crop.json"),
                               "--out", str(tmp_path / "cmp.json")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads((tmp_path / "cmp.json").read_text())
    all_ = result["metrics"]["v2.all"]
    assert result["seeds"] == [0, 1, 2] and all_["n"] == 3
    expected = [100 * (c["v2"]["all"] - f["v2"]["all"]) for f, c in zip(full["per_seed"], crop["per_seed"])]
    assert all_["deltas"] == pytest.approx(expected)
    assert result["other"]["crop"]["shape"] == "square" and isinstance(result["gate3_supported_increase"], bool)

    other = dict(crop, checkpoint={**crop["checkpoint"], "sha256": "0" * 64})
    (tmp_path / "other.json").write_text(json.dumps(other))
    assert compare_evals.main([str(tmp_path / "full.json"), str(tmp_path / "other.json")]) == 2


@pytest.mark.parametrize("deltas, verdict", [
    ([2.0, 3.0, 2.5], "supported increase"),
    ([-3.0, -2.5, -2.0], "supported decrease"),
    ([2.0, -1.0, 3.0], "not detected at n = 3"),
    ([0.5, 0.6, 0.7], "not detected at n = 3"),  # below the 1-point floor
    ([1.0, 9.0, 5.0], "not detected at n = 3"),  # mean 5 < 2 sd = 8
    ([4.0], "not detected at n = 1"),
])
def test_paired_verdicts(deltas, verdict):
    stats = compare_evals.paired_stats(deltas)
    assert stats["verdict"] == verdict
    if len(deltas) > 1:
        assert stats["threshold"] == pytest.approx(max(1.0, 2 * np.std(deltas, ddof=1)))
