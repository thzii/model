"""Stanford Cars (plan Phase 1): loader layouts, archive import and its safety checks, the SSB
split file, the factory and the CLIs. Fake images and devkit files only; real data is opt-in."""
import io
import json
import os
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from helpers import make_fake_scars
from planora_sf.cli import prepare_scars, train
from planora_sf.config import DataConfig, PlanoraSFConfig
from planora_sf.data import build_gcd_datasets, build_sources, build_split, make_gcd_split
from planora_sf.data.factory import canonical_dataset, default_root
from planora_sf.data.scars import ORIGINAL_MD5, StanfordCars, import_archives, safe_extract
from planora_sf.data.splits import ssb_classes, ssb_novel_tiers


# ---------------------------------------------------------------- loader

@pytest.mark.parametrize("layout, nested", [("torchvision", False), ("gcd", False), ("torchvision", True)])
def test_loader_reads_annotation_order_with_zero_based_classes(tmp_path, layout, nested):
    rows = make_fake_scars(tmp_path, layout=layout, nested=nested, greyscale_every=3)
    for split, is_train in (("train", True), ("test", False)):
        src = StanfordCars(tmp_path, train=is_train, check_files=True)
        assert src.paths == [f for f, _ in rows[split]]  # annotation order, as GCD/SelEx read it
        assert src.targets.tolist() == [c - 1 for _, c in rows[split]]
        img = src.load(0)  # index 0 is greyscale on disk
        assert img.mode == "RGB" and img.size == (24, 24)


def test_loader_errors_and_optional_test_labels(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_scars"):
        StanfordCars(tmp_path)
    make_fake_scars(tmp_path, test_labels=False)
    assert not StanfordCars.has_test_labels(tmp_path)
    with pytest.raises(FileNotFoundError):
        StanfordCars(tmp_path, train=False)
    (tmp_path / "stanford_cars" / "cars_train" / StanfordCars(tmp_path).paths[0]).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        StanfordCars(tmp_path, check_files=True)


# ---------------------------------------------------------------- SSB split

def test_scars_ssb_split_file():
    known, novel = ssb_classes("scars")
    tiers = ssb_novel_tiers("scars")
    assert len(known) == len(novel) == 98
    assert {t: len(v) for t, v in tiers.items()} == {"Hard": 7, "Medium": 15, "Easy": 76}
    assert sorted(known + novel) == list(range(196))  # a partition of the 0-based ids
    assert novel == tiers["Hard"] + tiers["Medium"] + tiers["Easy"]
    assert known[:5] == [1, 11, 25, 38, 46]  # as in scars_osr_splits.pkl (sha256 28d471f0...)
    assert ssb_classes("cub")[0][:3] == ssb_classes("cub")[0][:3]  # CUB unchanged
    with pytest.raises(KeyError):
        ssb_classes("aircraft")


def test_scars_scale_split_statistics():
    """196 classes at the real per-class counts would give 98 known classes and a half-labelled known set."""
    rng = np.random.default_rng(0)
    targets = np.repeat(np.arange(196), rng.integers(24, 68, 196))
    split = build_split("stanford_cars", type("S", (), {"targets": targets})(), DataConfig())
    known, _ = ssb_classes("scars")
    assert split.known_classes == tuple(known) and split.num_novel == 98
    n_known_images = int(np.isin(targets, known).sum())
    assert split.labelled_indices.size == int(0.5 * n_known_images)
    direct = make_gcd_split(targets, known_classes=known, novel_classes=ssb_classes("scars")[1])
    assert np.array_equal(direct.labelled_indices, split.labelled_indices)


# ---------------------------------------------------------------- factory and CLIs

def test_factory_with_and_without_test_labels(tmp_path):
    make_fake_scars(tmp_path / "a", num_classes=196, train_per_class=1, test_per_class=1, size=8)
    train_src, test_src = build_sources("stanford_cars", tmp_path / "a")
    split = build_split("scars", train_src, DataConfig())
    datasets = build_gcd_datasets(train_src, test_src, split, image_size=16)
    assert len(datasets.train) == 196 and datasets.test is not None and len(datasets.test) == 196
    make_fake_scars(tmp_path / "b", num_classes=196, train_per_class=1, test_per_class=1, size=8, test_labels=False)
    train_src, test_src = build_sources("scars", tmp_path / "b")
    assert test_src is None
    assert build_gcd_datasets(train_src, None, split, image_size=16).test is None
    with pytest.raises(ValueError, match="prepare_scars"):
        build_sources("scars", tmp_path / "b", download=True)
    with pytest.raises(ValueError):
        build_sources("scars")


def test_dataset_names_and_root_env(monkeypatch):
    assert canonical_dataset("stanford_cars") == "scars" and canonical_dataset("cub") == "cub"
    monkeypatch.setenv("SCARS_ROOT", "/data/scars")
    assert default_root("stanford_cars") == "/data/scars" and default_root("dummy") is None
    cfg = PlanoraSFConfig.for_dataset("scars")
    assert (cfg.fourier.cutoff, cfg.selex.smoothing_alpha, cfg.selex.balanced) == (27, 1.0, True)


def test_train_cli_accepts_stanford_cars(tmp_path, capsys, monkeypatch):
    make_fake_scars(tmp_path, num_classes=196, train_per_class=1, test_per_class=1, size=8)
    monkeypatch.setenv("SCARS_ROOT", str(tmp_path))
    from planora_sf.engine.train import FitResult

    built = {}

    def fake_fit(model, datasets, cfg, **kw):  # stops before any training; we only need the built inputs
        built.update(cfg=cfg, n=len(datasets.train))
        return FitResult()

    monkeypatch.setattr(train, "fit", fake_fit)
    assert train.main(["--dataset", "stanford_cars", "--device", "cpu", "--num-workers", "0"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert built["cfg"].dataset == "scars" and built["cfg"].fourier.cutoff == 27 and built["n"] == 196
    assert summary["split"]["known_classes"] == 98


# ---------------------------------------------------------------- archive import (prepare_scars)

def _tar(path, base, names):
    with tarfile.open(path, "w:gz") as tar:
        for name in names:
            tar.add(base / name, arcname=name)


def _original_release(tmp_path, **fake):
    """The four original-release files built from a fake tree: devkit.tgz, train.tgz, test.tgz, labels .mat."""
    src = tmp_path / "src"
    make_fake_scars(src, layout="gcd", **fake)
    out = tmp_path / "downloads"
    out.mkdir()
    _tar(out / "car_devkit.tgz", src, ["devkit/cars_train_annos.mat"])
    _tar(out / "cars_train.tgz", src, ["cars_train"])
    _tar(out / "cars_test.tgz", src, ["cars_test"])
    (out / "cars_test_annos_withlabels.mat").write_bytes((src / "devkit" / "cars_test_annos_withlabels.mat").read_bytes())
    return out


def test_prepare_scars_imports_the_original_release_files(tmp_path, capsys):
    out = _original_release(tmp_path, num_classes=4)
    files = [out / n for n in ("car_devkit.tgz", "cars_train.tgz", "cars_test.tgz", "cars_test_annos_withlabels.mat")]
    root = tmp_path / "data"
    rc = prepare_scars.main(["--root", str(root), "--check-files", "--from-archive", *map(str, files)])
    report = json.loads(capsys.readouterr().out)
    assert rc == 1  # 12 fake images, not 8,144: flagged, as download_cub does
    assert (report["train_images"], report["test_images"], report["classes"]) == (12, 8, 4)
    assert report["split"] is None  # the SSB split needs all 196 classes
    base = root / "stanford_cars"
    assert {p.name for p in base.iterdir()} == {"cars_train", "cars_test", "devkit", "cars_test_annos_withlabels.mat",
                                                "planora_manifest.json"}
    assert not (root / ".prepare_scars_staging").exists()
    assert [a["matches_original_release"] for a in report["archives"]] == [False] * 4  # fake content, real names
    manifest = json.loads((base / "planora_manifest.json").read_text())
    assert set(manifest["annotation_sha256"]) == {"devkit/cars_train_annos.mat", "cars_test_annos_withlabels.mat"}
    with pytest.raises(FileExistsError):
        import_archives(root, files)  # never overwrites an existing tree
    assert prepare_scars.main(["--root", str(root)]) == 1  # verify-only mode on the imported tree


def test_prepare_scars_imports_a_nested_zip_copy(tmp_path, capsys):
    """A Kaggle-style zip (cars_train/cars_train/...) plus the separately fetched devkit and test labels."""
    src = tmp_path / "src"
    make_fake_scars(src, layout="gcd", nested=True, num_classes=3)
    zpath = tmp_path / "archive.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for p in sorted((src / "cars_train").rglob("*.jpg")) + sorted((src / "cars_test").rglob("*.jpg")):
            zf.write(p, Path("stanford-cars-dataset") / p.relative_to(src))
    _tar(tmp_path / "car_devkit.tgz", src, ["devkit/cars_train_annos.mat"])
    labels = src / "devkit" / "cars_test_annos_withlabels.mat"
    rc = prepare_scars.main(["--root", str(tmp_path / "data"), "--check-files", "--from-archive", str(zpath),
                             str(tmp_path / "car_devkit.tgz"), str(labels)])
    report = json.loads(capsys.readouterr().out)
    assert rc == 1 and (report["train_images"], report["test_images"]) == (9, 6)
    assert report["archives"][0]["matches_original_release"] is None  # unknown name: no pinned MD5
    assert StanfordCars(tmp_path / "data", check_files=True).image_dir == tmp_path / "data" / "stanford_cars" / "cars_train"


def test_prepare_scars_without_test_labels_still_works(tmp_path, capsys):
    out = _original_release(tmp_path, num_classes=2)
    rc = prepare_scars.main(["--root", str(tmp_path / "d"), "--from-archive", str(out / "car_devkit.tgz"),
                             str(out / "cars_train.tgz")])
    assert rc == 1 and json.loads(capsys.readouterr().out)["test_images"] is None


def test_import_requires_train_images_and_annotations(tmp_path):
    out = _original_release(tmp_path, num_classes=2)
    with pytest.raises(FileNotFoundError, match="cars_train"):
        import_archives(tmp_path / "d", [out / "car_devkit.tgz"])
    assert not (tmp_path / "d" / "stanford_cars").exists() and not (tmp_path / "d" / ".prepare_scars_staging").exists()


def test_safe_extract_rejects_traversal_links_and_devices(tmp_path):
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(RuntimeError, match="outside"):
        safe_extract(evil_zip, tmp_path / "out1")
    link_zip = tmp_path / "link.zip"
    with zipfile.ZipFile(link_zip, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.external_attr = (0o120777 << 16)
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(RuntimeError, match="link"):
        safe_extract(link_zip, tmp_path / "out2")
    for kind, name in ((tarfile.SYMTYPE, "link"), (tarfile.CHRTYPE, "dev"), (tarfile.REGTYPE, "../../up.txt")):
        path = tmp_path / f"t{kind!r}.tgz"
        with tarfile.open(path, "w:gz") as tar:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.SYMTYPE:
                member.linkname = "/etc/passwd"
            data = b"x" if kind == tarfile.REGTYPE else b""
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
        with pytest.raises(RuntimeError):
            safe_extract(path, tmp_path / f"o{kind!r}")
    assert not (tmp_path / "escape.txt").exists() and not (tmp_path.parent / "up.txt").exists()


def test_original_release_md5s_are_pinned():
    assert ORIGINAL_MD5["cars_train.tgz"] == "065e5b463ae28d29e77c1b4b166cfe61"
    assert set(ORIGINAL_MD5) == {"car_devkit.tgz", "cars_train.tgz", "cars_test.tgz", "cars_test_annos_withlabels.mat"}


# ---------------------------------------------------------------- real data (opt-in)

@pytest.mark.skipif(not os.environ.get("SCARS_ROOT"), reason="set SCARS_ROOT to a prepared Stanford Cars copy")
def test_real_stanford_cars_counts_and_split():
    root = os.environ["SCARS_ROOT"]
    train_src = StanfordCars(root, train=True, check_files=True)
    assert len(train_src) == 8144 and len(set(train_src.targets.tolist())) == 196
    if StanfordCars.has_test_labels(root):
        assert len(StanfordCars(root, train=False, check_files=True)) == 8041
    summary = build_split("scars", train_src, DataConfig()).summary(train_src.targets)
    assert summary["known_classes"] == summary["novel_classes"] == 98
    assert 1900 <= summary["labelled_images"] <= 2100 and 6000 <= summary["unlabelled_images"] <= 6250  # GCD Table 1
