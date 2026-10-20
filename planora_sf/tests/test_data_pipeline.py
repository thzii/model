"""Sources, transforms and GCD datasets, including an offline mock of the CUB layout."""
import io
import tarfile

import numpy as np
import pytest
import torch
from torchvision import transforms

from helpers import make_fake_cub
from planora_sf.data import (
    CUB200,
    DummyGCDDataset,
    GCDTestDataset,
    GCDTrainDataset,
    MultiViewTransform,
    build_gcd_datasets,
    build_sources,
    build_split,
    build_transforms,
    download_cub,
    make_gcd_split,
)
from planora_sf.config import DataConfig
from planora_sf.data.sources import _safe_extract


def test_dummy_source_is_deterministic_and_class_dependent():
    src = DummyGCDDataset(num_classes=3, images_per_class=[2, 3, 4], image_size=16)
    assert len(src) == 9 and src.targets.tolist() == [0, 0, 1, 1, 1, 2, 2, 2, 2]
    a, b = np.asarray(src.load(4)), np.asarray(src.load(4))
    assert a.shape == (16, 16, 3) and a.dtype == np.uint8 and np.array_equal(a, b)
    means = [np.asarray(src.load(i)).reshape(-1, 3).mean(0) for i in (0, 2, 5)]
    assert np.abs(means[0] - means[1]).max() > 5 or np.abs(means[1] - means[2]).max() > 5


def test_transforms_shapes_and_randomness():
    image = DummyGCDDataset(num_classes=1, images_per_class=1, image_size=50).load(0)
    train_tf, test_tf = build_transforms(32)
    assert train_tf(image).shape == test_tf(image).shape == (3, 32, 32)
    assert torch.equal(test_tf(image), test_tf(image))  # deterministic
    assert any(not torch.equal(train_tf(image), train_tf(image)) for _ in range(5))  # random crop/flip
    _, raw_tf = build_transforms(32, normalize=False)
    x = raw_tf(image)
    assert x.min() >= 0 and x.max() <= 1
    views = MultiViewTransform(train_tf, n_views=2)(image)
    assert len(views) == 2 and views[0].shape == (3, 32, 32)


def test_train_dataset_is_labelled_first_and_hides_unlabelled_targets():
    src = DummyGCDDataset(num_classes=6, images_per_class=5, image_size=8)
    split = make_gcd_split(src.targets)
    ds = GCDTrainDataset(src, split, transforms.ToTensor())
    assert len(ds) == 30 and ds.num_labelled == int(0.5 * 15) == 7
    assert ds.is_labelled[:7].all() and not ds.is_labelled[7:].any()
    assert np.array_equal(ds.source_indices, np.concatenate([split.labelled_indices, split.unlabelled_indices]))
    assert torch.equal(ds.targets, torch.as_tensor(split.map_targets(src.targets[ds.source_indices])))
    assert (ds.train_targets[7:] == -1).all() and torch.equal(ds.train_targets[:7], ds.targets[:7])
    image, target, is_labelled, index = ds[3]
    assert image.shape == (3, 8, 8) and is_labelled and index == 3 and target == ds.targets[3]
    assert ds.with_transform(None)[0][0].size == (8, 8) and ds.transform is not None  # clone, original untouched


def test_test_dataset_maps_all_classes_with_train_order():
    train_src, test_src = DummyGCDDataset(4, 6, 8), DummyGCDDataset(4, 2, 8, seed=1)
    split = make_gcd_split(train_src.targets, known_classes=[3, 1])
    test = GCDTestDataset(test_src, split, transforms.ToTensor())
    assert test.targets.tolist() == [2, 2, 1, 1, 3, 3, 0, 0]  # class 3 -> 0, 1 -> 1, novel 0 -> 2, 2 -> 3


def test_cub_parser_on_mock_layout(tmp_path):
    rows = make_fake_cub(tmp_path, num_classes=6, train_per_class=3, test_per_class=2)
    train, test = CUB200(tmp_path, train=True, check_files=True), CUB200(tmp_path, train=False)
    assert len(train) == 18 and len(test) == 12
    expected_train = [r for r in rows if r[2] == 1]
    assert train.image_ids.tolist() == [r[0] for r in expected_train]  # images.txt order, test ids skipped
    assert train.targets.tolist() == [r[1] - 1 for r in expected_train]  # 0-based classes
    img = train.load(0)
    assert img.mode == "RGB" and img.size == (48, 48)
    assert download_cub(tmp_path) == tmp_path / "CUB_200_2011"  # already present: no network access


def test_cub_parser_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        CUB200(tmp_path / "missing")
    rows = make_fake_cub(tmp_path, num_classes=2, train_per_class=1, test_per_class=1)
    assert rows[0] == (1, 1, 1)  # image 1 is a training image of class 1
    (tmp_path / "CUB_200_2011" / "images" / "001.Class_1" / "img_00001.jpg").unlink()
    assert len(CUB200(tmp_path, train=True)) == 2  # metadata alone still parses
    with pytest.raises(FileNotFoundError, match="1 CUB images missing"):
        CUB200(tmp_path, train=True, check_files=True)
    CUB200(tmp_path, train=False, check_files=True)  # the test split is complete


def test_download_checks_md5_without_network(tmp_path):
    (tmp_path / "CUB_200_2011.tgz").write_bytes(b"not the real archive")
    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        download_cub(tmp_path)


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        data = b"x"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(RuntimeError, match="outside"):
        _safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_factory_builds_all_three_datasets():
    train_src, test_src = build_sources("dummy")
    split = build_split("dummy", train_src, DataConfig())
    ds = build_gcd_datasets(train_src, test_src, split, image_size=32)
    views, _, _, _ = ds.train[0]
    assert isinstance(views, list) and len(views) == 2 and views[0].shape == (3, 32, 32)
    assert ds.train_eval[0][0].shape == (3, 32, 32) and torch.equal(ds.train_eval[0][0], ds.train_eval[0][0])
    assert len(ds.test) == len(test_src) and ds.split is split
    with pytest.raises(ValueError):
        build_sources("cub")  # needs a root
    with pytest.raises(ValueError):
        build_sources("imagenet")
