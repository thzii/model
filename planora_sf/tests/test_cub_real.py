"""Real CUB-200-2011 checks. They download about 1.1 GB, so they are opt-in:

    PLANORA_SF_ALLOW_DOWNLOAD=1 CUB_ROOT=/data/cub python -m pytest tests/test_cub_real.py
"""
import os
from pathlib import Path

import pytest

from planora_sf.config import DataConfig
from planora_sf.data import CUB200, build_gcd_datasets, build_split

pytestmark = pytest.mark.skipif(
    os.environ.get("PLANORA_SF_ALLOW_DOWNLOAD") != "1",
    reason="downloads CUB-200-2011 (~1.1 GB); set PLANORA_SF_ALLOW_DOWNLOAD=1 (and optionally CUB_ROOT)",
)


@pytest.fixture(scope="module")
def cub_root(tmp_path_factory):
    return Path(os.environ.get("CUB_ROOT") or tmp_path_factory.mktemp("cub"))


@pytest.fixture(scope="module")
def cub_train(cub_root):
    return CUB200(cub_root, train=True, download=True, check_files=True)


def test_official_counts(cub_root, cub_train):
    test = CUB200(cub_root, train=False, check_files=True)
    assert (len(cub_train), len(test)) == (5994, 5794)
    assert sorted(set(cub_train.targets.tolist())) == list(range(200))


def test_ssb_gcd_split_counts(cub_train):
    split = build_split("cub", cub_train, DataConfig())
    known_images = int(sum(t in set(split.known_classes) for t in cub_train.targets.tolist()))
    assert split.labelled_indices.size == int(0.5 * known_images)
    assert split.labelled_indices.size + split.unlabelled_indices.size == 5994
    assert 1450 < split.labelled_indices.size < 1550  # SX-S Table: 1.5K labelled, 4.5K unlabelled


def test_items_load_at_224(cub_root, cub_train):
    split = build_split("cub", cub_train, DataConfig())
    datasets = build_gcd_datasets(cub_train, CUB200(cub_root, train=False), split, image_size=224)
    views, target, is_labelled, index = datasets.train[0]
    assert views[0].shape == (3, 224, 224) and is_labelled and index == 0 and 0 <= target < 100
