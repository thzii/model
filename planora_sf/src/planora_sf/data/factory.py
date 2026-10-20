"""Builds the three GCD datasets for a named benchmark or for synthetic data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from planora_sf.config import DataConfig
from planora_sf.data.gcd_dataset import GCDTestDataset, GCDTrainDataset
from planora_sf.data.sources import CUB200, DummyGCDDataset, ImageSource
from planora_sf.data.splits import GCDSplit, cub_ssb_classes, make_gcd_split
from planora_sf.data.transforms import MultiViewTransform, build_transforms


@dataclass
class GCDDatasets:
    split: GCDSplit
    train: GCDTrainDataset  # two augmented views per item, for training
    train_eval: GCDTrainDataset  # same samples with the test transform, for prototypes/transductive eval
    test: GCDTestDataset  # disjoint test split, test transform


def build_sources(
    name: str, root: Optional[Union[str, Path]] = None, *, download: bool = False
) -> tuple[ImageSource, ImageSource]:
    if name == "cub":
        if root is None:
            raise ValueError("the CUB dataset needs a root directory")
        return CUB200(root, train=True, download=download), CUB200(root, train=False)
    if name == "dummy":
        return DummyGCDDataset(seed=0), DummyGCDDataset(images_per_class=4, seed=1)
    raise ValueError(f"unknown dataset {name!r}; expected 'cub' or 'dummy'")


def build_split(name: str, train_source: ImageSource, cfg: DataConfig) -> GCDSplit:
    known = novel = None
    if name == "cub" and cfg.use_ssb_splits:
        known, novel = cub_ssb_classes()
    return make_gcd_split(
        train_source.targets,
        known_classes=known,
        novel_classes=novel,
        known_class_fraction=cfg.known_class_fraction,
        labelled_fraction=cfg.labelled_fraction,
        seed=cfg.split_seed,
        stratified=cfg.stratified,
    )


def build_gcd_datasets(
    train_source: ImageSource,
    test_source: ImageSource,
    split: GCDSplit,
    *,
    image_size: int,
    crop_pct: float = 0.875,
) -> GCDDatasets:
    train_tf, test_tf = build_transforms(image_size, crop_pct=crop_pct)
    train = GCDTrainDataset(train_source, split, MultiViewTransform(train_tf, n_views=2))
    return GCDDatasets(
        split=split,
        train=train,
        train_eval=train.with_transform(test_tf),
        test=GCDTestDataset(test_source, split, test_tf),
    )
