"""Builds the three GCD datasets for a named benchmark or for synthetic data."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from planora_sf.config import DataConfig
from planora_sf.data.bbox import BBoxCropSource, CropSpec
from planora_sf.data.gcd_dataset import GCDTestDataset, GCDTrainDataset
from planora_sf.data.scars import StanfordCars
from planora_sf.data.sources import CUB200, DummyGCDDataset, ImageSource
from planora_sf.data.splits import SSB_FILES, GCDSplit, make_gcd_split, ssb_classes
from planora_sf.data.transforms import MultiViewTransform, build_transforms

log = logging.getLogger(__name__)

DATASET_ALIASES = {"stanford_cars": "scars"}  # CLI spelling -> config key (CUTOFF_TABLE, README tables)
ROOT_ENV = {"cub": "CUB_ROOT", "scars": "SCARS_ROOT"}


def canonical_dataset(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def default_root(name: str) -> Optional[str]:
    """``$CUB_ROOT`` or ``$SCARS_ROOT`` for a dataset without ``--root``."""
    env = ROOT_ENV.get(canonical_dataset(name))
    return os.environ.get(env) if env else None


@dataclass
class GCDDatasets:
    split: GCDSplit
    train: GCDTrainDataset  # two augmented views per item, for training
    train_eval: GCDTrainDataset  # same samples with the test transform, for prototypes/transductive eval
    test: Optional[GCDTestDataset]  # disjoint test split, test transform; None without test labels (Cars)


def build_sources(
    name: str, root: Optional[Union[str, Path]] = None, *, download: bool = False, crop: Optional[CropSpec] = None
) -> tuple[ImageSource, Optional[ImageSource]]:
    """(train, test) image sources; ``crop`` wraps both in ``BBoxCropSource`` (evaluation only, Task #3)."""
    train, test = _raw_sources(canonical_dataset(name), root, download=download)
    if crop is None:
        return train, test
    if getattr(train, "bboxes", None) is None:
        raise ValueError(f"dataset {name!r} has no bounding boxes (CUB needs bounding_boxes.txt)")
    cropped = BBoxCropSource(train, crop)
    if cropped.num_degenerate:
        log.warning("%d degenerate bounding boxes: those images are used uncropped", cropped.num_degenerate)
    test_cropped = BBoxCropSource(test, crop) if test is not None and getattr(test, "bboxes", None) is not None else None
    return cropped, test_cropped


def _raw_sources(
    name: str, root: Optional[Union[str, Path]], *, download: bool
) -> tuple[ImageSource, Optional[ImageSource]]:
    if name == "cub":
        if root is None:
            raise ValueError("the CUB dataset needs a root directory")
        return CUB200(root, train=True, download=download), CUB200(root, train=False)
    if name == "scars":
        if root is None:
            raise ValueError("the Stanford Cars dataset needs a root directory (see planora_sf.cli.prepare_scars)")
        if download:
            raise ValueError("Stanford Cars has no official download; import it with planora_sf.cli.prepare_scars")
        test = StanfordCars(root, train=False) if StanfordCars.has_test_labels(root) else None
        if test is None:
            log.warning("no labelled Stanford Cars test annotations under %s: building without the test split", root)
        return StanfordCars(root, train=True), test
    if name == "dummy":
        return DummyGCDDataset(seed=0), DummyGCDDataset(images_per_class=4, seed=1)
    raise ValueError(f"unknown dataset {name!r}; expected 'cub', 'scars' ('stanford_cars') or 'dummy'")


def build_split(name: str, train_source: ImageSource, cfg: DataConfig) -> GCDSplit:
    known = novel = None
    name = canonical_dataset(name)
    if name in SSB_FILES and cfg.use_ssb_splits:
        known, novel = ssb_classes(name)
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
    test_source: Optional[ImageSource],
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
        test=GCDTestDataset(test_source, split, test_tf) if test_source is not None else None,
    )
