"""GCD split protocol (Vaze et al. 2022; SX-P §5.1, FX-P §5.1; spec §2, T-01).

A fraction of the classes is "known". A fraction of the training images of the
known classes is labelled. Every other training image (the rest of the known
classes and all of the novel classes) is unlabelled.

With ``stratified=False`` (the default) the labelled images are drawn globally
over all known-class images, using the same RNG call as SelEx/GCD
``subsample_instances``: ``np.random.seed(0)`` followed by
``np.random.choice(range(n), replace=False, size=int(prop * n))``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_SSB_FILE = Path(__file__).resolve().parent / "splits" / "cub_ssb_split.json"


@dataclass(frozen=True)
class GCDSplit:
    known_classes: tuple[int, ...]  # original class ids, in target-index order
    novel_classes: tuple[int, ...]
    labelled_indices: np.ndarray  # sorted indices into the source
    unlabelled_indices: np.ndarray  # sorted indices into the source

    @property
    def num_known(self) -> int:
        return len(self.known_classes)

    @property
    def num_novel(self) -> int:
        return len(self.novel_classes)

    @property
    def num_classes(self) -> int:
        return self.num_known + self.num_novel

    @property
    def class_to_target(self) -> dict[int, int]:
        """Known classes map to 0..K_s-1 and novel classes to K_s..K-1 (SelEx target transform)."""
        ordered = self.known_classes + self.novel_classes
        return {cls: i for i, cls in enumerate(ordered)}

    def map_targets(self, original: np.ndarray) -> np.ndarray:
        mapping = self.class_to_target
        return np.array([mapping[int(c)] for c in original], dtype=np.int64)

    def summary(self, targets: np.ndarray) -> dict[str, int]:
        targets = np.asarray(targets)
        return {
            "known_classes": self.num_known,
            "novel_classes": self.num_novel,
            "labelled_images": int(self.labelled_indices.size),
            "unlabelled_images": int(self.unlabelled_indices.size),
            "unlabelled_known_images": int(np.isin(targets[self.unlabelled_indices], self.known_classes).sum()),
            "unlabelled_novel_images": int(np.isin(targets[self.unlabelled_indices], self.novel_classes).sum()),
        }


def cub_ssb_classes() -> tuple[list[int], list[int]]:
    """SSB known and novel CUB classes, 0-based; novel are ordered Hard + Medium + Easy as in SelEx."""
    data = json.loads(_SSB_FILE.read_text(encoding="utf-8"))
    unknown = data["unknown_classes"]
    return list(data["known_classes"]), unknown["Hard"] + unknown["Medium"] + unknown["Easy"]


def make_gcd_split(
    targets: Sequence[int],
    *,
    known_classes: Optional[Sequence[int]] = None,
    novel_classes: Optional[Sequence[int]] = None,
    known_class_fraction: float = 0.5,
    labelled_fraction: float = 0.5,
    seed: int = 0,
    stratified: bool = False,
) -> GCDSplit:
    """Split a training set (given by its per-image class ids) into labelled and unlabelled parts.

    If ``known_classes`` is not given, the first ``int(fraction · #classes)``
    classes in sorted order are known, as with GCD's ``range(100)`` for CUB.
    """
    targets = np.asarray(targets, dtype=np.int64)
    classes = np.unique(targets)
    if not 0.0 < known_class_fraction <= 1.0 or not 0.0 < labelled_fraction <= 1.0:
        raise ValueError("known_class_fraction and labelled_fraction must be in (0, 1]")
    if known_classes is None:
        known_classes = classes[: int(known_class_fraction * len(classes))].tolist()
    known_classes = [int(c) for c in known_classes]
    if len(set(known_classes)) != len(known_classes) or not set(known_classes) <= set(classes.tolist()):
        raise ValueError("known_classes must be distinct class ids present in targets")
    if novel_classes is None:
        novel_classes = [int(c) for c in classes if int(c) not in set(known_classes)]
    novel_classes = [int(c) for c in novel_classes]
    if set(known_classes) | set(novel_classes) != set(classes.tolist()) or set(known_classes) & set(novel_classes):
        raise ValueError("known and novel classes must partition the classes present in targets")

    known_idx = np.nonzero(np.isin(targets, known_classes))[0]
    rng = np.random.RandomState(seed)
    if stratified:
        chosen = []
        for cls in known_classes:
            cls_idx = known_idx[targets[known_idx] == cls]
            chosen.append(rng.choice(cls_idx, size=int(labelled_fraction * cls_idx.size), replace=False))
        labelled = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    else:
        picks = rng.choice(range(known_idx.size), replace=False, size=int(labelled_fraction * known_idx.size))
        labelled = known_idx[picks]
    labelled = np.sort(labelled.astype(np.int64))
    unlabelled = np.setdiff1d(np.arange(targets.size), labelled)
    return GCDSplit(tuple(known_classes), tuple(novel_classes), labelled, unlabelled)
