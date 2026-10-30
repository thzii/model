"""Image sources: an indexable set of images with per-image class ids.

``CUB200`` reads the official CUB-200-2011 layout. ``DummyGCDDataset``
generates random images deterministically, for offline tests.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Protocol, Sequence, Union

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


class ImageSource(Protocol):
    targets: np.ndarray  # original class id of every image

    def __len__(self) -> int: ...

    def load(self, index: int) -> Image.Image: ...


class DummyGCDDataset:
    """Synthetic RGB images: a class-dependent base colour plus per-image noise.

    Images are generated from (seed, index), so every access returns the same image.
    """

    def __init__(
        self,
        num_classes: int = 8,
        images_per_class: Union[int, Sequence[int]] = 12,
        image_size: int = 40,
        seed: int = 0,
    ):
        counts = [images_per_class] * num_classes if isinstance(images_per_class, int) else list(images_per_class)
        if len(counts) != num_classes or min(counts) < 1:
            raise ValueError("images_per_class must give a positive count for every class")
        self.targets = np.repeat(np.arange(num_classes), counts).astype(np.int64)
        self.image_size = image_size
        self.seed = seed
        palette = np.random.RandomState(seed).randint(0, 256, size=(num_classes, 3))
        self._palette = palette.astype(np.float64)

    def __len__(self) -> int:
        return int(self.targets.size)

    def load(self, index: int) -> Image.Image:
        rng = np.random.RandomState((self.seed * 1_000_003 + index) % (2**32))
        noise = rng.normal(0.0, 40.0, size=(self.image_size, self.image_size, 3))
        pixels = np.clip(self._palette[self.targets[index]] + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(pixels, mode="RGB")


# CUB-200-2011. The old vision.caltech.edu URL returns 404; this record resolved
# when checked on 2026-10-20. MD5 is the one SelEx/GCD use.
CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
CUB_ARCHIVE = "CUB_200_2011.tgz"
CUB_MD5 = "97eceeb196236b17998738112f37df78"
CUB_NUM_TRAIN, CUB_NUM_TEST, CUB_NUM_CLASSES = 5994, 5794, 200


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, target: Path) -> None:
    target = target.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            dest = (target / member.name).resolve()
            if target not in dest.parents and dest != target:
                raise RuntimeError(f"refusing to extract {member.name!r} outside {target}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"refusing to extract link {member.name!r}")
        tar.extractall(target)


def download_cub(root: Union[str, Path], *, url: str = CUB_URL, md5: str = CUB_MD5) -> Path:
    """Download and extract CUB-200-2011 into ``root`` unless its metadata is already present."""
    root = Path(root)
    base = root / "CUB_200_2011"
    if (base / "images.txt").is_file():
        return base
    root.mkdir(parents=True, exist_ok=True)
    archive = root / CUB_ARCHIVE
    if not archive.is_file():
        log.info("downloading %s -> %s", url, archive)
        partial = archive.with_suffix(".part")
        with urllib.request.urlopen(url) as response, open(partial, "wb") as handle:
            shutil.copyfileobj(response, handle)
        partial.rename(archive)
    actual = _md5(archive)
    if actual != md5:
        raise RuntimeError(f"MD5 mismatch for {archive}: expected {md5}, got {actual}")
    _safe_extract(archive, root)
    return base


def _read_table(path: Path) -> dict[int, str]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            key, value = line.split(maxsplit=1)
            rows[int(key)] = value.strip()
    return rows


class CUB200:
    """CUB-200-2011 train or test split, in ``images.txt`` (img_id) order, with 0-based class ids.

    ``bboxes`` holds ``bounding_boxes.txt`` (x, y, width, height) as (left, top, right, bottom),
    or ``None`` when the file is absent.
    """

    def __init__(self, root: Union[str, Path], *, train: bool = True, download: bool = False, check_files: bool = False):
        root = Path(root)
        if download:
            download_cub(root)
        self.base = root / "CUB_200_2011"
        if not (self.base / "images.txt").is_file():
            raise FileNotFoundError(f"CUB metadata not found under {self.base}; use download=True")
        images = _read_table(self.base / "images.txt")
        labels = _read_table(self.base / "image_class_labels.txt")
        is_train = _read_table(self.base / "train_test_split.txt")
        wanted = "1" if train else "0"
        ids = [i for i in sorted(images) if is_train[i] == wanted]
        self.image_ids = np.array(ids, dtype=np.int64)
        self.paths = [images[i] for i in ids]
        self.targets = np.array([int(labels[i]) - 1 for i in ids], dtype=np.int64)
        self.bboxes = None
        if (self.base / "bounding_boxes.txt").is_file():
            boxes = _read_table(self.base / "bounding_boxes.txt")
            xywh = np.array([[float(v) for v in boxes[i].split()] for i in ids], dtype=np.float64).reshape(-1, 4)
            self.bboxes = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], axis=1)
        if check_files:
            missing = [p for p in self.paths if not (self.base / "images" / p).is_file()]
            if missing:
                raise FileNotFoundError(f"{len(missing)} CUB images missing, e.g. {missing[0]}")

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, index: int) -> Image.Image:
        with Image.open(self.base / "images" / self.paths[index]) as img:
            return img.convert("RGB")
