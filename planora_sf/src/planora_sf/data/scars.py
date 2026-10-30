"""Stanford Cars (Krause et al., 3dRR 2013): loader, archive import and verification (plan Phase 1).

The official site (ai.stanford.edu/~jkrause/cars) no longer serves the files, and torchvision's
``StanfordCars(download=True)`` now raises. So nothing here downloads. ``import_archives``
builds the layout from archives fetched by hand (original-release files or a Kaggle copy).

Layout (torchvision's; the GCD/SelEx layout without ``stanford_cars/`` is also read)::

    <root>/stanford_cars/
        cars_train/*.jpg                    8,144 images
        cars_test/*.jpg                     8,041 images (optional here)
        devkit/cars_train_annos.mat         fields bbox_x1..bbox_y2, class (1-based), fname
        devkit/cars_meta.mat                class_names (optional)
        cars_test_annos_withlabels.mat      labelled test annotations (optional here)

The test split is optional because the reported metric is transductive on the train split;
the disjoint test split is built but never evaluated (plan audit A4).
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

SCARS_DIR = "stanford_cars"
SCARS_NUM_TRAIN, SCARS_NUM_TEST, SCARS_NUM_CLASSES = 8144, 8041, 196
TEST_LABELS = "cars_test_annos_withlabels.mat"
DEVKIT_FILES = ("cars_train_annos.mat", "cars_meta.mat", "cars_test_annos.mat")

# MD5s of the original release, as torchvision pinned them before the site went down
# (torchvision 0.15 datasets/stanford_cars.py). Used to label files, never required.
ORIGINAL_MD5 = {
    "car_devkit.tgz": "c3b158d763b6e2245038c8ad08e45376",
    "cars_train.tgz": "065e5b463ae28d29e77c1b4b166cfe61",
    "cars_test.tgz": "4ce7ebf6a94d07f1952d94dd34c4d501",
    TEST_LABELS: "b0a2b23655a3edd16d84508592a98d10",
}


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_base(root: Union[str, Path]) -> Path:
    """``<root>/stanford_cars`` if present, else ``<root>`` itself (GCD/SelEx ``car_root`` layout)."""
    root = Path(root)
    return root / SCARS_DIR if (root / SCARS_DIR).is_dir() else root


def _image_dir(base: Path, name: str) -> Path:
    """``base/name``, descending into Kaggle's nested ``name/name`` if that is where the images are."""
    folder = base / name
    nested = folder / name
    return nested if nested.is_dir() and not any(folder.glob("*.jpg")) else folder


def _read_annotations(path: Path) -> tuple[list[str], np.ndarray]:
    from scipy.io import loadmat

    annotations = np.atleast_1d(loadmat(str(path), squeeze_me=True)["annotations"])
    names = annotations.dtype.names or ()
    if "fname" not in names or "class" not in names:
        raise ValueError(f"{path} has no 'fname'/'class' annotation fields (found {names})")
    fnames = [str(a["fname"]) for a in annotations]
    classes = np.array([int(a["class"]) - 1 for a in annotations], dtype=np.int64)  # 1-based on disk
    return fnames, classes


class StanfordCars:
    """Stanford Cars train or test split in annotation order (as GCD/SelEx read it), 0-based class ids."""

    def __init__(self, root: Union[str, Path], *, train: bool = True, check_files: bool = False):
        self.base = resolve_base(root)
        if train:
            annotations = self.base / "devkit" / "cars_train_annos.mat"
        else:
            candidates = [self.base / TEST_LABELS, self.base / "devkit" / TEST_LABELS]
            annotations = next((p for p in candidates if p.is_file()), candidates[0])
        if not annotations.is_file():
            raise FileNotFoundError(f"Stanford Cars annotations not found: {annotations}; see planora_sf.cli.prepare_scars")
        self.image_dir = _image_dir(self.base, "cars_train" if train else "cars_test")
        self.paths, self.targets = _read_annotations(annotations)
        if check_files:
            missing = [p for p in self.paths if not (self.image_dir / p).is_file()]
            if missing:
                raise FileNotFoundError(f"{len(missing)} Stanford Cars images missing, e.g. {missing[0]}")

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, index: int) -> Image.Image:
        with Image.open(self.image_dir / self.paths[index]) as img:
            return img.convert("RGB")  # some images are greyscale

    @staticmethod
    def has_test_labels(root: Union[str, Path]) -> bool:
        base = resolve_base(root)
        return (base / TEST_LABELS).is_file() or (base / "devkit" / TEST_LABELS).is_file()


# ---------------------------------------------------------------- archive import

def _check_member(target: Path, name: str) -> None:
    dest = (target / name).resolve()
    if dest != target and target not in dest.parents:
        raise RuntimeError(f"refusing to extract {name!r} outside {target}")


def safe_extract(archive: Path, target: Path) -> None:
    """Extract a tar (any compression) or zip archive, refusing links and paths outside ``target``."""
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                _check_member(target, info.filename)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError(f"refusing to extract link {info.filename!r}")
            zf.extractall(target)
        return
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            _check_member(target, member.name)
            if member.issym() or member.islnk():
                raise RuntimeError(f"refusing to extract link {member.name!r}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"refusing to extract special file {member.name!r}")
        if hasattr(tarfile, "data_filter"):  # Python >= 3.12 (and recent 3.10/3.11 patch releases)
            tar.extractall(target, filter="data")
        else:
            tar.extractall(target)


def _find_image_dir(staging: Path, name: str) -> Optional[Path]:
    """The directory named ``name`` that directly holds .jpg files (Kaggle copies nest it)."""
    hits = [p for p in staging.rglob(name) if p.is_dir() and any(p.glob("*.jpg"))]
    if len(hits) > 1:
        raise RuntimeError(f"several '{name}' image folders in the archives: {hits}")
    return hits[0] if hits else None


def _find_file(staging: Path, name: str) -> Optional[Path]:
    hits = [p for p in staging.rglob(name) if p.is_file()]
    if len(hits) > 1 and len({_digest(p, "sha256") for p in hits}) > 1:
        raise RuntimeError(f"several different '{name}' files in the archives: {hits}")
    return hits[0] if hits else None


def import_archives(root: Union[str, Path], archives: list[Union[str, Path]]) -> dict:
    """Build ``<root>/stanford_cars`` from archives and loose ``.mat`` files; returns a provenance record.

    Accepts the original-release files (``car_devkit.tgz``, ``cars_train.tgz``, ``cars_test.tgz``,
    ``cars_test_annos_withlabels.mat``) and zip/tar copies with other nesting (e.g. Kaggle).
    Refuses to overwrite an existing ``stanford_cars`` folder.
    """
    root = Path(root)
    base = root / SCARS_DIR
    if base.exists():
        raise FileExistsError(f"{base} already exists; remove it or verify it without --from-archive")
    staging = root / ".prepare_scars_staging"
    if staging.exists():
        raise FileExistsError(f"{staging} exists from an earlier attempt; remove it first")
    record: dict = {"archives": []}
    try:
        for i, archive in enumerate(Path(a) for a in archives):
            if not archive.is_file():
                raise FileNotFoundError(archive)
            md5 = _digest(archive, "md5")
            expected = ORIGINAL_MD5.get(archive.name)
            record["archives"].append({
                "name": archive.name, "md5": md5, "sha256": _digest(archive, "sha256"),
                "matches_original_release": None if expected is None else md5 == expected,
            })
            dest = staging / f"{i:02d}"
            if archive.suffix == ".mat":
                dest.mkdir(parents=True)
                shutil.copy2(archive, dest / archive.name)
            else:
                log.info("extracting %s", archive)
                safe_extract(archive, dest)

        train_images = _find_image_dir(staging, "cars_train")
        train_annos = _find_file(staging, "cars_train_annos.mat")
        if train_images is None or train_annos is None:
            raise FileNotFoundError("the archives contain no cars_train/*.jpg folder or no cars_train_annos.mat")
        base.mkdir(parents=True)
        (base / "devkit").mkdir()
        shutil.move(str(train_images), base / "cars_train")
        test_images = _find_image_dir(staging, "cars_test")
        if test_images is not None:
            shutil.move(str(test_images), base / "cars_test")
        for name in DEVKIT_FILES:
            found = _find_file(staging, name)
            if found is not None:
                shutil.move(str(found), base / "devkit" / name)
        labels = _find_file(staging, TEST_LABELS)
        if labels is not None:
            shutil.move(str(labels), base / TEST_LABELS)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return record


def write_manifest(root: Union[str, Path], record: Optional[dict] = None) -> Path:
    """SHA-256 of every annotation file (no official checksums exist), plus the import record."""
    base = resolve_base(root)
    mats = sorted(p for p in base.rglob("*.mat"))
    manifest = {
        "annotation_sha256": {str(p.relative_to(base)).replace("\\", "/"): _digest(p, "sha256") for p in mats},
        **(record or {}),
    }
    path = base / "planora_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
