"""Bounding-box cropping at the image-source level (master plan §2.1, Task #3).

Cropping happens before any transform, so the split, datasets, evaluator and metrics are
unchanged and a cropped evaluation pairs with an uncropped one seed by seed. Boxes are stored
as 0-based pixel ``(left, top, right, bottom)`` with exclusive right/bottom, PIL's crop convention.
The ±1 px left by the datasets' index conventions (Stanford Cars devkit: MATLAB 1-based) does not matter here.

``square`` (default) grows the box to a square before cropping: cars are about 2:1, and the
test transform (Resize then CenterCrop) would cut both ends off a tight crop. ``tight`` is the control.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from PIL import Image

from planora_sf.data.sources import ImageSource


@dataclass(frozen=True)
class CropSpec:
    margin: float = 0.1  # added on each side, as a fraction of the box width/height
    shape: str = "square"  # "square" | "tight"

    def __post_init__(self) -> None:
        if not 0.0 <= self.margin <= 2.0:
            raise ValueError(f"crop margin must be in [0, 2], got {self.margin}")
        if self.shape not in ("square", "tight"):
            raise ValueError(f"crop shape must be 'square' or 'tight', got {self.shape!r}")

    def to_dict(self) -> dict:
        return asdict(self)


def _fit(lo: float, hi: float, limit: int) -> tuple[float, float]:
    """Shift [lo, hi) inside [0, limit) without resizing it when it fits; clip otherwise."""
    if hi - lo >= limit:
        return 0.0, float(limit)
    if lo < 0:
        lo, hi = 0.0, hi - lo
    if hi > limit:
        lo, hi = lo - (hi - limit), float(limit)
    return lo, hi


def expand_box(box, spec: CropSpec, image_size: tuple[int, int]) -> Optional[tuple[int, int, int, int]]:
    """The crop box for ``box`` (left, top, right, bottom) in an image of ``(width, height)``.

    Returns ``None`` for a degenerate box (no area inside the image); callers then keep the full image.
    """
    width, height = image_size
    left, top, right, bottom = (float(v) for v in box)
    left, top = max(left, 0.0), max(top, 0.0)
    right, bottom = min(right, float(width)), min(bottom, float(height))
    w, h = right - left, bottom - top
    if w < 1 or h < 1:
        return None
    left, right = left - spec.margin * w, right + spec.margin * w
    top, bottom = top - spec.margin * h, bottom + spec.margin * h
    if spec.shape == "square":
        side = max(right - left, bottom - top)
        cx, cy = (left + right) / 2, (top + bottom) / 2
        left, right, top, bottom = cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2
        left, right = _fit(left, right, width)
        top, bottom = _fit(top, bottom, height)
    else:
        left, top = max(left, 0.0), max(top, 0.0)
        right, bottom = min(right, float(width)), min(bottom, float(height))
    box = (int(round(left)), int(round(top)), int(round(right)), int(round(bottom)))
    return box if box[2] > box[0] and box[3] > box[1] else None


def degenerate_boxes(bboxes: np.ndarray) -> int:
    """Boxes with no area, judged from the coordinates alone (image sizes are not read)."""
    b = np.asarray(bboxes, dtype=np.float64)
    return int(((b[:, 2] - b[:, 0] < 1) | (b[:, 3] - b[:, 1] < 1)).sum())


class BBoxCropSource:
    """An image source whose images are cropped to their bounding box (same order, same targets)."""

    def __init__(self, source: ImageSource, spec: CropSpec):
        bboxes = getattr(source, "bboxes", None)
        if bboxes is None:
            raise ValueError(f"{type(source).__name__} has no bounding boxes to crop with")
        if len(bboxes) != len(source):
            raise ValueError(f"{len(bboxes)} boxes for {len(source)} images")
        self.source = source
        self.spec = spec
        self.bboxes = np.asarray(bboxes, dtype=np.float64)
        self.targets = source.targets
        self.num_degenerate = degenerate_boxes(self.bboxes)

    def __len__(self) -> int:
        return len(self.source)

    def load(self, index: int) -> Image.Image:
        image = self.source.load(index)
        box = expand_box(self.bboxes[index], self.spec, image.size)
        return image if box is None else image.crop(box)
