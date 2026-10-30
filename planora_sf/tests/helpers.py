"""Synthetic builders shared by the tests (random tensors only: no images, no checkpoints)."""
from __future__ import annotations

import torch

from planora_sf.config import BackboneConfig, FourierConfig, HeadConfig, LatentConfig, PlanoraSFConfig


def make_tiny_config(**overrides) -> PlanoraSFConfig:
    """Small dimensions so the full 4-view forward/backward runs on CPU in well under a second."""
    cfg = PlanoraSFConfig(
        backbone=BackboneConfig(name="tiny", grad_from_block=2, image_size=32),
        head=HeadConfig(hidden_dim=64, bottleneck_dim=16, num_layers=3),
        latent=LatentConfig(dim=128),
        fourier=FourierConfig(cutoff=6),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_fake_cub(root, num_classes: int = 6, train_per_class: int = 3, test_per_class: int = 2, size: int = 48):
    """Write a miniature CUB_200_2011 tree (official metadata layout, tiny JPEGs) under ``root``.

    Image ids interleave train and test images, so split filtering is exercised.
    Returns the list of (img_id, class_1based, is_train) rows.
    """
    import numpy as np
    from pathlib import Path
    from PIL import Image

    base = Path(root) / "CUB_200_2011"
    pattern, t, s = [], train_per_class, test_per_class
    while t or s:  # alternate train/test within each class: 1, 0, 1, 0, ...
        if t:
            pattern.append(1)
            t -= 1
        if s:
            pattern.append(0)
            s -= 1
    rows, images, labels, flags = [], [], [], []
    img_id = 1
    for cls in range(1, num_classes + 1):
        folder = f"{cls:03d}.Class_{cls}"
        (base / "images" / folder).mkdir(parents=True, exist_ok=True)
        for is_train in pattern:
            rel = f"{folder}/img_{img_id:05d}.jpg"
            pixels = np.full((size, size, 3), (cls * 37) % 256, dtype=np.uint8)
            Image.fromarray(pixels).save(base / "images" / rel)
            images.append(f"{img_id} {rel}")
            labels.append(f"{img_id} {cls}")
            flags.append(f"{img_id} {is_train}")
            rows.append((img_id, cls, is_train))
            img_id += 1
    (base / "images.txt").write_text("\n".join(images) + "\n")
    (base / "image_class_labels.txt").write_text("\n".join(labels) + "\n")
    (base / "train_test_split.txt").write_text("\n".join(flags) + "\n")
    # x y width height, as in the official bounding_boxes.txt (float strings); box i starts at (i % 5, 2)
    boxes = [f"{i} {i % 5}.0 2.0 {size // 2}.0 {size // 3}.0" for i, _, _ in rows]
    (base / "bounding_boxes.txt").write_text("\n".join(boxes) + "\n")
    return rows


def _write_annotations(path, rows):
    """devkit-style ``annotations`` struct (bbox_x1..bbox_y2, class 1-based, fname) via scipy.io.savemat."""
    import numpy as np
    from scipy.io import savemat

    dt = [(f, "O") for f in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class", "fname")]
    ann = np.zeros((1, len(rows)), dtype=dt)
    for i, (fname, cls) in enumerate(rows):
        box = [np.array([[v]], dtype=np.uint16) for v in (1, 1, 20, 20)]
        ann[0, i] = (*box, np.array([[cls]], dtype=np.uint8), np.array([fname]))
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(path), {"annotations": ann})


def make_fake_scars(root, *, num_classes: int = 4, train_per_class: int = 3, test_per_class: int = 2, size: int = 24,
                    layout: str = "torchvision", nested: bool = False, test_labels: bool = True,
                    greyscale_every: int = 0):
    """Write a miniature Stanford Cars tree under ``root``; returns {"train": [(fname, cls)], "test": [...]}.

    ``layout="torchvision"`` uses ``root/stanford_cars/`` with the labelled test file at its top level;
    ``"gcd"`` puts everything directly in ``root`` with the labelled test file in ``devkit/`` (GCD/SelEx).
    ``nested`` reproduces Kaggle's ``cars_train/cars_train/``. Classes interleave across files, so the
    loader's annotation order (not file-name order) is exercised.
    """
    from pathlib import Path

    import numpy as np
    from PIL import Image

    base = Path(root) / "stanford_cars" if layout == "torchvision" else Path(root)
    rows = {}
    for split, per_class in (("train", train_per_class), ("test", test_per_class)):
        folder = base / f"cars_{split}"
        if nested:
            folder = folder / f"cars_{split}"
        folder.mkdir(parents=True, exist_ok=True)
        items = []
        for k in range(per_class * num_classes):
            cls = num_classes - (k % num_classes)  # 1-based, descending: annotation order != sorted order
            fname = f"{(k * 7919) % 100000:05d}.jpg"
            pixels = np.full((size, size, 3), (cls * 37 + k) % 256, dtype=np.uint8)
            img = Image.fromarray(pixels)
            if greyscale_every and k % greyscale_every == 0:
                img = img.convert("L")
            img.save(folder / fname)
            items.append((fname, cls))
        rows[split] = items
    _write_annotations(base / "devkit" / "cars_train_annos.mat", rows["train"])
    if test_labels:
        target = base / "cars_test_annos_withlabels.mat" if layout == "torchvision" else base / "devkit" / \
            "cars_test_annos_withlabels.mat"
        _write_annotations(target, rows["test"])
    return rows


def make_blobs(per_blob: list[int], dim: int = 8, spread: float = 0.05, scale: float = 10.0):
    """Well-separated Gaussian blobs: blob b is centred at scale·e_b. Returns (features, blob ids)."""
    feats, ids = [], []
    for b, n in enumerate(per_blob):
        centre = torch.zeros(dim)
        centre[b % dim] = scale * (1 + b // dim)
        feats.append(centre + spread * torch.randn(n, dim))
        ids.append(torch.full((n,), b, dtype=torch.long))
    return torch.cat(feats), torch.cat(ids)
