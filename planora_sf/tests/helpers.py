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
