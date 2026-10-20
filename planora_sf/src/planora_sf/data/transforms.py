"""GCD "imagenet" transforms (SelEx data/augmentations/__init__.py:7-31; spec §3.2 C1).

The Gaussian blur that FourEx applies before Fourier filtering (FX-P §5.1) is
not part of these image transforms. ``FourierViewGenerator`` applies it to the
batch on the device, to the input of the FFT only (U-03).
"""
from __future__ import annotations

from typing import Callable

from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    image_size: int = 224, *, crop_pct: float = 0.875, normalize: bool = True
) -> tuple[transforms.Compose, transforms.Compose]:
    """(train, test) transforms. Train: Resize(size/crop_pct) -> RandomCrop -> HFlip -> ToTensor -> Normalize.

    SelEx also lists ``ColorJitter()`` with no arguments; it is the identity and is omitted.
    """
    resize = int(image_size / crop_pct)
    tail = [transforms.ToTensor()] + ([transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)] if normalize else [])
    train = transforms.Compose(
        [
            transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            *tail,
        ]
    )
    test = transforms.Compose(
        [
            transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            *tail,
        ]
    )
    return train, test


class MultiViewTransform:
    """Apply ``transform`` independently ``n_views`` times (SelEx ContrastiveLearningViewGenerator)."""

    def __init__(self, transform: Callable, n_views: int = 2):
        self.transform = transform
        self.n_views = n_views

    def __call__(self, image):
        return [self.transform(image) for _ in range(self.n_views)]
