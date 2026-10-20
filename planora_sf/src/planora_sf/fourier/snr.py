"""Cut-off calibration via SNR (spec §4.3, FX-P §4).

SNR(f_c) = 10 log10( sum_{f <= f_c} |X(f)|^2 / sum_{f > f_c} |X(f)|^2 ).

The frequency metric, channel handling, DC handling and pre-processing are not
specified by FourEx (U-09); the defaults here are scaffold choices. The
acceptance test is to reproduce FX-A Table 7 on CUB (T = 23 at 15 dB), which
needs real images (Stage C).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

from planora_sf.fourier.filters import frequency_grid
from planora_sf.fourier.transforms import power_spectrum

SNR_CAP_DB = 300.0


def frequency_radius(height: int, width: int, metric: str = "box", device=None) -> torch.Tensor:
    """Scalar frequency of every DFT bin: max(|k_u|, |k_v|) ("box") or rounded L2 ("radial")."""
    ku = frequency_grid(height, device).abs()[:, None]
    kv = frequency_grid(width, device).abs()[None, :]
    if metric == "box":
        return torch.maximum(ku, kv)
    if metric == "radial":
        return torch.round(torch.sqrt((ku**2 + kv**2).to(torch.float64))).to(torch.long)
    raise ValueError(f"unknown frequency metric {metric!r}")


def mean_power_spectrum(images: torch.Tensor) -> torch.Tensor:
    """Mean over images of the channel-summed power spectrum, ``[N, C, H, W] -> [H, W]`` (U-09)."""
    if images.ndim != 4:
        raise ValueError(f"expected [N, C, H, W], got shape {tuple(images.shape)}")
    return power_spectrum(images.to(torch.float64)).sum(dim=1).mean(dim=0)


def snr_curve(power: torch.Tensor, metric: str = "box") -> torch.Tensor:
    """SNR in dB for every cut-off f_c = 0 .. R - 1, where R is the largest frequency radius.

    Entry ``f_c`` is the energy at radius <= f_c over the energy above it. The curve
    is non-decreasing in f_c. A cut-off with no energy above it (or below it) is
    reported as +SNR_CAP_DB (or -SNR_CAP_DB), keeping the curve finite.
    """
    radius = frequency_radius(power.shape[-2], power.shape[-1], metric, power.device)
    per_ring = torch.bincount(radius.flatten(), weights=power.flatten().to(torch.float64))
    cumulative = per_ring.cumsum(0)
    signal = cumulative[:-1]
    noise = cumulative[-1] - signal
    ratio_db = 10.0 * torch.log10(signal.clamp_min(1e-300) / noise.clamp_min(1e-300))
    capped = torch.where(noise > 0, ratio_db, torch.full_like(ratio_db, SNR_CAP_DB))
    return capped.clamp(-SNR_CAP_DB, SNR_CAP_DB)


def select_cutoff(curve_db: torch.Tensor, target_db: float = 15.0) -> int:
    """Smallest f_c whose SNR reaches ``target_db``; ``len(curve)`` if none does (saturation)."""
    hits = torch.nonzero(curve_db >= target_db).flatten()
    return int(hits[0]) if hits.numel() else int(curve_db.numel())


@dataclass
class CalibrationResult:
    cutoff: int
    per_class: dict[int, int] = field(default_factory=dict)


class ClassPowerAccumulator:
    """Running per-class sums of the channel-summed power spectrum, so calibration can stream batches."""

    def __init__(self) -> None:
        self.sums: dict[int, torch.Tensor] = {}
        self.counts: dict[int, int] = {}

    def add(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        if images.ndim != 4 or images.shape[0] != labels.shape[0]:
            raise ValueError("expected images [N, C, H, W] and labels [N]")
        power = power_spectrum(images.to(torch.float64)).sum(dim=1)
        for cls in torch.unique(labels).tolist():
            selected = power[labels == cls]
            key = int(cls)
            self.sums[key] = self.sums.get(key, 0) + selected.sum(dim=0).cpu()
            self.counts[key] = self.counts.get(key, 0) + int(selected.shape[0])

    def mean(self, cls: int) -> torch.Tensor:
        return self.sums[cls] / self.counts[cls]

    @property
    def classes(self) -> list[int]:
        return sorted(self.sums)


def calibrate_from_accumulator(
    acc: ClassPowerAccumulator, *, target_db: float = 15.0, metric: str = "box"
) -> CalibrationResult:
    """Per-class cut-offs averaged over classes (FX-P §4)."""
    if not acc.classes:
        raise ValueError("no images were accumulated")
    per_class = {cls: select_cutoff(snr_curve(acc.mean(cls), metric), target_db) for cls in acc.classes}
    mean = sum(per_class.values()) / len(per_class)
    return CalibrationResult(cutoff=int(math.floor(mean + 0.5)), per_class=per_class)


def calibrate_cutoff(
    images: torch.Tensor, labels: torch.Tensor, *, target_db: float = 15.0, metric: str = "box"
) -> CalibrationResult:
    """Per-class cut-offs averaged over classes (FX-P §4), for images held in memory.

    Pass only labelled, known-class images: novel-class labels are not available
    in GCD and using them would leak (U-10).
    """
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images and labels must have the same length")
    acc = ClassPowerAccumulator()
    acc.add(images, labels)
    return calibrate_from_accumulator(acc, target_db=target_db, metric=metric)


def calibrate_from_batches(
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    targets_db: Sequence[float] = (5.0, 10.0, 15.0, 20.0, 25.0),
    metric: str = "box",
) -> dict[float, CalibrationResult]:
    """Stream ``(images, labels)`` batches once and calibrate every SNR level (the rows of FX-A Table 7)."""
    acc = ClassPowerAccumulator()
    for images, labels in batches:
        acc.add(images, labels)
    return {float(db): calibrate_from_accumulator(acc, target_db=db, metric=metric) for db in targets_db}
