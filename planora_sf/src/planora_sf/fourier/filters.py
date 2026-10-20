"""Box low-/high-pass transfer functions on the signed DFT grid (spec §4.1-4.2, FX-P §3)."""
from __future__ import annotations

from typing import Optional, Union

import torch

CutoffLike = Union[int, float, torch.Tensor]


def frequency_grid(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Signed integer DFT frequency index in unshifted FFT order (spec §4.2 step 4, U-06).

    For ``n = 224`` this is ``0, 1, ..., 111, -112, ..., -1``, so ``|k|`` is the
    distance from DC along that axis.
    """
    return torch.round(torch.fft.fftfreq(n, device=device) * n).to(torch.long)


def _cutoff_tensor(cutoff: CutoffLike, device: Optional[torch.device]) -> torch.Tensor:
    f = torch.as_tensor(cutoff, device=device)
    if f.ndim > 1:
        raise ValueError(f"cutoff must be a scalar or a 1-D tensor, got shape {tuple(f.shape)}")
    if (f < 0).any():
        raise ValueError("cut-off frequencies must be non-negative")
    return f.reshape(-1, 1, 1, 1)


def lowpass_mask(
    height: int,
    width: int,
    cutoff: CutoffLike,
    *,
    strict: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """H_LP(u, v) = 1(|u| < f, |v| < f): a centred square of side 2f - 1 (2f + 1 if not strict).

    Returns a float mask of shape ``[B, 1, H, W]`` with ``B = 1`` for a scalar
    cut-off and ``B = len(cutoff)`` for per-image cut-offs.
    """
    if device is None and isinstance(cutoff, torch.Tensor):
        device = cutoff.device
    ku = frequency_grid(height, device).abs().view(1, 1, height, 1)
    kv = frequency_grid(width, device).abs().view(1, 1, 1, width)
    f = _cutoff_tensor(cutoff, device)
    if strict:
        mask = (ku < f) & (kv < f)
    else:
        mask = (ku <= f) & (kv <= f)
    return mask.to(torch.float32)


def highpass_mask(
    height: int,
    width: int,
    cutoff: CutoffLike,
    *,
    strict: bool = True,
    band_upper: Optional[CutoffLike] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """H_HP = 1 - H_LP (FX-P §3).

    With ``band_upper`` the mask is additionally limited to frequencies below it,
    turning it into the band-pass variant of U-07.
    """
    mask = 1.0 - lowpass_mask(height, width, cutoff, strict=strict, device=device)
    if band_upper is not None:
        mask = mask * lowpass_mask(height, width, band_upper, strict=strict, device=mask.device)
    return mask
