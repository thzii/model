"""Per-channel 2-D DFT helpers (FX-P Eq. 4-5)."""
from __future__ import annotations

import torch


def apply_frequency_mask(x: torch.Tensor, mask: torch.Tensor, *, return_complex: bool = False) -> torch.Tensor:
    """Filter ``x [..., H, W]`` by multiplying its 2-D DFT with ``mask`` and inverting.

    ``mask`` broadcasts against the spectrum (e.g. ``[B, 1, H, W]`` for ``[B, C, H, W]``
    input, filtering every channel independently). Point-symmetric masks give a
    real result up to round-off, so the real part is returned by default.
    """
    filtered = torch.fft.ifft2(torch.fft.fft2(x) * mask)
    return filtered if return_complex else filtered.real


def power_spectrum(x: torch.Tensor) -> torch.Tensor:
    """|F(u, v)|^2 over the last two dimensions."""
    return torch.fft.fft2(x).abs().pow(2)
