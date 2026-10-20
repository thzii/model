"""Latent subspace allocation (spec §3.3, FX-P §4 and Figs. 4-5, SX-P Eq. 13, FX-A Eq. 12).

With ``fraction_of_half=True`` (U-13) the low branch uses a prefix of the left
half ``[0, D/2)`` and the high branch a suffix of the right half ``[D/2, D)``.
"""
from __future__ import annotations

from typing import Union

import torch

WidthLike = Union[int, torch.Tensor]


def _round_half_up(x: torch.Tensor) -> torch.Tensor:
    return torch.floor(x + 0.5).to(torch.long)


def subspace_size(dim: int, *, fraction_of_half: bool = True) -> int:
    return dim // 2 if fraction_of_half else dim


def low_width(f_low: WidthLike, t_low: int, dim: int, *, fraction_of_half: bool = True) -> torch.Tensor:
    """d_l = round(f_l / T_l · D/2), at least 1 (FX-P §4, Fig. 4; rounding is U-12)."""
    base = subspace_size(dim, fraction_of_half=fraction_of_half)
    f = torch.as_tensor(f_low, dtype=torch.float64)
    if (f < 0).any() or (f > t_low).any():
        raise ValueError(f"f_l must lie in [0, T_l={t_low}]")
    return _round_half_up(f / t_low * base).clamp(1, base)


def high_width(
    f_high: WidthLike, t_high: int, dim: int, *, mode: str = "one_minus", fraction_of_half: bool = True
) -> torch.Tensor:
    """d_h = round((1 - f_h/T_h) · D/2) for ``mode="one_minus"`` (U-11 default), or
    round(f_h/T_h · D/2) for ``mode="ratio"``; at least 1."""
    base = subspace_size(dim, fraction_of_half=fraction_of_half)
    f = torch.as_tensor(f_high, dtype=torch.float64)
    if (f < 0).any() or (f > t_high).any():
        raise ValueError(f"f_h must lie in [0, T_h={t_high}]")
    if mode == "one_minus":
        fraction = 1.0 - f / t_high
    elif mode == "ratio":
        fraction = f / t_high
    else:
        raise ValueError(f"unknown high-fraction mode {mode!r}")
    return _round_half_up(fraction * base).clamp(1, base)


def prefix_mask(widths: torch.Tensor, dim: int) -> torch.Tensor:
    """Boolean ``[N, D]`` mask selecting dims ``[0, width_n)`` for each row."""
    widths = torch.as_tensor(widths).reshape(-1)
    return torch.arange(dim, device=widths.device)[None, :] < widths[:, None]


def suffix_mask(widths: torch.Tensor, dim: int) -> torch.Tensor:
    """Boolean ``[N, D]`` mask selecting dims ``[D - width_n, D)`` for each row."""
    widths = torch.as_tensor(widths).reshape(-1)
    return torch.arange(dim, device=widths.device)[None, :] >= (dim - widths)[:, None]


def selex_prefix_width(dim: int, level_k: int) -> int:
    """SelEx prefix length D / 2^k for hierarchy level k (k = 0 is the full D; SX-P Eq. 13)."""
    if level_k < 0:
        raise ValueError("level_k must be >= 0")
    return max(dim >> level_k, 1)


def lsse_widths(d_low: torch.Tensor, dim: int, level_k: int) -> torch.Tensor:
    """min(D / 2^k, d_l): the FourEx overlap rule for the k-th L_LSSE term (FX-A Eq. 12)."""
    return torch.as_tensor(d_low).clamp(max=selex_prefix_width(dim, level_k))
