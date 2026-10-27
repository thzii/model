"""L_total = L_base + α_low L_low + α_high L_high + α_cls L_cls (FX-P Eq. 9)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from planora_sf.config import LossWeights


@dataclass
class LossBreakdown:
    total: torch.Tensor
    base: torch.Tensor
    low: torch.Tensor
    high: torch.Tensor
    cls: torch.Tensor
    sdl: Optional[torch.Tensor] = None  # DebGCD Eq. 6, only when debgcd.enabled
    adl: Optional[torch.Tensor] = None  # DebGCD Eq. 13, only when debgcd.enabled
    stats: dict[str, float] = field(default_factory=dict)  # non-loss per-step monitors

    def as_floats(self) -> dict[str, float]:
        names = ["total", "base", "low", "high", "cls"] + [n for n in ("sdl", "adl") if getattr(self, n) is not None]
        return {**{name: float(getattr(self, name).detach()) for name in names}, **self.stats}


def combine_losses(
    base: torch.Tensor, low: torch.Tensor, high: torch.Tensor, cls: torch.Tensor, weights: LossWeights
) -> LossBreakdown:
    total = base + weights.low * low + weights.high * high + weights.cls * cls
    return LossBreakdown(total=total, base=base, low=low, high=high, cls=cls)


def add_debgcd(losses: LossBreakdown, sdl: torch.Tensor, adl: torch.Tensor, lam_sdl: float, lam_adl: float,
               stats: Optional[dict[str, float]] = None) -> LossBreakdown:
    """L_all = L_total + λ_sdl L_sdl + λ_adl L_adl (DebGCD Eq. 14, with L_gcd replaced by PLANORA-SF's L_total)."""
    return LossBreakdown(
        total=losses.total + lam_sdl * sdl + lam_adl * adl, base=losses.base, low=losses.low, high=losses.high,
        cls=losses.cls, sdl=sdl, adl=adl, stats=dict(stats or {}),
    )
