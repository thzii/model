"""L_total = L_base + α_low L_low + α_high L_high + α_cls L_cls (FX-P Eq. 9)."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from planora_sf.config import LossWeights


@dataclass
class LossBreakdown:
    total: torch.Tensor
    base: torch.Tensor
    low: torch.Tensor
    high: torch.Tensor
    cls: torch.Tensor

    def as_floats(self) -> dict[str, float]:
        return {name: float(getattr(self, name).detach()) for name in ("total", "base", "low", "high", "cls")}


def combine_losses(
    base: torch.Tensor, low: torch.Tensor, high: torch.Tensor, cls: torch.Tensor, weights: LossWeights
) -> LossBreakdown:
    total = base + weights.low * low + weights.high * high + weights.cls * cls
    return LossBreakdown(total=total, base=base, low=low, high=high, cls=cls)
