"""Multi-loss gradient monitor (master plan §2.3, Task #5).

On a sampled step, each loss group of ``LossBreakdown.parts`` is differentiated separately with
``torch.autograd.grad`` with respect to the trainable backbone blocks, the only parameters every
group shares (the DebGCD heads read the CLS, not z). The parts are:

    base      SelEx loss on z (the α_d-mixed hybrid under HypCD)
    fourier   w_low·L_low + w_high·L_high + w_cls·L_cls   (only when a FourEx weight is > 0)
    debias    λ_sdl·L_sdl + λ_adl·L_adl                    (only when DebGCD is on)
    hyp_dist  base's Poincaré-distance branch, before α_d   (HypCD only)
    hyp_ang   base's angle branch, before 1 - α_d           (HypCD only)

``base + fourier + debias`` is the total, so their gradients add up to the update direction;
``α_d·hyp_dist + (1 - α_d)·hyp_ang`` is ``base``. ``autograd.grad`` neither writes ``.grad``
nor draws random numbers, so monitoring leaves training bit-identical. Each sampled step costs
one extra backward pass per part.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Mapping, Optional, Sequence

import torch
from torch import nn

from planora_sf.models.backbone import BackboneAdapter

ADDITIVE_PARTS = ("base", "fourier", "debias")  # sum to L_total; the HypCD branches decompose "base"


def shared_parameters(backbone: BackboneAdapter) -> dict[str, list[nn.Parameter]]:
    """``{"block<i>": trainable params of block i}`` for every trainable backbone block, in order."""
    return {
        f"block{i}": [p for p in backbone.blocks[i].parameters() if p.requires_grad]
        for i in backbone.trainable_block_indices()
    }


def _cos(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return None
    return float(torch.dot(a, b) / (na * nb))


def gradient_stats(parts: Mapping[str, torch.Tensor], groups: Mapping[str, Sequence[nn.Parameter]]) -> dict[str, float]:
    """Norms and pairwise cosines of each part's gradient, over all groups together and per group.

    Keys: ``norm/<part>``, ``cos/<a>~<b>`` (all blocks), ``<group>/norm/<part>``, ``<group>/cos/<a>~<b>``.
    A cosine with a zero gradient is undefined and left out.
    """
    params = [p for group in groups.values() for p in group]
    sizes = {name: sum(p.numel() for p in group) for name, group in groups.items()}
    flat: dict[str, torch.Tensor] = {}
    for name, loss in parts.items():
        if not loss.requires_grad:
            continue
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        flat[name] = torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1).double() for g, p in zip(grads, params)
        ])
    out: dict[str, float] = {}
    scopes = {"": slice(None)}
    start = 0
    for name, size in sizes.items():
        scopes[f"{name}/"] = slice(start, start + size)
        start += size
    for prefix, sl in scopes.items():
        for name, g in flat.items():
            out[f"{prefix}norm/{name}"] = float(g[sl].norm())
        for a, b in combinations(flat, 2):
            cos = _cos(flat[a][sl], flat[b][sl])
            if cos is not None:
                out[f"{prefix}cos/{a}~{b}"] = cos
    return out


class GradientMonitor:
    """Accumulates ``gradient_stats`` over the sampled steps of one epoch."""

    def __init__(self, backbone: BackboneAdapter, every: int):
        if every < 1:
            raise ValueError(f"grad monitor needs every >= 1, got {every}")
        self.groups = shared_parameters(backbone)
        self.every = every
        self.steps = 0
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._negative: dict[str, int] = defaultdict(int)

    def due(self, step: int) -> bool:
        return bool(self.groups) and step % self.every == 0

    def observe(self, parts: Mapping[str, torch.Tensor]) -> None:
        for key, value in gradient_stats(parts, self.groups).items():
            self._sums[key] += value
            self._counts[key] += 1
            if "cos/" in key and value < 0:
                self._negative[key] += 1
        self.steps += 1

    def summary(self) -> dict:
        """``{"steps", "mean": {key: mean}, "negative_fraction": {cos key: share of steps with cos < 0}}``."""
        return {
            "steps": self.steps,
            "mean": {k: self._sums[k] / self._counts[k] for k in sorted(self._sums)},
            "negative_fraction": {k: self._negative[k] / self._counts[k] for k in sorted(self._sums) if "cos/" in k},
        }
