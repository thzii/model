"""ToPoincare: Euclidean head output -> Poincaré ball (HypCD map M(z) = exp_0^c(C_r(z)); plan Phase 3).

Mirrors upstream ``hyptorch/nn.py::ToPoincare`` (HypCD @ ffc895b, MIT) as Hyp-SelEx uses it:
fixed c, ``riemannian=False``, ``train_x=False`` (train_HypSelEx.py:774-777). It has no
parameters or buffers, so adding it changes neither initialisation draws nor state dicts.
"""
from __future__ import annotations

import torch
from torch import nn

from planora_sf.hyperbolic.poincare import clip_features, expmap0, project


class ToPoincare(nn.Module):
    def __init__(self, c: float, clip_r: float, project_eps: float = 1e-3):
        super().__init__()
        if c <= 0 or clip_r <= 0:
            raise ValueError(f"ToPoincare needs c > 0 and clip_r > 0, got c={c}, clip_r={clip_r}")
        self.c = float(c)
        self.clip_r = float(clip_r)
        self.project_eps = float(project_eps)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return project(expmap0(clip_features(v, self.clip_r), self.c), self.c, self.project_eps)

    def extra_repr(self) -> str:
        return f"c={self.c}, clip_r={self.clip_r}, project_eps={self.project_eps}"
