"""Poincaré-ball operations for HypCD (Liu et al., CVPR 2025, arXiv 2504.06120 §3; plan Phase 2).

Ball D_c^n = {x : c‖x‖² < 1}, curvature -c (c > 0), radius 1/√c. Formulas follow the
paper and the official code's ``hyptorch/pmath.py`` (Visual-AI/HypCD @ ffc895b, MIT licence,
Copyright (c) 2025 Visual AI Lab; itself partly based on geoopt). Pure functions, no parameters.
Differences from upstream are numerical only and are noted where they occur.
"""
from __future__ import annotations

import torch

_MIN_NORM = 1e-15  # guards divisions by ‖v‖ (upstream uses 1e-5; only affects ‖v‖ < 1e-5)
_ARTANH_MAX = 1 - 1e-5  # upstream Artanh clamp (pmath.py:19)
_TANH_MAX_ARG = 15.0  # upstream tanh clamp (pmath.py:11)
_SQ_EPS = 1e-12  # floor under squared distances before sqrt (as losses/contrastive.py)


def _sqrt_c(c: float) -> float:
    if c <= 0:
        raise ValueError(f"curvature c must be > 0, got {c}")
    return c ** 0.5


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x.norm(dim=-1, keepdim=True).clamp_min(_MIN_NORM)


def clip_features(v: torch.Tensor, r: float) -> torch.Tensor:
    """C_r(v) = min(1, r/‖v‖) · v (feature clipping; paper Eq. for C, upstream nn.py:154-160).

    As upstream, ``1e-5`` is added to ‖v‖ in the ratio.
    """
    if r <= 0:
        raise ValueError(f"clip radius r must be > 0, got {r}")
    factor = torch.clamp(r / (v.norm(dim=-1, keepdim=True) + 1e-5), max=1.0)
    return v * factor


def expmap0(v: torch.Tensor, c: float) -> torch.Tensor:
    """exp_0^c(v) = tanh(√c‖v‖) · v / (√c‖v‖)  (pmath.py:300-304)."""
    sc = _sqrt_c(c)
    n = _norm(v)
    return torch.tanh((sc * n).clamp(max=_TANH_MAX_ARG)) * v / (sc * n)


def logmap0(x: torch.Tensor, c: float) -> torch.Tensor:
    """log_0^c(x) = artanh(√c‖x‖) · x / (√c‖x‖)  (pmath.py:365-368)."""
    sc = _sqrt_c(c)
    n = _norm(x)
    return torch.atanh((sc * n).clamp(max=_ARTANH_MAX)) * x / (sc * n)


def project(x: torch.Tensor, c: float, eps: float = 1e-3) -> torch.Tensor:
    """Rescale points outside radius (1 - eps)/√c back onto it (pmath.py:98-103)."""
    max_norm = (1 - eps) / _sqrt_c(c)
    n = _norm(x)
    return torch.where(n > max_norm, x / n * max_norm, x)


def lambda_x(x: torch.Tensor, c: float) -> torch.Tensor:
    """Conformal factor λ_x^c = 2 / (1 - c‖x‖²), shape [..., 1]."""
    return 2 / (1 - c * x.pow(2).sum(dim=-1, keepdim=True))


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """x ⊕_c y (paper Eq. 5, pmath.py:171-177). Upstream adds 1e-5 to the denominator; this clamps it."""
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c**2 * x2 * y2
    return num / den.clamp_min(_MIN_NORM)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """Element-wise d_c(x, y) = (2/√c) · artanh(√c ‖(-x) ⊕_c y‖), the paper's form (pmath.py:205-208)."""
    sc = _sqrt_c(c)
    return 2 / sc * torch.atanh((sc * mobius_add(-x, y, c).norm(dim=-1)).clamp(max=_ARTANH_MAX))


def distance_from_squares(sq_dist: torch.Tensor, x_sq: torch.Tensor, y_sq: torch.Tensor, c: float) -> torch.Tensor:
    """d_c = (2/√c) · asinh(√c ‖x-y‖ / √((1 - c‖x‖²)(1 - c‖y‖²))), from squared quantities.

    Equal to the artanh/Möbius form via arcosh(1 + 2s²) = 2·asinh(s). ``sq_dist`` is ‖x-y‖²
    (any broadcastable shape); ``x_sq``, ``y_sq`` are ‖x‖², ‖y‖² broadcastable to it. Callers
    that restrict coordinates (masked prefixes) pass masked squares. asinh is smooth at 0, and
    the floor under ``sq_dist`` keeps the gradient finite at x = y (e.g. a matrix diagonal).
    """
    sc = _sqrt_c(c)
    conformal = ((1 - c * x_sq) * (1 - c * y_sq)).clamp_min(_MIN_NORM)
    return 2 / sc * torch.asinh(sc * sq_dist.clamp_min(_SQ_EPS).sqrt() / conformal.sqrt())


def pairwise_poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """``[N, M]`` Poincaré distances between ``x [N, D]`` and ``y [M, D]``.

    Same value as upstream ``dist_matrix`` (pmath.py:478-484) in O(N·M) memory instead of
    O(N·M·D): it never materialises the batched Möbius sum.
    """
    x_sq = (x * x).sum(dim=1, keepdim=True)
    y_sq = (y * y).sum(dim=1)[None, :]
    sq_dist = x_sq + y_sq - 2 * x @ y.T
    return distance_from_squares(sq_dist, x_sq, y_sq, c)


def dist0(x: torch.Tensor, c: float) -> torch.Tensor:
    """d_c(0, x) = (2/√c) · artanh(√c‖x‖)  (pmath.py:231-234)."""
    sc = _sqrt_c(c)
    return 2 / sc * torch.atanh((sc * x.norm(dim=-1)).clamp(max=_ARTANH_MAX))
