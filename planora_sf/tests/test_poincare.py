"""Poincaré-ball geometry (plan Phase 2): identities in float64, clipping, float32 numerics at
the real embedding size, and optional parity with upstream hyptorch. Synthetic tensors only."""
import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from planora_sf.hyperbolic import (
    clip_features,
    dist0,
    distance_from_squares,
    expmap0,
    lambda_x,
    logmap0,
    mobius_add,
    pairwise_poincare_distance,
    poincare_distance,
    project,
)

CURVATURES = [0.01, 0.05, 0.1, 1.0]
D64 = torch.float64


def _ball_points(n, dim, c, max_frac=0.95, dtype=D64):
    """Random points with ‖x‖ < max_frac / √c."""
    direction = F.normalize(torch.randn(n, dim, dtype=dtype), dim=1)
    radius = torch.rand(n, 1, dtype=dtype) * max_frac / math.sqrt(c)
    return direction * radius


def _tangent_vectors(n, dim, c, max_sqrt_c_norm=4.0):
    """Random v with √c‖v‖ < 4. (The artanh clamp 1 - 1e-5 makes log0 saturate beyond √c‖v‖ ≈ 6.1.)"""
    direction = F.normalize(torch.randn(n, dim, dtype=D64), dim=1)
    return direction * torch.rand(n, 1, dtype=D64) * max_sqrt_c_norm / math.sqrt(c)


def _arcosh_distance(x, y, c):
    """Third closed form: (1/√c) arcosh(1 + 2c‖x-y‖² / ((1-c‖x‖²)(1-c‖y‖²)))."""
    num = 2 * c * (x - y).pow(2).sum(-1)
    den = (1 - c * x.pow(2).sum(-1)) * (1 - c * y.pow(2).sum(-1))
    return torch.acosh(1 + num / den) / math.sqrt(c)


# ---------------------------------------------------------------- exp / log maps

@pytest.mark.parametrize("c", CURVATURES)
def test_log0_inverts_exp0_and_back(c):
    v = _tangent_vectors(200, 16, c)
    torch.testing.assert_close(logmap0(expmap0(v, c), c), v, rtol=1e-9, atol=1e-9)
    x = _ball_points(200, 16, c)
    torch.testing.assert_close(expmap0(logmap0(x, c), c), x, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", CURVATURES)
def test_exp0_lands_in_the_ball_and_preserves_direction(c):
    v = torch.randn(100, 8, dtype=D64) * 50
    x = expmap0(v, c)
    assert (x.norm(dim=1) < 1 / math.sqrt(c)).all()
    torch.testing.assert_close(F.normalize(x, dim=1), F.normalize(v, dim=1))


def test_exp0_is_identity_to_first_order_and_at_zero():
    v = torch.randn(10, 8, dtype=D64) * 1e-7
    torch.testing.assert_close(expmap0(v, 0.05), v, rtol=1e-9, atol=0)
    assert torch.equal(expmap0(torch.zeros(3, 8, dtype=D64), 0.05), torch.zeros(3, 8, dtype=D64))


@pytest.mark.parametrize("c", CURVATURES)
def test_distance_from_origin_of_exp0_is_twice_the_tangent_norm(c):
    v = _tangent_vectors(100, 12, c)
    torch.testing.assert_close(dist0(expmap0(v, c), c), 2 * v.norm(dim=1), rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------- Möbius addition

@pytest.mark.parametrize("c", CURVATURES)
def test_mobius_identities(c):
    x, y = _ball_points(100, 10, c), _ball_points(100, 10, c)
    zero = torch.zeros_like(x)
    torch.testing.assert_close(mobius_add(x, zero, c), x)
    torch.testing.assert_close(mobius_add(zero, y, c), y)
    torch.testing.assert_close(mobius_add(-x, x, c), zero, atol=1e-12, rtol=0)
    torch.testing.assert_close(mobius_add(-x, mobius_add(x, y, c), c), y, rtol=1e-9, atol=1e-10)  # left cancellation
    assert (mobius_add(x, y, c).norm(dim=1) < 1 / math.sqrt(c)).all()


# ---------------------------------------------------------------- distances

@pytest.mark.parametrize("c", CURVATURES)
def test_three_distance_forms_agree(c):
    x, y = _ball_points(150, 10, c), _ball_points(150, 10, c)
    artanh_form = poincare_distance(x, y, c)
    torch.testing.assert_close(artanh_form, _arcosh_distance(x, y, c), rtol=1e-9, atol=1e-9)
    asinh_form = distance_from_squares((x - y).pow(2).sum(-1), x.pow(2).sum(-1), y.pow(2).sum(-1), c)
    torch.testing.assert_close(asinh_form, artanh_form, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("c", CURVATURES)
def test_pairwise_matrix_matches_elementwise_distance(c):
    x, y = _ball_points(20, 10, c), _ball_points(30, 10, c)
    expected = poincare_distance(x[:, None, :], y[None, :, :], c)
    torch.testing.assert_close(pairwise_poincare_distance(x, y, c), expected, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("c", CURVATURES)
def test_metric_axioms(c):
    x, y, z = (_ball_points(300, 6, c) for _ in range(3))
    dxy = poincare_distance(x, y, c)
    torch.testing.assert_close(dxy, poincare_distance(y, x, c), rtol=1e-10, atol=1e-10)  # symmetry
    assert (dxy > 0).all()
    assert poincare_distance(x, x, c).abs().max() < 1e-6
    assert (dxy <= poincare_distance(x, z, c) + poincare_distance(z, y, c) + 1e-9).all()  # triangle
    # Diagonal of the matrix form: the 1e-12 floor under ‖x-y‖² gives d = 2e-6 / (1 - c‖x‖²) up to O(1e-12).
    xs = x[:50]
    bound = 2e-6 / (1 - c * xs.pow(2).sum(1)) * (1 + 1e-6)
    assert (pairwise_poincare_distance(xs, xs, c).diagonal() <= bound).all()


@pytest.mark.parametrize("c", CURVATURES)
def test_mobius_translation_is_an_isometry(c):
    """d(a ⊕ x, a ⊕ y) = d(x, y): checks the distance independently of its closed forms."""
    a, x, y = (_ball_points(200, 8, c, max_frac=0.8) for _ in range(3))
    torch.testing.assert_close(
        poincare_distance(mobius_add(a, x, c), mobius_add(a, y, c), c), poincare_distance(x, y, c), rtol=1e-8, atol=1e-8
    )


def test_small_curvature_limit_is_twice_euclidean():
    x, y = torch.randn(50, 8, dtype=D64), torch.randn(50, 8, dtype=D64)
    torch.testing.assert_close(poincare_distance(x, y, 1e-10), 2 * (x - y).norm(dim=1), rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(pairwise_poincare_distance(x, y, 1e-10), 2 * torch.cdist(x, y), rtol=1e-5, atol=1e-6)


def test_conformal_factor_and_curvature_validation():
    x = _ball_points(5, 4, 0.1)
    torch.testing.assert_close(lambda_x(x, 0.1).squeeze(1), 2 / (1 - 0.1 * x.pow(2).sum(1)))
    for fn in (lambda: expmap0(x, 0.0), lambda: poincare_distance(x, x, -1.0), lambda: clip_features(x, 0.0)):
        with pytest.raises(ValueError):
            fn()


# ---------------------------------------------------------------- clipping and projection

def test_clipping_caps_the_norm_and_keeps_short_vectors_and_direction():
    r = 2.3
    long = torch.randn(100, 16, dtype=D64) * 10
    short = F.normalize(torch.randn(100, 16, dtype=D64), dim=1) * (torch.rand(100, 1, dtype=D64) * 2.2)
    clipped = clip_features(long, r)
    assert (clipped.norm(dim=1) <= r).all() and (clipped.norm(dim=1) > r - 1e-4).all()
    torch.testing.assert_close(F.normalize(clipped, dim=1), F.normalize(long, dim=1))
    assert torch.equal(clip_features(short, r), short)


def test_active_clipping_has_no_radial_gradient():
    """∂C_r/∂v = (r/‖v‖)(I - v̂v̂ᵀ) when clipped: the Jacobian annihilates v (plan Phase 3 note)."""
    v_long = torch.randn(16, dtype=D64) * 10
    jac = torch.autograd.functional.jacobian(lambda t: clip_features(t, 2.3), v_long)
    n = v_long.norm()
    expected = (2.3 / (n + 1e-5)) * (torch.eye(16, dtype=D64) - torch.outer(v_long, v_long) / (n * (n + 1e-5)))
    torch.testing.assert_close(jac, expected, rtol=1e-6, atol=1e-8)
    assert (jac @ v_long).abs().max() < 1e-5
    v_short = F.normalize(torch.randn(16, dtype=D64), dim=0)
    torch.testing.assert_close(torch.autograd.functional.jacobian(lambda t: clip_features(t, 2.3), v_short),
                               torch.eye(16, dtype=D64))


@pytest.mark.parametrize("c, r", [(0.05, 2.3), (0.1, 2.0)])  # paper, upstream script
def test_clipped_embeddings_respect_the_radius_bounds(c, r):
    v = torch.randn(200, 32, dtype=D64) * 20
    z = expmap0(clip_features(v, r), c)
    assert (dist0(z, c) <= 2 * r + 1e-9).all()
    assert pairwise_poincare_distance(z, z, c).max() <= 4 * r + 1e-9
    max_cz2 = math.tanh(math.sqrt(c) * r) ** 2  # c‖z‖² ≤ tanh²(√c·r): 0.224 (paper), 0.313 (upstream)
    assert (c * z.pow(2).sum(1) <= max_cz2 + 1e-12).all()


def test_clipping_and_exp0_keep_cosines():
    """Both maps are radial, so cos(z_i, z_j) = cos(v_i, v_j): the paper's angle branch ignores c and r."""
    v = torch.randn(40, 16, dtype=D64) * torch.rand(40, 1, dtype=D64) * 8
    z = expmap0(clip_features(v, 2.3), 0.05)
    torch.testing.assert_close(F.normalize(z, dim=1) @ F.normalize(z, dim=1).T,
                               F.normalize(v, dim=1) @ F.normalize(v, dim=1).T)


def test_project_only_moves_points_beyond_the_safety_radius():
    c = 0.1
    inside = _ball_points(50, 8, c, max_frac=0.99)
    assert torch.equal(project(inside, c), inside)
    outside = F.normalize(torch.randn(50, 8, dtype=D64), dim=1) * 5 / math.sqrt(c)
    torch.testing.assert_close(project(outside, c).norm(dim=1), torch.full((50,), 0.999 / math.sqrt(c), dtype=D64))


# ---------------------------------------------------------------- gradients and float32 numerics

def test_gradcheck_float64():
    c = 0.05
    v = (torch.randn(4, 5, dtype=D64) * 2).requires_grad_()
    x = (_ball_points(4, 5, c, 0.8)).requires_grad_()
    y = (_ball_points(3, 5, c, 0.8)).requires_grad_()
    assert torch.autograd.gradcheck(lambda t: expmap0(t, c), (v,))
    assert torch.autograd.gradcheck(lambda t: logmap0(t, c), (x,))
    assert torch.autograd.gradcheck(lambda a, b: mobius_add(a, b[:1].expand_as(a), c), (x, y))
    assert torch.autograd.gradcheck(lambda a, b: poincare_distance(a[:3], b, c), (x, y))
    assert torch.autograd.gradcheck(lambda a, b: pairwise_poincare_distance(a, b, c), (x, y))
    assert torch.autograd.gradcheck(lambda t: clip_features(t, 1.5), (v,))


@pytest.mark.parametrize("c, r", [(0.05, 2.3), (0.1, 2.0)])
def test_float32_at_full_embedding_size_is_finite_including_the_diagonal(c, r):
    """2B = 256 head outputs of the real D = 65,536 (‖v‖ ≈ 16, as DINOHead at init): forward and backward."""
    v = (F.normalize(torch.randn(256, 65536), dim=1) * 16).requires_grad_()
    z = project(expmap0(clip_features(v, r), c), c)
    d = pairwise_poincare_distance(z, z, c)
    assert torch.isfinite(d).all() and d.max() <= 4 * r + 1e-3
    # float32 cancellation in ‖x‖² + ‖y‖² - 2x·y leaves ‖x-y‖² ~ 1e-6 on the diagonal, i.e. d(x, x) ~ 1e-2 at most.
    assert d.diagonal().max() < 1e-2
    d.sum().backward()  # the diagonal (x = y) is included on purpose
    assert torch.isfinite(v.grad).all()
    exact = poincare_distance(z[:16, None].double(), z[None, :16].double(), c)  # artanh/Möbius form, float64
    off = ~torch.eye(16, dtype=torch.bool)
    torch.testing.assert_close(d[:16, :16].double()[off], exact[off], rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------- optional parity with upstream hyptorch

UPSTREAM = os.environ.get("HYPCD_UPSTREAM")


def _upstream():
    root = Path(UPSTREAM)
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location("hyptorch_pmath", root / "hyptorch" / "pmath.py")
        pmath = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pmath)
        sys.modules["hyptorch.pmath"] = pmath  # nn.py does `import hyptorch.pmath as pmath`
        spec = importlib.util.spec_from_file_location("hyptorch_nn", root / "hyptorch" / "nn.py")
        hnn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hnn)
    finally:
        sys.path.remove(str(root))
        sys.modules.pop("hyptorch.pmath", None)
    return pmath, hnn


@pytest.mark.skipif(not UPSTREAM, reason="set HYPCD_UPSTREAM=/path/to/HypCD clone to compare with hyptorch")
@pytest.mark.parametrize("c, r", [(0.05, 2.3), (0.1, 2.0)])
def test_parity_with_upstream_hyptorch(c, r):
    from planora_sf.hyperbolic import ToPoincare

    pmath, hnn = _upstream()
    v = torch.randn(64, 128) * 3
    theirs = hnn.ToPoincare(c=c, riemannian=False, clip_r=r)(v)  # as Hyp-SelEx builds it
    ours = ToPoincare(c, r)(v)
    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-6)
    x = ours[:32]
    # Upstream adds 1e-5 to the Möbius denominator; agreement is therefore ~1e-5 relative.
    ours_d, theirs_d = pairwise_poincare_distance(x, x, c), pmath.dist_matrix(x, x, c=c)
    off = ~torch.eye(32, dtype=torch.bool)
    torch.testing.assert_close(ours_d[off], theirs_d[off], rtol=1e-4, atol=1e-4)
    # Diagonal: upstream's O(N²D) Möbius form gives exactly 0; the O(N²) form keeps the float32
    # cancellation floor of ‖x‖² + ‖y‖² - 2x·y (see the full-size float32 test).
    assert theirs_d.diagonal().abs().max() < 1e-4 and ours_d.diagonal().max() < 1e-2
    torch.testing.assert_close(poincare_distance(x[:16], x[16:], c), pmath.dist(x[:16], x[16:], c=c), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(logmap0(x, c), pmath.logmap0(x, c=c), rtol=1e-5, atol=1e-6)
