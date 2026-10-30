"""HypCD: Poincaré-ball geometry for the projection-head space (experiment E4)."""
from planora_sf.hyperbolic.layers import ToPoincare
from planora_sf.hyperbolic.poincare import (
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

__all__ = [
    "ToPoincare",
    "clip_features",
    "dist0",
    "distance_from_squares",
    "expmap0",
    "lambda_x",
    "logmap0",
    "mobius_add",
    "pairwise_poincare_distance",
    "poincare_distance",
    "project",
]
