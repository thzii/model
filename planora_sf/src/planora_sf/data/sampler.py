"""Sampler that balances labelled and unlabelled samples (SelEx ct.py:780-784, T-03).

Labelled samples get weight 1 and unlabelled samples weight L / (L + U). With
replacement, each draw is labelled with probability

    p_L = L / (L + U · L/(L+U)) = (L + U) / (L + 2U),

which is about 0.571 for the CUB split (L ≈ 1.5K, U ≈ 4.5K) instead of L/(L+U) ≈ 0.25.
Within each group, samples are drawn uniformly.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import WeightedRandomSampler


def gcd_sample_weights(num_labelled: int, num_unlabelled: int) -> torch.Tensor:
    """Per-sample weights for a dataset ordered labelled-first."""
    if num_labelled < 0 or num_unlabelled < 0 or num_labelled + num_unlabelled == 0:
        raise ValueError("need a non-empty dataset")
    total = num_labelled + num_unlabelled
    weights = torch.ones(total, dtype=torch.double)
    weights[num_labelled:] = num_labelled / total
    return weights


def expected_labelled_fraction(num_labelled: int, num_unlabelled: int) -> float:
    return (num_labelled + num_unlabelled) / (num_labelled + 2 * num_unlabelled) if num_labelled else 0.0


def build_gcd_sampler(
    num_labelled: int,
    num_unlabelled: int,
    *,
    num_samples: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> WeightedRandomSampler:
    """``num_samples`` defaults to the dataset size (one SelEx epoch); sampling is with replacement."""
    weights = gcd_sample_weights(num_labelled, num_unlabelled)
    return WeightedRandomSampler(
        weights, num_samples=num_samples or weights.numel(), replacement=True, generator=generator
    )
