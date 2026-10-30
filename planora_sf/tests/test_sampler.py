"""Statistical tests of the labelled/unlabelled-balancing sampler (SelEx ct.py:780-784).

With weights 1 (labelled) and L/(L+U) (unlabelled), every draw is labelled with
p = (L+U)/(L+2U). Within each group, draws are uniform. Draws are i.i.d., so
the labelled count per batch is Binomial(batch, p).
"""
import math

import pytest
import torch
from scipy.stats import chisquare
from torch.utils.data import DataLoader
from torchvision import transforms

from planora_sf.data import (
    DummyGCDDataset,
    GCDTrainDataset,
    build_gcd_sampler,
    expected_labelled_fraction,
    gcd_sample_weights,
    make_gcd_split,
)

L, U = 300, 900
P = (L + U) / (L + 2 * U)  # 4/7
N_DRAWS = 200_000


@pytest.fixture(scope="module")
def draws():
    sampler = build_gcd_sampler(L, U, num_samples=N_DRAWS, generator=torch.Generator().manual_seed(1234))
    return torch.tensor(list(sampler))


def test_weights_match_selex_formula():
    weights = gcd_sample_weights(L, U)
    assert weights.shape == (L + U,)
    assert torch.all(weights[:L] == 1.0) and torch.all(weights[L:] == L / (L + U))
    with pytest.raises(ValueError):
        gcd_sample_weights(0, 0)


def test_expected_fraction_closed_form():
    assert expected_labelled_fraction(L, U) == pytest.approx(4 / 7)
    assert expected_labelled_fraction(1498, 4496) == pytest.approx(5994 / 10490)  # CUB-sized split: ~0.5714
    assert expected_labelled_fraction(0, 10) == 0.0


def test_labelled_fraction_within_binomial_confidence(draws):
    assert draws.min() >= 0 and draws.max() < L + U
    frac = (draws < L).double().mean().item()
    sigma = math.sqrt(P * (1 - P) / N_DRAWS)  # ~0.0011
    assert abs(frac - P) < 4.5 * sigma, (frac, P)
    assert abs(frac - L / (L + U)) > 100 * sigma  # clearly not the unweighted 25%


def test_per_sample_frequencies_pass_chi_square(draws):
    observed = torch.bincount(draws, minlength=L + U).numpy()
    probs = gcd_sample_weights(L, U).numpy()
    expected = probs / probs.sum() * N_DRAWS
    stat, p_value = chisquare(observed, expected)
    assert p_value > 1e-3, (stat, p_value)
    # uniform within groups: a labelled index is drawn (L+U)/L = 4x as often as an unlabelled one
    ratio = observed[:L].mean() / observed[L:].mean()
    assert ratio == pytest.approx((L + U) / L, rel=0.02)


def test_batch_composition_is_binomial(draws):
    batch = 128
    per_batch = (draws[: N_DRAWS // batch * batch] < L).view(-1, batch).sum(dim=1).double()
    n_batches = per_batch.numel()
    mean_sigma = math.sqrt(batch * P * (1 - P) / n_batches)
    assert abs(per_batch.mean().item() - batch * P) < 4.5 * mean_sigma  # ~73.1 labelled per batch of 128
    assert 0.85 < per_batch.var().item() / (batch * P * (1 - P)) < 1.15


def test_sampler_through_dataloader_on_gcd_dataset():
    source = DummyGCDDataset(num_classes=8, images_per_class=12, image_size=8)
    split = make_gcd_split(source.targets)
    dataset = GCDTrainDataset(source, split, transforms.ToTensor())
    l, u = dataset.num_labelled, dataset.num_unlabelled
    assert (l, u) == (24, 72)
    n = 3200
    sampler = build_gcd_sampler(l, u, num_samples=n, generator=torch.Generator().manual_seed(7))
    flags = torch.cat([b[2] for b in DataLoader(dataset, batch_size=32, sampler=sampler)])
    p = expected_labelled_fraction(l, u)  # 96/168
    assert abs(flags.double().mean().item() - p) < 4.5 * math.sqrt(p * (1 - p) / n)
