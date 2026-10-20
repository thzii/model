"""Fourier mask correctness (spec §4.1-4.2, FX-P §3)."""
import pytest
import torch

from planora_sf.fourier.filters import frequency_grid, highpass_mask, lowpass_mask


def test_frequency_grid_is_signed_and_unshifted():
    grid = frequency_grid(224)
    assert grid[0] == 0 and grid[1] == 1 and grid[111] == 111 and grid[112] == -112 and grid[-1] == -1
    assert grid.min() == -112 and grid.max() == 111
    assert torch.unique(grid).numel() == 224
    odd = frequency_grid(31)
    assert odd.min() == -15 and odd.max() == 15


@pytest.mark.parametrize("f", [1, 5, 23, 55])
def test_lowpass_keeps_centred_square(f):
    strict = lowpass_mask(224, 224, f)
    assert strict.shape == (1, 1, 224, 224)
    assert int(strict.sum()) == (2 * f - 1) ** 2  # side 2f - 1 for |u| < f
    loose = lowpass_mask(224, 224, f, strict=False)
    assert int(loose.sum()) == (2 * f + 1) ** 2  # side 2f + 1 for |u| <= f (U-06)


def test_strict_cutoff_one_keeps_only_dc():
    mask = lowpass_mask(32, 32, 1)[0, 0]
    assert mask[0, 0] == 1 and int(mask.sum()) == 1


@pytest.mark.parametrize("size", [(224, 224), (31, 40)])
def test_masks_are_point_symmetric(size):
    h, w = size
    neg_u, neg_v = (-torch.arange(h)) % h, (-torch.arange(w)) % w
    for mask in (lowpass_mask(h, w, 9)[0, 0], highpass_mask(h, w, 9)[0, 0]):
        assert torch.equal(mask[neg_u][:, neg_v], mask)


def test_highpass_is_complement_and_band_option():
    lp, hp = lowpass_mask(64, 64, 7), highpass_mask(64, 64, 7)
    assert torch.equal(lp + hp, torch.ones_like(lp))
    assert hp[0, 0, 0, 0] == 0  # DC removed
    band = highpass_mask(64, 64, 7, band_upper=12)
    assert torch.equal(band, hp * lowpass_mask(64, 64, 12))
    grid = frequency_grid(64).abs()
    radius = torch.maximum(grid[:, None], grid[None, :])
    assert band[0, 0][radius >= 12].sum() == 0 and band[0, 0][(radius >= 7) & (radius < 12)].min() == 1


def test_per_image_cutoffs_match_scalar_masks():
    cutoffs = torch.tensor([1, 4, 9])
    batch = lowpass_mask(48, 48, cutoffs)
    assert batch.shape == (3, 1, 48, 48)
    for i, f in enumerate(cutoffs.tolist()):
        assert torch.equal(batch[i], lowpass_mask(48, 48, f)[0])


def test_invalid_cutoffs_rejected():
    with pytest.raises(ValueError):
        lowpass_mask(32, 32, -1)
    with pytest.raises(ValueError):
        lowpass_mask(32, 32, torch.ones(2, 2))
