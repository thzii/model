"""Latent subspace allocation (spec §3.3)."""
import pytest
import torch

from planora_sf.latent.partition import (
    high_width,
    low_width,
    lsse_widths,
    prefix_mask,
    selex_prefix_width,
    suffix_mask,
)

D = 65536


def test_low_width_reference_values():
    assert low_width(23, 23, D).item() == D // 2
    assert low_width(1, 23, D).item() == 1425  # round(32768 / 23)
    assert low_width(1, 23, D, fraction_of_half=False).item() == 2849  # round(65536 / 23)
    assert low_width(0, 23, D).item() == 1  # clamped to at least one dimension


def test_high_width_modes():
    assert high_width(0, 23, D).item() == D // 2  # f_h = 0: whole right half
    assert high_width(22, 23, D).item() == 1425  # f_h = T_h - 1: 1/T_h of the half (FX-P Fig. 5 caption)
    assert high_width(22, 23, D, mode="ratio").item() == round(22 / 23 * D / 2)
    with pytest.raises(ValueError):
        high_width(30, 23, D)
    with pytest.raises(ValueError):
        high_width(3, 23, D, mode="bogus")


def test_low_and_high_slices_live_in_disjoint_halves():
    f = torch.arange(0, 24)
    lo = prefix_mask(low_width(f.clamp(min=1), 23, D), D)
    hi = suffix_mask(high_width(f.clamp(max=22), 23, D), D)
    assert lo.shape == hi.shape == (24, D)
    assert not lo[:, D // 2 :].any() and not hi[:, : D // 2].any()
    assert torch.equal(lo.sum(1), low_width(f.clamp(min=1), 23, D))


def test_masks_select_exact_ranges():
    widths = torch.tensor([1, 3, 8])
    pre, suf = prefix_mask(widths, 8), suffix_mask(widths, 8)
    assert pre[1].tolist() == [True] * 3 + [False] * 5
    assert suf[1].tolist() == [False] * 5 + [True] * 3
    assert pre[2].all() and suf[2].all()


def test_selex_prefixes_for_cub_hierarchy():
    assert [selex_prefix_width(D, k) for k in range(8)] == [65536, 32768, 16384, 8192, 4096, 2048, 1024, 512]
    assert selex_prefix_width(4, 10) == 1


def test_lsse_min_rule():
    d_low = torch.tensor([100, 5000, 32768])
    assert lsse_widths(d_low, D, 1).tolist() == [100, 5000, 32768]
    assert lsse_widths(d_low, D, 4).tolist() == [100, 4096, 4096]
