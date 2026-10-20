"""Fourier view generation (spec §4.2): shapes, reality, complementarity, DC behaviour, U-04 claim."""
import pytest
import torch

from planora_sf.config import FourierConfig
from planora_sf.fourier.filters import highpass_mask, lowpass_mask
from planora_sf.fourier.transforms import apply_frequency_mask
from planora_sf.fourier.views import CutoffSampler, FourierViewGenerator, gaussian_blur


@pytest.fixture
def images():
    return torch.randn(2, 3, 224, 224) * 0.5 + torch.tensor([0.4, -0.2, 0.1]).view(1, 3, 1, 1)


def test_views_have_input_shape_and_are_finite(images, generator):
    gen = FourierViewGenerator(FourierConfig(cutoff=23))
    views = gen(images, generator=generator)
    for t in (views.low, views.high, views.anchor):
        assert t.shape == images.shape and t.dtype == images.dtype and torch.isfinite(t).all()
    assert views.f_low.shape == views.f_high.shape == (2,)
    assert ((views.f_low >= 1) & (views.f_low <= 23)).all() and ((views.f_high >= 0) & (views.f_high < 23)).all()


def test_inverse_is_real_for_symmetric_masks(images):
    filtered = apply_frequency_mask(images, lowpass_mask(224, 224, 23), return_complex=True)
    assert filtered.imag.abs().max() < 1e-4 * filtered.real.abs().max()


def test_low_plus_high_reconstructs_blurred_input(images):
    gen = FourierViewGenerator(FourierConfig(cutoff=23))
    views = gen(images, f_low=11, f_high=11)
    blurred = gaussian_blur(images, 5, 1.1)
    torch.testing.assert_close(views.low + views.high, blurred, atol=1e-4, rtol=1e-4)


def test_highpass_zero_mean_lowpass_keeps_mean(images):
    gen = FourierViewGenerator(FourierConfig(cutoff=23))
    views = gen(images, f_low=5, f_high=5)
    blurred = gaussian_blur(images, 5, 1.1)
    assert views.high.mean(dim=(-2, -1)).abs().max() < 1e-5  # DC removed: colour is lost (FX-P §5.3)
    torch.testing.assert_close(views.low.mean(dim=(-2, -1)), blurred.mean(dim=(-2, -1)), atol=1e-5, rtol=1e-5)


def test_dc_only_and_identity_edge_cases(images):
    gen = FourierViewGenerator(FourierConfig(cutoff=23))
    views = gen(images, f_low=1, f_high=0)
    blurred = gaussian_blur(images, 5, 1.1)
    channel_mean = blurred.mean(dim=(-2, -1), keepdim=True).expand_as(blurred)
    torch.testing.assert_close(views.low, channel_mean, atol=1e-5, rtol=1e-5)  # f_l = 1 keeps only DC
    torch.testing.assert_close(views.high, blurred, atol=1e-5, rtol=1e-5)  # f_h = 0 is the identity


def test_lowpass_energy_grows_with_cutoff(images):
    energies = [apply_frequency_mask(images, lowpass_mask(224, 224, f)).pow(2).sum() for f in range(1, 30, 4)]
    assert all(b >= a for a, b in zip(energies, energies[1:]))


def test_per_image_cutoffs_are_applied_per_sample(images):
    gen = FourierViewGenerator(FourierConfig(cutoff=23))
    batched = gen(images, f_low=torch.tensor([3, 17]), f_high=torch.tensor([2, 20]))
    for i, (fl, fh) in enumerate([(3, 2), (17, 20)]):
        single = gen(images[i : i + 1], f_low=fl, f_high=fh)
        torch.testing.assert_close(batched.low[i], single.low[0])
        torch.testing.assert_close(batched.high[i], single.high[0])


def test_normalisation_order_matters_only_for_highpass(images):
    """U-04: LPF keeps DC, so it commutes with per-channel affine normalisation; HPF does not."""
    mean = images.mean(dim=(0, 2, 3), keepdim=True)
    std = images.std(dim=(0, 2, 3), keepdim=True)
    lp, hp = lowpass_mask(224, 224, 9), highpass_mask(224, 224, 9)
    norm = lambda x: (x - mean) / std  # noqa: E731
    torch.testing.assert_close(apply_frequency_mask(norm(images), lp), norm(apply_frequency_mask(images, lp)), atol=1e-4, rtol=1e-4)
    hp_of_norm = apply_frequency_mask(norm(images), hp)
    torch.testing.assert_close(hp_of_norm, apply_frequency_mask(images, hp) / std, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(hp_of_norm, norm(apply_frequency_mask(images, hp)), atol=1e-3)


def test_band_pass_option_has_no_energy_above_t(images):
    gen = FourierViewGenerator(FourierConfig(cutoff=10, band_pass=True))
    views = gen(images, f_low=3, f_high=3)
    spectrum = torch.fft.fft2(views.high)
    above = 1.0 - lowpass_mask(224, 224, 10)
    assert (spectrum.abs() * above).max() < 1e-5 * spectrum.abs().max()


def test_gaussian_blur_preserves_constants_and_zero_input_is_finite():
    const = torch.full((1, 3, 16, 16), 0.7)
    torch.testing.assert_close(gaussian_blur(const, 5, 1.1), const)
    gen = FourierViewGenerator(FourierConfig(cutoff=4))
    views = gen(torch.zeros(2, 3, 16, 16))
    assert torch.isfinite(views.low).all() and torch.isfinite(views.high).all()
    assert views.low.abs().max() == 0 and views.high.abs().max() == 0


def test_cutoff_sampler_ranges_and_per_batch_mode(generator):
    f_low, f_high = CutoffSampler(6, 6).sample(4000, generator=generator)
    assert set(f_low.tolist()) == set(range(1, 7)) and set(f_high.tolist()) == set(range(0, 6))
    f_low, f_high = CutoffSampler(6, 6, per_image=False).sample(16, generator=generator)
    assert f_low.unique().numel() == 1 and f_high.unique().numel() == 1
    with pytest.raises(ValueError):
        CutoffSampler(6, 0)


def test_generator_requires_cutoff_and_4d_input():
    with pytest.raises(ValueError):
        FourierViewGenerator(FourierConfig())
    gen = FourierViewGenerator(FourierConfig(cutoff=4))
    with pytest.raises(ValueError):
        gen(torch.randn(3, 16, 16))
