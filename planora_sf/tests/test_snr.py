"""SNR cut-off calibration mechanics (spec §4.3) on synthetic spectra and images."""
import math

import pytest
import torch

from planora_sf.fourier.snr import (
    calibrate_cutoff,
    calibrate_from_batches,
    frequency_radius,
    mean_power_spectrum,
    select_cutoff,
    snr_curve,
)


def test_frequency_radius_metrics():
    box = frequency_radius(8, 8, "box")
    assert box[0, 0] == 0 and box[3, 1] == 3 and box[5, 0] == 3 and box.max() == 4
    radial = frequency_radius(8, 8, "radial")
    assert radial[3, 4] == 5
    with pytest.raises(ValueError):
        frequency_radius(8, 8, "hexagon")


def test_snr_curve_matches_hand_computation():
    power = torch.zeros(32, 32, dtype=torch.float64)
    power[0, 0] = 1000.0  # DC
    power[7, 0] = power[-7, 0] = 5.0  # a pair of bins at box radius 7
    power[0, 12] = 1.0  # one bin at radius 12
    curve = snr_curve(power)
    assert curve.shape == (16,)  # cut-offs 0 .. 15 at 32 px
    torch.testing.assert_close(curve[0], torch.tensor(10 * math.log10(1000 / 11), dtype=torch.float64))
    torch.testing.assert_close(curve[7], torch.tensor(10 * math.log10(1010 / 1), dtype=torch.float64))
    assert torch.isfinite(curve).all() and (curve[12:] > 100).all()  # no energy left above: clamped, finite
    assert select_cutoff(curve, 25.0) == 7
    assert select_cutoff(curve, 1e6) == curve.numel()  # saturation


def test_snr_curve_is_non_decreasing_on_random_images():
    curve = snr_curve(mean_power_spectrum(torch.rand(4, 3, 64, 64)))
    assert (curve[1:] >= curve[:-1] - 1e-9).all()


def test_calibration_orders_smooth_vs_noisy_classes():
    yy, xx = torch.meshgrid(torch.arange(32.0), torch.arange(32.0), indexing="ij")
    smooth = torch.stack(
        [torch.cos(2 * math.pi * (xx + p) / 32) + 0.5 * torch.sin(2 * math.pi * 2 * yy / 32) for p in range(6)]
    )
    smooth = smooth[:, None].expand(-1, 3, -1, -1) + 0.01 * torch.randn(6, 3, 32, 32)
    noisy = torch.randn(6, 3, 32, 32)
    images = torch.cat([smooth, noisy])
    labels = torch.tensor([0] * 6 + [1] * 6)
    result = calibrate_cutoff(images, labels, target_db=15.0)
    assert result.per_class[0] < result.per_class[1]
    assert result.cutoff == math.floor((result.per_class[0] + result.per_class[1]) / 2 + 0.5)


def test_streaming_calibration_equals_in_memory():
    images = torch.rand(12, 3, 32, 32)
    labels = torch.tensor([0, 1, 2] * 4)
    in_memory = {db: calibrate_cutoff(images, labels, target_db=db) for db in (5.0, 15.0)}
    batches = [(images[i : i + 5], labels[i : i + 5]) for i in range(0, 12, 5)]  # uneven batches
    streamed = calibrate_from_batches(batches, targets_db=(5.0, 15.0))
    for db in (5.0, 15.0):
        assert streamed[db].cutoff == in_memory[db].cutoff and streamed[db].per_class == in_memory[db].per_class
    with pytest.raises(ValueError):
        calibrate_from_batches([], targets_db=(15.0,))


@pytest.mark.skip(
    reason="acceptance check on real CUB: run `python -m planora_sf.cli.calibrate_snr --dataset cub --root ...` "
    "(scripts/stage_d_cub_check.sh) and compare with FX-A Table 7 (T = 23 at 15 dB)"
)
def test_reproduces_fx_a_table7_on_cub():
    pass
