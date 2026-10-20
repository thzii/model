"""Fourier view generator: blur -> FFT2 -> box mask -> iFFT2 (spec §4.2, FX-P §3-5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from planora_sf.config import FourierConfig
from planora_sf.fourier.filters import highpass_mask, lowpass_mask


def gaussian_kernel1d(kernel_size: int, sigma: float, *, dtype=torch.float32, device=None) -> torch.Tensor:
    half = (kernel_size - 1) / 2
    xs = torch.arange(kernel_size, dtype=dtype, device=device) - half
    kernel = torch.exp(-0.5 * (xs / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_blur(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.1) -> torch.Tensor:
    """Depthwise Gaussian blur with reflect padding, applied before filtering to limit
    Gibbs ringing (FX-P §5.1; sigma is U-03)."""
    if x.ndim != 4:
        raise ValueError(f"expected [B, C, H, W], got shape {tuple(x.shape)}")
    if kernel_size == 1:
        return x
    channels = x.shape[1]
    k1 = gaussian_kernel1d(kernel_size, sigma, dtype=x.dtype, device=x.device)
    kernel = torch.outer(k1, k1).expand(channels, 1, kernel_size, kernel_size)
    pad = kernel_size // 2
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel, groups=channels)


class CutoffSampler:
    """Random cut-offs f_l in {low_min .. T_l} and f_h in {high_min .. T_h - 1} (FX-P §4, U-02)."""

    def __init__(self, t_low: int, t_high: int, *, low_min: int = 1, high_min: int = 0, per_image: bool = True):
        if not low_min <= t_low:
            raise ValueError(f"need low_min <= T_l, got {low_min} > {t_low}")
        if not high_min < t_high:
            raise ValueError(f"need high_min < T_h, got {high_min} >= {t_high}")
        self.t_low, self.t_high = t_low, t_high
        self.low_min, self.high_min = low_min, high_min
        self.per_image = per_image

    def sample(
        self, batch_size: int, *, generator: Optional[torch.Generator] = None, device=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = batch_size if self.per_image else 1
        f_low = torch.randint(self.low_min, self.t_low + 1, (n,), generator=generator)
        f_high = torch.randint(self.high_min, self.t_high, (n,), generator=generator)
        if not self.per_image:
            f_low, f_high = f_low.expand(batch_size).clone(), f_high.expand(batch_size).clone()
        return f_low.to(device), f_high.to(device)


@dataclass
class FourierViews:
    anchor: torch.Tensor  # view the filtered images are paired with
    low: torch.Tensor  # l = F^-1(H_LP ⊙ F(x))
    high: torch.Tensor  # h = F^-1(H_HP ⊙ F(x))
    f_low: torch.Tensor  # [B] cut-offs used for l
    f_high: torch.Tensor  # [B] cut-offs used for h


def _per_image(f: Union[int, torch.Tensor], batch_size: int, device) -> torch.Tensor:
    f = torch.as_tensor(f, device=device).reshape(-1)
    if f.numel() == 1:
        f = f.expand(batch_size).clone()
    if f.numel() != batch_size:
        raise ValueError(f"expected {batch_size} cut-offs, got {f.numel()}")
    return f


class FourierViewGenerator(nn.Module):
    """Builds the low- and high-frequency views of a batch of images (spec §4.2).

    Works on whatever tensor it receives; feeding the ImageNet-normalised
    tensor is the U-04 scaffold default.
    """

    def __init__(self, cfg: FourierConfig):
        super().__init__()
        self.cfg = cfg
        self.sampler = CutoffSampler(
            cfg.t_low(),
            cfg.t_high(),
            low_min=cfg.low_cutoff_min,
            high_min=cfg.high_cutoff_min,
            per_image=cfg.per_image_cutoffs,
        )

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        f_low: Optional[Union[int, torch.Tensor]] = None,
        f_high: Optional[Union[int, torch.Tensor]] = None,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> FourierViews:
        if x.ndim != 4:
            raise ValueError(f"expected [B, C, H, W], got shape {tuple(x.shape)}")
        batch, _, height, width = x.shape
        if f_low is None or f_high is None:
            sampled_low, sampled_high = self.sampler.sample(batch, generator=generator, device=x.device)
            f_low = sampled_low if f_low is None else f_low
            f_high = sampled_high if f_high is None else f_high
        f_low = _per_image(f_low, batch, x.device)
        f_high = _per_image(f_high, batch, x.device)

        blurred = gaussian_blur(x, self.cfg.blur_kernel_size, self.cfg.blur_sigma)
        spectrum = torch.fft.fft2(blurred)
        strict = self.cfg.strict_inequality
        lp = lowpass_mask(height, width, f_low, strict=strict, device=x.device)
        band_upper = self.cfg.t_high() if self.cfg.band_pass else None
        hp = highpass_mask(height, width, f_high, strict=strict, band_upper=band_upper, device=x.device)
        low = torch.fft.ifft2(spectrum * lp).real.to(x.dtype)
        high = torch.fft.ifft2(spectrum * hp).real.to(x.dtype)
        anchor = blurred if self.cfg.blur_anchor else x
        return FourierViews(anchor=anchor, low=low, high=high, f_low=f_low, f_high=f_high)
