"""Estimating the number of categories K (Vaze et al.; FX-P §5.2, spec §7 step 6)."""
from __future__ import annotations

import torch


def estimate_num_classes(features: torch.Tensor, labelled_mask: torch.Tensor, targets: torch.Tensor, max_k: int) -> int:
    # TODO(Stage C): port SelEx methods/estimate_k/estimate_k.py. Search K (Brent's
    # method by default, or binary search) to maximise clustering accuracy on the
    # labelled subset. FourEx reports K = 231 (CUB) and 230 (Stanford Cars).
    raise NotImplementedError("K estimation is not scaffolded yet (Stage C)")
