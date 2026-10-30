"""Training-time diagnostics (master plan §1.3, Task #2b).

This is the only training-side module that reads the ground truth of unlabelled samples.
Its functions take no gradient, draw no random numbers and return plain Python numbers,
so switching them on (``loop.diagnostics``) cannot change a training trajectory.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Optional

import torch

from planora_sf.data.gcd_dataset import GCDTrainDataset
from planora_sf.eval.diagnostics import cluster_structure, routing_matrix
from planora_sf.eval.metrics import split_cluster_acc_v1, split_cluster_acc_v2
from planora_sf.hyperbolic.poincare import dist0
from planora_sf.pseudo_labels.hssk import PseudoLabelState


def _finite(x: float) -> Optional[float]:
    return float(x) if math.isfinite(x) else None


def pseudo_label_quality(state: PseudoLabelState, dataset: GCDTrainDataset) -> dict:
    """PL-acc L0: v1/v2 of the level-0 BSSK pseudo-labels on the unlabelled samples, plus routing.

    These are the labels training actually receives. With ``selex.reproduce_label_order_bug``
    they are misaligned with the samples (D-08), and this measures exactly that signal.
    """
    level = state.levels[0]
    mask = (~dataset.is_labelled).numpy()
    y = dataset.targets.numpy()[mask]
    p = level.labels.cpu().numpy()[mask]
    num_known = dataset.split.num_known
    known = y < num_known
    v1, v2 = split_cluster_acc_v1(y, p, known), split_cluster_acc_v2(y, p, known)
    structure = cluster_structure(y, p, num_known, level.num_known_clusters)
    return {
        "v1": {k: _finite(v) for k, v in asdict(v1).items()},
        "v2": {k: _finite(v) for k, v in asdict(v2).items()},
        "routing": routing_matrix(y, p, num_known, level.num_known_clusters),
        "purity": structure["purity"],
        "unclaimed_novel_classes": structure["unclaimed_novel_classes"],
        "num_levels": state.num_levels,
    }


@torch.no_grad()
def hyperbolic_radius_stats(tangent: torch.Tensor, z: torch.Tensor, c: float, clip_r: float) -> dict[str, float]:
    """Radius monitors of a HypCD batch (master plan §1.3).

    ``tangent`` is the head output v before clipping, ``z`` the ball point. ``hyp_clip_fraction``
    is the share of rows with ‖v‖ > r; the p5/p50/p95 of ‖v‖ show how far the clip saturates.
    ``hyp_radius_*`` is the hyperbolic distance d_c(0, z), which equals 2·min(‖v‖, r) before the
    safety projection. A std near 0 with clip fraction near 1 means the radius carries no signal.
    """
    norm = tangent.float().norm(dim=1)
    q = torch.quantile(norm, torch.tensor([0.05, 0.5, 0.95], dtype=norm.dtype, device=norm.device))
    radius = dist0(z.float(), c)
    return {
        "hyp_clip_fraction": float((norm > clip_r).float().mean()),
        "hyp_tangent_norm": float(norm.mean()),
        "hyp_tangent_norm_p5": float(q[0]),
        "hyp_tangent_norm_p50": float(q[1]),
        "hyp_tangent_norm_p95": float(q[2]),
        "hyp_radius_mean": float(radius.mean()),
        "hyp_radius_std": float(radius.std(unbiased=False)),
    }
