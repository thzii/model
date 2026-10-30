"""Helpers shared by the CLIs: strict JSON output, config diffs, the synthetic smoke config."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from planora_sf.config import BackboneConfig, FourierConfig, HeadConfig, LatentConfig, PlanoraSFConfig


def flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def config_diff(run: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, list]:
    """``{"dotted.key": [run_value, reference_value]}`` for every key whose values differ."""
    a, b = flatten(run), flatten(reference)
    return {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def json_safe(obj: Any) -> Any:
    """Plain-JSON copy of ``obj``: non-finite floats become None, numpy scalars become Python numbers."""
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def mean_std(values: Sequence[Optional[float]]) -> dict[str, Optional[float]]:
    """Mean and sample std (ddof=1) over the defined values; std is None for fewer than two."""
    vals = [v for v in values if v is not None and not np.isnan(v)]
    return {
        "mean": float(np.mean(vals)) if vals else None,
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smoke_config() -> PlanoraSFConfig:
    """Synthetic smoke-run config (``--dataset dummy``): tiny ViT, small latent space, no workers."""
    cfg = PlanoraSFConfig(
        backbone=BackboneConfig(name="tiny", grad_from_block=2, image_size=32),
        head=HeadConfig(hidden_dim=64, bottleneck_dim=16),
        latent=LatentConfig(dim=128),
        fourier=FourierConfig(cutoff=6),
    )
    cfg.loop.num_workers = 0
    return cfg
