"""Run provenance (master plan Task #2b): what produced a checkpoint, a summary or an evaluation.

Every new run records its seed, code commit, config hash and split hash, so that runs can be
paired by seed and refused when they do not share code or data (``cli.aggregate``). Nothing
here may fail a run: an unavailable git or CUDA is recorded as ``None``.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Optional

import torch

from planora_sf.config import PlanoraSFConfig
from planora_sf.data.splits import GCDSplit

_PACKAGE_DIR = Path(__file__).resolve().parents[1]


def canonical_sha256(obj: Any) -> str:
    """SHA-256 of the canonical JSON of ``obj`` (sorted keys, no whitespace)."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_sha256(cfg: PlanoraSFConfig) -> str:
    return canonical_sha256(cfg.to_dict())


def split_sha256(split: GCDSplit) -> str:
    """Hash of the class partition and the labelled/unlabelled index sets.

    Bounding-box cropping changes pixels only, so a cropped evaluation has the same hash.
    """
    return canonical_sha256({
        "known_classes": [int(c) for c in split.known_classes],
        "novel_classes": [int(c) for c in split.novel_classes],
        "labelled_indices": split.labelled_indices.tolist(),
        "unlabelled_indices": split.unlabelled_indices.tolist(),
    })


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(_PACKAGE_DIR), *args], capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state() -> dict[str, Any]:
    """``commit`` (HEAD sha) and ``commit_dirty`` (tracked files modified; untracked files ignored)."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no") if commit else None
    return {"commit": commit or None, "commit_dirty": None if status is None else bool(status)}


def environment(device: Optional[str] = None) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    name = None
    if cuda and device is not None and str(device).startswith("cuda"):
        name = torch.cuda.get_device_name(torch.device(device))
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),  # TorchVersion is not weights_only-loadable
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "device": str(device) if device is not None else None,
        "device_name": name,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def run_provenance(
    cfg: PlanoraSFConfig, split: GCDSplit, *, seed: Optional[int] = None, device: Optional[str] = None
) -> dict[str, Any]:
    """The provenance block of ``last.pt``, ``summary.json`` and ``cli.evaluate`` output."""
    from planora_sf.models.backbone import hub_source

    return {
        "seed": seed,
        **git_state(),
        "config_sha256": config_sha256(cfg),
        "split_sha256": split_sha256(split),
        "backbone": cfg.backbone.name,
        "hub": hub_source(cfg.backbone.name),
        **environment(device),
    }
