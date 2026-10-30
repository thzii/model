"""Paired comparison of two cli.evaluate outputs of the same weights (master plan Task #3, Gate 3).

Typical use: the same checkpoint scored on full images (BASE) and on bounding-box crops (OTHER):

    python -m planora_sf.cli.compare_evals runs/s3/e0_full.json runs/s3/e0_crop.json --out runs/s3/e0_cmp.json

Deltas are OTHER - BASE in percentage points, paired by eval seed. The two files must describe
the same weights (checkpoint SHA-256, or the same frozen backbone), the same dataset and split
hash, and the same clustering settings; anything else is refused. For each metric:

    Δ_s = OTHER_s - BASE_s,   σ_Δ = sd(Δ_s) (ddof 1),   threshold = max(1.0, 2 σ_Δ)
    supported increase:  mean Δ >= threshold and every Δ_s > 0   (Gate 3 criterion on v2 All)
    supported decrease:  mean Δ <= -threshold and every Δ_s < 0
    otherwise "not detected at n = <seeds>"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from planora_sf.cli.common import json_safe

METRICS = [(m, k) for m in ("v2", "v1") for k in ("all", "known", "novel")]
MIN_DELTA_PCT = 1.0  # master plan §1.2: δ = max(1.0, 2σ)


class IncomparableError(ValueError):
    """The two evaluations do not describe the same weights, data or evaluator."""


def weights_identity(result: Mapping[str, Any]) -> str:
    ckpt = result["checkpoint"]
    if ckpt.get("frozen_backbone"):
        return f"frozen:{result.get('backbone')}"
    return f"sha256:{ckpt['sha256']}" if ckpt.get("sha256") else f"path:{ckpt.get('path')}"


def _check_comparable(base: Mapping[str, Any], other: Mapping[str, Any]) -> None:
    problems = []
    if weights_identity(base) != weights_identity(other):
        problems.append(f"different weights: {weights_identity(base)} vs {weights_identity(other)}")
    for key in ("dataset", "split_sha256"):
        if base.get(key) != other.get(key):
            problems.append(f"different {key}: {base.get(key)} vs {other.get(key)}")
    for key in ("clustering", "n_init", "max_iter"):
        if base["evaluator"][key] != other["evaluator"][key]:
            problems.append(f"different evaluator.{key}: {base['evaluator'][key]} vs {other['evaluator'][key]}")
    if problems:
        raise IncomparableError("; ".join(problems))


def paired_stats(deltas: Sequence[float]) -> dict[str, Any]:
    """Mean, sd, SE and the supported/not-detected verdict of paired deltas (percentage points)."""
    d = np.asarray([x for x in deltas if x is not None and math.isfinite(x)], dtype=np.float64)
    n = int(d.size)
    if n == 0:
        return {"n": 0, "deltas": [], "mean": None, "sd": None, "se": None, "threshold": None, "verdict": "undefined"}
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else None
    threshold = max(MIN_DELTA_PCT, 2.0 * sd) if sd is not None else MIN_DELTA_PCT
    if n > 1 and mean >= threshold and (d > 0).all():
        verdict = "supported increase"
    elif n > 1 and mean <= -threshold and (d < 0).all():
        verdict = "supported decrease"
    else:
        verdict = f"not detected at n = {n}"
    return {
        "n": n, "deltas": d.tolist(), "mean": mean, "sd": sd, "se": None if sd is None else sd / math.sqrt(n),
        "threshold": threshold, "all_positive": bool((d > 0).all()), "all_negative": bool((d < 0).all()),
        "verdict": verdict,
    }


def _pct(x: Optional[float]) -> Optional[float]:
    return None if x is None else 100.0 * x


def compare(base: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """OTHER - BASE paired by eval seed; raises ``IncomparableError`` for unrelated evaluations."""
    _check_comparable(base, other)
    a = {r["seed"]: r for r in base["per_seed"]}
    b = {r["seed"]: r for r in other["per_seed"]}
    seeds = sorted(set(a) & set(b))
    if not seeds:
        raise IncomparableError("no eval seed in common")
    metrics = {}
    for m, k in METRICS:
        deltas = [None if a[s][m][k] is None or b[s][m][k] is None else _pct(b[s][m][k] - a[s][m][k]) for s in seeds]
        metrics[f"{m}.{k}"] = paired_stats(deltas)
    for key in ("absorption", "leakage"):
        deltas = [None if a[s]["routing"][key] is None or b[s]["routing"][key] is None
                  else _pct(b[s]["routing"][key] - a[s]["routing"][key]) for s in seeds]
        metrics[f"routing.{key}"] = paired_stats(deltas)
    tiers = {}
    if all(a[s].get("tiers") and b[s].get("tiers") for s in seeds):
        for tier in a[seeds[0]]["tiers"]:
            deltas = [None if a[s]["tiers"][tier]["sample_weighted"] is None
                      or b[s]["tiers"][tier]["sample_weighted"] is None
                      else _pct(b[s]["tiers"][tier]["sample_weighted"] - a[s]["tiers"][tier]["sample_weighted"])
                      for s in seeds]
            tiers[tier] = paired_stats(deltas)
    return {
        "weights": weights_identity(base),
        "dataset": base["dataset"],
        "split_sha256": base.get("split_sha256"),
        "seeds": seeds,
        "base": {"crop": base.get("crop"), "summary_v2_all_pct": _pct(base["summary"]["v2.all"]["mean"])},
        "other": {"crop": other.get("crop"), "summary_v2_all_pct": _pct(other["summary"]["v2.all"]["mean"])},
        "metrics": metrics,
        "novel_tiers_sample_weighted": tiers or None,
        "gate3_supported_increase": metrics["v2.all"]["verdict"] == "supported increase",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", type=Path, help="cli.evaluate JSON, e.g. full images")
    parser.add_argument("other", type=Path, help="cli.evaluate JSON of the same weights, e.g. --crop-bbox")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    base = json.loads(args.base.read_text(encoding="utf-8"))
    other = json.loads(args.other.read_text(encoding="utf-8"))
    try:
        result = compare(base, other)
    except IncomparableError as err:
        print(f"refusing to compare: {err}", file=sys.stderr)
        return 2
    result = {"base_file": str(args.base), "other_file": str(args.other), **result}
    text = json.dumps(json_safe(result), indent=2, allow_nan=False)
    print(text)
    for name in ("v2.all", "v2.known", "v2.novel"):
        r = result["metrics"][name]
        if r["mean"] is None:
            continue
        print(f"{name}: mean delta {r['mean']:+.2f} pt (sd {r['sd'] if r['sd'] is None else round(r['sd'], 2)}, "
              f"threshold {r['threshold']:.2f}) -> {r['verdict']}", file=sys.stderr)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
