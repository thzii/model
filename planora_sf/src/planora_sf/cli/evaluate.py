"""Re-score a checkpoint (or the frozen DINOv1 backbone) under controlled evaluator variants (plan T1, R0).

Features are extracted once; the transductive evaluator then runs once per eval seed.

Re-score a training run on CUB, five eval seeds:
    python -m planora_sf.cli.evaluate --dataset cub --root /data/cub --device cuda \
        --checkpoint runs/x/last.pt --eval-seeds 0,1,2,3,4 --out runs/r0/bssk_10x10.json

Frozen-backbone floor (pretrained DINOv1, no training):
    python -m planora_sf.cli.evaluate --dataset cub --root /data/cub --device cuda \
        --frozen-backbone --eval-seeds 0,1,2,3,4

``--root`` defaults to ``$CUB_ROOT``. The output is strict JSON: NaN (v1/v2 of an empty
subset) is written as ``null``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from planora_sf.config import PlanoraSFConfig
from planora_sf.data.factory import build_gcd_datasets, build_sources, build_split
from planora_sf.data.splits import cub_ssb_novel_tiers
from planora_sf.eval.evaluator import GCDEvaluator
from planora_sf.eval.protocol import extract_dataset_features
from planora_sf.models.model import build_model

_SUMMARY_KEYS = [(m, k) for m in ("v1", "v2") for k in ("all", "known", "novel")]


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


def _mean_std(values: Sequence[Optional[float]]) -> dict[str, Optional[float]]:
    """Mean and sample std (ddof=1) over the defined values; std is None for fewer than two."""
    vals = [v for v in values if v is not None and not np.isnan(v)]
    return {
        "mean": float(np.mean(vals)) if vals else None,
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
    }


def summarise(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {f"{m}.{k}": _mean_std([r[m][k] for r in per_seed]) for m, k in _SUMMARY_KEYS}
    for key in ("absorption", "leakage"):
        summary[f"routing.{key}"] = _mean_std([r["routing"][key] for r in per_seed])
    return summary


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


def parse_seeds(text: str) -> list[int]:
    seeds = [int(s) for s in text.split(",") if s.strip()]
    if not seeds or any(s < 0 for s in seeds):
        raise argparse.ArgumentTypeError(f"--eval-seeds needs non-negative integers, got {text!r}")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="checkpoint written by cli.train (last.pt)")
    source.add_argument("--frozen-backbone", action="store_true", help="pretrained DINOv1, no training (cub only)")
    parser.add_argument("--dataset", choices=["cub", "dummy"], default="cub")
    parser.add_argument("--root", default=os.environ.get("CUB_ROOT"), help="dataset root; default: $CUB_ROOT")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=None, help="default: loop.num_workers")
    parser.add_argument("--batch-size", type=int, default=None, help="feature extraction; default: optim.batch_size")
    parser.add_argument("--clustering", choices=["bssk", "ss_kmeans"], default=None, help="default: eval.clustering")
    parser.add_argument("--n-init", type=int, default=None, help="default: eval.n_init")
    parser.add_argument("--max-iter", type=int, default=None, help="default: eval.max_iter")
    parser.add_argument("--eval-seeds", type=parse_seeds, default=[0], help="comma-separated, e.g. 0,1,2,3,4")
    parser.add_argument("--out", type=Path, default=None, help="also write the JSON here")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frozen_backbone and args.dataset != "cub":
        parser.error("--frozen-backbone is only supported with --dataset cub")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ckpt = None
    if args.checkpoint is not None:
        # weights_only=False is acceptable only because this repo wrote the checkpoint.
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = PlanoraSFConfig.from_dict(ckpt["config"])
        if cfg.dataset is not None and cfg.dataset != args.dataset:
            parser.error(f"checkpoint was configured for dataset {cfg.dataset!r} but --dataset is {args.dataset!r}")
    else:
        cfg = PlanoraSFConfig.for_dataset("cub")

    # Rebuild data exactly as cli.train does.
    train_source, test_source = build_sources(args.dataset, args.root)
    split = build_split(args.dataset, train_source, cfg.data)
    datasets = build_gcd_datasets(
        train_source, test_source, split, image_size=cfg.backbone.image_size, crop_pct=cfg.loop.crop_pct
    )
    if ckpt is not None:
        model = build_model(cfg, num_known_classes=split.num_known, pretrained=False, num_classes=split.num_classes)
        model.load_state_dict(ckpt["model"])
    else:
        model = build_model(cfg, num_known_classes=split.num_known)
    model.to(args.device)

    overrides = {k: v for k, v in (("clustering", args.clustering), ("n_init", args.n_init), ("max_iter", args.max_iter))
                 if v is not None}
    eval_cfg = dataclasses.replace(cfg.eval, **overrides)
    dataset = datasets.train_eval
    feats = extract_dataset_features(
        model.backbone, dataset,
        batch_size=args.batch_size or cfg.optim.batch_size,
        num_workers=cfg.loop.num_workers if args.num_workers is None else args.num_workers,
        device=args.device,
    )
    tiers = None
    if args.dataset == "cub" and cfg.data.use_ssb_splits:
        to_target = split.class_to_target
        tiers = {name: [to_target[c] for c in ids] for name, ids in cub_ssb_novel_tiers().items()}

    evaluator = GCDEvaluator(split.num_known, split.num_classes, eval_cfg)
    per_seed = []
    for seed in args.eval_seeds:
        prediction = evaluator.predict(
            feats, dataset.is_labelled, dataset.train_targets, generator=torch.Generator().manual_seed(seed)
        )
        per_seed.append({"seed": seed, **evaluator.score_detailed(prediction, dataset.targets, ~dataset.is_labelled, tiers)})
        logging.info("eval seed %d: v2 %s", seed, per_seed[-1]["v2"])

    result = {
        "checkpoint": {
            "path": str(args.checkpoint) if ckpt is not None else None,
            "epoch": ckpt.get("epoch") if ckpt is not None else None,
            "frozen_backbone": ckpt is None,
        },
        "dataset": args.dataset,
        "evaluator": {**dataclasses.asdict(eval_cfg), "eval_seeds": list(args.eval_seeds)},
        "config": cfg.to_dict(),
        "config_diff_vs_cub_reference": config_diff(cfg.to_dict(), PlanoraSFConfig.for_dataset("cub").to_dict()),
        "split": split.summary(train_source.targets),
        "per_seed": per_seed,
        "summary": summarise(per_seed),
    }
    text = json.dumps(json_safe(result), indent=2, allow_nan=False)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
