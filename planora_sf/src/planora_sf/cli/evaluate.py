"""Re-score a checkpoint, foreign backbone weights or a frozen backbone under controlled evaluator variants.

Plan T1/R0 and master plan Tasks #2a, #3. Features are extracted once; the transductive evaluator
then runs once per eval seed.

Re-score a training run on CUB, five eval seeds, checking it against the v2 numbers its log printed:
    python -m planora_sf.cli.evaluate --dataset cub --root /data/cub --device cuda \
        --checkpoint runs/x/last.pt --eval-seeds 0,1,2,3,4 --logged-v2 60.10,74.05,53.12 --out runs/r0/x.json

Offline pseudo-label accuracy (PL-acc L0: phase A's level-0 BSSK on train_view0 features), per seed:
    python -m planora_sf.cli.evaluate ... --checkpoint runs/x/last.pt --pseudo-labels

Foreign weights (official SelEx/DebGCD/HypCD code; only the ViT is used, heads are ignored):
    python -m planora_sf.cli.evaluate --dataset scars --foreign-backbone hypcd/model.pt --eval-seeds 0,1,2,3,4

Frozen-backbone floor (pretrained DINO, no training):
    python -m planora_sf.cli.evaluate --dataset cub --root /data/cub --device cuda --frozen-backbone

``--root`` defaults to ``$CUB_ROOT`` (CUB) or ``$SCARS_ROOT`` (Stanford Cars). The output is strict
JSON: NaN (v1/v2 of an empty subset) is written as ``null``. Accuracies are fractions in [0, 1];
the ``logged`` link check works in percentage points, as logs print them.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from planora_sf.cli.common import config_diff, file_sha256, flatten, json_safe, mean_std, smoke_config
from planora_sf.config import PlanoraSFConfig
from planora_sf.data.factory import build_gcd_datasets, build_sources, build_split, canonical_dataset, default_root
from planora_sf.data.splits import SSB_FILES, ssb_novel_tiers
from planora_sf.engine.diagnostics import pseudo_label_quality
from planora_sf.engine.provenance import git_state, split_sha256
from planora_sf.engine.train import refresh_from_dataset
from planora_sf.eval.evaluator import GCDEvaluator
from planora_sf.eval.protocol import extract_dataset_features
from planora_sf.models.backbone import load_foreign_backbone
from planora_sf.models.model import build_model

__all__ = ["config_diff", "flatten", "json_safe", "main", "summarise"]

_SUMMARY_KEYS = [(m, k) for m in ("v1", "v2") for k in ("all", "known", "novel")]
LINK_SLACK_PCT = 0.5  # master plan §1.1 step 4: |rescored - logged| <= 2 σ_eval + 0.5 points
PL_NOVEL_GAP_FLAG = 0.05  # master plan §1.3: PL-acc Novel more than 5 points below evaluator Novel


def summarise(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {f"{m}.{k}": mean_std([r[m][k] for r in per_seed]) for m, k in _SUMMARY_KEYS}
    for key in ("absorption", "leakage"):
        summary[f"routing.{key}"] = mean_std([r["routing"][key] for r in per_seed])
    return summary


def resolve_root(args: argparse.Namespace) -> Optional[str]:
    """``--root``, else ``$CUB_ROOT`` / ``$SCARS_ROOT`` for the (canonical) dataset."""
    return args.root or default_root(args.dataset)


def parse_seeds(text: str) -> list[int]:
    seeds = [int(s) for s in text.split(",") if s.strip()]
    if not seeds or any(s < 0 for s in seeds):
        raise argparse.ArgumentTypeError(f"--eval-seeds needs non-negative integers, got {text!r}")
    return seeds


def parse_logged(text: str) -> dict[str, float]:
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        values = []
    if len(values) != 3 or not all(0.0 <= v <= 100.0 for v in values):
        raise argparse.ArgumentTypeError(f"--logged-v2 needs All,Known,Novel in percent, got {text!r}")
    return dict(zip(("all", "known", "novel"), values))


def reference_config(dataset: str) -> PlanoraSFConfig:
    """Config for weights without one of their own (frozen or foreign backbones)."""
    return smoke_config() if dataset == "dummy" else PlanoraSFConfig.for_dataset(dataset)


def link_check(logged: Mapping[str, float], summary: Mapping[str, Any]) -> dict[str, Any]:
    """Is this checkpoint the run whose log printed ``logged``? (master plan §1.1 step 4, Gate 1)."""
    mean = {k: summary[f"v2.{k}"]["mean"] for k in ("all", "known", "novel")}
    std = summary["v2.all"]["std"]
    sigma_pct = 100.0 * std if std is not None else None
    tolerance = 2.0 * (sigma_pct or 0.0) + LINK_SLACK_PCT
    delta = None if mean["all"] is None else 100.0 * mean["all"] - logged["all"]
    return {
        "logged_v2_pct": dict(logged),
        "rescored_v2_pct": {k: None if v is None else 100.0 * v for k, v in mean.items()},
        "delta_all_pct": delta,
        "sigma_eval_pct": sigma_pct,
        "tolerance_pct": tolerance,
        "linked": None if delta is None else abs(delta) <= tolerance,
        "note": None if sigma_pct is not None else "one eval seed: the tolerance omits sigma_eval",
    }


def pseudo_label_report(model, datasets, cfg: PlanoraSFConfig, seeds: Sequence[int], *, device, batch_size: int,
                        num_workers: int, evaluator_summary: Mapping[str, Any]) -> dict[str, Any]:
    """PL-acc L0 offline (master plan Task #2a): phase A exactly as training runs it, once per seed.

    ``train_view0`` features come from the random train transform, so each seed also fixes the
    crops and flips (``torch.manual_seed``) as well as the k-means initialisation.
    """
    pl_cfg = copy.deepcopy(cfg)
    pl_cfg.loop.num_workers = num_workers
    source = datasets.train if cfg.loop.prototype_features == "train_view0" else datasets.train_eval
    per_seed = []
    for seed in seeds:
        torch.manual_seed(seed)
        state = refresh_from_dataset(
            model, source, pl_cfg, device=device, batch_size=batch_size, generator=torch.Generator().manual_seed(seed)
        )
        per_seed.append({"seed": seed, **pseudo_label_quality(state, datasets.train)})
        logging.info("pseudo-labels seed %d: v2 %s", seed, per_seed[-1]["v2"])
    summary = {f"v2.{k}": mean_std([r["v2"][k] for r in per_seed]) for k in ("all", "known", "novel")}
    summary["routing.absorption"] = mean_std([r["routing"]["absorption"] for r in per_seed])
    pl_novel, eval_novel = summary["v2.novel"]["mean"], evaluator_summary["v2.novel"]["mean"]
    gap = None if pl_novel is None or eval_novel is None else pl_novel - eval_novel
    return {
        "features": cfg.loop.prototype_features,
        "bssk": {"max_iter": cfg.selex.kmeans_max_iter, "n_init": cfg.selex.kmeans_n_init,
                 "balanced": cfg.selex.balanced, "reproduce_label_order_bug": cfg.selex.reproduce_label_order_bug},
        "per_seed": per_seed,
        "summary": summary,
        "novel_minus_evaluator_novel": gap,
        "flag_novel_below_evaluator": None if gap is None else gap < -PL_NOVEL_GAP_FLAG,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="checkpoint written by cli.train (last.pt)")
    source.add_argument("--frozen-backbone", action="store_true", help="pretrained DINO, no training (cub, scars)")
    source.add_argument("--foreign-backbone", type=Path,
                        help="ViT weights saved by other code (SelEx/DebGCD/HypCD); evaluated with the reference config")
    parser.add_argument("--unsafe-load", action="store_true",
                        help="--foreign-backbone only: allow arbitrary pickled objects (only for files you trust)")
    parser.add_argument("--dataset", choices=["cub", "scars", "stanford_cars", "dummy"], default="cub")
    parser.add_argument("--root", default=None, help="dataset root; default: $CUB_ROOT or $SCARS_ROOT")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=None, help="default: loop.num_workers")
    parser.add_argument("--batch-size", type=int, default=None, help="feature extraction; default: optim.batch_size")
    parser.add_argument("--clustering", choices=["bssk", "ss_kmeans"], default=None, help="default: eval.clustering")
    parser.add_argument("--n-init", type=int, default=None, help="default: eval.n_init")
    parser.add_argument("--max-iter", type=int, default=None, help="default: eval.max_iter")
    parser.add_argument("--eval-seeds", type=parse_seeds, default=[0], help="comma-separated, e.g. 0,1,2,3,4")
    parser.add_argument("--pseudo-labels", action="store_true", help="also report PL-acc L0 per eval seed (Task #2a)")
    parser.add_argument("--logged-v2", type=parse_logged, default=None, metavar="ALL,KNOWN,NOVEL",
                        help="v2 the training log printed, in percent: checks the log <-> checkpoint link")
    parser.add_argument("--out", type=Path, default=None, help="also write the JSON here")
    return parser


def _load_model(args, cfg: PlanoraSFConfig, ckpt: Optional[Mapping[str, Any]], split):
    """``(model, checkpoint record, stored provenance)`` for the chosen weight source."""
    if ckpt is not None:
        model = build_model(cfg, num_known_classes=split.num_known, pretrained=False, num_classes=split.num_classes)
        model.load_state_dict(ckpt["model"])
        record = {"path": str(args.checkpoint), "epoch": ckpt.get("epoch"), "sha256": file_sha256(args.checkpoint)}
        return model, record, ckpt.get("provenance")
    if args.foreign_backbone is not None:
        model = build_model(cfg, num_known_classes=split.num_known, pretrained=False)
        obj = torch.load(args.foreign_backbone, map_location="cpu", weights_only=not args.unsafe_load)
        report = load_foreign_backbone(model.backbone.vit, obj)
        record = {"path": str(args.foreign_backbone), "epoch": None, "sha256": file_sha256(args.foreign_backbone),
                  "foreign_mapping": report}
        return model, record, None
    model = build_model(cfg, num_known_classes=split.num_known)
    return model, {"path": None, "epoch": None, "sha256": None}, None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.dataset = canonical_dataset(args.dataset)
    args.root = resolve_root(args)
    if args.frozen_backbone and args.dataset not in ("cub", "scars"):
        parser.error("--frozen-backbone needs a benchmark dataset (cub or scars)")
    if args.unsafe_load and args.foreign_backbone is None:
        parser.error("--unsafe-load applies to --foreign-backbone only")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ckpt = None
    if args.checkpoint is not None:
        # weights_only=False is acceptable only because this repo wrote the checkpoint.
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = PlanoraSFConfig.from_dict(ckpt["config"])
        if cfg.dataset is not None and cfg.dataset != args.dataset:
            parser.error(f"checkpoint was configured for dataset {cfg.dataset!r} but --dataset is {args.dataset!r}")
    else:
        cfg = reference_config(args.dataset)

    # Rebuild data exactly as cli.train does.
    train_source, test_source = build_sources(args.dataset, args.root)
    split = build_split(args.dataset, train_source, cfg.data)
    model, record, stored = _load_model(args, cfg, ckpt, split)
    datasets = build_gcd_datasets(
        train_source, test_source, split, image_size=cfg.backbone.image_size, crop_pct=cfg.loop.crop_pct
    )
    model.to(args.device)

    overrides = {k: v for k, v in (("clustering", args.clustering), ("n_init", args.n_init), ("max_iter", args.max_iter))
                 if v is not None}
    eval_cfg = dataclasses.replace(cfg.eval, **overrides)
    batch_size = args.batch_size or cfg.optim.batch_size
    num_workers = cfg.loop.num_workers if args.num_workers is None else args.num_workers
    dataset = datasets.train_eval
    feats = extract_dataset_features(
        model.backbone, dataset, batch_size=batch_size, num_workers=num_workers, device=args.device
    )
    tiers = None
    if args.dataset in SSB_FILES and cfg.data.use_ssb_splits:
        to_target = split.class_to_target
        tiers = {name: [to_target[c] for c in ids] for name, ids in ssb_novel_tiers(args.dataset).items()}

    evaluator = GCDEvaluator(split.num_known, split.num_classes, eval_cfg)
    per_seed = []
    for seed in args.eval_seeds:
        prediction = evaluator.predict(
            feats, dataset.is_labelled, dataset.train_targets, generator=torch.Generator().manual_seed(seed)
        )
        per_seed.append({"seed": seed, **evaluator.score_detailed(prediction, dataset.targets, ~dataset.is_labelled, tiers)})
        logging.info("eval seed %d: v2 %s", seed, per_seed[-1]["v2"])
    summary = summarise(per_seed)

    split_hash = split_sha256(split)
    result = {
        "checkpoint": {**record, "frozen_backbone": bool(args.frozen_backbone),
                       "foreign_backbone": args.foreign_backbone is not None},
        "dataset": args.dataset,
        "backbone": cfg.backbone.name,
        "evaluator": {**dataclasses.asdict(eval_cfg), "eval_seeds": list(args.eval_seeds)},
        "config": cfg.to_dict(),
        "config_diff_vs_cub_reference": config_diff(cfg.to_dict(), PlanoraSFConfig.for_dataset("cub").to_dict()),
        "split": split.summary(train_source.targets),
        "split_sha256": split_hash,
        "provenance": stored,
        "split_matches_training": None if not stored else stored.get("split_sha256") == split_hash,
        "code": git_state(),
        "per_seed": per_seed,
        "summary": summary,
        "logged": link_check(args.logged_v2, summary) if args.logged_v2 is not None else None,
        "pseudo_labels": pseudo_label_report(
            model, datasets, cfg, args.eval_seeds, device=args.device, batch_size=batch_size,
            num_workers=num_workers, evaluator_summary=summary,
        ) if args.pseudo_labels else None,
    }
    text = json.dumps(json_safe(result), indent=2, allow_nan=False)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
