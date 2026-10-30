"""Run the PLANORA-SF training loop (single device).

GPU dry run on CUB (a few steps, to check memory and timing; not a result):
    python -m planora_sf.cli.train --dataset cub --root /data/cub --device cuda \
        --epochs 1 --max-steps 5 --checkpoint-dir runs/dry

Offline smoke run on synthetic data with the tiny ViT:
    python -m planora_sf.cli.train --dataset dummy --epochs 1 --max-steps 2 --batch-size 8

Experiment rows (plan §4.2) on top of the chosen config:
    E0 SelEx           --no-use-fourex
    E1 SelEx+FourEx    --use-fourex            (the reference config already is E1)
    E2 SelEx+DebGCD    --no-use-fourex --use-debgcd
    E3 all three       --use-fourex --use-debgcd
    E4 Hyp-SelEx       --no-use-fourex --use-hypcd
DINOv2 instead of DINOv1: --backbone dinov2_vitb14_reg (or configs/dinov2/*.yaml).
Without a flag, a component stays as the config sets it.

With ``--checkpoint-dir`` the run writes ``last.pt`` and ``summary.json`` there, both with a
provenance block (seed, commit, config/split SHA-256; master plan Task #2b). ``--label`` names
the run for ``cli.aggregate`` (e.g. S2-E0); it defaults to the experiment row.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import torch

from planora_sf.cli.common import json_safe, smoke_config
from planora_sf.config import PlanoraSFConfig
from planora_sf.data.factory import build_gcd_datasets, build_sources, build_split, canonical_dataset, default_root
from planora_sf.engine.provenance import run_provenance
from planora_sf.engine.train import fit
from planora_sf.models.model import build_model


def build_config(args: argparse.Namespace) -> PlanoraSFConfig:
    if args.config:
        cfg = PlanoraSFConfig.from_yaml(args.config)
    elif args.dataset in ("cub", "scars"):
        cfg = PlanoraSFConfig.for_dataset(args.dataset)
    else:  # synthetic smoke run: tiny ViT, small latent space
        cfg = smoke_config()
    if args.backbone is not None:
        cfg.backbone.name = args.backbone
    if args.cutoff is not None:
        cfg.fourier.cutoff = args.cutoff
    if args.batch_size is not None:
        cfg.optim.batch_size = args.batch_size
    if args.max_steps is not None:
        cfg.loop.max_steps_per_epoch = args.max_steps
    if args.num_workers is not None:
        cfg.loop.num_workers = args.num_workers
    if args.diagnostics is not None:
        cfg.loop.diagnostics = args.diagnostics
    if args.keep_checkpoint_every is not None:
        cfg.loop.keep_checkpoint_every = args.keep_checkpoint_every
    return cfg.set_components(fourex=args.use_fourex, debgcd=args.use_debgcd, hypcd=args.use_hypcd)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["cub", "scars", "stanford_cars", "dummy"], default="cub")
    parser.add_argument("--root", default=None, help="dataset root; default: $CUB_ROOT or $SCARS_ROOT")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--config", type=Path, default=None, help="YAML config (default: reference config)")
    parser.add_argument("--epochs", type=int, default=None, help="default: optim.epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="cap steps per epoch (dry runs)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cutoff", type=int, default=None, help="Fourier cut-off T (default: Table 7 / config)")
    parser.add_argument("--backbone", choices=["dinov1_vitb16", "dinov2_vitb14_reg"], default=None,
                        help="default: backbone.name of the config (DINOv1 for the reference configs)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--label", default=None, help="run name for cli.aggregate (default: the experiment row)")
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=None,
                        help="per-epoch PL-acc L0 and routing (default: loop.diagnostics)")
    parser.add_argument("--keep-checkpoint-every", type=int, default=None,
                        help="also keep epoch_XXX.pt every n epochs (default: loop.keep_checkpoint_every)")
    parser.add_argument("--use-fourex", "--use_four_ex", dest="use_fourex", action=argparse.BooleanOptionalAction,
                        default=None, help="FourEx losses and L_cls head on/off (off = E0/E2)")
    parser.add_argument("--use-debgcd", "--use_debgcd", dest="use_debgcd", action=argparse.BooleanOptionalAction,
                        default=None, help="DebGCD detector + debiased classifier on/off (E2/E3)")
    parser.add_argument("--use-hypcd", "--use_hypcd", dest="use_hypcd", action=argparse.BooleanOptionalAction,
                        default=None, help="HypCD Poincaré-ball projection space on/off (E4)")
    args = parser.parse_args(argv)
    args.dataset = canonical_dataset(args.dataset)
    args.root = args.root or default_root(args.dataset)
    if args.backbone is not None and args.dataset == "dummy" and args.config is None:
        parser.error("--backbone needs a benchmark dataset or --config (the dummy smoke run uses the tiny ViT)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    torch.manual_seed(args.seed)
    cfg = build_config(args)
    train_source, test_source = build_sources(args.dataset, args.root, download=args.download)
    split = build_split(args.dataset, train_source, cfg.data)
    datasets = build_gcd_datasets(
        train_source, test_source, split, image_size=cfg.backbone.image_size, crop_pct=cfg.loop.crop_pct
    )
    model = build_model(cfg, num_known_classes=split.num_known, num_classes=split.num_classes)
    provenance = run_provenance(cfg, split, seed=args.seed, device=args.device)
    result = fit(
        model, datasets, cfg, device=args.device, epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir, generator=torch.Generator().manual_seed(args.seed), provenance=provenance,
    )
    components = cfg.components()
    evaluations = [{"epoch": e, **{k: asdict(v) for k, v in s.items()}} for e, s in result.evaluations]
    summary = {
        "label": args.label or components["row"] or "custom",
        "dataset": args.dataset,
        "backbone": cfg.backbone.name,
        "components": components,
        "provenance": provenance,
        "epochs_run": len(result.history),
        "split": split.summary(train_source.targets),
        "final": evaluations[-1] if evaluations else None,
        "history": [asdict(h) for h in result.history],
        "evaluations": evaluations,
    }
    text = json.dumps(json_safe(summary), indent=2, allow_nan=False)
    print(text)
    if args.checkpoint_dir is not None:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (args.checkpoint_dir / "summary.json").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
