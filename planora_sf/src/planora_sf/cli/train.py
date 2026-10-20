"""Run the PLANORA-SF training loop (single device).

GPU dry run on CUB (a few steps, to check memory and timing; not a result):
    python -m planora_sf.cli.train --dataset cub --root /data/cub --device cuda \
        --epochs 1 --max-steps 5 --checkpoint-dir runs/dry

Offline smoke run on synthetic data with the tiny ViT:
    python -m planora_sf.cli.train --dataset dummy --epochs 1 --max-steps 2 --batch-size 8
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

from planora_sf.config import BackboneConfig, FourierConfig, HeadConfig, LatentConfig, PlanoraSFConfig
from planora_sf.data.factory import build_gcd_datasets, build_sources, build_split
from planora_sf.engine.train import fit
from planora_sf.models.model import build_model


def build_config(args: argparse.Namespace) -> PlanoraSFConfig:
    if args.config:
        cfg = PlanoraSFConfig.from_yaml(args.config)
    elif args.dataset == "cub":
        cfg = PlanoraSFConfig.for_dataset("cub")
    else:  # synthetic smoke run: tiny ViT, small latent space
        cfg = PlanoraSFConfig(
            backbone=BackboneConfig(name="tiny", grad_from_block=2, image_size=32),
            head=HeadConfig(hidden_dim=64, bottleneck_dim=16),
            latent=LatentConfig(dim=128),
            fourier=FourierConfig(cutoff=6),
        )
        cfg.loop.num_workers = 0
    if args.cutoff is not None:
        cfg.fourier.cutoff = args.cutoff
    if args.batch_size is not None:
        cfg.optim.batch_size = args.batch_size
    if args.max_steps is not None:
        cfg.loop.max_steps_per_epoch = args.max_steps
    if args.num_workers is not None:
        cfg.loop.num_workers = args.num_workers
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["cub", "dummy"], default="cub")
    parser.add_argument("--root", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--config", type=Path, default=None, help="YAML config (default: reference config)")
    parser.add_argument("--epochs", type=int, default=None, help="default: optim.epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="cap steps per epoch (dry runs)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cutoff", type=int, default=None, help="Fourier cut-off T (default: Table 7 / config)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    torch.manual_seed(args.seed)
    cfg = build_config(args)
    train_source, test_source = build_sources(args.dataset, args.root, download=args.download)
    split = build_split(args.dataset, train_source, cfg.data)
    datasets = build_gcd_datasets(
        train_source, test_source, split, image_size=cfg.backbone.image_size, crop_pct=cfg.loop.crop_pct
    )
    model = build_model(cfg, num_known_classes=split.num_known)
    result = fit(
        model, datasets, cfg, device=args.device, epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir, generator=torch.Generator().manual_seed(args.seed),
    )
    summary = {
        "split": split.summary(train_source.targets),
        "history": [asdict(h) for h in result.history],
        "evaluations": [{"epoch": e, **{k: asdict(v) for k, v in s.items()}} for e, s in result.evaluations],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
