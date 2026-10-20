"""Calibrate the Fourier cut-off T from per-class SNR curves (spec §4.3, FX-P §4).

    python -m planora_sf.cli.calibrate_snr --dataset cub --root /data/cub --device cuda

Prints the cut-off at 5/10/15/20/25 dB next to FX-A Table 7 when a reference
exists (CUB: 2, 8, 23, 45, 55). It then says whether the 15 dB value matches.
The measurement details are unresolved in FourEx (U-09), so try
--metric/--normalize variants if it does not match. By default only labelled
known-class images are used (U-10).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from planora_sf.config import CUTOFF_TABLE, DataConfig
from planora_sf.data.factory import build_sources, build_split
from planora_sf.data.transforms import build_transforms
from planora_sf.fourier.snr import calibrate_from_batches

LEVELS_DB = (5.0, 10.0, 15.0, 20.0, 25.0)


class _Images(Dataset):
    def __init__(self, source, indices, transform):
        self.source, self.indices, self.transform = source, np.asarray(indices), transform

    def __len__(self):
        return int(self.indices.size)

    def __getitem__(self, i):
        index = int(self.indices[i])
        return self.transform(self.source.load(index)), int(self.source.targets[index])


def select_indices(targets: np.ndarray, split, which: str) -> np.ndarray:
    if which == "labelled":
        return split.labelled_indices
    if which == "known":
        return np.nonzero(np.isin(targets, split.known_classes))[0]
    if which == "all":
        return np.arange(targets.size)  # uses novel-class labels: a benchmark diagnostic only (U-10)
    raise ValueError(which)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["cub", "dummy"], default="cub")
    parser.add_argument("--root", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--classes", choices=["labelled", "known", "all"], default="labelled")
    parser.add_argument("--normalize", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--metric", choices=["box", "radial"], default="box")
    parser.add_argument("--image-size", type=int, default=None, help="default: 224 (cub) or 32 (dummy)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None, help="write results as JSON")
    args = parser.parse_args(argv)

    image_size = args.image_size or (224 if args.dataset == "cub" else 32)
    train_source, _ = build_sources(args.dataset, args.root, download=args.download)
    split = build_split(args.dataset, train_source, DataConfig())
    _, test_tf = build_transforms(image_size, normalize=args.normalize == "imagenet")
    images = _Images(train_source, select_indices(train_source.targets, split, args.classes), test_tf)
    loader = DataLoader(images, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    batches = ((x.to(args.device), y) for x, y in loader)
    results = calibrate_from_batches(batches, targets_db=LEVELS_DB, metric=args.metric)

    reference = CUTOFF_TABLE.get(args.dataset)
    print(f"dataset={args.dataset} images={len(images)} classes={args.classes} "
          f"metric={args.metric} normalize={args.normalize} size={image_size}")
    print(f"{'SNR (dB)':>9} {'T':>5} {'FX-A Table 7':>13}")
    for db in LEVELS_DB:
        ref = reference.get(int(db)) if reference else None
        print(f"{db:>9.0f} {results[db].cutoff:>5d} {ref if ref is not None else '-':>13}")
    if reference:
        verdict = "MATCH" if results[15.0].cutoff == reference[15] else "MISMATCH"
        print(f"15 dB acceptance check: T={results[15.0].cutoff} vs {reference[15]} -> {verdict}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "cutoffs": {str(int(db)): r.cutoff for db, r in results.items()},
            "per_class_15db": results[15.0].per_class,
            "reference": reference,
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    sys.exit(main())
