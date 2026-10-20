"""Download CUB-200-2011 (if needed) and report the GCD split counts.

    python -m planora_sf.cli.download_cub --root /data/cub
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from planora_sf.config import DataConfig
from planora_sf.data.factory import build_split
from planora_sf.data.sources import CUB200, CUB_NUM_CLASSES, CUB_NUM_TEST, CUB_NUM_TRAIN


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="directory that will contain CUB_200_2011/")
    parser.add_argument("--no-download", action="store_true", help="only verify an existing copy")
    parser.add_argument("--check-files", action="store_true", help="also check that every image file exists")
    args = parser.parse_args(argv)

    train = CUB200(args.root, train=True, download=not args.no_download, check_files=args.check_files)
    test = CUB200(args.root, train=False, check_files=args.check_files)
    split = build_split("cub", train, DataConfig())
    report = {
        "train_images": len(train),
        "test_images": len(test),
        "classes": int(len(set(train.targets.tolist()) | set(test.targets.tolist()))),
        "split": split.summary(train.targets),
    }
    print(json.dumps(report, indent=2))
    expected = (CUB_NUM_TRAIN, CUB_NUM_TEST, CUB_NUM_CLASSES)
    found = (report["train_images"], report["test_images"], report["classes"])
    if found != expected:
        print(f"WARNING: expected train/test/classes = {expected}, found {found}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
