"""Import Stanford Cars from hand-downloaded archives (if given), verify it, report the GCD split.

There is no official download any more (see planora_sf.data.scars). Fetch the files yourself:
either the four original-release files, whose MD5s are checked against torchvision's pins,

    python -m planora_sf.cli.prepare_scars --root /data/scars --from-archive \\
        car_devkit.tgz cars_train.tgz cars_test.tgz cars_test_annos_withlabels.mat

or a zip/tar copy with any nesting (e.g. Kaggle) plus whatever it lacks (devkit, test labels).
Without --from-archive it only verifies ``<root>/stanford_cars``. Exit code 1 if the counts are
not 8,144 train / 196 classes (and 8,041 test when the test labels are present).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from planora_sf.config import DataConfig
from planora_sf.data.factory import build_split
from planora_sf.data.scars import (
    SCARS_NUM_CLASSES,
    SCARS_NUM_TEST,
    SCARS_NUM_TRAIN,
    StanfordCars,
    import_archives,
    write_manifest,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="directory that will contain stanford_cars/")
    parser.add_argument("--from-archive", nargs="+", type=Path, default=None, metavar="FILE",
                        help="tar/tgz/zip archives and loose .mat files to import")
    parser.add_argument("--check-files", action="store_true", help="also check that every image file exists")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    record = import_archives(args.root, args.from_archive) if args.from_archive else None
    train = StanfordCars(args.root, train=True, check_files=args.check_files)
    test = StanfordCars(args.root, train=False, check_files=args.check_files) if StanfordCars.has_test_labels(
        args.root) else None
    manifest = write_manifest(args.root, record)
    classes = int(len(set(train.targets.tolist())))
    # The SSB split needs every class present; an incomplete copy is reported without it.
    split = build_split("scars", train, DataConfig()).summary(train.targets) if classes == SCARS_NUM_CLASSES else None
    report = {
        "base": str(train.base),
        "train_images": len(train),
        "test_images": len(test) if test is not None else None,
        "classes": classes,
        "split": split,
        "manifest": str(manifest),
        "archives": record["archives"] if record else [],
    }
    print(json.dumps(report, indent=2))
    problems = []
    if (report["train_images"], report["classes"]) != (SCARS_NUM_TRAIN, SCARS_NUM_CLASSES):
        problems.append(f"expected train/classes = {(SCARS_NUM_TRAIN, SCARS_NUM_CLASSES)}, "
                        f"found {(report['train_images'], report['classes'])}")
    if test is None:
        print("NOTE: no labelled test annotations; the (unevaluated) test split will be skipped", file=sys.stderr)
    elif len(test) != SCARS_NUM_TEST:
        problems.append(f"expected {SCARS_NUM_TEST} test images, found {len(test)}")
    mismatched = [a["name"] for a in report["archives"] if a["matches_original_release"] is False]
    if mismatched:
        print(f"NOTE: MD5 differs from the original release for {mismatched} (fine for re-packed copies)",
              file=sys.stderr)
    for problem in problems:
        print(f"WARNING: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
