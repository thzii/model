#!/usr/bin/env bash
# Stage D manual checks on a GPU machine: CUB download, SNR calibration, real-data
# tests and a short training dry run. Nothing here produces a reportable result.
#
# Usage:  CUB_ROOT=/data/cub DEVICE=cuda bash scripts/stage_d_cub_check.sh
set -euo pipefail

CUB_ROOT="${CUB_ROOT:-$HOME/data/cub}"
DEVICE="${DEVICE:-cuda}"
RUNS="${RUNS:-runs}"
cd "$(dirname "$0")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

echo "== 1. download CUB-200-2011 into $CUB_ROOT and report the GCD split"
python -m planora_sf.cli.download_cub --root "$CUB_ROOT" --check-files

echo "== 2. offline unit tests + real-data tests"
PLANORA_SF_ALLOW_DOWNLOAD=1 CUB_ROOT="$CUB_ROOT" python -m pytest

echo "== 3. SNR calibration (acceptance target: T = 23 at 15 dB, FX-A Table 7)"
python -m planora_sf.cli.calibrate_snr --dataset cub --root "$CUB_ROOT" --device "$DEVICE" \
  --output "$RUNS/snr/default.json"

echo "== 3b. U-09 variants, in case the default does not match"
for metric in box radial; do
  for norm in imagenet none; do
    python -m planora_sf.cli.calibrate_snr --dataset cub --root "$CUB_ROOT" --device "$DEVICE" \
      --metric "$metric" --normalize "$norm" --output "$RUNS/snr/${metric}_${norm}.json" | tail -1
  done
done

echo "== 4. training dry run: 1 epoch, 5 steps (memory/timing check only)"
python -m planora_sf.cli.train --dataset cub --root "$CUB_ROOT" --device "$DEVICE" \
  --epochs 1 --max-steps 5 --checkpoint-dir "$RUNS/dry_run"
