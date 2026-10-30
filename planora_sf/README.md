# PLANORA-SF research package

This is an isolated research package for fine-grained generalized category discovery (GCD). It builds on SelEx and adds three optional components:

- **FourEx:** Fourier self-supervision.
- **DebGCD:** an adapted port.
- **HypCD:** a Hyp-SelEx port.

It covers CUB-200-2011 and Stanford Cars with a DINOv1 or DINOv2 backbone. It does not import, and is not imported by, the PLANORA production software.

All tests run on CPU with synthetic data. GPU runs use the CLIs below, and every run records its own provenance.

## Setup and tests

```bash
cd planora_sf
pip install -r requirements.txt && pip install -e .   # editable install: provenance reads the git commit
python -m pytest -q                                    # 386 passed, 9 skipped (about 1.5 min on CPU)
```

- **Opt-in tests.** Set `PLANORA_SF_ALLOW_DOWNLOAD=1` to run the real DINOv1/DINOv2 hub loads and the CUB download. Set `SCARS_ROOT` to run the prepared-Cars check.
- **Other seeds.** `PLANORA_SF_TEST_SEED=<n>` reruns the suite under another random seed.

## Layout

```text
configs/default.yaml            every config key with its source (paper, upstream file:line, or open-item id)
configs/hypcd/e4_port_*.yaml    Hyp-SelEx with the upstream script settings (H-xx in docs/hypcd_port.md)
configs/dinov2/e0_scars.yaml    E0 on Stanford Cars with DINOv2 ViT-B/14 + registers
docs/debgcd_port.md             DebGCD paper-vs-code choices and open items (DG-xx)
docs/hypcd_port.md              HypCD paper-vs-code choices, numerics, Stanford Cars notes (H-xx)
src/planora_sf/
  config.py                     dataclasses; for_dataset(), set_components(), components()
  data/                         CUB, Stanford Cars (+ bounding boxes, crop), GCD split, transforms, sampler
  models/                       backbone adapter (pinned torch.hub DINO), DINOHead, L_cls head, model
  losses/ fourier/ latent/      SelEx, FourEx and GCD losses; Fourier views; latent partition
  pseudo_labels/                BSSK, HSSK hierarchy, USE target matrices
  debgcd/  hyperbolic/          DebGCD heads/losses/surrogate; Poincaré ball and ToPoincare
  engine/                       training loop, loss step, provenance, diagnostics, gradient monitor
  eval/                         transductive evaluator, Hungarian v1/v2, routing/structure diagnostics
  cli/                          command-line tools (below)
scripts/stage_d_cub_check.sh    CUB download + SNR calibration + dry run on a GPU machine
tests/                          pytest suite (synthetic data only)
```

## Command-line tools

Every CLI is `python -m planora_sf.cli.<name>`, and each one prints its full options with `--help`. Each accepts `--dataset dummy` (or a synthetic tree) for offline smoke runs.

| CLI | Purpose |
|---|---|
| `prepare_scars` | Imports Stanford Cars from hand-downloaded archives and verifies it. There is no official download any more. |
| `download_cub` | Downloads CUB-200-2011 and reports the GCD split. |
| `calibrate_snr` | Calibrates the Fourier cut-off T (FX-A Table 7). |
| `train` | Trains one run and writes `last.pt` and `summary.json` to `--checkpoint-dir`. Both carry provenance: seed, commit and dirty flag, config, split and hub. |
| `evaluate` | Re-scores a checkpoint, foreign ViT weights or a frozen DINO under controlled evaluator variants. |
| `compare_evals` | Pairs two evaluations of the same weights by eval seed (e.g. full images vs crops) and applies Gate 3. |
| `audit_checkpoint` | Identifies the dataset, backbone and components of a native or foreign checkpoint. Also derives w from logged All/Known/Novel. |
| `aggregate` | Groups runs by label, pairs them by seed and applies Decision Gates 1–4. Writes JSON and Markdown. |

### `train`

| Flag | Effect |
|---|---|
| `--dataset cub\|stanford_cars\|dummy`, `--root` | Data; `--root` defaults to `$CUB_ROOT` or `$SCARS_ROOT`. Without `--config`, the reference config for the dataset is used. |
| `--config FILE` | YAML config, applied over the generic defaults. |
| `--use-fourex` / `--no-use-fourex`, `--use-debgcd`, `--use-hypcd` | Component toggles. A component without a flag stays as the config sets it. |
| `--backbone dinov1_vitb16\|dinov2_vitb14_reg` | Overrides the config's backbone. |
| `--seed N`, `--label NAME` | Seed (default 1). The label is the run name the aggregator uses; it defaults to the experiment row. |
| `--diagnostics` | Logs per-epoch level-0 pseudo-label accuracy (PL-acc L0) and routing. It does not change training. |
| `--grad-monitor-every N` | Every N steps, records per-loss gradient norms and cosines on the trainable blocks. The losses are base, fourier, debias, hyp_dist and hyp_ang. It does not change training. |
| `--keep-checkpoint-every N` | Also keeps `epoch_XXX.pt` every N epochs, for offline learning curves. |
| `--epochs`, `--max-steps`, `--batch-size`, `--num-workers`, `--cutoff`, `--device` | Run-length and dry-run controls. |

Experiment rows:

| Row | Components | Flags on the reference config |
|---|---|---|
| E0 | SelEx | `--no-use-fourex` |
| E1 | SelEx + FourEx | none (the reference config) |
| E2 | SelEx + DebGCD | `--no-use-fourex --use-debgcd` |
| E3 | all three | `--use-debgcd` |
| E4 | Hyp-SelEx | `--no-use-fourex --use-hypcd` |

### `evaluate`

Choose one weight source:

- `--checkpoint last.pt`
- `--foreign-backbone model.pt`: a ViT saved by the official SelEx, DebGCD or HypCD code. Heads are ignored. Add `--unsafe-load` only for pickled files you trust.
- `--frozen-backbone`: pretrained DINO with no training. Add `--backbone dinov2_vitb14_reg` for DINOv2.

| Flag | Effect |
|---|---|
| `--eval-seeds 0,1,2,3,4` | One clustering per seed on features extracted once. Gives σ_eval. |
| `--clustering bssk\|ss_kmeans`, `--n-init`, `--max-iter` | Evaluator sensitivity. |
| `--logged-v2 A,K,N` | The v2 numbers (in %) that a training log printed. The check passes when \|rescored − logged\| ≤ 2σ_eval + 0.5. |
| `--pseudo-labels` | Reruns phase A per seed: BSSK on train_view0 features. Reports PL-acc L0 and its gap to the evaluator's Novel. |
| `--crop-bbox [--crop-margin 0.1] [--crop-shape square\|tight]` | Evaluates on bounding-box crops. The split and its hash are unchanged. |

The output JSON records the checkpoint SHA-256, split hash, crop setting, stored provenance and per-seed diagnostics: routing, per-class v2, cluster structure and SSB tiers.

## One-pass GPU runbook (Stanford Cars master plan)

Run everything from one clean checkout of a single commit. `aggregate` refuses runs that come from different commits. `runs/` and `audit/` are git-ignored.

```bash
# 0. Setup and data
git clone -b audit-checkpoint https://github.com/thzii/model.git && cd model/planora_sf
pip install -r requirements.txt && pip install -e . && python -m pytest -q
export SCARS_ROOT=/workspace/data/scars CUB_ROOT=/workspace/data/cub E=runs/eval
python -m planora_sf.cli.prepare_scars --root $SCARS_ROOT \
    --from-archive car_devkit.tgz cars_train.tgz cars_test.tgz cars_test_annos_withlabels.mat
PLANORA_SF_ALLOW_DOWNLOAD=1 python -m pytest -q tests/test_backbone.py tests/test_dinov2.py -k real  # caches DINOv1/v2

# 1. Step 1: audit existing checkpoints (no training). DS = the dataset the audit identifies.
python -m planora_sf.cli.audit_checkpoint $CK --logged v2=A,K,N --scars-root $SCARS_ROOT --cub-root $CUB_ROOT --out audit/x.json
python -m planora_sf.cli.evaluate --checkpoint $CK --dataset $DS --eval-seeds 0,1,2,3,4 --logged-v2 A,K,N --pseudo-labels --out $E/x_bssk.json
python -m planora_sf.cli.evaluate --checkpoint $CK --dataset $DS --eval-seeds 0,1,2,3,4 --clustering ss_kmeans --out $E/x_sskm.json
python -m planora_sf.cli.evaluate --checkpoint $CK --dataset $DS --eval-seeds 0,1,2,3,4 --n-init 100 --max-iter 200 --out $E/x_100x200.json
python -m planora_sf.cli.evaluate --foreign-backbone $BASELINE/model.pt --dataset scars --eval-seeds 0,1,2,3,4 --out $E/baseline.json
python -m planora_sf.cli.evaluate --frozen-backbone --dataset scars --eval-seeds 0,1,2,3,4 --out $E/frozen_v1_full.json

# 2. Step 2: reproduce the papers on Cars (Gate 2)
T="--dataset stanford_cars --device cuda --seed 42 --diagnostics"
python -m planora_sf.cli.train $T --no-use-fourex --label S2-E0 --keep-checkpoint-every 50 --checkpoint-dir runs/S2-E0/s42
python -m planora_sf.cli.train $T --config configs/hypcd/e4_port_scars.yaml --label S2-E4p --grad-monitor-every 50 \
    --checkpoint-dir runs/S2-E4p/s42

# 3. Step 3: bounding-box crop, paired on the same weights (Gate 3)
python -m planora_sf.cli.evaluate --checkpoint runs/S2-E0/s42/last.pt --dataset scars --eval-seeds 0,1,2,3,4 --out $E/e0_full.json
python -m planora_sf.cli.evaluate --checkpoint runs/S2-E0/s42/last.pt --dataset scars --eval-seeds 0,1,2,3,4 --crop-bbox --out $E/e0_crop.json
python -m planora_sf.cli.evaluate --frozen-backbone --dataset scars --eval-seeds 0,1,2,3,4 --crop-bbox --out $E/frozen_v1_crop.json
python -m planora_sf.cli.compare_evals $E/e0_full.json $E/e0_crop.json --out runs/cmp/e0.json

# 4a. Step 4a: combined losses on the E4 base (seed 42 screens; 3-seed confirmation also needs S2-E4p at 43, 44)
P="--config configs/hypcd/e4_port_scars.yaml --grad-monitor-every 50"
python -m planora_sf.cli.train $T $P --use-fourex --label S4-F --checkpoint-dir runs/S4-F/s42
python -m planora_sf.cli.train $T $P --use-debgcd --label S4-D --checkpoint-dir runs/S4-D/s42
python -m planora_sf.cli.train $T $P --use-fourex --use-debgcd --label S4-FD --checkpoint-dir runs/S4-FD/s42

# 4b. Step 4b: DINOv2 track
python -m planora_sf.cli.train $T --config configs/dinov2/e0_scars.yaml --label S4-v2-E0 --checkpoint-dir runs/S4-v2-E0/s42
python -m planora_sf.cli.evaluate --frozen-backbone --backbone dinov2_vitb14_reg --dataset scars --eval-seeds 0,1,2,3,4 \
    --out $E/frozen_v2_full.json

# 5. Decision gates
python -m planora_sf.cli.aggregate --runs runs/ --evals $E --audits audit/ --out-json runs/gates.json --out-md runs/gates.md
```

Notes:

- **Baseline log.** The HypCD baseline's log has v1 numbers only, so do not pass them to `--logged-v2`.
- **E0 needs no gradient monitor.** E0 has a single loss, so there is nothing to compare.
- **Cost of the options.** These are estimates, not measured on a GPU:
  - `--grad-monitor-every 50` adds about 10% time.
  - Each `epoch_XXX.pt` of an E0 run is about 0.6 GB.

### Decision gates (`aggregate`)

The aggregator matches runs to gate roles by their `--label`. The default labels are S2-E0, S2-E4p, S4-F, S4-D, S4-FD, S4-v2-E0 and S4-v2-E4; `--e0`, `--e4p` and similar flags override them. All values are v2 accuracies in percentage points, and δ_eval = max(1.0, 2σ_eval).

| Gate | Input | Rule |
|---|---|---|
| 1. Audit | `evaluate` JSONs, audit JSONs | **Exclude** a checkpoint whose log link fails.<br>**Evaluator-limited** if Novel moves more than 5 points across evaluator settings.<br>**σ_eval** = the largest σ over the trained checkpoints. |
| 2. Reproduction | `summary.json` of S2-E0 and S2-E4p | **Pass:** All within ±3 of the paper.<br>**Cars targets:** 56.9 (SelEx†) and 62.9 (Hyp-SelEx).<br>**CUB targets:** 76.4 with Known − Novel ≤ 3, and 79.8.<br>**Grey zone:** up to 1.5 points beyond the band, decided after 2 more seeds.<br>**Flag** Known or Novel more than 5 points off. |
| 3. Crop | crop/full pairs (frozen and trained) | **Supported:** mean Δ ≥ max(1, 2σ_Δ), with every Δ_s > 0. |
| 4. Combined losses | S4-* vs S2-E4p, paired by seed | **Screen:** ΔAll ≥ +1 and ΔKnown ≥ −2.<br>**Confirm** (3 seeds): all Δ > 0, mean ≥ δ_eval and mean − 2SE > 0.<br>**Additivity:** the trio vs the best single, plus the interaction.<br>**Gradient flags:** median cos < −0.1, or one norm more than 10× the others.<br>**DINOv2:** All ≥ 79 passes provisionally; > 80 needs every one of 3 seeds. |

The aggregator refuses runs from mixed commits (`--allow-mixed-commits` overrides), mixed split hashes and duplicate (label, seed) pairs.

## Configuration notes

- **Reference config.** `PlanoraSFConfig.for_dataset(name)` applies the FX-A Table 7 cut-off and the SelEx README smoothing. `set_components()` switches the rows.
- **SelEx code compatibility.** `selex_compat=True` reproduces the official SelEx code behaviour. The D-08 pseudo-label ordering bug is off by default.
- **Port documentation.** The DebGCD and HypCD port choices are listed in [docs/debgcd_port.md](docs/debgcd_port.md) and [docs/hypcd_port.md](docs/hypcd_port.md).
- **Pinned DINO hub code.** `models/backbone.py` pins `facebookresearch/dino@7c446df` and `facebookresearch/dinov2@7764ea0`. Each hub model's structure is checked at load.
- **DINOv2 open items.** `grad_from_block` and smoothing keep the DINOv1 values until a source fixes them.
