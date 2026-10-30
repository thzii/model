# PLANORA-SF research scaffold (Stage B)

This is an isolated research package for the PLANORA-SF method: SelEx + FourEx for fine-grained generalized category discovery. It implements the mechanics defined in [`PLANORA_SF_Method_Spec.md`](../../PLANORA_SF_Method_Spec.md). It does not import, and is not imported by, the PLANORA production software.

**Status:** mechanics and interfaces only. There is no training loop, no data loading, no checkpoint and no reported accuracy. All tests run on synthetic tensors.

## Run the tests

```bash
cd research/planora_sf
pip install -r requirements.txt   # torch, numpy, scipy, PyYAML, pytest
python -m pytest
```

- CPU is enough; the suite takes about 6 s.
- Set `PLANORA_SF_TEST_SEED=<n>` to rerun it under another random seed.
- Two tests are skipped by design:
  - Real DINOv1 loading through `torch.hub` needs network access. Enable it with `PLANORA_SF_ALLOW_DOWNLOAD=1`.
  - The FX-A Table 7 SNR acceptance test needs CUB images (Stage C).

## Layout

```text
configs/default.yaml          every config key, tagged with its spec source or U-id
src/planora_sf/
  config.py                   dataclasses: PlanoraSFConfig, FourierConfig, ..., CUTOFF_TABLE (FX-A Table 7)
  fourier/                    filters.py (box masks), transforms.py (FFT helpers), views.py (blur + views), snr.py
  latent/partition.py         d_l, d_h, SelEx prefixes, min() rule, prefix/suffix masks
  models/                     backbone.py (adapter + torch.hub loaders), projection_head.py (DINOHead),
                              cls_head.py (U-17 default), model.py, tiny_vit.py (synthetic test stand-in)
  losses/                     contrastive.py, selex.py (USE/SSE/SE), fourex.py (low/high/cls), gcd.py, total.py
  pseudo_labels/              bssk.py (balanced SS-k-means), hssk.py (hierarchy), target_matrix.py (USE targets)
  eval/                       evaluator.py (KNOWN / NOVEL_CLUSTER API), metrics.py (Hungarian v1/v2),
                              features.py, estimate_k.py (NotImplementedError)
  engine/                     step.py (one loss evaluation), optim.py (SGD + cosine), train.py (NotImplementedError)
  debgcd/                     DebGCD port (E2/E3): heads.py, losses.py (Eq. 4-13), surrogate.py (SelEx adapters)
  hyperbolic/                 HypCD (E4): poincare.py (ball operations), layers.py (ToPoincare)
tests/                        pytest suite, synthetic tensors only
```

## Stage C: data layer and training engine

- `data/`: the CUB-200-2011 loader and downloader, a synthetic `DummyGCDDataset`, the GCD splitter (RNG parity with SelEx/GCD `subsample_instances`, SSB class lists in `data/splits/cub_ssb_split.json`), transforms, a labelled-first dataset, and the balancing sampler.
- `engine/train.py`: `fit()` alternates phase A (features → BSSK/HSSK pseudo-labels, no grad) with phase B (4-view contrastive steps). The cosine LR steps once per epoch; transductive evaluation runs on the unlabelled training split.
- CLIs (all accept `--dataset dummy` for offline smoke runs):

```bash
export PYTHONPATH=src
python -m planora_sf.cli.download_cub  --root /data/cub                    # download + split report
python -m planora_sf.cli.calibrate_snr --dataset cub --root /data/cub --device cuda   # T vs FX-A Table 7
python -m planora_sf.cli.train         --dataset cub --root /data/cub --device cuda --epochs 1 --max-steps 5
```

On a GPU machine, `CUB_ROOT=/data/cub DEVICE=cuda bash scripts/stage_d_cub_check.sh` runs all of the above. It also runs the opt-in real-data tests and the U-09 calibration variants.

## Configuration

`PlanoraSFConfig()` holds the paper and SelEx-code reference values. Each `U-xx` scaffold default from spec §10 is a config field, so it can be ablated.

- `PlanoraSFConfig.for_dataset("cub")` applies the FX-A Table 7 cut-off and the SelEx README smoothing value.
- `selex_compat=True` reproduces the official SelEx code behaviour (spec §8). `selex.target_mode` can override it per component.
- The SelEx pseudo-label ordering bug (D-08) is **off** by default. Set `selex.reproduce_label_order_bug=True` to replicate it.

## Experiment rows and DebGCD

`cli.train` toggles the plan §4.2 rows on top of any config. A component without a flag stays as configured.

```bash
python -m planora_sf.cli.train --dataset cub --no-use-fourex                 # E0  SelEx
python -m planora_sf.cli.train --dataset cub                                 # E1  SelEx + FourEx (reference config)
python -m planora_sf.cli.train --dataset cub --no-use-fourex --use-debgcd    # E2  SelEx + DebGCD
python -m planora_sf.cli.train --dataset cub --use-debgcd                    # E3  all three
python -m planora_sf.cli.train --dataset cub --no-use-fourex --use-hypcd     # E4  Hyp-SelEx
```

The run summary reports `components` (including the row). DebGCD is the adapted port of the research plan (§2.2). Its paper-vs-code choices and open items are listed in [docs/debgcd_port.md](docs/debgcd_port.md).

The HypCD port, its paper-vs-code choices and its numerics are described in [docs/hypcd_port.md](docs/hypcd_port.md).

## Stanford Cars

There is no official download any more. Fetch the files by hand, then import and verify them:

```bash
python -m planora_sf.cli.prepare_scars --root /data/scars --from-archive     car_devkit.tgz cars_train.tgz cars_test.tgz cars_test_annos_withlabels.mat
export SCARS_ROOT=/data/scars
python -m planora_sf.cli.train --dataset stanford_cars --device cuda
```

`prepare_scars` also accepts zip/tar copies with other nesting (e.g. Kaggle). The labelled test file is optional, because the reported metric uses only the train split. See [docs/hypcd_port.md](docs/hypcd_port.md#stanford-cars).


## Audit, diagnostics and decision gates

These tools support the Stanford Cars master plan: audit, reproduction, crop and combined-loss steps. All of them run offline on `--dataset dummy` or on synthetic trees.

| CLI | Purpose |
|---|---|
| `cli.audit_checkpoint` | Fingerprints native and foreign checkpoints (dataset votes, backbone, components, provenance). Also derives w from logged All/Known/Novel. |
| `cli.train` | Writes `last.pt` and `summary.json` with provenance: seed, commit, config hash and split hash.<br>`--label` names the run for the aggregator.<br>`--diagnostics` logs per-epoch PL-acc L0.<br>`--grad-monitor-every N` records per-loss gradient cosines.<br>`--keep-checkpoint-every N` keeps periodic checkpoints.<br>`--backbone dinov2_vitb14_reg` switches to DINOv2. |
| `cli.evaluate` | Re-scores a checkpoint, foreign ViT weights (`--foreign-backbone`) or a frozen DINO.<br>`--eval-seeds` gives σ_eval.<br>`--logged-v2` checks the log ↔ checkpoint link.<br>`--pseudo-labels` reports offline PL-acc L0.<br>`--crop-bbox` evaluates on bounding-box crops. |
| `cli.compare_evals` | Paired, per-seed deltas between two evaluations of the same weights, e.g. full images vs crops. Applies the Gate 3 rule. |
| `cli.aggregate` | Groups `summary.json` runs by label, pairs them by seed and applies Gates 1–4.<br>Refuses mixed commits or splits.<br>Writes JSON and Markdown. |

DINO hub code is pinned to fixed commits in `models/backbone.py`. The DINOv2 reference config is `configs/dinov2/e0_scars.yaml`.
