# HypCD port (experiment E4) and Stanford Cars

| Field | Value |
|---|---|
| Paper | Liu, He & Han, *Hyperbolic Category Discovery*, CVPR 2025, arXiv 2504.06120 (§3, §4, supplement) |
| Upstream code | `github.com/Visual-AI/HypCD` @ `ffc895b7ff38be8502bb7e3bd851f1d2a55dff73` (2025-11-07), MIT licence. Read; only the pure-math `hyptorch` files were executed, by an opt-in parity test. |
| Variant | Hyp-SelEx: HypCD applied to SelEx (`train/train_HypSelEx.py`) |
| Code | `src/planora_sf/hyperbolic/` (`poincare.py`, `layers.py`); wired in `models/model.py`, `losses/contrastive.py`, `engine/step.py`, `engine/train.py` |
| Toggle | `hypcd.enabled` (default `false`), or `cli.train --use-hypcd` |

## What changes, and what does not

HypCD replaces the geometry of the projection-head output z, and only there.

| Component | E0–E3 (Euclidean) | With HypCD |
|---|---|---|
| Backbone CLS features | Euclidean | Unchanged |
| Pseudo-labels (BSSK/HSSK) and USE soft targets | ℓ2(CLS), pseudo-labels | Unchanged. The paper (§3.5) and the code (`extract_labeled_protos`) both cluster Euclidean backbone features. |
| Evaluation (`cli.evaluate`, v1/v2) | BSSK on ℓ2(CLS) | Unchanged (upstream `test()`, `test_kmeans()`) |
| z | ℓ2(DINOHead(CLS)) | proj(exp0_c(C_r(DINOHead(CLS)))); the ℓ2 normalisation is dropped, as upstream (`train_HypSelEx.py:275`) |
| USE, SSE, FourEx low/high losses | similarity −‖a−b‖ | α_d·ℓ[−d_c] + (1−α_d)·ℓ[S_ang] for every term |
| FourEx L_cls, DebGCD heads | backbone CLS | Unchanged (they never see z) |
| Parameters, initialisation, state dict | — | Unchanged: `ToPoincare` has no parameters (fixed c) |
| Optimiser | SGD | SGD. No Riemannian optimiser: every trainable parameter is Euclidean, and upstream Hyp-SelEx runs `riemannian=False` (`:674`). geoopt is not needed. |

So k-means needs neither logmap0 nor a Fréchet mean. Clustering logmap0(z) would just be Euclidean k-means on the clipped head output (log0 ∘ exp0 = id). That is a different method in a different space.

## Formulas

```
Ball        D_c^n = { x : c‖x‖² < 1 },  curvature −c (c > 0),  radius 1/√c
Clip        C_r(v)     = min(1, r/(‖v‖ + 1e-5)) · v                          upstream nn.py:154-160
Exp at 0    exp0_c(v)  = tanh(√c‖v‖) · v / (√c‖v‖)                           pmath.py:300-304
Log at 0    log0_c(x)  = artanh(√c‖x‖) · x / (√c‖x‖)                         pmath.py:365-368
Möbius      x ⊕_c y    = ((1 + 2c⟨x,y⟩ + c‖y‖²)x + (1 − c‖x‖²)y) / (1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²)
Distance    d_c(x,y)   = (2/√c) artanh(√c ‖(−x) ⊕_c y‖)                     paper; pmath.py:205-208
                       = (2/√c) asinh(√c‖x−y‖ / √((1 − c‖x‖²)(1 − c‖y‖²)))  computed form
Project     proj(x)    = x, or rescaled to radius (1 − 1e-3)/√c if outside     pmath.py:98-103

Hybrid      ℓ^H = α_d · ℓ[similarity −d_c] + (1 − α_d) · ℓ[similarity S_ang]
            S_ang = cos(z_i, z_j) (paper)   |   −‖z_i − z_j‖ (upstream Hyp-SelEx)
            α_d(e) = α_max · clamp((e + δ)/E, 0, 1),  e 0-based, E = optim.epochs
            L_SE^H = (1 − λ_b) L_USE^H + λ_b L_SSE^H,  λ_b = 0.35, τ_u = 1.0, τ_s = 0.07
```

The second distance form follows from arcosh(1 + 2s²) = 2·asinh(s). It is used for every pairwise matrix:

- **Memory.** It needs one N×N matrix product, like the Euclidean similarity. Upstream `dist_matrix` builds an N×N×D tensor: about 2.1 GB per call at N = 256, D = 8192, and about 17 GB at D = 65,536.
- **Gradients.** asinh is smooth at 0, so the self-pairs on the diagonal give finite gradients.

**Slices of ball points.** SSE prefixes and the FourEx partition slice coordinates of the ball points, as upstream does for the SSE prefixes (`:356-357`). A coordinate slice satisfies ‖z_S‖ ≤ ‖z‖ < 1/√c, so it lies in the lower-dimensional ball, and d_c on it is the distance there. Masked per-sample widths (FourEx L_LSSE) use the same masked-norm expansion as the Euclidean path.

## Paper vs upstream code

The paper is the default wherever the two differ. Upstream values belong to the E4-port reproduction, not the matrix row.

| ID | Item | Paper (default here) | Upstream code | Config |
|---|---|---|---|---|
| H-01 | Angle similarity | Cosine of the hyperbolic embeddings | `−cdist` on ball points (`:341`, `:353`); Hyp-GCD uses cosine | `angle_similarity: cosine \| neg_euclidean` |
| H-02 | Curvature c | 0.05 (fine-grained) | 0.1 for CUB and Cars with DINOv1 (`scripts/train_HypSelEx.cmd:13-14`) | `curvature` |
| H-03 | Clip radius r | 2.3 (best on CUB in an ablation) | 2.0 | `clip_r` |
| H-04 | Embedding dimension | "8,092" | 8192 (`:34`) | `latent.dim` |
| H-05 | Trainable blocks | Last 2 (CUB, Aircraft), last 3 (Cars) | `grad_from_block 10` (last 2) everywhere | `backbone.grad_from_block` |
| H-06 | α_d schedule | e·α_max/200 | (e+1)/200·α_max (`:361-364`) | `alpha_epoch_offset: 0 \| 1` |
| H-07 | USE smoothing on Cars | — | 0.5 (SelEx README: 1.0) | `selex.smoothing_alpha` |
| H-08 | Sampler weights | — | label_len/(unlabelled+labelled) | Not ported |
| H-09 | Checkpoint | — | Best by test-set Old accuracy | Not ported (last epoch) |
| H-10 | Pseudo-label order | — | Inherits SelEx's D-08 bug (`:496-521`) | `selex.reproduce_label_order_bug` |
| H-11 | Temperature of the cosine branch | Not stated | Not applicable | τ unchanged (open) |

**Why H-01 matters.**
- Clipping and exp0 only rescale along the radius, so cos(z_i, z_j) = cos(v_i, v_j). The paper's angle branch ignores c and r (tested).
- Upstream's branch on clipped points is SelEx's Euclidean similarity scaled by ρ = tanh(√c·r)/√c (about 2.1).
- At τ_u = 1, the cosine branch gives logits in [−1, 1], much flatter than upstream's [−2ρ, 0].
- Because α_d starts near 0, the two choices train differently for roughly the first half of the run.

**Not ported (pmath).**
- `pmath.arcosh` clamps its input to (−1, 1), outside arcosh's domain (x ≥ 1). It is unused in Hyp-SelEx and not ported.
- `RiemannianGradient` stores c in a class attribute shared by all instances. Hyp-SelEx doesn't use it.

## Numerics

- **Headroom.** With clipping, c‖z‖² ≤ tanh²(√c·r): 0.224 with the paper's values and 0.313 with upstream's. Points stay well inside the ball in float32, and `proj` is a guard that never fires.
- **Clamps (upstream's).** tanh argument ≤ 15; artanh argument ≤ 1 − 1e-5. So log0 inverts exp0 only for √c‖v‖ ≲ 6.1, which clipping keeps far away (√c·r ≈ 0.51).
- **Diagonal floor.** The 1e-12 floor under ‖x−y‖² gives d(x, x) = 2·10⁻⁶ / (1 − c‖x‖²) in float64.
- **Float32 diagonal.** The fast form computes ‖x−y‖² as ‖x‖² + ‖y‖² − 2x·y. In float32 that cancellation leaves d(x, x) up to about 1e-2 at D = 65,536, the same effect as the existing Euclidean similarity, larger because ball points are longer than unit vectors. Off-diagonal entries match the float64 Möbius form to 1e-4. The diagonal is masked out of every contrastive loss.

**Clipping saturation** (analysis, confirmed on the tiny model).
- DINOHead's last layer has unit-norm rows, so averaged over directions ‖v‖² = D/d_b (d_b = 256 bottleneck) throughout training.
- That is 32 at D = 8192 and 256 at D = 65,536, against r² ≈ 5.3.
- When clipping is active, ∂C_r/∂v = (r/‖v‖)(I − v̂v̂ᵀ): the radial gradient is zero.
- So most samples are expected to sit on the sphere of hyperbolic radius 2r, and the radius carries information only for samples with ‖v‖ < r.
- `hyp_clip_fraction` and `hyp_tangent_norm` are logged every epoch to check this. On the tiny test model (D/d_b = 8) the clipped share was 1.0 and the mean ‖v‖ 2.87, against a predicted √8 ≈ 2.83.

## Monitors

These are written into each epoch's history; none read labels:

| Key | Meaning |
|---|---|
| `hyp_alpha` | α_d this epoch |
| `hyp_clip_fraction` | share of rows with ‖v‖ > r |
| `hyp_tangent_norm` | mean ‖v‖ |
| `hyp_base_dist`, `hyp_base_ang` | the SelEx base loss under each branch |

## Rows

| Row | Definition | Target, v2 All / Known / Novel (DINOv1) |
|---|---|---|
| E4-port (CUB) | Upstream script: blocks 10–11, D = 8192, c = 0.1, r = 2.0, `neg_euclidean` angle, δ = 1, smoothing 1.0 | 79.8 / 75.8 / 81.8 (HypCD Table 1); gate G4: All ± 3 |
| E0-D8k | E0 with D = 8192 (control) | — |
| E4 | E0-D8k + HypCD, paper defaults | New |
| Cars E4-port | Upstream script, smoothing 0.5 | 62.9 / 80.0 / 54.7 |

- The E4-port rows are locked in `configs/hypcd/e4_port_cub.yaml` and `configs/hypcd/e4_port_scars.yaml`. `tests/test_port_configs.py` checks that they differ from `for_dataset()` in exactly the upstream keys. They also reproduce upstream's D-08 label order (H-10). Run one with `cli.train --config configs/hypcd/e4_port_cub.yaml --dataset cub`.
- `components()` names E4 only for SelEx + HypCD with FourEx and DebGCD off.
- HypCD + FourEx and HypCD + DebGCD run, but have no row. They are composed configurations (plan D3), kept out of the matrix until G4 passes.
- On Cars, every published method has Known − Novel between +25 and +30, so Cars gates check All ± 3 only.

## Stanford Cars

There is no official download. ai.stanford.edu/~jkrause/cars no longer serves the files, and torchvision's `StanfordCars(download=True)` raises.

**Layout read** (`data/scars.py`), torchvision's:

```
<root>/stanford_cars/
    cars_train/*.jpg                  8,144 images
    cars_test/*.jpg                   8,041 images (optional)
    devkit/cars_train_annos.mat       annotations: bbox_x1..bbox_y2, class (1-based), fname
    devkit/cars_meta.mat              class names (optional)
    cars_test_annos_withlabels.mat    labelled test annotations (optional)
```

- The GCD/SelEx layout (the same folders directly in `<root>`, labelled test file in `devkit/`) is also read, as is Kaggle's nested `cars_train/cars_train/`.
- Class ids are `class − 1`.
- Images are kept in annotation order, which matters: the labelled set must match GCD/SelEx's random split.
- Greyscale images are converted to RGB.

**Test labels are optional.** The reported metric is transductive on the train split. Without the labelled test file, the test split is skipped, with a warning.

**Split.** `data/splits/scars_ssb_split.json` comes from SSB's `scars_osr_splits.pkl` (sha256 `28d471f0…`). That file is byte-identical in SelEx@569ee70, HypCD@ffc895b and DebGCD@9ec30fe. It was read with an unpickler that refuses every class. It has 98 known and 98 novel classes (Hard 7, Medium 15, Easy 76).

**Reference config.** `PlanoraSFConfig.for_dataset("scars")`: T = 27, smoothing 1.0, balanced clusters. CLI name `stanford_cars` or `scars`. `--root` defaults to `$SCARS_ROOT`.

### Import

`python -m planora_sf.cli.prepare_scars --root DIR [--from-archive FILE ...] [--check-files]`

- **Input.** Any mix of `.tgz`/`.tar.gz`/`.tar`/`.zip` archives and loose `.mat` files. It locates the folder named `cars_train` (and `cars_test`) that directly contains `.jpg` files, and the devkit and label files by name, at any nesting depth.
- **Output.**
  - It moves them into `<root>/stanford_cars/` and verifies the counts.
  - It writes `planora_manifest.json` with SHA-256 hashes of every `.mat` and of each input file.
  - It prints the split report.
- **Exit code.** 1 if the counts are not 8,144 train / 196 classes, or 8,041 test when test labels are present.
- **Safety.** Extraction refuses paths outside the target, links and device files. The tool never overwrites an existing `stanford_cars/` and makes no network calls.
- **MD5 check.** Files with the original-release names are compared with the MD5s torchvision pinned (torchvision 0.15 `datasets/stanford_cars.py`). A mismatch is reported, not fatal, since re-packed copies differ.

| File (original release) | MD5 |
|---|---|
| `car_devkit.tgz` | `c3b158d763b6e2245038c8ad08e45376` |
| `cars_train.tgz` | `065e5b463ae28d29e77c1b4b166cfe61` |
| `cars_test.tgz` | `4ce7ebf6a94d07f1952d94dd34c4d501` |
| `cars_test_annos_withlabels.mat` | `b0a2b23655a3edd16d84508592a98d10` |

**Sources** (third-party, none official):

- **Devkit.** `car_devkit.tgz`, attached to torchvision issue #7545: `https://github.com/pytorch/vision/files/11644847/car_devkit.tgz`.
- **Images.** Kaggle, e.g. `jessicali9530/stanford-cars-dataset`. It needs a Kaggle account or API token, and its nesting is handled by the importer.
- **Labelled test file.** It circulates only through Kaggle notebooks. Being optional, it can be skipped.

The images are web-scraped. Use them for benchmark research only, and keep them, and models trained on them, out of production.

## Verification

All checks run on CPU:

| Area | Tests |
|---|---|
| Geometry (`tests/test_poincare.py`) | log0∘exp0 and exp0∘log0 = id; d(0, exp0(v)) = 2‖v‖; Möbius identities and left cancellation; three distance forms agree; metric axioms; Möbius translation is an isometry; c → 0 gives 2‖x−y‖; clipping bounds, direction, and zero radial Jacobian; radius bounds; cosine invariance; gradcheck; float32 at D = 65,536 with finite gradients through the diagonal |
| Upstream parity (opt-in, `HYPCD_UPSTREAM=/path/to/clone`) | `ToPoincare`, `dist`, `dist_matrix` (off-diagonal) and `logmap0` match `hyptorch` |
| Model (`tests/test_hypcd_model.py`) | Parameter-free; identical init and state dict; backbone features unchanged; gradients reach the head and the trainable blocks |
| Losses (`tests/test_hypcd_losses.py`) | α mixes every term linearly; finite losses and full gradient flow for E4, E4+FourEx, E4+DebGCD; cosine branch independent of c and r; cluster-relabelling invariance; unlabelled ground truth unused |
| Loop (`tests/test_hypcd_loop.py`) | α_d schedule (paper and upstream); epoch threaded to the loss; monitors; CLI train → `cli.evaluate` round trip |
| E0–E3 (`tests/test_rows_golden.py`) | Losses, v1/v2 and every parameter sum match values recorded on `2909cc6`, before HypCD; byte-identical on the machine that recorded them |
| Stanford Cars (`tests/test_scars.py`) | Both layouts and Kaggle nesting; annotation order; greyscale; optional test labels; SSB file; archive import from original-release files and a nested zip; traversal/link/device rejection; real-data counts (opt-in, `SCARS_ROOT`) |

## Open items

- **H-11:** temperature of the cosine branch (kept at τ_u/τ_s).
- **D1:** paper or upstream defaults for the matrix row. The paper is the default; upstream values apply to E4-port only.
- **D2:** run E4 at D = 8192 with the E0-D8k control.
- **Cars blocks (H-05):** the paper says 3, the code uses 2. The Cars E4-port run should record which was used.
