# DebGCD port (experiments E2 and E3)

| Field | Value |
|---|---|
| Paper | Liu et al., *Debiased Learning for Generalized Category Discovery*, ICLR 2025, arXiv 2504.04804, §4.2–4.3 (Eq. 4–14), §5.1, App. K |
| Upstream code | `github.com/Visual-AI/DebGCD` @ `9ec30fe3d54dfccd9934a862cf5cb9ff05de6f05` (2025-09-27), MIT licence. It was read but not run. |
| Port form | Adapted (research plan §2.2), chosen by the research lead |
| Code | `src/planora_sf/debgcd/`; wired in `engine/step.py::_add_debgcd` and `engine/train.py::align_to_previous` |
| Toggle | `debgcd.enabled` (default `false`), or `cli.train --use-debgcd` |

## What is ported

| DebGCD component | Paper | Here | Evidence tag |
|---|---|---|---|
| Semantic distribution detector ρ_s → ℓ2 → M one-vs-all heads | §4.2, App. K | `SemanticDistributionDetector`. It has 5 Linear-GELU layers (2048 wide, 256 bottleneck) and a cosine last layer with 2M outputs and logits divided by 0.1. Its input is the **un-normalised** CLS of x(1) and x(2). | PAPER + upstream `model.py:37-85` |
| L_sdl^s, hard negative | Eq. 4 | `sdl_labelled` | PAPER = upstream `ova_loss` (`train_DebGCD.py:37-47`) |
| L_sdl^u, entropy | Eq. 5 | `sdl_unlabelled` | PAPER, with the head normalisation taken from upstream (DG-02) |
| Score s, certainty d | Eq. 7, 10 | `distribution_score`, `certainty` | PAPER = upstream `:155-160` |
| Consistency filter F | Eq. 9 | `consistency`, applied when `debgcd.consistency_filter: true` | PAPER; absent upstream (P-01) |
| Debiased classifier ψ_s, K cosine prototypes / τ_a | Eq. 8 | `DebiasedPrototypeHead` on the ℓ2 CLS | PAPER = upstream `last_layer_deb` |
| L_adl^s, L_adl^u | Eq. 11–13 | `adl_labelled`, `adl_unlabelled`, `debgcd_terms` | PAPER |
| L_all = L_gcd + λ_sdl L_sdl + λ_adl L_adl | Eq. 14 | `add_debgcd`. Here PLANORA-SF's L_total (SelEx, plus FourEx in E3) takes the place of L_gcd. | PAPER, with the base swapped |

**Not ported:** SimGCD's prototype classifier, its self-distillation and its mean-entropy regulariser. That is the L_gcd that SelEx replaces (plan C1). ψ_s is training-only, and inference stays BSSK on the CLS (plan C10).

## Composed adapters (no paper; plan §2.1)

- **Class posterior p_i (C1).** p_i = softmax_k(cos(CLS_i, μ_k) / τ_p) over that epoch's level-0 BSSK centroids, with no gradient. It is computed per view, as upstream does with its own classifier. ŷ_i = argmax p_i, and ŷ_i counts as novel when ŷ_i ≥ K_s. τ_p = 0.1 (DG-01).
- **Stable class ids (C2).** After each phase A, level-0 novel clusters are Hungarian-matched to the previous epoch's by cosine similarity of their centres, and relabelled. Known clusters keep their ids.
  - SelEx and FourEx losses read pseudo-labels only through equality and `radii[labels]`, so relabelling does not change them. `test_permuting_level0_novel_ids_leaves_every_selex_and_fourex_loss_unchanged` checks this for all three target modes.
  - `test_zero_weight_debgcd_matches_disabled_including_cluster_alignment` checks it end to end: the run applies non-identity permutations and still matches the disabled run.
- **Known-cluster anchoring (DG-08).** Cluster j is known class j only if every known class has labelled samples. `align_to_previous` raises an error otherwise.

## Paper vs upstream code

The paper is the default wherever the two disagree. The CUB numbers in the paper may have been produced by the upstream settings rather than the paper values (P-04).

| ID | Item | Paper | Upstream code | Default here |
|---|---|---|---|---|
| P-01 | Consistency filter F in L_adl^u | Eq. 9 multiplies the weight by F | Absent: the weight is threshold × certainty (`train_DebGCD.py:174-177`) | Paper (`consistency_filter: true`); `false` gives the upstream weighting |
| P-02 | L_sdl^u | Entropy minimisation (Eq. 5) | 0.1 × entropy, plus an OpenMatch soft cross-view consistency term (`:143-151`) | Paper. The consistency term is not implemented. |
| P-03 | L_adl weighting | L_adl^s + L_adl^u (Eq. 13) | (1 − pl)·L_adl^s + pl·L_adl^u with pl = 0.5 (`:167`, `:179`) | Paper |
| P-04 | CUB hyperparameters | λ_sdl 0.01, λ_adl 1.0; τ 0.85 is the best value in the Aircraft ablation | `scripts/train_cub.cmd`: sdl 0.005, adl 4.0, thr 0.65, pl 0.5, grad_from_block 10, memax 2.0 | Paper (DG-07). The upstream values are noted in `config.py` comments. |
| P-05 | Pseudo-label source | GCD classifier | SimGCD head, per view (`:171-174`) | Centroid posterior (C1) |
| P-06 | Checkpoint | Not stated | Saves the "best" by test-set Old accuracy (`:213-218`) | Not ported. Last epoch only (U-28). |
| P-07 | Weight decay | Not stated | Biases and 1-D parameters excluded (`model.py:231-242`) | One SGD group, as for SelEx (DG-09) |

## Open items

| ID | Item | Status |
|---|---|---|
| DG-01 | τ_p of the centroid posterior | Open. 0.1 (= τ_a); untuned by design (plan §4.4). |
| DG-02 | Eq. 5 indexing and head normalisation | Resolved from code: mean over the M heads (`ova_ent`, `:50-54`). Summing instead would scale L_sdl^u by M (100 on CUB). |
| DG-03 | Eq. 9 text says it filters out agreeing samples; the formula keeps them | Open. Formula reading. Upstream has no F (P-01), so the code can't settle it. |
| DG-04 | View and gradient for p_i and s_i | Resolved: per view; OVA scores detached (`:155`) |
| DG-05 | Detector widths and input | Resolved: 5 layers, 2048 / 256, GELU, no BN, raw CLS input, gradient into the backbone (`model.py:37-85`) |
| DG-06 | Hard negative | Resolved: max over k ≠ y of −log o⁻ (`:45`), the same as the paper's min |
| DG-07 | Per-dataset hyperparameters | Paper values used (P-04). The upstream CUB values would be an E2 variant, not a silent default. |
| DG-08 | Known classes without labels | Guarded (raises an error). Never triggers on CUB. |
| DG-09 | Parameter groups | One group (P-07) |
| DG-10 | Eq. 11 normalisation | Resolved: 1/|B_u| over unlabelled rows of both views. Paper and code (`.mean()`, `:177`) agree. |
| DG-11 | Consistency filter missing upstream | Configurable (P-01) |

## Not yet implemented

These depend on the T2 diagnostics module (plan §5), which reads unlabelled ground truth in one audited place:

- utilisation split by ground-truth known/novel (only overall utilisation is logged now, as `adl_utilisation`);
- AUROC of s;
- ψ_s-argmax accuracy as a secondary metric;
- per-head gradient norms on blocks 9–11 (C3).
