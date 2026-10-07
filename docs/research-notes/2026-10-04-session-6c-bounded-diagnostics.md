# Session 6C bounded diagnostics: preflight, controls, and the routing record

**Date:** 2026-10-04 · **Milestone:** Session 6C (learnability audit, bounded package)
**Hardware:** NVIDIA L4 · **Runners:** `scripts/check_label_contract.py`,
`scripts/diagnose_costate_fitting.py` (extended), `scripts/diagnose_spatial_selection_bottleneck.py`
(zero/mean arms), `scripts/diagnose_norm_controls.py`, `scripts/diagnose_pair_interactions.py`
**Peer review:** ChatGPT routing-correction review (`docs/plans/2026-10-04-session-6c-peer-review-2.md`);
this package is its work-package items 2–4. Execution of Exp 2 remains gated.

## 1. Preflight: labels implement the declared contract

Artefact: `results/benchmarks/label_preflight/` (teacher SHA `cfbca4406f13df55…`, same backbone).

- **Batch invariance PASSES:** alone/batched/permuted/singletons agree to ≤9e-8 absolute
  against reference scale 0.031; alone-over-batched median norm ratio 1.0000001.
  Per-example gradients are sum-like — **no 1/B leak**, verified not assumed.
- **Finite differences agree at eps=0.01** (median rel 1.4e-3, max 3.2e-2), marginal at
  1e-3 (max 13%), noise-dominated at 1e-4 (median 35% — float32 floor on O(1) objectives,
  not a label defect). Labels match the production objective at resolvable scales.

## 2. Unregularised fit + scalar calibration (Exp-1 rerun, same 16+16 windows)

- **lstsq (float64 gelsd, rcond 1e-12): rank 16/16, residual exactly 0, gate TRUE.**
  σ_max 89.7 → σ_min 0.026 (condition ~3500, comfortable). Finite-sample labels are
  linearly representable — the production path's failure is optimisation/loss-path, not
  representability. Does not retroactively pass any production gate.
- **Scalar calibration:** composite a* = 0.0435 — fit vec 21.66 → **0.285** from one global
  scalar, confirming global miscalibration dominates its vector error; held-out 44.7 →
  2.15 (generalisation gap preserved, better localised). Pure a* = 0.90 (scale already
  roughly right). Ridge uncalibrated (noted limitation). Constant predictor published
  (fit 0.86 / held-out 1.31); held-out student errors exceed the zero-predictor reference
  1.0 — worse than predicting nothing on unseen episodes.
- Calibrated lstsq (a* = 1.0, no-op): fit 0, held-out 2.27 — even exact train-fit leaves
  held-out error > 2× the zero baseline.

## 3. ρ attribution: the learned head ≈ the train mean; zero beats it

Same scorer, only λ varies (zero Hessian, true Δz fixed):

| scorer | ρ | regret@2 | overlap |
|---|---|---|---|
| exact costate | +0.9771 | 0.000022 | 0.870 |
| train-mean λ | +0.0943 | 0.003995 | 0.277 |
| distilled λ̂ | +0.1151 | 0.003875 | 0.253 |
| zero λ (constant scores) | NaN (declared) | **0.003349** | 0.273 |
| patch norm | +0.0795 | 0.003916 | 0.236 |

The distilled head is indistinguishable from predicting the **train mean** (0.094/0.003995
vs 0.115/0.003875) — the H-target signature: E[λ|I] ≈ global mean, i.e. the conditioning
carries almost no usable signal. And constant-zero scores (fixed positional top-k) **beat**
the learned head (0.003349 < 0.003875). NaN for the constant arm is declared handling, not
a gap.

## 4. Norm controls: postencoder wins, neither saves work in this pipeline

Same windows, realised J, all budgets:

| k_cam | preview (cheap) | postencoder (DINOv2 norm) |
|---|---|---|
| 2 | 0.004593 | 0.003731 |
| 4 | 0.004436 | 0.003461 |
| 8 | 0.003112 | 0.002518 |

Postencoder wins ~20% throughout. Timing (L4, preregistered): preview 0.56 ms/frame,
DINOv2 re-encode 9.30 ms/frame (17×), top-k 0.52 ms, masked rollout 1.56 ms. Neither arm
saves work **in the current pipeline** — the encoder runs anyway for the world model, so the
postencoder norm is free and the preview is pointless *here*. Preview matters only with
pre-encoder pruning, which is not implemented. Cache-assisted readings would need the
9.3 ms/frame refresh declared, not cache-load time.

## 5. Interaction panel: singleton scores are nearly irrelevant to joint outcomes

200-pair exact manifest (within/cross/distance strata), empty mask verified legal
(mean J −2.16), actual rollout J throughout: mean |ε| = **1.57e-03** against mean pair gain
3.74e-05 — interactions are **~42×** the gains they modify. Best-pair vs top-2-singletons
at k=2: −0.0092 trimmed (negative → top singletons win; redundancy dominates). Singleton
additive scoring cannot see the dominant term. Any conditional selector is compared against
these same fixed/direct baselines with scoring calls charged — not yet built.

## 6. Routing record

Preflight passes → the label pipeline stands (R3 done except nothing further outstanding on
these windows). Exp-1 gate still failed (best E_rec 0.22 pure; lstsq gate passes but is
explicitly not a production pass). Per the frozen rule as corrected, Exp 2 requires either
a passed gate or a **prospective** amendment — this note is not that amendment. The
converging picture (mean-like λ̂, zero beating learned, 42× interactions, preview≈postencoder
cost ratio 17× with no pipeline savings) now bounds what any amendment could claim: the
next contribution, if any, is a conditional/interaction-aware allocator or a documented
limitation result — not a finer λ̂.

## 7. Reproduce

Substrate: cache rebuild (500 episodes, manifest lengths recorded) → backbone (final
selection score −2.1425, same as every prior run) → the four runners above in stage order
(preflight gates the rest). Same model/batch regime as the measured 1.7 GiB runs; no new
peak was profiled here, so no peak is claimed. L4 retained. Per-window parquet + manifests
committed for the rerun benchmark; JSON summaries for the diagnostics (raw per-window rows
for pairs/norm are summarised in-JSON; recompute from the committed manifests + checkpoint
to audit).
