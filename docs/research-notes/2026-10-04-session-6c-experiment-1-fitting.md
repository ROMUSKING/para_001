# Session 6C Experiment 1: can the costate training problem be fitted?

**Date:** 2026-10-04 · **Milestone:** Session 6C (learnability audit, Exp 1)
**Hardware:** NVIDIA L4 · **Artefact:** `results/benchmarks/costate_fitting/costate_fitting_summary.json`
**Runner:** `scripts/diagnose_costate_fitting.py` · **Plan:** `docs/plans/2026-10-04-session-6c-learnability-audit-plan.md` §1
**Peer review:** ChatGPT, CHANGES_REQUESTED with revisions recorded before execution (register
`docs/plans/peer-review-register.yaml`); this run implements the revised §1.
**Teacher:** `best_spatial.pt` SHA `cfbca4406f13df55…` (bit-identical backbone, −2.1425);
16 fit windows from 4 train episodes + 16 held-out windows from 4 other train episodes
(ids committed); 3 init seeds; one frozen teacher (not three replications).

## 1. Result: direction fits on train, nothing transfers to new episodes

| student (seed 0) | fit vec | fit dir | fit ρ | fit agree | hold vec | hold dir | hold ρ | hold agree | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| ridge | 0.322 | 0.055 | 0.882 | 0.438 | 1.819 | 1.026 | −0.103 | 0.000 | False |
| composite_h256 | 21.66 | 0.024 | 0.965 | 0.562 | 44.70 | 0.966 | 0.014 | 0.000 | False |
| pure_mse_h256 | 0.421 | 0.089 | 0.839 | 0.312 | 2.434 | 0.967 | −0.027 | 0.000 | False |
| train-mean constant | 0.864 | — | — | — | 1.311 | — | — | — | — |

(Seeds 1–2 same shape: composite fit-vec 24.58/26.85, pure 0.22/0.29; all gates False;
injection control reproduces the exact-λ scores with agreement 0.69/0.25/0.25 and overlap
0.86/0.92/0.98 at k=2/4/8 — the pipeline is sound, and the sub-1.0 agreement is the expected
first-order-score approximation gap, not a bug.)

Held-out overlap falls to near-chance levels (ridge 0.234, composite 0.297, pure 0.203 vs
0.125 chance at k=2) with agreement exactly 0.000 on all 16 windows for every student —
consistent with chance (1/496 per window), i.e. no signal rather than a broken metric.

## 2. Reading (corrected per peer review — routing held, interpretation narrowed)

> **Experiment 1 completed, but no reported model passed the preregistered
> reconstruction-fit gate. The composite head achieved high training directional and
> score-ranking agreement while retaining a large relative reconstruction error; held-out
> ranking correlations were approximately zero on the diagnostic split. These observations do
> not yet isolate information insufficiency or rule out optimisation, scaling,
> representation, or label-pipeline problems. Finite-difference and batch-invariance
> preflight remain pending. The next action is a bounded numerical/calibration
> investigation and completion of the Experiment 2 input manifests; any departure from the
> existing progression rule will be recorded prospectively.**

Corrections to the earlier reading, point by point:

- **No Exp-2 transition.** The gate failed everywhere (best E_rec 0.22), so the "train fits,
  held-out fails → Exp 2" branch does **not** fire. Exp 2 proceeds only under a passed gate
  or an explicit prospective amendment — currently neither. What *is* licensed now is the
  bounded package: finite differences + batch/chunk invariance, unregularised fitting,
  scalar calibration, and manifest preparation.
- **"SmoothL1 instability" downgraded to hypothesis.** The measurements establish a large
  relative reconstruction error, not its cause — inspect absolute target/prediction norms
  before any instability claim (a small denominator inflates relative error without large
  outputs). Required wording: *the composite head exhibits a large relative reconstruction
  error despite strong training directional agreement; its cause is unresolved.*
- **ρ attribution.** Reported ρ values come from `curvature_scores`, which mixes the learned
  co-state with the frozen curvature term — agreement there is not yet attributed to λ̂. The
  bounded package adds the same scorer under learned, exact, zero and mean co-state, all
  else fixed.
- **Held-out 0.000 agreement** needs overlap + regret context (recorded: overlap near chance,
  0.20–0.30 vs 0.125): zero exact-set matches are not independently decisive, though
  near-zero rank correlations remain concerning.
- **Preflight gaps remaining:** finite-difference label checks and batch-invariance (1/B)
  verification were specified (R3) but not executed — the injection control covers
  indexing/normalisation/identity only. They stay prerequisites for the interaction panel,
  which reuses these labels.

## 3. Executed snapshot (review requirement: base commit is not enough)

Base commit `1c1eee2a33d5efcb552b00f639caff0bf23c1dad` plus this working-tree delta
(hashes at write-up; the run executed from these working-tree files uploaded to the Colab
session — a base commit alone cannot identify that snapshot):

| File | SHA-256 (first 16) |
|---|---|
| scripts/diagnose_costate_fitting.py | febad21db0e3d10c |
| results/benchmarks/costate_fitting/costate_fitting_summary.json | e5632e24b5218f8e |
| docs/plans/2026-10-04-session-6c-learnability-audit-plan.md | 0042b7e06d105b27 |
| docs/plans/2026-10-04-session-6c-peer-review.md | 1d7236f6f4cde448 |
| docs/plans/2026-10-04-session-6c-peer-review-2.md | a28036a444a1aa48 |
| docs/plans/peer-review-register.yaml | 6d9f84a5a92acd44 |
| docs/research-notes/2026-10-04-session-6c-experiment-1-fitting.md | 7d6efffc1de1de3f |
| tests/test_spatial_selection.py | e7b1b37f9e9bed00 |
| tests/test_spatial_patch_dropout_backbone.py | c21390090cb4303e |
| README.md / CHANGELOG.md / docs/plans/WORKLOG.md | b4826acd / 00c42981 / 3378e457 |

Full 64-hex values in the session worklog entry. No commit was requested, so this stays
uncommitted — the record above is what makes the run re-identifiable without one.

## 4. Reproduce

```bash
python scripts/diagnose_costate_fitting.py --cache-dir /content/spatial_cache_e3_1 \
    --drive-root /content/local_runs --run-id spatial_patches_20261003 \
    --output-dir results/benchmarks/costate_fitting
```

Peak VRAM within the 1.7 GiB 6A envelope (same backbone, tiny batches); L4 retained.
Regret references are greedy-rollout lower bounds, as throughout Session 6.
