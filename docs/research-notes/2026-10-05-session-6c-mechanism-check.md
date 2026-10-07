# Session 6C mechanism check: additive-displacement cross-Hessian unsupported; branch closed

**Date:** 2026-10-05 · **Milestone:** Session 6C (learnability audit, mechanism check)
**Hardware:** NVIDIA L4 · **Artefact:** `results/benchmarks/mechanism_check/mechanism_check_summary.json`
**Runner:** `scripts/diagnose_mechanism_check.py` · **Design:** peer-reviewed (codex,
milestone `Session-6C-mechanism-design`) — train on TRAIN, measure additivity, A/B/C/D
arms, per-window correlations, top-k overlap, episode-clustered uncertainty.
**Head:** production composite, 300 steps on the TRAIN split (test windows disjoint);
backbone score −2.1425 again. 32 stratified test windows, frozen 200-pair manifest.

## 1. The additive-displacement premise fails (narrowly stated)

Median relative residual ‖δ_{pq} − (δ_p + δ_q)‖ / ‖δ_p + δ_q‖ = **1.235** (mean 1.237);
fraction of pairs below 10% residual: **0.0**. With r = z_{pq} − z_0 − δ_p − δ_q
non-negligible, the interaction expansion contains omitted terms — including −λᵀr plus
curvature terms involving r — beyond the tested −δ_pᵀHδ_q. So the precise conclusion is:
the **cross-Hessian approximation based on additive singleton latent displacements is
unsupported for the tested joint-mask interventions**. A local Taylor expansion of the
differentiable downstream loss can still exist; this closes the Hessian-correction branch
for this formula and intervention representation, nothing broader.

## 2. The predicted correction is four orders of magnitude too small and uncorrelated

Mean |predicted ε| 9.84e-08 vs mean |measured ε| 1.57e-03 (ratio ≈ 1/16,300);
correlation **0.23** (weak, uncertainty unreported — not literally zero, and immaterial to
the decision: the correction is negligible at any correlation). Arms A and B produce
**identical reported outcomes** on all 32 windows (ρ 0.0896, overlap 0.0469, regret
0.00489; B−A = 0.0 on every paired difference): the implemented correction supplies no
decision benefit on these windows. The Wilcoxon p = 1.0 follows the recorded
implementation's all-zero handling and is **not** a general equivalence result. Arm C
(oracle singletons + same correction): ρ 0.9887, overlap 0.875, regret ~0 — C's
performance is the additive ordering, already near-perfect without the correction.

## 3. Reading (adopted review conclusion)

The tested additive-displacement cross-Hessian correction is unsupported: its premise
fails, its predicted interaction magnitude is negligible, and it changes no evaluated
decisions. Oracle singleton gains nevertheless achieve near-optimal ordering on the
diagnostic candidate set. The immediate unresolved problem is therefore predicting
useful singleton benefits from deployment-available information. Conditional scoring
will be assessed against a matched learned singleton baseline, with oracle singletons
retained as a privileged reference. No broader conclusion about the usefulness of
Hessians or the learnability of the target is claimed.

## 4. Reproduce

Substrate rebuild (cache → backbone) then `scripts/diagnose_mechanism_check.py` with the
frozen manifest (`--n-pairs 200 --pair-seed 7`, 32 stratified windows). L4 retained.
