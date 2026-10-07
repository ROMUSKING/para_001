# Session 6C Experiment 1: follow-up review — correct the routing

**Review date:** 2026-10-04 · **Reviewer:** ChatGPT (external, delivered via user message)
**Basis:** the Exp-1 implementation, run output and research note as reported by the lead.
**Experiments executed for this review:** NONE · **Repository modified by reviewer:** NO
**Overall:** retain the implementation work; correct the experimental routing. The reviewer's
earlier R1–R10 stand; this adds routing discipline on top.

## Decision

The reported engineering checks are green; the reconstruction gate is not. Proceeding to
Experiment 2 as though the gate passed would contradict the frozen rule. Next: **one bounded
numerical/calibration investigation**, then a recorded routing decision — not another broad
training sweep. High training rank does not substitute for the gate (λ̂ = 25λ gives zero
directional error with relative error 24, and preserves every first-order ranking).

## Required items (lead's mapping to actions)

1. **Interpretation table** (note §1): composite direction ≈0.02/rank ≈0.96 coexists with
   vec error 21–27 — strong train agreement on those metrics only. Gate False everywhere:
   no automatic Exp-2 transition. Held-out ≈ 0: transfer failure on this split, not
   identified information insufficiency. Finite-difference/batch-invariance still open:
   label pipeline preflight incomplete.
2. **"SmoothL1 instability" is a hypothesis**, not established (SmoothL1's derivative alone
   does not explain magnitude growth). Required wording: large relative reconstruction error
   despite strong directional agreement; cause unresolved (scale, balance, reductions,
   optimisation). Inspect absolute norms first — small denominators inflate relative error.
   Also: clarify what reported ρ measures (shared frozen curvature term confound) — report
   the same scorer with learned, exact, zero and mean co-state, all else fixed.
3. **Three bounded diagnostics before another substantial experiment:**
   A. Derivative preflight (individual/batch/permuted/chunks; 1/B reduction convention;
      directional finite differences with absolute + scaled discrepancies).
   B. Unregularised linear fit (float64 lstsq, rank/singular values/residual/rcond —
      `torch.linalg.lstsq`, no normal equations); full-row-rank interpolation = finite-sample
      fitability only; must not retroactively pass the production gate.
   C. Train-fitted scalar calibration a* (held-out application only; record both; publish
      constant predictor; held-out errors exceed the zero-predictor reference 1.0).
4. **Check the "random gains" test**: random subsets agree by chance (1/6 for M=4,k=2);
   stochastic tests must assert distributions, not impossible universals. Held-out agreement
   0.000 needs overlap + regret context — zero exact matches are not independently decisive.
5. **Exp 2 stays next but gated**: either pass the gate after preflight + bounded
   diagnostics, or amend routing prospectively with limits stated. Manifests can be prepared
   now. Norm control + interaction panel proceed after shared rollout/scoring checks pass.
6. **Work package order:** (1) interpretation correction; (2) finite differences + batch
   invariance; (3) unregularised fit + calibration; (4) norm/interaction manifests;
   (5) recorded routing decision. Record base commit **plus content hash** of the executed
   snapshot including untracked files. Do not keep tuning against the 16 held-out windows
   while calling them untouched.
7. Adopt the suggested replacement conclusion in the research note (Exp 1 completed, gate
   failed, cause unresolved, preflight pending, next = bounded investigation + manifests).

Direction unchanged: the learnability audit. Nothing reopens retired VOI. The correction is
narrower: high training rank is worth investigating, not permission to relabel a failed
reconstruction gate as a generalisation-only failure.
