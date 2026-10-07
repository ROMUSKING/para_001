# Decision: close the Hessian-correction branch; prioritise learnable singleton gains

**Date:** 2026-10-04/05 · **Reviewer:** ChatGPT (external, delivered via user message)
**Basis:** reported mechanism-check results (review of reports, not an independent rerun).
**Decision:** close Hessian correction for this formula/intervention; next question is whether
a direct predictor of singleton benefits recovers useful allocation, and whether conditional
scoring improves on it enough to justify cost. Do NOT require beating hindsight ρ≈0.99.

## Recorded consequences

- No further Hessian-distillation sweep for this formula/intervention; preserve the negative
  result and artefacts. Closure rests on the failed approximation premise (stronger), not
  only on B's failure (which alone couldn't distinguish exact-H behavior).
- Taylor statement narrowed: local Taylor expansions still exist; what fails is the
  additive mask-effect simplification. Formalism: with r = z_pq − z_0 − δ_p − δ_q
  non-negligible, the interaction contains −λᵀr plus curvature-in-r terms beyond −δ_pᵀHδ_q.
  Bookkeeping, not a recommendation for elaborate Hessian models.
- A≡B reported as observed identity (32/32 zero differences); p=1 kept only per the
  recorded implementation, never as general equivalence.
- Two gaps replace one: singleton-estimation gap (learned vs oracle singleton regret) and
  residual joint headroom (oracle singleton vs best joint), same windows/constraints/loss.
  Large first gap + small second gap → learn scores first.
- Targets frozen at decision level: b_p(I) = E[L(S_0)−L(S_0∪{p})|I] singleton benefit,
  b_p(I,S) conditional; decision-benefit targets, not realised-gradient reconstruction.
  One benefit-regression baseline; ranking formulation only if reviewed. Future outcomes
  label-only, never in deployed inputs. Conditional training sets must resemble deployment
  (declared set-generation policy). Primary endpoint stays downstream loss/regret.
- Headroom table from existing artefacts before new training; frozen-DINOv2 control before
  expanding learned experiments; protected episode-disjoint evaluation + total cost for any
  retained method. Diagnostic windows stay diagnostic (no tuning-then-confirming on them).

Adopted session conclusion: the additive-displacement cross-Hessian correction is
unsupported (premise fails, magnitude negligible, no decision change). Oracle singleton
gains order near-optimally on the diagnostic set. The unresolved problem is predicting
useful singleton benefits from deployment-available information. Conditional scoring faces
a matched learned singleton baseline, with oracle singletons as privileged reference. No
broader Hessian/learnability claims.
