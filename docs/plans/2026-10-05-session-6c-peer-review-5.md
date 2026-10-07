# Decision: close Hessian branch and norm control; conditional experiment faces a learned singleton baseline

**Date:** 2026-10-05 · **Reviewer:** ChatGPT (external, delivered via user message)
**Basis:** reported mechanism-check + early-pruning results (review of reports, not reruns).
**Decision:** CLOSE the Hessian-correction branch and the norm control as complete; AMEND the
conditional experiment's success criterion (not oracle-ρ chasing).

## Recorded directions

- Hessian branch closed as a documented negative; no repair cycles. Taylor narrowed to the
  tested simplification (with the r-formalism: −λᵀr plus curvature-in-r omitted terms).
  A≡B as observed identity (p=1 per implementation only).
- Two gaps replace one: singleton-estimation gap vs residual joint headroom, same
  windows/constraints/loss. Large first + small second → learn scores first.
- Targets: b_p(I) singleton benefit, b_p(I,S) conditional — decision benefits, not gradient
  reconstruction. One benefit-regression baseline (+ ranking only if reviewed). Labels from
  futures; futures never in deployed inputs. Declared set-generation policy; count label
  and scoring work. Primary endpoint stays downstream loss/regret (Δ_k), never rank
  correlation; cost gate explicit (no quiet waivers).
- Headroom table H_k from existing artefacts before new training; absolute margins near
  zero regret; protected episode-disjoint eval prepared BEFORE model selection.
- Norm: three separate statements (quality / component cost / system savings); mask-identity
  qualification (masks may differ between arms — attribute to configurations); token-dropping
  surgery out of scope.
- Next comparison (frozen): fixed/norm control, learned singleton, matched conditional,
  oracle-singleton diagnostic reference, best-joint hindsight reference. Governed by the
  direct marginal-gain critic comparison (information, supervision, compute matched) —
  conditional success ≠ co-state validation.

Adopted open item: conditional-gain experiment comparing a deployment-valid conditional
predictor against a matched learned singleton predictor and the norm control, one frozen
pipeline, identical budgets, actual downstream loss; oracle ordering diagnostic only;
residual headroom quantified separately; protected eval + cost gates frozen before model
selection; retain conditional only on useful decision gain over its simpler comparator.
