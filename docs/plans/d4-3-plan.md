# D4-3: the varying-goal check (cheap versus cheap-adjoint, goal drawn per instance)

**Written:** 2026-09-30, before any D4-3 tuning or validation instance was generated · **Decision:** Roman, 2026-09-30 ("want the varying-goal check") · **Roadmap:** D4-3 · **Follows:** [D4-2](d4-2-plan.md) and its [note](../research-notes/2026-09-30-d4-2-flop-priced-scoring.md) (hypothesis 2 and next step 2) · **Precedence:** the cross-domain plan §5 (D4) and the roadmap stop-losses.

## 1. Question and scope

**Question.** D4-2 found that the tabulated co-state weight helps a cheap, non-learned estimator in only one of seven cells (`m` = 64, about 10 %), and never in the cells where a learned scorer paid. Each D4-2 cell had **one goal, drawn once**, so that result mixes the value of the weight with the luck of a single goal draw. D4-3 draws a **fresh random goal for every instance** of the same seven cells and asks again: does a co-state weight, given at lookup price, help a goal-blind cheap estimator to reach the targets with less compute, and in a cell where adaptivity pays?

**What it is.** A validation-only design study with non-learned arms. It reopens D4-1 (a learned direct critic against a co-state-featured critic) only in a cell where its frozen rules hold, and it would then need its own plan.

**What it is not.** There is no learned critic, so it says nothing about H2. It is also not a test of the claim that a learned direct critic "has nothing to learn the weight from" when the goal varies: the non-learned `cheap` arm is goal-blind in D4-2 and in D4-3 alike. What changes is that the goal-aware weight is now averaged over 20 independent goals per cell.

## 2. Instances

- **Cells:** the seven D4-2 cells (`base`, `m4`, `m64`, `wide`, `amp1`, `depth7`, `depth11`), same systems `A`, same pulses and initial states as D4-2's tuning (seed 3003, 10 instances) and validation (seed 1001, 20 instances) families. **Only the goal changes:** `highdim.varying_goal_instances` draws a random unit goal per instance from its own stream, `default_rng(FAMILY_SEEDS[family] + 10000 + m)`. Names carry `d4_3`. The scorer-training family is not used. **The test family (seed 2002) is never generated** (`cell_instances` refuses it).
- Generated from frozen seeds, so **this is not real data.**

## 3. Arms and prices

All marking arms use Dörfler's rule; `θ` is chosen per (cell, arm) on the tuning family from `{0.5, 0.7, 0.85, 0.95, 0.98, 0.995}` (the D4-2 grid).

| Arm | What it scores | Price |
|---|---|---|
| `uniform_pass` | nothing | none |
| `residual` | `‖τ̂‖` by step doubling (goal-blind) | `2n` CN steps per pass |
| `goal_local` | `|cᵀ τ̂|` by step doubling | `2n` |
| `adjoint` | `|Λᵀ τ̂|` by step doubling and a backward sweep | `3n` |
| `cheap` | `‖d_j‖ + ‖e_j‖`: forcing discrepancy plus solution-difference estimate (goal-blind) | FLOPs converted to CN steps, as in D4-2 |
| `cheap_adjoint` | the same two estimates weighted by the tabulated continuous co-state of **this instance's goal** | FLOPs, including two co-state dot products per interval; **the table is taken as given** |
| `cheap_adjoint_setup` | `cheap_adjoint` with the table's setup charged to each instance (`n_fine · 2m²` FLOPs, finest grid) | derived from the `cheap_adjoint` traces by adding the per-instance setup |
| `adjoint_free` | the `adjoint` arm with scoring priced at zero | **hypothetical ceiling**, read off the same traces |

`cheap_adjoint` answers the question "does the weight matter if it is available at lookup price", which is the premise of amortisation. `cheap_adjoint_setup` is the real price when every instance has its own goal and the table is built on the finest grid; it is a deliberately pessimistic upper bound (a co-state sweep on the current mesh is cheaper, and is what `adjoint` charges) and is reported, not gated.

## 4. Measure

Compute to reach 10 %, 3 % and 1 % of the initial-to-finest objective gap, priced by **log-log interpolation between passes (primary)**, with semilog interpolation and the staircase as sensitivities computed from the same traces. The objective is D4's `Σ|Λᵀ τ|` with each instance's own goal. Ratios are geometric means with an instance-bootstrap CI (10,000 resamples) of the mean log ratio. The compute cap is twice the finest grid, and targets not reached count as the cap (censored counts are reported).

## 5. Protocol

1. Generate the tuning instances; check that they equal D4-2's tuning instances except for goal and name, and that goals are distinct unit vectors.
2. Correctness and ledger checks (before anything is read): the FLOP formulas equal the operations counted from the code path (as in D4-2); the tabulated co-state is linear in the goal and close to the discrete co-state on a fine mesh; `setup_steps` equals the price report's table setup; the derived `cheap_adjoint_setup` rows equal `cheap_adjoint` plus the setup.
3. Choose `θ` per (cell, arm) on the tuning family. **Freeze** `θ`, the rules and the pair list, and write their SHA-256 (`reports/frozen_before_validation.json`).
4. Only then generate the validation instances and run every arm.

## 6. Frozen rules

"Beats" means the CI of `mean(log(compute_a / compute_b))` lies below 0 at **two adjacent targets** (primary method log-log), as in D4-2.

- **R3v, the co-state weight helps a cheap estimator:** `cheap_adjoint` beats `cheap`.
- **R4v, adaptivity pays with a goal-aware cheap arm:** `cheap_adjoint` beats `uniform_pass`.
- **D4-1 candidate cell:** R3v and R4v both hold.
- Reported, not gated: `cheap_adjoint_setup` against `cheap` and `uniform_pass`; `goal_local` against `residual` (does goal-aware marking matter with step doubling); `adjoint_free` against `uniform_pass` and `cheap_adjoint`; per-instance log-ratio quantiles and win rates; the same rules recomputed under the semilog and staircase pricings; and the D4-2 fixed-goal ratio `cheap_adjoint ÷ cheap` per cell, quoted from D4-2's committed validation summary for comparison.

**Decision rules.** A candidate cell → write a D4-1 plan for that regime (learned direct critic, given the goal vector, against a critic given the co-state features; matched size and training; 5 paired seeds), frozen before any critic is trained. R3v holds but not R4v in every such cell → D4-1 stays closed (no regime where adaptivity pays), and the note says the weight helps only inside a losing regime. R3v fails in every cell → record "the co-state weight adds nothing to a cheap estimator in D4 even when it is free", D4-1 closed for good, and D4 is not used as evidence for H2.

## 7. Artefacts

`src/adjointrwm/domains/highdim.py` (`varying_goal_instances`, `setup_steps`, `with_setup_charged`, `cell_rules_varying_goal`), `tests/test_varying_goal.py`, `notebooks/04-domains/d4_3_varying_goal.ipynb` (CPU, no scorer training), run directory `results/runs/d4_3_varying_goal_<UTC>/` imported with a README and a clean-worktree reproduction, `scripts/d4_3_tables.py`, and a research note.

## 8. Risks and limits

- **The co-state is exact and the table is given** in the primary arm; its price is the open part. The real-price variant is a pessimistic bound, so a pass of `cheap_adjoint` and a fail of `cheap_adjoint_setup` would mean "worth having if amortised, not worth building per instance on the finest grid".
- **`m` is confounded with the system draw** in the `m4`, `base` and `m64` cells, as in D4-2.
- **The goal is a random unit vector in `R^m`.** Goals aligned with a few state directions (a physical observable) may behave differently. Not tested.
- **`base`, `depth7` and `depth11` are near copies of one configuration** and count as one finding.
- **Twenty validation instances per cell** give wide intervals on small effects; an effect of a few percent cannot be resolved.

## 9. Changes after this plan was written

_None yet. Each change will be dated here, and any change made after the validation family was read will be marked as such._
