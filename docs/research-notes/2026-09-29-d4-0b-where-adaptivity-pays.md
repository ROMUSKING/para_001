# D4-0b: where does adaptive allocation pay at equal compute? (validation-only design study)

**Run:** `d4_0b_adaptivity_20260929T174347Z` (result) and `d4_0b_adaptivity_20260929T173428Z` (superseded first execution) · **Date:** 2026-09-29 · **Notebook:** `notebooks/04-domains/d4_0b_where_adaptivity_pays.ipynb` (SHA-256 `747f7f7c…97783`)
**Hardware:** CPU in the Claude Code sandbox (timings are not a systems claim) · **Package commit:** `b9fd11dc934f7aed896b031e9a3eddc3d25f290d` (clean tree) · **Config SHA-256:** `c82fbd69…4dda9f`
**Artefacts:** [`results/runs/d4_0b_adaptivity_20260929T174347Z/`](../../results/runs/d4_0b_adaptivity_20260929T174347Z/), [`…T173428Z/`](../../results/runs/d4_0b_adaptivity_20260929T173428Z/) · **Plan:** [`docs/plans/cross-domain-plan.md`](../plans/cross-domain-plan.md) §5 · **Follows:** [D4-0](2026-09-29-d4-0-adaptive-time-stepping.md)

**Status:** exploratory design study on a tuning and a validation family; **the test family was never generated or read.** Correctness ✅ · frozen rule: **2 of 9 cells are candidate regimes, both at hypothetical scoring prices (sharp ×0.25 and ×0), none at the real ledger (×1)** · no learned arm, so **no H2 statement.**

**In one paragraph.** Letting a policy refine many intervals per scoring pass does not rescue adaptive allocation at the real scoring price on the `smooth` family: uniform refinement still needs 2.2–3.0 times less compute than the co-state, residual or goal-local policy. Adaptive allocation pays at the real price only where the features are very localised (`sharper`, pulse widths 0.0001–0.0004), and there the cheap unweighted residual score pays at least as well as the co-state score in point estimate, so the co-state shows no benefit at the real price. The co-state weighting needs less compute than both cheap scores at two adjacent targets only when scoring is re-priced to a quarter or less of its real cost (×0.25 or ×0 in `sharp`, ×0 in `smooth`), and it beats uniform *and* both cheap scores together only in `sharp`. That is the regime a learned, amortised scorer is meant for, but no learned scorer exists in D4 yet, so its price is unmeasured. D4-1 therefore does not open at the real ledger.

---

## 1. What this run is

- **Domain and data:** the D4 domain of the D4-0 note (`y' = Ay + b(t)` on `[0, 1]`, Crank–Nicolson on a grid refined by bisection, exact matrix-exponential reference, exact discrete co-state `Λ_j = M_jᵀ Λ_{j+1}` computed from the known model). Instances are generated from frozen seeds; **this is not real data.** Tuning family: seed 3003, 10 instances per family. Validation family: seed 1001, 20 per family. Three families that differ in pulse width (with amplitude scale and grid depth changing alongside): `smooth` (0.01–0.04, the D4-0 family, finest grid 2,048 steps), `sharp` (0.001–0.004, 32,768) and `sharper` (0.0001–0.0004, 262,144). Amplitudes are scaled ×1, ×10 and ×100 so the forcing impulses stay comparable.
- **Objective:** the cancellation-free bound `Σ|Λᵀτ|` (D4-0 note §1).
- **Pass semantics (new).** A *pass* scores once, refines a set of `k` intervals and re-solves once from the leftmost refined interval, costing `n + k − j_min` CN steps. Scoring is charged per pass: `2n` steps for the step-doubling estimate (`residual`, `goal_local`), `3n` with the co-state sweep (`adjoint`); `uniform_pass` (refine every longest interval) is not scored.
- **Policies (all deployable):** `uniform_pass`; and Dörfler marking, the smallest set of intervals whose score covers a fraction `θ` of the total, on three scores that share the same `τ̂`: `residual` (`‖τ̂‖`), `goal_local` (`|cᵀτ̂|`) and `adjoint` (`|Λᵀτ̂|`). `θ ∈ {0.3, 0.5, 0.7, 0.85, 0.95}` is chosen per (family, policy) on the tuning family.
- **Measure:** compute to close a fraction of the gap between the initial and the finest-grid objective, with targets at 10 %, 3 % and 1 % of the gap remaining, one compute value per instance and policy. Ratios are geometric means over instances with a bootstrap CI over instances (10,000 resamples) of the mean log ratio. **No policy was censored in any cell** (`fraction_reached` is 1 everywhere in `artifacts/validation_summary.json`), so the cap of 2 finest grids never bound.
- **Scoring-price what-ifs.** Decisions never depend on costs, so the same traces are re-priced with the scoring cost scaled by ×1 (the real ledger), ×0.25 and ×0. **Scales below 1 are hypothetical**; scale `s` is equivalent to solves being `1/s` times as expensive relative to scoring, and ×0 is what allocation quality alone would buy. The co-state here is computed exactly from the known model, so ×0 gives the co-state policy its information for free.
- **Frozen decision rule** (in the notebook before the first execution's validation numbers were read, and unchanged since): a cell (family, scoring price) is a candidate regime for D4-1 if the co-state policy needs less compute than `uniform_pass` **and** than both `residual` and `goal_local`, each meaning the CI of the mean log compute ratio lies below 0 at two adjacent targets.
- **Design history (disclosed).**
  - A first execution used `θ ∈ {0.3, 0.5, 0.7}`. Its tuning table showed the top of that grid selected for all nine (family, policy) pairs, with cost still falling, so the grid was extended to 0.85 and 0.95 with nothing else changed. **The first execution's validation results had been printed and seen before that change.** The first execution is kept as `d4_0b_adaptivity_20260929T173428Z`; in it 1 of 9 cells met the rule (sharp ×0).
  - The `sharp` and `sharper` families were added *because* theory says adaptivity pays only on localised features and D4-0 showed no payoff on `smooth`. The study says where adaptivity pays, not that it does in general.
  - The interpolation check in §3.3 was designed **after** the frozen-rule results were read.

## 2. Gates

| Gate | Result | Source |
|---|---|---|
| Batch application equals sequential application | ✅ | `tests/correctness.json` |
| Error representation `cᵀ(y(T) − y_N) = Σ Λᵀτ` on batch grids | ✅ max abs difference 8.88e-16 | `tests/correctness.json` |
| Reference solution converged | ✅ max abs difference 3.76e-11 | `tests/correctness.json` |
| Test family unread | ✅ `test_family_read: false` | `reports/acceptance_report.json` |
| Frozen rule: candidate regimes for D4-1 | **2 of 9 cells, both hypothetical prices** | `reports/acceptance_report.json` |
| `θ` selected at the grid edge | ⚠️ **all 9 pairs at 0.95, the top of the grid** | `artifacts/theta_tuning.csv` |

## 3. Results

Ratios are compute ratios (below 1 means the first policy needs less). Every table below is printed by `scripts/d4_0b_tables.py` from the committed run files (§6).

### 3.1 Marking fraction (tuning family)

| Family | Policy | θ = 0.3 | θ = 0.5 | θ = 0.7 | θ = 0.85 | θ = 0.95 | Selected | At grid edge |
|---|---|---:|---:|---:|---:|---:|---:|---|
| smooth | residual | 1754 | 1057 | 790 | 587 | 511 | 0.95 | True |
| smooth | goal_local | 2033 | 1271 | 912 | 773 | 655 | 0.95 | True |
| smooth | adjoint | 1829 | 1154 | 829 | 657 | 599 | 0.95 | True |
| sharp | residual | 2754 | 1798 | 1407 | 1179 | 1138 | 0.95 | True |
| sharp | goal_local | 3342 | 2144 | 1588 | 1400 | 1289 | 0.95 | True |
| sharp | adjoint | 3663 | 2630 | 2041 | 1576 | 1314 | 0.95 | True |
| sharper | residual | 76946 | 16807 | 7724 | 6979 | 6815 | 0.95 | True |
| sharper | goal_local | 98319 | 22257 | 10033 | 8470 | 6592 | 0.95 | True |
| sharper | adjoint | 167836 | 42452 | 12273 | 11657 | 10318 | 0.95 | True |

In all nine pairs the geometric-mean compute falls at every step of the grid, and the best value is the largest one tried. Selective marking (`θ` well below 1) did not pay in any family; the tuned policies mark nearly everything that carries error. The true optimum may lie beyond 0.95 (θ = 1 degenerates to marking every interval with a non-zero score); that was not tried.

### 3.2 Real ledger (scoring price ×1), validation family

| Family | Target (gap left) | Median compute: uniform / residual / goal-local / co-state | Co-state ÷ uniform [95 % CI] | Residual ÷ uniform [95 % CI] | Co-state ÷ residual [95 % CI] |
|---|---|---|---|---|---|
| smooth | 10% | 96 / 331 / 352 / 224 | 2.63 [2.26, 3.09] | 2.55 [2.18, 3.02] | 1.03 [0.87, 1.20] |
| smooth | 3% | 224 / 674 / 692 / 494 | 2.50 [2.13, 2.97] | 2.45 [2.10, 2.84] | 1.02 [0.90, 1.14] |
| smooth | 1% | 480 / 983 / 1248 / 994 | 2.29 [1.94, 2.72] | 2.16 [1.83, 2.57] | 1.06 [0.91, 1.25] |
| sharp | 10% | 480 / 704 / 708 / 733 | 1.28 [1.07, 1.53] | 1.15 [0.94, 1.41] | 1.11 [0.97, 1.27] |
| sharp | 3% | 1504 / 1202 / 1309 / 1180 | 0.93 [0.78, 1.15] | 0.88 [0.73, 1.08] | 1.06 [0.93, 1.19] |
| sharp | 1% | 2016 / 1907 / 2050 / 1706 | 0.90 [0.75, 1.10] | 0.90 [0.75, 1.10] | 1.00 [0.89, 1.13] |
| sharper | 10% | 8160 / 4640 / 5322 / 4170 | 0.72 [0.52, 1.00] | 0.69 [0.53, 0.90] | 1.04 [0.82, 1.33] |
| sharper | 3% | 16352 / 5654 / 6874 / 6523 | 0.58 [0.40, 0.85] | 0.43 [0.34, 0.54] | 1.34 [0.97, 1.94] |
| sharper | 1% | 32736 / 7432 / 9435 / 9845 | 0.43 [0.31, 0.61] | 0.30 [0.24, 0.38] | 1.41 [1.05, 1.96] |

- **`smooth`:** `uniform_pass` needs 2.3–2.6 times less compute than the co-state policy and 2.2–2.6 times less than the residual policy, at every target (2.5–3.0 times less than goal-local marking, from `artifacts/validation_summary.json`). Batch marking did not fix the D4-0 result.
- **`sharp`:** uniform is better at the loosest target (co-state ÷ uniform 1.28 [1.07, 1.53]) and the CIs contain 1 at the two tighter targets (0.93, 0.90). No payoff.
- **`sharper`:** the co-state policy needs less compute than uniform at the two tighter targets (0.58 [0.40, 0.85], 0.43 [0.31, 0.61]) and is borderline at the loosest (0.72 [0.52, 1.00]). The **residual** policy pays more: 0.69 [0.53, 0.90], 0.43 [0.34, 0.54] and 0.30 [0.24, 0.38] of uniform's compute at the three targets, with every CI below 1.
- **Co-state against residual at the real price:** the point estimate is at or above 1 in every row (1.00–1.41), and the CI excludes 1 in only one row (`sharper`, 1 %: 1.41 [1.05, 1.96], the residual policy being cheaper). The co-state weighting is not better than the plain residual score at the real price in any row.

### 3.3 Robustness of the real-ledger reading to pass quantisation (post-hoc)

`uniform_pass` can only stop after a whole doubling of the grid, so its compute-to-target is a staircase that can overstate its cost by up to a factor 2, which flatters the adaptive policies. `scripts/d4_0b_interpolation.py` re-runs the validation family with the run's own `θ` and prices compute-to-target by linear interpolation of compute against `log(objective)` between the two bracketing passes, for every policy. It asserts that its targets equal the run's stored ones. It is not a realisable schedule; it is a sensitivity check written after the results above were read.

| Family | Target (gap left) | Median compute: uniform / residual / goal-local / co-state | Co-state ÷ uniform [95 % CI] | Residual ÷ uniform [95 % CI] | Co-state ÷ residual [95 % CI] |
|---|---|---|---|---|---|
| smooth | 10% | 81 / 181 / 200 / 201 | 2.58 [2.41, 2.76] | 2.32 [2.16, 2.51] | 1.11 [1.06, 1.16] |
| smooth | 3% | 178 / 397 / 422 / 438 | 2.50 [2.32, 2.70] | 2.34 [2.15, 2.56] | 1.07 [1.01, 1.14] |
| smooth | 1% | 334 / 744 / 806 / 825 | 2.44 [2.26, 2.62] | 2.32 [2.14, 2.53] | 1.05 [0.97, 1.14] |
| sharp | 10% | 433 / 579 / 603 / 579 | 1.52 [1.32, 1.81] | 1.33 [1.16, 1.54] | 1.15 [1.02, 1.29] |
| sharp | 3% | 961 / 943 / 990 / 948 | 1.18 [1.02, 1.40] | 1.09 [0.94, 1.27] | 1.09 [0.97, 1.20] |
| sharp | 1% | 1753 / 1394 / 1527 / 1444 | 0.98 [0.84, 1.17] | 0.92 [0.80, 1.08] | 1.06 [0.95, 1.17] |
| sharper | 10% | 5496 / 4237 / 5038 / 4081 | 0.91 [0.67, 1.26] | 0.83 [0.64, 1.09] | 1.09 [0.85, 1.39] |
| sharper | 3% | 10502 / 5475 / 6529 / 6061 | 0.77 [0.54, 1.13] | 0.57 [0.45, 0.73] | 1.34 [0.99, 1.91] |
| sharper | 1% | 18268 / 6919 / 8493 / 8798 | 0.58 [0.42, 0.81] | 0.40 [0.33, 0.50] | 1.42 [1.07, 1.97] |

- `smooth` is unchanged (uniform ahead by 2.4–2.6 times). In `sharp` the uniform policy is now clearly ahead at the two looser targets (co-state ÷ uniform 1.52 [1.32, 1.81], 1.18 [1.02, 1.40]) and tied at the tightest (0.98 [0.84, 1.17]).
- In `sharper` the co-state policy now beats uniform at the tightest target only (0.58 [0.42, 0.81]); its CIs contain 1 at the two looser ones. The residual policy still needs less compute than uniform at the two tighter targets (0.57 [0.45, 0.73] and 0.40 [0.33, 0.50], from `diagnostics/interpolated_summary.json`).
- **What survives:** at the real price, adaptivity pays only in `sharper` and only at the tighter targets, and the residual score does at least as well as the co-state score in point estimate (in the interpolated table the co-state ÷ residual CI excludes 1, in the residual's favour, in `smooth` 10 % and 3 %, `sharp` 10 % and `sharper` 1 %). The frozen rule for the real ledger (×1) is not met in any family either way.

The interpolation has a known bias. For a power-law error, a partial pass costs a geometric, not an arithmetic, share of the doubling; linear interpolation therefore overstates the compute of a fractional doubling pass by at most 6.1 % and by 4.1 % on average over the fractional position (§6). If convergence is close to a power law, this works against `uniform_pass` and toward the adaptive policies, by an amount small next to the effects above except in one cell (§3.4).

### 3.4 Hypothetical scoring prices, validation family

| Family | Scoring price | Target | Co-state ÷ uniform [95 % CI] | Residual ÷ uniform | Co-state ÷ residual [95 % CI] | Co-state ÷ goal-local [95 % CI] |
|---|---|---|---|---|---|---|
| smooth | ×0.25 | 10% | 1.36 [1.16, 1.61] | 1.54 | 0.88 [0.75, 1.03] | 0.74 [0.63, 0.86] |
| smooth | ×0.25 | 3% | 1.29 [1.09, 1.55] | 1.48 | 0.87 [0.77, 0.98] | 0.78 [0.68, 0.90] |
| smooth | ×0.25 | 1% | 1.19 [1.00, 1.43] | 1.31 | 0.91 [0.77, 1.07] | 0.80 [0.69, 0.93] |
| smooth | ×0 | 10% | 0.93 [0.79, 1.11] | 1.21 | 0.77 [0.65, 0.90] | 0.65 [0.55, 0.76] |
| smooth | ×0 | 3% | 0.89 [0.75, 1.07] | 1.16 | 0.77 [0.67, 0.86] | 0.69 [0.60, 0.80] |
| smooth | ×0 | 1% | 0.82 [0.69, 0.99] | 1.03 | 0.80 [0.68, 0.95] | 0.71 [0.61, 0.83] |
| sharp | ×0.25 | 10% | 0.58 [0.49, 0.70] | 0.64 | 0.91 [0.78, 1.05] | 0.83 [0.72, 0.95] |
| sharp | ×0.25 | 3% | 0.44 [0.37, 0.55] | 0.51 | 0.87 [0.76, 0.99] | 0.83 [0.72, 0.94] |
| sharp | ×0.25 | 1% | 0.44 [0.36, 0.53] | 0.52 | 0.83 [0.74, 0.95] | 0.78 [0.70, 0.88] |
| sharp | ×0 | 10% | 0.35 [0.29, 0.43] | 0.47 | 0.74 [0.62, 0.87] | 0.67 [0.57, 0.78] |
| sharp | ×0 | 3% | 0.28 [0.23, 0.35] | 0.38 | 0.73 [0.62, 0.84] | 0.69 [0.59, 0.80] |
| sharp | ×0 | 1% | 0.28 [0.23, 0.34] | 0.40 | 0.71 [0.62, 0.81] | 0.66 [0.59, 0.75] |
| sharper | ×0.25 | 10% | 0.29 [0.21, 0.41] | 0.34 | 0.86 [0.67, 1.10] | 0.75 [0.61, 0.92] |
| sharper | ×0.25 | 3% | 0.24 [0.16, 0.36] | 0.21 | 1.12 [0.81, 1.64] | 0.85 [0.64, 1.13] |
| sharper | ×0.25 | 1% | 0.18 [0.13, 0.26] | 0.15 | 1.19 [0.88, 1.67] | 0.88 [0.66, 1.17] |
| sharper | ×0 | 10% | 0.15 [0.11, 0.22] | 0.22 | 0.67 [0.52, 0.87] | 0.58 [0.47, 0.73] |
| sharper | ×0 | 3% | 0.12 [0.08, 0.19] | 0.14 | 0.89 [0.62, 1.33] | 0.65 [0.48, 0.88] |
| sharper | ×0 | 1% | 0.09 [0.07, 0.14] | 0.10 | 0.96 [0.69, 1.36] | 0.70 [0.51, 0.95] |

**Frozen decision rule (the run's own report):**

| Family | Scoring price | Co-state beats uniform | Co-state beats residual and goal-local | Candidate regime for D4-1 |
|---|---|---|---|---|
| smooth | ×1 | False | False | False |
| smooth | ×0.25 | False | False | False |
| smooth | ×0 | False | True | False |
| sharp | ×1 | False | False | False |
| sharp | ×0.25 | True | True | True |
| sharp | ×0 | True | True | True |
| sharper | ×1 | True | False | False |
| sharper | ×0.25 | True | False | False |
| sharper | ×0 | True | False | False |

- **Frozen result:** co-state marking beats uniform, residual and goal-local at two adjacent targets only in `sharp`, at ×0.25 and ×0. In `sharp` at ×0.25 the co-state ÷ residual ratios are 0.91 [0.78, 1.05], 0.87 [0.76, 0.99] and 0.83 [0.74, 0.95]; at ×0 they are 0.74, 0.73 and 0.71.
- **Headroom for a co-state over the cheap residual score is bounded:** across the two candidate cells the point estimates of co-state ÷ residual are 0.71–0.91 over the three targets, i.e. 9–29 % less compute, and that is with the exact co-state, computed from the known model, priced at a quarter or at nothing.
- **`sharper`:** the co-state policy beats uniform at every hypothetical price but never beats the residual policy at two adjacent targets (at ×0.25: 0.86, 1.12, 1.19), so the co-state has no demonstrated value over the residual score there.
- **`smooth` at ×0:** the co-state policy does beat the residual and goal-local scores at two adjacent targets (co-state ÷ residual 0.77 [0.65, 0.90], 0.77 [0.67, 0.86], 0.80 [0.68, 0.95]), but co-state ÷ uniform is 0.93 [0.79, 1.11], 0.89 [0.75, 1.07], 0.82 [0.69, 0.99], so the rule's first condition is not met and uniform refinement remains no more expensive.

**Interpolated re-application of the frozen rule.** With the pass-quantisation removed (`--interpolated`), the same rule gives:

| Family | Scoring price | Co-state beats uniform | Co-state beats residual and goal-local | Candidate regime for D4-1 |
|---|---|---|---|---|
| smooth | ×1 | False | False | False |
| smooth | ×0.25 | False | True | False |
| smooth | ×0 | True | True | True |
| sharp | ×1 | False | False | False |
| sharp | ×0.25 | True | True | True |
| sharp | ×0 | True | True | True |
| sharper | ×1 | False | False | False |
| sharper | ×0.25 | True | False | False |
| sharper | ×0 | True | False | False |

The candidate set is `sharp` ×0.25 and ×0 as before, plus `smooth` ×0, where co-state ÷ uniform is 0.91 [0.83, 1.00], 0.90 [0.81, 0.98] and 0.88 [0.80, 0.96]. That cell's saving of roughly 10 % is comparable in size to the interpolation bias in §3.3 (at most 6.1 % for the uniform policy), so it is **not** treated as a candidate; the frozen rule's own outcome (two `sharp` cells) stands, and the `smooth` ×0 cell is unresolved. The `sharper` ×1 cell drops out of "co-state beats uniform" under interpolation, as in §3.3.

## 4. Interpretation

**Observed** (from the tables above):

- Marking many intervals per pass does not make adaptive refinement pay at the real scoring price on the `smooth` family; the D4-0 result stands under the pass semantics.
- At the real price, adaptive refinement needs less compute than uniform only in the most localised family and at the tighter targets, and there the residual score is at least as cheap as the co-state score in point estimate.
- Re-pricing scoring to a quarter or less of the real price lets the co-state weighting beat the residual and goal-local scores at two adjacent targets in `sharp` (×0.25 and ×0) and in `smooth` (×0 only). Only in `sharp` does it also beat uniform refinement. In `sharp` the saving over the residual score is 9–29 % in point estimate (§3.4).
- Selective marking did not pay: every tuned `θ` is at the grid's top.

**Hypotheses (not tested here):**

1. *Localisation is what lets adaptivity pay.* The payoff of adaptive over uniform refinement appears and grows as pulses narrow (`smooth` → `sharp` → `sharper`). It is consistent with theory, but pulse width, amplitude scale and grid depth change together across the three families. **Diagnostic:** vary one factor at a time (D4-2).
2. *The co-state's information is not the bottleneck at this problem size; the price of getting it is.* At ×1 the residual score, which needs only the step-doubling estimate, does as well; the co-state pays only when its sweep is made cheap. **Diagnostic:** price a learned, amortised scorer at its measured cost in the same ledger (D4-2), instead of assuming a scale.
3. *A wider `θ` grid would help the adaptive policies.* Every tuned value is at the edge. **Diagnostic:** extend the grid (0.98, 0.99) on the tuning family only, then re-run the validation family; a second boundary optimum would suggest that marking-based adaptivity is close to uniform here.
4. *The `smooth` ×0 gain (≈ 10 % over uniform) is real but small.* **Diagnostic:** a power-law (log–log) interpolation, which would remove the bias in §3.3 if the error is close to a power law.

## 5. Design issues found

1. **`θ` is at the grid edge for all nine pairs (twice).** The first execution's grid ended at 0.7; the widened grid ends at 0.95 and the same thing happens. The adaptive policies may be understated; the direction of the bias is against them.
2. **`uniform_pass` compute-to-target is quantised** (up to 2×) in the frozen analysis, flattering the adaptive policies; §3.3 checks it post-hoc. The interpolated check has its own bias of at most 6.1 % against uniform.
3. **The first execution's validation results were seen before the grid was widened.** Disclosed in §1.
4. **The `sharp` and `sharper` families were designed after the D4-0 results were read.** They are a theory-motivated choice; the study says where adaptivity pays, not whether it does in general.
5. **Instance names repeat across families** (`d4_s<seed>_<i>`), so any code that joins on `instance` alone silently mixes families. The post-hoc script asserted on targets and caught this once; it now filters by family.
6. **The co-state is exact and free of learning error.** Priced at ×0.25 or ×0 it is a ceiling for a learned co-state, not a prediction of one. A learned scorer also has its own error, training cost and price.
7. **The comparison is not information-matched.** The co-state policy uses the known operator `M_j`, which `residual` and `goal_local` do not. That is the point of the co-state, but it means a win is a statement about privileged information being cheap to obtain, which a learned critic would have to replicate.
8. **Twenty validation instances per family** give wide CIs (for example 0.43 [0.31, 0.61] in `sharper`).

## 6. Commands

Every table in §3 is printed from committed files:

```bash
python scripts/d4_0b_tables.py results/runs/d4_0b_adaptivity_20260929T174347Z                  # frozen analysis
python scripts/d4_0b_tables.py results/runs/d4_0b_adaptivity_20260929T174347Z --interpolated   # post-hoc check
```

The interpolated summary was produced by `python scripts/d4_0b_interpolation.py results/runs/d4_0b_adaptivity_20260929T174347Z` (run README). The residual-over-uniform ratios quoted in §3.3 are in the interpolated table above and in that file (`differences["residual / uniform_pass"]`, exponentiated for the CI).

The interpolation-bias figures (6.1 % at most, 4.1 % on average over the fractional position) come from:

```bash
python - <<'EOF'
import numpy as np
f = np.linspace(0, 1, 100001)
r = (1 + f) / 2 ** f   # linear-interpolated compute over power-law compute for a doubling pass
print(r.max() - 1, r.mean() - 1)
EOF
```

The notebook re-executed from a clean worktree at the same commit reproduced every result file, byte-identical apart from the run identifier (run README).

## 7. What this run does and does not support

**Supports:**

- On this analytic family and ledger, at the real scoring price, batch marking does not make adaptive refinement cheaper than uniform refinement on `smooth` or `sharp`, and pays only in `sharper` at the tighter targets, where the cheap residual score does at least as well as the co-state score in point estimate.
- The co-state weighting beats the unweighted and goal-projected scores, together with uniform refinement, only in `sharp` and only if scoring costs a quarter or less of its real price, by 9–29 % over the residual score in point estimate (§3.4). That bounds the headroom for a co-state contribution in D4, with the exact co-state.
- The batch layer is correct (`tests/correctness.json`), and the frozen analysis was reproduced from a clean checkout (result files byte-identical apart from the run identifier; run README).

**Does not support:**

- Any statement that the adjoint helps or hurts relative to a matched direct critic (H2). No learned arm exists in D4.
- Any statement about a learned scorer's price. Scales ×0.25 and ×0 are hypothetical.
- Any test-family statement: the test family was not read.
- Any statement that adaptive allocation cannot pay in the `smooth` family for another scoring scheme; only the schemes in §1 were tried.
- Any statement about other domains, real data or cross-domain transfer.

**Corrects earlier statements.**

- The D4-0 note's next steps say that if a compute-efficient scheme fails, a harder family should be tried. Both were done here; pass semantics alone did not rescue `smooth`, and the harder families are where adaptivity pays at all.
- The roadmap row for D4-1 said it opens when D4-0b passes the equal-compute condition. D4-0b's conjunction (beat uniform *and* both cheap scores) is met only at hypothetical scoring prices; at the real ledger the residual policy beats uniform in `sharper`, and the co-state does not beat the residual. D4-1 does not open at the real ledger.

## 8. Next steps (roadmap IDs in `docs/plans/roadmap.md`)

1. **D4-2 (CPU, new):** price a learned, amortised scorer at its measured cost in the same ledger, on a higher-dimensional system, where a solve should cost more relative to a learned scorer (to be checked, not assumed), with one factor varied at a time (hypothesis 1), and a wider `θ` grid tuned on the tuning family only (hypothesis 3). Freeze the regime and the price before reading validation.
2. **D4-1** stays closed at the real ledger. It may open for `sharp` at a scoring price that D4-2 has measured to be at most ×0.25, with the co-state-featured critic compared to a direct critic at matched compute.
3. **The main H2 test is on real data:** `E1.1` (opportunity audit on the pilot checkpoint) then `B2` (pilot v2, five paired seeds). D4 informs the design but cannot replace it.
4. **DL** (licence survey) and the other domains stay deferred, per the D4-first decision.
