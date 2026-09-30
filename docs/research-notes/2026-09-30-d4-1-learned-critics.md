# D4-1: a learned direct critic against a co-state critic (validation-only design study)

**Run:** `d4_1_learned_critics_20260930T122312Z` · **Date:** 2026-09-30 · **Notebook:** `notebooks/04-domains/d4_1_learned_critics.ipynb` (SHA-256 `0f15d7d9…fd700c`)
**Hardware:** CPU in the Claude Code sandbox (NumPy; FLOPs are counted, not measured; 2177 s) · **Package commit:** `9308b9f…` (clean tree) · **Config SHA-256:** `1e3c74c5…82a75` · **Frozen-before-validation SHA-256:** `8f25f902…44fc1`
**Artefacts:** [`results/runs/d4_1_learned_critics_20260930T122312Z/`](../../results/runs/d4_1_learned_critics_20260930T122312Z/) · **Plan:** [`docs/plans/d4-1-plan.md`](../plans/d4-1-plan.md) · **Follows:** [D4-3](2026-09-30-d4-3-varying-goal.md)

**Status:** exploratory design study on a train, a tuning and a validation family; **the test family was never generated.** Correctness ✅ · **R0 fails in both cells: no learned critic beats uniform refinement at the real price** (the critics need 6.1 to 10.3 times uniform's compute in `m4` and 1.5 to 1.8 times in `m64`) · **the co-state critic does not beat the information-equivalent direct critic at either price** (compute ratio 1.02 to 1.10 in `m4`, 0.96 to 1.04 in `m64`, no interval lies entirely below 1) and is indistinguishable from its randomised control (0.96 to 1.03) · **`m4`:** at the *hypothetical* lookup price both learned critics beat uniform refinement (0.58 to 0.88) and, at all but the tightest target for the co-state critic, the goal-blind `cheap` estimator (0.88 to 0.94), and the exact co-state as a feature helps the head by about 8 % (0.92 [0.86, 0.99] and 0.92 [0.86, 0.98] at the two looser targets), but the learned estimator (error 0.43 against the exact co-state, floor met in 4 of 5 seeds) does not deliver it: exit class **teacher-only value** (staircase pricing: inconclusive) · **`m64`:** the estimator never met its floor (0 of 5 seeds, error 0.655 to 0.684) and even the exact co-state features do not help there (1.09 [0.96, 1.27]): exit class **inconclusive** · no H2 statement about real data.

**In one paragraph.** With the goal varying per instance, a small learned critic that is given the goal and the cheap error estimates can allocate refinement, but at the price a neural network actually costs (about 21 to 23 CN steps per interval in `m4`, about 1.1 in `m64`, against 1.19 and 0.16 for D4-3's exact-table lookup, from the price ratios in its `acceptance_report.json`) it loses to plain uniform refinement in both cells, so the real-price gate of the plan fails as expected before the run. Priced hypothetically like the exact table, a *direct* critic in `m4` needs 0.67 of uniform's compute at the 3 % target and keeps 0.66 of the headroom between `cheap` and the exact-table arm (mean over seeds; range 0.12 to 1.05), so most of what the exact co-state buys in D4-3 is reachable by a direct critic that sees the goal. Adding features from a learned co-state estimator does not change that (co-state critic over direct 1.03 [0.94, 1.12] at the 3 % target, over its randomised control 1.03 [0.96, 1.10]); adding the *exact* co-state as a feature does help the head by about 8 % in `m4`. The learned estimator is the bottleneck: a generic two-layer network reached a relative error of 0.43 in `m4` and 0.67 in `m64` against the exact co-state (predicting zero scores 1), and the ten co-state estimators hit their 200-epoch cap in nine of ten runs.

---

## 1. What this run is

- **Domain and data:** the D4 domain (`y' = Ay + b(t)`, Crank–Nicolson, exact reference, objective `Σ|Λᵀτ|`) in the two cells D4-3 opened, `m4` and `m64` (systems, pulses and initial states as D4-2 and D4-3). Train family: seed 4004, 30 instances; tuning: seed 3003, 10 (D4-3's); validation: seed 1001, 20 (D4-3's), each instance with its own random unit goal. **The validation instances had been read before, for the non-learned arms of D4-3, which chose these cells; no learned critic had been evaluated on any of them.** Generated from frozen seeds; **this is not real data.**
- **Critics** (plan §3): a co-state estimator `ŵ(t_b, c)` (two tanh layers of width 32) trained on the continuous co-state at the interval's right node; value heads (two tanh layers) trained on the gain label `log10|Λ_{j+1}ᵀ τ_j|` (the objective's own summand, exact local error, discrete co-state). Arms: **D** `direct` (P0 inputs: nine scalar features, the goal, the unit directions of the two cheap error vectors), **C** `costate_critic` (D's inputs plus `[log10|ŵᵀd|, log10|ŵᵀe|, log10‖ŵ‖]` from the trained estimator), **R** `costate_randomised` (C with the estimator trained on co-state labels permuted across rows), **T** `teacher_feature` (C's head on the exact co-state; privileged, lookup price only). Five paired seeds; D's width is matched to C's head plus the estimator by parameter count (the plan's information-equivalence rule: D has at least as many parameters and never fewer inputs).
- **Width and price:** head width chosen on the tuning family at the lookup price with D as the yardstick (8 in both cells; capacity probe not triggered, so D is reported as saturated); prices are counted FLOPs in CN steps, at the **real price** (gated) and at a **hypothetical lookup price** (each critic charged what `cheap_adjoint` is), from the same traces.
- **Frozen rules** (plan §7, written to `reports/frozen_before_validation.json` at 12:52:33.4 UTC; the validation instances were written at 12:52:33.5 UTC): R0, RC1/RC1-L, RC2-L, RC3/RC3-L, RT-L, the estimator floor (held-out relative error ≤ 0.5 in at least four of five seeds), and the exit class; the interval is a two-stage bootstrap (seeds, then instances).
- **Design history (disclosed).** The plan was redesigned once before any validation instance was generated (plan §10): a first version scored intervals with a bilinear head; a smoke run on train and tuning instances (scratch, not kept) showed it fit the gain label worse than a flat network and that its joint loss distorted the co-state estimate, so the arms were replaced by the governing protocol's structure (estimator trained alone, heads receiving direct features plus co-state features). The estimator budget (200 epochs, batch 256) was fixed from a probe on synthetic rows (`scripts/d4_1_estimator_probe.py`, plan §10). A smoke run of the notebook on a reduced configuration read no validation instance as validation (train instances stood in). The tuning-only training tables were read before the validation stage; nothing was changed after them.

## 2. Gates

| Gate | Result | Source |
|---|---|---|
| Propagator equals the tabulated co-state (max 1.9e-13); discrete sweep equals `domain.costate` (1.5e-13); exact-co-state features equal the `cheap_adjoint` score terms; gain label equals the objective's own summand; `flops_step` equals the counted operations; train goals and names disjoint from tuning; test family refused | ✅ | `tests/correctness.json` |
| Freeze written before the validation family was generated; test family never generated | ✅ (file times above; `test_family_read: false`) | `reports/frozen_before_validation.json`, `config/run_config.json` |
| Critic FLOP formula equals a count from the weights (asserted for all 40 trained critics inside the notebook) | ✅ | notebook cell 4 |
| **R0: a learned critic beats uniform at the real price** | **❌ in both cells** | §3.3, §3.5 |
| **RC1: the co-state critic beats the direct critic (real price; RC1-L at the lookup price)** | **❌ / ❌ in both cells** | §3.3, §3.4 |
| RC3: the co-state critic beats its randomised control | ❌ (no difference) | §3.3, §3.4 |
| RT-L: the exact co-state as a feature helps the head (lookup price) | ✅ in `m4` (gap 10 % and 3 %), ❌ in `m64` | §3.4 |
| Estimator floor (≥ 4 of 5 seeds at held-out error ≤ 0.5) | ✅ in `m4` (4 of 5), ❌ in `m64` (0 of 5) | §3.2 |
| Exit class | `m4`: teacher-only value (semilog: same; staircase: inconclusive, precision). `m64`: inconclusive (all three pricings) | `reports/acceptance_report.json` |

## 3. Results

Every table below is printed by `python scripts/d4_1_tables.py results/runs/d4_1_learned_critics_20260930T122312Z` from the committed run files; the script re-derives the primary intervals and rule flags from the per-instance parquet and stops if they differ from the stored ones. Compute ratios are ratios of geometric-mean compute (below 1: the first arm needs less compute); intervals are two-stage bootstrap intervals over five seeds and 20 instances. **Every lookup-price row and the `teacher_feature` arm are hypothetical.**

### 3.1 Gate summary and width selection

| Cell | Head width | Direct saturated | Estimator floor met (seeds of 5) | R0 | RC1 | RC1-L | RC2-L | RC3 | RC3-L | RT-L | Exit class (primary; semilog; staircase) |
|---|---:|---|---:|---|---|---|---|---|---|---|---|
| m4 | 8 | True | 4 | False | False | False | False | False | False | True | teacher-only value; teacher-only value; inconclusive: precision |
| m64 | 8 | True | 0 | False | False | False | False | False | False | False | inconclusive: precision; inconclusive: precision; inconclusive: precision |

| Cell | Head width | Direct width | Direct compute (lookup price, θ = 0.95) | Co-state compute (lookup price) | Direct val RMSE | Co-state val RMSE |
|---|---:|---:|---:|---:|---:|---:|
| m4 | 8 | 30 | 567 | 605 | 0.458 | 0.495 |
| m4 | 16 | 35 | 594 | 619 | 0.462 | 0.467 |
| m4 | 32 | 46 | 590 | 628 | 0.480 | 0.464 |
| m64 | 8 | 30 | 543 | 567 | 0.698 | 0.741 |
| m64 | 16 | 37 | 535 | 512 | 0.675 | 0.676 |
| m64 | 32 | 51 | 550 | 560 | 0.666 | 0.686 |

### 3.2 Training: what the critics learned and what they cost per interval

| Cell | Arm | Head parameters | Estimator parameters | Head val RMSE (log10) | Within-state Spearman | Estimator error vs exact co-state (mean [min, max]) | Steps per interval | Lookup price as a fraction of it |
|---|---|---:|---:|---:|---:|---|---:|---:|
| m4 | direct | 1621 | 0 | 0.525 | 0.664 |  | 20.887 | 0.06 |
| m4 | costate_critic | 281 | 1380 | 0.524 | 0.630 | 0.432 [0.311, 0.798] | 22.667 | 0.05 |
| m4 | costate_randomised | 281 | 1380 | 0.564 | 0.633 | 1.001 [1.001, 1.002] | 22.667 | 0.05 |
| m4 | teacher_feature | 281 | 0 | 0.312 | 0.859 |  | 6.284 | 0.19 |
| m64 | direct | 7021 | 0 | 0.695 | 0.302 |  | 1.053 | 0.15 |
| m64 | costate_critic | 1721 | 5280 | 0.675 | 0.321 | 0.670 [0.655, 0.684] | 1.091 | 0.15 |
| m64 | costate_randomised | 1721 | 5280 | 0.711 | 0.308 | 1.001 [1.001, 1.001] | 1.091 | 0.15 |
| m64 | teacher_feature | 1721 | 0 | 0.441 | 0.714 |  | 0.456 | 0.36 |

### 3.3 Comparisons at the real price

| Comparison (ratio of geometric-mean compute, 95 % interval) | 10% of the gap left | 3% of the gap left | 1% of the gap left | Seeds favouring the first (gap3%) |
|---|---|---|---|---|
| costate_critic / direct | 1.07 [0.98, 1.14] | 1.10 [1.02, 1.18] | 1.10 [1.02, 1.19] | 1 of 5 |
| costate_critic / costate_randomised | 0.98 [0.91, 1.06] | 0.97 [0.89, 1.06] | 0.96 [0.88, 1.03] | 3 of 5 |
| costate_critic / uniform_pass | 10.30 [9.67, 11.00] | 8.00 [7.48, 8.61] | 6.77 [6.30, 7.34] | 0 of 5 |
| direct / uniform_pass | 9.64 [9.02, 10.35] | 7.26 [6.80, 7.76] | 6.14 [5.76, 6.58] | 0 of 5 |
| costate_critic / cheap | 10.46 [9.82, 11.12] | 10.68 [10.07, 11.37] | 10.53 [9.91, 11.22] | 0 of 5 |
| direct / cheap | 9.79 [9.11, 10.53] | 9.69 [9.10, 10.34] | 9.56 [8.95, 10.22] | 0 of 5 |
| costate_critic / cheap_adjoint | 12.55 [11.86, 13.31] | 12.61 [11.87, 13.42] | 11.82 [10.91, 12.79] | 0 of 5 |
| direct / cheap_adjoint | 11.75 [11.10, 12.53] | 11.44 [10.81, 12.13] | 10.73 [9.89, 11.63] | 0 of 5 |
| costate_critic / adjoint_free | 23.31 [21.81, 24.93] | 23.86 [22.29, 25.61] | 22.00 [20.21, 23.89] | 0 of 5 |
| direct / adjoint_free | 21.82 [20.35, 23.54] | 21.65 [20.25, 23.21] | 19.97 [18.31, 21.75] | 0 of 5 |

| Comparison (ratio of geometric-mean compute, 95 % interval) | 10% of the gap left | 3% of the gap left | 1% of the gap left | Seeds favouring the first (gap3%) |
|---|---|---|---|---|
| costate_critic / direct | 0.96 [0.86, 1.10] | 1.02 [0.90, 1.15] | 1.00 [0.87, 1.15] | 3 of 5 |
| costate_critic / costate_randomised | 0.98 [0.91, 1.04] | 0.96 [0.90, 1.02] | 0.97 [0.91, 1.02] | 4 of 5 |
| costate_critic / uniform_pass | 1.76 [1.62, 1.91] | 1.59 [1.43, 1.77] | 1.47 [1.30, 1.66] | 0 of 5 |
| direct / uniform_pass | 1.82 [1.64, 2.04] | 1.56 [1.44, 1.70] | 1.46 [1.32, 1.62] | 0 of 5 |
| costate_critic / cheap | 1.80 [1.64, 1.99] | 2.03 [1.83, 2.26] | 2.04 [1.82, 2.31] | 0 of 5 |
| direct / cheap | 1.87 [1.67, 2.11] | 1.99 [1.83, 2.19] | 2.03 [1.83, 2.27] | 0 of 5 |
| costate_critic / cheap_adjoint | 1.96 [1.80, 2.15] | 2.32 [2.09, 2.59] | 2.48 [2.20, 2.80] | 0 of 5 |
| direct / cheap_adjoint | 2.03 [1.82, 2.29] | 2.28 [2.10, 2.51] | 2.47 [2.23, 2.76] | 0 of 5 |
| costate_critic / adjoint_free | 2.02 [1.84, 2.22] | 2.18 [1.95, 2.44] | 2.21 [1.96, 2.52] | 0 of 5 |
| direct / adjoint_free | 2.09 [1.86, 2.36] | 2.14 [1.95, 2.37] | 2.21 [1.97, 2.47] | 0 of 5 |

### 3.4 Comparisons at the hypothetical lookup price

| Comparison (ratio of geometric-mean compute, 95 % interval) | 10% of the gap left | 3% of the gap left | 1% of the gap left | Seeds favouring the first (gap3%) |
|---|---|---|---|---|
| costate_critic / direct | 1.02 [0.90, 1.11] | 1.03 [0.94, 1.12] | 1.05 [0.97, 1.13] | 1 of 5 |
| costate_critic / costate_randomised | 1.02 [0.93, 1.10] | 1.03 [0.96, 1.10] | 1.03 [0.97, 1.10] | 1 of 5 |
| teacher_feature / direct | 0.92 [0.86, 0.99] | 0.92 [0.86, 0.98] | 0.95 [0.89, 1.01] | 4 of 5 |
| costate_critic / uniform_pass | 0.88 [0.82, 0.94] | 0.70 [0.65, 0.75] | 0.61 [0.56, 0.66] | 5 of 5 |
| direct / uniform_pass | 0.86 [0.80, 0.95] | 0.67 [0.62, 0.73] | 0.58 [0.54, 0.62] | 5 of 5 |
| costate_critic / cheap | 0.89 [0.84, 0.95] | 0.93 [0.88, 0.98] | 0.94 [0.89, 1.00] | 5 of 5 |
| direct / cheap | 0.88 [0.82, 0.96] | 0.90 [0.84, 0.96] | 0.90 [0.85, 0.95] | 5 of 5 |
| costate_critic / cheap_adjoint | 1.07 [1.01, 1.13] | 1.10 [1.03, 1.16] | 1.06 [0.98, 1.14] | 0 of 5 |
| direct / cheap_adjoint | 1.05 [0.99, 1.15] | 1.06 [1.00, 1.13] | 1.01 [0.94, 1.08] | 1 of 5 |
| teacher_feature / cheap_adjoint | 0.97 [0.94, 1.01] | 0.98 [0.95, 1.00] | 0.95 [0.90, 1.01] | 5 of 5 |
| teacher_feature / uniform_pass | 0.80 [0.76, 0.85] | 0.62 [0.59, 0.66] | 0.55 [0.51, 0.59] | 5 of 5 |
| costate_critic / adjoint_free | 1.99 [1.86, 2.11] | 2.07 [1.95, 2.21] | 1.97 [1.82, 2.12] | 0 of 5 |
| direct / adjoint_free | 1.96 [1.83, 2.14] | 2.00 [1.89, 2.15] | 1.88 [1.74, 2.02] | 0 of 5 |

| Comparison (ratio of geometric-mean compute, 95 % interval) | 10% of the gap left | 3% of the gap left | 1% of the gap left | Seeds favouring the first (gap3%) |
|---|---|---|---|---|
| costate_critic / direct | 1.03 [0.91, 1.16] | 1.04 [0.92, 1.20] | 1.02 [0.88, 1.17] | 2 of 5 |
| costate_critic / costate_randomised | 1.03 [0.94, 1.11] | 0.99 [0.90, 1.06] | 0.99 [0.91, 1.06] | 2 of 5 |
| teacher_feature / direct | 1.09 [0.96, 1.27] | 1.08 [0.95, 1.22] | 1.01 [0.87, 1.16] | 2 of 5 |
| costate_critic / uniform_pass | 1.25 [1.11, 1.43] | 1.08 [0.96, 1.22] | 0.98 [0.88, 1.11] | 1 of 5 |
| direct / uniform_pass | 1.21 [1.12, 1.33] | 1.03 [0.95, 1.13] | 0.96 [0.87, 1.07] | 2 of 5 |
| costate_critic / cheap | 1.28 [1.13, 1.47] | 1.38 [1.23, 1.55] | 1.37 [1.23, 1.53] | 0 of 5 |
| direct / cheap | 1.25 [1.13, 1.38] | 1.32 [1.21, 1.45] | 1.34 [1.21, 1.49] | 0 of 5 |
| costate_critic / cheap_adjoint | 1.40 [1.24, 1.60] | 1.58 [1.41, 1.78] | 1.66 [1.49, 1.85] | 0 of 5 |
| direct / cheap_adjoint | 1.36 [1.24, 1.50] | 1.51 [1.39, 1.67] | 1.63 [1.47, 1.82] | 0 of 5 |
| teacher_feature / cheap_adjoint | 1.48 [1.33, 1.68] | 1.63 [1.46, 1.86] | 1.65 [1.47, 1.88] | 0 of 5 |
| teacher_feature / uniform_pass | 1.33 [1.19, 1.50] | 1.11 [0.99, 1.28] | 0.97 [0.86, 1.12] | 1 of 5 |
| costate_critic / adjoint_free | 1.44 [1.27, 1.65] | 1.48 [1.32, 1.68] | 1.48 [1.32, 1.67] | 0 of 5 |
| direct / adjoint_free | 1.39 [1.27, 1.55] | 1.42 [1.29, 1.57] | 1.46 [1.31, 1.63] | 0 of 5 |

### 3.5 Compute, retained headroom and break-even instances

| Cell | Arm | geometric-mean compute, 10% left | geometric-mean compute, 3% left | geometric-mean compute, 1% left | Ratio to uniform (gap3%) | Retained fraction of the cheap → cheap_adjoint headroom (gap3%, mean [min, max] over seeds) | Break-even instances (gap3%) |
|---|---|---:|---:|---:|---:|---|---:|
| m4 | direct | 3467 | 5827 | 8923 | 7.26 | -13.68 [-14.04, -13.27] |  |
| m4 | costate_critic | 3703 | 6422 | 9830 | 8.00 | -14.27 [-14.68, -13.87] |  |
| m4 | costate_randomised | 3769 | 6606 | 10267 | 8.23 | -14.44 [-14.93, -13.96] |  |
| m64 | direct | 479 | 960 | 1687 | 1.56 | -5.07 [-5.61, -4.57] |  |
| m64 | costate_critic | 462 | 978 | 1694 | 1.59 | -5.20 [-6.07, -4.34] |  |
| m64 | costate_randomised | 472 | 1016 | 1754 | 1.65 | -5.49 [-6.87, -4.50] |  |

| Cell | Arm | geometric-mean compute, 10% left | geometric-mean compute, 3% left | geometric-mean compute, 1% left | Ratio to uniform (gap3%) | Retained fraction of the cheap → cheap_adjoint headroom (gap3%, mean [min, max] over seeds) | Break-even instances (gap3%) |
|---|---|---:|---:|---:|---:|---|---:|
| m4 | direct | 311 | 539 | 838 | 0.67 | 0.66 [0.12, 1.05] | 356849 |
| m4 | costate_critic | 316 | 558 | 879 | 0.70 | 0.45 [0.06, 0.92] | 1893822 |
| m4 | costate_randomised | 311 | 542 | 853 | 0.68 | 0.62 [0.35, 0.82] | 741483 |
| m4 | teacher_feature | 287 | 497 | 793 | 0.62 | 1.14 [1.03, 1.22] | 58758 |
| m64 | direct | 319 | 637 | 1113 | 1.03 | -2.04 [-2.59, -1.50] | 82046 |
| m64 | costate_critic | 329 | 665 | 1133 | 1.08 | -2.36 [-3.62, -1.44] | 4843254 |
| m64 | costate_randomised | 321 | 674 | 1149 | 1.10 | -2.47 [-3.69, -1.36] | 114186 |
| m64 | teacher_feature | 349 | 685 | 1126 | 1.11 | -2.58 [-4.15, -1.69] | 50850 |

### 3.6 References

| Cell | Reference | median compute, 10% left | median compute, 3% left | median compute, 1% left |
|---|---|---:|---:|---:|
| m4 | uniform_pass | 383 | 795 | 1397 |
| m4 | cheap | 313 | 561 | 952 |
| m4 | cheap_adjoint | 295 | 486 | 741 |
| m4 | adjoint | 575 | 931 | 1499 |
| m4 | adjoint_free | 166 | 267 | 465 |
| m64 | uniform_pass | 271 | 600 | 1185 |
| m64 | cheap | 241 | 459 | 801 |
| m64 | cheap_adjoint | 234 | 407 | 636 |
| m64 | adjoint | 622 | 1198 | 2007 |
| m64 | adjoint_free | 217 | 416 | 729 |

## 4. What the run shows

- **At the real price no learned critic pays** (R0 fails). The critics' inputs grow with the dimension and the network is priced in FLOPs: 20.9 (D) and 22.7 (C) CN steps per interval in `m4` and 1.05 and 1.09 in `m64`, so each scoring pass costs more than step doubling's 2 steps per interval in `m4` and about half of it in `m64`. The expectation stated in plan §4 before the run held.
- **The learned co-state critic gives no measurable advantage over the direct critic** at either price, in either cell. In `m4` at the real price it is worse (1.10 [1.02, 1.18] at the 3 % target) which is consistent with its 8 % higher price per interval; at the lookup price the ratio is 1.03 [0.94, 1.12], and only one seed of five favours it. It is indistinguishable from the control whose estimator was trained on permuted labels (1.03 [0.96, 1.10]), so nothing in the run attributes any effect to the estimator's co-state.
- **The exact co-state, as a feature, helps the head in `m4`** (T over D: 0.92 [0.86, 0.99], 0.92 [0.86, 0.98], 0.95 [0.89, 1.01]; four of five seeds favour T at the 3 % target) and lifts the head's within-state rank correlation with the gain label from 0.66 to 0.86 (validation, seeds pooled). The teacher arm at the lookup price needs 0.62 of uniform's compute at the 3 % target, close to the exact-table arm's 0.63 (D4-3, `results/runs/d4_3_varying_goal_20260930T084725Z/`). This is the plan's *teacher-only value*: the co-state has value here, a learned estimator at this budget does not deliver it.
- **A direct critic that sees the goal captures most of the exact table's advantage in `m4`, hypothetically:** at the lookup price D needs 0.67 [0.62, 0.73] of uniform's compute at the 3 % target and keeps 0.66 of the `cheap` → exact-table headroom (mean over seeds; the range 0.12 to 1.05 shows how variable that is). The equivalence rule RC2-L (within ±5 % of the direct critic at all three targets) is **not** met: the intervals are about ±8 to 10 % wide, so "direct utility sufficient" is not established either.
- **In `m64` nothing helps.** The estimator error is 0.655 to 0.684 in every seed, the exact features do not help the head (1.09 [0.96, 1.27]), no critic beats uniform at the lookup price (ratios 0.96 to 1.33; every interval at the tightest target includes 1), and the retained fraction of the exact-table headroom is negative (−2.0 to −2.6). With a within-state rank correlation of only 0.30 to 0.32 between head and label (0.71 with the exact features), the head barely orders intervals better than the cheap features do.

## 5. What the run does not show

- It does not show that a *better* co-state estimator would help. The ten generic co-state estimators hit their 200-epoch cap in nine of ten runs; the probe recorded in plan §10 shows a wider estimator (width 64 instead of 32, same 200 epochs) lowering the error somewhat (`m4` 0.37 to 0.34, `m64` 0.66 to 0.61 on synthetic rows) but not below the floor in `m64`. A structured estimator (linear in the goal) would cost work of order `m²` per interval and was not tried.
- It does not show that the direct critic is sufficient (RC2-L fails on precision), nor that the co-state critic loses by more than the noise: the point estimates favour the direct critic by 3 to 10 % in `m4` and are within ±4 % in `m64`.
- It says nothing about real data or about H2 there: the system is linear, its co-state is linear in the goal, the loss is exact, and the lookup price is a construction.
- Training is not priced in the gated comparison and is very large against per-instance compute: about 5.2e8 CN steps for the `m4` co-state critic (estimator and head), against several hundred CN steps per instance (the reference arms' medians in §3.6) (§3.5's break-even counts are between 5e4 and 5e6 instances at the lookup price and undefined at the real price, where no critic saves compute).

## 6. Hypotheses (not tested here)

1. *The 200-epoch cap, not the architecture, limits the estimator in `m4`.* **Diagnostic:** train the estimator to a plateau on the same rows and repeat RC1-L; it needs a plan and a frozen estimator budget, and the teacher gap (about 8 %) bounds what it could gain.
2. *At `m64` the co-state operator is out of reach of a small generic network.* **Diagnostic:** an estimator that is linear in the goal (a matrix-valued function of time); it costs order `m²` per interval, so at the real price it would have to beat `cheap_adjoint`'s table lookup, which D4-3 shows costs more to build per instance than it saves.
3. *The head's rank correlation is the limit in `m64`* (0.30 to 0.32 with the cheap features, 0.71 with the exact co-state): richer inputs (for example the local solution, not just norms and directions of two estimates) may raise it for both critics alike.

## 7. Reproduce

`python scripts/d4_1_tables.py results/runs/d4_1_learned_critics_20260930T122312Z` reprints every table. The notebook `notebooks/04-domains/d4_1_learned_critics.ipynb` runs end to end on CPU in about 36 minutes (NumPy, no data files); the run's README records a re-execution from a clean worktree.

## 8. Recommendation

D4 has now answered on CPU what it can. It gives no support for H2 in either cell at the real price, a small teacher-only effect in `m4` at a hypothetical price, and nothing in `m64`. **I recommend not spending the test family and not planning a further D4 critic study unless Roman wants the structured-estimator question answered** (hypothesis 2); the open evidence for H2 is on real data (E1.1 and B2, Colab, L4), which is where the time should go.
