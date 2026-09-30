# D4-3: the varying-goal check (validation-only design study)

**Run:** `d4_3_varying_goal_20260930T084725Z` · **Date:** 2026-09-30 · **Notebook:** `notebooks/04-domains/d4_3_varying_goal.ipynb` (SHA-256 `58bbc65f…a60f`)
**Hardware:** CPU in the Claude Code sandbox (FLOPs are counted, not measured; timings are not a systems claim) · **Package commit:** `35c2daa9…` (clean tree) · **Config SHA-256:** `cfe1545d…75457` · **Frozen-before-validation SHA-256:** `6a66d322…f2fa`
**Artefacts:** [`results/runs/d4_3_varying_goal_20260930T084725Z/`](../../results/runs/d4_3_varying_goal_20260930T084725Z/) · **Plan:** [`docs/plans/d4-3-plan.md`](../plans/d4-3-plan.md) · **Follows:** [D4-2](2026-09-30-d4-2-flop-priced-scoring.md) (hypothesis 2, next step 2)

**Status:** exploratory design study on a tuning and a validation family; **the test family was never generated or read.** Correctness ✅ · frozen rules: **R3v (the co-state weight helps a goal-blind cheap estimator) holds in 2 of 7 cells (`m4`, `m64`), R4v (adaptivity pays with the goal-aware cheap arm) in 5 of 7, and both hold in `m4` and `m64`: two D4-1 candidate cells, so D4-1 reopens there** (it needs its own plan) · with the co-state table charged per instance the goal-aware arm loses everywhere (2.1 to 50 times `cheap`'s compute) · no learned critic, so **no H2 statement.**

**In one paragraph.** Giving every instance its own random goal (same systems, pulses and initial states as D4-2), the tabulated co-state weight lets a cheap, goal-blind estimator reach the targets with 11 to 17 % less compute in `m4` (0.83 [0.75, 0.92] to 0.89 [0.81, 0.99] of `cheap`'s compute) and 8 to 18 % less in `m64` (0.92 [0.86, 1.00] to 0.82 [0.79, 0.87]), and with about 10 % less in `base`, `depth7` and `depth11` (0.89 to 0.91) where the intervals just touch 1, so the frozen rule fails there by one bound and is sensitive to how a partial pass is priced. It does nothing in `wide` and `amp1`. In `m4`, `m64`, `base`, `depth7` and `depth11` the goal-aware cheap arm needs 0.53 to 0.60 of uniform refinement's compute at the 1 % target; in `wide` and `amp1` adaptive refinement does not pay at all. **But the co-state was taken as given.** Building the table for each instance's own goal costs 880 to 14,075 CN steps, more than the 341 to 741 steps the goal-aware arm itself needs to reach the 1 % target, so with the exact table charged per instance the arm needs 2.1 to 50 times the compute of `cheap`. With a goal per instance the co-state is worth having only if it is amortised across goals, which is the premise of H2 and exactly what a learned critic would have to deliver.

---

## 1. What this run is

- **Domain and data:** the D4 domain (`y' = Ay + b(t)` on `[0, 1]`, Crank–Nicolson, exact matrix-exponential reference, objective `Σ|Λᵀτ|`) on the seven D4-2 cells (`base`, `m4`, `m64`, `wide`, `amp1`, `depth7`, `depth11`; systems, pulses and initial states identical to D4-2's tuning (seed 3003, 10 instances) and validation (seed 1001, 20 instances) families). **Only the goal differs:** a fresh random unit vector per instance from its own stream (`default_rng(family seed + 10000 + m)`). Generated from frozen seeds; **this is not real data.**
- **Arms** (Dörfler marking except `uniform_pass`; `θ` per (cell, arm) on the tuning family): `residual`, `goal_local`, `adjoint` (step doubling, `2n`, `2n`, `3n` CN steps per pass); `cheap` (forcing discrepancy plus solution-difference estimate, goal-blind) and `cheap_adjoint` (the same two estimates weighted by the tabulated co-state of the instance's goal, **the table taken as given**), both priced in FLOPs; `cheap_adjoint_setup` (the same traces with the per-instance table setup added); `adjoint_free` (hypothetical: the `adjoint` arm with scoring priced at zero).
- **Measure:** compute to close 90 %, 97 % and 99 % of the initial-to-finest gap, log-log interpolation (primary), semilog and staircase as sensitivities; instance-bootstrap CIs of the mean log ratio (10,000 resamples).
- **Frozen rules** (plan §6, hashes in `reports/frozen_before_validation.json`): **R3v** `cheap_adjoint` beats `cheap`; **R4v** `cheap_adjoint` beats `uniform_pass`; a D4-1 candidate needs both; "beats" is a CI below 0 at two adjacent targets.
- **Design history (disclosed).** A smoke test of the notebook with a tiny configuration ran before the real run; it was fed the **train** family in place of validation, so no validation instance had been generated; its numbers are not evidence and led to no change. The real run generated the validation instances only after the freeze (file times in the run directory).

## 2. Gates

| Gate | Result | Source |
|---|---|---|
| FLOP formula equals the operations counted from the code path (all 7 cells) | ✅ | `tests/correctness.json` |
| Co-state table close to the discrete co-state on a fine mesh; table linear in the goal | ✅ max relative gap 2.3e-04; linearity gap below 1e-8 | `tests/correctness.json` |
| Instances equal D4-2's except goal and name; goals unit and distinct | ✅ (largest absolute cosine between two goals of a cell: 0.98) | `tests/correctness.json` |
| Error representation on batch grids; reference converged | ✅ (max representation difference 5.8e-13, max reference difference 1.4e-10) | `tests/correctness.json` |
| Test family refused and unread | ✅ `test_family_read: false` | `reports/acceptance_report.json` |
| Frozen before validation was generated | ✅ | `reports/frozen_before_validation.json` |
| **R3v** in `m4`, `m64` | ✅ (`base`, `depth7`, `depth11`, `wide`, `amp1`: ❌) | `reports/acceptance_report.json` |
| **R4v** in `base`, `m4`, `m64`, `depth7`, `depth11` | ✅ (`wide`, `amp1`: ❌) | `reports/acceptance_report.json` |
| **D4-1 candidate cells** | **`m4`, `m64`** | `reports/acceptance_report.json` |
| `θ` at the edge of its grid | ⚠️ 9 of 35 choices, all in `wide` and `amp1` | `artifacts/theta_tuning.csv` |
| Targets not reached (censored at the cap) | Only `goal_local` in `depth7` at the 1 % target (19 of 20 instances reached it); every other arm and target reached by all 20 | `artifacts/validation_summary.json` |

## 3. Results

Ratios are compute ratios (below 1 means the first arm needs less). Every table is printed by `scripts/d4_3_tables.py` from the committed run files (§6).

### 3.1 Scoring price (FLOPs, in CN steps)

| Cell | m | CN step (FLOPs) | Cheap ÷ step-doubling | Cheap-adjoint ÷ step-doubling | Co-state table setup, per instance (CN steps) |
|---|---:|---:|---:|---:|---:|
| base | 32 | 4,768 | 0.141 | 0.155 | 3519 |
| m4 | 4 | 204 | 0.556 | 0.596 | 1285 |
| m64 | 64 | 17,664 | 0.074 | 0.082 | 3799 |
| wide | 32 | 4,768 | 0.141 | 0.155 | 3519 |
| amp1 | 32 | 4,768 | 0.141 | 0.155 | 3519 |
| depth7 | 32 | 4,768 | 0.141 | 0.155 | 880 |
| depth11 | 32 | 4,768 | 0.141 | 0.155 | 14075 |

### 3.2 Validation: compute to reach each target (log-log interpolation)

| Cell | Target | Median compute: uniform / cheap / cheap-adjoint / cheap-adjoint + setup / adjoint / adjoint-free | Cheap ÷ uniform | Cheap-adjoint ÷ uniform | Cheap-adjoint ÷ cheap [95 % CI] | Setup-charged ÷ cheap | Goal-local ÷ residual | Adjoint-free ÷ cheap-adjoint |
|---|---|---|---|---|---|---|---|---|
| base | 10% | 337 / 287 / 256 / 3887 / 696 / 236 | 0.86 [0.78, 0.95] | 0.79 [0.65, 0.94] | 0.91 [0.79, 1.07] | 13.13 [10.86, 15.71] | 1.14 [1.01, 1.31] | 0.86 [0.74, 0.97] |
| base | 3% | 757 / 490 / 455 / 4084 / 1254 / 409 | 0.67 [0.62, 0.74] | 0.60 [0.51, 0.70] | 0.89 [0.80, 1.01] | 7.90 [6.84, 9.03] | 1.26 [1.15, 1.40] | 0.84 [0.75, 0.92] |
| base | 1% | 1347 / 784 / 735 / 4371 / 1949 / 613 | 0.59 [0.53, 0.65] | 0.53 [0.46, 0.61] | 0.90 [0.82, 0.99] | 5.29 [4.63, 5.97] | 1.36 [1.23, 1.53] | 0.81 [0.73, 0.88] |
| m4 | 10% | 383 / 313 / 295 / 1953 / 575 / 166 | 0.99 [0.89, 1.10] | 0.82 [0.75, 0.90] | 0.83 [0.75, 0.92] | 5.44 [4.56, 6.47] | 1.19 [1.01, 1.43] | 0.54 [0.50, 0.58] |
| m4 | 3% | 795 / 561 / 486 / 2127 / 931 / 267 | 0.75 [0.68, 0.84] | 0.63 [0.57, 0.71] | 0.85 [0.79, 0.91] | 3.57 [3.13, 4.08] | 1.23 [1.08, 1.42] | 0.53 [0.49, 0.57] |
| m4 | 1% | 1397 / 952 / 741 / 2490 / 1499 / 465 | 0.64 [0.58, 0.73] | 0.57 [0.49, 0.68] | 0.89 [0.81, 0.99] | 2.68 [2.39, 3.01] | 1.19 [1.06, 1.35] | 0.54 [0.50, 0.58] |
| m64 | 10% | 271 / 241 / 234 / 4062 / 622 / 217 | 0.98 [0.85, 1.13] | 0.90 [0.80, 1.02] | 0.92 [0.86, 1.00] | 15.95 [13.83, 18.35] | 1.20 [1.10, 1.30] | 0.97 [0.85, 1.12] |
| m64 | 3% | 600 / 459 / 407 / 4237 / 1198 / 416 | 0.78 [0.70, 0.87] | 0.68 [0.62, 0.76] | 0.87 [0.83, 0.93] | 8.86 [8.00, 9.75] | 1.26 [1.15, 1.39] | 1.07 [0.97, 1.18] |
| m64 | 1% | 1185 / 801 / 636 / 4488 / 2007 / 729 | 0.72 [0.64, 0.81] | 0.59 [0.53, 0.66] | 0.82 [0.79, 0.87] | 5.47 [5.01, 5.96] | 1.33 [1.22, 1.46] | 1.12 [1.02, 1.24] |
| wide | 10% | 76 / 86 / 86 / 3676 / 200 / 79 | 1.13 [1.13, 1.14] | 1.16 [1.13, 1.19] | 1.02 [1.00, 1.05] | 44.99 [41.09, 49.28] | 1.08 [1.05, 1.13] | 0.89 [0.86, 0.94] |
| wide | 3% | 169 / 191 / 185 / 3796 / 430 / 166 | 1.13 [1.12, 1.14] | 1.13 [1.10, 1.16] | 1.00 [0.98, 1.03] | 20.70 [19.18, 22.36] | 1.10 [1.06, 1.15] | 0.89 [0.86, 0.92] |
| wide | 1% | 318 / 360 / 341 / 3970 / 780 / 307 | 1.13 [1.12, 1.14] | 1.12 [1.08, 1.15] | 0.99 [0.96, 1.01] | 11.42 [10.69, 12.23] | 1.13 [1.08, 1.18] | 0.88 [0.86, 0.90] |
| amp1 | 10% | 107 / 122 / 129 / 3766 / 269 / 107 | 1.14 [1.13, 1.14] | 1.20 [1.06, 1.34] | 1.06 [0.93, 1.18] | 27.67 [21.24, 35.32] | 1.13 [1.08, 1.18] | 0.81 [0.75, 0.90] |
| amp1 | 3% | 285 / 325 / 316 / 3966 / 723 / 285 | 1.13 [1.13, 1.14] | 1.07 [0.94, 1.19] | 0.94 [0.83, 1.05] | 11.25 [8.94, 13.99] | 1.15 [1.06, 1.25] | 0.93 [0.85, 1.01] |
| amp1 | 1% | 634 / 708 / 585 / 4225 / 1598 / 629 | 1.13 [1.12, 1.14] | 0.93 [0.82, 1.05] | 0.82 [0.73, 0.92] | 5.93 [4.87, 7.14] | 1.15 [1.06, 1.26] | 1.07 [0.97, 1.17] |
| depth7 | 10% | 334 / 286 / 254 / 1180 / 690 / 235 | 0.86 [0.78, 0.95] | 0.79 [0.65, 0.94] | 0.91 [0.78, 1.07] | 4.02 [3.38, 4.75] | 1.14 [1.00, 1.30] | 0.87 [0.75, 0.98] |
| depth7 | 3% | 744 / 483 / 449 / 1343 / 1236 / 401 | 0.68 [0.62, 0.74] | 0.61 [0.52, 0.71] | 0.89 [0.79, 1.01] | 2.69 [2.37, 3.04] | 1.26 [1.15, 1.39] | 0.84 [0.75, 0.92] |
| depth7 | 1% | 1280 / 750 / 701 / 1604 / 1862 / 592 | 0.60 [0.54, 0.66] | 0.54 [0.47, 0.62] | 0.90 [0.82, 0.99] | 2.06 [1.84, 2.28] | 1.35 [1.21, 1.51] | 0.81 [0.74, 0.88] |
| depth11 | 10% | 338 / 287 / 256 / 14693 / 697 / 236 | 0.86 [0.78, 0.95] | 0.79 [0.65, 0.94] | 0.91 [0.79, 1.07] | 49.55 [40.76, 59.58] | 1.14 [1.01, 1.31] | 0.86 [0.74, 0.97] |
| depth11 | 3% | 758 / 490 / 456 / 14845 / 1255 / 410 | 0.67 [0.62, 0.74] | 0.60 [0.51, 0.70] | 0.89 [0.80, 1.01] | 28.82 [24.81, 33.13] | 1.26 [1.15, 1.40] | 0.84 [0.75, 0.92] |
| depth11 | 1% | 1351 / 787 / 737 / 15091 / 1955 / 616 | 0.59 [0.53, 0.65] | 0.53 [0.46, 0.61] | 0.90 [0.82, 0.99] | 18.33 [15.89, 20.87] | 1.36 [1.23, 1.53] | 0.81 [0.73, 0.88] |

### 3.3 Frozen rules, and the same rules under the sensitivity pricings

| Cell | R3v weight helps cheap | R4v cheap-adjoint beats uniform | D4-1 candidate | Setup-charged beats cheap | R3v semilog | R3v staircase | R4v semilog | R4v staircase |
|---|---|---|---|---|---|---|---|---|
| base | False | True | False | False | False | True | True | True |
| m4 | True | True | True | False | True | True | True | True |
| m64 | True | True | True | False | True | True | True | True |
| wide | False | False | False | False | False | False | False | False |
| amp1 | False | False | False | False | False | True | False | False |
| depth7 | False | True | False | False | False | True | True | True |
| depth11 | False | True | False | False | False | True | True | True |

### 3.4 The co-state weight with one goal per cell (D4-2) and a goal per instance (D4-3)

| Cell | Target | One goal per cell (D4-2) [95 % CI] | A goal per instance (D4-3) [95 % CI] |
|---|---|---|---|
| base | 10% | 1.05 [0.96, 1.16] | 0.91 [0.79, 1.07] |
| base | 3% | 0.97 [0.90, 1.06] | 0.89 [0.80, 1.01] |
| base | 1% | 0.97 [0.91, 1.04] | 0.90 [0.82, 0.99] |
| m4 | 10% | 0.88 [0.79, 0.97] | 0.83 [0.75, 0.92] |
| m4 | 3% | 0.89 [0.80, 1.01] | 0.85 [0.79, 0.91] |
| m4 | 1% | 0.88 [0.80, 0.98] | 0.89 [0.81, 0.99] |
| m64 | 10% | 0.91 [0.81, 0.99] | 0.92 [0.86, 1.00] |
| m64 | 3% | 0.90 [0.83, 0.97] | 0.87 [0.83, 0.93] |
| m64 | 1% | 0.88 [0.81, 0.94] | 0.82 [0.79, 0.87] |
| wide | 10% | 1.03 [1.01, 1.06] | 1.02 [1.00, 1.05] |
| wide | 3% | 1.02 [1.00, 1.04] | 1.00 [0.98, 1.03] |
| wide | 1% | 1.01 [0.98, 1.03] | 0.99 [0.96, 1.01] |
| amp1 | 10% | 1.27 [1.21, 1.35] | 1.06 [0.93, 1.18] |
| amp1 | 3% | 1.09 [1.00, 1.18] | 0.94 [0.83, 1.05] |
| amp1 | 1% | 1.00 [0.91, 1.08] | 0.82 [0.73, 0.92] |
| depth7 | 10% | 1.05 [0.96, 1.16] | 0.91 [0.78, 1.07] |
| depth7 | 3% | 0.98 [0.90, 1.06] | 0.89 [0.79, 1.01] |
| depth7 | 1% | 0.97 [0.91, 1.05] | 0.90 [0.82, 0.99] |
| depth11 | 10% | 1.05 [0.96, 1.16] | 0.91 [0.79, 1.07] |
| depth11 | 3% | 0.97 [0.90, 1.06] | 0.89 [0.80, 1.01] |
| depth11 | 1% | 0.97 [0.91, 1.04] | 0.90 [0.82, 0.99] |

### 3.5 Per-instance distribution of cheap-adjoint ÷ cheap

| Cell | Target | Instances | Win rate of cheap-adjoint | Ratio quantiles 10 / 25 / 50 / 75 / 90 % | Largest ratio |
|---|---|---:|---:|---|---:|
| base | 10% | 20 | 0.70 | 0.61 / 0.76 / 0.84 / 1.09 / 1.32 | 2.50 |
| base | 3% | 20 | 0.70 | 0.67 / 0.75 / 0.81 / 1.01 / 1.27 | 1.80 |
| base | 1% | 20 | 0.65 | 0.69 / 0.75 / 0.89 / 1.05 / 1.12 | 1.42 |
| m4 | 10% | 20 | 0.65 | 0.60 / 0.70 / 0.87 / 1.01 / 1.04 | 1.18 |
| m4 | 3% | 20 | 0.85 | 0.70 / 0.74 / 0.85 / 0.97 / 1.03 | 1.13 |
| m4 | 1% | 20 | 0.70 | 0.68 / 0.75 / 0.89 / 1.03 / 1.12 | 1.74 |
| m64 | 10% | 20 | 0.75 | 0.78 / 0.85 / 0.89 / 1.00 / 1.06 | 1.57 |
| m64 | 3% | 20 | 0.90 | 0.75 / 0.80 / 0.86 / 0.93 / 0.98 | 1.30 |
| m64 | 1% | 20 | 0.95 | 0.74 / 0.77 / 0.80 / 0.86 / 0.92 | 1.20 |
| wide | 10% | 20 | 0.45 | 0.97 / 0.99 / 1.00 / 1.03 / 1.14 | 1.16 |
| wide | 3% | 20 | 0.55 | 0.95 / 0.98 / 1.00 / 1.02 / 1.07 | 1.12 |
| wide | 1% | 20 | 0.55 | 0.92 / 0.96 / 1.00 / 1.03 / 1.06 | 1.08 |
| amp1 | 10% | 20 | 0.35 | 0.87 / 0.97 / 1.10 / 1.28 / 1.32 | 1.47 |
| amp1 | 3% | 20 | 0.45 | 0.73 / 0.81 / 1.05 / 1.14 / 1.19 | 1.27 |
| amp1 | 1% | 20 | 0.70 | 0.63 / 0.72 / 0.84 / 1.03 / 1.15 | 1.27 |
| depth7 | 10% | 20 | 0.70 | 0.61 / 0.76 / 0.84 / 1.09 / 1.32 | 2.51 |
| depth7 | 3% | 20 | 0.70 | 0.67 / 0.75 / 0.81 / 1.01 / 1.29 | 1.80 |
| depth7 | 1% | 20 | 0.60 | 0.68 / 0.75 / 0.90 / 1.05 / 1.16 | 1.42 |
| depth11 | 10% | 20 | 0.70 | 0.61 / 0.76 / 0.84 / 1.09 / 1.33 | 2.50 |
| depth11 | 3% | 20 | 0.70 | 0.67 / 0.75 / 0.81 / 1.01 / 1.27 | 1.80 |
| depth11 | 1% | 20 | 0.65 | 0.69 / 0.75 / 0.89 / 1.05 / 1.12 | 1.42 |

### 3.6 Fraction of validation instances reaching each target

| Cell | Target | uniform_pass | cheap | cheap_adjoint | residual | goal_local | adjoint |
|---|---|---:|---:|---:|---:|---:|---:|
| base | 10% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| base | 3% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| base | 1% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| m4 | 10% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| m4 | 3% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| m4 | 1% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| m64 | 10% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| m64 | 3% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| m64 | 1% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| wide | 10% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| wide | 3% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| wide | 1% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| amp1 | 10% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| amp1 | 3% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| amp1 | 1% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| depth7 | 10% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| depth7 | 3% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| depth7 | 1% | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 |
| depth11 | 10% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| depth11 | 3% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| depth11 | 1% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Lowest fraction of instances reaching a target: 0.95.

### 3.7 Marking fraction (tuning family)

| Cell | Arm | θ = 0.5 | θ = 0.7 | θ = 0.85 | θ = 0.95 | θ = 0.98 | θ = 0.995 | Selected | At grid edge |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| base | residual | 1442 | 1000 | 840 | 914 | 1049 | 1166 | 0.85 | False |
| base | goal_local | 1907 | 1370 | 1150 | 1012 | 1050 | 1139 | 0.95 | False |
| base | adjoint | 2013 | 1429 | 1205 | 1064 | 1171 | 1311 | 0.95 | False |
| base | cheap | 583 | 443 | 425 | 523 | 595 | 655 | 0.85 | False |
| base | cheap_adjoint | 618 | 460 | 402 | 417 | 451 | 540 | 0.85 | False |
| m4 | residual | 1489 | 1079 | 968 | 990 | 1088 | 1284 | 0.85 | False |
| m4 | goal_local | 1853 | 1383 | 1100 | 1099 | 1187 | 1207 | 0.95 | False |
| m4 | adjoint | 1909 | 1396 | 1150 | 969 | 1023 | 1206 | 0.95 | False |
| m4 | cheap | 970 | 771 | 705 | 723 | 819 | 1019 | 0.85 | False |
| m4 | cheap_adjoint | 909 | 738 | 670 | 569 | 575 | 681 | 0.95 | False |
| m64 | residual | 1593 | 1089 | 944 | 1041 | 1197 | 1296 | 0.85 | False |
| m64 | goal_local | 1958 | 1344 | 1099 | 1064 | 1153 | 1258 | 0.95 | False |
| m64 | adjoint | 2204 | 1523 | 1319 | 1233 | 1183 | 1423 | 0.98 | False |
| m64 | cheap | 591 | 457 | 458 | 576 | 632 | 690 | 0.7 | False |
| m64 | cheap_adjoint | 581 | 432 | 385 | 369 | 434 | 535 | 0.95 | False |
| wide | residual | 891 | 579 | 431 | 345 | 311 | 302 | 0.995 | True |
| wide | goal_local | 1186 | 758 | 564 | 467 | 406 | 341 | 0.995 | True |
| wide | adjoint | 1049 | 682 | 541 | 433 | 399 | 391 | 0.995 | True |
| wide | cheap | 377 | 262 | 209 | 180 | 171 | 170 | 0.995 | True |
| wide | cheap_adjoint | 362 | 259 | 212 | 187 | 174 | 171 | 0.995 | True |
| amp1 | residual | 1143 | 774 | 604 | 540 | 523 | 512 | 0.995 | True |
| amp1 | goal_local | 1535 | 1060 | 830 | 688 | 610 | 547 | 0.995 | True |
| amp1 | adjoint | 1448 | 988 | 760 | 672 | 660 | 640 | 0.995 | True |
| amp1 | cheap | 499 | 373 | 320 | 298 | 292 | 288 | 0.995 | True |
| amp1 | cheap_adjoint | 493 | 360 | 310 | 296 | 302 | 300 | 0.95 | False |
| depth7 | residual | 1421 | 977 | 822 | 893 | 1024 | 1135 | 0.85 | False |
| depth7 | goal_local | 1832 | 1339 | 1135 | 990 | 1037 | 1118 | 0.95 | False |
| depth7 | adjoint | 1939 | 1415 | 1183 | 1044 | 1153 | 1298 | 0.95 | False |
| depth7 | cheap | 568 | 432 | 416 | 511 | 581 | 638 | 0.85 | False |
| depth7 | cheap_adjoint | 606 | 451 | 394 | 409 | 442 | 528 | 0.85 | False |
| depth11 | residual | 1445 | 1001 | 841 | 915 | 1051 | 1168 | 0.85 | False |
| depth11 | goal_local | 1911 | 1372 | 1151 | 1014 | 1052 | 1141 | 0.95 | False |
| depth11 | adjoint | 2015 | 1431 | 1206 | 1065 | 1172 | 1313 | 0.95 | False |
| depth11 | cheap | 584 | 443 | 426 | 523 | 596 | 656 | 0.85 | False |
| depth11 | cheap_adjoint | 619 | 461 | 402 | 418 | 452 | 540 | 0.85 | False |

## 4. Interpretation

**Observed** (from the tables above):

- **Two candidate cells.** R3v and R4v both hold in `m4` and `m64`. The weight saves 11 to 17 % of `cheap`'s compute in `m4` and 8 to 18 % in `m64`, and the goal-aware cheap arm needs 0.57 to 0.82 of uniform's (`m4`) and 0.59 to 0.90 (`m64`) depending on the target.
- **A borderline group.** In `base`, `depth7` and `depth11` (one configuration at three finest depths, so one finding) the point estimates are 0.91, 0.89 and 0.90 at the three targets but the intervals reach 1 (upper bounds 1.07, 1.01, 0.99), so R3v fails by one bound; it fails under semilog pricing and holds under the staircase. R4v holds there (0.79, 0.60, 0.53 of uniform).
- **Nothing in `wide`, little in `amp1`.** In `wide` the weight changes compute by −1 % to +2 % (1.02, 1.00, 0.99). In `amp1` the ratios are 1.06, 0.94 and 0.82 at the three targets and only the last interval ([0.73, 0.92]) excludes 1, so the rule fails. The goal-aware cheap arm needs more compute than uniform refinement in `wide` at every target (1.12 to 1.16) and in `amp1` at two of three (1.20, 1.07).
- **With one goal per cell, the weight looked weaker.** `cheap_adjoint ÷ cheap` was 1.05, 0.97, 0.97 in `base` under D4-2 and is 0.91, 0.89, 0.90 here; in `amp1` 1.27, 1.09, 1.00 against 1.06, 0.94, 0.82; in `m4` and `m64` the two runs agree (0.88 to 0.91 against 0.82 to 0.92).
- **The exact co-state cannot be built per instance.** The table setup (880 steps at depth 7 to 14,075 at depth 11, 1,285 at `m4`, 3,519 at `base`) exceeds, in every cell, the compute the goal-aware arm itself needs to reach the 1 % target (341 to 741 steps); charged to each instance, the goal-aware arm needs 2.06 to 49.55 times `cheap`'s compute and never beats it or uniform refinement.
- **Weighting by the final-time goal alone does not help step doubling.** `goal_local ÷ residual` is 1.08 to 1.36 at every cell and target (21 of 21, every interval's lower bound at least 1.00): at the same price, `|cᵀ τ̂|` needs more compute than the goal-blind `‖τ̂‖`, because it ignores how the co-state propagates the local error to the goal.
- **Headroom beyond the cheap arms differs by cell.** `adjoint_free ÷ cheap_adjoint` is 0.53 to 0.54 in `m4` (a free exact co-state would need about half the compute), 0.81 to 0.86 in `base`, `depth7`, `depth11`, and 0.97 to 1.12 in `m64` (none left).
- **Instances differ.** In `base` the goal-aware arm wins on 65 to 70 % of instances and is up to 2.5 times worse on some (10 % target); in `m64` at the 1 % target it wins on 19 of 20.

**Hypotheses (not tested here):**

1. *D4-2's single goal per cell understated the average value of the weight.* Both runs estimate the average over goals of the log ratio; D4-2 had one draw per cell, this run has twenty. **Diagnostic:** more goals per cell would narrow the interval; it would not change the price finding.
2. *A learned critic could deliver the co-state at lookup price by amortising it over goals.* The exact table costs more than the solve when the goal varies, so the value measured here is reachable only through a learned or shared co-state. **Diagnostic:** D4-1, a learned direct critic given the goal against a critic trained on the co-state, matched in size and training, in `m4` and `m64`.
3. *The remaining headroom in `m4` (about half) could be captured by a better scorer.* **Diagnostic:** the same D4-1 comparison, with `adjoint_free` as the ceiling.

## 5. Design issues found

1. **The co-state is taken as given in the gated arm,** and its real per-instance price is prohibitive at the finest grid. The priced variant is a deliberately pessimistic bound (a sweep on the current mesh is what `adjoint` charges), so "charged, the arm loses" is a statement about this tabulation, not about every way of obtaining a co-state.
2. **D4-2 and D4-3 share pulses and initial states,** so their agreement is not independent confirmation; only the goals differ.
3. **`m` is confounded with the system draw** in `m4`, `base` and `m64`.
4. **`base`, `depth7` and `depth11` are near copies** and count as one finding; R3v fails there by one bound and its verdict depends on the pricing of a partial pass.
5. **`θ` is at the edge of its grid in `wide` and `amp1`** (9 of 35 choices), so the tuned arms there may be understated.
6. **The goals are random unit vectors.** A goal aligned with a few state directions (a physical observable) may behave differently. Not tested.
7. **Twenty validation instances** give intervals too wide to resolve effects of a few percent, and instances differ a lot.
8. **The cheap estimators are hand-built heuristics.**
9. **A smoke test preceded the run** (disclosed in §1).

## 6. Commands

Every table in §3 is printed from committed files; the frozen rules are re-applied for every interpolation method and the primary method is asserted equal to the run's own report:

```bash
python scripts/d4_3_tables.py results/runs/d4_3_varying_goal_20260930T084725Z
```

The reproduction from a clean worktree is described in the run README.

## 7. What this run does and does not support

**Supports:**

- With a goal per instance, the co-state weight helps a goal-blind cheap estimator by 8 to 18 % in `m4` and `m64`, by about 10 % (borderline) in `base`, `depth7` and `depth11`, and not in `wide` and `amp1` (§3.2, §3.3).
- Adaptive refinement with a goal-aware cheap arm needs 0.53 to 0.60 of uniform refinement's compute at the 1 % target in five of seven cells and does not pay in `wide` and `amp1` (§3.2).
- The frozen rule gives two D4-1 candidate cells, `m4` and `m64`.
- The exact tabulated co-state cannot be built per instance at this price (§3.2, column "setup-charged").
- Step-doubling scoring weighted by the final-time goal alone needs more compute than the goal-blind residual (§3.2).

**Does not support:**

- Any statement about a learned direct critic or a co-state-featured critic (H2).
- A co-state advantage at a price: the gated arm takes the table as given.
- Any statement about real data, goals aligned with physical observables, wall-clock time, hardware or the test family.
- Any claim that `m` alone causes the differences between `m4`, `base` and `m64`.

**Corrects earlier statements.**

- The D4-2 note (§1 and §4) said the co-state weight helps the cheap estimator only at `m` = 64. With a goal per instance it also helps in `m4` and, at the edge of the frozen rule, in `base`, `depth7` and `depth11`; D4-2's single goal per cell understated its average value in `base` and `amp1`.
- The roadmap and cross-domain plan said D4-1 stays closed unless D4-3 gave a confidence-bounded margin for the weight; it does in `m4` and `m64`, so **D4-1 reopens there.**

## 8. Next steps (roadmap IDs in `docs/plans/roadmap.md`)

1. **D4-1, in `m4` and `m64`:** write its plan first and freeze it before any critic is trained: a learned direct critic given the goal vector against a critic trained on the co-state, matched in size and training, five paired seeds, priced in FLOPs on the same ledger, with the non-learned `cheap` and `cheap_adjoint` arms as references and `adjoint_free` as the ceiling. CPU; no Colab needed.
2. Keep the test family unread until D4-1's design is frozen and its validation result is in.
