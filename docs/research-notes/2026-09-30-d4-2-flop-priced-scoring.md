# D4-2: FLOP-priced scoring on a higher-dimensional system (validation-only design study)

**Run:** `d4_2_flop_scoring_20260929T233556Z` · **Date:** 2026-09-30 · **Notebook:** `notebooks/04-domains/d4_2_flop_priced_scoring.ipynb` (SHA-256 `8f29012f…6a016e`)
**Hardware:** CPU in the Claude Code sandbox (FLOPs are counted, not measured; timings are not a systems claim) · **Package commit:** `b139a33a…` (clean tree) · **Config SHA-256:** `2ab9e446…3d88b4` · **Frozen manifest SHA-256:** `1a4e9c23…497607`
**Artefacts:** [`results/runs/d4_2_flop_scoring_20260929T233556Z/`](../../results/runs/d4_2_flop_scoring_20260929T233556Z/) · **Plan:** [`docs/plans/d4-2-plan.md`](../plans/d4-2-plan.md) · **Follows:** [D4-0b](2026-09-29-d4-0b-where-adaptivity-pays.md)

**Status:** exploratory design study on a scorer-training, a tuning and a validation family; **the test family was never generated or read.** Correctness ✅ · frozen rules: **R1 (an amortised scorer pays) holds in 3 of 7 cells, R3 (the co-state weight helps a cheap estimator) in 1, and no cell has both, so D4-1 stays closed** · no learned direct critic, so **no H2 statement.**

**In one paragraph.** With scoring priced in floating-point operations, the price of a small learned scorer relative to step-doubling scoring falls from 2.24 at `m` = 4 to 0.245 at `m` = 32 and 0.133 at `m` = 64, as hypothesised. At those prices, adaptive refinement needs less compute than uniform refinement in the three `m` = 32 cells with narrow pulses (the learned scorer: 0.77 [0.69, 0.88] and 0.68 [0.62, 0.76] of uniform's compute at the two tighter targets in the `base` cell), and in no wide-pulse or amplitude-1 cell. **But a hand-built cheap estimator with no learning does at least as well, at a lower price, in every cell**, so the learned scorer adds nothing in D4, and it needs about 5,000 to 10,000 instances to repay its training. The co-state weight helps the cheap estimator only at `m` = 64, by about 10 %; where the learned scorer pays, the weight does not help, and where the weight helps, the learned scorer misses the frozen rule by a confidence bound. In the `base` cell the cheap co-state-weighted estimator needs 0.63 [0.57, 0.70] of uniform's compute at the 1 % target against 0.66 [0.59, 0.73] for the exact co-state with free scoring (a hypothetical ceiling). D4 therefore leaves little headroom for a co-state contribution beyond cheap scoring when the goal is fixed.

---

## 1. What this run is

- **Domain and data:** the D4 domain (`y' = Ay + b(t)` on `[0, 1]`, Crank–Nicolson, exact matrix-exponential reference, objective the cancellation-free bound `Σ|Λᵀτ|`) on dense stable systems of dimension `m` = 4, 32 and 64 (damped oscillators rotated by a random orthogonal matrix). **`A` and the goal `c` are fixed per system;** instances differ in 2–4 Gaussian forcing pulses and the initial state. Generated from frozen seeds; **this is not real data.** Families: scorer training (seed 4004, 30 instances per cell), tuning (3003, 10), validation (1001, 20). Seven cells, one factor varied at a time from `base` (`m` = 32, pulse width 0.001–0.004, amplitude scale 10, finest grid 8,192 intervals): `m4`, `m64`, `wide` (width 0.01–0.04), `amp1` (scale 1), `depth7` (2,048 intervals) and `depth11` (32,768).
- **Arms** (all pass-based with Dörfler marking except `uniform_pass`): `residual`, `goal_local` and `adjoint` (step-doubling, priced `2n`, `2n`, `3n` CN steps per pass); `indicator` (forcing quadrature discrepancy; a diagnostic); `cheap` (that plus a third divided difference of the numerical solution); `cheap_adjoint` (the same weighted by a tabulated continuous co-state; its one-off setup is reported separately); `amortised` (a 9 → h → h → 1 MLP predicting the exact co-state-weighted local error from nine cheap features, no co-state input, `h` chosen from {8, 16, 32} on the tuning family); and `adjoint_free`, the `adjoint` arm with scoring priced at zero (**hypothetical**, read off the same traces).
- **Ledger:** operations counted from the code path (a CN step is `4 m² + 3 m + 2 P (2 m + 8)`), converted to CN steps; details in the plan §5. **Measure:** compute to close 90 %, 97 % and 99 % of the initial-to-finest gap (targets `10 %`, `3 %`, `1 %` of the gap left), priced by log-log interpolation between passes for every arm (primary, fixed in advance), with semilog interpolation and the staircase as sensitivities from the same traces. Ratios are geometric means with an instance-bootstrap CI (10,000 resamples) of the mean log ratio.
- **Frozen rules** (before validation was generated; hashes in `reports/frozen_before_validation.json`): **R1** `amortised` beats `uniform_pass`; **R3** `cheap_adjoint` beats `cheap`; a D4-1 candidate cell needs both. "Beats" means the CI lies below 0 at two adjacent targets.
- **Design history (disclosed).** All changes are dated in the plan §12. In short: the forcing-only `indicator` never reached even the 10 % target in a dry run on the `base` cell's train and tuning families, so `cheap` and `cheap_adjoint` were added as competent non-learned arms and R3 was moved to that pair; the scorer got a ninth input and its hidden size became a per-cell choice; a **smoke test with a tiny configuration printed validation medians for four instances of two cells**, and no design choice was changed because of them. A first launch of the real run was killed by a tool timeout during the scorer-training stage (four of seven scorers written), before the validation family was generated; it is not imported.

## 2. Gates

| Gate | Result | Source |
|---|---|---|
| FLOP formula equals the operations counted from the code path (all 7 cells) | ✅ | `tests/correctness.json` |
| Tabulated co-state close to the discrete co-state on a fine mesh | ✅ max relative gap 2.9e-04 | `tests/correctness.json` |
| Forcing indicator is zero without forcing; batch equals sequential; error representation on batch grids; reference converged | ✅ (max representation difference 5.1e-13, max reference difference 1.4e-10) | `tests/correctness.json` |
| Test family unread | ✅ `test_family_read: false` | `reports/acceptance_report.json` |
| Frozen manifest written before the validation family was generated | ✅ | `reports/frozen_before_validation.json` |
| Frozen rules: D4-1 candidate cells | **none** (R1 in `base`, `depth7`, `depth11`; R3 in `m64`) | `reports/acceptance_report.json` |
| `θ` at the edge of its grid (non-diagnostic arms) | ⚠️ 12 of 42 (arm, cell) choices, in `wide`, `amp1` and one in `m64` | `artifacts/theta_tuning.csv` |
| Targets not reached (censored at the cap) | The forcing-only `indicator`: no instance in five cells, and only 5–40 % of instances in `m4` and 5–50 % in `wide`; `goal_local` for one of 20 instances at one target in `depth7`. No other arm was censored | `artifacts/validation_summary.json` |

## 3. Results

Ratios are compute ratios (below 1 means the first arm needs less). Every table below is printed by `scripts/d4_2_tables.py` from the committed run files (§6). Cells are in the order the run reported them.

### 3.1 Scoring price (FLOPs, in CN steps)

| Cell | m | CN step (FLOPs) | Amortised (steps per interval) | Amortised ÷ step-doubling | Cheap ÷ step-doubling | Cheap-adjoint ÷ step-doubling | Indicator ÷ step-doubling | Co-state table setup (steps, one-off) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 32 | 4,768 | 0.49 | 0.245 | 0.141 | 0.155 | 0.087 | 3519 |
| m4 | 4 | 204 | 4.48 | 2.238 | 0.556 | 0.596 | 0.395 | 1285 |
| m64 | 64 | 17,664 | 0.27 | 0.133 | 0.074 | 0.082 | 0.045 | 3799 |
| wide | 32 | 4,768 | 0.49 | 0.245 | 0.141 | 0.155 | 0.087 | 3519 |
| amp1 | 32 | 4,768 | 0.49 | 0.245 | 0.141 | 0.155 | 0.087 | 3519 |
| depth7 | 32 | 4,768 | 0.49 | 0.245 | 0.141 | 0.155 | 0.087 | 880 |
| depth11 | 32 | 4,768 | 0.49 | 0.245 | 0.141 | 0.155 | 0.087 | 14075 |

The price of step-doubling scoring is `2` or `3` CN steps per interval at every `m`. The price of the cheap and learned scorers falls as `m` grows, because a CN step costs about `4 m²` operations and a scorer on a fixed feature vector costs about the same at any `m`. At `m` = 4 the learned scorer is more expensive than solving (2.24 times step-doubling); at `m` = 32 and 64 it is 0.245 and 0.133 times. The three `m` = 32 cells with the same `m` share one system, and `m` = 4, 32 and 64 use different random systems, so the effect of `m` in the results below includes the effect of the system draw.

### 3.2 Scorer size and training cost

| Cell | Hidden sizes tried: tuning compute (geometric mean) | Chosen | Held-out RMSE on training-family instances (log10) | Epochs | Training cost (CN steps) |
|---|---|---:|---:|---:|---:|
| base | 8: 502, 16: 537, 32: 798 | 8 | 0.47 | 53 | 2,688,693 |
| m4 | 8: 1545, 16: 2652, 32: 5624 | 8 | 0.44 | 30 | 35,780,246 |
| m64 | 8: 520, 16: 486, 32: 529 | 16 | 0.40 | 72 | 2,517,307 |
| wide | 8: 227, 16: 237, 32: 276 | 8 | 0.49 | 49 | 2,489,596 |
| amp1 | 8: 368, 16: 390, 32: 502 | 8 | 0.60 | 41 | 2,081,388 |
| depth7 | 8: 514, 16: 598, 32: 683 | 8 | 0.58 | 49 | 1,797,439 |
| depth11 | 8: 559, 16: 643, 32: 753 | 8 | 0.40 | 67 | 3,397,254 |

The smaller network won in six cells because its lower price outweighed its slightly lower accuracy; the 32-unit network was never chosen. Training cost is `3 ×` the forward FLOPs `×` training samples `×` epochs run, in CN steps; generating the exact labels needs reference solutions and is not charged.

### 3.3 Validation: compute to reach each target at the real price

| Cell | Target | Median compute: uniform / amortised / cheap / cheap-adjoint / adjoint / adjoint-free | Amortised ÷ uniform [95 % CI] | Cheap ÷ uniform | Cheap-adjoint ÷ uniform | Cheap-adjoint ÷ cheap | Adjoint ÷ uniform | Adjoint-free ÷ uniform |
|---|---|---|---|---|---|---|---|---|
| base | 10% | 251 / 250 / 227 / 255 / 660 / 236 | 0.97 [0.85, 1.14] | 0.92 [0.84, 1.02] | 0.97 [0.86, 1.09] | 1.05 [0.96, 1.16] | 2.49 [2.33, 2.68] | 0.90 [0.82, 0.98] |
| base | 3% | 639 / 448 / 429 / 468 / 1227 / 409 | 0.77 [0.69, 0.88] | 0.76 [0.69, 0.84] | 0.74 [0.66, 0.83] | 0.97 [0.90, 1.06] | 2.15 [1.98, 2.34] | 0.75 [0.68, 0.83] |
| base | 1% | 1181 / 738 / 731 / 746 / 1929 / 689 | 0.68 [0.62, 0.76] | 0.64 [0.60, 0.70] | 0.63 [0.57, 0.70] | 0.97 [0.91, 1.04] | 1.88 [1.71, 2.08] | 0.66 [0.59, 0.73] |
| m4 | 10% | 353 / 823 / 341 / 305 / 544 / 155 | 2.30 [2.05, 2.59] | 0.99 [0.89, 1.11] | 0.87 [0.75, 1.00] | 0.88 [0.79, 0.97] | 1.66 [1.44, 1.89] | 0.45 [0.37, 0.55] |
| m4 | 3% | 768 / 1365 / 578 / 509 / 932 / 245 | 1.88 [1.65, 2.15] | 0.82 [0.72, 0.94] | 0.73 [0.61, 0.88] | 0.89 [0.80, 1.01] | 1.31 [1.10, 1.55] | 0.37 [0.30, 0.45] |
| m4 | 1% | 1415 / 2234 / 941 / 846 / 1538 / 455 | 1.53 [1.38, 1.70] | 0.67 [0.60, 0.74] | 0.59 [0.50, 0.69] | 0.88 [0.80, 0.98] | 1.04 [0.88, 1.21] | 0.31 [0.26, 0.37] |
| m64 | 10% | 204 / 256 / 233 / 217 / 582 / 199 | 1.09 [0.95, 1.25] | 1.04 [0.92, 1.16] | 0.94 [0.82, 1.06] | 0.91 [0.81, 0.99] | 2.58 [2.31, 2.89] | 0.89 [0.77, 1.02] |
| m64 | 3% | 487 / 463 / 450 / 408 / 1165 / 393 | 0.89 [0.78, 1.01] | 0.85 [0.75, 0.96] | 0.76 [0.66, 0.87] | 0.90 [0.83, 0.97] | 2.21 [1.94, 2.52] | 0.76 [0.65, 0.88] |
| m64 | 1% | 953 / 807 / 746 / 675 / 1918 / 684 | 0.81 [0.71, 0.93] | 0.76 [0.67, 0.86] | 0.67 [0.57, 0.77] | 0.88 [0.81, 0.94] | 1.98 [1.69, 2.32] | 0.68 [0.57, 0.82] |
| wide | 10% | 66 / 89 / 74 / 76 / 175 / 69 | 1.27 [1.23, 1.32] | 1.13 [1.13, 1.14] | 1.17 [1.14, 1.20] | 1.03 [1.01, 1.06] | 2.62 [2.54, 2.74] | 1.03 [1.00, 1.08] |
| wide | 3% | 147 / 190 / 166 / 164 / 382 / 151 | 1.25 [1.23, 1.29] | 1.13 [1.13, 1.14] | 1.15 [1.12, 1.18] | 1.02 [1.00, 1.04] | 2.60 [2.53, 2.69] | 1.02 [1.00, 1.06] |
| wide | 1% | 278 / 354 / 314 / 304 / 712 / 281 | 1.24 [1.22, 1.27] | 1.13 [1.13, 1.14] | 1.14 [1.11, 1.17] | 1.01 [0.98, 1.03] | 2.57 [2.52, 2.65] | 1.01 [0.99, 1.04] |
| amp1 | 10% | 87 / 108 / 101 / 131 / 218 / 87 | 1.25 [1.24, 1.26] | 1.14 [1.13, 1.15] | 1.45 [1.38, 1.54] | 1.27 [1.21, 1.35] | 2.52 [2.49, 2.56] | 1.01 [0.99, 1.02] |
| amp1 | 3% | 237 / 295 / 269 / 301 / 597 / 238 | 1.25 [1.24, 1.26] | 1.14 [1.14, 1.15] | 1.24 [1.14, 1.35] | 1.09 [1.00, 1.18] | 2.52 [2.50, 2.54] | 1.00 [1.00, 1.01] |
| amp1 | 1% | 452 / 564 / 510 / 553 / 1140 / 454 | 1.26 [1.25, 1.27] | 1.14 [1.14, 1.15] | 1.14 [1.04, 1.24] | 1.00 [0.91, 1.08] | 2.53 [2.50, 2.56] | 1.01 [1.00, 1.02] |
| depth7 | 10% | 250 / 239 / 226 / 254 / 636 / 186 | 0.98 [0.87, 1.11] | 0.92 [0.84, 1.02] | 0.97 [0.86, 1.09] | 1.05 [0.96, 1.16] | 2.53 [2.24, 2.84] | 0.78 [0.68, 0.88] |
| depth7 | 3% | 629 / 473 / 424 / 462 / 1113 / 332 | 0.80 [0.72, 0.90] | 0.76 [0.69, 0.84] | 0.74 [0.67, 0.83] | 0.98 [0.90, 1.06] | 2.03 [1.79, 2.31] | 0.61 [0.54, 0.69] |
| depth7 | 1% | 1133 / 736 / 699 / 714 / 1750 / 550 | 0.70 [0.64, 0.77] | 0.65 [0.60, 0.71] | 0.63 [0.57, 0.71] | 0.97 [0.91, 1.05] | 1.69 [1.51, 1.90] | 0.52 [0.47, 0.58] |
| depth11 | 10% | 251 / 259 / 227 / 255 / 660 / 236 | 1.03 [0.91, 1.17] | 0.92 [0.84, 1.02] | 0.97 [0.86, 1.09] | 1.05 [0.96, 1.16] | 2.49 [2.33, 2.68] | 0.90 [0.82, 0.98] |
| depth11 | 3% | 639 / 476 / 429 / 468 / 1228 / 410 | 0.81 [0.73, 0.92] | 0.76 [0.69, 0.84] | 0.74 [0.66, 0.83] | 0.97 [0.90, 1.06] | 2.15 [1.98, 2.34] | 0.75 [0.68, 0.83] |
| depth11 | 1% | 1184 / 868 / 733 / 748 / 1932 / 690 | 0.73 [0.66, 0.81] | 0.64 [0.60, 0.70] | 0.63 [0.57, 0.70] | 0.97 [0.91, 1.04] | 1.88 [1.71, 2.08] | 0.66 [0.59, 0.73] |

### 3.4 The learned scorer against the cheap arms and the free ceiling

| Cell | Target | Amortised ÷ cheap [95 % CI] | Amortised ÷ cheap-adjoint | Adjoint-free ÷ amortised | Adjoint ÷ amortised | Residual ÷ uniform | Goal-local ÷ uniform | Indicator ÷ uniform (censored at the cap) |
|---|---|---|---|---|---|---|---|---|
| base | 10% | 1.06 [0.98, 1.14] | 1.01 [0.91, 1.11] | 0.92 [0.81, 1.04] | 2.56 [2.21, 2.93] | 1.85 [1.65, 2.10] | 2.22 [1.99, 2.49] | 66.12 [52.85, 83.99] |
| base | 3% | 1.02 [0.95, 1.09] | 1.04 [0.97, 1.11] | 0.98 [0.88, 1.07] | 2.80 [2.49, 3.09] | 1.49 [1.34, 1.67] | 2.00 [1.76, 2.29] | 28.51 [23.58, 35.09] |
| base | 1% | 1.06 [1.00, 1.12] | 1.09 [1.02, 1.16] | 0.97 [0.89, 1.04] | 2.76 [2.54, 2.99] | 1.29 [1.18, 1.43] | 1.75 [1.57, 1.97] | 15.02 [12.73, 17.92] |
| m4 | 10% | 2.32 [2.06, 2.56] | 2.64 [2.31, 3.05] | 0.20 [0.16, 0.24] | 0.72 [0.61, 0.84] | 1.36 [1.23, 1.51] | 1.45 [1.32, 1.61] | 11.94 [5.02, 26.14] |
| m4 | 3% | 2.29 [2.09, 2.49] | 2.57 [2.21, 2.99] | 0.20 [0.16, 0.23] | 0.70 [0.59, 0.82] | 1.12 [1.00, 1.27] | 1.23 [1.09, 1.39] | 15.05 [7.65, 25.82] |
| m4 | 1% | 2.30 [2.12, 2.48] | 2.60 [2.30, 2.97] | 0.20 [0.17, 0.23] | 0.68 [0.58, 0.78] | 0.93 [0.84, 1.02] | 1.02 [0.90, 1.17] | 10.79 [8.27, 13.66] |
| m64 | 10% | 1.05 [0.98, 1.13] | 1.16 [1.08, 1.27] | 0.81 [0.75, 0.88] | 2.36 [2.16, 2.57] | 2.06 [1.90, 2.22] | 2.12 [2.03, 2.22] | 71.68 [58.50, 86.11] |
| m64 | 3% | 1.05 [0.98, 1.13] | 1.17 [1.07, 1.26] | 0.85 [0.79, 0.91] | 2.48 [2.33, 2.65] | 1.77 [1.64, 1.91] | 2.13 [2.01, 2.27] | 30.63 [25.81, 35.88] |
| m64 | 1% | 1.07 [1.01, 1.13] | 1.21 [1.13, 1.31] | 0.84 [0.78, 0.91] | 2.44 [2.27, 2.62] | 1.60 [1.49, 1.73] | 2.06 [1.95, 2.17] | 16.19 [13.80, 18.72] |
| wide | 10% | 1.12 [1.09, 1.17] | 1.09 [1.05, 1.13] | 0.81 [0.77, 0.85] | 2.06 [1.96, 2.17] | 2.00 [2.00, 2.01] | 2.23 [2.16, 2.31] | 50.83 [22.37, 110.19] |
| wide | 3% | 1.11 [1.09, 1.14] | 1.09 [1.06, 1.12] | 0.82 [0.79, 0.85] | 2.07 [2.00, 2.15] | 2.01 [2.00, 2.02] | 2.28 [2.19, 2.39] | 76.58 [54.09, 102.53] |
| wide | 1% | 1.09 [1.08, 1.11] | 1.08 [1.06, 1.11] | 0.82 [0.79, 0.84] | 2.08 [2.02, 2.14] | 2.01 [2.00, 2.02] | 2.38 [2.25, 2.55] | 52.43 [43.97, 59.83] |
| amp1 | 10% | 1.10 [1.09, 1.10] | 0.86 [0.81, 0.91] | 0.81 [0.80, 0.82] | 2.02 [2.00, 2.05] | 2.02 [2.00, 2.03] | 2.22 [2.15, 2.30] | 180.67 [151.82, 216.58] |
| amp1 | 3% | 1.10 [1.09, 1.10] | 1.01 [0.93, 1.10] | 0.80 [0.79, 0.81] | 2.01 [1.99, 2.02] | 2.03 [2.01, 2.04] | 2.22 [2.12, 2.36] | 69.28 [58.84, 81.67] |
| amp1 | 1% | 1.10 [1.09, 1.11] | 1.10 [1.02, 1.20] | 0.80 [0.79, 0.81] | 2.02 [1.99, 2.04] | 2.03 [2.02, 2.04] | 2.25 [2.15, 2.39] | 35.32 [30.16, 41.70] |
| depth7 | 10% | 1.06 [0.99, 1.13] | 1.01 [0.92, 1.10] | 0.79 [0.73, 0.86] | 2.59 [2.34, 2.86] | 1.86 [1.65, 2.10] | 2.23 [1.99, 2.49] | 16.63 [13.31, 21.11] |
| depth7 | 3% | 1.05 [0.98, 1.12] | 1.08 [1.00, 1.17] | 0.76 [0.70, 0.84] | 2.54 [2.30, 2.83] | 1.50 [1.35, 1.68] | 2.01 [1.77, 2.30] | 7.26 [6.03, 8.89] |
| depth7 | 1% | 1.07 [1.03, 1.12] | 1.10 [1.03, 1.18] | 0.74 [0.69, 0.80] | 2.42 [2.23, 2.64] | 1.31 [1.19, 1.44] | 1.75 [1.58, 1.95] | 3.93 [3.37, 4.65] |
| depth11 | 10% | 1.12 [1.05, 1.19] | 1.07 [0.96, 1.18] | 0.87 [0.79, 0.95] | 2.42 [2.16, 2.69] | 1.85 [1.65, 2.10] | 2.22 [1.99, 2.49] | 264.36 [211.31, 335.86] |
| depth11 | 3% | 1.08 [0.99, 1.18] | 1.10 [1.02, 1.20] | 0.92 [0.83, 1.02] | 2.64 [2.36, 2.94] | 1.49 [1.34, 1.67] | 2.00 [1.76, 2.29] | 113.92 [94.20, 140.23] |
| depth11 | 1% | 1.13 [1.06, 1.21] | 1.16 [1.08, 1.26] | 0.90 [0.82, 1.00] | 2.58 [2.33, 2.85] | 1.29 [1.18, 1.43] | 1.74 [1.57, 1.97] | 59.91 [50.71, 71.50] |

### 3.5 Frozen rules and sensitivity to how a partial pass is priced

| Cell | R1 amortised beats uniform | R3 co-state weight helps the cheap estimator | D4-1 candidate | R1 semilog | R1 staircase | R3 semilog | R3 staircase |
|---|---|---|---|---|---|---|---|
| base | True | False | False | True | True | False | False |
| m4 | False | False | False | False | False | False | False |
| m64 | False | True | False | True | False | True | False |
| wide | False | False | False | False | False | False | False |
| amp1 | False | False | False | False | False | False | False |
| depth7 | True | False | False | True | True | False | False |
| depth11 | True | False | False | True | True | False | False |

- **R1** holds in `base`, `depth7` and `depth11` (for example `base`: 0.77 [0.69, 0.88] and 0.68 [0.62, 0.76] at the 3 % and 1 % targets). It fails in `m4` (the scorer costs more than it saves), `wide` and `amp1` (uniform is cheaper), and in `m64`, where 0.89 [0.78, 1.01] at 3 % misses by a confidence bound: under semilog pricing `m64` passes R1.
- **R3** holds only in `m64` (0.91 [0.81, 0.99], 0.90 [0.83, 0.97], 0.88 [0.81, 0.94]). In `base`, `depth7` and `depth11` the ratio is 0.97–1.05, and in `m4` 0.88–0.89 with the 3 % CI reaching 1.01.
- **Under semilog pricing `m64` would satisfy both rules** (R1 and R3 both true). The frozen primary pricing is log-log, under which it does not; the decision below follows the frozen rule.

### 3.6 Retained headroom and break-even instance counts

| Cell | Retained headroom: 10 % | 3 % | 1 % | Amortised break-even instances | Cheap-adjoint break-even instances |
|---|---:|---:|---:|---:|---:|
| base | 0.24 | 0.92 | 0.92 | 6,853 | 7.7 |
| m4 | -1.04 | -0.63 | -0.36 | n/a | 2.2 |
| m64 | -0.76 | 0.41 | 0.55 | 10,352 | 9.6 |
| wide | n/a | n/a | n/a | n/a | n/a |
| amp1 | n/a | n/a | n/a | n/a | n/a |
| depth7 | 0.09 | 0.45 | 0.55 | 5,022 | 2.1 |
| depth11 | -0.28 | 0.72 | 0.76 | 9,858 | 30.8 |

Retained headroom is the share of the free exact co-state's advantage over uniform refinement that the learned scorer keeps, from geometric-mean compute (1 is all of it, negative is worse than uniform, n/a where the free arm does not beat uniform). Break-even counts are the one-off cost (scorer training for `amortised`, the co-state table for `cheap_adjoint`) divided by the mean compute saved per instance against uniform at the tightest target, and are n/a where nothing is saved.

### 3.7 Marking fraction and network size (tuning family)

| Cell | Arm | θ = 0.5 | θ = 0.7 | θ = 0.85 | θ = 0.95 | θ = 0.98 | θ = 0.995 | Selected | At grid edge |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| base | residual | 1439 | 1006 | 863 | 937 | 1101 | 1228 | 0.85 | False |
| base | goal_local | 1853 | 1364 | 1143 | 1188 | 1293 | 1336 | 0.85 | False |
| base | adjoint | 1955 | 1457 | 1334 | 1313 | 1295 | 1529 | 0.98 | False |
| base | indicator | 16384 | 16384 | 16384 | 16384 | 16384 | 16384 | 0.5 | True |
| base | cheap | 585 | 449 | 436 | 537 | 626 | 692 | 0.85 | False |
| base | cheap_adjoint | 627 | 471 | 426 | 419 | 480 | 567 | 0.95 | False |
| base | amortised | 691 | 505 | 442 | 502 | 593 | 684 | 0.85 | False |
| m4 | residual | 1489 | 1087 | 1036 | 1010 | 1178 | 1410 | 0.95 | False |
| m4 | goal_local | 1641 | 1244 | 1079 | 1033 | 1069 | 1225 | 0.95 | False |
| m4 | adjoint | 1638 | 1213 | 1074 | 1016 | 1109 | 1345 | 0.95 | False |
| m4 | indicator | 10043 | 9649 | 9383 | 9375 | 9203 | 9222 | 0.98 | False |
| m4 | cheap | 976 | 791 | 731 | 752 | 904 | 1125 | 0.85 | False |
| m4 | cheap_adjoint | 1066 | 830 | 680 | 593 | 633 | 778 | 0.95 | False |
| m4 | amortised | 2244 | 1711 | 1394 | 1545 | 1896 | 2482 | 0.85 | False |
| m64 | residual | 1682 | 1148 | 939 | 932 | 1033 | 1067 | 0.95 | False |
| m64 | goal_local | 2398 | 1621 | 1233 | 1100 | 1103 | 1099 | 0.995 | True |
| m64 | adjoint | 2284 | 1499 | 1241 | 1132 | 1162 | 1263 | 0.95 | False |
| m64 | indicator | 16384 | 16384 | 16384 | 16384 | 16384 | 16384 | 0.5 | True |
| m64 | cheap | 652 | 492 | 457 | 516 | 541 | 564 | 0.85 | False |
| m64 | cheap_adjoint | 660 | 501 | 429 | 403 | 448 | 505 | 0.95 | False |
| m64 | amortised | 679 | 510 | 450 | 486 | 567 | 596 | 0.85 | False |
| wide | residual | 972 | 629 | 464 | 380 | 339 | 325 | 0.995 | True |
| wide | goal_local | 1187 | 771 | 581 | 470 | 414 | 363 | 0.995 | True |
| wide | adjoint | 1137 | 737 | 549 | 454 | 419 | 417 | 0.995 | True |
| wide | indicator | 11787 | 11104 | 10719 | 10419 | 10417 | 10023 | 0.995 | True |
| wide | cheap | 423 | 293 | 232 | 201 | 188 | 183 | 0.995 | True |
| wide | cheap_adjoint | 402 | 288 | 229 | 200 | 187 | 180 | 0.995 | True |
| wide | amortised | 514 | 351 | 272 | 227 | 213 | 203 | 0.995 | True |
| amp1 | residual | 1272 | 863 | 683 | 604 | 587 | 574 | 0.995 | True |
| amp1 | goal_local | 1637 | 1108 | 832 | 751 | 712 | 647 | 0.995 | True |
| amp1 | adjoint | 1565 | 1096 | 892 | 799 | 764 | 706 | 0.995 | True |
| amp1 | indicator | 16384 | 16384 | 16384 | 16384 | 16384 | 16384 | 0.5 | True |
| amp1 | cheap | 548 | 409 | 352 | 331 | 328 | 323 | 0.995 | True |
| amp1 | cheap_adjoint | 550 | 405 | 347 | 319 | 322 | 327 | 0.95 | False |
| amp1 | amortised | 672 | 475 | 390 | 368 | 370 | 353 | 0.995 | True |
| depth7 | residual | 1413 | 985 | 846 | 918 | 1077 | 1199 | 0.85 | False |
| depth7 | goal_local | 1831 | 1353 | 1120 | 1165 | 1318 | 1306 | 0.85 | False |
| depth7 | adjoint | 1907 | 1433 | 1294 | 1297 | 1302 | 1512 | 0.85 | False |
| depth7 | indicator | 4096 | 4096 | 4096 | 4096 | 4096 | 4096 | 0.5 | True |
| depth7 | cheap | 574 | 440 | 428 | 526 | 613 | 675 | 0.85 | False |
| depth7 | cheap_adjoint | 616 | 462 | 418 | 411 | 472 | 555 | 0.95 | False |
| depth7 | amortised | 647 | 478 | 435 | 514 | 603 | 695 | 0.85 | False |
| depth11 | residual | 1441 | 1007 | 864 | 938 | 1103 | 1230 | 0.85 | False |
| depth11 | goal_local | 1856 | 1366 | 1144 | 1190 | 1295 | 1338 | 0.85 | False |
| depth11 | adjoint | 1957 | 1459 | 1336 | 1315 | 1296 | 1531 | 0.98 | False |
| depth11 | indicator | 65536 | 65536 | 65536 | 65536 | 65536 | 65536 | 0.5 | True |
| depth11 | cheap | 586 | 449 | 436 | 537 | 627 | 693 | 0.85 | False |
| depth11 | cheap_adjoint | 628 | 472 | 426 | 419 | 481 | 567 | 0.95 | False |
| depth11 | amortised | 710 | 538 | 497 | 559 | 657 | 751 | 0.85 | False |

Compute at the top of the widened grid (0.995) is best in `wide` and `amp1` for almost every arm, which means marking nearly everything, i.e. close to uniform refinement, and is consistent with adaptivity not paying there. The forcing-only `indicator` never reaches a target, so its rows sit at the compute cap for every `θ` and its selected value is arbitrary.

## 4. Interpretation

**Observed** (from the tables above):

- The price of a small learned scorer relative to step-doubling scoring falls with dimension: 2.24 (`m` = 4), 0.245 (`m` = 32), 0.133 (`m` = 64).
- Where adaptive refinement pays at all (the `m` = 32 narrow-pulse cells and `m` = 64 at the two tighter targets, `m` = 4 for the cheap arms at the two tighter targets), a non-learned cheap arm pays at least as much as the learned scorer: `amortised ÷ cheap` is 1.02–1.13 wherever the scorer is affordable, with CIs that include 1 in some cells and exclude it in the scorer's disfavour in others.
- The co-state weight on a cheap estimator matters little: `cheap_adjoint ÷ cheap` is 0.97–1.05 in `base`, `depth7` and `depth11`, 1.01–1.03 in `wide`, 1.00–1.27 in `amp1` (worst at the 10 % target, 1.27 [1.21, 1.35]), 0.88–0.89 in `m4` and 0.88–0.91 in `m64`.
- In `base` the learned scorer matches the hypothetical free exact adjoint (`adjoint_free ÷ amortised` 0.92 [0.81, 1.04], 0.98 [0.88, 1.07], 0.97 [0.89, 1.04]); in `m64`, `depth7` and `depth11` some headroom remains (ratios 0.74–0.92).
- The step-doubling `adjoint` arm at its real price needs 1.04 to 2.62 times uniform's compute in every cell, as in D4-0b.
- In `wide` and `amp1`, no adaptive arm is detectably cheaper than uniform refinement at any target; the free exact adjoint ties (`adjoint_free ÷ uniform` 1.00–1.03), and every arm at a real price needs more (1.13 or more).
- The cost of learning is large: 1.8 million to 35.8 million CN steps of training, which is 5,022 to 10,352 instances of break-even where it saves anything, against 2.1 to 30.8 instances for the tabulated co-state of `cheap_adjoint`.

**Hypotheses (not tested here):**

1. *The learned scorer is limited by what its features carry, not by its size.* Its features are the ones the cheap arm uses plus a few logs, and its predictions are at best as good as `cheap`; a richer feature set (for example a neighbourhood of solution values) might close the gap but would raise its price. **Diagnostic:** a new design with a stated price budget.
2. *The co-state weight is worth little here because the goal is fixed and the forcing pulses are few.* With one goal the weight is a fixed function of time; when the goal varies per instance a goal-blind estimator has nothing to learn the weight from. **Diagnostic:** repeat the non-learned pair `cheap` versus `cheap_adjoint` on instances with a goal that varies per instance (no training needed, but the co-state table becomes per goal).
3. *`m` = 64 is the cell to revisit.* It has the lowest price, the only clear co-state gain, and misses R1 by one confidence bound at one target. **Diagnostic:** more validation instances in that cell under a new, separately frozen design (this run's rule is not reopened).

## 5. Design issues found

1. **`m` is confounded with the system draw.** Each `m` has its own random system (seed `7000 + m`), so the `m4`, `base` and `m64` cells differ in more than dimension.
2. **The FLOP price is a count, not a runtime.** `tests/correctness.json` records that in this NumPy implementation the features and network for 1,024 intervals at `m` = 64 took about 2.9 times as long as one CN step on all of them, where the ledger prices the scorer at 0.27 steps per interval, i.e. about 0.27 times. Interpreted as wall-clock time the learned scorer would be about ten times more expensive than the ledger says. The result is a statement about operation counts.
3. **The forcing-only `indicator` mostly cannot reach a target** (it ignores the homogeneous error), so its ratios to uniform (3.9–264) are censored at the cap and understate its true cost; it is a diagnostic.
4. **The frozen R1 in `m64` is sensitive to the pricing of a partial pass** (fails under log-log and staircase, passes under semilog).
5. **`θ` is at the edge of the widened grid in `wide` and `amp1`.** Their best marking fraction is 0.995, close to marking everything. The tuned policies there may still be understated by the grid.
6. **The scorer is trained on 30 instances per cell and evaluated on the same system.** Its validation RMSE is 0.40–0.60 in log10 of the weighted error. Generalisation to a new system was not tested.
7. **The exact-label generation for training is not charged**, which understates the scorer's one-off cost.
8. **The scorer sees the goal only through time.** With a fixed goal the co-state weight is a fixed function of time; this is the simplification that D4-1 would remove.
9. **The forcing indicator and the solution-difference estimate are hand-built heuristics**; other cheap estimators might change the picture.
10. **`base`, `depth7` and `depth11` are near copies of one configuration.** They differ only in the finest grid, so R1 holding in all three is one finding, not three.
11. **A smoke test printed validation medians for four instances of two cells**, disclosed in §1.

## 6. Commands

Every table in §3 is printed from committed files:

```bash
python scripts/d4_2_tables.py results/runs/d4_2_flop_scoring_20260929T233556Z
```

The count of `θ` choices at the grid edge among the six non-diagnostic arms:

```bash
python - <<'EOF'
import csv
rows = [r for r in csv.DictReader(open("results/runs/d4_2_flop_scoring_20260929T233556Z/artifacts/theta_tuning.csv")) if r["arm"] != "indicator"]
print(sum(r["selected_at_grid_edge"] == "True" for r in rows), "of", len(rows))
EOF
```

The wall-clock figures in §5 are in `tests/correctness.json` under `wall_clock_sanity_seconds`.

## 7. What this run does and does not support

**Supports:**

- The FLOP price of cheap and learned scorers on this analytic family and its fall with `m` (§3.1).
- At those prices, adaptive refinement with a cheap non-learned estimator needs less compute than uniform refinement in the narrow-pulse cells at the tighter targets, and does not in the `wide` and `amp1` cells (§3.3).
- A small learned scorer adds nothing over the cheap estimator in any cell of this family, and costs thousands of instances to repay its training (§3.3, §3.4, §3.6).
- The co-state weight helps the cheap estimator in `m` = 64 only, by about 10 %, and not in the cells where the learned scorer pays (§3.5).
- The frozen rules yield no D4-1 candidate cell. The run reproduces from a clean checkout on the same machine: every scorer file, the tuning and validation tables and the manifest are byte-identical between two executions at the same commit, and the other files differ only in run id, a timestamp or a timing ([`reproduction.md`](../../results/runs/d4_2_flop_scoring_20260929T233556Z/reproduction.md); one repetition, same machine and thread count).

**Does not support:**

- Any statement about a learned direct critic or a co-state-featured critic (H2).
- Any statement about a varying goal: the goal is fixed per system.
- Any statement about wall-clock speed, hardware, real data, other domains or the test family.
- A claim that a learned scorer cannot pay under a different feature set, network or training regime.
- Any claim that `m` alone causes the differences between the `m4`, `base` and `m64` cells.

**Corrects earlier statements.**

- The D4-0b note's hypothesis 2 said the co-state pays only when its sweep is made cheap, and its recommendation implied a learned scorer was the way to make it cheap. With FLOP pricing, a non-learned tabulated co-state on a cheap estimator does that at least as well, so the learned scorer is not needed for the price question.
- The roadmap (§2.1) said D4-1 "may open for a regime that D4-2 shows at scoring price ≤ ×0.25". The price condition is met at `m` ≥ 32, but the frozen rule requires R1 and R3 in the same cell and none has both, so it does not open.

## 8. Next steps (roadmap IDs in `docs/plans/roadmap.md`)

1. **D4-1 stays closed** under the frozen rule. `m` = 64 is the one cell to revisit, only in a new, separately frozen design (§4, hypothesis 3).
2. **The remaining D4 question that matters for H2 is the varying goal**, checked first with the non-learned pair `cheap` versus `cheap_adjoint` (hypothesis 2). If the co-state weight does not matter there either, D4 offers no H2 headroom and should be de-prioritised.
3. **Return to the real-data path** (`E1.1`, then `B2`) and to the licence-cleared domain cards (D1-0 with SMD, D2-0 with Qwen3 and MuSiQue or Qasper), for which D4-2 offers one design lesson: price every scorer in the same units as the solver, and compare it with a competent non-learned baseline.
