# D1-0b: a forecasting objective for sensing allocation on SMD (Rung-0 redesign, validation machines only)

**Run:** `d1_0b_forecast_sensing_20260930T085746Z` · **Date:** 2026-09-30 · **Notebook:** `notebooks/04-domains/d1_0b_forecast_sensing.ipynb` (SHA-256 `38277ad8…c2849`)
**Hardware:** CPU in the Claude Code sandbox · **Package commit:** `2b3bb994…` (clean tree) · **Config SHA-256:** `d3fd7272…c50d82f` · **Frozen-before-validation SHA-256:** `23d6b37d…e0696` · **Data manifests:** tuning `8692ee22…69a46`, validation `077ee8c3…e5e69e`
**Artefacts:** [`results/runs/d1_0b_forecast_sensing_20260930T085746Z/`](../../results/runs/d1_0b_forecast_sensing_20260930T085746Z/) · **Plan:** [`docs/plans/d1-0b-plan.md`](../plans/d1-0b-plan.md) · **Follows:** [D1-0](2026-09-30-d1-0-sensor-opportunity.md)

**Status:** exploratory Rung-0 study on seven validation machines; **the 14 test machines were never downloaded.** Correctness ✅ · **G0 (the forecast error rises when channels are held instead of observed): ✅ on the tuning machines (r = 0.522); ✅ on validation by the frozen point rule but not robust: r = 0.298 [0.023, 0.743], and one validation machine is negative (−0.174)** · **G1 (oracle headroom over the best fixed allocation): passes by the frozen rule, 0.875 [0.295, 1.347], but the ratio is unstable** (interval above 1; [−1.427, 2.982] at `L` = 120) · **G2: neither deployable dynamic policy keeps any of the headroom** (−0.24 [−5.54, 1.28] and −0.10 [−5.04, 1.18]); the best static policy beats both · value-of-information comparison inconclusive · no learned critic, no co-state, **no H2 statement.** This note recommends **not** designing D1-1 on this domain.

**In one paragraph.** With the loss changed to the native endpoint (a frozen linear forecaster's 5-minute-ahead error against the true future), holding instead of observing raises the forecast error by 30 % on average over the validation machines, unevenly: from −17 % on one machine to +75 % on another. A privileged oracle that sees the window's states, but not the targets, closes almost all of that gap with a handful of channels (normalised excess 0.105 with one channel open, 0.047 with eight), while the best fixed allocation closes some of it (area 0.190 against 0.298 for hold) and **no deployable policy does better than the best fixed one**: `recent_change` is at 0.230 and `recent_change_weighted` at 0.206, against 0.169 for the best static policy on validation, `top_sensitivity`. Opening a few channels can make the forecast much worse than holding all of them (excess 1.97 with one channel for `top_sensitivity`, against 0.298 for hold), because the linear forecaster expects a consistent snapshot. The forecaster itself has little skill on these test streams (mean error 1.02 in units where the training variance is 1), which limits what sensing can add. The window-start snapshot therefore carries little usable information about which channels will matter later in the hour beyond static sensitivity and volatility, and with a linear forecaster the sensitivity is a fixed matrix row, so no state-dependent co-state advantage can arise here.

---

## 1. What this run is

- **Data:** SMD as in D1-0 (MIT, register id `smd`; per-file SHA-256 in the two manifests are **identical to D1-0's**): 7 tuning, 7 validation, 14 test machines (never downloaded), split by index mod 4 before windowing. **The validation machines are the ones D1-0 used**, and D1-0's validation results (the detector's poor labelled F1) motivated this redesign, so this validation set is not pristine for the choice of objective; the forecasting numbers themselves had not been computed on any machine before the freeze. The test machines remain reserved for a confirmatory stage.
- **Problem:** a 60-minute window starts with a snapshot of all 38 channels; the allocator opens `k` channels (observed every minute) and the rest are held at the snapshot. 2,998 validation windows.
- **Loss (the native endpoint):** the mean over the window's minutes `t = 1 … 54` and channels of the squared error of `ŷ_{t+5} = z̃_t B` against the true `z_{t+5}`, `z` standardised with floor 0.05, `B` a ridge regression (`λ = 1`) fitted on the machine's training stream. Reported as the **excess over full observation**, divided by the machine's full-observation loss: `0` means as good as observing everything, and a machine's hold-only value is its `r`.
- **Policies (non-learned):** fixed `round_robin`, `top_volatility`, `top_sensitivity` (`‖B[c, :]‖²`), `top_weighted_volatility`; dynamic `recent_change`, `recent_change_weighted`; privileged `oracle_fidelity` (per window and size, the better of greedy forward selection and backward elimination of the subset whose forecast is closest to the full-information forecast; it sees the window states, not the targets). The **reference** is the oracle's curve alone, with unused budget legal.
- **Gates and rules** (plan §6, frozen in `reports/frozen_before_validation.json`): G0, `r ≥ 0.05`; G1, relative headroom ≥ 0.15; G2 and the value-of-information comparisons descriptive. The best fixed policy per setting was chosen on the tuning machines (`top_weighted_volatility` at the primary setting).
- **Design history (disclosed).** Dated in the plan §10: after a first smoke run on tuning machines crashed because one machine has a negative hold-only excess at `h` = 15, the normalisation was changed from the hold-only excess to the full-observation loss (before any validation machine was read; only tuning numbers had been computed). Two smoke runs on tuning machines preceded the real run; they read no validation machine as validation (the second was fed tuning machines in both roles).

## 2. Gates

| Gate | Result | Source |
|---|---|---|
| G3: split partition; fetch refuses a test machine; `B` equals the ridge solution; full-observation loss equals a direct computation; fidelity is 0 at full observation; the incremental update equals a direct recomputation (both directions, 7.1e-15); deployable policies ignore the window after the snapshot; budget 38 is full observation for every policy | ✅ | `tests/correctness.json` |
| Test machines never downloaded; frozen choices written before the validation losses were computed | ✅ (the validation *files* were already in the local cache from D1-0) | `config/run_config.json`, `config/data_manifest_validation.json` |
| **G0 on the tuning machines (stage A)**, `r` at `h` = 5, `L` = 60 | **✅ 0.522** (per machine 0.26, 0.66, 0.27, 1.27, 0.62, 0.013, 0.55) | `artifacts/tuning_summary.json` |
| **G0 on validation** | **✅ by the point rule, not robust: 0.298 [0.023, 0.743]**; machines below 0.05: 1 of 7 (−0.174) | `artifacts/validation_summary.json` |
| **G1** at `h` = 5, `L` = 60 | **passes: 0.875 [0.295, 1.347]** (lower bound above 0.15; the interval exceeds 1, so the ratio is unstable) | `artifacts/validation_summary.json` |
| G1 at the four sensitivities | pass by the point rule at all four (0.896, 0.940, 0.876, 0.584); interval lower bounds 0.477, 0.750, 0.794, −1.427 (`L` = 120 is not robust) | `figures/phase_table.csv` |
| **G2** | **neither dynamic policy keeps any headroom** | `artifacts/validation_summary.json` |

## 3. Results

Loss curves are excess forecast error over full observation, divided by the machine's full-observation error, averaged over machines. Every table is printed by `scripts/d1_0b_tables.py` from the committed run files (§6).

### 3.1 Stage A: the endpoint-moves statistic and the fixed policies on the tuning machines

| Setting | Endpoint moves `r` (mean over tuning machines) | Best fixed | round_robin area | top_volatility area | top_sensitivity area | top_weighted_volatility area |
|---|---:|---|---:|---:|---:|---:|
| h15_L60 | 0.227 | top_weighted_volatility | 0.3677 | 0.2244 | 0.1710 | 0.1602 |
| h1_L60 | 1.018 | top_weighted_volatility | 1.3371 | 0.9878 | 0.6750 | 0.5591 |
| h5_L120 | 1.027 | top_weighted_volatility | 0.9861 | 0.5100 | 0.4759 | 0.4264 |
| h5_L30 | 0.303 | top_sensitivity | 0.3983 | 0.2896 | 0.1906 | 0.2216 |
| h5_L60 | 0.522 | top_weighted_volatility | 0.5574 | 0.3493 | 0.2687 | 0.2650 |

| Tuning machine | Windows | Endpoint moves `r` |
|---|---:|---:|
| machine-1-1 | 472 | 0.260 |
| machine-1-5 | 393 | 0.662 |
| machine-2-1 | 392 | 0.273 |
| machine-2-5 | 392 | 1.270 |
| machine-2-9 | 476 | 0.623 |
| machine-3-4 | 392 | 0.013 |
| machine-3-8 | 476 | 0.552 |

### 3.2 Gates G0 and G1 and the pre-registered sensitivities (validation machines)

| Setting | Best fixed | Endpoint moves `r` [95 % CI] | G0 (≥ 0.05) | Area: fixed | Area: oracle | Relative headroom [95 % CI] | G1 (≥ 0.15) | Note |
|---|---|---|---|---:|---:|---|---|---|
| h15_L60 | top_weighted_volatility | 0.203 [-0.028, 0.550] | pass | 0.2273 | 0.0137 | 0.940 [0.750, 1.062] | pass | robust |
| h1_L60 | top_weighted_volatility | 0.425 [-0.051, 1.420] | pass | 0.3171 | 0.0331 | 0.896 [0.477, 1.254] | pass | robust |
| h5_L120 | top_weighted_volatility | 0.322 [0.069, 0.765] | pass (robust) | 0.0805 | 0.0334 | 0.584 [-1.427, 2.982] | pass | not robust |
| h5_L30 | top_sensitivity | 0.146 [-0.038, 0.407] | pass | 0.1171 | 0.0146 | 0.876 [0.794, 0.976] | pass | robust |
| h5_L60 | top_weighted_volatility | 0.298 [0.023, 0.743] | pass | 0.1900 | 0.0238 | 0.875 [0.295, 1.347] | pass | robust |

### 3.3 Policies in the primary run

| Policy | Kind | Normalised excess area [95 % CI] | Headroom kept (fixed to oracle) [95 % CI] | Fidelity area (the oracle's objective) |
|---|---|---|---|---:|
| round_robin | fixed | 0.5741 [0.0321, 1.5490] |  | 1.3893 |
| top_volatility | fixed, uncertainty-only | 0.1908 [-0.0324, 0.5911] |  | 0.6386 |
| top_sensitivity | fixed, sensitivity-only | 0.1693 [0.0439, 0.3441] |  | 0.3798 |
| top_weighted_volatility (best fixed) | fixed, sensitivity × uncertainty | 0.1900 [-0.0299, 0.5505] |  | 0.5867 |
| recent_change | dynamic, uncertainty-only | 0.2296 [-0.0336, 0.6902] | -0.24 [-5.54, 1.28] | 0.6508 |
| recent_change_weighted | dynamic, sensitivity × uncertainty | 0.2062 [-0.0232, 0.5309] | -0.10 [-5.04, 1.18] | 0.5612 |
| oracle_fidelity | privileged | 0.0238 [0.0072, 0.0508] |  | 0.0683 |

### 3.4 Paired differences of areas (G2 and the value-of-information check)

| Paired difference of normalised areas (negative: first is better) | h15_L60 | h1_L60 | h5_L120 | h5_L30 | h5_L60 |
|---|---|---|---|---|---|
| oracle_fidelity - best fixed | -0.2137 [-0.5349, 0.0233] | -0.2841 [-1.0375, 0.0567] | -0.0470 [-0.2523, 0.0907] | -0.1025 [-0.2094, -0.0211] | -0.1662 [-0.5149, 0.0424] |
| recent_change - best fixed | +0.0228 [-0.3543, 0.4337] | -0.0524 [-0.8241, 0.3998] | +0.0462 [-0.1313, 0.1597] | +0.0336 [-0.0401, 0.1599] | +0.0396 [-0.3715, 0.3940] |
| recent_change_weighted - recent_change | +0.0119 [-0.1941, 0.1355] | +0.0158 [-0.1768, 0.1001] | +0.0412 [-0.0407, 0.1538] | +0.0084 [-0.0572, 0.0837] | -0.0234 [-0.2018, 0.0633] |
| recent_change_weighted - top_sensitivity | -0.0131 [-0.2307, 0.2681] | -0.0034 [-0.2411, 0.2607] | +0.0108 [-0.1109, 0.2024] |  | +0.0370 [-0.0904, 0.2232] |
| recent_change_weighted - best fixed | +0.0347 [-0.3110, 0.3355] | -0.0366 [-0.7969, 0.3196] | +0.0874 [-0.1269, 0.2700] | +0.0421 [-0.0443, 0.1670] | +0.0162 [-0.3458, 0.2564] |
| top_weighted_volatility - top_sensitivity | -0.0478 [-0.2102, 0.1794] | +0.0332 [-0.2127, 0.6389] | -0.0767 [-0.1848, 0.1111] |  | +0.0208 [-0.1245, 0.3398] |
| top_weighted_volatility - top_volatility | +0.0619 [-0.0244, 0.2046] | +0.0233 [-0.0739, 0.1196] | -0.0569 [-0.1758, 0.0079] | +0.0004 [-0.0461, 0.0540] | -0.0008 [-0.0684, 0.0930] |
| top_weighted_volatility - best fixed |  |  |  | -0.0283 [-0.1151, 0.1444] |  |

### 3.5 Machines

| Machine | Windows | Endpoint moves `r` | Headroom (best fixed to oracle) |
|---|---:|---:|---:|
| machine-1-2 | 392 | 0.506 | 0.956 |
| machine-1-6 | 392 | 0.084 | 0.735 |
| machine-2-2 | 393 | 0.746 | 0.779 |
| machine-2-6 | 477 | 0.161 | 0.838 |
| machine-3-1 | 476 | -0.174 | nan |
| machine-3-5 | 392 | 0.099 | 0.801 |
| machine-3-9 | 476 | 0.666 | 0.958 |

### 3.6 Mean normalised excess forecast error by budget (primary run)

| Policy | k = 0 | k = 1 | k = 2 | k = 4 | k = 8 | k = 12 | k = 16 | k = 24 | k = 38 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round_robin | 0.298 | 0.376 | 1.517 | 0.646 | 0.479 | 0.607 | 0.766 | 0.740 | 0.000 |
| top_volatility | 0.298 | 0.341 | 0.280 | 0.297 | 0.363 | 0.301 | 0.281 | 0.101 | 0.000 |
| top_sensitivity | 0.298 | 1.974 | 0.351 | 0.211 | 0.438 | 0.142 | 0.088 | 0.027 | 0.000 |
| top_weighted_volatility | 0.298 | 1.965 | 0.345 | 0.357 | 0.424 | 0.135 | 0.102 | 0.061 | 0.000 |
| recent_change | 0.298 | 0.364 | 0.533 | 1.228 | 0.150 | 0.137 | 0.225 | 0.112 | 0.000 |
| recent_change_weighted | 0.298 | 1.854 | 0.382 | 0.217 | 0.174 | 0.287 | 0.271 | 0.104 | 0.000 |
| oracle_fidelity | 0.298 | 0.105 | 0.093 | 0.068 | 0.047 | 0.020 | 0.006 | 0.000 | 0.000 |

### 3.7 Derived quantities quoted in this note

- Validation windows (primary setting): 2998 over 7 machines.
- Mean native forecast error (standardised units squared, per channel-minute) with every channel observed: 1.0220 (machine range 0.3180 to 2.7612); hold only: 1.2090 (range 0.5188 to 3.0356).
- Endpoint moves `r` per machine: min -0.174, max 0.746; machines below 0.05: 1 of 7.
- Deployable policies with a lower normalised area than the privileged oracle (primary): none.
- Headroom against the validation-best fixed policy: 0.859 (top_sensitivity, area 0.1693); the tuning-chosen best fixed (top_weighted_volatility) has 0.1900.

## 4. Interpretation

**Observed** (from the tables above):

- **The endpoint moves, weakly and unevenly.** Holding instead of observing raises the forecast error by 0.298 on average (from −0.174 to +0.746 across the validation machines), with an interval that just clears the 0.05 threshold at its lower end. At `h` = 1 it is 0.425 and at `h` = 15 it is 0.203, both with intervals that include 0.
- **The privileged headroom is large.** The oracle reaches 0.105, 0.093, 0.068 and 0.047 of the normalised excess with 1, 2, 4 and 8 channels; its area is 0.0238 against 0.1900 for the tuning-chosen best fixed policy (headroom 0.875) and 0.1693 for the best fixed policy on validation, `top_sensitivity` (headroom 0.859).
- **No deployable policy exploits it.** `recent_change` (area 0.2296) and `recent_change_weighted` (0.2062) are not better than the best fixed policy (paired differences +0.0396 [−0.3715, 0.3940] and +0.0162 [−0.3458, 0.2564]), and `top_sensitivity` (0.1693) is lower than both. No deployable policy has a lower area than the oracle.
- **The value-of-information check is inconclusive.** `recent_change_weighted − recent_change` is −0.0234 [−0.2018, 0.0633]; `top_weighted_volatility − top_sensitivity` is +0.0208 [−0.1245, 0.3398]; `recent_change_weighted − top_sensitivity` is +0.0370 [−0.0904, 0.2232]. Every interval includes 0.
- **A few opened channels can hurt.** With one channel open, `top_sensitivity` and `top_weighted_volatility` have excess 1.97 and 1.97 against 0.298 for hold, and `round_robin` has 1.52 with two; the error falls below hold only from about 12 channels for the sensitivity policies.
- **The forecaster has little skill here.** The mean full-observation error is 1.022 (machine range 0.318 to 2.761) in units where the training variance is 1.
- **Heterogeneity dominates.** Per-machine headroom is 0.735 to 0.958 on six machines; the seventh (`machine-3-1`, `r` = −0.174) has no defined headroom because holding is better than observing there.

**Hypotheses (not tested here):**

1. *Partial snapshots hurt because the forecaster was fitted on consistent states.* Opening a few channels makes the input inconsistent (some fresh, some up to an hour old). **Diagnostic:** a forecaster trained on stale-and-fresh mixtures, or an allocation that keeps the set of opened channels consistent across windows.
2. *The forecaster's weak skill limits the value of sensing.* **Diagnostic:** compare its error with persistence and with the training mean on each machine, and repeat with a richer forecaster (several lags), on tuning machines first.
3. *A nonlinear downstream model would make the sensitivity state-dependent,* which is where a co-state could differ from a static weight. **Diagnostic:** not testable here; it needs a nonlinear frozen downstream model.

## 5. Design issues found

1. **The validation machines were used by D1-0, and D1-0's validation outcome motivated D1-0b.** The forecasting losses were computed on them for the first time here, but the choice of objective is not independent of what D1-0 showed. The test machines are untouched.
2. **The ratio measures are unstable.** When a machine has a hold-only excess near or below zero, its normalised curve is near or below zero, and ratios of averaged areas (relative headroom, retained fractions) have intervals that run past 1 or below 0: headroom 0.875 [0.295, 1.347], retained fractions [−5.54, 1.28] and [−5.04, 1.18], and `L` = 120 headroom [−1.427, 2.982]. The frozen rule's "robust" flag at the primary setting is therefore weaker than it looks; the absolute areas and the paired differences are more stable.
3. **The forecaster was not compared with persistence or the mean,** and its full-observation error is about 1 in standardised units.
4. **The normalisation was changed after a smoke crash** (plan §10); the change is logged and was made before any validation read.
5. **The best fixed policy was chosen on the tuning machines** (`top_weighted_volatility`) and is not the best on validation (`top_sensitivity`); the headroom barely changes (0.875 against 0.859).
6. **The oracle is privileged and greedy** and optimises the fidelity of the forecast, not the native error; its native error is what is reported.
7. **A linear forecaster** has a state-independent sensitivity, so this domain cannot show a co-state advantage.
8. **Policy computation and the window-start snapshot are not charged.**
9. **Seven machines, one dataset, one operator.**
10. **Smoke runs on tuning machines** preceded the real run (§1).

## 6. Commands

Every table in §3 is printed from committed files, and the primary setting's areas and headroom are recomputed from the per-window parquet and asserted equal to the stored summary:

```bash
python scripts/d1_0b_tables.py results/runs/d1_0b_forecast_sensing_20260930T085746Z
```

The reproduction from a clean worktree is described in the run README.

## 7. What this run does and does not support

**Supports:**

- On these machines, with this frozen linear forecaster, holding instead of observing every channel raises the native forecast error by about 30 % on average, unevenly, and in one machine lowers it (§3.2, §3.5).
- A privileged oracle that sees the window states could remove almost all of that excess with a few channels (§3.3).
- No non-learned deployable policy tried here, dynamic or static, beats the best static policy, and opening a few channels can be worse than holding (§3.3, §3.6).
- The value-of-information comparison cannot separate sensitivity times uncertainty from either factor at seven machines (§3.4).

**Does not support:**

- Any statement about H2, a co-state, a learned direct critic or a learned critic with gradient features.
- Any claim that adaptive sensing beats fixed allocation on this domain, or that a learned policy could not.
- Any statement about the forecaster's usefulness, other datasets, nonlinear downstream models, wall-clock time, energy or the 14 test machines.

**Corrects earlier statements.**

- The D1-0 note (§8, step 1) proposed D1-0b as a redesign "so that the labelled endpoint moves with sensing", to be tested first with a cheap diagnostic. The forecasting endpoint does move on average, but weakly, unevenly and with one negative machine, and the plan's rule (G0 and G1 pass, so a D1-1 design may be written) is met only by the point rules. This note recommends against writing D1-1 here.

## 8. Next steps (roadmap IDs in `docs/plans/roadmap.md`)

1. **Do not open D1-1 on SMD forecasting.** Record D1 as a Rung-0 result without a deployable opportunity for H2: a privileged oracle has headroom, no deployable policy keeps any, and the downstream model is linear. D1 remains a reusable sensing-allocation testbed for a later nonlinear downstream model.
2. **If D1 is to be kept,** the cheap diagnostics in §4 (forecaster against persistence; a forecaster trained on mixed-freshness inputs; several lags) come first and only on tuning machines.
3. **The effort moves to D4-1** (reopened by D4-3 in `m4` and `m64`) and to the Colab items (B2 probe).
