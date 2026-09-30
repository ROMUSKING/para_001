# D1-0: sensing allocation on the Server Machine Dataset (Rung-0 gate, validation machines only)

**Run:** `d1_0_sensor_opportunity_20260930T062900Z` · **Date:** 2026-09-30 · **Notebook:** `notebooks/04-domains/d1_0_sensor_opportunity.ipynb` (SHA-256 `bacede2b…0178a`)
**Hardware:** CPU in the Claude Code sandbox · **Package commit:** `7c5c89c5…` (clean tree) · **Config SHA-256:** `3c9fb6c7…4fd78f` · **Frozen-before-validation SHA-256:** `d128a670…1d0d95` · **Data manifests:** tuning `44bff97c…35d222`, validation `424bbed7…bb6a5f`
**Artefacts:** [`results/runs/d1_0_sensor_opportunity_20260930T062900Z/`](../../results/runs/d1_0_sensor_opportunity_20260930T062900Z/) · **Plan:** [`docs/plans/d1-0-plan.md`](../plans/d1-0-plan.md) · **Follows:** [licence survey, pass 2](../licences/survey-2026-09-29-pass2.md) (register id `smd`, MIT) and [D4-2](2026-09-30-d4-2-flop-priced-scoring.md)

**Status:** exploratory Rung-0 gate on seven validation machines; **the 14 test machines were never downloaded.** Correctness ✅ · **G1 (opportunity against the best fixed allocation, privileged reference) ✅, robust: relative headroom 0.920 [0.886, 0.949]** · G2: three non-learned deployable policies keep 0.42 to 0.49 of that headroom and cannot be told apart · **native endpoint ❌: detection F1 against the dataset labels does not improve with more sensing (0.259 with hold only, 0.192 with every channel observed), so the fidelity loss that G1 measures does not track the labelled task** · no learned critic, no co-state, **no H2 statement.**

**In one paragraph.** On seven validation machines, a privileged allocation that knows which channels will move within a 60-minute window removes almost all of the score-fidelity loss with a handful of channels (normalised loss 0.390 with 1 of 38 channels opened, 0.073 with 4, 0.033 with 8), while the best fixed allocation removes almost none (0.947 with 4, 0.778 with 8). The gate of the plan passes at every window length tried and under the decision-level objective. Simple deployable policies that look only at the window-start snapshot keep 0.42 to 0.49 of the gap between the best fixed allocation and the privileged reference, and "sensitivity times volatility" is no better than either factor alone (paired area differences +0.0238 [−0.0573, 0.1340] against `recent_change` and +0.0369 [−0.0153, 0.1266] against `sensitivity`). **But the loss that passes the gate is fidelity to a frozen detector that is poor at the labelled task** (mean per-machine F1 of 0.141, false alarms on 12 % to 48 % of unlabelled minutes on five of seven machines), and the labelled F1 is flat or falls as sensing rises. The gate therefore shows room to allocate sensing for that fidelity, not room to improve anomaly detection. D1-1 (a learned critic against a gradient-featured critic) should not be designed on this loss.

---

## 1. What this run is

- **Data:** SMD (the `NetManAIOps/OmniAnomaly` repository, MIT): 28 server machines, 38 channels at one-minute sampling, each with a normal training stream, a test stream and per-minute anomaly labels for the test stream. **Real data, not committed;** per-file SHA-256 in the two data manifests. **Split by machine, before windowing:** index mod 4 = 0 tuning (`machine-1-1, 1-5, 2-1, 2-5, 2-9, 3-4, 3-8`), 1 validation (`1-2, 1-6, 2-2, 2-6, 3-1, 3-5, 3-9`), 2 or 3 test (14 machines, never downloaded).
- **Problem:** a window is 60 minutes of a machine's test stream, starting with a snapshot of all 38 channels. The allocator opens `k` channels (observed at every minute) and the rest are held at their snapshot value. 2,998 validation windows in total (392 to 477 per machine); the first two windows of each stream are dropped.
- **Loss `J`:** the mean over the window's minutes of the squared difference between `log(1 + s)` on the partially observed stream and on the fully observed stream, `s` being the score of a frozen Mahalanobis detector fitted on the machine's training stream (shrinkage 0.1; alarm threshold at the 0.995 quantile of its training scores). Lower is better; `J = 0` when every channel is opened. **This is a proxy for a downstream decision, not the labelled task.**
- **Detector constants:** the floor and ridge were chosen on the seven tuning machines, by the fully observed detector's mean F1 against the tuning labels, from a 2 × 3 grid: **floor 0.01, ridge 10⁻⁶** (mean F1 0.3205; two other settings are within 0.0009).
- **Policies (non-learned):** fixed: `round_robin`, `top_volatility`, `top_weighted_volatility`; dynamic, deployable at the window start: `recent_change`, `sensitivity` (`|P z|` at the snapshot), `sensitivity_x_volatility`; privileged: `oracle_greedy` (per window and budget, the better of greedy forward selection and greedy backward elimination, using the true window values). The **reference** at each (window, budget) is the minimum over the oracle at every size up to the budget and over all deployable policies. The **best fixed allocation** is the fixed policy with the lowest mean normalised area on the tuning machines (`top_volatility` at every window length).
- **Budgets:** `k` ∈ {0, 1, 2, 4, 8, 12, 16, 24, 38} channels opened per window. Policy computation is **not charged**.
- **Measures:** per machine, the mean loss curve divided by the machine's hold-only loss, its area over `k / 38`, averaged over machines; relative headroom = `(A_fixed − A_reference) / A_fixed`; a two-stage bootstrap (machines resampled, then windows within them, 10,000 resamples, seed 0) for 95 % intervals.
- **Design history (disclosed).** All changes are dated in the plan §10. In short: the detector's floor and ridge are chosen on the tuning machines because the originally written constants (floor 0.01, ridge 10⁻⁶) gave a condition number of 1.4 × 10⁷ and alarms on 35.5 % of unlabelled minutes on the one tuning machine that was probed (`machine-1-1`); the oracle was widened to forward-or-backward greedy after a smoke run on tuning machines showed forward-only greedy stalling at about 0.10 of the hold-only loss while deployable policies kept falling. **No validation machine had been downloaded when either change was made:** the freeze was written at 06:29:18.9 UTC and the first validation file was retrieved at 06:29:20.1 UTC. **The grid then selected the originally written constants (floor 0.01, ridge 10⁻⁶), by a hair:** the mean tuning F1 is 0.3205 for that pair, 0.3199 for (0.05, 10⁻⁶) and 0.3196 for (0.01, 0.1), so the step did not change the detector, and the smoke-run choice (floor 0.01, ridge 0.1, three machines) did not fix it.

## 2. Gates

| Gate | Result | Source |
|---|---|---|
| G3: split is a partition; the fetch code refuses a test machine; `P` symmetric positive definite; `τ` is the training quantile; full observation has zero loss; the incremental greedy update equals direct recomputation (both directions, both objectives); deployable policies ignore the window after the snapshot | ✅ (largest incremental-update gap 2.3 × 10⁻¹² for the score objective, 0 for the decision objective; condition number of `P` on `machine-1-1` 1.4 × 10⁷ with ridge 10⁻⁶) | `tests/correctness.json` |
| Test machines never downloaded | ✅ `test_machines_downloaded: false`; the data root held exactly the 14 tuning and validation machines | `config/run_config.json`, README |
| Frozen choices written before any validation file existed | ✅ freeze 06:29:18.9 UTC, first validation retrieval 06:29:20.1 UTC | `reports/frozen_before_validation.json`, `config/data_manifest_validation.json` |
| **G1**: relative headroom ≥ 0.15 (primary: `L` = 60, score objective) | **✅ 0.920 [0.886, 0.949], robust** | `artifacts/validation_summary.json` |
| G1 at `L` = 30, `L` = 120, and with the decision-level objective | ✅ ✅ ✅ (0.929, 0.915, 0.872) | `figures/phase_table.csv` |
| Native endpoint moves with sensing (not a gate of the plan; reported because the plan says the proxy and the decision must be shown to agree) | **❌ F1 is flat or falls** | `artifacts/validation_summary.json` |

## 3. Results

Losses are divided by each machine's hold-only loss, so 1 means no better than hold and 0 means full fidelity. Every table is printed by `scripts/d1_0_tables.py` from the committed run files (§6).

### 3.1 Detector choice on the tuning machines

| Floor | Ridge | Mean F1 (tuning machines) | Median alarm rate on unlabelled minutes | Selected |
|---:|---:|---:|---:|---|
| 0.01 | 1e-06 | 0.320 | 0.029 | yes |
| 0.01 | 0.01 | 0.309 | 0.023 |  |
| 0.01 | 0.1 | 0.320 | 0.027 |  |
| 0.05 | 1e-06 | 0.320 | 0.028 |  |
| 0.05 | 0.01 | 0.311 | 0.060 |  |
| 0.05 | 0.1 | 0.287 | 0.044 |  |

### 3.2 Best fixed allocation on the tuning machines (normalised area, lower is better)

| Fixed policy | L = 120 | L = 30 | L = 60 |
|---|---:|---:|---:|
| round_robin | 0.6832 | 0.6043 | 0.6256 |
| top_volatility | 0.6151 (selected) | 0.5152 (selected) | 0.6164 (selected) |
| top_weighted_volatility | 0.6807 | 0.5488 | 0.6267 |

### 3.3 Gate G1 and the pre-registered sensitivities (validation machines)

| Run | L | Objective | Best fixed | Area: fixed | Area: reference | Area: greedy oracle | Relative headroom [95 % CI] | G1 (≥ 0.15) | Robust |
|---|---:|---|---|---:|---:|---:|---|---|---|
| L120 | 120 | score | top_volatility | 0.5720 | 0.0485 | 0.0506 | 0.915 [0.878, 0.944] | pass | yes |
| L30 | 30 | score | top_volatility | 0.5297 | 0.0377 | 0.0385 | 0.929 [0.888, 0.957] | pass | yes |
| decision | 60 | decision | top_volatility | 0.5561 | 0.0710 | 0.0723 | 0.872 [0.845, 0.901] | pass | yes |
| primary | 60 | score | top_volatility | 0.5483 | 0.0440 | 0.0452 | 0.920 [0.886, 0.949] | pass | yes |

### 3.4 Policies in the primary run

| Policy | Kind | Normalised area [95 % CI] | Headroom kept (fraction of fixed-to-reference) [95 % CI] |
|---|---|---|---|
| round_robin | fixed | 0.5455 [0.5025, 0.5982] |  |
| top_volatility (best fixed) | fixed | 0.5483 [0.4719, 0.6368] |  |
| top_weighted_volatility | fixed | 0.4993 [0.3991, 0.6135] |  |
| recent_change | dynamic | 0.3133 [0.2361, 0.4040] | 0.47 [0.29, 0.61] |
| sensitivity | dynamic | 0.3002 [0.2285, 0.4023] | 0.49 [0.31, 0.62] |
| sensitivity_x_volatility | dynamic | 0.3371 [0.2259, 0.4988] | 0.42 [0.19, 0.62] |
| oracle_greedy | privileged greedy | 0.0452 [0.0315, 0.0598] |  |
| reference | best-known reference | 0.0440 [0.0308, 0.0581] |  |

### 3.5 Paired differences of areas

| Paired difference of normalised areas (negative: first is better) | L120 | L30 | decision | primary |
|---|---|---|---|---|
| oracle_greedy - best fixed | -0.5215 [-0.6117, -0.4361] | -0.4913 [-0.6071, -0.3814] | -0.4838 [-0.5412, -0.4396] | -0.5031 [-0.6007, -0.4205] |
| recent_change - best fixed | -0.2247 [-0.3205, -0.1132] | -0.2691 [-0.3739, -0.1519] | -0.1462 [-0.2320, -0.0455] | -0.2350 [-0.3209, -0.1354] |
| sensitivity_x_volatility - recent_change | +0.0527 [-0.0318, 0.1626] | +0.1096 [-0.0081, 0.2369] | -0.0108 [-0.0714, 0.0521] | +0.0238 [-0.0573, 0.1340] |
| sensitivity_x_volatility - sensitivity | +0.0361 [-0.0083, 0.1129] | +0.0920 [-0.0271, 0.2257] | +0.0029 [-0.0215, 0.0396] | +0.0369 [-0.0153, 0.1266] |
| sensitivity_x_volatility - best fixed | -0.1720 [-0.2977, -0.0073] | -0.1595 [-0.2448, -0.0820] | -0.1571 [-0.2596, -0.0419] | -0.2112 [-0.2986, -0.1042] |

### 3.6 Machines

| Machine | Windows | Windows with a labelled anomaly | Headroom (best fixed to reference) | Fully observed detector: F1 | Alarm rate on unlabelled minutes |
|---|---:|---:|---:|---:|---:|
| machine-1-2 | 392 | 20 | 0.851 | 0.137 | 0.004 |
| machine-1-6 | 392 | 90 | 0.949 | 0.411 | 0.479 |
| machine-2-2 | 393 | 60 | 0.880 | 0.078 | 0.030 |
| machine-2-6 | 477 | 16 | 0.923 | 0.135 | 0.121 |
| machine-3-1 | 476 | 10 | 0.905 | 0.097 | 0.148 |
| machine-3-5 | 392 | 17 | 0.979 | 0.078 | 0.348 |
| machine-3-9 | 476 | 9 | 0.916 | 0.049 | 0.336 |

### 3.7 Mean normalised loss by budget (primary run)

| Policy | k = 0 | k = 1 | k = 2 | k = 4 | k = 8 | k = 12 | k = 16 | k = 24 | k = 38 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round_robin | 1.000 | 0.983 | 0.977 | 0.895 | 0.828 | 0.714 | 0.636 | 0.465 | 0.000 |
| top_volatility | 1.000 | 0.961 | 0.968 | 0.947 | 0.778 | 0.710 | 0.618 | 0.493 | 0.000 |
| top_weighted_volatility | 1.000 | 0.977 | 0.964 | 0.825 | 0.799 | 0.685 | 0.611 | 0.361 | 0.000 |
| recent_change | 1.000 | 0.791 | 0.677 | 0.512 | 0.436 | 0.387 | 0.343 | 0.247 | 0.000 |
| sensitivity | 1.000 | 0.752 | 0.613 | 0.515 | 0.435 | 0.372 | 0.335 | 0.223 | 0.000 |
| sensitivity_x_volatility | 1.000 | 0.853 | 0.717 | 0.547 | 0.455 | 0.410 | 0.380 | 0.273 | 0.000 |
| oracle_greedy | 1.000 | 0.390 | 0.211 | 0.073 | 0.033 | 0.019 | 0.012 | 0.001 | 0.000 |
| reference | 1.000 | 0.390 | 0.211 | 0.070 | 0.029 | 0.017 | 0.010 | 0.001 | 0.000 |

### 3.8 Detection F1 against the dataset labels, by budget

| Policy (primary) | k = 0 | k = 1 | k = 2 | k = 4 | k = 8 | k = 12 | k = 16 | k = 24 | k = 38 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round_robin | 0.259 | 0.256 | 0.251 | 0.253 | 0.243 | 0.236 | 0.223 | 0.215 | 0.192 |
| top_volatility | 0.259 | 0.261 | 0.262 | 0.258 | 0.258 | 0.253 | 0.253 | 0.184 | 0.192 |
| top_weighted_volatility | 0.259 | 0.252 | 0.252 | 0.252 | 0.254 | 0.255 | 0.258 | 0.188 | 0.192 |
| recent_change | 0.259 | 0.256 | 0.254 | 0.208 | 0.212 | 0.212 | 0.216 | 0.218 | 0.192 |
| sensitivity | 0.259 | 0.261 | 0.263 | 0.226 | 0.208 | 0.199 | 0.196 | 0.190 | 0.192 |
| sensitivity_x_volatility | 0.259 | 0.260 | 0.257 | 0.218 | 0.204 | 0.201 | 0.194 | 0.187 | 0.192 |
| oracle_greedy | 0.259 | 0.258 | 0.256 | 0.187 | 0.190 | 0.191 | 0.192 | 0.192 | 0.192 |

### 3.9 Derived quantities quoted in this note

- Validation windows: 2998, of which 222 contain a labelled anomalous minute (0.074).
- Fully observed detector, mean per-machine F1: 0.141 (range 0.049 to 0.411).
- Fully observed detector, alarm rate on unlabelled minutes: median 0.148, range 0.004 to 0.479; machines above 0.10: 5 of 7.
- Pooled detection F1 at k = 0 (hold) and k = 38 (full observation): 0.259 and 0.192.
- Best fixed policy on validation by area: top_weighted_volatility (0.4993); the tuning-chosen best fixed (top_volatility) has 0.5483. Headroom against the validation-best fixed policy: 0.912.
- Greedy oracle share of the reference area: 0.0452 against 0.0440.
- Policies with a headroom-kept fraction whose interval excludes 0 in the primary run: recent_change, sensitivity, sensitivity_x_volatility.

## 4. Interpretation

**Observed** (from the tables above):

- **The gate passes easily and robustly:** 0.920 [0.886, 0.949] at `L` = 60, 0.929 at `L` = 30, 0.915 at `L` = 120 and 0.872 [0.845, 0.901] with the decision-level objective. Every machine is between 0.851 and 0.979. Against the validation-best fixed policy (`top_weighted_volatility`) the headroom is 0.912, so the choice of the best fixed policy on the tuning machines does not drive it.
- **The loss sits in a few channels per window.** The privileged oracle reaches 0.390 of the hold-only loss with one channel opened, 0.073 with four and 0.033 with eight, and the reference differs from it by at most 0.0044 (at `k` = 8) on the budget grid. The oracle knows the window's future (privilege P3), so this is an upper bound on what can be obtained, not a gain.
- **Deployable dynamic policies beat the best fixed allocation.** `recent_change` is lower than it by 0.2350 in area [−0.3209, −0.1354] and `sensitivity_x_volatility` by 0.2112 [−0.2986, −0.1042]; `sensitivity` has the lowest dynamic area (0.3002) and, like the other two, keeps about half of the fixed-to-reference gap (0.49 [0.31, 0.62], 0.47 [0.29, 0.61], 0.42 [0.19, 0.62]). At `k` = 8 (21 % of the channels) they are at 0.435 to 0.455 of the hold loss against 0.778 for the best fixed policy and 0.029 for the reference.
- **No separation among the dynamic policies, and the product is not better than its factors.** `sensitivity_x_volatility − recent_change` = +0.0238 [−0.0573, 0.1340] and `sensitivity_x_volatility − sensitivity` = +0.0369 [−0.0153, 0.1266] in the primary run (positive means the product is worse); the same pairs at other window lengths have intervals that also span zero (and one at `L` = 30 almost excludes it, +0.1096 [−0.0081, 0.2369]). With seven machines this says only that the data cannot rank them.
- **The fixed policies are close to one another:** areas 0.5455, 0.5483 and 0.4993 with overlapping intervals; the margin by which the tuning machines preferred `top_volatility` at `L` = 60 (0.6164 against 0.6256 and 0.6267) was small.
- **The native endpoint does not follow the proxy.** The fully observed detector has a mean per-machine F1 of 0.141 (0.049 to 0.411), and on five of seven validation machines it alarms on more than 10 % of minutes that carry no anomaly label (median 0.148, up to 0.479). Pooled F1 against the labels is 0.259 with hold only and 0.192 with every channel observed; across the policies and budgets in §3.8 it lies between 0.184 and 0.263 and shows no systematic increase with sensing.

**Hypotheses (not tested here):**

1. *The proxy fails as a decision metric because the frozen detector does not transfer from each machine's training stream to its test stream.* Its alarm rate on unlabelled minutes has a median of 0.029 on the tuning machines and 0.148 on the validation machines. More sensing then makes the detector alarm more, not detect better. **Diagnostic:** repeat the gate with a detector that is calibrated on the start of the test stream, or with a labelled decision loss (misses and false alarms against the labels), on tuning machines first.
2. *Most of the headroom is knowledge of the window's future, and the deployable policies succeed through persistence.* A channel that is far from normal at the snapshot tends to stay far, so `|P z|` (and `recent_change`) pick channels that matter. **Diagnostic:** add a policy ranking channels by `|z|` at the snapshot and a policy ranking by the previous window's realised loss contribution, and compare them with `sensitivity`.
3. *The three dynamic policies differ, but seven machines cannot show it.* **Diagnostic:** more machines (the 14 test machines are reserved for a confirmatory stage and are not to be used for this).
4. *A learned critic would keep more than half of the headroom.* That is the Rung-1 question and this run cannot speak to it.

## 5. Design issues found

1. **`J` is a proxy for a detector that is poor on these machines,** and the labelled F1 does not move with sensing. This is the main limit on what the gate means.
2. **Seven validation machines** give wide intervals on the dynamic-policy quantities (for example 0.19 to 0.62 for one headroom fraction) and the machines differ a lot (false-alarm rate on unlabelled minutes from 0.004 to 0.479).
3. **The detector's constants were chosen among near-ties** (mean tuning F1 0.3205, 0.3199, 0.3196), so the tuning step left the originally written detector in place; it is weakly conditioned (condition number of `P` 1.4 × 10⁷ on `machine-1-1`), and its calibration did not transfer: the median alarm rate on unlabelled minutes is 0.029 on the tuning machines and 0.148 on the validation machines.
4. **The best fixed policy was chosen by a small margin** and is not the best fixed policy on validation. The conclusion does not depend on it (§4) but the frozen rule was applied as written.
5. **The oracle is greedy, not exhaustive.** The reference includes every deployable policy, so it is never above them, but it may still be above the true optimum.
6. **Policy computation and the window-start snapshot are not charged** (only `k · L` channel-minutes are). D4-2 showed that an unpriced scorer can change a conclusion; the sensitivity policies here need about `38²` multiply-adds per window.
7. **The hold-only loss is the normaliser,** so a machine whose windows move little has a noisy normalised curve; the machine-first bootstrap reflects this.
8. **The data-manifest hash includes `retrieved_utc`,** so it changes when files come from the cache; the per-file SHA-256 values are the identity of the data (run README).
9. **Smoke tests on tuning machines** preceded the run and led to the oracle change; they read no validation machine (run README).
10. **One dataset, three machine groups from one operator.** Nothing here generalises beyond SMD.

## 6. Commands

Every table in §3 is printed from committed files, and the primary run's areas and headroom are recomputed from the per-window parquet and asserted equal to the stored summary:

```bash
python scripts/d1_0_tables.py results/runs/d1_0_sensor_opportunity_20260930T062900Z
```

The reproduction from a clean worktree is described in the run README (byte-identical results; differences only in run id, timestamps, timings and hashes that cover them).

## 7. What this run does and does not support

**Supports:**

- On SMD, with this frozen detector and loss, there is large headroom for sensing allocation: a privileged allocation removes almost all of the score-fidelity loss with few channels and the best fixed allocation removes little (§3.3, §3.7).
- Simple non-learned policies that read only the window-start snapshot keep about half of that headroom (§3.4).
- No evidence that multiplying sensitivity by volatility helps over either factor (§3.5), with low power.
- The fully observed detector used here is a poor anomaly detector on these machines, and its labelled F1 does not improve with sensing (§3.6, §3.8).
- The analysis reproduces exactly on the same machine (run README).

**Does not support:**

- Any statement about H2, a co-state, a learned direct critic or a learned critic with gradient features.
- Any claim that adaptive sensing improves anomaly detection on SMD, or that an operator would benefit; the labelled endpoint does not show it.
- A claim that "sensitivity is not VOI" holds or fails in D1: the test is inconclusive.
- Any statement about wall-clock time, energy, the cost of the policies, other datasets or the 14 test machines.

**Corrects earlier statements.**

- The cross-domain plan (§5, D1) says D1 is where "sensitivity ≠ VOI" can be tested. This run does not test it (no separation between the product and its factors); it only shows the test is not decided by the data at seven machines.
- The plan's rule says that a G1 pass licenses a Rung-1 design. This note recommends against designing Rung 1 on this loss, because its labelled endpoint does not follow it.

## 8. Next steps (roadmap IDs in `docs/plans/roadmap.md`)

1. **Do not open D1-1 on this loss.** Decide between: (a) **D1-0b**, a redesign of the objective so that the labelled endpoint moves with sensing, tested first on tuning machines with a cheap diagnostic (does any simple detector's labelled F1 improve with more sensing at all?); (b) a forecasting objective on the same stream; (c) leaving D1 and moving to D2-0 or D3-0. This needs Roman's decision; nothing is started.
2. **Keep the window-start-snapshot allocation problem** as a reusable design: it has an exact loss, a cheap greedy oracle and a clear reference, and the dynamic policies here are the baselines any learned critic must beat.
3. **The real-data path is unchanged:** E1.1 and B2 (Colab) remain the critical path; D2-0 and D3-0 follow the same card and gate pattern, with the lesson of D4-2 and D1-0 that the loss must be checked against the domain's native endpoint before any critic is built.
