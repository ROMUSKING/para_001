# D1-0b: a forecasting objective for sensing allocation on SMD (Rung-0 redesign)

**Written:** 2026-09-30, before any forecasting loss was computed on any machine · **Decision:** Roman, 2026-09-30 ("redesign its objective (D1-0b)") · **Roadmap:** D1-0b · **Follows:** [D1-0](d1-0-plan.md) and its [note](../research-notes/2026-09-30-d1-0-sensor-opportunity.md) · **Precedence:** comprehensive plan §7.2B-1 for the opportunity gate, cross-domain plan §2 for the card.

## 1. Why a redesign, and what it must fix

D1-0 passed its gate on a **fidelity proxy**: the loss was how far a frozen detector's score on a partial stream was from its score on the full stream. That detector was poor on the validation machines (mean per-machine F1 0.141), and the labelled F1 did not improve with sensing (0.259 with hold only, 0.192 with every channel observed), so the gate said nothing about the labelled task.

D1-0b changes the loss so that **the loss is the native endpoint**: the error of a frozen forecaster's prediction of the machine's own future metrics. No label and no second model stands between the allocation and the quantity that is scored.

**Scope.** A Rung-0 study with non-learned policies on validation machines, preceded by a gate on the tuning machines. No learned critic and no claim about H2. The 14 test machines are never downloaded.

## 2. Domain card changes (everything not listed is as in the D1-0 plan §2)

| # | Requirement | D1-0b |
|---|---|---|
| 2 | Downstream objective | `J_w(S)`, lower is better: the mean over the window's minutes of the squared error of an `h`-minute-ahead forecast of **all 38 channels** (in standardised units, averaged over channels), made by a frozen linear forecaster from the stream as seen under allocation `S`. Targets are the true future values (privilege P3, used only to score). |
| 3 | Legal actions | As in D1-0: open a channel for the whole window; the rest are held at the window-start snapshot. |
| 5 | Measurable effects | Exact for one allocation. The oracle over subsets is **approximate**: greedy selection (better of forward and backward) of the subset that makes the forecast closest to the forecaster's full-information forecast, knowing the true window states **but not the forecast targets**. |
| 6 | Baselines | Fixed, uncertainty-only, sensitivity-only, sensitivity × uncertainty, oracle (§4). |
| 10 | Regime | Acquisition of unknown information (VOI). For a linear forecaster the sensitivity of the forecast to a channel is a fixed matrix row, so the state-dependent part of the value of sensing a channel is its unknown staleness; a co-state from `∂J/∂state` would need the unknown target, so **nothing in D1-0b is called a co-state** (AGENTS.md rule 4). |

## 3. Data, forecaster and windows (all frozen)

- **Machines and splits:** exactly as in D1-0 (`sensor.split_machines`): 7 tuning, 7 validation, 14 test (never downloaded). The data are the files already pinned by the D1-0 manifests; the new run pins them again.
- **Standardisation:** `z = (x − μ) / max(σ, 0.05)`, `μ, σ` from the machine's training stream. (Not tuned. D1-0's floor was a detector constant chosen for scoring; here 0.05 keeps near-constant channels from dominating a squared error.)
- **Forecaster:** `ŷ_{t+h} = z̃_t B`, `B = (Xᵀ X + λ I)⁻¹ Xᵀ Y` fitted on the machine's training stream (`X = z_t`, `Y = z_{t+h}`), **`λ = 1`**, **`h = 5` minutes**. Not tuned; the horizon is fixed before any data were scored and the other horizons are sensitivities (§7).
- **Windows:** `L = 60` minutes tiled from the start of the test stream, first two windows dropped, snapshot of all channels at the window start, exactly as in D1-0. The loss covers the minutes `t0 + 1 … t0 + L − 1 − h`, so every target lies inside the window.
- **Budgets:** `k ∈ {0, 1, 2, 4, 8, 12, 16, 24, 38}` channels opened per window. Policy computation is not charged.

## 4. Policies (non-learned, frozen)

Each deployable policy ranks the channels from P0 information (the snapshot, the previous window's snapshot, training statistics) and opens the top `k`; ties go to the lower channel index. `s_c = ‖B[c, :]‖²` is the squared norm of the row of `B` for input channel `c` (how much that input moves the forecasts), `d_c` is the training standard deviation of the `L`-minute difference of `z_c`, `Δ_c = z_{t0,c} − z_{t0−L,c}`.

| Policy | Score of channel `c` | Kind |
|---|---|---|
| `round_robin` | window `w` opens channels `(w k + j) mod 38` | fixed |
| `top_volatility` | `d_c` | fixed, uncertainty-only |
| `top_sensitivity` | `s_c` | fixed, sensitivity-only |
| `top_weighted_volatility` | `s_c d_c²` | fixed, first-order value of information |
| `recent_change` | `|Δ_c|` | dynamic, uncertainty-only |
| `recent_change_weighted` | `s_c Δ_c²` | dynamic, first-order value of information |
| `oracle_fidelity` | per window and size, the better of greedy forward selection and greedy backward elimination of the subset minimising `mean_t ‖(z̃_t − z_t) B‖²`, using the true window values (not the targets) | **privileged** |

- **Best fixed allocation** = the fixed policy (of four) with the lowest mean normalised excess area on the **tuning** machines.
- **Reference** = the oracle's curve with unused budget legal (per window and budget, the best size `j ≤ k` by the oracle's own objective). Deployable policies are **not** mixed into the reference (the native loss is noisy per window and a minimum over arms would select on that noise); a deployable policy that sits below the oracle is reported as such.

## 5. Measures

- **Native loss** `J_w(k)` per window and policy; **excess** over full observation `E_w(k) = J_w(k) − J_w(38)` (`J_w(38)` is the forecaster's own error with every channel observed).
- **Normalised excess curve (per machine):** the mean excess curve divided by the machine's mean full-observation loss `J(38)`; it starts at that machine's `r` at `k = 0` and is 0 at `k = 38`. **Normalised area** over `k / 38`, averaged over machines (see §10 for why this replaced division by the hold-only excess). Relative headroom, retained fraction and the two-stage bootstrap (machines, then windows; 10,000 resamples; seed 0) are as in D1-0.
- **Endpoint-moves statistic** `r = (mean_w J_w(0) − mean_w J_w(38)) / mean_w J_w(38)`, per machine, then averaged over machines, with the same two-stage bootstrap.
- **Secondary (descriptive):** the fidelity loss of each policy (the oracle's objective), and the same tables for `h = 1` and `h = 15` and `L = 30` and `L = 120`.

## 6. Protocol and frozen rules

1. **Stage A, tuning machines only** (downloaded and read as in D1-0): correctness checks on real windows; the native loss, the endpoint-moves statistic and the four fixed policies' areas; choose the best fixed policy. **Gate G0 (the endpoint moves with sensing):** `r ≥ 0.05` on the tuning machines (point estimate of the machine mean). **If G0 fails on the tuning machines, stage B is not run**, the validation machines are not read, and the note reports that the forecasting endpoint does not move with sensing on SMD at this horizon.
2. Freeze the detector-free constants above, the best fixed policy (per horizon and window length), the rules, and write their SHA-256.
3. **Stage B, validation machines.** Correctness is re-checked first. Then:
   - **G0 on validation:** `r ≥ 0.05`, point estimate; **robust** if the 95 % interval's lower bound is also ≥ 0.05.
   - **G1 (opportunity):** relative headroom of the oracle over the best fixed allocation ≥ 0.15 on the normalised excess areas, point estimate; robust if the interval's lower bound is too. Only meaningful if G0 holds; if G0 fails, G1 is reported and labelled vacuous.
   - **G2 (descriptive):** the fraction of the headroom kept by `recent_change` and `recent_change_weighted`.
   - **G-VOI (descriptive, the "sensitivity is not VOI" check):** paired differences of normalised areas: `top_weighted_volatility − top_sensitivity`, `top_weighted_volatility − top_volatility`, `recent_change_weighted − recent_change`, `recent_change_weighted − top_sensitivity`. A negative difference means the product is better than that factor.
4. **G3 (correctness, before anything is read on validation):** the incremental fidelity update equals a direct recomputation (both directions); the native loss at full observation equals the forecaster's error computed directly; `J(38) − J(38) = 0`; `B` reproduces the ridge solution; deployable policies ignore everything after the snapshot; the split is a partition; the fetch code refuses test machines.

**Decision rules.** G0 fails → no sensing opportunity on SMD for forecasting at `h = 5`; D1 is deprioritised behind D2/D3 and this is recorded as a D1 negative result. G0 passes and G1 fails → redesign the candidates (sub-window opening, retaining a band). G0 and G1 pass → a D1-1 design (learned critic against a critic given sensitivity features, five paired seeds) may be written, with the note's G2 and G-VOI as its baselines, and needs Roman's go-ahead.

## 7. Pre-registered sensitivities (validation, reported next to the primary result)

`h = 1` and `h = 15`; `L = 30` and `L = 120` (best fixed re-chosen on the tuning machines at each setting, `B` refitted). The primary result is `h = 5`, `L = 60`.

## 8. Artefacts

`src/adjointrwm/domains/sensor_forecast.py` and `tests/test_sensor_forecast.py` (reusing the D1-0 loader, windows and oracle-mask code); `notebooks/04-domains/d1_0b_forecast_sensing.ipynb` (CPU; downloads nothing new if the D1-0 cache is present, otherwise about 250 MB); run `results/runs/d1_0b_forecast_sensing_<UTC>/` imported with a README and a clean-worktree reproduction; `scripts/d1_0b_tables.py`; a research note.

## 9. Risks and limits

- **The oracle is privileged and greedy**, so a large headroom is an upper bound on what sensing could give, not a gain a policy can have. G0 and G2 carry more information than G1.
- **A linear forecaster** has a state-independent sensitivity, so this domain cannot show a state-dependent co-state advantage; a nonlinear downstream model would be needed for that, and is not built here.
- **Short-horizon forecasts of slowly varying metrics are dominated by persistence.** G0 exists to show whether a stale snapshot costs anything measurable.
- **Seven validation machines** give wide intervals and machine heterogeneity.
- **Policy computation is not charged**, and the window-start snapshot is a fixed cost excluded from the budget.
- **One dataset,** three machine groups from one operator.

## 10. Changes after this plan was written

- **2026-09-30, before any validation machine was read:** a smoke run of the notebook on tuning machines crashed at the `h = 15` sensitivity because `machine-1-1` has a **negative** hold-only excess there (its `r` is −0.152), so dividing a machine's excess curve by its hold-only excess is undefined; a direct computation of `r` on the seven tuning machines then showed, at the primary setting, `r` = 0.26, 0.66, 0.27, 1.27, 0.62, 0.013 and 0.55 (mean 0.52, so G0 holds on the tuning machines) and, for `machine-3-4`, a tiny `r` at every setting (0.006 to 0.066) together with a very large full-observation error (about 16 against 0.1 to 0.4 elsewhere), so dividing by its hold-only excess would give it an arbitrary weight. The normalisation is therefore changed from the hold-only excess to the machine's **mean full-observation loss**: the curve starts at `r` and ends at 0, is defined whenever the full-observation loss is positive, and weights machines by how much sensing matters to them. Everything else, including `r`, G0, G1, G2, G-VOI and all thresholds, is as planned. Only tuning-machine numbers had been computed when this was written.
