# D1-0: sensor-stream domain card and Rung-0 opportunity gate (SMD)

**Written:** 2026-09-30, before any validation machine was downloaded · **Decision:** Roman, 2026-09-30, "proceed as recommended" (D4-1 stays closed; D1/D2/D3 candidates from the licence survey are confirmed; E1.1 and B2 remain Colab jobs) · **Roadmap:** D1-0 (Tier P) · **Precedence:** the comprehensive plan §7.2B-1 for the opportunity gate; the cross-domain plan §2 for the card.

## 1. Question and scope

**Question.** On a public multivariate server-metrics stream, is there room for adaptive *sensing* allocation: does an allocation that knows which channels will move (a privileged oracle) beat the best fixed allocation by at least 15 % of the normalised area, at equal sensing budget, and how much of that room can a deployable policy that sees only the start of the window keep?

**This is a Rung-0 gate.** It runs non-learned policies only and makes **no statement about H2**: there is no learned critic and no co-state. Passing the gate would license a Rung-1 design; failing it says to redesign the candidates (checklist row 9). The result is exploratory and uses validation machines only; **the test machines are never downloaded.**

**Why this shape.** The cross-domain plan (§5, D1) says sampling acquires unknown information (VOI), so this domain is where "sensitivity is not VOI" can be tested. D4-2 taught two things this plan applies from the start: price every policy in the same units as the resource, and compare with a competent non-learned baseline before any learned scorer.

## 2. Domain card (cross-domain plan §2)

| # | Requirement | D1-0 |
|---|---|---|
| 1 | Structured state | A channel × time array from one machine: 38 channels at 1-minute sampling. The hierarchy is *window → channel*: a window is `L` consecutive minutes; a channel inside a window is the unit that can be opened. |
| 2 | Downstream objective | `J`, lower is better: the **score-fidelity loss** of a frozen detector, `J_w(S) = mean over t in window of (φ(s̃_t) − φ(s_t))²`, with `s` the detector score on the fully observed stream, `s̃` the score on the stream as seen under allocation `S`, and `φ(s) = log(1 + s)`. Secondary endpoints (reported, not optimised): alarm disagreement with the fully observed detector and detection F1 against the dataset labels. |
| 3 | Legal actions | `open(c)`: observe channel `c` at every minute of the window (kind `sample`). `hold` (default for an unopened channel: keep its window-start value) and `stop` (leave budget unused) are generic. |
| 4 | Constrained resource | Sensor samples (ledger R): `k · L` channel-minutes per window, on top of a fixed window-start snapshot of all 38 channels. Policy computation is **not charged** here (see §9). C, L and Q ledgers are zero. |
| 5 | Measurable effects | Exact for one allocation: the score under any `S` is recomputed from the data. The oracle over subsets is **approximate**: greedy forward selection, not exhaustive (`C(38, k)` is too large). `DomainSpec.oracle_support = "approximate"`. |
| 6 | Baselines | Fixed: round-robin, top volatility, top precision-weighted volatility. Uncertainty-only: recent change. Sensitivity-only, and sensitivity × volatility. Oracle: greedy, privileged. **A direct critic and a co-state critic are Rung 1 and not built here.** |
| 7 | Privilege | P0 (deployable at the window start): the snapshot at the window start, the snapshots of the two previous windows, and statistics fitted on the machine's training stream. P3 (evaluation and oracle only): every value inside the window after the snapshot, and the labels. |
| 8 | Data | Server Machine Dataset (SMD), MIT (repository `LICENSE`, register id `smd`, verdict `adopt`, `docs/licences/register.csv`). Downloaded from the owners' repository, kept out of git, pinned by SHA-256 in the run's manifest. |
| 9 | Opportunity | The gate of §6. |
| 10 | Regime | Acquisition of unknown information (VOI). A sensitivity or co-state alone is not enough; the volatility of a channel is a second factor. |

## 3. Data, splits and detector (all frozen)

- **Machines.** 28 machines in three groups: `machine-1-1 … 1-8`, `machine-2-1 … 2-9`, `machine-3-1 … 3-11`, in that order, indexed `0 … 27`. **Split by machine, before windowing:** index mod 4 = 0 → **tuning** (7 machines); = 1 → **validation** (7); = 2 or 3 → **test** (14 machines, never downloaded or read).
- **Streams.** Each machine has a training stream (normal behaviour), a test stream and per-minute labels for the test stream. The detector is fitted on the training stream only. Windows are taken from the test stream. Labels are used only for the secondary endpoint.
- **Standardisation.** `z = (x − μ) / max(σ, f)` with `μ, σ` from the machine's training stream and a floor `f` that guards near-constant channels. **`f` is chosen on the tuning machines** (§3, detector).
- **Detector.** Mahalanobis score `s(z) = zᵀ P z`, `P = ((1 − λ) Σ + λ diag Σ + ρ I)⁻¹`, `Σ` the covariance of `z` on the training stream, **`λ = 0.1`**. Alarm threshold `τ` = the 0.995 quantile of `s` on the training stream. `λ` and the quantile are fixed. **The floor `f ∈ {0.01, 0.05}` and the ridge `ρ ∈ {10⁻⁶, 10⁻², 10⁻¹}` are chosen on the seven tuning machines** as the pair with the highest mean per-machine F1 of the fully observed detector against the test labels at `τ` (labels are used here on tuning machines only; validation labels are read only for the secondary endpoint). The choice is frozen before any validation machine is downloaded.
- **Windows.** Length **`L = 60`** minutes, non-overlapping, tiled from the start of the test stream; a final partial window is dropped; **windows 0 and 1 are dropped** (their policies would lack two previous snapshots). Window `w` starts at `t0 = wL`. The snapshot `z_{t0}` observes all 38 channels.
- **Sensing.** An unopened channel is held at its snapshot value for the rest of the window; an opened channel is observed at every minute. The loss covers `t = t0 + 1 … t0 + L − 1`. Targets are in the future of the decision (comprehensive plan rule 5), and windows are independent given the data.
- **Budgets.** `k ∈ {0, 1, 2, 4, 8, 12, 16, 24, 38}` channels opened per window. `k = 38` observes everything (`J = 0`); `k = 0` is hold only.

## 4. Policies (all non-learned, frozen)

Each deployable policy ranks the 38 channels from P0 information and opens the top `k`. Ties break toward the lower channel index.

| Policy | Score of channel `c` | Kind |
|---|---|---|
| `round_robin` | window `w` opens channels `(w k + j) mod 38`, `j = 0 … k−1` | fixed |
| `top_volatility` | `d_c`: the standard deviation of the `L`-minute difference of `z_c` on the training stream | fixed |
| `top_weighted_volatility` | `d_c² P_cc` (the expected contribution of an unseen move to the score, ignoring cross terms) | fixed |
| `recent_change` | `|z_{t0,c} − z_{t0−L,c}|` (change between the two latest snapshots) | dynamic, uncertainty-only |
| `sensitivity` | `|(P z_{t0})_c|`, proportional to the gradient of `s` with respect to `z_c` at the snapshot | dynamic, sensitivity-only |
| `sensitivity_x_volatility` | `|(P z_{t0})_c| · d_c` (first-order surrogate of the effect of a typical move) | dynamic |
| `oracle_greedy` | forward selection that adds, at each step, the channel that most lowers `J_w`, using the true window values | **privileged** |

- **Best fixed allocation** = the fixed policy with the lowest mean normalised area on the **tuning** machines (chosen there, applied to validation).
- **Oracle reference.** Greedy forward selection is not optimal and `J_w` is not monotone in `S`, so the reference curve at each `(window, k)` is the **minimum over `oracle_greedy` at every size `j ≤ k` and over all deployable policies** (best-known, as in `adjointrwm.domains.metrics`). Using fewer than `k` samples is legal (`stop`).
- Nothing here is called a co-state: `sensitivity` uses `∂s/∂z`, but `J` is a comparison against an unobserved full-information score, so no `∂J/∂state` is supervised or derived. The terminology rule (AGENTS.md rule 4) is kept for Rung 1, where `J` and its gradient would have to be defined.

## 5. Measures

- Per window and policy: `J_w(k)` for each budget. Per machine: the mean over its windows. Overall: the mean over machines (equal weight per machine).
- **Normalised area (per machine):** for each machine, the mean `J` curve of a policy is divided by that machine's hold-only value `J(k = 0)`, and the trapezoidal area is taken over `k / 38`, so a machine's number lies in `[0, 1]` for a policy that never does worse than hold. The overall value is the mean of the machines' normalised areas.
- **Opportunity** (the rule of `adjointrwm.domains.metrics.opportunity_over_budgets`, applied to the machine-averaged normalised curves): `relative headroom = area(J_fixed − J_oracle) / area(J_fixed)` over the budget grid, with `J_fixed` the best fixed allocation and `J_oracle` the reference above.
- **Uncertainty:** a two-stage bootstrap (7 validation machines resampled with replacement, then windows within each resampled machine), 10,000 resamples, percentile 95 % interval. With 7 machines the interval is wide by construction.

## 6. Frozen gate and rules

- **G1 (gate, plan §7.2B-1):** relative headroom ≥ 0.15 on the validation machines, point estimate. **Robust** if the 95 % interval's lower bound is also ≥ 0.15. Reported either way; a pass that is not robust is called a pass that is not robust.
- **G2 (descriptive, no decision):** the fraction of the oracle headroom that each dynamic policy keeps, `(J_fixed − J_policy) / (J_fixed − J_oracle)` on the area, and the paired difference of areas between `sensitivity_x_volatility`, `sensitivity` and `recent_change` (the "sensitivity is not VOI" check).
- **G3 (correctness, before anything else is read):** the incremental score update equals a direct recomputation; `J(k = 38) = 0`; the detector's `P` is symmetric positive definite and `τ` is the stated quantile of the training scores; a deployable policy's choice does not change when the window values after the snapshot are replaced; the machine split is a partition and the fetch code refuses a test machine.
- **Decision rules.** G3 fails → fix and rerun, and say so. G1 passes → a Rung-1 design may be written (a learned direct critic against a critic given the gradient features, five paired seeds), and it needs Roman's go-ahead. G1 fails → redesign the candidates (open a channel for a sub-window; retain a band; query history), or record that D1 on SMD has no sensing opportunity at this granularity.

## 7. Pre-registered sensitivity analyses (validation, reported next to the primary result)

- **Window length:** `L = 30` and `L = 120`, everything else unchanged. The primary result is `L = 60`.
- **Decision-level objective:** the same gate with `J` replaced by the alarm disagreement rate with the fully observed detector (fraction of the window's minutes on which the alarm decision differs), the oracle choosing greedily on that objective.
- **Secondary endpoint:** detection F1 against the labels at `τ` under the reconstructed stream, per policy and budget, pooled over validation windows. This is the native endpoint; it is descriptive because the detector is imperfect at full observation.

## 8. Artefacts

- `scripts/fetch_smd.py` downloads only tuning and validation machines to `~/adjointrwm_data/smd/`, writes `manifest.json` (URL, size, SHA-256, retrieval time per file), and refuses test machines.
- `src/adjointrwm/domains/sensor.py` and `tests/test_sensor.py`: the loader, detector, windows, policies, oracle and gate.
- `notebooks/04-domains/d1_0_sensor_opportunity.ipynb` (CPU): manifest, G3 checks, tuning machines (best fixed), then validation, then the report. Run directory `results/runs/d1_0_sensor_opportunity_<UTC>/`, imported with a README; research note `docs/research-notes/2026-09-30-d1-0-sensor-opportunity.md`.
- The tuning machines are used for two choices only: the detector's `(f, ρ)` and the best fixed family. Nothing else is tuned.

## 9. Risks and limits

- **Policy computation is not charged.** A policy needs `P z_{t0}` (about `38²` multiply-adds per window). Whether that is negligible depends on a device profile that D1-0 does not have; Rung 1 must price it in the resource's own units, as D4-2 did.
- **The oracle is greedy and privileged.** It is an upper bound only in the best-known sense of §4. A gate that fails with a weak oracle would not prove there is no opportunity; the reference includes every deployable policy so that it can never sit above them.
- **Score fidelity is a proxy.** `J` measures how closely a partially observed detector matches a fully observed one, not whether an operator would act correctly. The alarm-level and F1 endpoints in §7 are there to show whether the proxy and the decision agree.
- **Stale values may be good enough.** Server metrics at one-minute sampling move slowly for most channels. If hold at the window start is already close, `J(k = 0)` is small and every policy has little to gain; the gate then fails honestly.
- **Seven validation machines** give wide intervals and a machine-level heterogeneity that a pooled window count would hide, which is why the bootstrap resamples machines first.
- **Distribution.** SMD is three groups from one company's servers. Nothing here generalises beyond that.

## 10. Changes after this plan was written

- **2026-09-30, before any code:** the file format was checked on one **tuning** machine (`machine-1-1`): a comma-separated file of 38 columns, 28,479 rows in each of the training and test streams, and 9.5 % of its test minutes labelled anomalous, in segments of 400 to 700 minutes plus a few short ones. No validation machine had been downloaded. The interpretation labels are not used by any policy or measure, so they are not fetched by the run.

- **2026-09-30, before any code, still before any validation machine was downloaded:** with the originally written detector (`f = 0.01`, `ρ = 10⁻⁶`) the same tuning machine gave a covariance condition number of 1.4 × 10⁷ (8 of its 38 channels have zero training variance), and the detector alarmed on 35.5 % of the minutes that carry no anomaly label in that machine's test stream. A near-degenerate channel then dominates the score, which would turn the allocation problem into "watch the constant channels". The floor and the ridge are therefore chosen on the tuning machines from the small grid above, by the detector's own F1. The numbers came from a quick probe of that one machine (`f, ρ` = 0.01, 10⁻⁶; 0.01, 10⁻²; 0.01, 10⁻¹; 0.05, 10⁻⁶; 0.05, 10⁻²; 0.05, 10⁻¹ gave alarm rates on normal minutes of 35.5, 30.0, 12.5, 35.5, 6.0 and 4.4 %); the grid is not narrowed by them, and the run recomputes the choice on all seven tuning machines.

Each later change will be dated here, and any change made after a validation machine was read will be marked as such.
