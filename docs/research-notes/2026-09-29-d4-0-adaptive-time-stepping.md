# D4-0: goal-oriented adaptive time stepping, the first cross-domain reference domain

**Run:** `d4_time_stepping_20260929T162518Z` (main) and `d4_time_stepping_20260929T162914Z` (exploratory sensitivity variant) · **Date:** 2026-09-29 · **Notebook:** `notebooks/04-domains/d4_adaptive_time_stepping.ipynb` (SHA-256 `fb2c37cb…4ff311`)
**Hardware:** CPU in the Claude Code sandbox (timings are not a systems claim) · **Package commit:** `be9c6ed36a0e6ce7df331254a87b353e2987a9a1` (clean tree) · **Config SHA-256:** `9a6ec9e9…50d85` (main), `90397e2c…7e7a` (variant)
**Artefacts:** [`results/runs/d4_time_stepping_20260929T162518Z/`](../../results/runs/d4_time_stepping_20260929T162518Z/), [`…T162914Z/`](../../results/runs/d4_time_stepping_20260929T162914Z/) · **Plan:** [`docs/plans/cross-domain-plan.md`](../plans/cross-domain-plan.md) §5

**Status:** correctness ✅ · rate-budget opportunity gate (validation) ✅ · **equal-compute payoff (validation) ❌** · **D4-1 stays closed.**

**In one paragraph.** At equal *refinement count*, weighting the local-error estimate by the discrete co-state gives lower regret than the unweighted and goal-projected versions on all 40 test instances. At equal *total compute*, plain uniform refinement beats every adaptive policy at every compute level, on both the validation and test families, and under both objectives tried. The adaptive policies re-score the whole grid before every single refinement, and that scoring costs more than it saves at this problem size. So D4-0 gives no evidence that adaptive allocation pays for itself here, and the learned-critic comparison (D4-1) does not open.

---

## 1. What this run is

- **Domain:** `y' = Ay + b(t)` on `[0, 1]`: a damped oscillator coupled to one decaying or growing mode, forced by 2–4 Gaussian pulses. It is integrated with Crank–Nicolson (CN) on a grid that starts at 16 uniform intervals and is refined by bisecting intervals (finest grid 2,048). The quantity of interest is `cᵀy(T)` for a unit goal vector `c`.
- **Data:** generated in this repository from frozen seeds, like `HJoinBench-0`. **This is not real data.** Validation family: seed 1001, 40 instances. Test family: seed 2002, 40 instances (`config/instances.json`).
- **Declared objective (lower is better):** the cancellation-free goal-oriented error `Σ_j |Λ_{j+1}ᵀ τ_{j+1}|`, an upper bound on `|cᵀ(y(T) − y_N)|`, computed with the exact local errors `τ` from a matrix-exponential reference solution. The signed QoI error is a secondary metric (and the objective of the exploratory variant, §3.4).
- **Co-state:** the discrete recursion `Λ_j = M_jᵀ Λ_{j+1}` (comprehensive plan §4.2), **computed exactly from the known model**. It is a "Mode A" co-state (computed), not a learned one, and it is with respect to the discrete solution state. No model is trained.
- **Policies** (one refinement per step, greedy):

  | Policy | Score | Deployable |
  |---|---|---|
  | `uniform` | longest interval first | yes |
  | `random` | uniform random (8 draws per instance) | yes |
  | `residual` | step-doubling local-error norm `‖τ̂_j‖` | yes |
  | `goal_local` | `|cᵀτ̂_j|` (goal projection, no propagation) | yes |
  | `adjoint` | `|Λ_{j+1}ᵀ τ̂_j|` (co-state weighting) | yes |
  | `exact_tau_adjoint` | same with the true `τ` | no (diagnostic ceiling) |
  | `one_step_oracle` | exact one-step gain for every candidate | no |

  `residual`, `goal_local` and `adjoint` use the same `τ̂`, so they differ only in how they weight it.
- **Reference curve for regret:** the pointwise minimum over all seven policies at each budget (the "best-known" curve). It upper-bounds the true oracle, so the regret below is non-negative and a lower bound on true oracle regret. It depends on the policy set.
- **Ledger (CN steps).** A refinement is charged the steps re-solved from the refined interval to the end (`n + 1 − j`). Scoring is charged at every decision: 2 steps per interval for the estimate (`residual`, `goal_local`), 3 with the backward sweep (`adjoint`). Privileged policies are not charged and are left out of the equal-compute comparison.
- **Execution history (disclosed).**
  - A first execution at commit `ab7c7ba` charged 1 step per refinement and compared policies at equal refinement count only. I read its test-family refinement-count results.
  - A ledger audit against the plan's matched-`C_total` rule found the under-charge. A validation-only preview then showed uniform refinement ahead of every adaptive policy at equal compute.
  - The corrected ledger, the equal-compute check, its compute levels (1000, 2000, 4000, 8000 CN steps, from ledger arithmetic on validation) and the stricter D4-1 opening rule were added after that preview. They are stated in the notebook before the test family is read at equal compute.
  - The refinement-count results are unchanged by the ledger fix, because no policy reads costs. The files listed as identical are in the run README.
  - The first execution is not in the repository.

## 2. Gates

| Gate | Result | Source |
|---|---|---|
| Error representation `cᵀ(y(T) − y_N) = Σ Λᵀτ` | ✅ max abs difference 1.24e-14 (80 instances × 2 grids) | `tests/correctness.json` |
| Reference solution converged | ✅ max abs difference in `y(T)` between the finest grid and one twice as fine: 3.10e-12 (5 instances) | `tests/correctness.json` |
| Rate-budget opportunity (validation): area(uniform − best-known) ≥ 15 % of area(uniform) over `B = 0..48` | ✅ 0.457 (33.29 / 72.89) | `artifacts/opportunity_gate.json` |
| — instances passing individually | 39 / 40 (0.975); the exception is `d4_s1001_001` at 0.039 | `artifacts/opportunity_per_instance.csv` |
| Equal-compute payoff (validation): CI of `mean(J_adjoint − J_uniform)` below 0 at two adjacent levels | ❌ CI above 0 at all four levels | `artifacts/equal_compute_gate.json` |
| **D4-1 opens** | **No** | `reports/acceptance_report.json` |

## 3. Results

### 3.1 Equal refinement count (test family, 40 instances)

Regret is measured against the best-known curve. AURC is the trapezoidal area under the regret curve over `B = 0..48`. Source: `artifacts/test_summary.json`.

| Policy | Mean AURC | Mean regret at `B = 48` | Deployable |
|---|---:|---:|---|
| `exact_tau_adjoint` | 0.83 | 0.0049 | no |
| `adjoint` | **1.84** | **0.0127** | yes |
| `one_step_oracle` | 2.49 | 0.0860 | no |
| `residual` | 14.24 | 0.0989 | yes |
| `goal_local` | 17.11 | 0.1162 | yes |
| `uniform` | 31.30 | 0.0996 | yes |
| `random` (expected, 8 draws) | 56.62 | 1.0326 | yes |

Paired differences in AURC, bootstrap over the 40 test instances (10,000 resamples). Negative means the first policy has lower regret. Source: the same file.

| Comparison | Estimate | 95 % CI |
|---|---:|---|
| `adjoint − residual` | −12.41 | [−17.09, −8.30] |
| `adjoint − goal_local` | −15.27 | [−20.80, −10.47] |
| `adjoint − uniform` | −29.46 | [−38.11, −21.44] |
| `goal_local − residual` | +2.87 | [+1.51, +4.50] |
| `residual − uniform` | −17.05 | [−22.53, −12.16] |
| `exact_tau_adjoint − adjoint` | −1.01 | [−2.00, −0.21] |

The `adjoint` policy has lower AURC than `residual`, `goal_local` and `uniform` on **each of the 40 instances** (command in §6). Mean regret by budget (`figures/mean_regret_by_budget.csv`):

| `B` | `adjoint` | `residual` | `goal_local` | `uniform` | `one_step_oracle` |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.0762 | 0.6351 | 0.6934 | 1.5750 | 0.0125 |
| 16 | 0.0441 | 0.3983 | 0.4625 | 0.3702 | 0.0275 |
| 32 | 0.0179 | 0.1707 | 0.2678 | 0.5251 | 0.0761 |
| 48 | 0.0127 | 0.0989 | 0.1162 | 0.0996 | 0.0860 |

### 3.2 Equal total compute (deployable policies; validation gate, test read once)

Mean objective at fixed totals of CN steps. The `adjoint − uniform` column is the instance-bootstrap CI of the mean difference (positive means uniform is better). Sources: `artifacts/equal_compute_gate.json` (validation), `artifacts/test_equal_compute_summary.json` (test).

| Compute | Split | `adjoint` | `residual` | `uniform` | `adjoint − uniform` [95 % CI] | Refinements: `adjoint` / `uniform` |
|---:|---|---:|---:|---:|---|---|
| 1000 | validation | 1.167 | 1.220 | 0.317 | +0.849 [+0.500, +1.275] | 12.4 / 52.0 |
| 1000 | test | 0.886 | 1.062 | 0.268 | +0.618 [+0.356, +0.928] | 12.6 / 52.0 |
| 2000 | validation | 0.629 | 0.674 | 0.282 | +0.347 [+0.175, +0.564] | 21.1 / 72.0 |
| 2000 | test | 0.536 | 0.616 | 0.235 | +0.300 [+0.159, +0.466] | 21.3 / 72.0 |
| 4000 | validation | 0.352 | 0.368 | 0.080 | +0.272 [+0.159, +0.413] | 34.2 / 121.0 |
| 4000 | test | 0.301 | 0.323 | 0.067 | +0.233 [+0.131, +0.354] | 34.1 / 121.0 |
| 8000 | validation | 0.180 | 0.186 | 0.072 | +0.108 [+0.058, +0.175] | 53.1 / 160.0 |
| 8000 | test | 0.153 | 0.168 | 0.060 | +0.094 [+0.049, +0.147] | 53.3 / 160.0 |

- **Uniform refinement has the lower objective at every level, on both splits.** In the main run, `adjoint`, `residual` and `goal_local` also beat `random` at every level on both splits (`random` objective at 8000: 0.505 validation, 0.471 test), so the ordering is uniform, then the adaptive policies, then random. (In the signed-error variant, §3.4, `goal_local` is worse than `random` at 1000 on the test split.)
- **Within the adaptive family, on the test split**, `adjoint − residual` is negative at 1000, 2000 and 8000 (for example −0.176 [−0.293, −0.076] at 1000) and not distinguishable from zero at 4000 ([−0.059, +0.019]). `adjoint − goal_local` is negative at all four levels. So the extra backward sweep pays for itself against the unweighted estimate at three of four levels, but the whole adaptive family loses to uniform.

### 3.3 Other observations

- **`goal_local` is worse than `residual`** at equal refinement count (+2.87 [+1.51, +4.50]). Projecting the local error onto the goal without propagating it through the dynamics did not help; it hurt.
- **The greedy one-step oracle falls behind the co-state policies** as the budget grows. Its mean regret rises from 0.0125 at `B = 8` to 0.0860 at `B = 48`, while `adjoint` falls from 0.0762 to 0.0127.
- **`uniform` is not monotone** in the budget (regret 1.575, 0.370, 0.525, 0.0996 at `B = 8, 16, 32, 48`): it is good only right after a full pass over the grid (`B = 16` and `B = 48`).
- **The fraction of oracle advantage is undefined for 1 of 40 test instances** (`d4_s2002_037`, where `uniform` already equals the best-known curve at `B = 48`).

### 3.4 Exploratory variant: signed QoI error as the objective

Run `…T162914Z`, designed after the main results were known and **not used for any gate decision**. Source: its `artifacts/test_summary.json`, `artifacts/equal_compute_gate.json` and `artifacts/test_equal_compute_summary.json`.

- Rate-budget opportunity (validation): ✅ 0.702. Equal-compute payoff (validation): ❌. D4-1 stays closed.
- Equal refinement count: `adjoint − residual` −5.39 [−9.64, −2.22], `adjoint − goal_local` −7.55 [−12.74, −3.58], `adjoint − uniform` −8.62 [−13.62, −4.44]. `exact_tau_adjoint − adjoint` is −1.25 [−2.74, +0.03], not distinguishable from zero.
- Equal compute (test): `adjoint − uniform` is +0.410, +0.197, +0.111, +0.054 at 1000, 2000, 4000, 8000 (all CIs above 0). `adjoint − residual` is not distinguishable from zero at any level.
- So the two qualitative conclusions do not depend on which objective is used. The finer ranking inside the adaptive family (`adjoint` vs `residual` at equal compute) does.

## 4. Interpretation

**Observed** (from the tables above):

- At equal refinement count, co-state weighting of the same `τ̂` lowers regret against both unweighted and goal-projected weighting, on every test instance, and the effect survives a change of objective.
- The co-state's advantage over the unweighted estimate mostly survives paying for its extra sweep, in the main run. It does not survive the change to the signed objective.
- Plain uniform refinement beats every deployable adaptive policy at equal total compute, on both splits, at all four levels, under both objectives.

**Hypotheses (not tested here):**

1. *The compute deficit comes from scoring the whole grid before each single refinement.* For `n` intervals, an adaptive step pays `3n` scoring steps (`2n` for `residual` and `goal_local`) on top of the re-solve, while a uniform step pays only the re-solve. That is why `uniform` gets 52 refinements at 1000 steps against 12–16 for the adaptive policies (§3.2). **Diagnostic:** a scheme that marks many intervals per scoring pass, or updates estimates incrementally, evaluated at equal compute on validation (roadmap D4-0b).
2. *At this problem size, resolving the pulses uniformly is cheap enough that adaptivity has little left to win.* At 8000 steps `uniform` reaches about 16 + 160 = 176 intervals (mean width about 0.006, from the 160 refinements in §3.2), while the pulse widths are drawn from 0.01–0.04 (`sample_instances` in `src/adjointrwm/domains/linear_ode.py`). **Diagnostic:** the same comparison on a harder family (sharper pulses, longer horizon, stiffer coupling) or a larger system, where an equal-compute win is at least possible.
3. *The gap between `adjoint` and `exact_tau_adjoint` reflects the error estimator, not the co-state.* The step-doubling `τ̂` can miss error on under-resolved pulses. **Diagnostic:** a better local estimator, holding the co-state fixed.
4. *`goal_local` loses to `residual` because projecting onto `c` discards error components that reach the goal through the dynamics.* **Diagnostic:** replace the goal vector by a random rotation of it and check that `goal_local` degrades.

## 5. Design issues found

1. **The action-cost ledger was wrong in the first execution** (1 step per refinement instead of the re-solve). Fixed before anything was committed. It changed no refinement-count result, but it would have understated every equal-compute number. See §1 for what was and wasn't affected.
2. **Signed QoI error rewards lucky cancellations.** A greedy oracle on `|cᵀe_N|` won by finding cancellations of opposite-sign local errors, so the declared objective is the bound `Σ|Λᵀτ|` (plan §5, D4).
3. **The greedy one-step oracle is not a true oracle** and can be beaten, which would give negative regret. Regret uses the best-known curve, which depends on the policy set. `exhaustive_oracle_curve` measures the gap to the true oracle only on tiny instances.
4. **Policies that score every interval before every refinement are a poor match for an equal-compute comparison.** That is a property of the policy implementation, not of adaptive allocation in general (hypothesis 1).
5. **Privileged references are not charged for their scoring**, so equal-compute numbers exist only for deployable policies.
6. **One instance in 40 fails the rate-budget opportunity individually** (`d4_s1001_001`, 0.039). The family gate passes.
7. **The D4-1 opening rule was tightened after a validation preview.** The tightening is conservative (it can only keep D4-1 closed) and is documented in the notebook, but it is a post-preview change and is reported as such.

## 6. Commands

The per-instance win counts in §3.1 come from:

```bash
python - <<'EOF'
import pandas as pd
w = pd.read_csv("results/runs/d4_time_stepping_20260929T162518Z/artifacts/test_aurc_by_instance.csv").pivot_table(index="instance", columns="policy", values="aurc")
for other in ("residual", "goal_local", "uniform"):
    print("adjoint has lower AURC than", other, "on", int((w.adjoint < w[other]).sum()), "of", len(w), "instances")
EOF
```

Every other number is read from the named files. The candidate-cost formulas, the ledger and the best-known curve are in `src/adjointrwm/domains/` and covered by `tests/test_domains.py`.

## 7. What this run does and does not support

**Supports:**

- The D4 machinery is correct: the exact error representation holds to rounding and the reference solution is converged (`tests/correctness.json`).
- On this analytic family, at equal refinement count, co-state weighting of a step-doubling local-error estimate lowers regret against unweighted and goal-projected weighting (single-refinement greedy policies, 40 instances, one family).
- On this analytic family, with this ledger and single-refinement greedy scoring, plain uniform refinement beats every deployable adaptive policy at equal total compute.
- The result is reproducible: a clean re-execution of the same commit reproduced every result file byte for byte (run README).

**Does not support:**

- Any statement that the adjoint helps or hurts relative to a direct critic (H2). No learned arm exists in D4-0.
- Any statement that adaptive allocation is useless. The equal-compute result is for one problem size, one family and one scoring scheme (hypotheses 1–2).
- Any statement about other domains, real data, or cross-domain transfer.
- Any systems claim: the runs were on CPU, and timings are not reported.

**Corrects earlier statements.**

- The first draft of the run README said D4-1 "opens". That was written before the equal-compute check existed and was replaced.
- My chat summary of the cross-domain work said that if the opportunity gate passed, D4-1 would be designed next. With the equal-compute condition, D4-1 stays closed.

## 8. Next steps (roadmap IDs in `docs/plans/roadmap.md`, Track D)

1. **D4-0b:** a compute-efficient scoring scheme (mark many intervals per pass, incremental estimates, or a learned amortised scorer), validated at equal compute against `uniform` on validation. Test data stays unread until the scheme is frozen. If it also fails, try a harder family (hypothesis 2) as a new, separately gated benchmark.
2. **D4-1** stays closed until D4-0b passes the equal-compute condition.
3. **DL:** the licence survey for permissively licensed data and models for the other domains, per Roman's decision. Nothing for D1–D3 is built before it reports.
4. **Carry the lesson to Track E:** in pilot v2 (B2), charge the co-state and critic scoring costs explicitly and compare at matched compute. The pilot's own systems benchmark already shows the gap the amortised co-state exists to close (exact teacher 15.3 ms against 0.28 ms amortised at p50, `results/runs/droid100_adjoint_20260929T070629Z/benchmarks/systems_benchmark.json`).
