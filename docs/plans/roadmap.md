# Roadmap

**Last updated:** 2026-09-29, after run `droid100_adjoint_20260929T070629Z`.
**Governing document:** `docs/research-plan/adjoint_guided_recursive_world_model_research_plan.md` (protocol dated 24 Sep 2026). When this roadmap and the plan disagree, the plan wins.

## Where we are

| Plan stage | State | Evidence in repo |
|---|---|---|
| Stage 0: LQTree analytic correctness | Notebooks exist on Drive but aren't reviewed or committed | none yet |
| I.2: realised direct baseline and coder | Not started. The pilot has a direct critic but no saturation or realisation protocol, and its coder isn't canonical. | — |
| I.3A: adjoint contribution under C0 | First real-data pilot: **tie with the critic; both collapsed; lost to random** | `results/runs/droid100_adjoint_20260929T070629Z/` |
| Stage A: controlled multiscale visual dynamics | Not started | — |
| Phase II: manipulation or embodied | **Locked** by the plan's unlock gate | — |

The DROID pilot is manipulation data but is evaluated offline on prediction and allocation. It is useful as an engineering shakedown. Under the plan, though, it is **not** a Stage A or Phase II result, and it must not be reported as one.

---

## Milestone 1: diagnose the pilot failure (days, L4, no retraining)

| # | Task | Output | Done when |
|---|---|---|---|
| 1.1 | Run `notebooks/02-diagnostics/opportunity_audit.ipynb` on the existing checkpoint | `results/runs/<run>/diagnostics/opportunity_audit.json` | JSON committed |
| 1.2 | Read it against the decision table in the notebook | Research note appended to the findings | One of the five outcomes identified |
| 1.3 | Check whether the train and test oracle shares differ (collapse hypothesis) | `train_vs_test` block | Confirmed or rejected |

**Decision point.**
- If `exact_costate_first_order` doesn't beat `random_expected` on validation, stop tuning the amortised estimator. The benchmark or the candidate design is the problem.
- If the opportunity proxy fails (`passes: false`), redesign the benchmark (Milestone 2b) before any H2 comparison.

## Milestone 2: fix the evaluation contract (1 week)

Changes to `AdjointRWM_Production_Pilot.ipynb`. Bump it to `…_v2`, keep v1 for provenance, and record each change in `CHANGELOG.md`.

1. **Hold candidate:** add a zero-effect, zero-cost candidate so "don't refine" is legal (plan §3.4).
2. **Consistent semantics:** train candidate effects in the regime they are evaluated in.
   - Option (a): single choice, i.e. a one-hot mask during stage 1.
   - Option (b): budgeted subset selection at evaluation, the plan's best-first allocator (§5.7).
   - Pick one and freeze it before the next run.
3. **Baselines:** exact expected-random regret, `always_c_k` for every k, and the oracle.
4. **Traces:** log `exact_gain[N, K]`, the chosen index, the gate probability and the stop objective. The `adjointrwm.analysis` functions then run on the traces alone.
5. **Gate target:** replace the fixed `> 0.002` threshold with a margin set on validation, or train the gate on the sign of the regret difference with class weighting.
6. **Ceiling arm:** score candidates with the **exact** co-state in the same first-order rule. This is the P2/P3 teacher arm in the plan's privilege taxonomy.
7. **Per-horizon reporting:** report dynamics RMSE for each horizon step, not only the mean.

### 2b. If the opportunity gate fails: redesign the candidates

- Make candidates structurally different: resolution or crop refinements on the visual input, a second exterior camera as a queryable port (Track X), or extra rollout depth. Replace the four learned latent nudges capped at 0.10·tanh.
- Tune until the oracle beats the best fixed choice by at least 15 % on validation (plan §7.2B-1), with a CI. Only then open the test split.

## Milestone 3: minimal confirmatory-style rerun (2–3 weeks, L4)

- **Seeds and splits:** 5 seeds, paired across the adjoint and critic arms, with the same episode split. Record the split hash in the run config.
- **Primary endpoint:** oracle regret per decision (the plan's Allocation track). Report adjoint − critic with a 95 % episode-cluster bootstrap CI (`episode_bootstrap_ci`), plus AURC over at least 3 budgets.
- **Matching:** parameters, inputs, labels and tuning budget must be matched between the adjoint and critic heads. Log head parameter counts.
- **Data:** if DROID-100 is kept, consider growing to more episodes (full DROID via RLDS) so that 10 test episodes aren't the unit of inference. Otherwise use k-fold over episodes.
- **Promotion rule:** a run is promoted only if dynamics, opportunity and allocation gates all pass on validation. The test split is read once per frozen config.

## Milestone 4: put the plan's early stages on a firm footing (in parallel)

- Review the latest `Stage0_LQTree_Benchmark.ipynb` from Drive. Move it to `notebooks/00-stage0/`, strip outputs, and commit the logs of its gate checks: analytic vs autograd co-state, candidate-effect correctness, determinism.
- Write the preregistration freeze manifest (`prereg/freeze_manifest.yaml`) for the fields the plan says must be locked before confirmatory runs: `delta_oracle_min`, `delta_H2_min`, the seed schedule and the split contract.

## Milestone 5: code hygiene (ongoing)

- Lift reusable pieces out of the pilot notebook into `src/adjointrwm/`: `WindowDataset`, the model, `objective_per_sample`, `exact_adjoint_targets` and the evaluation code. Add tests for causality (no future leakage), for the exact co-state against finite differences, and for the checkpoint round-trip. The notebook should then import from the package.
- Add `nbstripout` as a pre-commit hook so notebook outputs don't bloat the repo. Executed copies live on Drive, listed in `docs/DRIVE_INVENTORY.csv`.
- Keep large binaries (checkpoints, `.npz` tensors) on Drive. Record their SHA-256 in each run's README.

## Milestone 6: write-up (only after Milestone 3)

- **Negative or tie result:** write a workshop paper or tech report. It covers the protocol, the matched direct critic, the opportunity gate, and the collapse failure mode with its diagnosis. This is still a useful contribution.
- **Positive result that survives the CIs:** extend to Stage A's basis classification before any main-track submission.
- **Retire the current draft's unsupported claims** (see `papers/drafts/REVIEW.md`).

## Compute

Everything through Milestone 3 fits on the Colab L4. The pilot peaked at 0.6 GiB, with the exact adjoint at about 15 ms per batch. An A100 becomes worth it only for Stage A/L-scale representation training, or when a run is measured to be compute-bound.
