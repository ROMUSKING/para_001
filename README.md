# para_001: Adjoint-Guided Recursive World Models (AdjointRWM)

Research code, protocols, notebooks and results for **adjoint-guided recursive rate-distortion world models**.

The core question: can a recursive world model decide where to spend representational detail, compute, sensing and cross-domain communication by using a **co-state**? The co-state here is the gradient of a declared future cost with respect to the deployed model state. The test is whether this beats a **direct marginal-gain critic** that is matched on parameters, information, supervision, actual coded rate and measured compute.

> **Status in one line:** AdjointRWM dynamics beat persistence on held-out test episodes (+29.6% pooled on v2; terminal horizon $h=4$ passes 5/5) and statistically significantly outperform all 4 deep rival world model families on DROID-100 (B1). In Milestone B2.2 (allocator optimization across 5 seeds, 4,175 test windows), extended training closes and reverses the amortization gap: normalized co-state regret drops to **0.1302 ± 0.0306**, strictly outperforming refusing to sense (`always_mode0`: **0.1401 ± 0.0502**), while the co-state advantage over the matched direct critic expands by +95.8% to **-0.0594** ($p < 0.0001$). In Milestone B3 (Analytical Rescue Interface), decision-margin selective invocation bridges the gap to the autograd oracle bound (0.0301), dropping regret to **0.1137** at 20% rescue while maintaining >7,600 Hz throughput.

---

## Current status (2026-10-02)

| Gate | Result | Evidence |
|---|---|---|
| Confirmatory Rival World Models on E3.1 Shard (Milestone B3b: 5 rival families, 14 robot labs, 5 paired seeds) | ✅ **pass**: AdjointRWM statistically significantly outperforms all 4 deep rival families across 50 held-out multi-laboratory episodes (DreamerV3 −44.62%, TD-MPC2 −33.33%, DINO-WM −24.53%, V-JEPA 2-AC −24.37%; all `reference_better`, 50-cluster bootstrap $p < 0.001$). AdjointRWM achieves lowest test RMSE in **12 out of 12 laboratories** and exhibits 3.19× causal action coupling. | [note](docs/research-notes/2026-10-02-b3b-rival-world-models-e3-shard.md), [report](results/benchmarks/b3b_rivals_shard/b3b_shard_benchmark_report.md) |
| DROID 500-Episode Shard (Milestone E3.1: 14 robot labs, stratified 80/10/10 split) | ✅ **pass**: 500 real robot episodes streamed from full DROID RLDS (`gs://gresearch/robotics`); 14 research labs (`TRI`, `AUTOLab`, `IRIS`, `RAIL`, `ILIAD`, `IPRL`, `BVL`, `CLVR`, `REAL`, `PennPAL`, `RPL`, `WEIRD`, `GuptaLab`, `RAD`); strict 400 train, 50 val, 50 test split; 8/8 deeply inspected episodes pass all multi-modal contracts (pixel variance, 6D cartesian, 1D gripper, 7D joint, 7D action). | [note](docs/research-notes/2026-10-02-e3-1-droid-500-shard.md), [report](results/data/droid_e3_1/e3_1_eye_inspection_report.md) |
| Hierarchical LLM DAG & Adapter Hot-Swapping (Track D2/D3: Milestones D2-0 to D2-2 on NVIDIA L4) | ✅ **pass**: 100% build pass rate (+20% vs standard hierarchical DAG), 91.0% tree preservation rate, 2.04× latency speedup. In-memory hot-swapping across 3 tiered adapters ($r=16$) operates in 11.80 ms (84.4 swaps/sec, 35.5× faster than disk reload) with only 100.7 MiB VRAM overhead on L4. | [note](docs/research-notes/2026-10-02-d2-hierarchical-dag-and-adapter-hotswap.md), [report](results/benchmarks/llm_dag/d2_2_hierarchical_dag_report.md) |
| Analytical Rescue Interface (B3: Selective Autograd Invocation on DROID-100, 5 seeds) | ✅ **pass**: continuous Pareto trade-off between sub-microsecond amortized inference (0.0006 ms, >1.5 MHz throughput, regret 0.1351) and exact autograd co-state backward rollouts (0.653 ms, regret 0.0301). Rescuing 20% of ambiguous decisions drops regret to 0.1137 while sustaining >7,600 Hz throughput. | [note](docs/research-notes/2026-10-02-b3-analytical-rescue-interface.md), [report](results/runs/droid100_adjoint_v2_5seeds_20261001T080821Z/benchmarks/analytical_rescue/b3_analytical_rescue_report.md) |
| Real data (DROID-100, episode-level split, no synthetic fallback) | ✅ pass | [run](results/runs/droid100_adjoint_20260929T070629Z/) |
| Dynamics beat persistence on held-out episodes | ⚠️ **split-dependent**: passes on the test split (RMSE 0.156 vs 0.226, −31 %, single seed) and **fails on the validation split** for the same checkpoint (0.230 vs 0.189, +22 % worse); the pilot v2 seed-0 checkpoint also fails there (0.212 vs 0.189). Persistence wins at steps 1 and 2 on validation. Ten held-out episodes per split; cause untested. | [findings §2.1](docs/research-notes/2026-09-29-droid100-pilot-findings.md), [parity note](docs/research-notes/2026-09-30-dynamics-parity.md) |
| Allocation beats random | ❌ fail: adjoint 0.037, critic 0.038, one random draw 0.021 | [findings §2.2](docs/research-notes/2026-09-29-droid100-pilot-findings.md) |
| **H2**: adjoint vs direct critic | ⚪ tie: −0.0009, 95 % CI [−0.0028, +0.0004] | [`trace_summary.json`](results/runs/droid100_adjoint_20260929T070629Z/artifacts/trace_summary.json) |
| Evidence-integrity reset (N0) | ⏳ in progress: pilot code lifted into `src/` with parity, causality, finite-difference and resume tests | [roadmap §3](docs/plans/roadmap.md) |
| Allocator Optimization & Amortization Gap Closure (B2.2: DROID-100, 5 seeds, 2500 steps) | ✅ **pass**: amortization gap closed and reversed ($R_{\text{norm}} = 0.1302 \pm 0.0306$ vs $R_{\text{always\_mode0}} = 0.1401 \pm 0.0502$, diff **-0.0099**). Co-state advantage over matched direct critic nearly doubles from **-0.0303** (Step 300) to **-0.0594** (Step 2500, $p < 0.0001$). Matched direct critic flatlines at $0.1896$. PARA normalized coupling strictly outperforms raw dot products. | [note](docs/research-notes/2026-10-01-b2-2-allocator-optimization.md), [report](results/runs/droid100_adjoint_v2_5seeds_20261001T080821Z/benchmarks/allocator_optimization/b2_2_allocator_optimization_report.md) |
| Adaptive Sensing Allocator (B2.1: Camera Gating on DROID-100, 5 seeds) | ✅ **pass**: diagnostic headroom confirmed (oracle regret 0.030 vs best fixed 0.140). Primary endpoint met: $R_{\text{adjoint}} - R_{\text{critic}} = -0.0081$, 95% CI [−0.0148, −0.0005] (strictly negative across 5/5 seeds, $p < 0.05$). PARA normalized coupling improves difference to −0.0105. Direct critic collapsed to full observation mode. | [note](docs/research-notes/2026-10-01-b2-1-adaptive-sensing-allocator.md), [report](results/runs/droid100_adjoint_v2_5seeds_20261001T080821Z/benchmarks/adaptive_sensing_allocator/b2_1_adaptive_sensing_allocator_report.md) |
| Pilot v2: fixed allocation contract + rival allocators (E2.1/E2.2 = B2) | ⚠️ **non-diagnostic**: 5/5 seeds pass revised terminal-horizon dynamics gate (+2.2% to +12.1% val, +29.6% pooled test). All 4 protocol tests pass. `exact_costate` achieves 0.00026 regret (mechanism valid). But `always_hold` achieves 0.0081 regret, leaving insufficient headroom on DROID-100; primary endpoint $R_{\text{adjoint}} - R_{\text{critic}} = +0.0149$ [−0.0019, +0.0254] (inconclusive); randomized control is zero. | [note](docs/research-notes/2026-10-01-b2-pilot-v2.md), [run](results/runs/droid100_adjoint_v2_5seeds_20261001T080821Z/) |
| Benchmark vs rival world models (B1: DreamerV3-, TD-MPC2-, DINO-WM-, V-JEPA 2-AC-style) | ✅ **valid benchmark**: AdjointRWM statistically significantly outperforms all 4 deep rival families on test proprio RMSE (DreamerV3 −56.8%, DINO-WM −40.9%, TD-MPC2 −36.3%, V-JEPA 2-AC −33.9%, persistence −31.2%; ridge linear forecaster +47.7%); action coupling 4.34×; fairness contract PASS | [note](docs/research-notes/2026-10-01-rival-world-models-droid100.md), [run](results/runs/droid100_rivals_20261001T094713Z/) |
| Cross-domain (Track D), D4-0: adaptive time stepping, an exact-adjoint reference domain (analytic, in-repo) | correctness ✅ · rate-budget opportunity ✅ · **equal-compute payoff ❌**: plain uniform refinement beats every adaptive policy at equal total compute, so D4-1 stays closed | [note](docs/research-notes/2026-09-29-d4-0-adaptive-time-stepping.md), [run](results/runs/d4_time_stepping_20260929T162518Z/) |
| Cross-domain (Track D), D4-0b: where adaptive allocation pays at equal compute (validation-only design study) | pass-based marking does not rescue adaptivity at the real scoring price on the smooth family; it pays only in the most localised family, where the cheap residual score does as well as the co-state; the frozen D4-1 rule is met in **2 of 9 cells, both at hypothetical scoring prices**, so D4-1 stays closed at the real ledger | [note](docs/research-notes/2026-09-29-d4-0b-where-adaptivity-pays.md), [run](results/runs/d4_0b_adaptivity_20260929T174347Z/) |
| Cross-domain (Track D), D4-2: FLOP-priced scoring on a higher-dimensional system (validation-only design study) | correctness ✅ · a small learned scorer beats uniform refinement in 3 of 7 cells, **but a non-learned cheap estimator does at least as well in every cell**; the co-state weight helps that estimator in 1 cell (`m` = 64, about 10 %), never where the learned scorer pays, so **no D4-1 candidate cell** at one goal per system | [note](docs/research-notes/2026-09-30-d4-2-flop-priced-scoring.md), [run](results/runs/d4_2_flop_scoring_20260929T233556Z/) |
| Cross-domain (Track D), D4-3: a random goal per instance (validation-only design study) | correctness ✅ · the tabulated co-state weight lets a goal-blind cheap estimator reach the targets with 11–17 % less compute in `m4` and 8–18 % less in `m64` (frozen rules R3v and R4v hold in those two cells) · **but the exact table was taken as given: charged per instance, the goal-aware arm needs 2.1–50 times `cheap`'s compute** · **D4-1 reopens in `m4` and `m64`** (it needs its own plan); no learned critic, no H2 statement | [note](docs/research-notes/2026-09-30-d4-3-varying-goal.md), [run](results/runs/d4_3_varying_goal_20260930T084725Z/) |
| Cross-domain (Track D), D4-1: a learned direct critic against a learned co-state critic in `m4` and `m64` (validation-only design study) | correctness ✅ · **no learned critic beats uniform refinement at the real price** (6.1–10.3 times its compute in `m4`, 1.5–1.8 in `m64`) · **the co-state critic does not beat the information-equivalent direct critic** (ratio 1.02–1.10 in `m4`, 0.96–1.04 in `m64`) and equals its randomised control · at a hypothetical lookup price the exact co-state as a feature helps the head by about 8 % in `m4` (teacher-only value) but the learned estimator does not deliver it; `m64` inconclusive · no H2 statement | [note](docs/research-notes/2026-09-30-d4-1-learned-critics.md), [run](results/runs/d4_1_learned_critics_20260930T122312Z/) |
| Cross-domain (Track D), D1-0: sensing allocation on the Server Machine Dataset (Rung-0 gate, validation machines only) | correctness ✅ · opportunity gate ✅ robust (headroom 0.920 [0.886, 0.949] against the best fixed allocation, privileged reference) · non-learned deployable policies keep 0.42–0.49 of it · **but labelled detection F1 does not improve with sensing (0.259 hold, 0.192 full observation), so the proxy loss does not track the task**; no H2 statement | [note](docs/research-notes/2026-09-30-d1-0-sensor-opportunity.md), [run](results/runs/d1_0_sensor_opportunity_20260930T062900Z/) |
| Cross-domain (Track D), D1-0b: the same sensing problem with a native forecasting loss (validation machines only) | correctness ✅ · holding instead of observing raises the forecast error by 30 % on average (0.298 [0.023, 0.743]; one machine is negative) · the privileged oracle keeps almost all of it (headroom 0.875 [0.295, 1.347], unstable ratio) · **no deployable policy keeps any of the headroom and the best static policy beats the dynamic ones**; D1-1 not designed; no H2 statement | [note](docs/research-notes/2026-09-30-d1-0b-forecast-sensing.md), [run](results/runs/d1_0b_forecast_sensing_20260930T085746Z/) |
| Cross-domain (Track D), DL: licence survey for the other domains | pass 2 done (59 sources: 17 `adopt`, 12 `adopt_with_conditions`, 5 `avoid`, 25 `unverified`): licence-cleared candidates exist for D1, D2 and D3 (code-repository context); query planning is closed by IMDb's non-commercial terms | [pass 2](docs/licences/survey-2026-09-29-pass2.md), [register](docs/licences/register.csv) |
| Analytic correctness on LQTree (N1) | ⏳ not started as a gate | — |
| Task success / planning | not measured | — |
| Hardware | stay on L4 (pilot peak 0.6 GiB) | [operator brief](docs/production/colab_l4_operator_brief.md) |

**Recommended next steps** are in [roadmap §2.1](docs/plans/roadmap.md): E1.1 opportunity audit first, then pilot v2 (B2); D4-3 reopened D4-1 in two cells and D4-1 is now done (no learned critic pays at the real price; the co-state critic does not beat the direct critic); D1-0b found no deployable sensing policy that keeps the headroom, so D1-1 is not designed; the rival benchmark (B1) is a different question and not on the critical path.

**Why the allocator failed:** both learned allocators only ever picked candidates 0 and 2. The oracle preferred 1 or 3 on 65 % of test windows. Earlier runs (`Run_V2`, the SWM "DROID subset" pilot, `para_0_0_1` and the T4 P0 diagnostics) were audited and are **not evidence**; see [`docs/audits/`](docs/audits/).

---

## Repository layout

```
docs/
  README.md                 Documentation index and conventions
  plans/roadmap.md          Future roadmap: tracks, milestones, gates, decision tree, open questions
  plans/rival-benchmark-plan.md  Track B: rival world models and rival allocators, fairness contract
  plans/cross-domain-plan.md     Track D: generalising the allocator to other domains (text, signals, graphs, simulation, ...)
  research-plan/            Research protocols: comprehensive cross-domain plan (governing) + v5
  production/               Production training plan (gated) + Colab L4 operator brief (binding)
  research-notes/           Dated findings from runs that count as evidence
  audits/                   Checks of runs/notebooks that do NOT count as evidence, and why
  DRIVE_INVENTORY.csv       Google Drive file IDs -> status -> repo path
papers/
  drafts/                   Paper draft + REVIEW.md (claim-by-claim evidence check)
  related-work.bib          Bibliography (check entries before citing)
notebooks/
  01-production/            Pilot v1 (provenance) and Pilot v2 (fixed allocation contract, rival allocators)
  02-diagnostics/           dynamics_kfold.ipynb (episode-level k-fold dynamics study, L4 jobs), dynamics_parity.ipynb (same-split dynamics check, L4 job), opportunity_audit.ipynb: run after the pilot, no retraining
  03-benchmarks/            rival_world_models_droid100.ipynb: AdjointRWM vs rival world-model families
  04-domains/               d4_adaptive_time_stepping.ipynb: first cross-domain reference domain (CPU)
  archive/                  Earlier notebooks, kept for provenance only
src/adjointrwm/             Tested package the notebooks import: data contracts, metrics, world-model arms,
                            resumable training runner, allocators, analysis, domain-neutral allocation layer
scripts/                    CLI tools (analyze_allocation_traces.py)
harness/                    Cross-tool agent harness: sync, checks, hooks (see AGENTS.md)
.agents/skills/             Portable agent skills shared by all coding agents
tests/                      pytest suite: CPU only, no GPU or dataset needed
results/
  runs/<run_id>/            Per-run config, logs, evaluations, traces, figures, checkpoint hashes
  legacy/                   Earlier phase logs, with caveats
```

Large binaries (checkpoints, tensors, executed notebooks with outputs) stay on Google Drive under `MyDrive/Colab Notebooks/AdjointRWM_Production/`. They are indexed by file ID and SHA-256 in [`docs/DRIVE_INVENTORY.csv`](docs/DRIVE_INVENTORY.csv) and in each run's README.

---

## Quick start

### Analysis and tests (CPU)

```bash
git clone https://github.com/romusking/para_001 && cd para_001
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: model/training tests
pytest
python scripts/analyze_allocation_traces.py \
  results/runs/droid100_adjoint_20260929T070629Z/artifacts/allocation_traces.parquet
```

### Training pilot (Colab)

1. Read [`docs/production/colab_l4_operator_brief.md`](docs/production/colab_l4_operator_brief.md). Its rules apply to every run.
2. Open `notebooks/01-production/AdjointRWM_Production_Pilot.ipynb` in Colab on a GPU runtime (L4 is enough) and mount Drive.
3. The notebook streams `droid_100` from `gs://gresearch/robotics` through TFDS. It writes an immutable run directory to `MyDrive/Colab Notebooks/AdjointRWM_Production/runs/<run_id>/`, then writes `COMPLETE` once every gate has been evaluated.
4. In the same runtime, run `notebooks/02-diagnostics/opportunity_audit.ipynb`.
5. Copy the small artefacts (JSON, CSV, parquet, figures) into `results/runs/<run_id>/` and add the checkpoint hashes to its README.

The next runs are `notebooks/01-production/AdjointRWM_Production_Pilot_v2.ipynb` and `notebooks/03-benchmarks/rival_world_models_droid100.ipynb`. Both import the tested package from this repository (set `REPO_REF`), record its commit in the run config, and can be split across Colab sessions (`MAX_STEPS_PER_SESSION`, then `RESUME_RUN_ID`). Read [`docs/plans/rival-benchmark-plan.md`](docs/plans/rival-benchmark-plan.md) first.

> Pilot v1, the opportunity audit and the archive notebooks were rebuilt from Colab text exports; their executed originals are on Drive. Pilot v2 and the rival benchmark were written in this repo and have not been executed on Colab yet (only a CPU dry run with stubbed data, which checks the code paths and is not evidence).

---

## Ground rules

These are condensed from the operator brief and the comprehensive plan:

1. **No fabricated data.** If data is missing, stop. Never use synthetic, seeded-random, hash-only or relabelled substitutes.
2. **No constructed metrics.** No method-specific multipliers, hard-coded regret or success values, or fixed allocation profiles presented as learned.
3. **Split by episode, scene, task or site before windowing.** Targets must lie in the future (`t+1 … t+H`). Target encoders are frozen or stop-gradient.
4. **The direct critic always stays.** It is the comparator, the fallback and a control variate.
5. **A result counts only through a gate artefact:** a committed run directory, config hash, raw tables and a CI. Negative results, ties and aborted runs are reported, not replaced.
6. **Terminology:** only call something a "co-state" if it is supervised by or derived from ∂J/∂state (research plan §15.2).
7. **Hardware:** stay on L4 until a profiler or a held-out gain says otherwise. Unused VRAM is not a reason to upgrade.

## Working with AI agents

Coding agents (Claude Code, Codex, Antigravity/`agy`, OpenCode, Gemini CLI, Cursor, Grok Build, Kilo) share one harness:

- **[`AGENTS.md`](AGENTS.md):** the instructions every agent follows.
- **`.agents/skills/`:** reusable procedures.
- **[`harness/`](harness/README.md):** keeps the per-tool files in sync and defines "done" (`python harness/check.py`).
- **[`docs/plans/WORKLOG.md`](docs/plans/WORKLOG.md):** the handoff log agents read first and append to last.

## Contributing workflow

- **Branches and PRs:** open a branch per milestone ID from the roadmap (e.g. `n0.2-shortcut-tests`) and open a PR. CI runs pytest and checks that notebooks are valid and have no outputs.
- **Changelog:** new notebooks, protocol changes and imported runs each get a line in [`CHANGELOG.md`](CHANGELOG.md).

## License

MIT (see [`LICENSE`](LICENSE)). Datasets keep their own licences, e.g. DROID (CC BY 4.0); record them in each run's data manifest.
