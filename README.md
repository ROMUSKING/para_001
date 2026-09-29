# para_001: Adjoint-Guided Recursive World Models (AdjointRWM)

Research code, protocols, notebooks and results for **adjoint-guided recursive rate-distortion world models**.

The core question: can a recursive world model decide where to spend representational detail, compute, sensing and cross-domain communication by using a **co-state**? The co-state here is the gradient of a declared future cost with respect to the deployed model state. The test is whether this beats a **direct marginal-gain critic** that is matched on parameters, information, supervision, actual coded rate and measured compute.

> **Status in one line:** the real-data dynamics model works. The allocator doesn't yet. There is **no evidence yet** that the adjoint beats the direct critic. Production promotion is blocked, and the next milestones are in the [roadmap](docs/plans/roadmap.md).

---

## Current status (2026-09-29)

| Gate | Result | Evidence |
|---|---|---|
| Real data (DROID-100, episode-level split, no synthetic fallback) | ✅ pass | [run](results/runs/droid100_adjoint_20260929T070629Z/) |
| Dynamics beat persistence on held-out episodes | ✅ pass: RMSE 0.156 vs 0.226 (−31 %), single seed. Persistence still wins at one step ahead. | [findings §2.1](docs/research-notes/2026-09-29-droid100-pilot-findings.md) |
| Allocation beats random | ❌ fail: adjoint 0.037, critic 0.038, one random draw 0.021 | [findings §2.2](docs/research-notes/2026-09-29-droid100-pilot-findings.md) |
| **H2**: adjoint vs direct critic | ⚪ tie: −0.0009, 95 % CI [−0.0028, +0.0004] | [`trace_summary.json`](results/runs/droid100_adjoint_20260929T070629Z/artifacts/trace_summary.json) |
| Evidence-integrity reset (N0) | ⏳ in progress: pilot code lifted into `src/` with parity, causality, finite-difference and resume tests | [roadmap §3](docs/plans/roadmap.md) |
| Pilot v2: fixed allocation contract + rival allocators (E2.1/E2.2 = B2) | ⏳ notebook ready, **not run** | [plan §5](docs/plans/rival-benchmark-plan.md) |
| Benchmark vs rival world models (B1: DreamerV3-, TD-MPC2-, DINO-WM-, V-JEPA 2-AC-style) | ⏳ notebook ready, **not run** | [plan](docs/plans/rival-benchmark-plan.md) |
| Cross-domain (Track D), D4-0: adaptive time stepping, an exact-adjoint reference domain (analytic, in-repo) | correctness ✅ · rate-budget opportunity ✅ · **equal-compute payoff ❌**: plain uniform refinement beats every adaptive policy at equal total compute, so D4-1 stays closed | [note](docs/research-notes/2026-09-29-d4-0-adaptive-time-stepping.md), [run](results/runs/d4_time_stepping_20260929T162518Z/) |
| Cross-domain (Track D), D4-0b: where adaptive allocation pays at equal compute (validation-only design study) | pass-based marking does not rescue adaptivity at the real scoring price on the smooth family; it pays only in the most localised family, where the cheap residual score does as well as the co-state; the frozen D4-1 rule is met in **2 of 9 cells, both at hypothetical scoring prices**, so D4-1 stays closed at the real ledger and D4-2 is next | [note](docs/research-notes/2026-09-29-d4-0b-where-adaptivity-pays.md), [run](results/runs/d4_0b_adaptivity_20260929T174347Z/) |
| Analytic correctness on LQTree (N1) | ⏳ not started as a gate | — |
| Task success / planning | not measured | — |
| Hardware | stay on L4 (pilot peak 0.6 GiB) | [operator brief](docs/production/colab_l4_operator_brief.md) |

**Recommended next steps** are in [roadmap §2.1](docs/plans/roadmap.md): E1.1 opportunity audit first, then pilot v2 (B2); D4-2 on CPU in parallel; the rival benchmark (B1) is a different question and not on the critical path.

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
  02-diagnostics/           opportunity_audit.ipynb: run after the pilot, no retraining
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

Coding agents (Claude Code, Codex, Antigravity/`agy`, OpenCode, Gemini CLI, Cursor) share one harness:

- **[`AGENTS.md`](AGENTS.md):** the instructions every agent follows.
- **`.agents/skills/`:** reusable procedures.
- **[`harness/`](harness/README.md):** keeps the per-tool files in sync and defines "done" (`python harness/check.py`).
- **[`docs/plans/WORKLOG.md`](docs/plans/WORKLOG.md):** the handoff log agents read first and append to last.

## Contributing workflow

- **Branches and PRs:** open a branch per milestone ID from the roadmap (e.g. `n0.2-shortcut-tests`) and open a PR. CI runs pytest and checks that notebooks are valid and have no outputs.
- **Changelog:** new notebooks, protocol changes and imported runs each get a line in [`CHANGELOG.md`](CHANGELOG.md).

## License

MIT (see [`LICENSE`](LICENSE)). Datasets keep their own licences, e.g. DROID (CC BY 4.0); record them in each run's data manifest.
