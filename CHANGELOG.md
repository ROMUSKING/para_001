# Changelog

## 2026-09-30 (l): D1-0 (sensor streams on SMD) run and imported

- **`adjointrwm.domains.sensor`** (26 tests in `tests/test_sensor.py`): SMD machine split by index mod 4 (test machines are refused by the download and load code), a frozen Mahalanobis detector, windows, six non-learned policies, a privileged greedy oracle (better of forward selection and backward elimination), the best-known reference, normalised areas, the opportunity gate and a machine-clustered bootstrap. `scripts/fetch_smd.py` downloads tuning and validation machines only.
- **New plan `docs/plans/d1-0-plan.md`** (domain card, frozen choices, dated changes); **new notebook `notebooks/04-domains/d1_0_sensor_opportunity.ipynb`**; **run `results/runs/d1_0_sensor_opportunity_20260930T062900Z/` imported** (15 files, 1.8 MB, with README; data not committed) and **research note `docs/research-notes/2026-09-30-d1-0-sensor-opportunity.md`**. Correctness ✅; G1 ✅ robust (relative headroom 0.920 [0.886, 0.949]); non-learned deployable policies keep 0.42 to 0.49 of it; **the labelled detection F1 does not improve with sensing (0.259 hold, 0.192 full observation), so the proxy loss does not track the task and D1-1 is not designed on it.** The run reproduces byte-for-byte from a clean worktree (`reproduction` section of the run README). `scripts/d1_0_tables.py` prints every table in the note and recomputes the primary areas from the per-window parquet.
- **Docs:** roadmap, cross-domain plan, README and notebooks README updated; the candidates for D1, D2 and D3 are recorded as confirmed by Roman ("proceed as recommended", 2026-09-30); D1-0b (objective redesign) proposed and waiting for a decision.
- **Tests:** 204 (was 178).

## 2026-09-30 (k): D4-2 run imported; D4-1 stays closed

- **Run `results/runs/d4_2_flop_scoring_20260929T233556Z/` imported** (24 files, about 1.6 MB, with README). Research note `docs/research-notes/2026-09-30-d4-2-flop-priced-scoring.md`. Frozen rules: R1 holds in 3 of 7 cells, R3 in 1, none has both, so **no D4-1 candidate cell**; a non-learned cheap estimator does at least as well as the learned scorer in every cell. Validation only; the test family was never generated. A clean-checkout re-execution at the same commit gave byte-identical scorers, tuning and validation files (`reproduction.md` in the run folder).
- **`scripts/d4_2_tables.py`** prints every table in the note from the run files and re-applies R1 and R3 under each interpolation method (asserts the primary method equals the run's report).
- **Docs:** roadmap, cross-domain plan, README and notebooks README updated; D4-3 (varying goal, non-learned pair) proposed and waiting for Roman; plan §12 records the killed first launch.
- **Tests:** 178 (was 177).

## 2026-09-29 (j): D4-2 code, tests and notebook

- **`adjointrwm.domains.highdim`** (26 tests in `tests/test_highdim.py`): dense stable systems with a fixed goal, the seven one-factor cells, a FLOP ledger for scoring converted to CN steps, cheap solve-free estimators (forcing quadrature discrepancy; a third divided difference of the solution) with and without a tabulated continuous co-state, an amortised MLP scorer with NumPy training, Dörfler policies for each, `tune_theta`, and the frozen-rule helpers. The test family (seed 2002) cannot be generated.
- **Runner:** log-log interpolation of compute-to-target (`method='loglog'`), and `evaluate_work_precision(methods=...)`, which prices every target under staircase, semilog and log-log from the same traces.
- **New plan `docs/plans/d4-2-plan.md`** with its dated list of changes, including a disclosure that a smoke test printed validation numbers for four instances of two cells.
- **New notebook `notebooks/04-domains/d4_2_flop_priced_scoring.ipynb`. Not run yet.**
- **Tests:** 177 (was 151).

## 2026-09-29 (i): licence survey pass 2; `.gitattributes`

- **`.gitattributes`:** `results/runs/**` is marked `linguist-generated`, so GitHub collapses the run-artefact diffs by default (they were 33,561 of 43,388 added lines on the branch). No file changes; `instances.json` is not compacted (Roman's decision).
- **Licence survey pass 2** (network access widened): register grown to 59 rows (17 `adopt`, 12 `adopt_with_conditions`, 5 `avoid`, 25 `unverified`), 124 evidence URLs, 14 probes. New in `scripts/licence_survey.py`: licence fields from JSON answers and HTML catalogue markup, Hugging Face gating, commit and parameter counts, structure probes of Hugging Face datasets through the datasets server, and a licence census over the repositories behind SWE-bench Verified. `tests/test_licences.py` has 16 tests. See `docs/licences/survey-2026-09-29-pass2.md`.
- **Findings:** Gemma 4 is Apache-2.0 where Gemma 3 was custom; TriviaQA's and LongBench v1's owner statements conflict or are silent; JOB's IMDb data is non-commercial, which closes the query-planning route for D3; 490 of 500 SWE-bench Verified instances come from repositories with a permissive-family licence file.
- **Tests:** 151 (was 143).

## 2026-09-29 (h): licence survey, pass 1 (milestone DL)

- **Roman's decision:** licence survey first, then the rest.
- **New `docs/licences/`:** `register.csv` (53 sources across D1, D2, D3, the query-planning alternative and Tier 2; the licence as read, what it covers, a verification level, evidence URLs, gaps, a verdict), `evidence.jsonl` (per-URL status, size, SHA-256, retrieval time and licence lines; no texts), `probes.jsonl` (structure-only usability tests of data that is reachable), and `survey-2026-09-29.md`.
- **New `scripts/licence_survey.py`** (`snapshot`, `probe`; standard library only, so it also runs on Colab) and `tests/test_licences.py` (8 tests): a row that claims a read must point at fetched evidence, and `adopt` requires a permissive licence for the asset itself, not just for its code.
- **Result:** 4 `adopt`, 10 `adopt_with_conditions`, 3 `avoid`, 36 `unverified`. The sandbox blocks Hugging Face, UCI, Zenodo, PhysioNet and other hosts, so no model checkpoint card and no non-GitHub dataset page was read; pass 2 needs those hosts reachable. QuALITY's articles are under Project Gutenberg, OANC and CC BY 4.0 licences, but its annotations have no stated licence.
- **Tests:** 143 (was 135).

## 2026-09-29 (g): D4-0b design study, robustness check and recommended sequence

- **Batch (pass) allocation** in `adjointrwm.domains`: `apply_batch`/`batch_cost` on the domain (a pass of `k` refinements costs `n + k − j_min` CN steps), `BatchPolicy`, `run_batch_policy`, `uniform_pass_policy`, `marking_policy` (Dörfler marking on the residual, goal-local or co-state score), work-precision evaluation (`evaluate_work_precision`, `work_precision_summary`), scoring-price what-ifs, and three localisation families (`smooth`, `sharp`, `sharper`). 32 domain tests.
- **`interpolated_compute_to_target`**: removes the pass quantisation of compute-to-target (a post-hoc robustness check, applied to every policy).
- **Scripts:** `scripts/d4_0b_tables.py` prints every table in the D4-0b note from committed run files (and re-applies the frozen rule to the interpolated summary); `scripts/d4_0b_interpolation.py` writes the robustness diagnostics.
- **New notebook `notebooks/04-domains/d4_0b_where_adaptivity_pays.ipynb`** (validation-only; the test family is never generated).
- **Runs imported:** `results/runs/d4_0b_adaptivity_20260929T174347Z/` (the result; reproduced from a clean checkout) and `…T173428Z/` (first execution, superseded: its marking-fraction grid was truncated).
- **Result (validation only):** at the real scoring price pass-based marking does not rescue adaptivity on `smooth`, pays only in `sharper` where the residual score does at least as well as the co-state score, and the frozen rule is met in 2 of 9 cells, both `sharp` at hypothetical prices (×0.25, ×0). Every tuned marking fraction is at the top of its grid. See `docs/research-notes/2026-09-29-d4-0b-where-adaptivity-pays.md`.
- **Roadmap §2.1:** recommended sequence (E1.1 → B2; D4-2 → D4-1 on CPU in parallel; B1 off the critical path; DL and other domains deferred), stop-losses, and what would change it.
- **Tests:** 135 (was 123).

## 2026-09-29 (f): D4-0 run, ledger fix and equal-compute check; cross-domain decision recorded

- **Decision recorded (Roman):** D4 first; the other domains are subject to future research into permissively licensed content and testing. The plan gains a licence survey milestone (DL); D1–D3, ERP and any amendment of N6 wait for it.
- **Ledger fix (`adjointrwm.domains.linear_ode`).** A refinement is now charged the CN steps it forces the solver to redo (`n + 1 − j`), not 1. Found by auditing the ledger against the plan's matched-`C_total` rule, before any D4 run was committed.
- **Equal-compute analysis.** `run_policy(compute_budget=…)`, `evaluate_at_compute`, `objective_at_compute` and `compute_level_summary`, with tests. The D4-0 notebook compares deployable policies at fixed compute levels (1000–8000 CN steps). D4-1 now opens only if the co-state beats uniform refinement at equal compute on validation; this is stricter than the plan's rate-budget gate, and was added after a validation-only preview.
- **Runs imported:**
  - `results/runs/d4_time_stepping_20260929T162518Z/`: the main D4-0 run, at a clean commit, reproduced byte for byte from a separate clean checkout.
  - `…T162914Z/`: an exploratory variant with the signed QoI error as objective.
- **Result:** correctness ✅, rate-budget opportunity ✅, **equal-compute payoff ❌**. At equal refinement count, co-state weighting beats unweighted and goal-projected weighting on all 40 test instances. At equal total compute, plain uniform refinement beats every adaptive policy at every level. D4-1 stays closed. See `docs/research-notes/2026-09-29-d4-0-adaptive-time-stepping.md`.
- **Tests:** 123 (was 119).

## 2026-09-29 (e): cross-domain track and the D4 reference domain

- **Plan.** Added `docs/plans/cross-domain-plan.md` (Track D). It turns Roman's domain-portfolio brief into a programme reconciled with the governing plan:
  - a domain admission checklist mapped to `DomainSpec`;
  - portfolio A–P mapped to plan families, tiers and phases;
  - how the brief's `AllocationDomain` interface is implemented;
  - cards for D1–D4;
  - an experimental ladder from specialists through shared allocator and leave-one-domain-out to N6;
  - universal metrics using the plan's definitions;
  - governance for high-stakes and proprietary domains;
  - open questions.

  The brief ranks LLM context first; the plan proposes D4 first, consistent with the governing N1 → N2 → N6 order. This is flagged as a question for Roman.
- **Package `adjointrwm.domains`** (NumPy only):
  - `Cost` and `Candidate` with R/C/L/Q ledgers;
  - a `DomainSpec` subset;
  - the `AllocationDomain` interface;
  - a policy runner that enforces privilege (deployable policies never receive the instance);
  - generic one-step and exhaustive oracles;
  - metrics: regret, `AURC`, adaptive gain, fraction of oracle advantage, opportunity over budgets, `TransferMacro`/`TransferWorst`, and a best-known reference curve.
- **D4 reference domain** (`linear_ode.py`): goal-oriented adaptive time stepping for a linear ODE with Crank–Nicolson, an exact matrix-exponential reference, the discrete co-state recursion (plan §4.2) and a frozen-seed instance family.
  - Declared objective: the cancellation-free error bound `Σ|Λᵀτ|`.
  - The signed QoI error rewarded lucky cancellations, so it is a secondary metric.
- **New notebook `notebooks/04-domains/d4_adaptive_time_stepping.ipynb` (D4-0). Not run yet.** It runs on CPU and covers correctness, the opportunity gate on validation, and a test-family comparison of residual, goal-local and co-state weighting.
- **Tests.** 119 tests (16 new). New coverage:
  - the error representation is exact to rounding;
  - the reference matches a closed form;
  - the co-state matches finite differences;
  - CN is second order;
  - deployable policies never read the exact solution;
  - the exhaustive oracle bounds the greedy one;
  - ledgers and metrics.

## 2026-09-29 (d): rival-model benchmark track, pilot v2, tested training package

- **Plan.** Added `docs/plans/rival-benchmark-plan.md` (Track B). It covers:
  - the two rival questions, kept apart: rival world models (B1/B3) and rival allocators (B2 = E2.1/E2.2);
  - the rivals, with their constants checked against the official code and configs;
  - a fairness contract enforced in code;
  - endpoints, the decision rule, milestones B0–B5, allowed wording per outcome, risks and open questions.

  The roadmap gains a Track B table and progress notes on N0.1, N0.3 and E2.1.
- **New notebook `notebooks/03-benchmarks/rival_world_models_droid100.ipynb` (B1). Not run yet.**
  - Arms: AdjointRWM (pilot architecture) against DreamerV3-, TD-MPC2-, DINO-WM- and V-JEPA 2-AC-style re-implementations, plus persistence and ridge baselines.
  - Setup: DROID-100 with the pilot's split, prediction-path parameters matched within ±10 %, an equal validation-only LR search, and 5 paired seeds.
  - Scoring: two-level bootstrap comparisons and an action-shuffle control.
  - The run is resumable across Colab sessions.
- **New notebook `notebooks/01-production/AdjointRWM_Production_Pilot_v2.ipynb` (E2.1/E2.2). Not run yet.** v1 is unchanged. Changes from v1:
  - a hold option;
  - single-choice candidate training;
  - exact expected-random, `always_hold` and `always_c_k` baselines;
  - full `exact_gain[N, K+1]` in the traces;
  - co-state, critic and gate trained as separate jobs, with the critic's width matched to the co-state head's parameters;
  - a gate label from the regret sign, with its class weight taken from validation;
  - uncertainty-only and randomized-co-state allocators;
  - opportunity and critic-floor gates on validation;
  - the dynamics gate moved from test to validation;
  - 5 paired seeds;
  - FP32 exact labels inside BF16 training.
- **Package.** The notebooks now import tested code from `src/adjointrwm/`:
  - `data`: DROID parsing, feature cache, the pilot's split, windows and normalisation;
  - `eval`: method-blind metrics, persistence, ridge, paired bootstrap, classification;
  - `models`: the pilot model lifted verbatim, four rival arms, and parameter matching;
  - `training`: resumable runner with the operator brief's checkpoint contract;
  - `allocators`, `features`, `benchmark` (fairness contract) and `io`;
  - `analysis`: new policy summaries and the critic floor.
- **Tests.** 103 tests, up from 14. torch-dependent modules skip without torch. New coverage:
  - pilot parity: parameter count and stage-1 loss;
  - split and window parity with the pilot;
  - future-target invariance for every world-model arm and every deployable allocator;
  - exact co-state against float64 finite differences;
  - bit-exact resume;
  - checkpoint integrity;
  - the fairness contract.
- **CI** installs CPU-only PyTorch so the model tests run. `pyproject.toml` gains a `torch` extra.
- **Fix found by the resume test.** Creating a `DataLoader` iterator draws from the global torch RNG, which shifted dropout after a resume. The runner now gives the loader its own generator. v1 has no resume path, so no v1 result is affected.

## 2026-09-29 (c): cross-tool agent harness

- Added `AGENTS.md` (root, `notebooks/`, `results/`) as the single instruction source for Claude Code, Codex, Antigravity/`agy`, OpenCode, Gemini CLI and Cursor.
- Added five portable skills in `.agents/skills/`: `import-run`, `audit-run`, `claim-check`, `notebook-hygiene` and `research-note`.
- Added `harness/`:
  - `sync.py`: generates the Claude Code adapters and slash-command shims from `manifest.json`, including MCP fan-out.
  - `check.py`: the definition of done, also run in CI.
  - `hooks/protect_paths.py`: blocks edits to immutable run artefacts, dated audits and generated files.
  - `README.md`: design notes and a tool matrix checked against each tool's docs.
- Added `.claude/settings.json`, `opencode.json` and `.gemini/settings.json` with permission guardrails.
- Added `docs/plans/WORKLOG.md`, an append-only handoff log.
- CI now runs `harness/check.py` with immutability checks against the base ref.

## 2026-09-29 (b): documentation update and roadmap

- Imported these documents:
  - `docs/research-plan/adjoint_guided_comprehensive_research_plan.md`, the cross-domain N0–N9 protocol, now the governing plan
  - `docs/production/production_training_plan.md` (P0–P11)
  - `docs/production/colab_l4_operator_brief.md`
- Imported two outside audits into `docs/audits/`:
  - `Run_V2` (Ailerons)
  - the SWM "DROID subset" pilot (seeded-random surrogate)
- Moved the legacy notebook audit into `docs/audits/` and added `docs/audits/README.md`, an evidence register.
- Rewrote `docs/plans/roadmap.md`. It now has:
  - two tracks: research N0–N3 and embodied engineering E1–E3
  - the G-H2 decision gate
  - the production P-stages split into "start now" and "gated"
  - hardware policy, publication tracks, risks and open questions
- Rewrote `README.md` and added the `docs/README.md` index.
- Rewrote `.gitignore`. It now excludes weights, tensors, archives, Colab caches, profiler traces and credentials, and it re-includes small run artefacts under `results/runs/`.
- The paper draft upload was byte-identical to `papers/drafts/`, so nothing changed there.

## 2026-09-29: repository created

- Imported from Google Drive:
  - research plan v5
  - paper draft
  - production pilot notebook (source)
  - `para_0_0_1` and signal-filtering notebooks (archive)
  - run `droid100_adjoint_20260929T070629Z` artefacts
  - legacy phase logs
- Added `docs/DRIVE_INVENTORY.csv`, which maps 59 Drive files and folders to a status and repo path.
- Added research notes:
  - DROID-100 pilot findings, including a re-analysis of the allocation traces: the allocators collapsed onto candidates {0, 2}.
  - Audit of the earlier notebooks and phase logs.
- Added `papers/drafts/REVIEW.md`, a claim-by-claim evidence check of the paper draft.
- Added `docs/plans/roadmap.md`, milestones M1–M6.
- Added the `adjointrwm.analysis` package (trace summaries, episode-cluster bootstrap, opportunity audit) with tests and a CLI.
- Added `notebooks/02-diagnostics/opportunity_audit.ipynb`.
