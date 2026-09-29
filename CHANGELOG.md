# Changelog

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
