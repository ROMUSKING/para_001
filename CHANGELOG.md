# Changelog

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
