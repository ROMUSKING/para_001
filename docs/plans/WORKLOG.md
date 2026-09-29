# Work log

Append-only handoff log shared by every agent and human working in this repo. Newest entries go **at the top**. Read the latest entries before starting; add one before ending a session.

Each entry records:

- what changed;
- what was verified, and how (commands and results);
- what's open or blocked;
- what the next agent should do first.

Reference roadmap IDs (`docs/plans/roadmap.md`). Keep each entry under about 15 lines.

```
## YYYY-MM-DD HH:MM (tz) · <agent/tool> · <roadmap IDs>
- Changed:
- Verified:
- Open:
- Next:
```

---

## 2026-09-29 13:25 (BST) · Claude Code (web) · B0, E2.1, E2.2, N0.1, N0.3

- **Changed:** added Track B, the benchmark against rival models. It has four parts:
  - the plan, `docs/plans/rival-benchmark-plan.md`;
  - two notebooks: `03-benchmarks/rival_world_models_droid100.ipynb` (B1) and `01-production/AdjointRWM_Production_Pilot_v2.ipynb` (B2 = E2.1/E2.2);
  - the tested modules they import: `data`, `eval`, `models`, `training`, `allocators`, `features`, `benchmark` and `io`;
  - updates to the roadmap, READMEs, CI (CPU torch) and `pyproject`.
- **Verified:**
  - `pytest`: 103 passed (56 passed and 3 modules skipped without torch). `python harness/check.py` passes.
  - The lifted pilot model has 27,360,798 parameters, matching `model_manifest.json`. The stage-1 loss equals a verbatim copy of the pilot's.
  - The exact co-state matches float64 finite differences.
  - Pause-and-resume gives bit-identical weights. This test found a `DataLoader` RNG drift, now fixed.
  - Both notebooks ran end to end on CPU with stubbed Colab/TFDS and random fixture episodes (code paths only, not evidence), including a run resumed over 5 sessions.
  - Rival constants come from the official GitHub configs. arXiv was blocked from the sandbox.
- **Open:**
  - Nothing has run on real data or a GPU.
  - Until this branch is merged, set `REPO_REF` in both notebooks to the branch name.
  - The split-parity check needs the pilot's `data_manifest.json` on Drive; it is listed in `DRIVE_INVENTORY.csv`.
  - Roadmap open questions 5–7 (endpoint, TD-MPC2 input, B4) need Roman's decision.
  - The CUDA resume check is still to do.
- **Next:**
  1. E1.1: the opportunity audit on the v1 checkpoint. It is cheap and informs the B2 reading.
  2. B2: run pilot v2.
  3. B1: run the rival benchmark on L4.
  4. Import each run with `/import-run`, then write the note with `/research-note`.

## 2026-09-29 11:30 (BST) · Claude (claude.ai agent) · harness

- **Changed:** added the cross-tool agent harness:
  - `AGENTS.md` (root plus `notebooks/` and `results/`), the single source of truth;
  - `.agents/skills/` with 5 portable skills;
  - `harness/sync.py`, which generates `CLAUDE.md` imports, `.claude/skills`, `.agents/workflows` and `.opencode/commands`;
  - `harness/check.py`, the definition of done;
  - the `protect_paths` hook, `.claude/settings.json`, `opencode.json` and `.gemini/settings.json`;
  - a CI job.
- **Verified:**
  - `python harness/check.py --base origin/main` passes 7/7, with 14 tests.
  - The hook blocks edits to existing run artefacts, dated audits and generated files, and allows new files.
  - Negative tests: hand-edits, stale skills and a modified run README are all caught.
  - Per-tool paths were checked against the docs; see the matrix in `harness/README.md`. Mid-build, `.agent/*` and `GEMINI.md` copies were removed as redundant, and the OpenCode rule order was fixed (last match wins).
- **Open:** Codex project-level `.codex/config.toml` is unverified (only matters once MCP servers exist). The MCP server list is empty.
- **Next:** roadmap E1.1 (run the opportunity audit notebook in Colab), then N0.1 (move the pilot code into `src/`).

## 2026-09-29 10:44 (BST) · Claude (claude.ai agent) · docs

- **Changed:** imported the comprehensive plan, the production docs and two audits; rewrote the roadmap, README and `.gitignore`.
- **Verified:** links resolve and 8/8 tests pass.
- **Next:** E1.1 and E1.2 (import the second pilot run `droid100_adjoint_20260929T090015Z`).
