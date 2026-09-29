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
