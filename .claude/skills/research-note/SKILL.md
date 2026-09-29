---
name: research-note
description: Write a dated, evidence-backed research note summarising what a run or analysis shows, including failures and ties. Use after importing or analysing a run, or when the user asks for findings, results or a write-up of an experiment.
---

# Research note

**File:** `docs/research-notes/YYYY-MM-DD-<slug>.md`. Model it on `docs/research-notes/2026-09-29-droid100-pilot-findings.md`.

## Required structure

1. **Header:** run ID(s), date, notebook path, hardware, config SHA-256, and artefact path.
2. **Status line:** which gates passed or failed, e.g. `data ✅ · dynamics ✅ · allocation ❌`.
3. **What the run is:** data, split, model size, candidates and objective, in a few bullets.
4. **Results tables:**
   - Every number comes from a file in `results/runs/<id>/`, or from a command given in the note.
   - Include baselines: persistence, random (expected), best-fixed, oracle and direct critic.
   - Include CIs from `adjointrwm.analysis.episode_bootstrap_ci`.
   - State the number of seeds and episodes.
5. **Interpretation:** separate observed from hypothesised. Label hypotheses as such and say which diagnostic would test each one.
6. **Design issues found:** code-level problems that make the result hard to interpret.
7. **What it does and does not support:** an explicit list. If the note contradicts earlier writing (a paper draft or README), say so and link to it.
8. **Next steps:** reference roadmap milestone IDs (`docs/plans/roadmap.md`).

## Rules

- Negative results and ties get the same prominence as positive ones.
- If a previous statement you or another agent made turns out to be wrong, correct it explicitly in the note. Don't let it quietly disappear.
- **Update affected files:**
  - `README.md` status table, if the headline changes;
  - the roadmap status block;
  - `CHANGELOG.md`;
  - a WORKLOG entry.
