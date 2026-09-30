# Run `d4_0b_adaptivity_20260929T173428Z` (D4-0b design study, **first execution, superseded**)

- **Status: superseded.** Kept as a record of the design history, not as the study's result. The result is the later run, written up in `docs/research-notes/2026-09-29-d4-0b-where-adaptivity-pays.md` (a plain path, not a link, because that note is written after this README is committed).
- **Why superseded:** its tuning table (`artifacts/theta_tuning.csv`) selected **`θ = 0.7`, the top of the grid `{0.3, 0.5, 0.7}`, for all nine (family, policy) pairs**, with cost still falling as `θ` rose. The adaptive policies were therefore tuned against a truncated grid and may be understated. The grid was extended to 0.85 and 0.95 in a later commit and nothing else was changed; the decision rule is identical. **The validation results in this run had already been printed and seen before that change**, so the extension is not blind to them, although the reason for it (a boundary optimum) is visible in the tuning table alone.
- **What it is:** an exploratory design study of the analytic D4 benchmark (goal-oriented adaptive time stepping): where does adaptive, co-state-guided refinement need less compute than uniform refinement? Tuning family (seed 3003, 10 instances per family) and validation family (seed 1001, 20 per family). **The test family (seed 2002) was never generated or read** (`test_family_read: false`). Not real data, no learned arm.
- **Where it ran:** the Claude Code sandbox, on CPU, in about 474 s. The notebook `notebooks/04-domains/d4_0b_where_adaptivity_pays.ipynb` as of commit `10c0557a3e05d65e94ce7007598aa7f06d1b980c` (notebook SHA-256 `f6f251714f542517830442bcef71a8527a54dd16926eeaf64f5360a77c2ac34c`; that version is no longer the file at `HEAD`) was executed unchanged with `nbclient` on a clean working tree (`repo_dirty: false`).
- **Frozen rule outcome in this run:** 1 of 9 (family, scoring-price) cells met the candidate-regime rule: `sharp` at scoring price ×0, a hypothetical re-pricing (`reports/acceptance_report.json`).

| Hash | Value |
|---|---|
| Config SHA-256 | `bedf87b69b74818403bba078040a8bc91cf1dae8479972e75449113114eb6e0d` (re-derived from `config/run_config.json`; equals `config_hash` in `reports/acceptance_report.json`) |
| Instance manifest SHA-256 | `7b507f50e4931de99e934e86c0f4d1cd2231fc4574db3e4e2de97a3c2bf5a9f9` (the file's own hash) |
| Package commit | `10c0557a3e05d65e94ce7007598aa7f06d1b980c` |

## Files

Copied **byte-for-byte** from `~/adjointrwm_runs/d4_0b_adaptivity_20260929T173428Z/` (whole-tree SHA-256 comparison before and after).

| Path | Origin |
|---|---|
| `config/run_config.json`, `config/instances.json`, `config/environment.json` | Notebook cell 1 |
| `tests/correctness.json`, `tests/batch_checks.csv`, `tests/reference_convergence.csv` | Cell 2 (batch layer correctness) |
| `artifacts/theta_tuning.csv` | Cell 3 (tuning family; shows the boundary optimum) |
| `artifacts/validation_work_precision.parquet`, `artifacts/validation_summary.json` | Cell 4 (validation family) |
| `figures/phase_table.csv`, `figures/phase_diagram.png` | Cell 5 |
| `reports/acceptance_report.json`, `reports/run_summary.md`, `COMPLETE` | Cell 5 |

**Not copied:** `reports/executed_notebook.ipynb` (SHA-256 `5631468f5bdb9c1dc6d16fb2db8a41713745147df4d62aedb76455944a10c812`), which is not preserved.

## Checkpoints

None.
