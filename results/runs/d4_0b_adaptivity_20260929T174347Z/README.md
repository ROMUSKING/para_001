# Run `d4_0b_adaptivity_20260929T174347Z` (D4-0b design study, exploratory)

- **What it is:** an exploratory design study of the analytic D4 benchmark (goal-oriented adaptive time stepping). The question is where adaptive, co-state-guided refinement needs less **compute** than uniform refinement, once refinements are charged for re-solving downstream and scoring is charged per pass. Tuning family (seed 3003, 10 instances per family) and validation family (seed 1001, 20 per family), three families (`smooth`, `sharp`, `sharper`). **The test family (seed 2002) was never generated or read** (`test_family_read: false`). Not real data, no learned arm, so no H2 statement follows.
- **Status:** this is the run the study's conclusions rest on. It replaces `d4_0b_adaptivity_20260929T173428Z`, which is kept as a superseded design-history record (its marking-fraction grid was truncated; see that README). The write-up is `docs/research-notes/2026-09-29-d4-0b-where-adaptivity-pays.md`.
- **Where it ran:** the Claude Code sandbox, on CPU. The notebook `notebooks/04-domains/d4_0b_where_adaptivity_pays.ipynb` at commit `b9fd11dc934f7aed896b031e9a3eddc3d25f290d` (notebook SHA-256 `747f7f7c60829f2ef0edde1059fd869824ae6f4a3a4dbdc5cf2f90bfe2597783`, unchanged since) was executed with `nbclient` on a clean working tree (`repo_dirty: false`).
- **Frozen rule outcome (`reports/acceptance_report.json`):** 2 of 9 (family, scoring-price) cells met the candidate-regime rule: `sharp` at hypothetical scoring prices ×0.25 and ×0. Neither is the real ledger (×1). Every selected marking fraction `θ` is at the top of the grid (0.95), so the adaptive policies may still be understated (`theta_selected_at_grid_edge` is true for all nine pairs).

| Hash | Value |
|---|---|
| Config SHA-256 | `c82fbd69767c3b53c1f0ee011a0cf25ba0f84d051be304bb04505783c44dda9f` (`config_sha256` in `config/run_config.json`; re-derived with `adjointrwm.io.sha256_json` on its `config` field; equals `config_hash` in `reports/acceptance_report.json`) |
| Instance manifest SHA-256 | `7b507f50e4931de99e934e86c0f4d1cd2231fc4574db3e4e2de97a3c2bf5a9f9` (the file's own hash; identical to the first D4-0b run, since seeds and families are the same) |
| Package commit | `b9fd11dc934f7aed896b031e9a3eddc3d25f290d` |

## Files

Everything outside `diagnostics/` was copied **byte-for-byte** from `~/adjointrwm_runs/d4_0b_adaptivity_20260929T174347Z/` (whole-tree SHA-256 comparison before and after).

| Path | Origin |
|---|---|
| `config/run_config.json`, `config/instances.json`, `config/environment.json` | Notebook cell 1 |
| `tests/correctness.json`, `tests/batch_checks.csv`, `tests/reference_convergence.csv` | Cell 2 (batch-layer correctness: batch equals sequential, error representation on batch grids, reference converged) |
| `artifacts/theta_tuning.csv` | Cell 3 (tuning family; geometric-mean compute per marking fraction) |
| `artifacts/validation_work_precision.parquet`, `artifacts/validation_summary.json` | Cell 4 (validation family; compute to reach each target, censored at 2 finest grids) |
| `figures/phase_table.csv`, `figures/phase_diagram.png` | Cell 5 |
| `reports/acceptance_report.json`, `reports/run_summary.md`, `COMPLETE` | Cell 5 |
| `diagnostics/interpolated_summary.json`, `diagnostics/interpolated_work_precision.parquet` | **Produced here, after the results above were read**, by `python scripts/d4_0b_interpolation.py results/runs/d4_0b_adaptivity_20260929T174347Z` at commit `2ea8afc`. It re-runs the validation family with the run's own selected `θ` and prices compute-to-target by interpolation between passes for every policy. It asserts that its targets equal the stored ones. Post-hoc and exploratory; the frozen rule was **not** re-decided from it. |

**Not copied:** `reports/executed_notebook.ipynb`, which is not preserved.

## Reproducibility

The notebook was re-executed from a clean `git worktree` at the same commit (`b9fd11d`) as run `d4_0b_adaptivity_20260929T175553Z` (not imported). Comparing files with `sha256sum`:

- **Byte-identical:** `config/instances.json`, `artifacts/theta_tuning.csv`, `artifacts/validation_summary.json`, `artifacts/validation_work_precision.parquet`, `tests/correctness.json`, `tests/batch_checks.csv`, `tests/reference_convergence.csv`, `figures/phase_table.csv`.
- **Differ only in the run identifier:** `config/run_config.json` and `reports/acceptance_report.json` (their only differing key is `run_id`), `reports/run_summary.md` (identical after masking the run id), and `figures/phase_diagram.png` (its title contains the run id; pixels were not compared).

## Checkpoints

None.
