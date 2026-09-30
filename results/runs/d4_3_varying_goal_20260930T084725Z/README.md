# Run `d4_3_varying_goal_20260930T084725Z` (D4-3 design study, exploratory)

- **What it is:** an exploratory, validation-only design study of the analytic D4 benchmark (goal-oriented adaptive time stepping on `y' = Ay + b(t)`), the follow-up to D4-2 that gives **every instance its own random unit goal**. It has the same seven cells as D4-2 (dimension `m` = 4, 32, 64; pulse width, amplitude and finest depth varied one at a time from `base`; identical systems, pulses and initial states), a tuning family (seed 3003, 10 instances) and a validation family (seed 1001, 20 instances); the goal of every instance comes from its own stream (`default_rng(family seed + 10000 + m)`). **The test family (seed 2002) was never generated** (`test_family_read: false` in `reports/acceptance_report.json` and `config/run_config.json`). The data are generated from frozen seeds; **this is not real data.** No learned critic is involved, so **no H2 statement follows.**
- **Status:** this is the run the D4-3 conclusions rest on. The write-up is `docs/research-notes/2026-09-30-d4-3-varying-goal.md`; the design is `docs/plans/d4-3-plan.md` (§9: no change after the plan was written).
- **Where it ran:** the Claude Code sandbox, on CPU (`config/environment.json`: Python 3.11.15, NumPy 2.4.6, pandas 3.0.6; 726 s). The notebook `notebooks/04-domains/d4_3_varying_goal.ipynb` (SHA-256 `58bbc65f9bc9a9163d76bd4015c2059814a511db6fcb2c54973192231a45d60f`) was executed with `nbclient`, detached from the tool call, at commit `35c2daa98a98a5fe0d2d44c73bc43ad8f3aba45d` on a clean working tree (`repo_dirty: false`).
- **Frozen rule outcome (`reports/acceptance_report.json`):** two cells are D4-1 candidates (`candidate_cells_for_d4_1` = `m4`, `m64`): R3v (the tabulated co-state weight helps the goal-blind `cheap` estimator) holds in `m4` and `m64`; R4v (the goal-aware cheap arm beats uniform refinement) holds in `base`, `m4`, `m64`, `depth7` and `depth11`. With the table's setup charged per instance the goal-aware arm does not beat `cheap` in any cell. Every table in the note is printed from these files by `python scripts/d4_3_tables.py results/runs/d4_3_varying_goal_20260930T084725Z`.

| Hash | Value |
|---|---|
| Config SHA-256 | `cfe1545d2bdf65f16326dc9bcebad9458828b260724fd1283c41591e38a75457` (`config_sha256` in `config/run_config.json`; equals `config_hash` in `reports/acceptance_report.json` and `config_sha256` in `reports/frozen_before_validation.json`) |
| Frozen-before-validation SHA-256 | `6a66d322ba40964a60b48115566203bbdd93d510fa1e3ce74c377a4f4b2af2fa` (`sha256` in `reports/frozen_before_validation.json`, written 2026-09-30T08:53:33.3 UTC, before the validation family was generated; equals `frozen_sha256` in the acceptance report) |
| Tuning instance manifest SHA-256 | `6c6bbfa2b2a9b288a3a494eae36a9572c32a7b29d0f71815038058b44e403b71` (`sha256` in `config/instances_manifest_tuning.json`; full parameters in `config/instances_tuning.parquet`) |
| Validation instance manifest SHA-256 | `4555658bdd1746a380c0ed512636690466e7496359d591b637f832fcbe586416` (`sha256` in `config/instances_manifest_validation.json`; full parameters in `config/instances_validation.parquet`) |
| Package commit | `35c2daa98a98a5fe0d2d44c73bc43ad8f3aba45d` |

## Files

Everything was copied **byte-for-byte** from `~/adjointrwm_runs/d4_3_varying_goal_20260930T084725Z/` (per-file SHA-256 comparison after the copy).

| Path | Origin |
|---|---|
| `config/run_config.json`, `config/environment.json`, `config/instances_manifest_tuning.json`, `config/instances_tuning.parquet` | Notebook cell 1 (frozen configuration and the tuning family) |
| `tests/correctness.json`, `tests/ledger_and_goal_checks.csv` | Cell 2 (FLOP formulas against the operations counted from the code path; the tabulated co-state is linear in the goal and close to the discrete one on a fine mesh; goals are unit and distinct; instances equal D4-2's except goal and name; the test family is refused; the setup-charged rows equal `cheap_adjoint` plus the setup) |
| `artifacts/theta_tuning.csv`, `reports/frozen_before_validation.json` | Cell 3 (`θ` per (cell, arm) on the tuning family; frozen and hashed) |
| `config/instances_manifest_validation.json`, `config/instances_validation.parquet`, `artifacts/validation_work_precision.parquet`, `artifacts/validation_summary.json` | Cell 4 (validation family generated only after the freeze; compute to each target for every arm, three interpolation methods) |
| `figures/phase_table.csv`, `figures/compute_ratios.png`, `reports/acceptance_report.json`, `reports/run_summary.md`, `COMPLETE` | Cell 5 (frozen rules, price ratios, table, figure and report) |

**Not copied:** `reports/executed_notebook.ipynb` (SHA-256 `f19a5ccbe00e5c0ade67d5e66c717d18387ab4859347cc8b79b781f6491c141b` in the run directory; not committed, as for earlier runs). **No `diagnostics/` folder:** no post-hoc analysis was run on this run.

## Related runs not imported

- A smoke test of this notebook with a tiny configuration ran before the real run. It was fed the **train** family in place of validation, so no validation instance was generated; its numbers are not evidence and led to no change.
- `d4_3_varying_goal_20260930T114142Z`: the reproduction (below).

## Reproducibility

The notebook was re-executed from a clean `git worktree` at the same commit (`35c2daa`), on the same machine and Python environment, as run `d4_3_varying_goal_20260930T114142Z` (not imported; 675 s). Comparing files with SHA-256:

- **Byte-identical (11 of 18 files):** `artifacts/theta_tuning.csv`, `artifacts/validation_summary.json`, `artifacts/validation_work_precision.parquet`, `config/environment.json`, `config/instances_manifest_tuning.json`, `config/instances_manifest_validation.json`, `config/instances_tuning.parquet`, `config/instances_validation.parquet`, `figures/phase_table.csv`, `tests/correctness.json`, `tests/ledger_and_goal_checks.csv`.
- **Differ only in a run id or a timestamp** (compared key by key): `config/run_config.json` (only `run_id`); `reports/acceptance_report.json` (only `run_id`); `reports/frozen_before_validation.json` (only `written_utc`; its `sha256` is identical, so the freeze content is identical); `reports/run_summary.md` (identical after masking the run id); `COMPLETE` (a timestamp); `figures/compute_ratios.png` (573 pixels differ, all within rows 15 to 29 and columns 409 to 479 of a 647 × 1796 image, where the title with the run id sits); `reports/executed_notebook.ipynb` (not committed).
- So every number in the run reproduces exactly on the same machine. One repetition; no comparison across BLAS builds, CPU types or thread counts was made.

## Checkpoints

None (no learned model).
