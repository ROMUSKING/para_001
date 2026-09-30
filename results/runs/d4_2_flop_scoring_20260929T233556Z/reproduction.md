# Reproduction of `d4_2_flop_scoring_20260929T233556Z`

Added after the first commit of this folder (which is why it is a separate file; see the README, Reproducibility).

**What was run:** the same notebook (`notebooks/04-domains/d4_2_flop_priced_scoring.ipynb`, SHA-256 `8f29012f747d56ccb84aa03cd998bd19aaf3adad9b6b224ad1fc0b0cd66a016e`) executed with `nbclient` from a clean `git worktree` at the same package commit `b139a33a17ae54376e1f4fe8a701eab80ad2da55`, on the same machine and Python environment (4 CPUs), as run `d4_2_flop_scoring_20260930T000230Z` (not imported; `repo_dirty: false`). The run id's timestamp is its start (00:02:30 UTC); its `COMPLETE` file records the finish (00:28:38 UTC), about 26 minutes. The imported run started 23:35:56 UTC and finished at 00:01:28 UTC (`COMPLETE`), about 26 minutes.

**Comparison** (SHA-256 of every file; every file of the imported run exists in the reproduction):

- **Byte-identical (17 files):** all seven `artifacts/scorers/*.json`, `artifacts/hidden_size_selection.csv`, `artifacts/theta_tuning.csv`, `artifacts/validation_summary.json`, `artifacts/validation_work_precision.parquet`, `config/environment.json`, `config/instances.parquet`, `config/instances_manifest.json`, `config/systems.parquet`, `figures/phase_table.csv` and `tests/ledger_and_batch_checks.csv`. The trained scorer weights, the chosen `θ` and network sizes and every validation number are therefore the same on the two executions.
- **Differ only in the run identifier or a timestamp or a timing** (compared key by key for the JSON files; the two `nan` retained-headroom values that a naive comparison flags are equal `NaN`s):
  - `config/run_config.json`: only `run_id`.
  - `reports/acceptance_report.json`: only `run_id` and the two `wall_clock_sanity_seconds` timings.
  - `reports/frozen_before_validation.json`: only `written_utc` (its `sha256` field is identical, so the frozen choices and rules are the same).
  - `tests/correctness.json`: only the two `wall_clock_sanity_seconds` timings.
  - `reports/run_summary.md`: identical after replacing the run id.
  - `COMPLETE`: the completion timestamp.
  - `figures/compute_ratios.png`: its title contains the run id; pixels were not compared.
- **Only in the reproduction:** `reports/executed_notebook.ipynb`, which is not committed for either run.

**Scope of this check:** one repetition, same machine, same thread count. Identical scorer bytes show that the NumPy training is deterministic here; they do not show that it is bit-identical across BLAS builds, CPU types or thread counts, and no such comparison was made.
