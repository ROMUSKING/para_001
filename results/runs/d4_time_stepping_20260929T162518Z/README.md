# Run `d4_time_stepping_20260929T162518Z` (Track D, milestone D4-0)

- **Status:** correctness ✅ · rate-budget opportunity gate (validation) ✅ · **equal-compute payoff (validation) ❌** · D4-1 **stays closed** (`d4_1_opens: false` in `reports/acceptance_report.json`)
- **Findings:** [`docs/research-notes/2026-09-29-d4-0-adaptive-time-stepping.md`](../../../docs/research-notes/2026-09-29-d4-0-adaptive-time-stepping.md)
- **What it is:** goal-oriented adaptive time stepping for a linear ODE, the analytic reference domain D4 of [`docs/plans/cross-domain-plan.md`](../../../docs/plans/cross-domain-plan.md). The instances are generated in the repository with frozen seeds. **This is not real data**, and no learned model is involved.
- **Where it ran:** the Claude Code sandbox, on CPU, in about 224 s. The notebook `notebooks/04-domains/d4_adaptive_time_stepping.ipynb` (SHA-256 `fb2c37cb77be8ede8645613ac528326f356abfa5ed0b4dcef19d7e60af4ff311`) was executed unchanged, with `nbclient`, at commit `be9c6ed36a0e6ce7df331254a87b353e2987a9a1` with a clean working tree (`repo_dirty: false`). There is no Drive copy.
- **Companion run:** [`d4_time_stepping_20260929T162914Z`](../d4_time_stepping_20260929T162914Z/README.md), an exploratory variant that swaps the objective for the signed QoI error.

| Hash | Value |
|---|---|
| Config SHA-256 | `9a6ec9e927b6a8520d1b5f41da8ae2b06320ad5ba4c30976727f5513fb550d85` (re-derived from `config/run_config.json`; equals `config_hash` in `reports/acceptance_report.json`) |
| Instance family SHA-256 | `14a6abafdb6c8f04af37afb413f716ab4a9e7fd48074c6be8529e4ed95aa6717` of `config/instances.json` (the file's own SHA-256; the `sha256` field inside it is the hash of its instance list) |
| Domain spec SHA-256 | `3aecf7854e470f51…` prefix, full value in `config/run_config.json` (`domain_spec_sha256`) |
| Package commit | `be9c6ed36a0e6ce7df331254a87b353e2987a9a1` |

## Files

Every file below was copied **byte-for-byte** from the run directory the notebook wrote (`~/adjointrwm_runs/d4_time_stepping_20260929T162518Z/`). SHA-256 values were compared for the whole tree before and after the copy.

| Path | Origin |
|---|---|
| `config/run_config.json`, `config/instances.json`, `config/environment.json` | Notebook cell 1 |
| `tests/correctness.json`, `tests/error_representation.csv`, `tests/reference_convergence.csv` | Cell 2 (correctness gate) |
| `artifacts/validation_curves.parquet`, `artifacts/opportunity_gate.json`, `artifacts/opportunity_per_instance.csv` | Cell 3 (validation family, rate-budget opportunity gate) |
| `artifacts/validation_equal_compute.parquet`, `artifacts/equal_compute_gate.json` | Cell 3 (equal-compute check on validation; the D4-1 opening rule) |
| `artifacts/test_curves.parquet`, `artifacts/test_aurc_by_instance.csv`, `artifacts/test_summary.json` | Cell 4 (test family at equal refinement count, read once) |
| `artifacts/test_equal_compute.parquet`, `artifacts/test_equal_compute_summary.json` | Cell 4 (test family at equal total compute, read once) |
| `figures/mean_regret_by_budget.csv`, `figures/mean_regret_by_budget.png` | Cell 5 (raw table first, then the figure) |
| `reports/acceptance_report.json`, `reports/run_summary.md`, `COMPLETE` | Cell 5 |

**Not copied:** `reports/executed_notebook.ipynb`, the executed copy with outputs (SHA-256 `61b89755ae20e7eb553d9425884b68a58b633afb52d7283f1983d35fad9c6a7b`). It stayed in the ephemeral sandbox and is not preserved.

## History and what changed between executions

There was an earlier execution of an earlier version of this notebook, at commit `ab7c7bad6351fddaf86d8595832ea21cd65e8363` (run `d4_time_stepping_20260929T125504Z`). It was never committed to this repository. It differed from this run in two ways:

1. **Ledger.** It charged **1 CN step per refinement**. A refinement must actually re-solve every step from the refined interval onward, so this run charges `n + 1 − j` steps. Scoring costs were charged correctly in both.
2. **Equal-compute check.** The earlier version compared policies at equal refinement count only. A ledger audit found the under-charge, and a validation-only preview then showed plain uniform refinement ahead of every adaptive policy at equal compute. The equal-compute check, its frozen compute levels (1000, 2000, 4000, 8000 CN steps, from ledger arithmetic on validation) and the stricter D4-1 opening rule were added after that preview. **The test family had already been read at equal refinement count in the earlier execution**; it had not been read at equal compute.

Comparing the two executions:

- **Identical, byte for byte:** `tests/correctness.json`, `tests/error_representation.csv`, `tests/reference_convergence.csv`, `artifacts/opportunity_per_instance.csv`, `artifacts/opportunity_gate.json`, `artifacts/test_aurc_by_instance.csv`, `artifacts/test_summary.json`, `figures/mean_regret_by_budget.csv`, `config/instances.json`.
- **Same values:** the `objective` and `decision_compute` columns of `validation_curves.parquet` and `test_curves.parquet`. No policy reads costs, so the ledger cannot change what any policy chooses.
- **Changed:** the `action_compute` column of those two files, and `config/run_config.json` (the config gained the compute fields, so its hash differs).

## Reproducibility

The notebook was executed a second time from a separate clean checkout of commit `be9c6ed36a0e6ce7df331254a87b353e2987a9a1` (a `git worktree`, unchanged notebook, same environment, about 232 s). That run is not imported. Compared with this run:

- **byte-identical:** `tests/*` (all three files), `artifacts/test_aurc_by_instance.csv`, `artifacts/opportunity_per_instance.csv`, `artifacts/opportunity_gate.json`, `artifacts/equal_compute_gate.json`, `artifacts/test_summary.json`, `artifacts/test_equal_compute_summary.json`, `figures/mean_regret_by_budget.csv`, `config/instances.json`;
- **equal frames:** `validation_curves.parquet`, `test_curves.parquet`, `validation_equal_compute.parquet`, `test_equal_compute.parquet`;
- **equal:** the config hash, and `reports/acceptance_report.json` apart from the run ID.

The benchmark is deterministic on this software stack (Python 3.11.15, NumPy 2.4.6, pandas 3.0.6; see `config/environment.json`): re-executing the notebook at the commit above reproduces every number. This was not tested on other library versions.

## Checkpoints

None. Nothing is trained in D4-0.
