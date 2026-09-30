# Run `d4_time_stepping_20260929T162914Z` (Track D, D4-0 sensitivity variant, **exploratory**)

- **What it is:** the D4-0 notebook executed with one change, `D4Config(objective_kind='abs_error')`. The objective is the signed quantity-of-interest error `|cᵀ(y(T) − y_N)|` instead of the declared cancellation-free bound `Σ|Λᵀτ|`. Everything else, including the instance family, is identical to the main run [`d4_time_stepping_20260929T162518Z`](../d4_time_stepping_20260929T162518Z/README.md).
- **Why it exists:** a robustness check on the choice of objective. The declared objective is the bound; this run is **not a substitute** for it and no gate is decided from it. It was designed after the main run's results were known.
- **Status:** correctness ✅ · rate-budget opportunity (validation) ✅ · equal-compute payoff (validation) ❌ · `d4_1_opens: false`.
- **Findings:** [`docs/research-notes/2026-09-29-d4-0-adaptive-time-stepping.md`](../../../docs/research-notes/2026-09-29-d4-0-adaptive-time-stepping.md), the sensitivity section.
- **Where it ran:** the Claude Code sandbox, on CPU, in about 123 s. The committed notebook (SHA-256 `fb2c37cb77be8ede8645613ac528326f356abfa5ed0b4dcef19d7e60af4ff311`) was executed with `nbclient` at commit `be9c6ed36a0e6ce7df331254a87b353e2987a9a1`, clean working tree, **with one in-memory source substitution**: the single occurrence of `CFG = D4Config()` was replaced by `CFG = D4Config(objective_kind='abs_error')  # exploratory sensitivity run: signed-QoI-error objective`. The patched notebook is not committed; `config/run_config.json` records the resulting config.

| Hash | Value |
|---|---|
| Config SHA-256 | `90397e2c5d5a70ab9b138c43e6d6c53033f5ce5592ddb073efb368c8bb807e7a` (re-derived from `config/run_config.json`; equals `config_hash` in `reports/acceptance_report.json`) |
| Instance family | Same `config/instances.json` as the main run (SHA-256 `14a6abafdb6c8f04af37afb413f716ab4a9e7fd48074c6be8529e4ed95aa6717`) |
| Package commit | `be9c6ed36a0e6ce7df331254a87b353e2987a9a1` |

## Files

Copied **byte-for-byte** from `~/adjointrwm_runs/d4_time_stepping_20260929T162914Z/`, with the same layout and origins as the main run's README. `reports/executed_notebook.ipynb` (SHA-256 `88d5afddf9d3f7e00b1b9532264f5b468716a023d1caa162dfa809e64ea0c87c`) was not copied and is not preserved.

An earlier version of this variant, run at commit `ab7c7ba` with the under-charged ledger, is not imported. Its refinement-count results were the same as this run's (the ledger does not affect them).

## Checkpoints

None.
