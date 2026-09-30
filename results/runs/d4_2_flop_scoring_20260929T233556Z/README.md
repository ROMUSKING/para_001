# Run `d4_2_flop_scoring_20260929T233556Z` (D4-2 design study, exploratory)

- **What it is:** an exploratory, validation-only design study of the analytic D4 benchmark (goal-oriented adaptive time stepping on `y' = Ay + b(t)`) with scoring priced in floating-point operations and converted to Crank–Nicolson (CN) step units. Seven cells (dimension `m` = 4, 32, 64; pulse width, amplitude and finest depth varied one at a time from `base`), each with a scorer-training family (seed 4004, 30 instances), a tuning family (seed 3003, 10) and a validation family (seed 1001, 20). **The test family (seed 2002) was never generated** (`test_family_read: false` in `reports/acceptance_report.json` and `config/run_config.json`). The data are generated from frozen seeds; **this is not real data.**
- **Status:** this is the run the D4-2 conclusions rest on. The write-up is `docs/research-notes/2026-09-30-d4-2-flop-priced-scoring.md`; the design is `docs/plans/d4-2-plan.md` (dated changes in §12).
- **Where it ran:** the Claude Code sandbox, on CPU (`config/environment.json`: Python 3.11.15, NumPy 2.4.6, pandas 3.0.6). The notebook `notebooks/04-domains/d4_2_flop_priced_scoring.ipynb` (SHA-256 `8f29012f747d56ccb84aa03cd998bd19aaf3adad9b6b224ad1fc0b0cd66a016e`) was executed with `nbclient`, detached from the tool call, at commit `b139a33a17ae54376e1f4fe8a701eab80ad2da55` on a clean working tree (`repo_dirty: false`).
- **Frozen rule outcome (`reports/acceptance_report.json`):** no cell is a candidate for D4-1 (`candidate_cells_for_d4_1` is empty): R1 (the amortised scorer beats uniform refinement) holds in `base`, `depth7` and `depth11`; R3 (the co-state weight helps the cheap estimator) holds in `m64`. Every table in the note is printed from these files by `python scripts/d4_2_tables.py results/runs/d4_2_flop_scoring_20260929T233556Z`, which also re-applies R1 and R3 for each interpolation method and asserts that the primary method equals the run's own report.

| Hash | Value |
|---|---|
| Config SHA-256 | `2ab9e44685312fbf39b43ebe96a254229a1074dcd17623e47adba156283d88b4` (`config_sha256` in `config/run_config.json`; equals `config_hash` in `reports/acceptance_report.json` and `config_sha256` in `reports/frozen_before_validation.json`) |
| Frozen-before-validation SHA-256 | `1a4e9c23e9e7e75f04b5b5aa30c313a7bf1fc9f4894d9bc2045c9198fe497607` (`sha256` in `reports/frozen_before_validation.json`, written 2026-09-29T23:56:23 UTC, before the validation family was generated; equals `frozen_sha256` in the acceptance report) |
| Instance manifest SHA-256 | `858caf764942956be999cd268bb422df23886b9b4bb28e768f13d15bfb408af5` (the hash of `config/instances_manifest.json`; full instance and system parameters are in `config/instances.parquet` and `config/systems.parquet`) |
| Domain-spec SHA-256 | `3aecf7854e470f51d654050c208a2534300755ad5d68d49613436c99b151f7d5` (`domain_spec_sha256` in `config/run_config.json`) |
| Package commit | `b139a33a17ae54376e1f4fe8a701eab80ad2da55` |

## Files

Everything was copied **byte-for-byte** from `~/adjointrwm_runs/d4_2_flop_scoring_20260929T233556Z/` (per-file SHA-256 comparison after the copy).

| Path | Origin |
|---|---|
| `config/run_config.json`, `config/instances_manifest.json`, `config/instances.parquet`, `config/systems.parquet`, `config/environment.json` | Notebook cell 1 (frozen configuration, cells, instances) |
| `tests/correctness.json`, `tests/ledger_and_batch_checks.csv` | Cell 2 (FLOP formula against operations counted from the code path, tabulated co-state against the discrete co-state, forcing-indicator and batch checks, reference convergence; `wall_clock_sanity_seconds` records timings, which are not a systems claim) |
| `artifacts/scorers/*.json` | Cell 3 (trained scorer weights per cell; small JSON, not checkpoints) |
| `artifacts/theta_tuning.csv`, `artifacts/hidden_size_selection.csv` | Cell 3 (tuning family: marking fraction per arm; network size per cell) |
| `reports/frozen_before_validation.json` | Cell 3 (frozen choices, scorer hashes and rules, written before validation) |
| `artifacts/validation_work_precision.parquet`, `artifacts/validation_summary.json` | Cell 4 (validation family; compute to reach each target under staircase, semilog and log-log pricing) |
| `figures/compute_ratios.png`, `figures/phase_table.csv`, `reports/acceptance_report.json`, `reports/run_summary.md`, `COMPLETE` | Cell 5 (frozen rules, one-off costs, price ratios, table, figure, report) |

**Not copied:** `reports/executed_notebook.ipynb` (SHA-256 `c27f96a1eb202e94f609df38bbb5e78cda84814254de18b003c95b114ac56f17` in the run directory; not committed, as for earlier runs). **No `diagnostics/` folder:** no post-hoc analysis was run on this run.

## Related runs not imported

- `d4_2_flop_scoring_20260929T224306Z` (renamed `ABORTED_…_killed_at_600s` in the local runs folder): the first launch, started inside a tool call with a 600 s limit and killed during scorer training (four of seven scorers written). It produced no tuning or validation output, so no validation number was seen from it. It was relaunched with unchanged code and configuration. See plan §12.
- A smoke test of the notebook with a tiny configuration was run before this run and printed validation medians for four instances of two cells; no design choice was changed because of them (plan §12).

## Reproducibility

**Not yet established at the time of the first commit of this folder.** A re-execution of the notebook from a clean `git worktree` at the same commit (run `d4_2_flop_scoring_20260930T000230Z`, not imported) was started before this commit and had not finished. Its file-by-file SHA-256 comparison with this run will be added to this folder as `reproduction.md` (a new file, since files in a committed run are not edited). Until that file exists, treat determinism of the training stage (NumPy, BLAS threading) as unverified. Timing fields in `tests/correctness.json` are expected to differ.

## Checkpoints

None. The scorer files are small JSON weight dumps (the largest, `m64.json`, is about 16 KB).
