# Run `d1_0b_forecast_sensing_20260930T085746Z` (D1-0b Rung-0 forecasting objective, exploratory)

- **What it is:** the redesign of the D1 sensor-stream opportunity gate after D1-0's detector-fidelity objective was found to be a proxy that does not move the native endpoint. A 60-minute window of the Server Machine Dataset (SMD) starts with a snapshot of all 38 channels; an allocator opens `k` channels (observed every minute) and the rest are held at the snapshot. The loss is now the **native endpoint**: the squared error of a frozen per-machine ridge forecaster's 5-minute-ahead prediction against the true future, reported as the excess over full observation divided by the machine's full-observation loss. **Seven validation machines** gave the gates; seven tuning machines gave the endpoint-moves statistic `r`, the best fixed policy per setting and the G0 pre-check; **the 14 test machines were never downloaded** (`test_machines_downloaded: false`). No learned critic and no co-state: **no H2 statement.** The data are real and **not committed** (MIT licence, register id `smd`).
- **Status:** this is the run the D1-0b conclusions rest on. The write-up is `docs/research-notes/2026-09-30-d1-0b-forecast-sensing.md`; the design and its dated changes are `docs/plans/d1-0b-plan.md`.
- **Where it ran:** the Claude Code sandbox, on CPU (`config/environment.json`: Python 3.11.15, NumPy 2.4.6, pandas 3.0.6). The notebook `notebooks/04-domains/d1_0b_forecast_sensing.ipynb` (SHA-256 `38277ad848a9323e020242e822ce62c85191396416e4af30b52f6f5383cc2849`) was executed with `nbclient`, detached from the tool call, at commit `2b3bb994530527667b52a02a402b1ca6d4e6b157` on a clean working tree (`repo_dirty: false`).
- **Outcome (`reports/acceptance_report.json`):** correctness passed; G0 passes on the tuning machines (`r` = 0.522) and by the frozen point rule on validation but not robustly (0.298, 95 % interval [0.023, 0.743]; one validation machine is negative); G1 passes by the frozen rule (0.875 [0.295, 1.347], an unstable ratio); neither deployable dynamic policy keeps any of the headroom (G2). See the note for what that does and does not mean.

| Hash | Value |
|---|---|
| Config SHA-256 | `d3fd72720addd367e9dd9b2ac717211d45d92b9c3634be0dfcabc326cc50d82f` (`config_sha256` in `config/run_config.json`; equals `config_hash` in `reports/acceptance_report.json` and `config_sha256` in `reports/frozen_before_validation.json`) |
| Frozen-before-validation SHA-256 | `23d6b37d3d579be85079662048f022bb29bcea56d7f07c33985dbe38757e0696` (`sha256` in `reports/frozen_before_validation.json`, written 2026-09-30T08:58:07.8 UTC; it covers the config hash, the tuning-manifest hash, the forecaster constants, the G0 result on tuning, the best-fixed choice per setting and the rules) |
| Tuning data manifest SHA-256 | `8692ee22edaaf86278bb8acb6d59b9a900918e5fabf0b652367f54a9a2569a46` (`sha256` in `config/data_manifest_tuning.json`: 21 files) |
| Validation data manifest SHA-256 | `077ee8c3b2eabb9a7c3931b22cae4ad36c1e6e039f7b2db54a3b4a9595e5e69e` (`sha256` in `config/data_manifest_validation.json`: 21 files) |
| Package commit | `2b3bb994530527667b52a02a402b1ca6d4e6b157` |

## Files

Everything was copied **byte-for-byte** from `~/adjointrwm_runs/d1_0b_forecast_sensing_20260930T085746Z/` (per-file SHA-256 comparison after the copy).

| Path | Origin |
|---|---|
| `config/run_config.json`, `config/environment.json`, `config/data_manifest_tuning.json` | Notebook cell 1 (frozen configuration; tuning machines pinned) |
| `artifacts/tuning_summary.json` | Cell 2 (stage A: `r` and the fixed policies' areas per setting on the tuning machines; the best fixed policy per setting) |
| `tests/correctness.json` | Cell 3 (correctness checks on real tuning windows) |
| `reports/frozen_before_validation.json`, `config/data_manifest_validation.json` | Cell 4 (everything frozen and hashed, then the validation machines pinned) |
| `artifacts/validation_window_losses_primary.parquet` | Cell 5 (per-window native loss and full-observation loss of every policy at every budget, primary setting, validation machines) |
| `artifacts/validation_summary.json`, `figures/phase_table.csv`, `figures/loss_curves.png`, `reports/acceptance_report.json`, `reports/run_summary.md`, `COMPLETE` | Cell 6 (gates, bootstrap, sensitivities, report) |

**Not copied:** `reports/executed_notebook.ipynb` (SHA-256 `4d15353fc76d900550985809c6d08d2bef3360894b66e4d628d8826601ab914d` in the run directory; not committed, as for earlier runs). **No `diagnostics/` folder:** no post-hoc analysis was run on this run. `python scripts/d1_0b_tables.py <this folder>` prints every table in the note and recomputes the primary setting's areas and headroom from the per-window parquet, asserting they equal the stored summary.

## Data provenance

- Source: the same SMD files as D1-0 (`https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset/{train,test,test_label}/<machine>.txt`), seven tuning machines (`machine-1-1`, `1-5`, `2-1`, `2-5`, `2-9`, `3-4`, `3-8`) and seven validation machines (`machine-1-2`, `1-6`, `2-2`, `2-6`, `3-1`, `3-5`, `3-9`). Files are not in git.
- **All 42 manifest rows are `retrieved_utc: "cached"`:** every file was read from the local cache that D1-0 filled. The per-file SHA-256 values equal those in D1-0's two manifests, which is what makes the data identical. **The validation machines are the ones D1-0 used**, and D1-0's validation results (the detector's poor labelled F1, the unchanged native-endpoint F1) motivated this redesign; the forecasting numbers themselves had not been computed on any validation machine before the freeze, but this validation set is not pristine for the choice of objective. The test machines remain reserved for a confirmatory stage.

## Related runs not imported

- Two smoke tests of this notebook with a reduced configuration preceded the real run. They read no validation machine as validation (the second was fed tuning machines in both roles); their numbers are not evidence. The first crashed because one machine has a negative hold-only excess at `h` = 15, after which the normalisation was changed from the hold-only excess to the full-observation loss (plan §10, dated, before any validation machine was read).
- `d1_0b_forecast_sensing_20260930T114142Z`: the reproduction (below).

## Reproducibility

The notebook was re-executed from a clean `git worktree` at the same commit (`2b3bb99`), on the same machine and Python environment, as run `d1_0b_forecast_sensing_20260930T114142Z` (not imported; 113 s; all data files read from the cache). Comparing files with SHA-256:

- **Byte-identical:** `artifacts/validation_window_losses_primary.parquet`, `artifacts/tuning_summary.json`, `tests/correctness.json`, `config/environment.json`, `figures/phase_table.csv`, `config/data_manifest_tuning.json`, `config/data_manifest_validation.json` (both manifests are identical to each other because every row is `"cached"` in both runs).
- **Differ only in a run id, a timestamp or a timing** (compared key by key): `config/run_config.json` (only `run_id`); `artifacts/validation_summary.json` (only the five `wall_clock_seconds` values; the per-machine headroom of `machine-3-1` is `NaN` in both runs, because `sn.headroom` returns `NaN` when the fixed policy's normalised area is not positive, and `NaN` compares unequal to itself in a key-by-key check); `reports/acceptance_report.json` (only `run_id`); `reports/frozen_before_validation.json` (only `written_utc`; its `sha256` is identical, so the freeze content is identical); `reports/run_summary.md` (identical after masking the run id); `COMPLETE` (a timestamp); `figures/loss_curves.png` (578 pixels differ, all within rows 68 to 82 and columns 501 to 571 of a 721 × 991 image, where the title with the run id sits).
- So every number in the run reproduces exactly on the same machine. One repetition; no comparison across BLAS builds, CPU types or thread counts was made.

## Checkpoints

None.
