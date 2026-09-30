# Run `d1_0_sensor_opportunity_20260930T062900Z` (D1-0 Rung-0 opportunity gate, exploratory)

- **What it is:** an exploratory Rung-0 gate for the D1 sensor-stream domain on the Server Machine Dataset (SMD): does a privileged allocation of sensing (which of 38 channels to observe at every minute of a 60-minute window, the rest held at the window-start snapshot) beat the best fixed allocation by at least 15 % of the normalised area, and how much of that can non-learned deployable policies keep? The loss is the score-fidelity of a frozen Mahalanobis detector on the partially observed window. **Seven validation machines** gave the gate; seven tuning machines chose the detector's `(floor, ridge)` and the best fixed family; **the 14 test machines were never downloaded** (`test_machines_downloaded: false`; `~/adjointrwm_data/smd/train/` held exactly the 14 tuning and validation machines). No learned critic and no co-state: **no H2 statement.** The data are real; they are **not committed** (MIT licence, register id `smd`).
- **Status:** this is the run the D1-0 conclusions rest on. The write-up is `docs/research-notes/2026-09-30-d1-0-sensor-opportunity.md`; the design and its dated changes are `docs/plans/d1-0-plan.md`.
- **Where it ran:** the Claude Code sandbox, on CPU (`config/environment.json`: Python 3.11.15, NumPy 2.4.6, pandas 3.0.6). The notebook `notebooks/04-domains/d1_0_sensor_opportunity.ipynb` (SHA-256 `bacede2b25583ecb2a9a84f915ec860869e619d64d0a2435d3b68d0879f0178a`) was executed with `nbclient`, detached from the tool call, at commit `7c5c89c52936347298f29ba06f1a1f9d66f367d8` on a clean working tree (`repo_dirty: false`).
- **Outcome (`reports/acceptance_report.json`):** correctness passed; G1 at the primary run (`L` = 60, score objective) passes robustly; see the note for what that does and does not mean (the native-endpoint F1 does not improve with sensing).

| Hash | Value |
|---|---|
| Config SHA-256 | `3c9fb6c74d8a42dd112052a4ed495ac0bf45d20a1b0e449aa0f06185534fd78f` (`config_sha256` in `config/run_config.json`; equals `config_hash` in `reports/acceptance_report.json` and `config_sha256` in `reports/frozen_before_validation.json`) |
| Frozen-before-validation SHA-256 | `d128a670eb03d4e527a94182782769ebf3b8cda55d533e90c3abfc48f91d0d95` (`sha256` in `reports/frozen_before_validation.json`, written 2026-09-30T06:29:18.9 UTC; it covers the config hash, the tuning-manifest hash, the detector choice and the best-fixed choice per window length) |
| Tuning data manifest SHA-256 | `44bff97cb2597e0dab28e88e107aa76cf4cfe8a28274b0448bc5945daa35d222` (`sha256` in `config/data_manifest_tuning.json`: 21 files, 123.9 MB) |
| Validation data manifest SHA-256 | `424bbed770bbd9b6ad37789d4e9d71e8c3606ba0703c7830db4eec0bdfbb6a5f` (`sha256` in `config/data_manifest_validation.json`: 21 files, 124.1 MB) |
| Domain-spec SHA-256 | `b35a66acbb79c06f9a8da6a9950cd7b8104c980534d4c443a286ca559dca1f4b` |
| Package commit | `7c5c89c52936347298f29ba06f1a1f9d66f367d8` |

**A caveat on the manifest hashes.** Each manifest row includes `retrieved_utc`, so the manifest's own SHA-256 changes when a file is read from the local cache instead of downloaded. The identity of the data is the per-file `sha256` in each row, which is what to compare.

## Files

Everything was copied **byte-for-byte** from `~/adjointrwm_runs/d1_0_sensor_opportunity_20260930T062900Z/` (per-file SHA-256 comparison after the copy).

| Path | Origin |
|---|---|
| `config/run_config.json`, `config/environment.json`, `config/data_manifest_tuning.json` | Notebook cell 1 (frozen configuration; tuning machines downloaded and pinned) |
| `artifacts/detector_selection.json`, `artifacts/best_fixed_tuning.json` | Cell 2 (detector `(floor, ridge)` and the best fixed family at each window length, tuning machines only) |
| `tests/correctness.json` | Cell 3 (correctness checks on real tuning windows) |
| `reports/frozen_before_validation.json`, `config/data_manifest_validation.json` | Cell 4 (everything frozen and hashed, then the validation machines downloaded and pinned) |
| `artifacts/validation_window_losses_primary.parquet` | Cell 5 (per-window loss of every policy at every budget, primary run, validation machines) |
| `artifacts/validation_summary.json`, `figures/phase_table.csv`, `figures/loss_curves.png`, `reports/acceptance_report.json`, `reports/run_summary.md`, `COMPLETE` | Cell 6 (gate, bootstrap, sensitivities, report) |

**Not copied:** `reports/executed_notebook.ipynb` (SHA-256 `a0e3c9bc668229e9e6f0637e09acc70663bc9b6bcaf78868610b79b1ec600d4c` in the run directory; not committed, as for earlier runs). **No `diagnostics/` folder:** no post-hoc analysis was run on this run. `python scripts/d1_0_tables.py <this folder>` prints every table in the note and recomputes the primary run's areas and headroom from the per-window parquet.

## Data provenance

- Source: `https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset/{train,test,test_label}/<machine>.txt` for the seven tuning machines (`machine-1-1`, `1-5`, `2-1`, `2-5`, `2-9`, `3-4`, `3-8`) and the seven validation machines (`machine-1-2`, `1-6`, `2-2`, `2-6`, `3-1`, `3-5`, `3-9`). Files are not in git.
- Five tuning machines (`1-1`, `1-5`, `2-1`, `2-5`, `2-9`) were read from the local cache (`retrieved_utc: "cached"`): they had been downloaded by a smoke test of the notebook earlier the same day (below). The other two tuning machines were downloaded by this run before the freeze. **All seven validation machines were downloaded by this run, from 2026-09-30T06:29:20 UTC, after the freeze was written at 06:29:18.9 UTC**; no validation file existed before.

## Related runs not imported

- Two smoke tests of this notebook with a reduced configuration (three tuning machines standing in for tuning, two other tuning machines standing in for validation, 300 bootstrap resamples) ran before the real run. They read **no validation machine** and their numbers are not evidence; after the first one, the oracle was changed from forward selection to the better of forward selection and backward elimination (plan §10).
- `d1_0_sensor_opportunity_20260930T063142Z`: the reproduction (below).

## Reproducibility

The notebook was re-executed from a clean `git worktree` at the same commit (`7c5c89c`), on the same machine and Python environment, as run `d1_0_sensor_opportunity_20260930T063142Z` (not imported; all data files read from the cache). Comparing files with SHA-256:

- **Byte-identical:** `artifacts/validation_window_losses_primary.parquet`, `artifacts/detector_selection.json`, `artifacts/best_fixed_tuning.json`, `tests/correctness.json`, `config/environment.json`, `figures/phase_table.csv`.
- **Differ only in a run id, a timestamp, a timing or a hash that covers a timestamp** (compared key by key): `config/run_config.json` (only `run_id`); `artifacts/validation_summary.json` (only the four `wall_clock_seconds` values); `reports/acceptance_report.json` (only `run_id` and `frozen_sha256`); `reports/frozen_before_validation.json` (only `written_utc`, `tuning_manifest_sha256` and therefore `sha256`); the two data manifests (only `retrieved_utc`, `"cached"` against a timestamp, and the manifest's own `sha256`; every per-file `sha256` and size is identical); `reports/run_summary.md` (identical after masking the run id); `COMPLETE`; `figures/loss_curves.png` (its title contains the run id; pixels were not compared).
- So every number in the run reproduces exactly on the same machine. One repetition; no comparison across BLAS builds, CPU types or thread counts was made.

## Checkpoints

None.
