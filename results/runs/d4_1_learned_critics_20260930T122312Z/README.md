# Run `d4_1_learned_critics_20260930T122312Z` (D4-1 design study, exploratory)

- **What it is:** an exploratory, validation-only design study of the analytic D4 benchmark (goal-oriented adaptive time stepping on `y' = Ay + b(t)`) with **learned critics**, in the two cells D4-3 opened (`m4` and `m64`, dimension 4 and 64; systems, pulses and initial states as in D4-2 and D4-3). Per cell and for five paired seeds: a co-state estimator `ŵ(t_b, c)` trained on the co-state label, and four value heads trained on the gain label: `direct` (P0 inputs), `costate_critic` (P0 inputs plus three features from the trained estimator), `costate_randomised` (the same with an estimator trained on permuted labels) and `teacher_feature` (features from the exact co-state; a privileged diagnostic, lookup price only). Train family: seed 4004, 30 instances; tuning: seed 3003, 10; validation: seed 1001, 20 (the tuning and validation instances are D4-3's, each with its own random unit goal). **The test family (seed 2002) was never generated** (`test_family_read: false` in `reports/acceptance_report.json` and `config/run_config.json`). The data are generated from frozen seeds; **this is not real data.** No H2 statement about real data follows.
- **Status:** this is the run the D4-1 conclusions rest on. The write-up is `docs/research-notes/2026-09-30-d4-1-learned-critics.md`; the design and its dated changes are `docs/plans/d4-1-plan.md` (§10: the redesign made before any validation instance was generated, the estimator budget, and the execution record).
- **Where it ran:** the Claude Code sandbox, on CPU (`config/environment.json`: Python 3.11.15, NumPy 2.4.6, pandas 3.0.6; 2177 s). The notebook `notebooks/04-domains/d4_1_learned_critics.ipynb` (SHA-256 `0f15d7d969a727fe6806dff83b59a677015b8e99a3554af75b805ee0c3fd700c`) was executed with `nbclient`, detached from the tool call, at commit `9308b9ff1e0b87985e63ed73c426385fc4a4d7f4` on a clean working tree (`repo_dirty: false`).
- **Outcome (`reports/acceptance_report.json`):** correctness passed; the estimator floor (held-out relative error against the exact co-state ≤ 0.5 in at least four of five seeds) is met in `m4` (4 of 5) and not in `m64` (0 of 5); R0 (a learned critic beats uniform refinement at the real price) fails in both cells; the co-state critic does not beat the direct critic at either price (RC1, RC1-L fail) and does not beat its randomised control (RC3, RC3-L fail); RT-L (the exact co-state as a feature helps the head, lookup price) holds in `m4` and fails in `m64`. Exit classes: `m4` teacher-only value (semilog: same; staircase: inconclusive, precision); `m64` inconclusive (all three pricings). Every table in the note is printed from these files by `python scripts/d4_1_tables.py results/runs/d4_1_learned_critics_20260930T122312Z`, which also re-derives the primary intervals and rule flags from the per-instance parquet and stops if they differ from the stored ones.

| Hash | Value |
|---|---|
| Config SHA-256 | `1e3c74c5dc4f3c40b055cb88a03b17a21c40b7cca512b7cc163bec53c7e82a75` (`config_sha256` in `config/run_config.json`; equals `config_hash` in `reports/acceptance_report.json` and `config_sha256` in `reports/frozen_before_validation.json`) |
| Frozen-before-validation SHA-256 | `8f25f90283342898760fd333b6c65035641d7ca740bd68668059f6123b044fc1` (`sha256` in `reports/frozen_before_validation.json`, written 2026-09-30T12:52:33.4 UTC; it covers the head widths, the selected `θ` per (cell, arm, seed, price), the estimator-floor result, the rules and the SHA-256 of every critic file; the validation instances were written at 12:52:33.5 UTC) |
| Train instance manifest SHA-256 | `f61ee0a3527f19ca47d555a1c7a790762d3e565dd45a7d6367df4019b36aa1d9` (`config/instances_manifest_train.json`; parameters in `config/instances_train.parquet`) |
| Tuning instance manifest SHA-256 | `87ad06c41043c63c3eba1fff65ab016c366d67c02f5fdc1eb743ca010428c2b0` (`config/instances_manifest_tuning.json`; `config/instances_tuning.parquet`) |
| Validation instance manifest SHA-256 | `889c3a62e0353ce98dc0ffaebf946ed12f2c33333b26ddb8a3a2e8c77e82a2b9` (`config/instances_manifest_validation.json`; `config/instances_validation.parquet`) |
| Package commit | `9308b9ff1e0b87985e63ed73c426385fc4a4d7f4` |

## Files

Everything was copied **byte-for-byte** from `~/adjointrwm_runs/d4_1_learned_critics_20260930T122312Z/` (per-file SHA-256 comparison after the copy).

| Path | Origin |
|---|---|
| `config/run_config.json`, `config/environment.json`, `config/instances_manifest_train.json`, `config/instances_manifest_tuning.json`, `config/instances_train.parquet`, `config/instances_tuning.parquet` | Notebook cell 2 (frozen configuration, train and tuning instances) |
| `tests/correctness.json`, `tests/ledger_and_label_checks.csv` | Cell 3 (propagator against the tabulated co-state, discrete sweep against `domain.costate`, exact-co-state features against the `cheap_adjoint` score terms, gain label against the objective's own summand, FLOP formula against the counted operations, train/tuning disjointness, test family refused) |
| `artifacts/width_selection.csv` | Cell 4 (head width per cell on the tuning family at the lookup price; capacity probe not triggered) |
| `artifacts/training_summary.csv` | Cell 5 (per cell, arm and seed: head and estimator fit, parameters, price per interval, training FLOPs; the critic-FLOP formula is asserted against a count from the weights inside the notebook) |
| `artifacts/theta_tuning.csv`, `artifacts/critics_<cell>_<arm>.json` (8 files, five seeds each), `reports/frozen_before_validation.json` | Cell 6 (`θ` per (cell, arm, seed, price) on the tuning family; the trained critics; the freeze) |
| `config/instances_validation.parquet`, `config/instances_manifest_validation.json`, `artifacts/validation_work_precision.parquet` | Cell 7 (validation family generated only after the freeze; compute to each target for every arm, references and critics, three interpolation methods) |
| `artifacts/validation_summary.json`, `figures/comparisons.csv`, `figures/retained_and_break_even.csv`, `figures/phase_table.csv`, `figures/compute_ratios.png`, `reports/acceptance_report.json`, `reports/run_summary.md`, `COMPLETE` | Cell 8 (frozen rules, two-stage bootstrap intervals, retained headroom, break-even counts, figure, report) |

**Not copied:** `reports/executed_notebook.ipynb` (SHA-256 `e3cc4818db44b094a165187b3fdb358541137c4c1b42be65a4710d237dc24fc7` in the run directory; not committed, as for earlier runs). **No `diagnostics/` folder:** no post-hoc analysis was run on this run.

## Related runs not imported

- Smoke runs of this notebook with a reduced configuration (one cell, eight train instances, two seeds, `validation` replaced by train instances) ran before the real run; they read no validation instance as validation and their numbers are not evidence. An earlier smoke run of a first, different design (bilinear heads) on train and tuning instances led to the redesign recorded in plan §10; its scratch script and logs were not kept.
- `d4_1_learned_critics_20260930T154712Z`: the reproduction (below).

## Reproducibility

A re-execution of the notebook from a clean `git worktree` at the same commit (`9308b9f`), on the same machine and Python environment, produced the run `d4_1_learned_critics_20260930T154712Z` (not imported). Its file-by-file comparison with this run (SHA-256 of every file, and a key-by-key comparison of the JSON files that differ) is recorded in the file `reproduction.md` in this folder, which is added after the reproduction finishes; until it exists, no reproduction claim is made. The run took 2177 s.

## Checkpoints

The critics are small NumPy MLPs and are committed as JSON (`artifacts/critics_<cell>_<arm>.json`, 0.07 to 1.4 MB each, five seeds per file); their SHA-256 are in `reports/frozen_before_validation.json` (`critic_sha256`). There are no other checkpoints.
