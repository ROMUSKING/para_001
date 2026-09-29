# Run `droid100_adjoint_20260929T070629Z`

- **Status:** data ✅ · dynamics ✅ · allocation ❌ · production **BLOCKED**
- **Findings:** [`docs/research-notes/2026-09-29-droid100-pilot-findings.md`](../../../docs/research-notes/2026-09-29-droid100-pilot-findings.md)
- **Drive:** `MyDrive/Colab Notebooks/AdjointRWM_Production/runs/droid100_adjoint_20260929T070629Z` (folder ID `1vHJS7kdFqJxomm2RSjnPFtfaVbF1Obq2`)

| Hash | Value |
|---|---|
| Config SHA-256 | `433e7473f00c795e812346de93bacad23d47d1bd25189f16205c8e2836affa87` (re-verified from `config/run_config.json`) |
| Data manifest SHA-256 | `109b3a871d7a2ad1c4c3b10e1a4d4b4b223a2228d684354c4d4b9174ea75950b` |
| Source SHA-256 | `18d7cbb47a524e3c1309d23a172ffb7987d3d30cc4d45dbe0773bb14687205cf` |

## Files

| Path | Origin |
|---|---|
| `config/*.json` | Copied byte-for-byte from the Drive run directory |
| `tests/data_audit.json` | Copied from Drive |
| `tests/test_results.json` | Extracted from the acceptance report (8 tests; `allocator_negative_control` FAIL) |
| `artifacts/acceptance_report.json` | The final `REPORT` JSON printed by the notebook |
| `artifacts/dynamics_evaluation.json`, `allocator_evaluation.json`, `allocation_traces.parquet` | Copied from Drive |
| `artifacts/trace_summary.json` | Produced here by `scripts/analyze_allocation_traces.py` (episode-bootstrap CIs, choice shares) |
| `benchmarks/microbatch_training_benchmark.csv` | Copied from Drive |
| `benchmarks/systems_benchmark.json`, `rate_benchmark.json` | Extracted from the acceptance report |
| `logs/allocator_train.jsonl` | Allocator training log, steps 1–750 (all 31 logged points), parsed from the notebook output |
| `logs/dynamics_train_steps_1-825.jsonl` | Dynamics training log, steps 1–825 of 1,500. The notebook output was truncated after that; the full log is on Drive under `logs/` |
| `figures/droid100_episode_samples.png` | Sample DROID episodes (exterior + wrist frame, state[0]/action[0] traces) |

## Checkpoints (Drive only)

| File | SHA-256 |
|---|---|
| `checkpoints/best_dynamics.pt` | `4b0f07177221ae63467bd0ce7f8ae97545e9fc92a27b3a0731551648709fcc3d` |
| `checkpoints/best_allocator.pt` | `95e1dab2eee371c3785a669cbeb60d5e30d12863511efab7c0be8792c875d1a2` |
| `Phase_N3_Logs/optimized_adjoint_teacher.pt` (legacy, read-only) | `507e06694887e11d15696cc986a5079f96ae66cdadb31b37f08d0a17ed8cd70e` |
