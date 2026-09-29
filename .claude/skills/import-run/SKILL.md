---
name: import-run
description: Import a finished Colab training run from Google Drive into results/runs/<run_id>/ with provenance, hashes and a trace summary. Use when the user says a run finished, shares a run directory or acceptance report, or asks to add run results to the repo.
---

# Import a run

**Goal:** get the run's small artefacts into `results/runs/<run_id>/` exactly as the run produced them, with a README that makes every file traceable. Checkpoints stay on Drive.

## Inputs

- **Run ID:** e.g. `droid100_adjoint_20260929T070629Z`. On Drive, runs live at `MyDrive/Colab Notebooks/AdjointRWM_Production/runs/<run_id>/`.
- **Source:** the Drive folder, if a Drive tool is available. Otherwise use files or notebook output the user attached.

## Procedure

1. **Check it's new.** Make sure `results/runs/<run_id>/` doesn't exist yet. Existing run folders are immutable (`results/AGENTS.md`).
2. **List the Drive folder.** Take `config/`, `tests/`, `artifacts/`, `benchmarks/` and `logs/`. Copy **byte-for-byte**, keep the folder names, and after each download compare the byte size with the Drive metadata.
   - Take: `*.json`, `*.jsonl`, `*.csv`, `*.parquet` under ~5 MB, and plots.
   - Skip: `checkpoints/*.pt`, `normalisation.npz`, `pip_freeze.txt`, the large `data_manifest.json`. Record these in `docs/DRIVE_INVENTORY.csv` with their Drive ID, and put checkpoint SHA-256 values in the README.
3. **Re-verify the config hash:**
   ```python
   import json, hashlib
   c = json.load(open("config/run_config.json"))["config"]
   hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
   ```
   It must equal `config_hash` in `acceptance_report.json`. If it doesn't, stop and report the mismatch.
4. **Pasted output only.** If the only source is pasted notebook output, you may parse it (e.g. Python-dict log lines via `ast.literal_eval`). Name the resulting files to show where they came from and whether they're complete, e.g. `logs/dynamics_train_steps_1-825.jsonl`. Never fill gaps.
5. **Summarise the traces, if present:**
   ```bash
   python scripts/analyze_allocation_traces.py results/runs/<id>/artifacts/allocation_traces.parquet \
     --num-candidates <K> --out results/runs/<id>/artifacts/trace_summary.json
   ```
6. **Write `results/runs/<id>/README.md`.** Use `results/runs/droid100_adjoint_20260929T070629Z/README.md` as the template: gate status line, hash table, file-origin table, checkpoint table.
7. **Update the records:** add the new files to `docs/DRIVE_INVENTORY.csv` (`status=copied`), add a line to `CHANGELOG.md`, and add a WORKLOG entry.
8. **Verify** with `python harness/check.py`.
9. **Write it up.** If the run changes any conclusion, use the `research-note` skill. The import itself makes no claims.

## Don'ts

- Don't round, reformat or "clean" JSON values.
- Don't create an artefact the run didn't produce.
- Don't overwrite an existing run folder. Use a new run ID, or a new audit if something was wrong.
