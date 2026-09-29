# Notebooks

| Folder | Notebook | Role | Runs on |
|---|---|---|---|
| `01-production/` | `AdjointRWM_Production_Pilot.ipynb` | Main pipeline. Trains the DROID-100 world model, the amortised co-state and the direct critic, evaluates all gates and writes an immutable run directory to Drive. | Colab GPU (L4 is enough) |
| `02-diagnostics/` | `opportunity_audit.ipynb` | Runs in the same kernel after the pilot. Logs full per-window gain matrices and answers: is there headroom over the best fixed candidate, and does the exact co-state rank candidates? | Same runtime as the pilot |
| `archive/` | `para_0_0_1.ipynb` | Earlier "Specialist World Model" notebook. **Not evidence.** It hard-codes metrics and uses mislabelled data (see the audit note). | — |
| `archive/` | `General_Signal_Filtering_Benchmark_PARTIAL.md` | Truncated source excerpt of the synthetic denoising benchmark. **Sanity check only.** The full v6 notebook is on Drive. | — |

## Conventions

- **Numbering and folders:** notebooks are numbered by stage. `00-stage0/` is reserved for the LQTree correctness notebook (roadmap M4). New notebooks get the next number, and superseded ones move to `archive/` with a line in `CHANGELOG.md`.
- **Outputs:** commit notebooks without outputs. Executed copies live on Drive and are listed in `docs/DRIVE_INVENTORY.csv`. The small run artefacts that matter (JSON, CSV, parquet, figures) go in `results/runs/<run_id>/`.
- **Origin of these copies:** the copies here were rebuilt from Colab text exports into valid `.ipynb`. Every code cell was syntax-checked, except `para_0_0_1` cell 86, which is truncated in the original export.
