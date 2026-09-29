# para_001: Adjoint-Guided Recursive World Models (AdjointRWM)

Research repository for **adjoint-guided recursive rate-distortion world models**. The core question: can a recursive world model decide where to spend representational detail, compute and sensing by using a **co-state**, the gradient of a declared future cost with respect to the model state? The test is whether this beats a parameter-, information- and compute-matched **direct marginal-gain critic**.

## Current status (2026-09-29)

| Gate | Result |
|---|---|
| Real data (DROID-100, episode-level split, no synthetic fallback) | ✅ pass |
| Dynamics beat persistence on held-out episodes | ✅ pass: RMSE 0.156 vs 0.226 (−31 %), single seed |
| Allocation: adjoint beats random | ❌ **fail**: adjoint 0.037, critic 0.038, one random draw 0.021 |
| Adjoint vs critic (H2) | ⚪ tie: −0.0009, 95 % CI [−0.0028, +0.0004] |
| Task success | not measured (offline data only) |

**Diagnosis so far:** both learned allocators collapsed onto candidates {0, 2} and never chose 1 or 3, which the oracle prefers on 65 % of test windows. Details are in [`docs/research-notes/2026-09-29-droid100-pilot-findings.md`](docs/research-notes/2026-09-29-droid100-pilot-findings.md) and the next steps in [`docs/plans/roadmap.md`](docs/plans/roadmap.md).

## Layout

```
docs/
  research-plan/      Full preregistration-style research plan (v5, 24 Sep 2026)
  research-notes/     Dated findings: pilot results, audit of earlier notebooks
  plans/roadmap.md    Milestones, decision points, next-run changes
  DRIVE_INVENTORY.csv Every Drive file found, with ID, size, status and repo path
papers/
  drafts/             Paper draft + REVIEW.md (claim-by-claim evidence check)
  related-work.bib    Core bibliography (verify entries before citing)
notebooks/
  01-production/      AdjointRWM_Production_Pilot.ipynb: the main pipeline (Colab, L4/A100)
  02-diagnostics/     opportunity_audit.ipynb: run after the pilot, no retraining
  archive/            Earlier notebooks kept for provenance; NOT evidence (see audit note)
src/adjointrwm/       Tested Python code lifted out of notebooks (analysis helpers so far)
scripts/              CLI tools, e.g. analyze_allocation_traces.py
tests/                pytest suite (runs on CPU, no GPU or dataset needed)
results/
  runs/<run_id>/      Small artefacts per run: config, logs, evaluations, traces, figures
  legacy/             Earlier phase logs, kept with caveats
```

Checkpoints (~100–290 MB each) and executed notebooks with outputs stay on Google Drive. [`docs/DRIVE_INVENTORY.csv`](docs/DRIVE_INVENTORY.csv) maps each one to its Drive file ID.

## Quick start

```bash
pip install -e ".[dev]"
pytest
python scripts/analyze_allocation_traces.py \
  results/runs/droid100_adjoint_20260929T070629Z/artifacts/allocation_traces.parquet
```

To reproduce the pilot, open `notebooks/01-production/AdjointRWM_Production_Pilot.ipynb` in Colab with a GPU runtime and mount Drive. The notebook pulls `droid_100` from `gs://gresearch/robotics` through TFDS and writes an immutable run directory under `MyDrive/Colab Notebooks/AdjointRWM_Production/runs/`.

> The notebook copies here were rebuilt from Colab text exports. The code is complete, but cell outputs were not preserved. The executed originals are on Drive.

## Ground rules

- A result counts only if it traces to a committed run directory with its config hash and the gate that passed.
- Report every outcome class, including negative results and ties (plan §2.4 and "Key falsifiers").
- Don't describe a head as a "co-state" unless it is supervised by or derived from ∂J/∂state (plan §15.2).

## License

MIT (see `LICENSE`).
