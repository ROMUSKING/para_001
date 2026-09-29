# AGENTS.md: results/

These rules are in addition to the root `AGENTS.md`.

- **Immutability.**
  - `results/runs/<run_id>/` is **immutable after its first commit**. You may add new files, such as a later `diagnostics/` folder, but you may not modify or delete existing ones. `harness/check.py --base origin/main` enforces this.
  - If an imported artefact is wrong, record that in a new audit under `docs/audits/`; don't edit the artefact.
- **What goes in a run folder.** Every run folder has a `README.md` listing:
  - where each file came from (copied byte-for-byte, extracted from the report, or produced here by which script);
  - the config, data-manifest and source hashes;
  - checkpoint SHA-256 values (the checkpoints themselves stay on Drive).
- **Keep it small.** Only small artefacts belong here: JSON, JSONL, CSV, parquet under ~5 MB, and figures.
- **`legacy/`** holds non-evidential material. Never cite it (see `docs/audits/`).
- **Importing.** Use the `import-run` skill.
