# AGENTS.md: notebooks/

These rules are in addition to the root `AGENTS.md`.

- **Colab only.** Notebooks are Colab orchestrators that need a GPU runtime, a Drive mount and DROID data. Don't execute them locally on the CPU sandbox. Dispatch and monitor them directly using the `colab-cli` skill (`colab exec -s <session> -f <notebook>.ipynb`). Worker scripts (`scripts/colab_worker.py`, `notebooks/05-ops/colab_worker.ipynb`) and the Drive job queue (`jobs/inbox/`, `05-ops/allowlist.json`) serve as a fallback when direct CLI execution is not used. (The CPU notebooks in `04-domains/` are the exception and were run in the sandbox.)
- **No outputs in git.** Commit notebooks with no outputs. CI rejects any code cell that has `outputs`. Executed copies stay on Drive and are listed in `docs/DRIVE_INVENTORY.csv`.
- **Folders.** Number folders by stage: `00-stage0/`, `01-production/`, `02-diagnostics/`, … A notebook that is superseded moves to `archive/` with a line in `CHANGELOG.md`; it is not deleted.
- **Version bumps.** To change the production pilot, make a new version (`…_v2.ipynb`) and keep v1 for provenance. List every change in `CHANGELOG.md`.
- **Reusable logic.** Anything used in two places belongs in `src/adjointrwm/` with tests. The notebook imports it.
- **Required contracts for every training notebook** (from `docs/production/colab_l4_operator_brief.md`):
  - an immutable run directory;
  - a full checkpoint contract: model, optimizer, scheduler, scaler, sampler, RNG, and hashes;
  - an episode-level split before windowing;
  - no synthetic fallback;
  - gates written to `acceptance_report.json`.
- **Editing.** Validate with `python harness/check.py`. The `notebook-hygiene` skill covers conversion from Colab text exports.
