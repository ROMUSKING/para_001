# AGENTS.md: notebooks/

These rules are in addition to the root `AGENTS.md`.

- **Colab only.** Notebooks are Colab orchestrators that need a GPU runtime, a Drive mount and DROID data. Don't execute them locally. (The CPU notebooks in `04-domains/` are the exception and were run in the sandbox. `05-ops/` holds the Colab job worker and its smoke test; see `docs/plans/colab-handoff.md` §5. Only notebooks on `05-ops/allowlist.json` on `origin/main` can be started through the queue.)
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
