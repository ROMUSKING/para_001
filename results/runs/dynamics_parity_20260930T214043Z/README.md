# Run `dynamics_parity_20260930T214043Z` (same-split dynamics parity, diagnostic)

- **What it is:** a no-training diagnostic. It scores two checkpoints against persistence (the last observed state repeated over the horizon) on the same windows: the pilot's `best_dynamics.pt` and the pilot v2 probe's seed-0 `best.pt`, on the **train** and **validation** splits only, in `base` and `full` prediction mode, with BF16 autocast and in FP32. Each checkpoint is scored in its own input normaliser (the two are identical). The episode split is the pilot's (split parity true). **The test split was never built or read** (`test_split_read: false` in `reports/acceptance_report.json` and `config/run_config.json`). Real DROID-100 features from the probe run's cache; no synthetic data.
- **Status:** `status: OK`, `anchors_ok: true`, `COMPLETE` written. The write-up is `docs/research-notes/2026-09-30-dynamics-parity.md`. This is a diagnostic of dynamics against persistence; it is not evidence for or against H2.
- **Where it ran:** Colab, NVIDIA L4, through the job queue (job `dynamics-parity-r1`; worker commit `8b5550fca319ac1e8377b23f8226d1f5d87a8aca`; Python 3.13.15, torch 2.11.0+cu128; `repo_dirty: false`). Started 2026-09-30 21:40:36 UTC, result written 21:42:17 UTC. The notebook is `notebooks/02-diagnostics/dynamics_parity.ipynb` at that commit.
- **Outcome (`reports/acceptance_report.json`):** all five anchors within tolerance (the two pilot v2 anchors reproduce the probe's stored numbers exactly; the three pilot anchors differ from the pilot's stored validation metrics by 0.0021 to 0.0036, tolerance 0.005, cause untested). Both checkpoints fail the 2 % gate on the validation split in every mode and precision; both pass on the train split. The numbers are in the note; `python scripts/dynamics_parity_tables.py results/runs/dynamics_parity_20260930T214043Z` prints every table from these files and stops if the CSV, the JSON and the report disagree, if a relative improvement does not recompute from its two RMSEs, or if the config hash does not verify.

| Hash | Value |
|---|---|
| Config SHA-256 | `f913d72ac322c557bb9bb739d5fe391f808d97c5d63b7375e2b9468df0299d1a` (`config_sha256` in `config/run_config.json`; equals `config_hash` in `reports/acceptance_report.json`; recomputed from the `config` block with `json.dumps(sort_keys=True, separators=(",", ":"))` at import) |
| Package commit | `8b5550fca319ac1e8377b23f8226d1f5d87a8aca` |
| Pilot checkpoint scored | `4b0f07177221ae63467bd0ce7f8ae97545e9fc92a27b3a0731551648709fcc3d` (`droid100_adjoint_20260929T070629Z/checkpoints/best_dynamics.pt`, step 1400, Drive only; hash-checked by the run before loading) |
| Pilot v2 seed-0 checkpoint scored | `66c80f464a442da0ba9e3ee424aef692a1be4b76eff860fe1d8640c08555882b` (`droid100_adjoint_v2_20260930T165409Z/jobs/seed_0/dynamics/best.pt`, step 1000, Drive only; hash-checked by the run before loading) |

## Files

Copied **byte-for-byte** from the Drive run folder `MyDrive/Colab Notebooks/AdjointRWM_Production/runs/dynamics_parity_20260930T214043Z/` (folder ID `1EdVEDEb_3Cg9peBgoZAyQWYpRR0biHPk`), which Roman downloaded as a zip (`dynamics_parity_20260930T214043Z-20260930T215241Z-1-001.zip`, 9,803 bytes; zip integrity test passed). After the copy, each file's SHA-256 was compared with the file in the zip and each size with the Drive metadata (all equal).

| Path | Bytes | SHA-256 | Origin |
|---|---:|---|---|
| `COMPLETE` | 33 | `0f4dde370780ecd4dbd3dcf45e24a54f57a2d0a8c3a153e7a5d535db35374f8f` | Written by the notebook only when its own status is `OK` (2026-09-30T21:42:14.29 UTC) |
| `config/run_config.json` | 1197 | `2976277004a574ede3984bb19292493f13501eea8746957e24adb856bc9e6a03` | Notebook cell 2: frozen configuration, repository commit, GPU, environment |
| `artifacts/dynamics_parity.json` | 11965 | `11f29db5528b73b6a44ca00e066ab5b44c1613cc971e94036967e1dd457aad0b` | Cell 5: the 16 scored rows (checkpoint x mode x split x precision), with by-horizon lists |
| `artifacts/dynamics_parity.csv` | 1567 | `611f0d5a8c14d4babadfb64275262a9f49b535f90abd8be5c0bee6608235b4bc` | Cell 5: the same rows as a table, without the by-horizon lists |
| `reports/acceptance_report.json` | 16829 | `3188f146d9468045ecb2958a427db7ad92b8645dabd2cd51aa9fb17e2c5e189c` | Cell 6: status, anchors, checkpoint facts, normaliser comparison, split parity, rows, mechanical readings |
| `reports/run_summary.md` | 1775 | `62e6f3cc039cc40ac934f5a817d12ee22dddb76b38cf843fe2747c9a5788c4f3` | Cell 6: the table and the readings as text |

**Not copied:** the executed notebook (`executed_notebook.ipynb`, 37,547 bytes) and the job's `result.json`, which are in the job's result folder on Drive (`jobs/results/dynamics-parity-r1/`, folder ID `1xkivIvd7-fULOkgunWRDrUh0I3IvGAzN`), not in the run folder; and the empty `tests/` folder the notebook created (it wrote nothing there). **No `diagnostics/` folder:** no post-hoc analysis was run on this run.

## Related runs not imported

- `droid100_adjoint_v2_20260930T165409Z`, the one-seed B2 probe (incomplete; recorded in `docs/audits/2026-09-30_b2_probe_seed0_dynamics_gate_audit.md`). Its `dynamics_gate.json` is one of this run's anchors.
- The pilot `droid100_adjoint_20260929T070629Z` is imported (`results/runs/droid100_adjoint_20260929T070629Z/`); its committed `artifacts/dynamics_evaluation.json` supplies the test-split figures quoted in the note.

## Reproducibility

Not re-executed. The run needs the two Drive checkpoints, the probe run's feature cache and an L4, so it can only be repeated on Colab (`python scripts/colab_job.py make --notebook notebooks/02-diagnostics/dynamics_parity.ipynb --commit <sha>`). What can be repeated here is the table check above.

## Checkpoints

None were written. The two scored checkpoints stay on Drive; their SHA-256 values are in the table above and in `reports/acceptance_report.json`.
