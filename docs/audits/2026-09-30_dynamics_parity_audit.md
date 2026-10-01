# Audit: same-split dynamics parity `dynamics_parity_20260930T214043Z` (diagnostic, no training)

**Date:** 2026-09-30. **Auditor:** OpenCode.
**Classification:** `DIAGNOSTIC_EXECUTION: PASS` (status `OK`, `COMPLETE` written, anchors within tolerance), `DYNAMICS_GATE_VALIDATION (both checkpoints, both modes, both precisions): FAIL` (a finding, not a pipeline failure), `DATA_CONTRACT: PASS` (with the custody caveat below), `H2_EVIDENCE: NONE` (by design), `REAL_DATA_TRAINING: NOT APPLICABLE` (no training).

## Subject

The queue job `dynamics-parity-r1` (notebook `notebooks/02-diagnostics/dynamics_parity.ipynb` at commit `8b5550fca319ac1e8377b23f8226d1f5d87a8aca`, NVIDIA L4, 2026-09-30 21:40:36–21:42:17 UTC, status `ok`): scores the pilot checkpoint (`droid100_adjoint_20260929T070629Z/checkpoints/best_dynamics.pt`, step 1400) and the pilot v2 probe seed-0 checkpoint (`droid100_adjoint_v2_20260930T165409Z/jobs/seed_0/dynamics/best.pt`, step 1000) against persistence on the same train (5,910 windows) and validation (796 windows, 10 episodes) windows, in `base` and `full` modes, BF16 and FP32. No training, no allocator, no test split. The imported run folder is `results/runs/dynamics_parity_20260930T214043Z/` (six files + `README.md`); the write-up is `docs/research-notes/2026-09-30-dynamics-parity.md`.

## How this audit was done

Static inspection of committed files only. This container has no Python (`python3` not found), so `scripts/dynamics_parity_tables.py` and the config-hash recomputation were **not** re-executed here. Every number below is quoted from a committed file named in the evidence column. The import-time verification (byte equality with the zip, sizes vs Drive metadata, config-hash recomputation, CSV/JSON/report agreement) is recorded in `results/runs/dynamics_parity_20260930T214043Z/README.md` lines 6, 10, 17–26 and `CHANGELOG.md` entry (z); it is cited, not re-claimed as done in this session.

## Checklist

| # | Check | Verdict | Evidence (file and line) |
|---|---|---|---|
| D1 | Real, publisher-sourced data with episode IDs, real observation payloads, not hash strings | PASS (with custody caveat) | `results/runs/dynamics_parity_20260930T214043Z/README.md:3` (real DROID-100 features from the probe cache, no synthetic); `notebooks/02-diagnostics/dynamics_parity.ipynb:170-172` (`restore_cache` with hash verification, records with `episode_id`/`cached_path`); `src/adjointrwm/data/windows.py:233-249` (loads `states`, `actions`, exterior/wrist embeddings from the cached `.npz`). Caveat: the cache bytes and executed notebook live on Drive; the import verified sizes and SHA-256 vs the downloaded zip (`README.md:17-26`), not Drive checksums. Timestamps were not separately checked. |
| D2 | No seeded-random or synthetic substitutes | PASS | `README.md:3` (no synthetic data); notebook cell 2 builds windows from the cache with no `np.random.seed` for data; `src/adjointrwm/eval/dynamics_parity.py:19-35` (deterministic scoring of inputs). The lag-1 autocorrelation test was not run in this session (no Python); no synthetic signature was found (window counts 5910/796 and split-varying RMSEs are inconsistent with a white-noise fixture). The bootstrap helpers (`dynamics_parity.py:133-143,171-182`) exist but are unused by this run. |
| D3 | No relabelling of unrelated tabular data as states/actions (the Ailerons case) | PASS | States/actions/visual embeddings come from the DROID cache (`windows.py:233-249`; notebook cell 2). No tabular source appears anywhere in the notebook, config, or report. |
| D4 | Split by episode before windowing; split disjointness verified | PASS | `src/adjointrwm/data/windows.py:7-8,45-66` (episode split rule, before windowing); `notebooks/02-diagnostics/dynamics_parity.ipynb:174-181` (`episode_split` before any `WindowDataset`, `compare_splits` vs the pilot manifest, `assert` disjoint train/validation/test); `results/runs/dynamics_parity_20260930T214043Z/reports/acceptance_report.json:598-603` (`split_parity_with_pilot.matches: true`); datasets built per split at notebook line 231. |
| O1 | Targets are in the future (`t+1…t+H`), not the current step | PASS | `src/adjointrwm/data/windows.py:14-15` (`target_state = state[s+T : s+T+H]`, strictly after context); `slice_window` lines 164-175; notebook `SPEC` horizon 4 / context 8 (lines 119-122); `dynamics_gate_row` compares `pred_state` vs `target_state` (`dynamics_parity.py:19-29`). |
| O2 | Target encoders frozen or stop-gradient, not trained jointly | PASS (vacuous: no training) | No encoder is trained — the run scores `state_mean` vs `target_state` with fixed normalisers. Each checkpoint is scored in its own native input units; the two normalisers are identical (`acceptance_report.json:99-110`, `max_abs_diff_overall: 0.0`). Normaliser construction is fixed before scoring (notebook lines 193-194, 227-230). |
| O3 | Co-state supervised by ∂J/∂state or directional gain | NOT ASSESSABLE | No co-state is trained or scored here. The report's `claim_boundary` states this explicitly (`acceptance_report.json:95`; notebook lines 355-356). Nothing in this run is called a co-state, and this audit does not either. |
| O4 | Critic on counterfactual marginal gain; gate on measured net benefit | NOT ASSESSABLE | No critic or allocator is trained. The only "gate" is the dynamics-vs-persistence margin 0.02 (`config/run_config.json:13,24`; `acceptance_report.json` `required_margin`), which is not an allocator gate. |
| M1 | No hard-coded values | PASS | Grep over `notebooks/02-diagnostics/dynamics_parity.ipynb`, `src/adjointrwm/eval/dynamics_parity.py`, `scripts/dynamics_parity_tables.py` finds no literal metric assignments (`regret =`, `success =`, `mse =`, `task_success_rate`) and no `1.15`/`0.95`. Metrics are computed in `dynamics_gate_row` (lines 26-34); the table script asserts CSV/JSON/report agreement and recomputation of every relative improvement (lines 37-44). |
| M2 | No method-specific multipliers or constants on shared losses | PASS | Both checkpoints, modes, splits, and precisions go through the same `dynamics_gate_row` with the same margin; `MODEL_FIELDS` are identical for both builds (notebook lines 237-244). No `×1.15/×0.95`-style constant was found (same grep as M1). |
| M3 | Baselines present (persistence + linear for dynamics; random/best-fixed/oracle/direct critic for allocation) | FAIL (partial) | Persistence: present — both checkpoints share identical persistence RMSEs (0.2101 train, 0.1886 validation; `artifacts/dynamics_parity.csv:2-17`). Linear dynamics baseline: absent. Allocation baselines: not applicable (no allocation tested, by design). |
| M4 | Held-out evaluation, with per-horizon and per-episode breakdowns | FAIL (partial) | Held-out validation (796 windows) plus in-sample train are reported; per-horizon lists are present (`improvement_by_horizon` in `artifacts/dynamics_parity.json:5-9` and `acceptance_report.json:119-124`). Per-episode breakdown: absent — aggregates only; the note itself flags this (research note §H-B: per-episode predictions were not saved). Test split correctly unread (`test_split_read: false` in `config/run_config.json:38` and `acceptance_report.json:605`; notebook lines 115-116, 132, 185). |
| M5 | Seeds reported; CI (episode-cluster bootstrap) | FAIL | One v2 seed (`v2_seed: 0`, `config/run_config.json:27`) plus one pilot checkpoint; no confidence interval or bootstrap in this run. The 2 % gate is a point comparison on ten validation episodes. (`cluster_bootstrap_relative_improvement` exists in `dynamics_parity.py:133-143` but is unused here; it belongs to the planned k-fold study.) |
| R1 | Checkpoint contract (model, optimizer, scheduler, scaler, sampler, RNG, hashes; every module weights + optimizer state) | NOT ASSESSABLE (no new checkpoint; scored payloads listed but not module-verified) | This run wrote no checkpoint (`README.md:39-41`). Scored payloads: v1 keys include `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `rng_state`, `sampler_state` but no `scaler_state_dict`; v2 keys include all of those plus `scaler_state_dict` (`acceptance_report.json:46-63,71-91`). Both were hash-checked before loading (notebook lines 217-220). Module-level weights-vs-optimizer completeness was not verified here (checkpoints are Drive-only; no torch in this container). |
| R2 | Config, data-manifest, source hashes recorded; config hash re-verified | FAIL (partial: config PASS, manifests FAIL) | Config hash `f913d72ac322c557bb9bb739d5fe391f808d97c5d63b7375e2b9468df0299d1a` is recorded in both `config/run_config.json:30` and `acceptance_report.json:96`; the table script re-verifies it (`scripts/dynamics_parity_tables.py:30-31`) and the import README records recomputation (`README.md:10`). This run's own `run_config.json` (lines 2-40) carries **no** `data_manifest_hash` or `source_hash`; provenance rests on split parity, the two checkpoint SHAs, and the repo commit instead. |
| R3 | Resume-equivalence test, not just reload | NOT ASSESSABLE | No training, so nothing to resume. (The CPU pause-and-resume check belongs to `tests/test_training.py` per the roadmap, not to this run.) |
| C1 | Every claim maps to a computed artefact; hardware per the operator brief | PASS (one hedge recorded) | `reports/run_summary.md:5-22` matches `dynamics_parity.csv:2-17` row for row (values quoted: v1 validation `base` 0.2325 vs 0.1886 → −0.233; v2 validation `base` 0.2125 vs 0.1886 → −0.127); `claim_boundary` is stated in the report and summary (`acceptance_report.json:95`; `run_summary.md:26`); the research note labels H-A–H-D untested and excludes H2 (note §§4-6). Anchors: v2 reproduces exactly (diff 0.0), pilot within 0.005 (diffs 0.0021–0.0036; `acceptance_report.json:2-37`); the 0.002–0.004 pilot offset is unexplained (H-C). Hardware: L4 (`config/run_config.json:33`; notebook L4 guard lines 100-104); no larger GPU is recommended. |

## What the run shows (from the committed files, not re-computed here)

- Both checkpoints **fail** the 2 % dynamics gate on the validation split in both modes and both precisions, and pass on the train split (`run_summary.md:7-22`; `dynamics_parity.csv:2-17`). The probe's seed-0 failure is therefore **not specific to the v2 training run**: on the same 796 validation windows the pilot checkpoint is further below persistence (BF16: −0.233 `base`, −0.219 `full`) than v2 (−0.127 `base`, −0.125 `full`).
- The pilot checkpoint passes on test (+0.312; `results/runs/droid100_adjoint_20260929T070629Z/artifacts/dynamics_evaluation.json:2-9`: model 0.1555880680680275 vs persistence 0.22619297169148922) and fails on validation (−0.219 `full`). The gate verdict flips between two ten-episode held-out splits for one checkpoint.
- BF16 vs FP32 differ by at most 7.6e-05 in model RMSE — quoted from the research note §3, which the table script enforces via the committed rows; not re-computed in this session.

## Hashes of what was audited

| Artefact | SHA-256 |
|---|---|
| Config (`config` block) | `f913d72ac322c557bb9bb739d5fe391f808d97c5d63b7375e2b9468df0299d1a` (`config/run_config.json:30`; equals `config_hash` in `acceptance_report.json:96`) |
| Pilot checkpoint scored | `4b0f07177221ae63467bd0ce7f8ae97545e9fc92a27b3a0731551648709fcc3d` (`config/run_config.json:23`; `acceptance_report.json:64`) |
| Pilot v2 seed-0 checkpoint scored | `66c80f464a442da0ba9e3ee424aef692a1be4b76eff860fe1d8640c08555882b` (`config/run_config.json:25`; `acceptance_report.json:92`) |
| Package commit | `8b5550fca319ac1e8377b23f8226d1f5d87a8aca` (`config/run_config.json:35`; `acceptance_report.json:114`) |
| `COMPLETE` (33 B) | `0f4dde370780ecd4dbd3dcf45e24a54f57a2d0a8c3a153e7a5d535db35374f8f` (`README.md:21`) |
| `config/run_config.json` (1197 B) | `2976277004a574ede3984bb19292493f13501eea8746957e24adb856bc9e6a03` (`README.md:22`) |
| `artifacts/dynamics_parity.json` (11965 B) | `11f29db5528b73b6a44ca00e066ab5b44c1613cc971e94036967e1dd457aad0b` (`README.md:23`) |
| `artifacts/dynamics_parity.csv` (1567 B) | `611f0d5a8c14d4babadfb64275262a9f49b535f90abd8be5c0bee6608235b4bc` (`README.md:24`) |
| `reports/acceptance_report.json` (16829 B) | `3188f146d9468045ecb2958a427db7ad92b8645dabd2cd51aa9fb17e2c5e189c` (`README.md:25`) |
| `reports/run_summary.md` (1775 B) | `62e6f3cc039cc40ac934f5a817d12ee22dddb76b38cf843fe2747c9a5788c4f3` (`README.md:26`) |

## Do not reuse for

- H2 or any adjoint-vs-critic claim (no co-state, critic, or allocator here; `claim_boundary` excludes it).
- A multi-seed or uncertainty-quantified result (one v2 seed, no CI; ten validation episodes).
- A general "dynamics works / does not work" verdict (pilot: +31 % test, −22 % validation; train passes are in-sample).
- A v2-regime-vs-pilot-regime comparison (one seed each; direction consistent but unpowered — note H-D).
- A linear-baseline or per-episode analysis (neither is in this run).
- An explanation of the splits' difference (H-A–H-D untested) or of the 0.002–0.004 pilot anchor offset (H-C untested).
- Weights (none were written; both scored checkpoints stay on Drive).

## Open (not blockers for this diagnostic)

1. The k-fold study (planned, frozen, not yet run) answers H-A with pooled episode-cluster intervals and per-episode errors.
2. If a follow-up diagnostic runs first, save per-episode predictions (H-B) and re-score the pilot with its own batch-mean weighting (H-C).
3. This audit did not re-execute the table script or hash recomputation (no Python in this container); a future session with Python can re-run `python scripts/dynamics_parity_tables.py results/runs/dynamics_parity_20260930T214043Z` and record the output.
