# Dynamics parity: the pilot checkpoint and the pilot v2 seed-0 checkpoint on the same windows (2026-09-30)

- **Run:** `dynamics_parity_20260930T214043Z` (queue job `dynamics-parity-r1`, started 2026-09-30 21:40:36 UTC, result written 21:42:17 UTC, status `ok`; the run wrote `COMPLETE`, which the notebook writes only when its own status is `OK`).
- **Notebook:** `notebooks/02-diagnostics/dynamics_parity.ipynb` at repository commit `8b5550fca319ac1e8377b23f8226d1f5d87a8aca`. **Hardware:** NVIDIA L4 (Colab). **Config SHA-256:** `f913d72ac322c557bb9bb739d5fe391f808d97c5d63b7375e2b9468df0299d1a` (`config_hash` in `reports/acceptance_report.json`).
- **Checkpoints scored (hash-checked against the committed hashes):** pilot `droid100_adjoint_20260929T070629Z/checkpoints/best_dynamics.pt` (SHA-256 `4b0f0717…cc3d`, step 1400, trained with `mask_mode='subset'`) and pilot v2 probe `droid100_adjoint_v2_20260930T165409Z/jobs/seed_0/dynamics/best.pt` (SHA-256 `66c80f46…882b`, step 1000, `single_choice`).
- **Artefact path and provenance.** `results/runs/dynamics_parity_20260930T214043Z/` (six files, copied byte for byte from the Drive run folder; every size equals the Drive metadata, the config hash verifies; see its `README.md`). Every table below is printed from those files by `python scripts/dynamics_parity_tables.py results/runs/dynamics_parity_20260930T214043Z`, which also stops if the CSV, the JSON and the report disagree or a relative improvement does not recompute from its two RMSEs; `tests/test_dynamics_parity_tables.py` checks that this note contains those tables. The pilot's test-split figures come from `results/runs/droid100_adjoint_20260929T070629Z/artifacts/dynamics_evaluation.json`. The test split was not built or read by this run (`test_split_read: false`).
- **Status:** data ✅ (split parity with the pilot true, normalisers identical) · anchors ✅ (within my tolerances, see §2) · **pilot checkpoint, dynamics gate on the validation split ❌** · **pilot v2 seed 0, dynamics gate on the validation split ❌** · H2 not tested.

## 1. What this run is

- **Question.** The one-seed B2 probe failed its dynamics gate on the validation split (model RMSE 0.2125 against persistence 0.1886; audit `docs/audits/2026-09-30_b2_probe_seed0_dynamics_gate_audit.md`). The pilot's pass (+31.2 %) had been measured on a different split with a different training regime. Is the failure specific to the v2 run?
- **Method.** No training. Both checkpoints, each in its own input normaliser (the two normalisers are identical, max absolute difference 0.0), scored against persistence (the last observed state repeated over the horizon) on the same windows: the **train** split (5,910 windows) and the **validation** split (796 windows; 10 episodes), in `base` and `full` prediction mode, with BF16 autocast and in FP32. Metric: horizon-mean RMSE in normalised units; relative improvement is (persistence − model) / persistence; the gate is 2 %.
- **Data:** DROID-100, 80 / 10 / 10 episodes split by episode before windowing, the pilot's split (`split_parity_with_pilot.matches: true`), real image payloads, no synthetic fallback.

## 2. Anchors (checked before any reading)

| Anchor | Recomputed | Stored by the original run | Absolute difference | Tolerance |
|---|---:|---:|---:|---:|
| v2 validation `base` RMSE, against the probe's `dynamics_gate.json` | 0.2124711301360882 | 0.2124711301360882 | 0.0 | 0.001 |
| v2 validation persistence RMSE, same file | 0.18857847256011034 | 0.18857847256011034 | 0.0 | 0.001 |
| Pilot validation `base` RMSE, against the `best_metrics` stored in its checkpoint (FP32 rows) | 0.23252931343221625 | 0.22893401980400085 | 0.0035952936282153958 | 0.005 |
| Pilot validation `full` RMSE, same | 0.22983233099167405 | 0.22631007805466652 | 0.0035222529370075306 | 0.005 |
| Pilot validation persistence RMSE, same | 0.18857847256011034 | 0.18652507103979588 | 0.0020534015203144684 | 0.005 |

The two tolerances are policy values I fixed before the run, not measurements. The v2 numbers reproduce exactly. The pilot's do not reproduce exactly: the recomputed RMSEs are 0.002 to 0.004 above the stored ones (about 1 to 1.6 %). The cause is **not established**; the pilot's `evaluate_dynamics` (read from the committed pilot notebook) accumulates per-batch mean squared errors and divides by the number of batches, which weights the last, smaller batch (28 of the 796 windows at batch size 64 or 128) like a full one, whereas this diagnostic weights windows equally; whether that accounts for the offset is a hypothesis (H-C). The offset does not matter for the conclusions below: computed from the pilot's own stored validation numbers (this session, from the two values above), the gate's relative improvement is −0.213 (`full`) and −0.227 (`base`), against −0.219 and −0.233 here.

## 3. Results

**Horizon-mean RMSE against persistence, BF16** (the pipeline of pilot v2's own gate; `v1` = the pilot checkpoint, `v2` = pilot v2 seed 0):

| Checkpoint | Mode | Split | Windows | Model RMSE | Persistence RMSE | Relative improvement | Passes 2 % |
|---|---|---|---:|---:|---:|---:|---|
| v1 | base | train | 5910 | 0.1497 | 0.2101 | +0.288 | yes |
| v1 | base | validation | 796 | 0.2325 | 0.1886 | -0.233 | no |
| v1 | full | train | 5910 | 0.1429 | 0.2101 | +0.320 | yes |
| v1 | full | validation | 796 | 0.2298 | 0.1886 | -0.219 | no |
| v2 | base | train | 5910 | 0.1542 | 0.2101 | +0.266 | yes |
| v2 | base | validation | 796 | 0.2125 | 0.1886 | -0.127 | no |
| v2 | full | train | 5910 | 0.1524 | 0.2101 | +0.275 | yes |
| v2 | full | validation | 796 | 0.2122 | 0.1886 | -0.125 | no |

The FP32 rows differ from the BF16 rows by at most 7.6e-05 in model RMSE (computed over the eight pairs from the CSV); the FP32 table is in the appendix. Persistence's RMSE is the same for both checkpoints (0.2101 on train, 0.1886 on validation).

**Relative improvement over persistence by horizon step (BF16; negative = persistence is better):**

| Checkpoint, mode, split | Step 1 | Step 2 | Step 3 | Step 4 |
|---|---:|---:|---:|---:|
| v1, `full`, validation | −1.123 | −0.336 | −0.064 | +0.064 |
| v2, `base`, validation | −0.889 | −0.223 | +0.003 | +0.109 |
| v2, `full`, validation | −0.895 | −0.230 | +0.005 | +0.119 |
| v1, `full`, train | −0.117 | +0.258 | +0.397 | +0.460 |
| v2, `base`, train | −0.206 | +0.202 | +0.350 | +0.415 |
| pilot checkpoint, `full`, **test** (the pilot's committed `dynamics_evaluation.json`) | −0.138 | +0.258 | +0.387 | +0.446 |

**The pilot checkpoint across the three splits** (`full` mode; train and validation from this run, test from `results/runs/droid100_adjoint_20260929T070629Z/artifacts/dynamics_evaluation.json`): model RMSE 0.1429 (train), **0.2298 (validation)**, 0.1556 (test); persistence RMSE 0.2101, **0.1886**, 0.2262. Relative improvement +0.320, **−0.219**, +0.312.

## 4. Interpretation

**Observed.**

1. Both checkpoints fail the 2 % gate on the validation split in both prediction modes and in both precisions. The probe's failure is therefore **not specific to the v2 training run**: the pilot checkpoint, on the same 796 windows, is further below persistence (−0.219 to −0.233 against −0.125 to −0.127 for v2, BF16).
2. **The pilot checkpoint passes on the test split (+31.2 %) and fails on the validation split (−21.9 %).** The verdict of the dynamics gate flips between two held-out splits of ten episodes each, for one and the same checkpoint.
3. Both checkpoints pass on the train split (+26.6 % to +32.0 %), which is in-sample.
4. Both checkpoints were selected by their own validation RMSE (steps 1400 and 1000), so the validation failure is not a bad checkpoint choice; if anything, selecting on this split makes its numbers optimistic.
5. On validation the models beat persistence only at the longest steps (step 4 for all three rows, step 3 by under one percent for v2) and lose heavily at steps 1 and 2. On train and test they beat it from step 2.
6. Persistence is easiest on the validation split (RMSE 0.1886, against 0.2101 on train and 0.2262 on test) and the pilot model's error is highest there (0.2298, against 0.1429 and 0.1556).

**Hypotheses (not shown), with the diagnostic that would test each.**

- **H-A: with ten held-out episodes per split, the pass or fail is dominated by which episodes are held out.** Test: predict every episode while it is held out (episode-level k-fold over all 100 episodes) and report the pooled relative improvement with an episode-cluster bootstrap interval, plus the per-episode spread.
- **H-B: the validation episodes contain more near-static motion, which persistence predicts well.** Test: per-episode persistence RMSE and model-to-persistence ratio; this diagnostic saved only aggregates, so it needs per-window or per-episode predictions saved.
- **H-C: the pilot anchors' 0.002 to 0.004 offset comes from batch-mean weighting in the pilot's loop.** Test: re-score the pilot checkpoint with the pilot's weighting and see whether 0.22893 and 0.22631 come back.
- **H-D: the v2 regime (`single_choice` masks, `base` prediction mode) generalises better to validation than the pilot's.** It is one seed each and the same ten episodes, so this run cannot support it; it is listed only because the direction is consistent in all four validation rows.

## 5. Design issues found

- **B2's gate will likely block most or all seeds as designed.** Pilot v2 gates each seed on the validation split with a 2 % margin and blocks allocator training for a seed that fails. One seed of the v2 regime is at −12.7 % and the pilot regime at −21.9 % on that split. If the other four seeds behave alike, the five-seed run would end with every seed `NOT_EVALUATED` and no H2 comparison. This is an expectation from two checkpoints, not a measurement of the other seeds.
- **The gate is a point comparison on ten episodes**, with no uncertainty interval, and a single split decides it.
- The failed-gate report path of the pilot v2 notebook still crashes when no seed passes (see the probe audit); it is not fixed.

## 6. What this run supports and does not support

- **Supports:** the dynamics gate's verdict depends on the split: the pilot checkpoint passes on test and fails on validation. The validation failure seen in the probe is shared by the pilot's own checkpoint. The diagnostic's pipeline reproduces the probe's stored numbers exactly and the pilot's to within 1.6 %. BF16 and FP32 agree to 8e-05 in RMSE.
- **Does not support:** any statement about why the splits differ (H-A to H-D are untested); any statement about H2 or the allocator; that the dynamics model "works" or "does not work" in general; that the v2 regime is better than the pilot's (one seed each).
- **Contradicts earlier writing.** The README status row "Dynamics beat persistence on held-out episodes ✅ pass (−31 %)" and the line "the real-data dynamics model works" hold for the **test split only**. On the validation split the same checkpoint fails by 22 %. The pilot findings note (`2026-09-29-droid100-pilot-findings.md` §2.1, "Dynamics: pass" and "Supports: … beats persistence on DROID-100 held-out episodes at horizon steps 2–4 (single seed)") is correct for the test split and silent on validation. The README has been updated to say so; the dated note is not edited.

## 7. Next steps

- **Recommended, not started (needs a go-ahead):** an episode-level k-fold dynamics study (roadmap E3.2's question, asked over all 100 episodes) with a decision rule frozen before it runs. Measured cost basis: the probe's 1500-step dynamics job took 496.1 s on the L4 (`DONE.json`), so five folds are about 41 minutes of training (5 × 496 s) plus evaluation. It answers H-A directly and tells B2 whether its dynamics gate can be met at all.
- **Hold the five-seed B2 (E2.1 / E2.2)** until the gate's design is decided; that is a protocol decision (and a deviation-log entry), not mine to make.
- **If another diagnostic is run first,** save per-episode predictions (H-B) and re-score the pilot with its own weighting (H-C).

## Appendix: the same rows in FP32

| Checkpoint | Mode | Split | Windows | Model RMSE | Persistence RMSE | Relative improvement | Passes 2 % |
|---|---|---|---:|---:|---:|---:|---|
| v1 | base | train | 5910 | 0.1497 | 0.2101 | +0.288 | yes |
| v1 | base | validation | 796 | 0.2325 | 0.1886 | -0.233 | no |
| v1 | full | train | 5910 | 0.1428 | 0.2101 | +0.320 | yes |
| v1 | full | validation | 796 | 0.2298 | 0.1886 | -0.219 | no |
| v2 | base | train | 5910 | 0.1542 | 0.2101 | +0.266 | yes |
| v2 | base | validation | 796 | 0.2124 | 0.1886 | -0.126 | no |
| v2 | full | train | 5910 | 0.1523 | 0.2101 | +0.275 | yes |
| v2 | full | validation | 796 | 0.2121 | 0.1886 | -0.125 | no |
