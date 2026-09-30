# Audit: one-seed B2 probe `droid100_adjoint_v2_20260930T165409Z` (pilot v2, seed 0)

**Date:** 2026-09-30. **Classification:** `RUN_STATUS: INCOMPLETE` (no `acceptance_report.json`, no `COMPLETE`), `DYNAMICS_GATE (seed 0, validation): FAIL`, `DATA_CONTRACT: PASS`, `H2_EVIDENCE: NONE`.

## Subject

The queue job `b2-probe-seed0-r3` (Drive `jobs/results/b2-probe-seed0-r3/`): `notebooks/01-production/AdjointRWM_Production_Pilot_v2.ipynb` at commit `e738c786805620f1b8e650f58351308875b9c436`, override `seeds=[0]`, on an NVIDIA L4 under worker commit `01d2e30fadc9ff35498f7eb09c59c7dcf47e4bd8`. It started 2026-09-30 16:53:56 UTC and its result was written 17:03:57 UTC (status `failed`). The run directory on Drive holds 120 files.

## How these facts were obtained

Each value below was read in this session through the Google Drive connector, as text, from the file named. The connector returns a text rendering, not the bytes, so this is **not** a byte-for-byte import and nothing was added to `results/runs/`. A formal import needs the run folder copied from Drive (`/import-run`).

| Fact | Value | Drive file |
|---|---|---|
| Dynamics gate, seed 0, split `validation`, `prediction_mode` `base` | model RMSE 0.2124711301360882, persistence RMSE 0.18857847256011034, relative improvement −0.12669875437856223 (recomputed: (persistence − model) / persistence = −0.12669875437856223), required margin 0.02, `passed: false` | `artifacts/dynamics_gate.json` |
| Dynamics training | `DONE`; 1500 steps in 496.09 s (3.02 steps/s); best validation score 0.2124711301360882 at step 1000; peak VRAM 0.497 GiB | `jobs/seed_0/dynamics/DONE.json` |
| Hashes | config `c8cd4bf281d6e91159e0d5a16cf9dcf8bd530ed20adeaca9e02f4c92de11b6f8`, data manifest `3ae7e05cac8dccc7c69e5a76e228caeb02d07876d237a5b00712ba1470d6e59f`, source `ef7989fc4b44571361b5608a4e30ba8c203d91934a19e9560ddc57cd7740df3b`, best checkpoint `66c80f464a442da0ba9e3ee424aef692a1be4b76eff860fe1d8640c08555882b` | `DONE.json` |
| Data contract | `status: PASS`, 100 valid episodes, splits 80 / 10 / 10, disjointness verified, real image payloads verified, `synthetic_fallback_used: false`, split parity with the pilot `matches: true` | `data_audit.json` |
| Allocator stage | `gate_class_balance.json` is `{}`: no seed reached allocator training | `artifacts/gate_class_balance.json` |
| Failure | `ValueError: No objects to concatenate` at `ALL_TRACES = pd.concat(TRACES, ignore_index=True)` (notebook section 6); `runs[0].complete: false` | `result.json` |

## What the run shows

- For seed 0, with the pilot v2 configuration, the dynamics model's validation RMSE (0.2125) is **higher** than persistence's (0.1886): 12.7 % worse, against a required margin of 2 % better. The gate failed. A failed gate is a result and is reported as such; the seed is not replaced and the settings are not changed to get a pass.
- The data contract held: the real DROID-100 episodes were read and the split is the same as the pilot's.

## What it does not show

- **Nothing about H2.** Allocator training is blocked for a seed whose dynamics gate fails (notebook section 5: "reported, not replaced"), so there are no traces, no opportunity gate and no adjoint or critic comparison.
- **Nothing about other seeds.** One seed.
- **No comparison with the pilot's dynamics result is justified yet.** The pilot (`docs/research-notes/2026-09-29-droid100-pilot-findings.md` §2.1) reported a gate mean RMSE of 0.1556 against persistence 0.2262 (+31.2 %), single seed, from a different training run and, as far as the record shows, a different split: the note's table does not name its split, the roadmap (E2.1 entry) says pilot v2 evaluates the gate on validation instead of test, and the persistence RMSE differs (0.2262 there, 0.1886 here), which a change of split would explain but which has not been checked. The two numbers have not been put on the same split, and no cause of the difference is established. The pilot's logged throughput was about 22 steps/s and this run's 3.02 steps/s; that is also unexplained.

## Why the run crashed after a failed gate

By design a failed gate leaves `TRACES` empty, and the notebook then calls `pd.concat` on the empty list (section 6), so the run ends before the report. This is a defect in the failed-gate path of the notebook, not in the gate: later cells also assume traces exist (`test_hold_gain_is_exactly_zero` asserts `len(traces)`, and the report counts that test toward `COMPLETE`). The production pilot is not edited in place (`notebooks/AGENTS.md`); a fix would be a new notebook version with its changes listed in `CHANGELOG.md`, and it changes what a run with no passing seed reports, so it needs a decision first.

## Open

1. Copy the run folder from Drive for a formal import, or accept this audit as the record of the gate value.
2. Decide what to do about a failed dynamics gate before any five-seed B2 spend: (a) investigate the difference from the pilot on the same split, (b) fix the failed-gate report path in a new notebook version, (c) both. This audit recommends not starting the five-seed run until (a) is done, because a seed that fails the gate cannot contribute to the H2 comparison.
