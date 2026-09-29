# Audits

These are independent checks of runs and notebooks, kept so that nothing gets reused or cited by mistake. Audits are append-only: a new finding goes in a new dated file, and existing files aren't rewritten.

| Date | Audit | Subject | Classification |
|---|---|---|---|
| 2026-09-28 | [`2026-09-28_run_v2_ailerons_audit.md`](2026-09-28_run_v2_ailerons_audit.md) | `AdjointRWM_Production/Run_V2` (archive SHA-256 `7c2bd647…00edd8`) | `PIPELINE_EXECUTION: PARTIAL_PASS`, `WORLD_MODEL_OBJECTIVE: FAIL`, A100 deferred |
| 2026-09-28 | [`2026-09-28_swm_pilot_synthetic_surrogate_audit.md`](2026-09-28_swm_pilot_synthetic_surrogate_audit.md) | `swm_pilot_checkpoint.pt` "DROID subset" (checkpoint SHA-256 `b0e61713…943656`) | `PIPELINE_SMOKE_TEST_PASSED`, `REAL_DATA_TRAINING_FAILED`, promotion blocked |
| 2026-09-29 | [`2026-09-29_legacy_notebook_audit.md`](2026-09-29_legacy_notebook_audit.md) | `para_0_0_1`, signal-filtering benchmark, phase logs N3–N10, paper draft | Non-evidential |

The first two audits were written outside this repo and imported verbatim. They independently confirm the problems the third audit found in the SWM and `para_0_0_1` line of work. The pseudo-random "DROID" data, the hard-coded promotion metrics, and the co-state, critic and gate heads trained to constants all appear in both.

The comprehensive research plan (section "Current evidence status and scientific-integrity gate") separately audits the T4 `Untitled0.ipynb` P0 diagnostics. There, the reported 17.39 % adjoint advantage is fixed by the ratio `0.95 / 1.15` before any allocation decision is made.
