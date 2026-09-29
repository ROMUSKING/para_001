# Audit of pre-pilot notebooks, phase logs and paper draft

**Date:** 2026-09-29. **Scope:** material on Google Drive from 25–29 Sep 2026 that predates or runs alongside the DROID pilot.

**Verdict:** apart from the production pilot (see `2026-09-29-droid100-pilot-findings.md`), none of the numbers in this material can be used as evidence for the architecture. Several are hard-coded, derived from other numbers by formula, or computed on data that isn't what its label says. The material is kept in the repo, under `notebooks/archive/` and `results/legacy/`, so the history is traceable. Each item is marked so that nobody cites it by accident.

## 1. `para_0_0_1.ipynb` (Specialist World Model / "SWM")

**Status: archive, non-evidential.** The repo copy is rebuilt from a text export, so code is kept but outputs are not. The original, with outputs (1.3 MB), stays on Drive; the file ID is in `docs/DRIVE_INVENTORY.csv`.

| Issue | Where | Consequence |
|---|---|---|
| Promotion-report metrics are typed-in constants, e.g. `projected_regret = 0.0475`, `task_success_rate = 94.1` (and 98.4 % in a later cell) | "Execute Training and Run SWM Hardware Promotion Report" cells | Reported "task success" is not measured |
| The "real telemetry" is the OpenML F-16 **Ailerons** regression dataset, relabelled as flight/robot telemetry | "Fetching Real-World Telemetry Data (OpenML)" | Not a world-model or robotics benchmark |
| The "DROID subset" is 20 deterministic synthetic episodes, and images are replaced by hash strings | "Generate 20 deterministic episodes representing a DROID subset" | Not DROID data |
| Regret is derived from MSE by a constant, not from any allocation decision | Evaluation cell | "Regret" doesn't measure allocation |
| Some training targets are random tensors or placeholder projections | Production loop cells | Training signal is partly noise |
| Cell 86 is truncated mid-`print(f…` in the export | — | The archived notebook won't run as-is |

The `swm_*_checkpoint.pt` files on Drive (4 × ~286 MB) come from this notebook. They are listed in the inventory but not copied into git and are not used anywhere.

## 2. `General_Signal_Filtering_End_to_End_Benchmark` (versions 1–6)

**Status: archive, sanity check only.** Only a truncated source excerpt (through model definitions) is in the repo, at `notebooks/archive/General_Signal_Filtering_Benchmark_PARTIAL.md`. The latest full notebook (v6, modified 2026-09-29) is on Drive.

The benchmark denoises synthetic signals in three families (sinusoids, step "shocks", AR(1) drift). The comparison is tilted towards "Ours":

- **Unequal information.** "Ours" receives a domain-ID embedding and a bilateral filter front end. The "JEPA" baseline gets neither. This violates the plan's information-matching contract.
- **Baselines are mislabelled or weak.**
  - The "EKF" is a random-walk Kalman filter with no dynamics model and untuned q/r, so it isn't an extended filter.
  - The "Wavelet BayesShrink" filter is a one-level Haar filter with a fixed threshold factor.
  - The "JEPA" is a supervised denoiser with no embedding-prediction target.
- **The "adjoint head" isn't a co-state.** It is a sigmoid saliency score with no ∂J/∂s supervision. The research plan (§15.2) explicitly says not to call such a head a co-state.
- **Capacity is mismatched.** "Ours" and the JEPA are about 5 M-parameter transformers. The EKF, wavelet and moving-average baselines have essentially no learned parameters.
- **One seed, 240 steps.** No confidence intervals are possible.

## 3. `StageA_Deep_Neural_Adjoint_World_Model*.ipynb` (8 copies, 26–28 Sep) and `Stage0_LQTree_Benchmark*.ipynb` (6 copies, 25–26 Sep)

**Status: not reviewed.** These notebooks are listed in the inventory only. Stage 0 (LQTree) is the plan's first hard gate: analytic co-state correctness, candidate effects, coding, determinism. It is the right place to restart the confirmatory sequence (see the roadmap), so the latest Stage 0 notebook should be reviewed and brought into `notebooks/00-stage0/` next.

## 4. Phase logs N3, N8, N9, N10 (`results/legacy/phase_logs/`)

**Status: non-evidential.** Each file reports `"verification_status": "SUCCESS"`, but none contains the evidence the plan requires.

| File | What it contains | Why it isn't evidence |
|---|---|---|
| `phase_n3_optimization_run.json` | One final MSE (1.92e-4) for B=16, N=8, D=32 and `causal_taint_check: CLEAN` | No task, split, baseline, seed or data provenance |
| `phase_n8_deployment_assurance.json` | One "normal" loss and one "anomalous" loss of 9824.9 → "ANALYTICAL_RESCUE" | A single injected anomaly; a fallback firing once isn't an assurance result |
| `phase_n9_replication_release.json` | Marks N3 and N8 as "verified" | Self-referential; nothing was re-run independently |
| `phase_n10_embodied_telemetry.json` | One telemetry step: control loss 8.35e-5, 250 bits, on an "A100-SXM4-80GB" in float64 | One step on synthetic state; no embodiment or task |

`optimized_adjoint_teacher.pt` (70 KB, SHA-256 `507e0669…d70e`) is the Phase N3 artefact. The production pilot records it as a "read-only audit asset; no semantic compatibility assumed", and it should stay that way.

## 5. Paper draft

See `papers/drafts/REVIEW.md` for the claim-by-claim check. In short: the draft's central empirical claim, that the adjoint allocator systematically outperforms direct baselines, is unsupported, and the only rigorous run to date contradicts it.
