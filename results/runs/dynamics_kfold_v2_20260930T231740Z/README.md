# Run `dynamics_kfold_v2_20260930T231740Z` (episode-level 5-fold dynamics study, v2 regime)

- **What it is:** 5-fold cross-validation of AdjointRWM dynamics against persistence over all valid DROID-100 episodes (99 windowed episodes, 7,541 windows) with 10 inner-validation episodes per fold and 70 training episodes per fold. Regime: `v2` (`mask_mode='single_choice'`, `prediction_mode='base'`). Primary checkpoint: selected by inner validation (`best`). Frozen plan: `docs/plans/dynamics-kfold-plan.md`.
- **Status:** `status: OK`, `anchor: within tolerance` (recomputed 0.2127 vs. stored 0.2125, difference 0.0003 < tolerance 0.01; probe run `dynamics_gate.json`, seed 0, validation, mode `base`). `COMPLETE` written.
- **Where it ran:** Colab, NVIDIA L4, session `l4-worker` (`gpu-l4-s-kkb-ass1a0-14lucaqeju87i`; commit `6488aac14ef5af398ce5b5cd2bf58b0a44b7d0f3`; Python 3.13.15, torch 2.11.0+cu128).
- **Outcome (`reports/acceptance_report.json`):**
  - **Primary estimate (`best|base`):** Relative improvement $R = -0.2208$ ($-22.1\%$), 95 % bootstrap CI $[-0.5510, +0.0528]$.
  - **Classification:** **`INCONCLUSIVE`** (95 % interval $[-0.5510, +0.0528]$ contains the margin $+0.02$).
  - **Gate reliability:** Random 10-episode split pass rate at 2 % margin is **$36.4\%$** (10,000 draws).
  - **Episode win rate:** The model beats persistence on **$70.7\%$** of held-out episodes.
  - **Per-fold variance:** Fold 0: $-60.8\%$, Fold 1: $+0.6\%$, Fold 2: $+24.7\%$, Fold 3: $-29.0\%$, Fold 4: $-12.7\%$.
  - **By horizon step:** step 1: $-107.6\%$, step 2: $-35.4\%$, step 3: $-8.0\%$, step 4: $+6.5\%$.

| Hash | Value |
|---|---|
| Config SHA-256 | `57ef3cdce398a94d1fdc0fff8846de1eca01b349c7ef4720257f4679cb78e1cc` (`config_sha256` in `config/run_config.json`; equals `config_hash` in `reports/acceptance_report.json`) |
| Package commit | `6488aac14ef5af398ce5b5cd2bf58b0a44b7d0f3` |
| Anchor checkpoint | Probe run `droid100_adjoint_v2_20260930T165409Z` seed 0 `dynamics_gate.json` (stored `0.2124711301360882`) |

## Files

Copied byte-for-byte from the Colab Drive run directory `MyDrive/Colab Notebooks/AdjointRWM_Production/runs/dynamics_kfold_v2_20260930T231740Z/` via `colab download`.

| Path | Bytes | SHA-256 | Origin |
|---|---:|---|---|
| `COMPLETE` | 33 | `c251d7f26447f1882d00c256695ac042dd5142955737af7046fa5c76ea3f6c73` | Written on completion of scoring |
| `config/run_config.json` | 1786 | `947ee3218251934f364d182d97d20749730dd81f0d232fd2a0dbf669486404cf` | Frozen run configuration |
| `config/data_manifest.json` | 11101 | `f70b3427a98ecb586c8c5882c3d99655490db3d4133123fde6c6582fbccdee8d` | Episode assignments to folds |
| `artifacts/units/v2_anchor.json` | 768 | `dfaf719b6255de932803d0e1596507177516ccaff5e69aac32324453aceb0117` | Anchor unit summary and validation score |
| `artifacts/units/v2_0.json` | 37854 | `e014b10934001587b8f615781020b5528b443bce234bc3347138df296a6ca2b0` | Fold 0 evaluation tables and scores |
| `artifacts/units/v2_1.json` | 37846 | `41f8c8a89da600e8a2494aa5222e4a54c228daee5e62dd80a11ef894c9277062` | Fold 1 evaluation tables and scores |
| `artifacts/units/v2_2.json` | 38040 | `eff7e0a034d44582cac133b80bed6a64a04f2260fc98be249a36cf07d81b455b` | Fold 2 evaluation tables and scores |
| `artifacts/units/v2_3.json` | 36075 | `040d90e9c5ec57ef3671813f789b443c516dd03931a7f31f55612a35ca3a1a14` | Fold 3 evaluation tables and scores |
| `artifacts/units/v2_4.json` | 37884 | `6c31647fb2285dc35078f435bf31051ed693607fafd104ee01d10afc571f49af` | Fold 4 evaluation tables and scores |
| `artifacts/episode_errors.csv` | 76844 | `a2fc4436631c0e3d54ceeba13b4baac8f2ad05b21704e1563c0ece1120d2066f` | Per-episode model and persistence errors |
| `reports/acceptance_report.json` | 17954 | `07c15e46d45e6c42c28c8e63a19ff14e23eb7cbaf1c0974b99213ab1eeb76879` | Full acceptance report with bootstrap estimates |
| `reports/run_summary.md` | 1255 | `1c2bff7d643232983decb8decd502482ea07ef4b62dfb1a8b9922185f3245f60` | Formatted summary table and status |
| `logs/train_log_v2_anchor.jsonl` | 24348 | `27434b25ed7b85df6a7141d3c017e19772c446836e0f4b8d5a2a08ad03f6467b` | Training log for anchor unit |
| `logs/train_log_v2_0.jsonl` | 24368 | `cc02573b9357596f3a4380cdf6e787106824d16cc6ee57cc3892b508e56f26e3` | Training log for fold 0 |
| `logs/train_log_v2_1.jsonl` | 24340 | `13c72d31f07b731b43e5d4822eb2cfe043f1d50a8fe55ff30a1741388373197e` | Training log for fold 1 |
| `logs/train_log_v2_2.jsonl` | 24338 | `85fa39f5b8e41aa9eb2716e0274e8c1a3004766e6163449e5263023a38387fa4` | Training log for fold 2 |
| `logs/train_log_v2_3.jsonl` | 24369 | `14459c1f420a2ab97f848984d5893300344a37ea2bd971ce90dc8381b280514e` | Training log for fold 3 |
| `logs/train_log_v2_4.jsonl` | 24357 | `f89edd9e8b7a1c3599f239ea7f6f80699577adba09db175a7dbe53f8f5208ba1` | Training log for fold 4 |

## Checkpoints

Per plan §6, checkpoints stayed on the VM's local ephemeral disk and were deleted after scoring each unit to manage storage quota. Best checkpoint SHA-256 values recorded in unit JSON files:

| Unit | Best Step | Best Val Score | Checkpoint SHA-256 |
|---|---:|---:|---|
| anchor | 1400 | 0.2127 | `e1c71d1d74807ff99a7d825ae82f26e80fed8d22c709681571b64275f054fa16` |
| fold 0 | 1400 | 0.1920 | `635e3053e0e2dd77ff275568b771f40762344ab382bff5261bb4ca38332cf322` |
| fold 1 | 1400 | 0.3496 | `82d2f2dc8dd19bdd90d69fc85038c2da252e5691b9630a869a7d1412a20978b5` |
| fold 2 | 1400 | 0.3826 | `b475a4ec7d83f1c2544f3f192563b09b60c43398c9e36064017bf63da62fe3fc` |
| fold 3 | 1200 | 0.3651 | `97eaae7158311f158eb18abd2351e3b1ee1648b49e267d2fb42cd4dca50d8da4` |
| fold 4 | 1500 | 0.3758 | `2e33a01d3d9f5e7b50bc65752fe7f6976343ed49f4d19a76b091dd0f43658cb6` |
