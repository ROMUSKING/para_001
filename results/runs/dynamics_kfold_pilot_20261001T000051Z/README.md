# Run `dynamics_kfold_pilot_20261001T000051Z` (episode-level 5-fold dynamics study, pilot regime)

- **What it is:** 5-fold cross-validation of AdjointRWM dynamics against persistence over all valid DROID-100 episodes (99 windowed episodes, 7,541 windows) with 10 inner-validation episodes per fold and 70 training episodes per fold. Regime: `pilot` (`mask_mode='subset'`, `prediction_mode='full'`). Primary checkpoint: selected by inner validation (`best`). Frozen plan: `docs/plans/dynamics-kfold-plan.md`.
- **Status:** `status: OK`, `anchor: within tolerance` (recomputed 0.2133 vs. stored 0.2263, difference 0.0130 < tolerance 0.02; pilot checkpoint `best_metrics`, validation, mean of `full_rmse_by_horizon`). `COMPLETE` written.
- **Where it ran:** Colab, NVIDIA L4, session `l4-worker` (`gpu-l4-s-kkb-ass1a0-14lucaqeju87i`; commit `6488aac14ef5af398ce5b5cd2bf58b0a44b7d0f3`; Python 3.13.15, torch 2.11.0+cu128).
- **Outcome (`reports/acceptance_report.json`):**
  - **Primary estimate (`best|full`):** Relative improvement $R = -0.2207$ ($-22.1\%$), 95 % bootstrap CI $[-0.5471, +0.0511]$.
  - **Classification:** **`INCONCLUSIVE`** (95 % interval $[-0.5471, +0.0511]$ contains the margin $+0.02$).
  - **Gate reliability:** Random 10-episode split pass rate at 2 % margin is **$36.3\%$** (10,000 draws).
  - **Episode win rate:** The model beats persistence on **$70.0\%$** of held-out episodes.
  - **Per-fold variance:** Fold 0: $-59.8\%$, Fold 1: $+0.9\%$, Fold 2: $+23.2\%$, Fold 3: $-29.0\%$, Fold 4: $-13.9\%$.
  - **By horizon step:** step 1: $-108.7\%$, step 2: $-35.5\%$, step 3: $-7.7\%$, step 4: $+6.8\%$.
- **Comparison to `v2` (`scripts/dynamics_kfold_compare.py`):**
  - Paired difference $R(\text{v2}) - R(\text{pilot}) = -0.000$ (95 % bootstrap CI $[-0.007, +0.008]$).
  - Label: **`NOT_DISTINGUISHED`**. The two regimes are virtually identical in held-out generalization.

| Hash | Value |
|---|---|
| Config SHA-256 | `8d503e1aac5accbe1f4ad65e6e84510edba117206d902e12944d9b90aa07ab6a` (`config_sha256` in `config/run_config.json`; equals `config_hash` in `reports/acceptance_report.json`) |
| Package commit | `6488aac14ef5af398ce5b5cd2bf58b0a44b7d0f3` |
| Anchor checkpoint | Pilot run `droid100_adjoint_20260929T070629Z` `best_dynamics.pt` (stored validation full RMSE `0.22631007805466652`) |

## Files

Copied byte-for-byte from the Colab Drive run directory `MyDrive/Colab Notebooks/AdjointRWM_Production/runs/dynamics_kfold_pilot_20261001T000051Z/` via `colab download`.

| Path | Bytes | SHA-256 | Origin |
|---|---:|---|---|
| `COMPLETE` | 33 | `6d3a8821f0850318bc67cebe02ef33d1bd49485999228c98f86653156b022761` | Written on completion of scoring |
| `config/run_config.json` | 1792 | `87fe85b90aeb0af82dfcf843f5682dc619f90c374d393aedb9e445e66647d5e8` | Frozen run configuration |
| `config/data_manifest.json` | 11101 | `f70b3427a98ecb586c8c5882c3d99655490db3d4133123fde6c6582fbccdee8d` | Episode assignments to folds |
| `artifacts/units/pilot_anchor.json` | 776 | `3d847d7f84b187dd8488b67e4598a906023f89afd8c7cfe724def27ed728ed2b` | Anchor unit summary and validation score |
| `artifacts/units/pilot_0.json` | 37855 | `b8abe15f1a363a562104e331bc69b526f16efd5926902336f5c126ba5c2cf740` | Fold 0 evaluation tables and scores |
| `artifacts/units/pilot_1.json` | 37849 | `b2079c3599280e3064ceee5eb893570067cfc94660689c9e93bed997be17b94b` | Fold 1 evaluation tables and scores |
| `artifacts/units/pilot_2.json` | 38058 | `c04dc0a5d5053eb2972535a55a2d2c187c3ad5d06820f57d84e6fd6039b43b91` | Fold 2 evaluation tables and scores |
| `artifacts/units/pilot_3.json` | 36070 | `4297c00e4a96a4028e105e343e9639c991241bbb7f66ef4a7b5b1fceec53f33d` | Fold 3 evaluation tables and scores |
| `artifacts/units/pilot_4.json` | 37876 | `a4b92da2640e765351e1663b5e28e3719fd9dbe33f4a83e4a4383f346ca41a77` | Fold 4 evaluation tables and scores |
| `artifacts/episode_errors.csv` | 77997 | `2bd356f010ad4fcd31008bbf3cd21ce6804130eb72eb8eb82e7a41c38212b29c` | Per-episode model and persistence errors |
| `reports/acceptance_report.json` | 18026 | `1dd11eb734cd0c89844864a90c327b7592e598e4ff8d8e0a02da78931aaf2b7b` | Full acceptance report with bootstrap estimates |
| `reports/run_summary.md` | 1273 | `beef503ef46d93a50d02fe47dbfbf6e42aee01bdaf175336665ceaf3bfdd1660` | Formatted summary table and status |
| `logs/train_log_pilot_anchor.jsonl` | 24365 | `0737fadd1b62b558efedb5855f5b3266373c29f585ba54ecc67bd42f880d4604` | Training log for anchor unit |
| `logs/train_log_pilot_0.jsonl` | 24367 | `4691991c002640846ec25ddc8b45438a47910eba5a717a5eebe1479cce17d158` | Training log for fold 0 |
| `logs/train_log_pilot_1.jsonl` | 24308 | `9208278e47394c1ff35ff60adcb354fbc6c70012b6d557e04261f95549b26fb3` | Training log for fold 1 |
| `logs/train_log_pilot_2.jsonl` | 24315 | `71707d08479217017266e600e4a01fd4be56c9d4c45372ffa82fdd9cdf12c297` | Training log for fold 2 |
| `logs/train_log_pilot_3.jsonl` | 24305 | `2e807c63ee8d5054007f738f12282d876bb02cd02d79d6a0e8bdaa92c5e06921` | Training log for fold 3 |
| `logs/train_log_pilot_4.jsonl` | 24306 | `ec3bcc2cda1a0ca1d3e6fa37b851ac38394bb92e31a0a20df2ad4f3cc74db32c` | Training log for fold 4 |

## Checkpoints

Per plan §6, checkpoints stayed on the VM's local ephemeral disk and were deleted after scoring each unit to manage storage quota. Best checkpoint SHA-256 values recorded in unit JSON files:

| Unit | Best Step | Best Val Score | Checkpoint SHA-256 |
|---|---:|---:|---|
| anchor | 1400 | 0.2133 | `1140a32b992f5130772a705f21de21ae970294d8bacad3fc6e33d7966eac4a9f` |
| fold 0 | 1400 | 0.1906 | `ca62feabd76d189fad2c74b22276f0a338ebec944765c651283d7c56fe9ec215` |
| fold 1 | 1200 | 0.3468 | `2c3068cecc2ee4bd1470823fde31335fae1b7f8b228b63a4188e16390c6ec407` |
| fold 2 | 1400 | 0.3827 | `1856cbc6f7e14f930542b1c0da198fd6c388dc0585b0e4ea247df964105f3a4a` |
| fold 3 | 1500 | 0.3650 | `894ca70de41bd4bffba41cc76a3450aeb9acb92e960b200bf69abbbebf86b194` |
| fold 4 | 1200 | 0.3703 | `f5cb0cb8ef233fa347d08a9954a880006c7acebc8e5937d0b8147a965f294eb6` |
