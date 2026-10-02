# Milestone E3.2: Dynamics Pilot on E3.1 Stratified Shard

**Date:** 2026-10-02 · **Execution Time:** 880.5 s · **Device:** NVIDIA L4
**Manifest:** `cache_manifest.json` · **Test Split:** 50 held-out episodes, 4,154 windows across 12 robotics laboratories
**Architecture:** AdjointRWM Transformer-Adjoint (24,731,164 parameters) trained on 399 multi-site episodes

---

## 1. Executive Summary & Findings

AdjointRWM was trained directly on the 399 training episodes of the E3.1 multi-site shard (same-data protocol) with native normalisation.

### Primary Results:
- **Test Proprioception RMSE (Mean ± Std over 2 seeds):**
  - **AdjointRWM:** 0.2586 ± 0.0049
  - **Persistence Baseline:** 0.2021
  - **In-Domain Ridge Forecaster:** 0.1102
- **Paired Comparisons (5,000 cluster bootstrap resamples):**
  - **vs Persistence:** +28.01% (95% CI [-31.42%, +82.34%]) -> `inconclusive`
  - **vs In-Domain Ridge:** +134.70% (95% CI [+29.60%, +221.17%]) -> `rival_better`

---

## 2. Cross-Site Performance Across 12 Laboratories

| site    |   num_episodes |   num_windows |   adjoint_rwm_rmse |   persistence_rmse |   ridge_rmse |
|:--------|---------------:|--------------:|-------------------:|-------------------:|-------------:|
| ILIAD   |              6 |          1024 |          0.459025  |           0.199379 |    0.116131  |
| TRI     |             15 |           781 |          0.129138  |           0.221457 |    0.100395  |
| AUTOLab |              4 |           512 |          0.135507  |           0.242981 |    0.0985275 |
| IRIS    |              5 |           500 |          0.211419  |           0.263995 |    0.148687  |
| IPRL    |              6 |           376 |          0.133234  |           0.204964 |    0.123159  |
| RPL     |              1 |           197 |          0.0780649 |           0.131679 |    0.0679651 |
| PennPAL |              3 |           185 |          0.153413  |           0.170817 |    0.141651  |
| BVL     |              3 |           183 |          0.0847567 |           0.17576  |    0.077815  |
| RAD     |              1 |           144 |          0.127239  |           0.108881 |    0.0536747 |
| REAL    |              2 |            96 |          0.12251   |           0.215979 |    0.112212  |
| WEIRD   |              1 |            80 |          0.120912  |           0.214228 |    0.10755   |
| RAIL    |              3 |            76 |          0.123086  |           0.246375 |    0.115739  |

---

## 3. Inference Latency on NVIDIA L4

- **p50 Latency:** 8.06 ms
- **p95 Latency:** 8.51 ms
- **Mean Latency:** 8.07 ms
