# Hybrid Adjoint Recursive World Model (HARP): Dynamics on E3.1 Stratified Shard

**Date:** 2026-10-02 · **Execution Time:** 2054.2 s · **Device:** NVIDIA L4
**Manifest:** `cache_manifest.json` · **Test Split:** 50 held-out episodes, 4,154 windows across 12 robotics laboratories
**Architecture:** HybridAdjointRWM Kinematic-Residual Transformer (24,731,472 parameters) trained on 399 multi-site episodes

---

## 1. Executive Summary & Findings

The Hybrid Adjoint Recursive World Model (HARP architecture) combines an analytical kinematic state-action continuation base with a 6-layer Transformer + GRU neural residual dynamics network.

### Primary Results:
- **Test Proprioception RMSE (Mean ± Std over 2 seeds):**
  - **Hybrid Adjoint RWM:** 0.2652 ± 0.0012
  - **Persistence Baseline:** 0.2021
  - **In-Domain Ridge Forecaster:** 0.1102
- **Paired Comparisons (5,000 cluster bootstrap resamples):**
  - **vs Persistence:** +31.27% (95% CI [+9.22%, +52.10%]) -> `rival_better`
  - **vs In-Domain Ridge:** +140.68% (95% CI [+105.99%, +167.29%]) -> `rival_better`

---

## 2. Cross-Site Performance Across 12 Laboratories

| site    |   num_episodes |   num_windows |   hybrid_adjoint_rwm_rmse |   persistence_rmse |   ridge_rmse |
|:--------|---------------:|--------------:|--------------------------:|-------------------:|-------------:|
| ILIAD   |              6 |          1024 |                  0.285272 |           0.199379 |    0.116131  |
| TRI     |             15 |           781 |                  0.249232 |           0.221457 |    0.100395  |
| AUTOLab |              4 |           512 |                  0.234447 |           0.242981 |    0.0985275 |
| IRIS    |              5 |           500 |                  0.439175 |           0.263995 |    0.148687  |
| IPRL    |              6 |           376 |                  0.296426 |           0.204964 |    0.123159  |
| RPL     |              1 |           197 |                  0.119179 |           0.131679 |    0.0679651 |
| PennPAL |              3 |           185 |                  0.263848 |           0.170817 |    0.141651  |
| BVL     |              3 |           183 |                  0.192472 |           0.17576  |    0.077815  |
| RAD     |              1 |           144 |                  0.116322 |           0.108881 |    0.0536747 |
| REAL    |              2 |            96 |                  0.299664 |           0.215979 |    0.112212  |
| WEIRD   |              1 |            80 |                  0.196998 |           0.214228 |    0.10755   |
| RAIL    |              3 |            76 |                  0.255743 |           0.246375 |    0.115739  |

---

## 3. Inference Latency on NVIDIA L4

- **Batch-1 p50 Latency:** 7.80 ms
- **Batch-1 p95 Latency:** 8.02 ms
- **Batch-128 p50 Latency:** 8.42 ms
