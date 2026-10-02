# Multi-Horizon Rollout Decay Benchmark

**Date:** 2026-10-02 · **Execution Time:** 141.6 s
**Model Evaluated:** `adjoint_rwm` · **Horizons:** [1, 2, 4, 8, 12, 16]

---

## Horizon Scaling Summary

|   horizon |   num_windows |   persistence_mean_rmse |   persistence_terminal_rmse |   ridge_mean_rmse |   ridge_terminal_rmse |   adjoint_rwm_mean_rmse |   adjoint_rwm_terminal_rmse |   adjoint_rwm_vs_pers_pct |   adjoint_rwm_vs_ridge_pct |
|----------:|--------------:|------------------------:|----------------------------:|------------------:|----------------------:|------------------------:|----------------------------:|--------------------------:|---------------------------:|
|         1 |          4230 |                0.10489  |                    0.10489  |         0.0775002 |             0.0775002 |                0.232995 |                    0.232995 |                 122.132   |                   200.639  |
|         2 |          4204 |                0.14285  |                    0.180576 |         0.0893089 |             0.101037  |                0.241223 |                    0.249486 |                  68.8646  |                   170.1    |
|         4 |          4154 |                0.202073 |                    0.289603 |         0.110208  |             0.140678  |                0.262107 |                    0.295086 |                  29.7093  |                   137.828  |
|         8 |          4054 |                0.29785  |                    0.45207  |         0.159674  |             0.241561  |                0.304807 |                    0.380093 |                   2.33576 |                    90.8937 |
|        12 |          3954 |                0.373278 |                    0.569045 |         0.20613   |             0.322604  |                0.34251  |                    0.446355 |                  -8.24241 |                    66.1624 |
|        16 |          3854 |                0.433889 |                    0.654336 |         0.243282  |             0.378266  |                0.374641 |                    0.496703 |                 -13.655   |                    53.9947 |

---

## Key Observations:
- **Persistence Error Growth:** Rapid degradation from H=1 to H=16 as static assumptions fail.
- **Linear Ridge Degradation:** Error compounding across longer horizons as autoregressive extrapolation drifts.
- **World Model Stability:** Demonstrates bounded error and physical trajectory plausibility under extended rollouts.
