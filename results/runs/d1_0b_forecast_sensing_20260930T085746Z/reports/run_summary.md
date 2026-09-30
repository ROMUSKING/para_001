# D1-0b `d1_0b_forecast_sensing_20260930T085746Z` (exploratory)

Correctness: **PASS** · G0 on the tuning machines (primary): **PASS** (0.522) · stage B (validation) run: **True**
G0 on validation: **PASS** (not robust) r = 0.298 [0.023, 0.743] · G1: **PASS** (robust) headroom 0.875 [0.295, 1.347]

| Setting | Best fixed | r [95 % CI] | G0 | Relative headroom [95 % CI] | G1 |
|---|---|---|---|---|---|
| h5_L60 | top_weighted_volatility | 0.298 [0.023, 0.743] | True | 0.875 [0.295, 1.347] | True |
| h1_L60 | top_weighted_volatility | 0.425 [-0.051, 1.420] | True | 0.896 [0.477, 1.254] | True |
| h15_L60 | top_weighted_volatility | 0.203 [-0.028, 0.550] | True | 0.940 [0.750, 1.062] | True |
| h5_L30 | top_sensitivity | 0.146 [-0.038, 0.407] | True | 0.876 [0.794, 0.976] | True |
| h5_L120 | top_weighted_volatility | 0.322 [0.069, 0.765] | True | 0.584 [-1.427, 2.982] | True |

Exploratory Rung-0 study on SMD machines with non-learned policies and a frozen linear forecaster. The oracle is a privileged greedy selection that sees the window states but not the forecast targets. No learned critic and no co-state are involved, so no H2 statement follows. Policy computation is not charged.
