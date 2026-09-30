# D4-0 `d4_time_stepping_20260929T162518Z`

Correctness: **PASS** · Opportunity (validation): **PASS** (relative headroom 0.4567, required 0.15) · Equal-compute payoff (validation): **FAIL** · D4-1 opens: **False**

| Paired AURC difference at equal refinement count (test) | Estimate | 95 % CI | Reading |
|---|---:|---|---|
| adjoint - residual | -12.41 | [-17.09, -8.295] | adjoint lower AURC |
| adjoint - goal_local | -15.27 | [-20.8, -10.47] | adjoint lower AURC |
| adjoint - uniform | -29.46 | [-38.11, -21.44] | adjoint lower AURC |
| goal_local - residual | +2.866 | [+1.514, +4.503] | residual lower AURC |
| residual - uniform | -17.05 | [-22.53, -12.16] | residual lower AURC |
| exact_tau_adjoint - adjoint | -1.007 | [-1.997, -0.2103] | exact_tau_adjoint lower AURC |

| Total compute (CN steps) | Mean J adjoint | Mean J uniform | adjoint − uniform, 95 % CI | Reading |
|---:|---:|---:|---|---|
| 1000 | 0.8861 | 0.2682 | +0.6179 [+0.3564, +0.9284] | uniform lower objective |
| 2000 | 0.5356 | 0.2354 | +0.3002 [+0.1594, +0.466] | uniform lower objective |
| 4000 | 0.3005 | 0.06745 | +0.2331 [+0.1307, +0.3545] | uniform lower objective |
| 8000 | 0.1532 | 0.05958 | +0.09363 [+0.04925, +0.1469] | uniform lower objective |

Analytic benchmark generated in-repo (frozen seeds). The co-state is computed exactly from the known model (Mode A); no learned arms, so no H2 statement follows. exact_tau_adjoint and one_step_oracle are privileged diagnostics.
