# D4-0 `d4_time_stepping_20260929T162914Z`

Correctness: **PASS** · Opportunity (validation): **PASS** (relative headroom 0.7019, required 0.15) · Equal-compute payoff (validation): **FAIL** · D4-1 opens: **False**

| Paired AURC difference at equal refinement count (test) | Estimate | 95 % CI | Reading |
|---|---:|---|---|
| adjoint - residual | -5.391 | [-9.635, -2.216] | adjoint lower AURC |
| adjoint - goal_local | -7.552 | [-12.74, -3.582] | adjoint lower AURC |
| adjoint - uniform | -8.616 | [-13.62, -4.444] | adjoint lower AURC |
| goal_local - residual | +2.16 | [+0.5206, +4.15] | residual lower AURC |
| residual - uniform | -3.225 | [-6.276, -0.388] | residual lower AURC |
| exact_tau_adjoint - adjoint | -1.249 | [-2.736, +0.03261] | no detectable difference |

| Total compute (CN steps) | Mean J adjoint | Mean J uniform | adjoint − uniform, 95 % CI | Reading |
|---:|---:|---:|---|---|
| 1000 | 0.4925 | 0.08254 | +0.4099 [+0.1988, +0.6584] | uniform lower objective |
| 2000 | 0.2691 | 0.07239 | +0.1967 [+0.0973, +0.3143] | uniform lower objective |
| 4000 | 0.1309 | 0.01986 | +0.1111 [+0.05431, +0.1806] | uniform lower objective |
| 8000 | 0.07241 | 0.01794 | +0.05446 [+0.02297, +0.09466] | uniform lower objective |

Analytic benchmark generated in-repo (frozen seeds). The co-state is computed exactly from the known model (Mode A); no learned arms, so no H2 statement follows. exact_tau_adjoint and one_step_oracle are privileged diagnostics.
