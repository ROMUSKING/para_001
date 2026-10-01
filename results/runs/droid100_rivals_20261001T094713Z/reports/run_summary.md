# Rival world-model benchmark `droid100_rivals_20261001T094713Z`

Fairness contract: **PASS**

| Comparison (reference vs rival) | Reference RMSE | Rival RMSE | Relative diff. | 95 % CI | Classification |
|---|---:|---:|---:|---|---|
| adjoint_rwm_vs_dreamerv3_rssm | 0.1562 | 0.3616 | -0.568 | [-0.652, -0.486] | reference_better |
| adjoint_rwm_vs_tdmpc2 | 0.1562 | 0.2454 | -0.363 | [-0.424, -0.271] | reference_better |
| adjoint_rwm_vs_dino_wm | 0.1562 | 0.2643 | -0.409 | [-0.465, -0.326] | reference_better |
| adjoint_rwm_vs_vjepa2_ac | 0.1562 | 0.2363 | -0.339 | [-0.440, -0.236] | reference_better |
| adjoint_rwm_vs_persistence | 0.1562 | 0.2272 | -0.312 | [-0.394, -0.185] | reference_better |
| adjoint_rwm_vs_ridge | 0.1562 | 0.1058 | +0.477 | [+0.275, +0.805] | rival_better |

Negative relative difference = the reference (AdjointRWM) has lower RMSE.

Rivals are re-implementations ("-style") at a matched ~25M prediction-path budget on DROID-100 (10 test episodes). No H2, task-success or SOTA claim follows.
