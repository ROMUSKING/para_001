# Dynamics k-fold `dynamics_kfold_pilot_20261001T000051Z`

## Regime `pilot` (own mode `full`): **OK**, class **INCONCLUSIVE**

Anchor: recomputed best validation 0.2133 against stored 0.2263 (pilot checkpoint best_metrics, validation, mean of full_rmse_by_horizon), difference 0.0130, tolerance 0.02, within: True.

| Checkpoint | Mode | Relative improvement R | 95 % interval | Class | Random 10-episode splits passing 2 % | Episodes where the model beats persistence |
|---|---|---:|---|---|---:|---:|
| best | base | -0.230 | [-0.562, +0.047] | INCONCLUSIVE | 0.354 | 0.70 |
| best | full (primary) | -0.221 | [-0.547, +0.051] | INCONCLUSIVE | 0.363 | 0.70 |
| final | base | -0.228 | [-0.561, +0.048] | INCONCLUSIVE | 0.357 | 0.70 |
| final | full | -0.218 | [-0.544, +0.053] | INCONCLUSIVE | 0.367 | 0.70 |

Per fold (primary): -0.598, +0.009, +0.232, -0.290, -0.139
By horizon step (primary, estimate [95 % interval]): step 1: -1.087 [-1.712, -0.588]; step 2: -0.355 [-0.716, -0.048]; step 3: -0.077 [-0.365, +0.161]; step 4: +0.068 [-0.167, +0.265]

Status: **OK**.

Episode-level k-fold dynamics against persistence on DROID-100, one training seed per fold, 70 training episodes per fold. Diagnostic models; not evidence for or against H2; B2 does not reuse them.
