# Dynamics k-fold `dynamics_kfold_v2_20260930T231740Z`

## Regime `v2` (own mode `base`): **OK**, class **INCONCLUSIVE**

Anchor: recomputed best validation 0.2127 against stored 0.2125 (probe run dynamics_gate.json, seed 0, validation, mode base), difference 0.0003, tolerance 0.01, within: True.

| Checkpoint | Mode | Relative improvement R | 95 % interval | Class | Random 10-episode splits passing 2 % | Episodes where the model beats persistence |
|---|---|---:|---|---|---:|---:|
| best | base (primary) | -0.221 | [-0.551, +0.053] | INCONCLUSIVE | 0.364 | 0.71 |
| best | full | -0.210 | [-0.536, +0.058] | INCONCLUSIVE | 0.375 | 0.71 |
| final | base | -0.222 | [-0.552, +0.053] | INCONCLUSIVE | 0.366 | 0.70 |
| final | full | -0.211 | [-0.536, +0.059] | INCONCLUSIVE | 0.377 | 0.71 |

Per fold (primary): -0.608, +0.006, +0.247, -0.290, -0.127
By horizon step (primary, estimate [95 % interval]): step 1: -1.076 [-1.701, -0.581]; step 2: -0.354 [-0.718, -0.045]; step 3: -0.080 [-0.373, +0.161]; step 4: +0.065 [-0.176, +0.264]

Status: **OK**.

Episode-level k-fold dynamics against persistence on DROID-100, one training seed per fold, 70 training episodes per fold. Diagnostic models; not evidence for or against H2; B2 does not reuse them.
