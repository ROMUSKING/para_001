# D4-0b `d4_0b_adaptivity_20260929T173428Z` (exploratory)

Correctness: **PASS** · candidate regimes for D4-1: **1** of 9 cells

| Family | Median pulse width | Scoring price | Adjoint beats uniform | Adjoint beats residual and goal-local | Candidate for D4-1 |
|---|---:|---:|---|---|---|
| smooth | 0.02505 | x1 | False | False | False |
| smooth | 0.02505 | x0.25 | False | True | False |
| smooth | 0.02505 | x0 | False | True | False |
| sharp | 0.00251 | x1 | False | False | False |
| sharp | 0.00251 | x0.25 | True | False | False |
| sharp | 0.00251 | x0 | True | True | True |
| sharper | 0.00025 | x1 | True | False | False |
| sharper | 0.00025 | x0.25 | True | False | False |
| sharper | 0.00025 | x0 | True | False | False |

Exploratory design study on a tuning and a validation family of the analytic D4 benchmark; the test family was not read. Scoring-price scales below 1 are hypothetical re-pricings. No learned arm exists, so no H2 statement follows. The sharp and sharper families were added after D4-0 to vary localisation.
