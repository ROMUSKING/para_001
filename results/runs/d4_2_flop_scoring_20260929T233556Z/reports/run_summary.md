# D4-2 `d4_2_flop_scoring_20260929T233556Z` (exploratory)

Correctness: **PASS** · D4-1 candidate cells: **0** of 7

| Cell | m | Hidden | R1 amortised beats uniform | R3 co-state weight helps cheap | Candidate | Amortised price / step-doubling |
|---|---:|---:|---|---|---|---:|
| base | 32 | 8 | True | False | False | 0.245 |
| m4 | 4 | 8 | False | False | False | 2.238 |
| m64 | 64 | 16 | False | True | False | 0.133 |
| wide | 32 | 8 | False | False | False | 0.245 |
| amp1 | 32 | 8 | False | False | False | 0.245 |
| depth7 | 32 | 8 | True | False | False | 0.245 |
| depth11 | 32 | 8 | True | False | False | 0.245 |

Exploratory design study on a train, a tuning and a validation family of the analytic D4 benchmark with a fixed goal per system; the test family was not read. FLOPs are counted from the code path, not measured. adjoint_free is a hypothetical ceiling. The amortised scorer is supervised by the adjoint-weighted error and is not a direct critic, so no H2 statement follows.
