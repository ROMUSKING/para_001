# D4-3 `d4_3_varying_goal_20260930T084725Z` (exploratory)

Correctness: **PASS** · D4-1 candidate cells: **2** of 7

| Cell | m | R3v co-state weight helps cheap | R4v cheap-adjoint beats uniform | Candidate | Setup charged beats cheap |
|---|---:|---|---|---|---|
| base | 32 | False | True | False | False |
| m4 | 4 | True | True | True | False |
| m64 | 64 | True | True | True | False |
| wide | 32 | False | False | False | False |
| amp1 | 32 | False | False | False | False |
| depth7 | 32 | False | True | False | False |
| depth11 | 32 | False | True | False | False |

Exploratory design study on a tuning and a validation family of the analytic D4 benchmark with a random unit goal per instance; the test family was not read. FLOPs are counted from the code path, not measured. adjoint_free is a hypothetical ceiling and cheap_adjoint takes the tabulated co-state as given. No learned critic is involved, so no H2 statement follows.
