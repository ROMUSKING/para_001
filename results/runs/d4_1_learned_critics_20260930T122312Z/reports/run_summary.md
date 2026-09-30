# D4-1 `d4_1_learned_critics_20260930T122312Z` (exploratory)

Correctness: **PASS**

| Cell | Head width | D saturated | Estimator floor met (seeds) | R0 | RC1 | RC1-L | RC2-L | RC3 | RC3-L | RT-L | Exit class |
|---|---:|---|---:|---|---|---|---|---|---|---|---|
| m4 | 8 | True | 4 | False | False | False | False | False | False | True | teacher-only value |
| m64 | 8 | True | 0 | False | False | False | False | False | False | False | inconclusive: precision |

Exploratory design study on a train, a tuning and a validation family of the analytic D4 benchmark with a random unit goal per instance; the test family was not read. FLOPs are counted from the code path, not measured. The lookup price and the teacher_feature arm are hypothetical; the validation instances were read before for the non-learned arms of D4-3. No statement about real data or about H2 there follows.
