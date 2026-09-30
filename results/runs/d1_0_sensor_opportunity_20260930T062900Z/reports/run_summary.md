# D1-0 `d1_0_sensor_opportunity_20260930T062900Z` (exploratory)

Correctness: **PASS** · G1 at the primary length: **PASS** (robust) · relative headroom 0.920 [0.886, 0.949]

| Run | Length | Objective | Best fixed | Relative headroom | 95 % interval | G1 | Robust |
|---|---:|---|---|---:|---|---|---|
| primary | 60 | score | top_volatility | 0.920 | [0.886, 0.949] | True | True |
| decision | 60 | decision | top_volatility | 0.872 | [0.845, 0.901] | True | True |
| L30 | 30 | score | top_volatility | 0.929 | [0.888, 0.957] | True | True |
| L120 | 120 | score | top_volatility | 0.915 | [0.878, 0.944] | True | True |

Exploratory Rung-0 gate on seven validation machines of the Server Machine Dataset with non-learned policies. The oracle is a privileged greedy selection and the reference is the best-known curve over it and every deployable policy. The loss is a score-fidelity proxy for a frozen detector. No learned critic and no co-state are involved, so no H2 statement follows. Policy computation is not charged.
