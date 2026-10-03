# Milestone B3c HARP Selective Analytical Rescue Benchmark

**Date:** 2026-10-03T11:27:32Z · **Model:** `HARP (hybrid_adjoint_rwm)` · **Seeds:** 3
**Multi-Site Test Set:** 12462 held-out windows across 12 robotics laboratories

## 1. Baselines on Multi-Site Shard

| Policy | Mean Regret | Std Regret | Description |
|---|:---:|:---:|:---|
| **Exact Autograd Oracle** | **`0.00000`** | `±0.00000` | Privileged theoretical upper bound |
| **Refusal Baseline (`always_mode0`)** | **`0.72139`** | `±0.55545` | Static refusal to sense |
| **Amortized HARP Forward Pass** | **`0.39904`** | `±0.27897` | Pure sub-microsecond inference (tau=0) |
| **Matched Direct Critic** | **`0.47105`** | `±0.33675` | Direct marginal-gain baseline |

## 2. Multi-Site Pareto Frontier (Confidence Gating Sweep)

| Gating Threshold (tau) | Actual Invocation Rate | Mean Test Regret | Systems Latency (ms) | Effective Throughput (Hz) |
|:---:|:---:|:---:|:---:|:---:|
| `tau = 0.00` | **`  0.0%`** | **`0.39904 ± 0.27897`** | `0.000 ms` | **`2613918.7 Hz`** |
| `tau = 0.05` | **`  5.0%`** | **`0.39506 ± 0.27654`** | `0.182 ms` | **`24522.6 Hz`** |
| `tau = 0.10` | **` 10.0%`** | **`0.32966 ± 0.24560`** | `0.363 ms` | **`12344.3 Hz`** |
| `tau = 0.20` | **` 20.0%`** | **`0.26876 ± 0.18864`** | `0.725 ms` | **` 6200.5 Hz`** |
| `tau = 0.30` | **` 30.0%`** | **`0.25553 ± 0.17906`** | `1.087 ms` | **` 4140.0 Hz`** |
| `tau = 0.50` | **` 50.0%`** | **`0.15725 ± 0.14140`** | `1.812 ms` | **` 2485.9 Hz`** |
| `tau = 0.80` | **` 80.0%`** | **`0.05739 ± 0.04836`** | `2.898 ms` | **` 1554.6 Hz`** |
| `tau = 1.00` | **`100.0%`** | **`0.00505 ± 0.00384`** | `3.623 ms` | **` 1243.8 Hz`** |

## 3. Systems Conclusion

Gating HARP inference with selective autograd rescue achieves a continuous, strictly monotonic trade-off on multi-site robotics data. Rescuing 20% of ambiguous decisions drops regret significantly while sustaining >5,000 Hz throughput on NVIDIA L4.
