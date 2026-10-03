# Session 1: Horizon Stress Testing & Autograd Scaling Report

**Date (UTC):** 2026-10-03T09:02:08.244559+00:00  
**Device:** NVIDIA L4 (22.03 GiB VRAM)  
**Evaluated Horizons:** [4, 8, 16, 32, 64]  
**Evaluated Batch Sizes:** [16, 32, 64]  

---

## 1. Empirical Scaling Summary

| Horizon $H$ | Batch $B$ | Forward (ms) | BPTT Backward (ms) | HVP 2nd-Order (ms) | Total Step (ms) | Peak VRAM (MiB) | VRAM Util % |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4 | 16 | 289.43 | 52.07 | 117.68 | 459.18 | 281.3 | 1.2% |
| 4 | 32 | 12.26 | 15.49 | 37.61 | 65.35 | 440.1 | 1.9% |
| 4 | 64 | 14.15 | 17.07 | 38.85 | 70.07 | 558.5 | 2.5% |
| 8 | 16 | 12.46 | 16.61 | 41.12 | 70.19 | 398.2 | 1.8% |
| 8 | 32 | 12.67 | 16.17 | 42.47 | 71.31 | 451.5 | 2.0% |
| 8 | 64 | 15.13 | 18.83 | 42.51 | 76.47 | 571.0 | 2.5% |
| 16 | 16 | 15.15 | 21.04 | 51.76 | 87.94 | 406.5 | 1.8% |
| 16 | 32 | 17.03 | 22.21 | 56.72 | 95.96 | 495.8 | 2.2% |
| 16 | 64 | 18.16 | 23.94 | 55.27 | 97.37 | 613.4 | 2.7% |
| 32 | 16 | 20.04 | 29.06 | 76.32 | 125.42 | 422.1 | 1.9% |
| 32 | 32 | 22.48 | 30.70 | 79.48 | 132.65 | 559.9 | 2.5% |
| 32 | 64 | 26.10 | 33.30 | 77.79 | 137.19 | 708.2 | 3.1% |
| 64 | 16 | 29.75 | 45.71 | 123.77 | 199.22 | 511.3 | 2.3% |
| 64 | 32 | 34.10 | 47.71 | 129.86 | 211.68 | 707.7 | 3.1% |
| 64 | 64 | 41.72 | 52.64 | 128.48 | 222.83 | 904.3 | 4.0% |

---

## 2. Key Observations & Hardware Gating Insights

1. **Memory Growth Scaling $d(\text{VRAM})/dH$:**
   Unrolling the full computational graph with double-backpropagation retains activations across all unrolled recurrence steps.
2. **Compute Latency Scaling:**
   Evaluates whether long horizons ($H \ge 32$) induce quadratic attention bottlenecks or linear recurrence scaling.
3. **Hopper G4 Justification Gate:**
   Checks whether $H=64$ at nominal batch size $B=64$ exceeds L4 24 GiB VRAM or requires gradient checkpointing.
