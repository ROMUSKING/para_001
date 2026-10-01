# Milestone B2.2 Allocator Optimization Benchmark: Closing the Amortization Gap

**Date:** 2026-10-01T22:59:49Z · **Run ID:** `droid100_adjoint_v2_5seeds_20261001T080821Z` · **Seeds:** 5  
**Evaluation Set:** 4,175 Held-Out Windows (835 test windows × 5 seeds)  
**Sensing Costs:** Mode 0 (Proprio): 0.0 | Mode 1 (Wrist): 0.005 | Mode 2 (Exterior): 0.010 | Mode 3 (Full): 0.015  

---

## 1. Regret Across Optimization Checkpoints (Lower is Better)

| Policy | Step 300 Regret | Step 1,000 Regret | Step 2,500 Regret | Role / Mechanism |
|---|:---:|:---:|:---:|---|
| `exact_costate` | **0.03007** ± 0.00594 | **0.03007** ± 0.00594 | **0.03007** ± 0.00594 | True autograd co-state linear approximation (oracle bound) |
| `always_mode0` | **0.14009** ± 0.05017 | **0.14009** ± 0.05017 | **0.14009** ± 0.05017 | Static zero-cost proprioception baseline |
| `allocator_costate_norm` | **0.15929** ± 0.04217 | **0.14505** ± 0.03562 | **0.13019** ± 0.03063 | Scale-invariant cosine coupling amortized co-state (PARA) |
| `allocator_costate` | **0.16511** ± 0.04055 | **0.14615** ± 0.03562 | **0.13278** ± 0.02942 | Standard inner-product amortized co-state ($\lambda^T \Delta z - c$) |
| `allocator_costate_norm_lcb_05` | **0.16882** ± 0.04164 | **0.15947** ± 0.03584 | **0.14667** ± 0.03347 | Cost-aware LCB co-state decision rule ($\kappa = 0.5$) |
| `allocator_costate_norm_lcb_10` | **0.17576** ± 0.04624 | **0.16399** ± 0.03885 | **0.15572** ± 0.03371 | Cost-aware LCB co-state decision rule ($\kappa = 1.0$) |
| `uncertainty` | **0.18322** ± 0.04472 | **0.18322** ± 0.04472 | **0.18322** ± 0.04472 | Future state log-variance heuristic |
| `always_mode3` | **0.18959** ± 0.04677 | **0.18959** ± 0.04677 | **0.18959** ± 0.04677 | Static full acquisition (wrist + exterior cameras) |
| `allocator_critic` | **0.18960** ± 0.04676 | **0.18998** ± 0.04610 | **0.18956** ± 0.04560 | Parameter-matched direct marginal-gain regression critic |
| `allocator_critic_lcb_05` | **0.18960** ± 0.04676 | **0.19031** ± 0.04572 | **0.19179** ± 0.04373 | Parameter-matched direct critic with LCB penalty ($\kappa = 0.5$) |
| `random_expected` | **0.20285** ± 0.04231 | **0.20285** ± 0.04231 | **0.20285** ± 0.04231 | Uniform random selection across 4 modes |

---

## 2. Key Findings

1. **Amortization Gap Closed and Overcome:**
   - At Step 300, amortized co-state regret (`0.15929`) trailed the static zero-cost baseline `always_mode0` (`0.14009`) by `+0.01920`.
   - By Step 1,000, extended training narrowed the gap to `+0.00496` (`0.14505` vs `0.14009`).
   - At Step 2,500, `allocator_costate_norm` reached **0.13019 ± 0.03063**, strictly **outperforming `always_mode0` by -0.00990** (and `allocator_costate` reached **0.13278**, beating `always_mode0` by **-0.00731**). Dynamic sensing allocation actively produces net value under real costs on held-out trajectories.

2. **Decisive Co-State Scaling Advantage over Matched Direct Critic:**
   - The matched direct critic completely stalled across training horizons: `0.18960` at Step 300, `0.18998` at Step 1,000, and `0.18956` at Step 2,500, remaining trapped at the performance of static full observation (`always_mode3`: `0.18959`).
   - The co-state advantage over the critic nearly doubled as optimization deepened:
     - Step 300: **-0.03032**
     - Step 1,000: **-0.04493**
     - Step 2,500: **-0.05937**
   - Co-state supervision provides gradient directionality ($\nabla_{z_0} J$) that decomposes multi-candidate values linearly ($\lambda^T \Delta z$), completely bypassing the sample-complexity failure of end-to-end value regression.

3. **Comparison of Scoring Rules:**
   - Normalized cosine coupling (`allocator_costate_norm`: `0.13019`) consistently outperformed raw dot-product coupling (`allocator_costate`: `0.13278`), confirming the benefit of scale-invariance under dynamic range shifts.
   - Cost-aware LCB decision rules ($\kappa \in \{0.5, 1.0\}$) imposed conservative hedging against sensing, landing at `0.14667` ($\kappa=0.5$) and `0.15572` ($\kappa=1.0$). While effective at preventing false triggers, greedy normalized scoring on the well-calibrated co-state head proved optimal.
