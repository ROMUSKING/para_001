# Candidate Redesign Evaluation on DROID-100: Adaptive Compute (Depth) vs Adaptive Sensing (Camera Gating)

**Date:** 2026-10-01  
**Run Evaluated:** `droid100_adjoint_v2_5seeds_20261001T080821Z`  
**Config SHA-256:** `eaad4782d247e9d085b97e0330e2beb1fbbc00bfe97d7d1f5107f573727358a5`  
**Dataset:** DROID-100 (10 held-out test episodes, 835 test windows/seed × 5 seeds = 4,175 evaluation windows; 10 validation episodes, 796 windows/seed × 5 seeds = 3,980 windows)  
**Hardware:** Remote NVIDIA L4 (Google Colab `b2-worker`), verified terminated after execution  
**Artifacts:** [`candidate_comparison_summary.json`](file:///home/r/git/para_001/results/runs/droid100_adjoint_v2_5seeds_20261001T080821Z/diagnostics/candidate_comparison/candidate_comparison_summary.json), [`candidate_comparison_report.md`](file:///home/r/git/para_001/results/runs/droid100_adjoint_v2_5seeds_20261001T080821Z/diagnostics/candidate_comparison/candidate_comparison_report.md)

---

## 1. Motivation & Context

In Milestone B2 ([research note](2026-10-01-b2-pilot-v2.md)), the allocator comparison was classified as `NON_DIAGNOSTIC` despite $5/5$ seeds passing the dynamics gate. The root cause was candidate cost asymmetry: operational costs ($c_k \in [0.002, 0.004]$) penalized random latent perturbations ($z + \delta_k$) more than the marginal prediction reduction ($\Delta J \approx 0.0005$), leading the oracle to choose `always_hold` on $>85\%$ to $100\%$ of windows.

Per Roadmap §6 and user directives, we designed and empirically benchmarked the two candidate redesign paradigms across the identical **4,175 held-out test windows** over all 5 seeds on DROID-100:
1. **Candidate 1 (Adaptive Compute / Recursive Depth):** Refinement passes $k \in \{0, 1, 2, 3\}$ along the autograd co-state trajectory ($z^{(k)} = z^{(k-1)} - \eta \lambda^{(k-1)}$) with calibrated step costs $c_k = k \cdot c_1$.
2. **Candidate 2 (Adaptive Sensing / Camera Gating):** Sensory modality inclusion $m \in \{0, 1, 2, 3\}$:
   - Mode 0: Proprioception only (exterior and wrist camera embeddings masked to zero, cost $c_0 = 0.0$).
   - Mode 1: Proprioception + Wrist camera (exterior masked, wrist active, cost $c_1$).
   - Mode 2: Proprioception + Exterior camera (wrist masked, exterior active, cost $c_2$).
   - Mode 3: Full Observation (both exterior and wrist cameras active, cost $c_3 = c_1 + c_2$).

---

## 2. Direct Head-to-Head Comparison (Pooled 5 Seeds, 4,175 Test Windows)

Data read from `results/runs/droid100_adjoint_v2_5seeds_20261001T080821Z/diagnostics/candidate_comparison/candidate_comparison_summary.json`:

| Metric | Candidate 1: Adaptive Compute (Depth) | Candidate 2: Adaptive Sensing (Camera Gating) | Decisive Winner |
|---|:---:|:---:|:---:|
| **Opportunity Gate Pass Rate** | **0% (0/5 seeds)** | **100% (5/5 seeds)** | **Candidate 2** |
| **Validation Pass Rate** | **0% (0/5 seeds)** | **100% (5/5 seeds)** | **Candidate 2** |
| **Peak Relative Headroom over Best Fixed** | **0.0%** | **70.9%** (Gate threshold $\ge 15\%$) | **Candidate 2** |
| **Peak Absolute Headroom** | 0.0000 | **+0.1454** | **Candidate 2** |
| **Oracle Choice Entropy** (max 2.0 bits) | 0.099 bits | **1.914 bits** | **Candidate 2** |
| **First-Order Co-State Correlation ($r$)** | **1.000** | **0.944** | Candidate 1 |
| **Raw Loss Reduction Range ($\Delta J$)** | [+0.0715, +0.1917] | [−0.1286, −0.0345]* | — |

*\*In Candidate 2, Mode 0 is the zero-sensing baseline; values represent loss reductions relative to Mode 0.*

---

## 3. Detailed Empirical Analysis

### Candidate 1: Adaptive Compute / Recursive Depth
- **Monotonic Reduction with Step Depth:** The autograd co-state provides the exact descent direction of the prediction loss. Successive gradient steps achieve monotonic loss reduction:
  - Step 1: $\Delta J = +0.0715 \pm 0.041$
  - Step 2: $\Delta J = +0.1350 \pm 0.078$
  - Step 3: $\Delta J = +0.1917 \pm 0.111$
  - First-order Taylor correlation: $r = 1.0000$ (step 1), $0.9999$ (step 2), $0.9998$ (step 3).
- **Failure Mode (Fixed Baseline Dominance):** Because every window benefits monotonically from unrolling, at any step cost $c_1 \le 0.005$, the deepest candidate (`depth_3`) provides superior net gain over shallower steps on $>99\%$ of test windows. The static baseline `always_depth_3` captures essentially 100% of the oracle's gain, collapsing adaptive headroom over best fixed to **0.0%** ($0/5$ seeds pass the opportunity gate).

### Candidate 2: Adaptive Sensing / Camera Gating
- **Genuine Multimodal Diversity:** Unlike 1D depth unrolling, sensory channels represent orthogonal physical perspectives (wrist camera captures gripper-object contact; exterior camera captures global robot positioning).
- **Oracle Choice Distribution Across Test Windows:**
  - Mode 0 (Proprioception only): **29.1%** to **34.3%**
  - Mode 1 (Proprioception + Wrist): **19.0%** to **20.5%**
  - Mode 2 (Proprioception + Exterior): **28.7%** to **29.6%**
  - Mode 3 (Full Observation): **16.5%** to **22.3%**
- **Balanced Entropy:** Oracle choice entropy reaches **1.914 bits** (out of theoretical maximum $2.000$ bits for 4 candidates), demonstrating that no single sensory modality dominates.
- **Robust Opportunity Headroom:**
  - Cost $c_1 = 0.001$: Headroom = **+0.1454**, Relative Headroom = **70.9%**, Pass Rate = **100% (5/5)**
  - Cost $c_1 = 0.005$: Headroom = **+0.1401**, Relative Headroom = **67.9%**, Pass Rate = **100% (5/5)**
  - Cost $c_1 = 0.010$: Headroom = **+0.1339**, Relative Headroom = **64.3%**, Pass Rate = **100% (5/5)**
  - Cost $c_1 = 0.020$: Headroom = **+0.1229**, Relative Headroom = **57.5%**, Pass Rate = **100% (5/5)**
  - Cost $c_1 = 0.050$: Headroom = **+0.1032**, Relative Headroom = **44.5%**, Pass Rate = **100% (5/5)**
- **High Co-State Guidance Signal:** The first-order inner product $-\langle \lambda_0, \Delta z_m \rangle$ achieves a correlation of **$r = 0.944$** with the true empirical loss reduction of adding visual modalities ($r = 0.956$ for Wrist, $0.941$ for Exterior, $0.937$ for Full).

---

## 4. Scientific Conclusions & Recommendation for Milestone B2.1

1. **Resolution of the NON_DIAGNOSTIC Verdict:** Candidate 2 (Adaptive Sensing / Camera Gating) completely eliminates the headroom collapse. With $70.9\%$ relative headroom and $100\%$ pass rate across 5 seeds, it creates an ideal diagnostic environment for evaluating learned allocators.
2. **Empirical Recommendation:** Adopt **Candidate 2 (Adaptive Sensing / Camera Gating)** as the definitive candidate formulation for the Milestone B2.1 allocator re-run on DROID-100.
3. **Compute Efficiency:** The experiments were run on Colab NVIDIA L4 with direct CLI management; the remote session was terminated immediately after artifact download to conserve compute units.
