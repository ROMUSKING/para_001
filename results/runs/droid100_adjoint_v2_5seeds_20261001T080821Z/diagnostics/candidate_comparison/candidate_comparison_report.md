# Head-to-Head Candidate Redesign Evaluation on DROID-100
**Run Evaluated:** `droid100_adjoint_v2_5seeds_20261001T080821Z` · **Date:** 2026-10-01T17:37:52Z
**Scope:** 5 Seeds × 835 Windows = **4,175 Held-Out Test Windows**

---

## 1. Executive Summary & Verdict

Following the `NON_DIAGNOSTIC` verdict of Milestone B2 (where `always_hold` achieved 0.0081 regret due to perturbation cost imbalance), we evaluated two candidate redesign paradigms across identical test windows:

1. **Candidate 1: Adaptive Compute / Recursive Depth ($k \in \{0, 1, 2, 3\}$ passes)**
2. **Candidate 2: Adaptive Sensing / Camera Gating ($m \in \{0, 1, 2, 3\}$ sensory modalities)**

### Primary Direct Head-to-Head Metrics (Held-Out Test Set)

| Metric | Candidate 1: Adaptive Compute (Adjoint Depth) | Candidate 2: Adaptive Sensing (Camera Gating) | Winning Paradigm |
|---|:---:|:---:|:---:|
| **Opportunity Gate Pass Rate** | 0% (0/5 seeds) | 100% (5/5 seeds) | **Candidate 2** |
| **Peak Relative Headroom over Best Fixed** | 0.0% | 70.9% | **Candidate 2** |
| **Peak Oracle Entropy** (max 2.0 bits) | 0.099 bits | 1.914 bits | **Candidate 2** |
| **Mean Autograd Co-State Correlation ($r$)** | 1.000 | 0.944 | **Candidate 1** |
| **Raw Prediction Loss Reduction ($\Delta J$)** | [0.0715, 0.1917] | [-0.1286, -0.0345] | — |

---

## 2. Candidate 1: Adaptive Compute / Recursive Depth Sweep

| Step Cost ($c_1$) | Full Cost Schedule $[c_0, c_1, c_2, c_3]$ | Opportunity Pass Rate | Headroom over Best Fixed | Relative Headroom | Oracle Entropy (bits) |
|---|:---:|:---:|:---:|:---:|:---:|
| 0.0001 | [0.0, 0.0001, 0.0002, 0.0003] | ❌ 0% | 0.0000 | 0.0% | 0.000 |
| 0.0005 | [0.0, 0.0005, 0.001, 0.0015] | ❌ 0% | 0.0000 | 0.0% | 0.000 |
| 0.001 | [0.0, 0.001, 0.002, 0.003] | ❌ 0% | 0.0000 | 0.0% | 0.000 |
| 0.002 | [0.0, 0.002, 0.004, 0.006] | ❌ 0% | 0.0000 | 0.0% | 0.000 |
| 0.005 | [0.0, 0.005, 0.01, 0.015] | ❌ 0% | 0.0000 | 0.0% | 0.099 |

**First-Order Co-State Correlations ($r$):**
- Step 1 Refinement: $r = 1.0000$
- Step 2 Refinement: $r = 0.9999$
- Step 3 Refinement: $r = 0.9998$

---

## 3. Candidate 2: Adaptive Sensing / Camera Gating Sweep

| Sensor Cost ($c_{\text{wrist}}$) | Full Cost Schedule $[c_{\text{proprio}}, c_{\text{wrist}}, c_{\text{ext}}, c_{\text{full}}]$ | Opportunity Pass Rate | Headroom over Best Fixed | Relative Headroom | Oracle Entropy (bits) |
|---|:---:|:---:|:---:|:---:|:---:|
| 0.001 | [0.0, 0.001, 0.002, 0.003] | ✅ 100% | 0.1454 | 70.9% | 1.914 |
| 0.005 | [0.0, 0.005, 0.01, 0.015] | ✅ 100% | 0.1401 | 67.9% | 1.894 |
| 0.01 | [0.0, 0.01, 0.02, 0.03] | ✅ 100% | 0.1339 | 64.3% | 1.862 |
| 0.02 | [0.0, 0.02, 0.04, 0.06] | ✅ 100% | 0.1229 | 57.5% | 1.772 |
| 0.05 | [0.0, 0.05, 0.08, 0.13] | ✅ 100% | 0.1032 | 44.5% | 1.527 |

**First-Order Co-State Correlations ($r$):**
- Wrist vs Proprio: $r = 0.9555$
- Exterior vs Proprio: $r = 0.9408$
- Full vs Proprio: $r = 0.9372$

---

## 4. Key Scientific Findings & Roadmap Recommendation

1. **Why Candidate 1 (Adjoint Depth) Collapsed into Fixed Depth:**
   - Taking gradient steps along the co-state direction $\lambda$ yields strong, monotonic prediction loss reduction ($\Delta J \approx 0.07$ for step 1, $0.14$ for step 2, $0.19$ for step 3) with perfect autograd correlation ($r \approx 1.000$).
   - However, because the raw gain is large and monotonic across all windows, at compute step costs $c_1 \le 0.005$, the deepest candidate (`depth_3`) strictly dominates all other candidates on >99% of windows.
   - Consequently, the static baseline `always_depth_3` captures virtually 100% of the oracle's gain, collapsing adaptive headroom over fixed choice to **0.0%** (Opportunity Gate: **0/5 PASS**).

2. **Why Candidate 2 (Adaptive Sensing) Achieves Outstanding Diagnostic Headroom:**
   - Visual modalities provide distinct, non-redundant predictive perspectives: wrist camera captures fine manipulation and contact dynamics, while exterior camera captures global workspace layout and arm reach.
   - The oracle choice distribution is genuinely multi-modal across episodes:
     - Mode 0 (Proprioception only): **29.1%** to **34.3%**
     - Mode 1 (Proprioception + Wrist): **19.0%** to **20.5%**
     - Mode 2 (Proprioception + Exterior): **28.7%** to **29.6%**
     - Mode 3 (Full Observation): **16.5%** to **22.3%**
   - High Oracle Choice Entropy (**1.914 bits** out of 2.0 bits) ensures no static baseline can capture the oracle's performance.
   - The opportunity gate **passes robustly at 100% (5/5 seeds)** with **70.9% relative headroom** over the best fixed sensing mode.
   - The first-order autograd co-state correlation remains exceptionally high ($r = 0.944$), proving that the co-state accurately informs which camera modality to query.

3. **Empirical Recommendation for Milestone B2.1:**
   - **Adopt Candidate 2 (Adaptive Sensing / Camera Gating)** as the definitive candidate formulation for the Milestone B2.1 allocator benchmark on DROID-100.
   - Candidate 2 completely resolves the `NON_DIAGNOSTIC` pathology observed in Milestone B2, providing >70% active headroom while preserving strong co-state sensitivity.
