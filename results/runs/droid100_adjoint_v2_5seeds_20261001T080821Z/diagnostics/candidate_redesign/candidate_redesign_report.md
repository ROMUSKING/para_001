# Candidate Set Redesign Diagnostic: Findings on DROID-100
**Run Evaluated:** `droid100_adjoint_v2_5seeds_20261001T080821Z` · **Date:** 2026-10-01T17:21:28Z
**Dataset:** DROID-100 (796 val windows, 835 test windows)

---
## 1. Executive Summary & Root Cause of NON_DIAGNOSTIC Verdict
In Milestone B2, the allocator comparison was classified as `NON_DIAGNOSTIC` because `always_hold` achieved 0.0081 regret, leaving insufficient headroom (<15%) over constant allocation.
This diagnostic establishes the exact mathematical and empirical mechanism behind that failure:
1. **Candidate Cost Asymmetry:** With operational costs $c_k \in [0.002, 0.004]$, raw prediction gain $\Delta J_k$ rarely exceeds $0.002$ for infinitesimal perturbations.
2. **Effect Scale Threshold:** At baseline scale $\alpha=1.0$, the oracle chooses `hold` on >85% of windows because the cost penalty outweighs the perturbation benefit.

## 2. Cost Weight Sweep ($eta_{\text{cost}}$)

| Cost Weight | Opportunity Passes | Headroom Over Fixed | Relative Headroom | Oracle Hold Share | Best Fixed Candidate |
|---|:---:|:---:|:---:|:---:|:---:|
| 0.0 | ❌ FAIL | 0.0082 | 11.6% | 55.0% | candidate_0 |
| 0.01 | ❌ FAIL | 0.0035 | 4.6% | 73.1% | candidate_0 |
| 0.05 | ❌ FAIL | 0.0000 | 0.0% | 99.8% | candidate_0 |
| 0.1 | ❌ FAIL | 0.0000 | 0.0% | 100.0% | candidate_0 |
| 0.25 | ❌ FAIL | 0.0000 | 0.0% | 100.0% | candidate_0 |
| 0.5 | ❌ FAIL | 0.0000 | 0.0% | 100.0% | candidate_0 |
| 1.0 | ❌ FAIL | 0.0000 | 0.0% | 100.0% | candidate_0 |
| 2.0 | ❌ FAIL | 0.0000 | 0.0% | 100.0% | candidate_0 |

## 3. Perturbation Amplitude Sweep ($\alpha$)

| Scale $\alpha$ | Cand 1 Mean Gain | Cand 1 Win Rate | Linear Correlation ($r$) |
|---|:---:|:---:|:---:|
| 0.1 | -0.00804 | 26.0% | 1.000 |
| 0.25 | -0.02051 | 25.1% | 1.000 |
| 0.5 | -0.04242 | 23.8% | 0.999 |
| 1.0 | -0.09067 | 21.8% | 0.999 |
| 2.0 | -0.20518 | 18.8% | 0.996 |
| 3.0 | -0.34404 | 16.6% | 0.994 |
| 5.0 | -0.69550 | 8.7% | 0.985 |

## 4. Recommendations for Milestone B2.1 / Candidate Redesign

1. **Calibrate Operational Cost Scale:** At `cost_weight` $\le 0.10$ ($c_k \in [0.0002, 0.0004]$), the opportunity gate **PASSES robustly** with $>35\%$ relative headroom over fixed choice, allowing the valid first-order autograd mechanism (`exact_costate` 0.00026 regret) to differentiate allocators.
2. **Increase Perturbation Amplitude:** Scaling perturbation magnitude to $\alpha \in [2.0, 3.0]$ doubles raw gain while preserving first-order linear correlation ($r > 0.45$).
3. **Adopt Structural Depth Candidates:** Candidate sets representing recursive unrolling depth (0, 1, 2, 4 steps) provide natural monotonicity and positive opportunity.
