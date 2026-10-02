# Next Research Steps: Amortization Gap Closure, HARP Selective Rescue, and Closed-Loop Control

**Date:** 2026-10-02  
**Author:** Antigravity & OpenCode Research Team  
**Governing Documents:** `docs/research-plan/adjoint_guided_comprehensive_research_plan.md`, `docs/plans/roadmap.md`, `docs/plans/WORKLOG.md`

---

## 1. Executive Summary: What Has Been Established

The AdjointRWM research programme has systematically validated the core tenets of adjoint-guided world models across multiple empirical gates on real robot data (DROID dataset):

1. **Empirical Substrate Superiority (Milestones B1 & B3b):**
   - On both the 100-episode DROID pilot and the 500-episode stratified multi-site shard across 14 robotics laboratories, AdjointRWM decisively outperformed all four deep rival world model families on test proprioception RMSE:
     - DreamerV3-RSSM: **−44.62%** [−57.45%, −37.77%]
     - TD-MPC2: **−33.33%** [−39.94%, −25.90%]
     - DINO-WM: **−24.53%** [−35.52%, −18.19%]
     - V-JEPA 2-AC: **−24.37%** [−30.69%, −18.79%]
   - Won in 100% of individual robot laboratories (12/12 test sites).
   - Maintained strong action coupling (3.19× to 4.00×).

2. **Multi-Horizon Crossover Dynamics (Milestone E3.3):**
   - Disproved the short-horizon linear extrapolation advantage: closed-form Ridge forecasters appear competitive only on trivial short horizons ($H=1 \dots 4$).
   - At extended horizons ($H \ge 8$), linear extrapolation compounds error exponentially (+388% for Ridge, +524% for Persistence from $H=1 \to 16$).
   - AdjointRWM error is bounded (+113%), crossing over Persistence at $H^*=8$ (terminal) and $H^*=11$ (mean), widening its lead to −24.09% terminal RMSE at $H=16$.

3. **Hybrid Kinematic-Residual State-of-the-Art (Milestone E3.4 - HARP):**
   - Decoupled linear joint momentum $s^{\text{kin}}_h(s_0, v_0, u_{1:h})$ from the non-linear transformer-GRU neural residual $\Delta s^{\text{residual}}_h(z_h)$.
   - Established state-of-the-art across all models on physical task coordinates:
     - Cartesian 6D Pose RMSE: **0.0328** (−46.1% vs Persistence, −4.1% vs Ridge)
     - Gripper Aperture RMSE: **0.0418** (−39.9% vs Persistence, −7.7% vs Ridge)
   - Halved Step-1 discontinuity error by 44.4% (0.2330 to 0.1295).
   - Real-time batch-1 CUDA-synchronized latency: **7.80 ms p50** on NVIDIA L4 GPU.

---

## 2. Immediate Research Agenda & Milestones

### Direction 1: Closing the Amortization Gap via Pairwise Margin Ranking & Listwise Losses (Milestone B2.3)
- **Background:**
  In Milestone B2.2, amortized costate allocation ($0.13019$ regret) beat the parameter-matched direct critic ($0.18956$) and beat fixed refusal ($0.14009$). However, the exact autograd oracle co-state reached $0.03007$.
- **Core Diagnosis:**
  The current allocator training loss (`AllocatorJob.training_loss`) uses standard Cross-Entropy on `s["adjoint"]` against the hard argmax `oracle`.
  This has two severe mathematical deficiencies:
  1. It discards continuous relative gains: misordering two candidates with gains $+0.051$ and $+0.049$ is penalized the same as misordering $+0.051$ and $-0.500$.
  2. It trains on unnormalized dot products rather than the scale-invariant cosine coupling scores (`adjoint_norm`) that won in B2.2.
- **Intervention:**
  Implement Pairwise Margin Ranking Loss:
  $$\mathcal{L}_{\text{margin}}(\hat{s}, y) = \frac{1}{\binom{K+1}{2}} \sum_{i < j} \max\left(0, \Delta y_{ij} - \text{sign}(y_i - y_j)(\hat{s}_i - \hat{s}_j)\right)$$
  and Plackett-Luce Listwise Softmax KL Loss:
  $$\mathcal{L}_{\text{PL}}(\hat{s}, y) = D_{\text{KL}}\left(\text{Softmax}(y / \tau) \parallel \text{Softmax}(\hat{s} / \tau)\right)$$
- **Expected Outcome:**
  Drive amortized regret below $0.100$, cutting the amortization gap to the oracle by $>30\%$.

### Direction 2: Unified HARP Selective Analytical Rescue Interface (Milestone B3c)
- **Background:**
  Milestone B3 demonstrated that confidence gating ($\tau$) creates a strictly monotonic Pareto frontier: sub-microsecond amortized inference (0.0006 ms) up to exact autograd rollouts (0.653 ms). Rescuing just 20% of ambiguous decisions achieved a 20% improvement toward the oracle bound.
- **Intervention:**
  Integrate the Selective Analytical Rescue Interface with the HARP architecture (`HybridAdjointRecursiveWorldModel`).
  HARP outputs calibrated residual log-variances $\sigma^2_{\Delta s, h}$. The gating condition combines:
  1. Decision margin: $\Delta \hat{S} = \hat{S}_{(1)} - \hat{S}_{(2)}$
  2. Epistemic uncertainty: $\bar{\sigma} = \frac{1}{H}\sum_{h=1}^H \|\sigma_{\Delta s, h}\|_1$
  When $\Delta \hat{S} < \tau_{\text{margin}}$ or $\bar{\sigma} > \tau_{\text{uncert}}$, trigger exact autograd co-state backward step.
- **Expected Outcome:**
  A production-ready Pareto frontier on the multi-site E3.1 shard with $>5\text{ kHz}$ throughput and near-oracle regret.

### Direction 3: Belief-Space Co-States and True Value of Information (VOI)
- **Background:**
  Independent Peer Review 3 highlighted that $\nabla_z J$ is state perturbation sensitivity under a deterministic or point estimate, not expected value of information (reduction of covariance $\Delta \Sigma_z$).
- **Intervention:**
  Formalize second-order Laplace approximation:
  $$\text{VOI}_k \approx \nabla_z J^\top \Delta \mu_{z, k} + \frac{1}{2} \text{Tr}\left(\nabla^2_{zz} J \cdot \Delta \Sigma_{z, k}\right)$$
  using a diagonal empirical Fisher Information Matrix (FIM) or Gauss-Newton approximation.
- **Expected Outcome:**
  Measure whether second-order uncertainty reduction provides statistically significant headroom over first-order co-states in multi-camera gating.

### Direction 4: Closed-Loop Interactive Evaluation (Milestone B5)
- **Background:**
  All tests on DROID are open-loop trajectory evaluations on offline datasets.
- **Intervention:**
  Closed-loop interactive simulation benchmark (e.g., ManiSkill3 Franka manipulation tasks) comparing HARP vs TD-MPC2 and DreamerV3 under realistic latency constraints (20 Hz loop). Slow models that incur inference lag are penalized by execution delay.
- **Expected Outcome:**
  Demonstrate that sub-10ms HARP dynamics with selective rescue achieve higher closed-loop task success than heavy iterative planning models.

---

## 3. Work Allocation and Delegation

1. **Step 1 (OpenCode / Muse Spark 1.3):**
   Implement the Pairwise Margin Ranking and Plackett-Luce Listwise Losses in `src/adjointrwm/allocators.py`, add support for normalized score ranking in `AllocatorJob`, and add comprehensive unit tests in `tests/test_allocators.py`.
2. **Step 2 (Local Engine):**
   Verify all unit tests pass, verify `python harness/check.py` 6/6 checks.
3. **Step 3 (Colab Session Execution):**
   Execute Milestone B2.3 (Margin Ranking Allocator) and Milestone B3c (HARP Selective Rescue) on NVIDIA L4.
