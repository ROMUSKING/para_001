# Session 4 Specification: Real-Data Second-Order Curvature & Belief-Space VOI Allocation on NVIDIA L4

**Date:** 2026-10-03  
**Status:** APPROVED (Post Peer Critic Review)  
**Author:** Antigravity (Lead Agent)  
**Peer Critic:** OpenCode (`space-bunny-free`)  
**Target Milestone:** Session 4 (Roadmap B3c / Direction 3)  
**Target Runtime:** Google Colab NVIDIA L4 GPU (22.03 GiB VRAM)  

---

## 1. Executive Summary & Context

Session 4 advances the empirical investigation of **Direction 3: Second-Order Curvature and Belief-Space VOI Allocation**. In previous sessions:
- **Session 0:** Established baseline data loader profiling and CUDA utilization (Gate G4-1: wait > 15%).
- **Session 1:** Established horizon stress scaling, proving that second-order Hessian-vector products (HVP) on pooled latents are memory-light (<1 GiB VRAM for $H=64, B=64$).
- **Session 2:** Confirmed representation headroom of multi-token spatial representations (−48.73% RMSE, Gate G4-2 PASS).
- **Session 3:** Validated HARP selective analytical rescue on the 500-episode multi-site E3.1 shard across 12 robotics laboratories, demonstrating −42.9% regret reduction over the matched critic at >6.2 kHz throughput.

Session 4 evaluates whether **second-order curvature ($\nabla_z^2 J$) and belief-space value-of-information (VOI)** provide superior decision allocation over pure first-order gradient co-states ($\nabla_z J$) and matched direct critics when allocating non-linear sensory acquisitions on real multi-site robotics data.

---

## 2. Peer Critic Review & Lead Agent Decisions

Prior to final implementation, this specification was submitted to OpenCode (`space-bunny-free`) as an adversarial Peer Critic. The critique produced four key findings:

1. **P0-1: HVP Autograd Mathematical Formulation:**
   - *Critique:* The initial HVP snippet omitted `create_graph=True` on the first backward pass (triggering a PyTorch runtime error) and took `autograd.grad(inner.sum(), z0)`, which sums across all candidate directions before differentiating. By linearity, this contaminated the directional curvature $\kappa_k = \Delta z_k^\top H \Delta z_k$ with cross-terms $\sum_{j \neq k} \Delta z_k^\top H \Delta z_j$, producing up to 191× distortion.
   - *Decision:* **ACCEPTED**. Autograd HVP now explicitly sets `create_graph=True` and computes per-direction projections along each candidate vector $\Delta z_k$ individually, yielding exact directional second derivatives $\Delta z_k^\top H \Delta z_k$ with zero cross-term distortion.
2. **P0-2: Parameter-Matching Discipline (Rule 6):**
   - *Critique:* `CurvatureCostateEstimator` possesses 1,577,472 parameters (+20.0% over `CostateEstimator`'s 1,314,816 parameters) due to the diagonal Hessian head. Comparing it only against a 1.315M parameter critic creates a 16.6% capacity shortfall, risking a violation of Research Integrity Rule 6.
   - *Decision:* **ACCEPTED**. We evaluate two direct critic baselines: `direct_critic` (first-order matched, 1.315M parameters) and `direct_critic_curv_matched` (hidden dimension scaled to 1,536 to match 1.577M parameters exactly).
3. **P0-3: Zero-Gradient Bug in Prior Curvature Script:**
   - *Critique:* In the legacy script, `heads["curvature"]` was trained purely with cosine similarity on $\hat{\lambda}$, leaving `hessian_head.weight` and `hessian_head.bias` with `grad = None` (zero gradient).
   - *Decision:* **ACCEPTED**. Supervise $\hat{h}$ via a multi-objective loss combining exact directional HVP curvature alignment, Plackett-Luce decision ranking loss over true net gains, and first-order co-state cosine similarity.
4. **P1-1: Belief-Space VOI Uncertainty Proxy:**
   - *Critique:* Using $\Delta \Sigma_k \approx (\Delta z_k)^2$ directly conflates mean perturbation with uncertainty collapse.
   - *Decision:* **PARTIALLY ADOPTED**. We benchmark both the documented squared perturbation proxy $\Delta \Sigma_k = (\Delta z_k)^2$ and the normalized empirical feature variance across context timesteps, documenting both clearly.

---

## 3. Mathematical Formulation & Allocation Policies

### 3.1 Taylor Expansion & Curvature Correction
Let $z_0$ denote the baseline latent state under Mode 0 (proprioceptive hold). When considering candidate sensory refinement $k \in \{1, \dots, K\}$ with latent perturbation $\Delta z_k = z_k - z_0$ and cost $c_k$, the second-order Taylor expansion of prediction loss $J(z_k)$ around $z_0$ is:
$$J(z_0 + \Delta z_k) \approx J(z_0) + \nabla_z J(z_0)^\top \Delta z_k + \frac{1}{2} \Delta z_k^\top \nabla_z^2 J(z_0) \Delta z_k$$

The net gain of candidate $k$ over hold is:
$$G_k = J(z_0) - J(z_k) - c_k \approx -\lambda^\top \Delta z_k - \frac{1}{2} \Delta z_k^\top H \Delta z_k - c_k$$
where $\lambda = \nabla_z J(z_0)$ is the first-order co-state and $H = \nabla_z^2 J(z_0)$ is the Hessian.

### 3.2 Diagonal Hessian Parameterization
To maintain $\mathcal{O}(D)$ latency and memory scaling, we parameterize the Hessian as a non-negative diagonal matrix $H \approx \operatorname{diag}(h)$, with $h = \operatorname{softplus}(W_h \phi(z_0)) \ge 0$.
The second-order curvature score is:
$$s_k^{(2nd)} = -\lambda^\top \Delta z_k - \frac{1}{2} \sum_{d=1}^D h_d (\Delta z_{k, d})^2 - c_k$$
with $s_0^{(2nd)} \equiv 0$ (hold option).

### 3.3 Belief-Space Value of Information (VOI)
Under state uncertainty $\Sigma_0 = \operatorname{Cov}(z_0)$, acquiring sensor modality $k$ reduces posterior entropy/covariance by $\Delta \Sigma_k$. In belief space, the expected value of information balances mean reduction with uncertainty reduction weighted by curvature:
$$s_k^{(VOI)} = -\lambda^\top \Delta z_k + \frac{1}{2} \kappa_{uncert} \sum_{d=1}^D h_d \Delta \Sigma_{k, d} - c_k$$
where $\kappa_{uncert} \in [0.1, 1.0]$.

### 3.4 Evaluated Policies
1. `exact_costate` (Oracle Floor): first-order scores using autograd co-state $\lambda^* = \nabla_{z_0} J(z_0)$.
2. `always_mode0` (Refusal Baseline): always hold proprio-only, gain $\equiv 0$.
3. `direct_critic`: parameter-matched MLP (1.315M params) mapping $(z_0, \Delta z_k, c_k) \mapsto \hat{G}_k$.
4. `direct_critic_curv_matched`: capacity-matched MLP (1.577M params) matching `CurvatureCostateEstimator`.
5. `first_order`: learned $\hat{\lambda}$ with inner product $-\hat{\lambda}^\top \Delta z_k - c_k$.
6. `normalized_first_order`: learned $\hat{\lambda}$ with cosine coupling.
7. `second_order_curvature`: learned $(\hat{\lambda}, \hat{h})$ with diagonal quadratic curvature penalty.
8. `belief_space_voi`: learned $(\hat{\lambda}, \hat{h})$ with belief-space VOI information bonus.

---

## 4. Ground-Truth Curvature Supervision & Training Objectives

1. **Exact Directional Curvature via Autograd HVP:**
   For $z_0$:
   ```python
   z0_req = z0.detach().clone().requires_grad_(True)
   with torch.enable_grad():
       p0_req = model.rollout(z0_req, future_actions, context_state=context_state)
       loss_0 = prediction_objective(p0_req, target_state, target_visual).sum()
       lambda0 = torch.autograd.grad(loss_0, z0_req, create_graph=True)[0]
       
       exact_curvatures = []
       for k in range(1, num_candidates):
           dz_k = effects[:, k]
           inner_k = torch.sum(lambda0 * dz_k)
           hvp_k = torch.autograd.grad(inner_k, z0_req, retain_graph=True)[0]
           curv_k = torch.sum(hvp_k * dz_k, dim=-1).detach()
           exact_curvatures.append(curv_k)
       exact_curvatures = torch.stack(exact_curvatures, dim=1) # [B, K-1]
   ```
2. **Multi-Objective Loss:**
   $$\mathcal{L} = \mathcal{L}_{cos}(\hat{\lambda}, \lambda^*) + \alpha \mathcal{L}_{curv}(\hat{h} \odot \Delta z_k^2, \kappa_k^*) + \beta \mathcal{L}_{rank}(\hat{s}, G^*)$$
   where $\mathcal{L}_{rank}$ is the Plackett-Luce ranking loss over true net gains $G^*$.

---

## 5. Empirical Evaluation Protocol

- **Dataset:** Stratified E3.1 shard on Google Colab (`/content/cache_e3_1`), comprising 500 episodes across 12 robotics laboratories:
  AUTOLab, BVL, ILIAD, IPRL, IRIS, PennPAL, RAD, RAIL, REAL, RPL, TRI, WEIRD.
- **Test Split:** 50 held-out episodes, 4,154 test windows (evaluated over 3 seeds = 12,462 window evaluations).
- **Candidate Modalities:**
  - Mode 0: Proprioceptive hold ($c=0$)
  - Mode 1: Wrist camera ($c=0.001$)
  - Mode 2: Exterior camera ($c=0.001$)
  - Mode 3: Both cameras ($c=0.002$)
- **Primary Metrics:**
  - Mean test regret $\mathbb{E}[G^* - G_\pi]$ (mean ± std across seeds).
  - Regret reduction vs `first_order` and vs `direct_critic` (%).
  - Win rate vs `direct_critic` and vs `always_mode0`.
  - Wall-clock decision throughput (decisions / sec) and per-window latency on NVIDIA L4.
- **Session 4 Exit Gate:**
  - Real-data test regret reduction of `second_order_curvature` or `belief_space_voi` over `first_order` co-state ($p < 0.05$).
  - Decision throughput exceeding 5,000 decisions/second (5 kHz) on NVIDIA L4.
