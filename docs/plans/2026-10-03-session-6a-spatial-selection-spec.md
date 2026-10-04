# Session 6A Specification: Fixed-Budget Spatial Patch Selection Benchmark

**Date:** 2026-10-03  
**Status:** APPROVED (Finalized post Peer Critic Protocol Review with OpenCode `space-bunny-free`)  
**Lead Agent:** Antigravity  
**Peer Critic:** OpenCode (`space-bunny-free`)  
**Target Milestone:** Session 6A (Spatial Value-of-Information at Fixed Token Budgets)  
**Preceding Milestone:** Stage 5.1 Evidence Reconciliation (Audit PASS, `docs/audits/2026-10-03_session_5_loso_and_cost_reconciliation_audit.md`)  

---

## 1. Research Question & Experimental Objective

Following the user's directive and the Session 5 audit:
> *"At a fixed processing budget, does belief-space sensitivity select more useful visual regions than a competent direct gain predictor, curvature, and inexpensive token-selection methods?"*

Spatial allocation must be established at fixed patch budgets **before** introducing policy distillation (Session 6B) or end-to-end hardware acceleration (Session 6C).

### Core Research Question
When the visual patch budget is held strictly constant ($k \in \{4, 8, 16\}$ total patches retained out of $P = 32$ spatial tokens, partitioned symmetrically as $k/2$ per camera to isolate spatial selection from camera modality selection), does an adjoint-guided co-state selector allocate spatial compute to decision-critical image regions more effectively than:
1. Deterministic spatial coverage (center / uniform grid)?
2. Spatially stratified random selection?
3. Inexpensive early visual feature saliency (raw DINOv2 patch norm / CLS-attention energy)?
4. A capacity-matched direct ranking critic (decision-focused learning)?
5. A privileged direct critic (given $\hat{\lambda}, \hat{H}, \Delta z$ as inputs)?
6. Second-order curvature selection?

---

## 2. Peer Critic Synthesis & Architectural Design Decisions

Under the `AGENTS.md` Peer Critic Protocol, OpenCode (`space-bunny-free`) conducted an adversarial review of the initial draft. All major recommendations were adopted with documented rationales:

| Issue Flagged by Peer Critic | Rationale & Adopted Resolution |
|---|---|
| **B1: Non-additive $\Delta z_p$ & Information Boundary** | In `SpatialPatchAdapter`, spatial tokens are pooled with a single learned query: $\text{pooled} = \operatorname{Attn}(q, h, h)$. Computing $\Delta z_p$ via $P$ individual rollouts violates the information boundary. **Adopted:** We use the exact single-query attention perturbation identity: $\Delta \text{pooled}_p = \alpha_p (v_p - \text{pooled})$. A **single backward pass** through temporal encoder and rollout yields cotangent $g = \partial J / \partial \text{tokens}$, scoring all $P$ patches in $O(1)$ forward + backward operations. |
| **B1: Attention Head & Sign Conventions** | MHA default `average_attn_weights=True` hides per-head weights; spec sign was $+ \lambda^\top \Delta z$. **Adopted:** Configure `average_attn_weights=False`; strictly enforce repo sign convention: Gain $= -\lambda^\top \Delta z - \frac{1}{2} \Delta z^\top H \Delta z$. |
| **B2: True Epistemic Belief-Space VOI** | In Session 4/5, `delta_cov = effects.pow(2)`, reducing VOI and curvature to the same functional family up to a sign flip. **Adopted:** Compute true epistemic variance reduction $\Delta \Sigma_p = \operatorname{diag}(\sigma^2(z_S)) - \operatorname{diag}(\sigma^2(z_{S \cup p}))$ using the world model's own predictive variance head (`state_logvar_head` from `rollout()`). Include $\beta \in \{0, 0.5, 1\}$ and $-\beta$ sign-flip controls. |
| **B3: Privilege & Primary Comparator** | Deployable policies must use distilled $\hat{\lambda} = \text{CurvatureCostateEstimator}(z)$, not autograd $\lambda_0$ (which leaks future targets). A direct critic seeing only early features is handicapped. **Adopted:** Add `direct_critic_privileged` (fed $\hat{\lambda}, \hat{H}, \Delta z$) as the primary comparator. Structure evaluation into Tier 0 (cheap heuristics), Tier 1 (deployable $O(P)$ scorers), and Tier 2 (non-deployable ceilings). |
| **B4: Raw vs Normalized Early Norm** | Scoring post-LayerNorm patch embeddings causes norm collapse. **Adopted:** Score raw pre-LayerNorm DINOv2 patch token norms and CLS self-attention energy. |
| **B5: Combinatorial Explosion & Camera Entanglement** | $\binom{32}{8} \approx 10.5 \times 10^6$ combinations. Unconstrained top-$k$ can empty one camera, confounding spatial selection with modality selection. **Adopted:** Partition budget symmetrically as $k/2$ patches per camera ($P_{\text{cam}} = 16$, $k_{\text{cam}} \in \{2, 4, 8\}$). Use greedy/Lazy-Greedy search with an optimality gap calibrated against the exhaustive oracle on small subsets ($k=2, 3$). |
| **B6: Spatial World Model & Patch-Dropout Prerequisite** | Existing 12-site checkpoint is pooled ResNet18. Evaluating a model trained on $P=32$ at $k=4$ without patch-dropout causes distribution shift. **Adopted:** Pre-train / fine-tune spatial backbone with random patch-dropout ($k \sim \mathcal{U}\{4, 8, 16, 32\}$) on the 12-site DROID shard. |
| **Q4: Deferring Latency Claims** | Profiler indicates DROID dataloader is IO-bound (83-95% wait). **Adopted:** Defer wall-clock system latency claims to Session 6C; evaluate Session 6A strictly on prediction loss and decision regret at fixed token budgets. |

---

## 3. Policy Suite & Evaluation Tiers

Total spatial patches: $P = 32$ (16 wrist camera + 16 exterior camera, DINOv2 ViT-S/14 Grid 4×4).  
Evaluated budgets: $k \in \{4, 8, 16\}$ total patches ($k_{\text{cam}} \in \{2, 4, 8\}$ patches per camera).

### Tier 0: Inexpensive Heuristics ($O(P)$ Deployable, No Learned Models)
1. `uniform_grid`: Deterministic uniform spatial subsampling (e.g. 4 corners at $k_{\text{cam}}=4$).
2. `stratified_random`: Random patch selection stratified across the 4 spatial quadrants.
3. `early_feature_norm`: Score patches by raw DINOv2 patch token $L_2$ norm.
4. `early_cls_attention`: Score patches by DINOv2 ViT last-layer self-attention weight from CLS token to patch token.

### Tier 1: Deployable Learned Allocators ($O(P)$ Deployable, No Future Targets)
5. `direct_ranking_critic`: Capacity-matched MLP predicting patch utility scores, trained via pairwise margin ranking loss against true downstream loss reduction.
6. `direct_critic_privileged`: Capacity-matched MLP receiving $[\hat{\lambda}, \hat{H}, \Delta z_p]$ as inputs, trained to rank patches. (Decisive comparator: tests whether VOI adds value over learned non-linear combination of co-state features).
7. `second_order_curvature`: Ranks patches via $-\hat{\lambda}^\top \Delta z_p - \frac{1}{2} \Delta z_p^\top \hat{H} \Delta z_p$ using distilled $\hat{\lambda}$ and $\hat{H}$.
8. `belief_space_voi`: Ranks patches via $-\hat{\lambda}^\top \Delta z_p - \frac{1}{2} \Delta z_p^\top \hat{H} \Delta z_p + \frac{\beta}{2} \Delta \Sigma_p^\top \hat{H} \Delta \Sigma_p$ where $\Delta \Sigma_p$ is derived from the model's predictive logvar head.

### Tier 2: Non-Deployable References (Performance Ceilings & Diagnostics)
9. `exact_costate_reference`: First-order autograd co-state $\lambda_0 = \partial J / \partial z$ (requires ground-truth future targets; diagnostic ceiling for $\hat{\lambda}$ distillation).
10. `calibrated_greedy_oracle`: Greedy search evaluating full rollouts for patch additions (calibrated against exhaustive oracle on $k=2, 3$).

---

## 4. Evaluation Metrics & Statistical Analysis

1. **Downstream Multi-Step Prediction Loss:** $J(z_S)$ on held-out horizon $H=4$ targets ($t+1 \dots t+4$).
2. **Decision Regret:** $r(S) = J(z_S) - J(z_{\text{oracle}})$.
3. **Primary Statistical Metrics:**
   - 10% Trimmed Mean Regret (primary headline).
   - Median Regret (outlier-robust).
   - Paired Wilcoxon Signed-Rank Test on $d_i = r_i^{\text{VOI}} - r_i^{\text{direct\_ranking}}$.
   - Site-Clustered Bootstrap 95% Confidence Intervals (resampling over the 12 laboratories to account for intra-site correlation).
4. **Diagnostic Metrics:**
   - Additivity $R^2$: Linear regression $R^2$ of $J(S)$ on $\sum_{p \in S} s_p$, separating non-additivity error from score ranking error.
   - Distillation Gap: Regret difference between exact reference $\lambda_0$ and distilled $\hat{\lambda}$.
   - Submodularity Violation Rate: Frequency of diminishing marginal returns violations across nested patch subsets.

---

## 5. Pre-Registered Exit Gate

- **Criterion 1 (Primary Advantage):** Belief-space VOI must achieve a lower 10% trimmed mean regret than `uniform_grid` and `early_feature_norm` by $\ge 8.0\%$ across evaluated budgets.
- **Criterion 2 (Non-Inferiority vs Privileged Critic):** Belief-space VOI must be non-inferior to `direct_critic_privileged` (within $3.0\%$ margin on trimmed mean regret).
- **Tie / Inconclusive Branch:** If VOI and curvature tie within statistical error ($p > 0.05$ on paired Wilcoxon), conclude that second-order curvature captures the dominant spatial allocation signal without requiring epistemic covariance reduction.
- **Negative Result Branch:** If `early_feature_norm` or `uniform_grid` matches or outperforms learned allocators, report plainly that spatial patch allocation is dominated by geometric coverage rather than downstream sensitivity.
