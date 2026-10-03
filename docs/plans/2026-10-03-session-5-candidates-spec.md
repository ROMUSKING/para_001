# Session 5 Specification: Regret Robustness, Cost Sensitivity & Multi-Horizon Generalization on NVIDIA L4

**Date:** 2026-10-03  
**Status:** APPROVED (Post Peer Critic Review)  
**Author:** Antigravity (Lead Agent)  
**Peer Critic:** OpenCode (`space-bunny-free`)  
**Target Milestone:** Session 5 (10-Session Operational Research Campaign)  
**Target Hardware:** Google Colab NVIDIA L4 GPU (22.03 GiB VRAM, Active Session `l4-worker`)  

---

## 1. Context & Hardware Gate Audit

In the 10-Session Colab campaign (`docs/plans/2026-10-03-10-session-colab-hopper-plan.md`):
- **Session 0:** Established profiler baseline (Gate G4-1: wait 21% > 15%).
- **Session 1:** Proved $H=64, B=64$ double-backprop uses only 904 MiB VRAM.
- **Session 2:** Confirmed representation headroom of multi-token spatial representations (−48.73% RMSE, Gate G4-2 pilot).
- **Session 3:** Validated HARP selective analytical rescue on the 500-episode multi-site E3.1 shard across 12 laboratories (−42.9% regret vs critic, 6.2 kHz throughput).
- **Session 4:** Validated second-order curvature (−54.6% regret) and belief-space VOI (−62.15% regret vs first-order, −22.84% vs critic) on real E3.1 multi-site data at 27.2 kHz throughput.

### G4 Switchover Gate Audit:
Per §4 of the campaign plan, upgrading to Hopper G4 (96 GiB) requires all three entry gates:
- Gate G4-1 (Profiler saturation): FAIL (data loading is not GPU-bound).
- Gate G4-2 (Representation value): PASS (pilot representation gain).
- Gate G4-3 (Physical L4 OOM): FAIL (peak VRAM remains < 1.2 GiB on L4).

**Mandatory Hardware Policy Decision:** The campaign **strictly remains on NVIDIA L4 for Session 5** (logged in `prereg/deviation_log.yaml`).

---

## 2. Peer Critic Synthesis & Decision

Antigravity submitted Candidates A, B, and C to OpenCode (`space-bunny-free`) for adversarial peer review under the Peer Critic Protocol.

### Peer Critic Findings & Recommendations:
1. **Candidate C is Blocked on Evidence Inconsistency:** OpenCode discovered a hard-coded verdict string in `results/benchmarks/repo_code_gen/repo_code_gen_summary.json` claiming "47.1% token savings" while the computed metric is `-16.5%`. Furthermore, cross-domain FLOP-regret unification is a category error across continuous control and discrete DAG repair.
2. **Session 4 Advantage is Heavily Site-Dependent:** A leave-one-site-out analysis revealed that dropping laboratory `IRIS` reduces the headline advantage of `belief_space_voi` vs `direct_critic` from −22.8% to −6.0%. IRIS accounts for 71.5% of the refusal regret mass. Per-site breakdowns and leave-one-site-out metrics are mandatory.
3. **Regret Metric Depends on Hand-Set Cost Vector:** $c_k = (0, 1.0, 1.0, 1.5, 2.0) \times 0.002$ was unmeasured. A cost-sensitivity sweep is required to prove that the adjoint/curvature advantage is not an artifact of the cost weight or modality penalty ratios.
4. **Dead Conditioning Feature Bug:** In `scripts/benchmark_curvature_voi_allocator.py`, `horizon` was set to `torch.ones(B)`, meaning the horizon input was dead across all heads. True horizon conditioning $H \in \{2, 4, 8\}$ must be passed as dynamic input.
5. **Recommendation:** OpenCode recommended rejecting A, B, and C in favor of **Candidate D: Regret Metric Robustness, Cost Sensitivity & Multi-Horizon Generalization**.

### Lead Agent Disposition:
- **P0 #1 (G4-2 Pilot Citing):** ACCEPT. Retain L4; do not extrapolate pilot findings to justify G4.
- **P0 #2 (Leave-One-Site-Out & Per-Site Win Rates):** ACCEPT. Implement full LOSO panel across all 12 sites and win-rate statistics.
- **P0 #3 (Cost Model Sensitivity Sweep):** ACCEPT. Evaluate 5 distinct cost regimes (Zero-Cost, Uniform, Default, High-Penalty, Latency-Weighted).
- **P0 #4 (Model-Internal Oracle Semantics):** ACCEPT. Explicitly report that $G_k$ is the model's rollout objective $J$.
- **P0 #5 (Track D Hardcoded Metric):** ACCEPT & REMEDIATED. Created `docs/audits/2026-10-03_repo_code_gen_verdict_audit.md` and corrected `scripts/run_repo_code_generation_benchmark.py`.
- **P1 #10 (Horizon Conditioning Bug):** ACCEPT & FIXED. Pass variable `horizon` tensor to all heads.

---

## 3. Session 5 Execution Plan: Candidate D

### Experimental Grid:
1. **Model Backbone:** Frozen `hybrid_adjoint_rwm` teacher checkpoints (`runs/harp_hybrid_dynamics_20261002/`).
2. **Dataset:** Stratified held-out test windows ($N = 4,154 \times 3 \text{ seeds} = 12,462 \text{ evaluations}$) across 12 robotics laboratories on DROID E3.1 shard.
3. **Horizon Sweep:** $H \in \{2, 4, 8\}$ with dynamic `horizon` tensor conditioning.
4. **Cost Regimes Evaluated:**
   - **Regime 0 (Zero Cost):** $c_k = (0, 0, 0, 0)$ (Pure predictive value-of-information).
   - **Regime 1 (Uniform Cost):** $c_k = (0, 1, 1, 1) \times 0.002$.
   - **Regime 2 (Default Cost):** $c_k = (0, 1.0, 1.0, 1.5, 2.0) \times 0.002$.
   - **Regime 3 (High Active Penalty):** $c_k = (0, 1.0, 1.0, 3.0, 5.0) \times 0.002$.
   - **Regime 4 (Latency-Weighted):** $c_k = (0, \text{lat}_1, \text{lat}_2, \text{lat}_3) \times 0.002$.
5. **Statistical Robustness Panel:**
   - Mean regret, standard deviation.
   - Median regret (tail-insensitive).
   - 10% trimmed mean regret.
   - Paired bootstrap 95% confidence intervals (1,000 resamples).
   - Window-level win rate vs direct critic and vs capacity-matched critic.
   - Leave-one-site-out (LOSO) regret delta for all 12 robotics laboratories.
