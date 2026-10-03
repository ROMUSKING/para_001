# Session 3 Benchmark Specification: HARP Selective Analytical Rescue on Full E3.1 Real Shard

**Date:** 2026-10-03  
**Status:** ACTIVE EXECUTION  
**Milestone:** Session 3 / Milestone B3c  
**Target Hardware:** NVIDIA L4 GPU (`l4-worker`, 22.03 GiB VRAM)  
**Governing Documents:** `docs/plans/2026-10-03-10-session-colab-hopper-plan.md`, `docs/plans/2026-10-02-next-research-steps-plan.md`, `AGENTS.md`

---

## 1. Research Question & Objective

### Question
Can decision-margin confidence gating ($\tau \in [0.0, 1.0]$) and analytical rescue dynamically bridge the amortization gap across 12 diverse robotics research laboratories on the full E3.1 stratified real-data shard, establishing a high-throughput, low-regret Pareto frontier?

### Background & Prior Evidence
- **Milestone B3 (DROID-100):** Rescuing 20% of ambiguous decisions ($\tau = 0.20$) reduced regret by ~20% toward the autograd oracle bound while preserving >7,600 Hz throughput (`docs/research-notes/2026-10-02-b3-analytical-rescue-interface.md`).
- **Milestone B2.3:** Continuous ranking losses (margin, listwise) failed to close the amortization gap on average (+0.0008 to +0.0048 vs CE), confirming that selective analytical rescue—rather than loss reshaping—is the mathematically rigorous mechanism to achieve near-oracle performance under amortized capacity limits (`docs/research-notes/2026-10-02-b2-3-ranking-allocator-benchmark.md`).

---

## 2. Experimental Design & Data Protocol

### 2.1 Multi-Site Stratified Shard (E3.1)
- Evaluated on `/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production/data/cache_e3_1`.
- 499 cached real-robot episodes spanning 14 robotics laboratories (`TRI`, `AUTOLab`, `IRIS`, `ILIAD`, `IPRL`, `BVL`, `RAIL`, `REAL`, `WEIRD`, etc.).
- Evaluated on 4,154 held-out test windows across 12 laboratories (exact split parity with B3b confirmatory benchmark).

### 2.2 Model & Allocator Architecture
- **HARP Dynamics Backbone:** `HybridAdjointRecursiveWorldModel` (24.7M parameters) combining linear kinematic continuation ($\hat{s}^{\text{kin}}_{h+1} = \hat{s}_h + A_v v_h + B_u u_h$) with a 6-layer Transformer-GRU residual and Pontryagin co-state sensitivity.
- **Decision Margin Gating:**
  $$m(x) = s_{(1)} - s_{(2)}$$
  Windows where $m(x) \le \tau_{\text{quantile}}$ trigger exact analytical autograd co-state rollouts.
- **Sensory Modality Allocation Modes:**
  1. Mode 0: Proprioception only (Refusal baseline, zero sensory cost).
  2. Mode 1: Wrist camera only ($c_1 = 0.05$).
  3. Mode 2: Exterior camera only ($c_2 = 0.05$).
  4. Mode 3: Full Observation ($c_3 = 0.10$).

---

## 3. Evaluation Endpoints & Exit Gate

1. **Primary Metric:** Net allocation regret ($R_{\text{alloc}} = \mathbb{E}[J_{\text{oracle}} - J_{\text{policy}}]$) across all 12 test laboratories.
2. **Pareto Frontier:** Regret vs Effective Inference Latency (ms) / Throughput (Hz) across $\tau \in \{0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00\}$.
3. **Exit Gate (Session 3):**
   - Amortization gap closed below $\tau = 0.20$ threshold ($R_{\tau=0.20} < R_{\text{refusal}}$ and $R_{\tau=0.20} < R_{\text{amortized}}$).
   - Effective decision throughput exceeding 5,000 decisions/second (>5 kHz).
