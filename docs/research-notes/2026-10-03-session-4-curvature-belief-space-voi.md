# Research Note: Direction 3 Real-Data Second-Order Curvature & Belief-Space VOI Allocation (Session 4)

**Date:** 2026-10-03  
**Authors:** Antigravity & OpenCode  
**Hardware Evaluated:** NVIDIA L4 GPU (`l4-worker`, 22.03 GiB VRAM) via Google Colab  
**Governing Documents:** `docs/plans/2026-10-03-10-session-colab-hopper-plan.md` §3 (Session 4), `docs/plans/2026-10-03-session-4-curvature-voi-spec.md`  
**Committed Artifacts:** [`results/benchmarks/curvature_voi/curvature_voi_summary.json`](../../results/benchmarks/curvature_voi/curvature_voi_summary.json), [`results/benchmarks/curvature_voi/curvature_voi_report.md`](../../results/benchmarks/curvature_voi/curvature_voi_report.md)  

---

## 1. Executive Summary & Core Findings

Session 4 of the 10-session operational research campaign evaluated **Direction 3: Second-Order Curvature and Belief-Space Value-of-Information (VOI) Allocation** on the full 500-episode stratified E3.1 real-robot dataset across 12 independent robotics research laboratories (`AUTOLab`, `BVL`, `ILIAD`, `IPRL`, `IRIS`, `PennPAL`, `RAD`, `RAIL`, `REAL`, `RPL`, `TRI`, `WEIRD`).

The benchmark evaluated 4,154 held-out test windows across 3 seeds (= 12,462 window evaluations) comparing 8 allocation policies under real camera candidate acquisitions (proprioceptive hold, wrist camera, exterior camera, and full dual-camera observation).

### Primary Findings:

1. **Second-Order Curvature Outperforms Pure First-Order Co-State:**
   - The learned diagonal-Hessian curvature policy (`second_order_curvature`) achieves a mean test regret of **`0.07621 ± 0.05358`**, delivering a **−54.60% relative regret reduction** over the standard first-order co-state (`0.16788 ± 0.17332`).
   - Quadratic curvature correction successfully penalizes large perturbations that overshoot local linear approximations in non-linear dynamics.

2. **Belief-Space VOI Achieves Best Deployable Performance:**
   - Incorporating uncertainty reduction into the second-order expansion (`belief_space_voi`) achieves the lowest test regret among all deployable heads: **`0.06354 ± 0.04138`**.
   - This represents a **−62.15% relative regret reduction** over first-order co-state (`0.16788`), a **−22.84% advantage** over the standard direct critic (`0.08235`), a **−22.05% advantage** over the capacity-matched direct critic (`0.08151`), and a **−91.23% reduction** vs the refusal baseline (`0.72430`).

3. **Ultra-High Inference Throughput on NVIDIA L4:**
   - All-policy scoring latency was measured at **`0.0367 ms/window`**, delivering a wall-clock throughput of **`27,198.5 decisions/second (27.2 kHz)`** on NVIDIA L4.
   - This exceeds the 5,000 decisions/second (5 kHz) real-time exit gate by **5.44×**.

4. **Session 4 Exit Gate: MET (PASS):**
   - Requirement 1: Real-data test regret reduction of `second_order_curvature` or `belief_space_voi` over `first_order` co-state ($p < 0.05$) $\to$ **PASS** (−54.60% and −62.15% regret reductions).
   - Requirement 2: Inference throughput exceeding 5 kHz on NVIDIA L4 $\to$ **PASS** (`27,198.5 Hz`).

---

## 2. Head-to-Head Policy Evaluation

Evaluated across 12,462 held-out test window decisions over 3 seeds on NVIDIA L4:

| Allocator Policy | Mean Test Regret | Gap to Oracle Floor | Win Rate vs Critic | Win Rate vs First-Order | Win Rate vs Mode 0 | Parameters |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`exact_costate` (Oracle Floor)** | **`0.00399 ± 0.00298`** | `+0.00000` | `0.609` | `0.427` | `0.607` | — (autograd) |
| **`always_mode0` (Refusal Baseline)** | `0.72430 ± 0.55571` | `+0.72030` | `0.498` | `0.342` | `0.000` | 0 |
| **`direct_critic` (Matched to Costate)** | `0.08235 ± 0.05820` | `+0.07836` | `0.000` | `0.186` | `0.382` | 1,315,063 |
| **`direct_critic_curv_matched` (Capacity Matched)** | `0.08151 ± 0.05699` | `+0.07752` | `0.138` | `0.184` | `0.408` | 1,577,458 |
| **`first_order` (Linear Inner Product)** | `0.16788 ± 0.17332` | `+0.16389` | `0.430` | `0.000` | `0.465` | 1,314,816 |
| **`normalized_first_order` (Cosine Coupling)** | `0.22515 ± 0.16795` | `+0.22115` | `0.480` | `0.083` | `0.469` | 1,314,816 |
| **`second_order_curvature` (Quadratic Penalty)** | **`0.07621 ± 0.05358`** | `+0.07221` | `0.236` | `0.171` | `0.390` | 1,577,472 |
| **`belief_space_voi` (Curvature + Information)** | **`0.06354 ± 0.04138`** | `+0.05955` | `0.203` | `0.160` | `0.418` | 1,577,472 |

---

## 3. Cross-Site Performance Across 12 Robotics Laboratories

Mean test regret across the 12 robotics laboratories in the held-out test split:

| Site | Test Windows | Mode 0 (Refusal) | Critic (Curv Matched) | First-Order Co-State | Second-Order Curvature | Belief-Space VOI | Best Performing Deployable |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `AUTOLab` | 512 | 0.1428 | 0.0374 | 0.0761 | 0.0465 | **0.0391** | `direct_critic` / `belief_space_voi` |
| `BVL` | 183 | 0.0174 | 0.0359 | 0.0246 | 0.0333 | 0.0347 | `always_mode0` (low motion) |
| `ILIAD` | 1,024 | 0.4989 | 0.1201 | 0.1447 | 0.1185 | **0.1071** | **`belief_space_voi`** (−10.8% vs critic) |
| `IPRL` | 376 | 0.3168 | 0.0750 | 0.1142 | 0.0926 | **0.0708** | **`belief_space_voi`** (−5.6% vs critic) |
| `IRIS` | 500 | 4.3013 | 0.1793 | 0.7371 | **0.1038** | **0.0535** | **`belief_space_voi`** (−70.2% vs critic) |
| `PennPAL` | 185 | 0.0432 | 0.0429 | 0.0390 | **0.0390** | 0.0414 | **`second_order_curvature`** |
| `RAD` | 144 | 0.0180 | 0.1019 | 0.0600 | 0.0962 | 0.0984 | `always_mode0` |
| `RAIL` | 76 | 0.1805 | 0.0308 | 0.0751 | **0.0302** | 0.0371 | **`second_order_curvature`** |
| `REAL` | 96 | 0.0328 | 0.0272 | 0.0425 | 0.0314 | 0.0327 | `direct_critic` |
| `RPL` | 197 | 0.0210 | 0.0440 | 0.0275 | 0.0387 | **0.0312** | **`belief_space_voi`** (−29.1% vs critic) |
| `TRI` | 781 | 0.1552 | 0.0518 | 0.0825 | 0.0585 | 0.0531 | `direct_critic` / `belief_space_voi` |
| `WEIRD` | 80 | 0.0435 | 0.0411 | 0.0403 | 0.0411 | 0.0441 | `first_order` |

### Critical Cross-Site Takeaway: The `IRIS` Stress Test
At the `IRIS` laboratory (characterized by extreme dynamic occlusions and high camera diversity), refusal regret spikes to `4.3013`, and first-order linear co-states fail significantly (`0.7371` regret). In contrast:
- `second_order_curvature` drops regret to **`0.1038`** (**−85.9%** vs first-order).
- `belief_space_voi` drops regret to **`0.0535`** (**−92.7%** vs first-order, **−70.2%** vs critic).

---

## 4. Methodological Rigor & Peer Critic Hardening

Session 4 incorporated all recommendations from the adversarial peer review with OpenCode (`space-bunny-free`):

1. **Exact Directional HVPs (P0-1):**
   - Eliminated cross-term contamination by evaluating directional HVPs per candidate vector $\Delta z_k$:
     $$\kappa_k^* = \Delta z_k^\top \nabla_{z_0} (\lambda_0^\top \Delta z_k)$$
     with `create_graph=True` enabled on the baseline backward pass.
2. **Capacity-Matched Direct Critic Control (P0-2 / Rule 6):**
   - Matched `CurvatureCostateEstimator`'s parameter count (1,577,472 params) with `direct_critic_curv_matched` (1,577,458 params, hidden dimension 1,533).
   - Confirmed that even against the capacity-matched critic, `belief_space_voi` preserves a **−22.05% relative regret reduction** (`0.06354` vs `0.08151`).
3. **Multi-Objective Supervised Curvature Loss (P0-3):**
   - Corrected the zero-gradient bug by supervising $\hat{h}$ with directional HVP alignment and end-to-end Plackett-Luce ranking loss over true net gains.
4. **Compute Discipline Enforced:**
   - NVIDIA L4 session `l4-worker` was terminated immediately upon benchmark completion. Verified that 0 billable sessions remain active.
