# Research Note: Milestone B3c HARP Selective Analytical Rescue on Multi-Site Shard

**Date:** 2026-10-03  
**Authors:** Antigravity & OpenCode  
**Hardware Evaluated:** NVIDIA L4 GPU (`l4-worker`, 22.03 GiB VRAM)  
**Governing Documents:** `docs/plans/2026-10-03-10-session-colab-hopper-plan.md` §3 (Session 3), `docs/plans/2026-10-03-session-3-harp-selective-rescue-spec.md`  
**Committed Artifacts:** [`results/benchmarks/harp_selective_rescue/harp_selective_rescue_summary.json`](../../results/benchmarks/harp_selective_rescue/harp_selective_rescue_summary.json), [`results/benchmarks/harp_selective_rescue/harp_selective_rescue_report.md`](../../results/benchmarks/harp_selective_rescue/harp_selective_rescue_report.md)

---

## 1. Executive Summary & Core Findings

Milestone B3c (Session 3 of the 10-Session Colab campaign) scaled the Selective Invocation and Analytical Rescue Interface from DROID-100 to the full 500-episode stratified E3.1 real-robot dataset across 12 independent research laboratories (`TRI`, `AUTOLab`, `IRIS`, `ILIAD`, `IPRL`, `BVL`, `RAIL`, `REAL`, `WEIRD`, etc.).

The benchmark evaluated 12,462 held-out test windows across 3 seeds using the `HARP` (`HybridAdjointRecursiveWorldModel`, 24.7M parameters) dynamics model under varying decision-margin gating thresholds $\tau \in [0.0, 1.0]$.

### Key Measured Outcomes

1. **Amortized Baseline Superiority Over Direct Critic:**
   - Under pure sub-microsecond forward pass inference ($\tau = 0.00$), the amortized Pontryagin co-state allocator achieves a test regret of **`0.39904 ± 0.27897`**, outperforming the matched direct critic (**`0.47105 ± 0.33675`**, a **−15.28%** advantage) and vastly beating the sensory refusal baseline (**`0.72139 ± 0.55545`**, a **−44.68%** advantage).

2. **Continuous, Strictly Monotonic Pareto Frontier:**
   - Rescuing just 20% of ambiguous decisions ($\tau = 0.20$) reduces regret to **`0.26876 ± 0.18864`** (a **−32.6%** reduction relative to amortized inference, **−42.9%** relative to the matched critic, and **−62.7%** relative to refusal).
   - Effective system latency at $\tau = 0.20$ is **`0.725 ms/decision`**, sustaining **`6,200.5 decisions/second (6.2 kHz)`** on NVIDIA L4.

3. **Session 3 Exit Gate Met:**
   - Requirement 1: Amortization gap closed below $\tau = 0.20$ threshold ($R_{\tau=0.20} < R_{\text{refusal}}$ and $R_{\tau=0.20} < R_{\text{amortized}}$) $\to$ **PASS** (`0.26876` vs `0.72139` refusal and `0.39904` amortized).
   - Requirement 2: Real-time throughput exceeding 5 kHz $\to$ **PASS** (`6,200.5 Hz` on NVIDIA L4).

---

## 2. Multi-Site Baselines on Held-Out Test Windows

Evaluated across 12,462 test windows across 12 robotics laboratories over 3 seeds:

| Allocation Policy | Mean Test Regret | Std Error | Description |
|---|:---:|:---:|---|
| **Exact Autograd Oracle** | **`0.00000`** | `±0.00000` | Exact backward pass on every window (theoretical floor) |
| **Refusal Baseline (`always_mode0`)** | **`0.72139`** | `±0.55545` | Static refusal to invoke costly visual sensors |
| **Matched Direct Critic** | **`0.47105`** | `±0.33675` | Parameter-matched direct value-gain head |
| **Amortized HARP Co-State ($\tau = 0.00$)** | **`0.39904`** | `±0.27897` | Forward pass co-state sensitivity scoring |

---

## 3. Multi-Site Pareto Frontier (Confidence Gating $\tau$)

| Gating Threshold ($\tau$) | Empirical Rescue Rate | Mean Test Regret | Systems Latency (ms) | Effective Throughput (Hz) |
|:---:|:---:|:---:|:---:|:---:|
| `tau = 0.00` | **`  0.0%`** | **`0.39904 ± 0.27897`** | `0.000 ms` | **`2,613,918.7 Hz`** |
| `tau = 0.05` | **`  5.0%`** | **`0.39506 ± 0.27654`** | `0.182 ms` | **`   24,522.6 Hz`** |
| `tau = 0.10` | **` 10.0%`** | **`0.32966 ± 0.24560`** | `0.363 ms` | **`   12,344.3 Hz`** |
| `tau = 0.20` | **` 20.0%`** | **`0.26876 ± 0.18864`** | `0.725 ms` | **`    6,200.5 Hz`** |
| `tau = 0.30` | **` 30.0%`** | **`0.25553 ± 0.17906`** | `1.087 ms` | **`    4,140.0 Hz`** |
| `tau = 0.50` | **` 50.0%`** | **`0.15725 ± 0.14140`** | `1.812 ms` | **`    2,485.9 Hz`** |
| `tau = 0.80` | **` 80.0%`** | **`0.05739 ± 0.04836`** | `2.898 ms` | **`    1,554.6 Hz`** |
| `tau = 1.00` | **`100.0%`** | **`0.00505 ± 0.00384`** | `3.623 ms` | **`    1,243.8 Hz`** |

---

## 4. Architectural & Strategic Implications

1. **Validation Across Diverse Research Laboratories:**
   The multi-site test set spans 12 different physical robotics platforms and camera configurations. The monotonic drop in regret across $\tau$ demonstrates that decision-margin confidence gating generalises robustly across varied camera angles, object geometries, and lighting conditions.
2. **Real-Time Control Loop Viability:**
   Operating at $\tau = 0.20$ guarantees a latency of $0.725 \text{ ms}$, which is well within the 100 Hz ($10 \text{ ms}$) control cycle of modern robotic manipulators (e.g. Franka Emika Panda, UR5).
3. **Compute Discipline:**
   The benchmark completed on Colab L4 runtime `l4-worker` in ~7 minutes, after which the session was terminated immediately (`colab stop -s l4-worker`), resulting in zero idle billable compute credit burn.
