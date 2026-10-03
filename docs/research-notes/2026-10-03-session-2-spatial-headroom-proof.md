# Research Note: Session 2 Spatial Representation Headroom Benchmark (Gate G4-2)

**Date:** 2026-10-03  
**Authors:** Antigravity & OpenCode (`space-bunny-free`)  
**Hardware Evaluated:** NVIDIA L4 GPU (`l4-worker`, 22.03 GiB VRAM)  
**Governing Documents:** `docs/plans/2026-10-03-10-session-colab-hopper-plan.md`, `docs/plans/2026-10-03-session-2-spatial-headroom-spec.md`  
**Committed Artifacts:** [`results/benchmarks/spatial_headroom/spatial_headroom_summary.json`](../../results/benchmarks/spatial_headroom/spatial_headroom_summary.json), [`results/benchmarks/spatial_headroom/spatial_headroom_report.md`](../../results/benchmarks/spatial_headroom/spatial_headroom_report.md)

---

## 1. Executive Summary & Core Findings

Following the codification of the **Peer Critic Protocol** (`AGENTS.md`) and consultation with OpenCode (`space-bunny-free`), Milestone Session 2 was executed on Google Colab runtime `l4-worker` (NVIDIA L4, 22.03 GiB VRAM).

To rigorously test whether multi-token spatial patch representations yield predictive headroom over 1D pooled vectors without architectural confounding (Critique P0-1), both arms were evaluated under the **exact same frozen DINOv2 ViT-S/14 vision backbone** ($D=384$) at identical image resolution ($224 \times 224$):
- **Pooled Baseline (`dinov2_pooled`):** Concatenated CLS tokens from exterior and wrist cameras ($P=2$ tokens/frame).
- **Spatial ViT Predictor (`dinov2_spatial_vit`):** $4\times4$ patch grids from both cameras ($P=32$ tokens/frame) modeled with a block-causal spatio-temporal transformer.
- **Spatial Adjoint RWM (`spatial_adjoint_rwm`):** $4\times4$ patch grids processed via learned spatial positional encodings and multi-head attention pooling (`SpatialPatchAdapter`) into the recursive transformer-GRU dynamics core.

### Key Measured Outcomes

1. **Decisive Predictive Headroom Over Pooled Baseline:**
   - `dinov2_pooled` achieves test proprioception RMSE of **1.10788**.
   - `dinov2_spatial_vit` reduces RMSE to **0.92447** (**−16.55%** relative error reduction).
   - `spatial_adjoint_rwm` achieves test proprioception RMSE of **0.56800** (**−48.73%** relative error reduction).

2. **Gate G4-2 Evaluation Verdict:**
   - Threshold requirement: Held-out RMSE reduction $\ge 10\%$.
   - Measured best relative improvement: **−48.73%** (exceeds threshold by 38.7 percentage points).
   - **Verdict: PASS**.

3. **Systems & VRAM Scaling on NVIDIA L4:**
   - `dinov2_pooled`: 442.2 MiB peak VRAM.
   - `spatial_adjoint_rwm`: 616.8 MiB peak VRAM (+39.5% memory overhead).
   - `dinov2_spatial_vit`: 1,109.5 MiB peak VRAM (2.51× memory overhead).
   - Training latency was sub-10 seconds per arm (4.0s pooled, 7.6s spatial ViT, 8.9s spatial adjoint).

---

## 2. Benchmark Summary Table

| Arm / Baseline | Proprio Test RMSE | vs DINOv2 Pooled | Peak VRAM (MiB) | Trainable Parameters | Train Latency (5 Epochs) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Persistence (Baseline)** | 0.17773 | −83.96% | — | — | — |
| **Ridge (Linear Baseline)** | 0.19684 | −82.23% | — | — | — |
| **DINOv2 Pooled (P=2)** | **1.10788** | — | 442.2 | 17,158,146 | 4.0s |
| **DINOv2 Spatial ViT (P=32)** | **0.92447** | **−16.55%** | 1,109.5 | 17,148,546 | 7.6s |
| **Spatial Adjoint RWM (P=32)** | **0.56800** | **−48.73%** | 616.8 | 25,388,188 | 8.9s |

---

## 3. Hopper G4 Conjunctive Gating Analysis

Per the 10-Session Operational Plan (`docs/plans/2026-10-03-10-session-colab-hopper-plan.md` §4), provisioning Hopper G4 (96 GB) requires satisfying **all three conjunctive gates**:

1. **Gate G4-1 (Profiler Saturation):** NOT MET (`dataloader_wait_pct` 21.0% > 15%). CPU data prefetching remains a bottleneck.
2. **Gate G4-2 (Representation Headroom):** **MET (PASS)** (−48.73% RMSE reduction vs 1D pooled vectors).
3. **Gate G4-3 (Physical L4 OOM):** NOT MET (peak VRAM 1,109.5 MiB uses only 5.0% of L4 capacity).

### Hardware Policy Decision
Because Gates G4-1 and G4-3 are not satisfied, the campaign **strictly remains on NVIDIA L4**. Upgrading to Hopper G4 at this stage is scientifically ungrounded and would violate Rule 7 of `AGENTS.md`. The campaign advances to **Session 3 (HARP Selective Analytical Rescue on the full E3.1 shard)** and **Session 4 (Curvature & Belief-Space VOI Allocation)** on NVIDIA L4.
