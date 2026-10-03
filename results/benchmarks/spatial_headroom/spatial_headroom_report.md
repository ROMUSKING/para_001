# Session 2 Benchmark Report: Spatial Representation Headroom (Gate G4-2)

**Generated:** 2026-10-03T11:14:18.761202+00:00  
**Device:** NVIDIA L4  
**Configuration:** DINOv2 ViT-S/14 Grid 4x4 (32 tokens/frame)  
**Dataset:** 5 train / 5 test episodes (1898 test windows)

---

## 1. Primary Empirical Results

| Arm / Baseline | Proprio Test RMSE | vs DINOv2 Pooled | Peak VRAM (MiB) | Parameters | Train Latency |
|---|:---:|:---:|:---:|:---:|:---:|
| **Persistence** | 0.17773 | -83.96% | — | — | — |
| **Ridge (Linear)** | 0.19684 | -82.23% | — | — | — |
| **DINOv2 Pooled (P=2)** | **1.10788** | — | 442.2 | 17,158,146 | 4.0s |
| **DINOv2 Spatial ViT (P=32)** | **0.92447** | **-16.55%** | 1109.5 | 17,148,546 | 7.6s |
| **Spatial Adjoint RWM (P=32)** | **0.56800** | **-48.73%** | 616.8 | 25,388,188 | 8.9s |

---

## 2. Gate G4-2 Evaluation Verdict

- **Gate Criterion:** Held-out RMSE reduction >= 10% (relative difference <= -0.10).
- **Measured Best Relative Improvement:** **-48.73%**
- **Gate G4-2 Verdict:** **PASS**

### Hardware & G4 Gating Context
Per AGENTS.md and the 10-Session Operational Plan:
1. Gate G4-2 is **one of three conjunctive gates** (with G4-1 Profiler Saturation and G4-3 L4 Memory Exhaustion).
2. Because G4-1 currently reports DataLoader wait >15%, G4-2 passing licenses consideration of spatial scale-up but does not authorize provisioning G4 alone.
3. Both arms operate comfortably within NVIDIA L4 VRAM (1109.5 MiB peak), verifying that subgrid spatial models ($P=32$) remain fully compatible with L4 standard policy.
