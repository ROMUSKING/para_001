# Research Note: Spatial Token Activation Scaling, Memory-Mapped Caching, and Curvature Autograd Mechanics

**Date:** 2026-10-03  
**Authors:** Antigravity & OpenCode (`space-bunny-free`, `muse-spark-1.3-contributor-free`)  
**Hardware Evaluated:** NVIDIA L4 GPU (`l4-worker`, 22.03 GiB VRAM) & Hopper G4 Analytical Model  
**Artifacts:** [`results/benchmarks/profiler/`](../../results/benchmarks/profiler/), [`results/benchmarks/horizon_stress/`](../../results/benchmarks/horizon_stress/)

---

## 1. Executive Summary & Breakthrough Findings

In parallel with executing Sessions 0 and 1 of the 10-Session Colab campaign, an autonomous research study was conducted to resolve the exact memory and systems mechanics governing the transition from 1D pooled vectors to high-dimensional spatial patch tokens (Workload A) and long-horizon curvature autograd (Workload B).

This study produced three concrete, measured findings:

1. **The 1,462× Disk Decompression Bottleneck:**
   Saving spatial feature tokens in compressed `.npz` format incurs a catastrophic CPU decompression penalty that scales with token count: **356.38 ms/item** at full grid ($257 \text{ tokens} \times 384 \text{ dim}$), throttling multi-worker data loading to **0.35 steps/second**. Replacing `.npz` with uncompressed `.npy` arrays accessed via `np.load(..., mmap_mode='r')` reduces read latency to **0.24 ms/item (1,462× speedup)**, enabling **513 steps/second**.

2. **Empirical Activation Memory Model for Hopper G4 (Workload A):**
   Using `torch.autograd.graph.saved_tensors_hooks` on depth-6 transformer blocks in BF16, activation volume scales as:
   $$\text{Memory}(N, B) \approx \frac{(c_1 N + c_2 N^2) \times \text{depth} \times B}{2 \times 1024^3} \text{ GiB}$$
   where $N = T \times C \times P$ is the sequence token count.
   - For 1D pooled ResNet vectors ($N=24, B=64$): Memory is only **0.22 GiB** (comfortably fits L4).
   - For 3 cameras $\times 256$ spatial patches ($T=16, N=12,288$ tokens):
     * At $B=16$: Activation memory is **38.44 GiB** (**Strict OOM on L4 22.03 GiB; fits Hopper G4 96 GiB**).
     * At $B=32$: Activation memory is **76.88 GiB** (**Strict OOM on L4; fits Hopper G4**).
     * Maximum batch fitting L4 without gradient checkpointing is $B \le 9$. On Hopper G4, $B \le 39$ fits uncheckpointed.

3. **Autograd Curvature Mechanics in PyTorch SDPA:**
   cuDNN / FlashAttention SDPA (`aten::_scaled_dot_product_efficient_attention_backward`) does **not** implement second-order derivatives (`create_graph=True`). Attempting double-backpropagation for second-order co-state curvature throws a runtime error. Exact second-order curvature requires executing under the Math SDP context:
   ```python
   with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
       ...
   ```
   Under Math SDP, exact second-order Hessian-vector products execute in **128.48 ms** at $H=64, B=64$ on NVIDIA L4 with only **904.3 MiB peak VRAM (4.0% capacity)**.

---

## 2. Activation Memory Scaling Matrix

| Workload Configuration | Cameras $C$ | Tokens / Frame $P$ | Horizon $T$ | Total Tokens $N$ | Batch $B$ | GiB / Sample | Total VRAM (GiB) | Hardware Verdict |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (ResNet-18 Pooled)** | 2 | 1 | 12 | 24 | 64 | 0.003 | 0.22 | **Fits L4** |
| **DINOv2 Grid=4** | 2 | 17 | 12 | 408 | 32 | 0.060 | 1.91 | **Fits L4** |
| **DINOv2 Grid=8** | 2 | 65 | 12 | 1,560 | 32 | 0.236 | 7.54 | **Fits L4** |
| **DINOv2 Grid=16 (2 Cams)** | 2 | 257 | 16 | 8,224 | 8 | 1.469 | 11.75 | **Fits L4** |
| **DINOv2 Grid=16 (2 Cams)** | 2 | 257 | 16 | 8,224 | 16 | 1.469 | 23.51 | **OOM L4 / Fits G4** |
| **Workload A (3 Cams $\times$ 256 Patches)** | 3 | 256 | 16 | 12,288 | 16 | 2.403 | 38.44 | **OOM L4 / Fits G4** |
| **Workload A (3 Cams $\times$ 256 Patches)** | 3 | 256 | 16 | 12,288 | 32 | 2.403 | 76.88 | **OOM L4 / Fits G4** |
| **Workload A (3 Cams $\times$ 256 Patches)** | 3 | 256 | 16 | 12,288 | 64 | 2.403 | 153.76 | **OOM Both (Needs Checkpoint)** |

---

## 3. Data Pipeline & Serialization Architecture

### The Compressed NPZ Flaw
In existing feature extractors (`scripts/build_e3_feature_cache.py`), features were stored using `np.savez_compressed()`. For compact 512-d pooled vectors, decompression took only 2.12 ms/item. However, for spatial token grids ($16\times16$ patches, $D=384$, float16), each episode expands to 118.5 MB:
- `np.savez_compressed` CPU inflate: **356.38 ms/item** $\to$ **11,404 ms per batch of 32**.
- Even with 4 DataLoader workers, batch fetch wall-clock latency is **2,851 ms**, limiting GPU training to **0.35 steps/second**.

### The Zero-Copy Memory-Mapped Solution
Storing episodes as uncompressed `.npy` files and loading them via `np.load(path, mmap_mode='r')`:
- Kernel page cache reads slices without decompressing the full episode array.
- Latency per item drops to **0.24 ms/item** $\to$ **7.8 ms per batch of 32**.
- Sustained pipeline throughput: **513 steps/second (1,462× speedup)**.

---

## 4. Alignment with the 10-Session Operational Plan

These findings confirm the exact execution strategy for upcoming sessions:
1. **Session 2 (Spatial Headroom Proof on L4):**
   Must be executed with `grid=4` or `grid=8` ($P \in [17, 65]$ tokens/view) or with small batch sizes ($B \le 8$ at `grid=16`) to fit comfortably within L4's 22.03 GiB VRAM while testing for held-out prediction/regret gain.
2. **Session 5 (Switchover to Hopper G4):**
   Once Session 2 confirms that spatial patch tokens produce statistically significant held-out gains ($p < 0.05$), Session 5 scales directly to full 3-camera `grid=16` ($P=256$ tokens/view, $B=32$), utilizing **76.9 GiB VRAM** on NVIDIA Hopper G4 (96 GB).
