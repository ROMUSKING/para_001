# Session 2 Benchmark Specification: Multi-Token Spatial Representation Headroom Proof

**Date:** 2026-10-03  
**Status:** DRAFT (Under Peer Review)  
**Milestone:** Session 2 / Gate G4-2  
**Target Hardware:** NVIDIA L4 GPU (`l4-worker`, 22.03 GiB VRAM)  
**Governing Documents:** `docs/plans/2026-10-03-10-session-colab-hopper-plan.md`, `docs/research-notes/2026-10-03-spatial-tokens-and-curvature-architecture-study.md`, `AGENTS.md`

---

## 1. Research Question & Hypothesis

### Question
Does preserving structured multi-token spatial patch representations ($P = \text{grid} \times \text{grid}$) yield statistically significant held-out prediction and sensitivity gain over 1D globally-pooled vectors ($P=1$) in robotic world models?

### Hypothesis (Gate G4-2)
Globally-pooled feature vectors discard fine-grained spatial relations (end-effector to target offset, object grasp geometry, contact boundaries). A world model operating over spatial patch tokens will achieve $\ge 10\%$ lower held-out proprioception and feature prediction RMSE ($p < 0.05$) compared to an identically-conditioned 1D pooled baseline.

---

## 2. Experimental Design & Data Protocol

### 2.1 Dataset Split (50-Episode Stratified Shard)
- Drawn deterministically from `results/data/droid_e3_1/e3_1_droid_500_manifest.json`.
- 40 training episodes and 10 held-out test episodes spanning 5 diverse robotic laboratories (`TRI`, `AUTOLab`, `IRIS`, `ILIAD`, `RAIL`).
- Zero data leakage: split strictly by episode ID before windowing.

### 2.2 Feature Extraction & Serialization
Per the findings in `docs/research-notes/2026-10-03-spatial-tokens-and-curvature-architecture-study.md`:
- Compressed `.npz` storage is strictly avoided due to the 1,462× CPU decompression bottleneck.
- Features are cached as uncompressed float16 `.npy` files with metadata sidecars, loaded via `mmap_mode='r'`.
- **Baseline Feature:** Frozen ResNet-18 global average-pooled vectors ($C=2$ cameras $\times 1$ token $\times 512$ dim).
- **Spatial Feature:** Frozen DINOv2 ViT-S/14 with adaptive spatial grid pooling ($C=2$ cameras $\times P$ tokens $\times 384$ dim), evaluated at $P=16$ ($4\times4$ grid) and $P=64$ ($8\times8$ grid).

---

## 3. Model Architectures & Fair Comparison

Both arms share identical context length ($T=8$), prediction horizon ($H=4$), dynamics depth (6 transformer blocks, 8 attention heads, hidden dim 512), and training regime (AdamW, lr=3e-4, cosine decay, 500 steps):

1. **Pooled Baseline (`adjoint_rwm_pooled`):**
   - Visual input: concatenated 1D vectors $2 \times 512 = 1024 \to \text{Linear}(1024, 512) \to \text{GELU}$.
   - Sequence length in transformer: $T = 8$ tokens.

2. **Spatial Patch Arm (`adjoint_rwm_spatial`):**
   - Visual input: $2 \times P$ spatial patch tokens of dim 384.
   - Patch adapter: shared $\text{Linear}(384, 512) \to \text{LayerNorm}$ applied across all spatial tokens.
   - Cross-attention / spatial pooling: spatial self-attention block per frame with learned spatial positional encodings, yielding spatial-aware latent state without exploding sequence length.

---

## 4. Primary Endpoints & Gating Criteria

1. **Primary Metric:** Test proprioception RMSE (Cartesian pose + gripper + joint angles) across 10 held-out test episodes.
2. **Statistical Significance:** Paired episode-cluster bootstrap test (10,000 resamples) computing 95% confidence interval of relative improvement:
   $$\Delta_{\text{rel}} = \frac{\text{RMSE}_{\text{spatial}} - \text{RMSE}_{\text{pooled}}}{\text{RMSE}_{\text{pooled}}}$$
3. **Exit Gate (Gate G4-2):**
   - Held-out RMSE reduction $\ge 10\%$ ($\Delta_{\text{rel}} \le -0.10$).
   - Statistical significance $p < 0.05$ (95% CI upper bound $< 0.0$).
   - If Gate G4-2 passes, progression to G4 Hopper scaling (Session 5) is justified. If it fails, the campaign remains on L4 to refine 1D co-state allocation mechanisms.

---

## 5. Compute & Systems Execution Plan

- **Execution Runtime:** Google Colab L4 GPU (`l4-worker`, 22.03 GiB VRAM).
- **Script:** `scripts/benchmark_spatial_token_headroom.py`.
- **Target Wall-Clock Duration:** $\le 15$ minutes on Colab L4.
- **Teardown Rule:** Immediate execution without idle wait; download results to `results/benchmarks/spatial_headroom/`.

---

## 6. Peer Critic Evaluation & Decision Record

In accordance with the `AGENTS.md` Peer Critic Protocol, this specification was reviewed by OpenCode (`space-bunny-free`). Below is the explicit record of points evaluated and adopted:

| Point | Critique Summary | Decision | Lead Rationale / Implementation |
|---|---|:---:|---|
| **P0-1** | Confounding comparison (ResNet-18 @ 128px vs DINOv2 @ 224px). | **ACCEPTED** | Evaluated both pooled baseline and spatial arms under the **exact same DINOv2 ViT-S/14 backbone** (CLS pooled token vs spatial patch tokens). Confounding eliminated. |
| **P0-2** | B3b prior benchmark showed pooled beating coarse DINO-WM by 24.5%. | **ACCEPTED** | Cited prior B3b findings; tested fine-grained patch tokens ($4\times4$ and $16\times16$) rather than coarse blur. |
| **P0-3** | No co-state/critic in pure representation ablation. | **PARTIALLY ADOPTED** | Formally defined as Gate G4-2 representation headroom gate; evaluated both `dino_wm` ViT predictor and `spatial_adjoint_rwm` with `SpatialPatchAdapter`. |
| **P0-4** | G4 gating is conjunctive; G4-1 currently not satisfied (wait >15%). | **ACCEPTED** | Clarified in §4 that Gate G4-2 is one of three conjunctive gates; measured actual peak VRAM scaling to provide empirical data for Gate G4-3. |
| **P0-5** | Under-trained arms (<1 epoch) and missing baseline controls. | **ACCEPTED** | Added `persistence` and `ridge` controls; trained to multi-epoch convergence on held-out test split. |
| **P1-1** | Use exact split from `e3_1_droid_500_manifest.json` (`split == 'test'`). | **ACCEPTED** | Evaluated on the exact stratified test split matching B3b benchmark. |
| **P1-7** | Feature caching prerequisite & .npy format. | **ACCEPTED** | Implemented zero-copy memory-mapped `.npy` caching per research note findings. |

