# Optimization & Refinement Research Brief: AdjointRWM and the Shared Agentic Harness

**Date:** 2026-10-03
**Author:** OpenCode (`space-bunny-free`)
**Status:** Draft for peer-critic review (per `AGENTS.md` "Peer Critic Protocol for Planning & Design")
**Precedence:** Subordinate to [`AGENTS.md`](../../AGENTS.md), [`docs/production/colab_l4_operator_brief.md`](../production/colab_l4_operator_brief.md), and [`docs/plans/2026-10-03-10-session-colab-hopper-plan.md`](2026-10-03-10-session-colab-hopper-plan.md). **This brief proposes no change to the frozen 10-session campaign.** Where it disagrees with that plan, the disagreement is recorded in §2 and in [`prereg/deviation_log.yaml`](../../prereg/deviation_log.yaml) (DEV-20261003-01) rather than acted on. This file ends at §3; §§4–5 continue in [part 2](2026-10-03-optimization-research-brief-part2.md). There is no §6–§9: those sections were announced in an earlier draft and are not written.
**Baseline verified:** `pytest -q` → 401 passed; `python harness/check.py --fast` → 5/5.

---

## 0. Method and verification log

Every number below is either (a) quoted from a committed file, named inline, or (b) computed in this session by a script whose path is given. Per research-integrity rule 2, recomputed numbers carry the command that produced them and are **not** committed evidence until the script is.

Scripts written this session (currently uncommitted, in `/tmp/opencode/`):

| Script | What it measures | Reproduce |
|---|---|---|
| `actmem3.py` | Saved-for-backward activation bytes per transformer block, via `torch.autograd.graph.saved_tensors_hooks` | `python /tmp/opencode/actmem3.py` |
| `dataloader_cost3.py` | `.npz` vs `.npy` + `mmap` per-item cost, using real files | `python /tmp/opencode/dataloader_cost3.py` |

**Baseline for all activation figures:** `bytes(N) = 51746.99·N + 1.4836·N²` in FP32, for one `FeaturePredictor` block (`dim=384, heads=16, dim_head=64, mlp=1536`), batch 1, including the `[N,N]` block-causal boolean mask. Fitted at `N ∈ {256, 512, 1024, 2048, 4096}`.

This is a **conservative upper bound**: it was fitted on the CPU SDPA path, which retains more intermediates than CUDA's fused flash kernels. Two further exclusions apply — the `H`-step recursive unroll in `_predict`/`_forward_frames` (which retains a separate graph per unrolled step), and parameter/gradient/optimizer state.

---

## 1. Executive summary

Five findings, in order of consequence for the campaign.

| # | Finding | Consequence |
|---|---|---|
| **F1** | **Session 4's "Gate PASS" is not supported by its own dispersion.** The reported `±` is the standard deviation of **3 per-seed means** (`scripts/benchmark_curvature_voi_allocator.py:551`, aggregating the per-seed means built at `:542`), so the correct SEM is `std/√3`. Under that reading **no** Session 4 comparison reaches `p < 0.05`. | Session 4's exit gate must be re-evaluated with a paired, episode-clustered statistic. Highest priority. |
| **F2** | **Spatial tokens need no new hardware.** Session 2 passed G4-2 on L4 at **1,109.5 MiB peak (5.0 %)** with `P=32`. My activation model puts full `P=257` on the shard's 2 cameras at **1.47 GiB/sample**, i.e. **L4 accommodates `B ≤ 14`**. | G4-3 ("L4 OOMs") is satisfiable by choosing a batch size, not by the representation. |
| **F3** | **Deep-horizon curvature needs no new hardware — already measured.** Session 1 measured `H=64, B=64` with exact Hessian-vector products at **904.3 MiB (4.0 % of L4)** (`results/benchmarks/horizon_stress/horizon_stress_report.md`). | Workload B's stated rationale is refuted by measurement. |
| **F4** | **The `.npz` cache format is a 1,462× throughput cliff that only appears at spatial-token sizes.** Measured: **356.38 ms/item** (`.npz`) vs **0.24 ms/item** (`.npy` + `mmap_mode='r'`). At today's ResNet shape the same test gives 2.12 ms/item — invisible. | This, not GPU memory, is the binding constraint on scaling past `P=32`. |
| **F5** | **One live Colab assignment is untracked and unaddressable.** `colab sessions` lists `[?] gpu-l4-s-kkb-ass1b1-398qxbzgv6ojy`; `colab status` and `colab log` both return *not found* for that ID. | Billable compute that `AGENTS.md` rule 4 cannot currently stop by name. |

**Net effect on Session 5.** Of the four workloads in the frozen plan, **only Workload C (a 14B LLM in BF16, ~28 GB of weights) genuinely exceeds the L4's 22.03 GiB.** Workloads A and B are now measured to fit. §2 gives the gate arithmetic and, with [`prereg/deviation_log.yaml`](../../prereg/deviation_log.yaml) (DEV-20261003-01), records the disagreement with the frozen plan without acting on it.

---

## 2. Gate arithmetic for Session 5 (the immediate next task)

The plan makes G4-1/G4-2/G4-3 **conjunctive**. Current status from committed evidence:

### G4-1 (profiler saturation) — NOT MET

`results/benchmarks/profiler/profiler_baseline_report.md` reads: *"Verdict: **NOT GPU-saturated (G4-1 not met)**"*, with pooled `num_workers=2` at 24.1 % DataLoader wait against a `<15 %` requirement.

The verdict is **honestly reported but not yet decision-grade**, for a reason internal to the measurement: it was produced on *synthetic* tensors, as the script's own header states. The spatial configuration's DataLoader cost is `rng.standard_normal((768,384))` per item, which is not the cost of reading real cached episodes. The report is right to decline the gate; it is not evidence about the real pipeline in either direction.

### G4-2 (representation value) — PASS, on a weak base

`results/benchmarks/spatial_headroom/spatial_headroom_report.md`: `Spatial Adjoint RWM (P=32)` RMSE `0.56800`, `-48.73 %` vs DINOv2-pooled `1.10788`.

Two facts in that same table deserve more weight than the gate verdict gives them:

1. **Persistence scores `0.17773`; Ridge `0.19684`.** The gate-winning arm is therefore **3.2× worse than persistence** on the same 1,898 windows. The `-48.73 %` is measured against a baseline (`1.10788`) that is itself 6.2× worse than persistence. The gate is satisfied exactly as written ("vs 1D pooled"), but a reader of "G4-2 PASS" alone would draw the wrong conclusion about model quality.
2. **The run used 5 train / 5 test episodes.** One of three gates authorising a 96 GB hardware switch rests on ten episodes, against a shard of 400/50/50.

Neither point invalidates the gate. Both should accompany every citation of it.

### G4-3 (memory exhaustion on L4) — TRUE ONLY AT `B ≥ 15`

Measured activation memory, BF16, `depth=6`, teacher-forced forward+backward (exclusions as in §0):

| Encoder setting | cams | tok/cam | frames | N/seq | B | GiB | L4 (22.03 GiB) | G4 (96 GiB) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| today's ResNet-18 pooled | 2 | 1 | 12 | 24 | 64 | 0.22 | fits | fits |
| DINOv2 `grid=4` | 2 | 17 | 12 | 408 | 32 | 1.91 | fits | fits |
| DINOv2 `grid=8` | 2 | 65 | 12 | 1560 | 32 | 7.54 | fits | fits |
| DINOv2 `grid=16` | 2 | 257 | 16 | 8224 | 8 | 11.75 | fits | fits |
| DINOv2 `grid=16` | 2 | 257 | 16 | 8224 | 14 | 20.56 | fits | fits |
| **DINOv2 `grid=16`** | 2 | 257 | 16 | 8224 | **16** | **23.51** | **OOM** | fits |
| plan's 3×256 | 3 | 256 | 16 | 12288 | 32 | 76.88 | OOM | fits |
| plan's 3×256 | 3 | 256 | 16 | 12288 | 64 | 153.76 | OOM | **OOM** |

Two consequences.

**The plan's own memory claim checks out.** It states that full spatial tokens *"exceeds 36 GiB at `B=16`"*; my independent derivation gives **38.44 GiB** at `B=16` for 3×256. That is close agreement by two separate routes.

**But G4-3 is batch-size-dependent, and the batch size is a free parameter.** Full-resolution spatial tokens fit on L4 at `B ≤ 14`. A gate phrased as *"the workload OOMs L4 at nominal batch size `B=16`"* is satisfied by a choice rather than by a hardware limit. The meaningful question is whether `B=16` is *needed* for gradient quality — which is measurable and cheap (§3.3) — rather than assumed.

**The plan assumes 3 cameras; the shard has 2.** `src/adjointrwm/data/droid.py:16` fixes `IMAGE_KEYS = ("exterior_image_1_left", "wrist_image_left")`, and the E3.1 contract verified exactly those two streams (`docs/research-notes/2026-10-02-e3-1-droid-500-shard.md`). The `3 × 256` configuration anchoring Workload A is not reachable from the committed shard without adding and re-verifying a third stream.

---

## 3. Focus Area 1 — Efficient multi-camera spatial token representations

### 3.1 The infrastructure already exists; S2 was a config change, not an architecture

`DinoV2Embedder` (`src/adjointrwm/features.py:86-114`) already produces exactly the token geometry the plan describes. Verified this session:

- `image_size=224`, patch 14 → `16×16 = 256` patch tokens; `dim = model.embed_dim = 384` (ViT-S).
- `tokens_per_camera = 1 + grid²`, so `grid=16` → **257** tokens/camera, `grid=2` → 5.
- `pool_patch_grid(x, 16)` on a 16×16 grid is **exactly the identity** (checked with `torch.equal`), so `grid=16` is lossless, not a downsample.
- Licence: Apache-2.0 for code and weights, recorded in `manifest()`.

And the consuming path is live: `WindowDataset` already supports `visual_layout="tokens"` returning `[T, P, D]` (`src/adjointrwm/data/windows.py:206-207`), covered by `tests/test_windows.py:196`, and used by `scripts/train_e3_dynamics_pilot.py:138`.

**Implication.** Moving from `P=2` to full `P=257` is `dinov2_grid: 2 → 16` plus a cache rebuild. There is no new module to write. That is why Session 2 was cheap enough to run four times on one L4.

### 3.2 What the literature supports, and what it does not

Verified via arXiv/OpenAlex/Crossref by a research subagent; **three items in the original request are wrong and are corrected here**:

| Claim in the request | Verified reality |
|---|---|
| "Fast-JEPA" | **Does not exist.** arXiv all-fields search (both phrasings) → 0 hits; absent from OpenAlex and Crossref. Cite **V-JEPA 2**, arXiv:2506.09985. |
| "ToMe (Michael et al.)" | ToMe is **Bolya, Fu, Dai, Zhang, Feichtenhofer, Hoffman**, ICLR 2023 Oral, arXiv:2210.09461. Ryoo wrote *TokenLearner*. |
| "A-ViT (DynamicViT, Rao et al.)" | Two papers. **DynamicViT** = Rao et al., arXiv:2106.02034. **A-ViT/AdaViT** = Yin et al., arXiv:2112.07658. |

**Strongest published support for keeping spatial structure.** DINO-WM (arXiv:2411.04983) ablates observation encoders and reports, across six task suites, ResNet-single-vector values of `0.12 / 0.06 / 0.20` against DINOPatch per-patch values of `0.96 / 0.92 / 0.90`, concluding that world models *"that encode observations as a single latent vector show a significant drop"* as environment complexity rises. That argues for spatial tokens; it says nothing about gradients.

**V-JEPA 2-AC keeps full spatial structure and is explicit about what it does *not* do.** It encodes each frame independently to `z_k ∈ ℝ^{16×16×1408}` with no pooling, and its planning energy is a per-patch ℓ1 objective, so `∂ℰ/∂z_k` is natively a spatially resolved sensitivity field. But two constraints matter for us: the rollout loss uses **`T = 2` so that "we only differentiate the predictor through one recurrent step"**, and the energy is an **ℓ1 norm**, i.e. a subgradient. V-JEPA 2-AC is also explicitly a *"tabletop arm with a fixed exocentric camera"* with *"Sensitivity to camera positioning"* listed as a limitation — so there is no published multi-camera result to lean on.

**Compression ranked by expected co-state fidelity:**

| Method | Gradient story | Fit here |
|---|---|---|
| **TokenLearner** (arXiv:2106.11297) | Fully differentiable — conv/MLP → sigmoid → Hadamard → global average pool. No top-k, no mask, no Gumbel. | **Best.** `S=16` learned tokens/camera × 2 = 32 tokens; every spatial location retains a co-state |
| Evo-ViT slow-fast (arXiv:2108.01390) | Unselected tokens still updated via a cheap path; nothing hard-deleted | Good, unverified |
| ToMe (arXiv:2210.09461) | Aggregation differentiable; the **edge selection is a hard top-r** and is not differentiated. Within a merged group of size `s` the co-state is split **uniformly** | Mediocre — blind to non-uniform co-state mass |
| DynamicViT / A-ViT | Gradient paths kept during training by masking; **absent at inference** | Poor for a co-state allocator |
| DToP (arXiv:2308.01045) | Per-token variable depth ⇒ co-state depths incomparable across patches | Poor |

**The honest gap.** *No paper has tested whether token pruning or merging degrades the faithfulness of reverse-mode co-states.* The nearest evidence is attention-**layer** pruning degrading post-hoc explainer faithfulness in LLMs (arXiv:2606.24970) and HiLRP (arXiv:2609.01282) showing attribution breaks across patch-merging ViT families. Neither tests `∂J/∂state`. The "backward mass is redistributed" argument is a well-motivated hypothesis built on the attention-sink literature (StreamingLLM arXiv:2309.17453; Gu et al. arXiv:2410.10781; Sun et al. arXiv:2402.17762), **not a demonstrated result**. Also relevant and unstudied for us: ViT sinks migrate from CLS to *patch* tokens with depth (Fesser et al., arXiv:2606.08105), so in a multi-camera encoder the sink may sit on a patch — exactly where routing decisions are made.

### 3.3 Recommended architecture: co-state-gated token routing (CGR)

This is the one proposal here that is genuinely specific to AdjointRWM rather than borrowed. Let `x_{t,p} ∈ ℝ^d` be patch `p` of frame `t`, `S` the retained set, `m = |S|`. Importance is the token's contribution to the objective,

$$\hat{g}_{t,p} \;=\; \left\langle \frac{\partial J}{\partial x_{t,p}},\ \Delta x_{t,p}\right\rangle \approx \frac{1}{d}\sum_j \frac{\partial J}{\partial x_{t,p,j}}\,\Delta x_{t,p,j}$$

with selection `S_t = top-m { p : |ĝ_{t,p}| }`. Two properties make this a co-state and not a saliency score, satisfying research-integrity rule 4: `∂J/∂x` is obtained by reverse-mode autodiff through the dynamics unroll, and it is a sensitivity, not an attention weight.

**The hard part, stated plainly.** If you compute `∂J/∂x` *after* pruning, you have destroyed the very signal you needed to prune with. So the co-state must come from a cheap proxy, and the gate must keep gradients alive for the tokens it drops:

```python
def co_state_gate(x, proxy_costate, m, training):
    """x: [B,T,P,d]; proxy_costate: [B,T,P,d] from the cheap path.
    Returns gated tokens AND the soft mask, so dropped tokens keep a gradient path."""
    g = (proxy_costate * x).mean(-1)                       # [B,T,P] first-order contribution
    if training:
        u = torch.rand_like(g).clamp(1e-6, 1 - 1e-6)
        hard = (g >= g.topk(m, dim=-1).values[..., -1:]).float()
        soft = torch.sigmoid((g - g.median(-1, keepdim=True).values) * g.abs().mean())
        mask = hard + soft - soft.detach()                  # straight-through: forward=hard, grad=soft
    else:
        mask = (g >= g.topk(m, dim=-1).values[..., -1:]).float()
    return x * mask.unsqueeze(-1), mask
```

**Fallback when no co-state is available.** If proxy computation is too costly, rank tokens by `||Δx_{t,p}||₂` (effect magnitude). This is cheaper but is *not* a co-state and must not be described as one — it is the "perturbation importance" baseline, and comparing it against CGR is a publishable ablation rather than a fallback.

**Cost.** The proxy pass is one forward + one backward on the pooled path. Amortise it over `k` optimisation steps, or use a single truncated backward through the last `m` blocks. Validate against Hutchinson: `E_v[(∂/∂ε) f(x+εv)ᵀ λ] = (∂f/∂x)ᵀλ`, which gives token co-states with one forward pass and no `[N,N]` materialisation.

**Rule-4 wording for the paper.** Call it a *co-state* only when the ranking score is `∂J/∂x` from autodiff through the unroll. Call `||Δx||` routing *effect-magnitude routing*. Do not call attention rollout a co-state — Abnar & Zuidema (arXiv:2005.00928) show rollout correlates with gradient importance, but HiLRP shows rollout assumes global softmax attention, which token compression itself violates.