# Optimization & Refinement Research Brief, part 2 (Focus Areas 2 and 3)

**Date:** 2026-10-03
**Author:** OpenCode (`space-bunny-free`)
**Status:** Draft for peer-critic review. Continuation of [`2026-10-03-optimization-research-brief.md`](2026-10-03-optimization-research-brief.md) (§§4–5); it ends at §5 and there is no §6–§9.
**Precedence:** Subordinate to [`AGENTS.md`](../../AGENTS.md) and part 1.

---

## 4. Focus Area 2 — Low-rank / Gauss-Newton curvature approximations

### 4.1 First, a correction to the framing

The request asks for curvature **without storing full dense Hessians over long horizons `H ≥ 32`**. The measured premise is that this is expensive. It is not.

`results/benchmarks/horizon_stress/horizon_stress_report.md`, `H=64, B=64`, including exact second-order Hessian-vector products: **222.83 ms total step, 904.3 MiB peak VRAM, 4.0 % of L4.** Memory grows from 281.3 MiB at `H=4, B=16` to 904.3 MiB at `H=64, B=64` — close to linear in `B`, strongly sub-linear in `H`.

So the reason to approximate curvature here is **latency and amortised decision throughput**, not memory. The relevant constraint is the allocator's 27.2 kHz budget (Session 4), not a 22 GiB ceiling. Framing Workload B as a memory problem has already been refuted by the campaign's own measurement.

### 4.2 The implementation is already the right one

`second_order_curvature_scores` (`src/adjointrwm/allocators.py:181`) computes

$$s_k = -\hat{\lambda}^{\top}\Delta z_k - \tfrac12 \sum_d H_{dd}\,(\Delta z_{k,d})^2 - c_k$$

This is exact under one interpretation, and that interpretation should be stated in the paper. It is the Newton decrease for the candidate effect used *as the step*:

$$\Delta J_k \approx \hat{\lambda}^{\top}\Delta z_k + \tfrac12 \Delta z_k^{\top}\nabla^2 J\,\Delta z_k =: -s_k$$

Consequences that follow, and that the current docs do not state:

1. **The sign convention is correct, and it is not arbitrary.** Under a *fixed-step-norm* budget `||Δa|| ≤ ρ`, the achievable decrease is `-ρ||∇J|| + ½ρ²·λ_max(∇²J)`, which penalises large effects for a different reason. The implemented formula is the `Δa = Δz_k` reading, which is natural when effects are predicted per-unit-action. Say which one is meant.
2. **`F.softplus` on the Hessian head (`allocators.py:229`) guarantees a positive-definite diagonal**, so the curvature term can only ever *demote* a candidate. A candidate with a large effect norm is penalised regardless of whether its curvature is favourable. This is a consequence of the parameterisation, not a modelling insight, and should be checked against an unconstrained-diagonal ablation.
3. **The zero-curvature limit is already tested** as an invariant: `H = 0` reduces exactly to first-order.

### 4.3 What the literature does and does not license

I could not verify a canonical citation for *action-space* curvature specifically. It is an unusual object and should be presented as such rather than dressed up as natural gradient.

| Method | Cost | Verdict here |
|---|---|---|
| Diagonal Fisher / empirical Fisher (what is implemented) | `O(K·d)`, one backward | **Correct choice.** `d ≈ 384`, `K = 4`; the score is already `O(d)` per the Session 4 note |
| Levenberg–Marquardt damping `H̃ = H + λI` | free | **Should be added.** Without it, `H_dd → 0` on a direction the network is insensitive to and the correction silently vanishes. A `λ` sweep passing through `λ→∞` recovers first-order and is the cleanest possible falsification test |
| K-FAC (Grosse & Martens) | parameter-space Kronecker factors | **Not applicable.** It approximates *parameter* curvature; ours is over effects, ~384-dim and structured |
| Woodbury / low-rank (`H = D + UCUᵀ`) | `O(d·K_c + K_c³)` | **Worth one experiment** (§4.4) — the only upgrade that adds rank information without changing the estimator's character |
| L-BFGS two-loop recursion | `O(n·m)` | Not applicable: this is inference-time amortised scoring, not optimisation |
| TENGraD, Shampoo, LOBC | parameter-space | Not applicable |

**Meta-learning caution.** MAML (Finn et al.) is the closest analogue and it is contested — the "Critiquing MAML" line of work argues much of its benefit comes from feature reuse rather than second-order gradients. Do not cite MAML as evidence that second-order adaptation helps here.

**Cost basis for second-order information**, from Griewank & Walther's AD taxonomy: forward-over-first-reverse ("2-3-1 mode") gives one Hessian-vector product per call for `O(n)` VHPs; complex-step differentiation gives derivatives to machine precision at roughly forward cost but does not extend cleanly to higher derivatives. For an amortised scorer running at 27 kHz, the practical answer is **one VHP per policy evaluation**, amortised over a batch — not a full Hessian.

### 4.4 The strongest available upgrade: restrict curvature to the span of the candidates

The allocator only ever scores `K` candidates. So the curvature that matters is curvature **restricted to `span{Δz_k}`** — a `K`-dimensional object, not a `d`-dimensional one. This is exact on that subspace, not an approximation of it:

$$H_S = S^{\top}\nabla^2 J\,S, \qquad S = [\Delta z_1,\dots,\Delta z_K] \in \mathbb{R}^{d \times K}$$

$$s_k = -\hat{\lambda}^{\top} e_k - \tfrac12 (H_S)_{kk} - c_k$$

Cost: `K` Hessian-vector products (one backward each with `create_graph=True`), independent of `d`. With `K=4` that is 4 backwards — comfortably inside a 27 kHz budget when amortised over a batch — and it captures **off-diagonal** curvature the current diagonal estimator cannot represent.

```python
def subspace_curvature_scores(costate, effects, costs, vhp, lam=1e-3):
    """effects: [d, K]; vhp: callable v -> H v (one VHP each). Returns [B, K+1]."""
    K = effects.shape[-1]
    cols = [vhp(effects[:, k]) for k in range(K)]
    H_S = torch.stack([cols[k] @ effects for k in range(K)], dim=-1)   # [d, K]
    H_S = 0.5 * (H_S + H_S.T)                        # symmetrise: VHPs are noisy
    H_S = H_S + lam * torch.eye(K, device=effects.device)             # LM damping
    first = -(costate.unsqueeze(1) * effects).sum(-1)
    scores = first - 0.5 * torch.diagonal(H_S) - costs.view(1, -1)
    return torch.cat([costate.new_zeros(costate.shape[0], 1), scores], dim=1)
```

**The practical caveat, which Session 1 already hit.** `results/benchmarks/horizon_stress/horizon_stress_report.md` records that cuDNN FlashAttention does not support double-backpropagation under `create_graph=True`, forcing a Math-SDP fallback. Any VHP-based method inherits that constraint, and it costs latency. Benchmark it before proposing it.

### 4.5 The belief-space VOI term

`belief_space_voi_scores` (`allocators.py:195`) adds `½·w·Tr(diag(H)·ΔΣ)`. This is the standard belief-space information term and is correctly signed — information is valuable where the objective is sensitive. Two notes:

- `w` is a **free hyperparameter with no learned component**, defaulting to `0.5`. It is a method-specific constant, and research-integrity rule 1 forbids unstated multipliers. Either sweep it and report the sweep, or make it a learned head. As written, `belief_space_voi` beating `second_order_curvature` (0.06354 vs 0.07621) is partly an artefact of tuning `w`.
- VOI and second-order curvature are **not competing allocators** — one is a variance-reduction term, the other a curvature penalty. Reporting them as rival rows in one table invites reading a difference that is really a hyperparameter.

### 4.6 Literature grounding for §4 (subagent-verified; corrections to the above)

**The formula already has a canonical name.** It is the negative of the improvement estimate in **Differential Dynamic Programming / iLQR**, evaluated per candidate and truncated at second order in the perturbation of the control trajectory: **Tassa, Erez & Todorov, "Synthesis and stabilization of complex behaviors through online trajectory optimization", IROS 2012**, DOI [10.1109/IROS.2012.6386025](https://doi.org/10.1109/IROS.2012.6386025). Cite that rather than describing the estimator as ad hoc.

**Action-space curvature does have a literature — I was too cautious.** The closest verified match is **Tangkaratt, Abdolmaleki & Sugiyama, "Guide Actor-Critic for Continuous Control", ICLR 2018**, arXiv:[1705.07606](https://arxiv.org/abs/1705.07606): *"GAC updates the guide actor by performing second-order optimization in the action space where the curvature matrix is based on the Hessians of the critic."* Also **Furmston & Lever, "A Gauss-Newton Method for Markov Decision Processes", arXiv:[1507.08271](https://arxiv.org/abs/1507.08271)**, which analyses the MDP-objective Hessian, drops the hard-to-estimate terms, shows the result is negative definite, and links the method to both EM and natural gradient.

**The VOI term is a Gaussian moment identity, and it has a precedent that partly rescues it.** For diagonal `H`, `½·tr(H·ΔΣ) = E[½·ΔθᵀH·Δθ]` when `Δθ ~ N(0, ΔΣ)`. That is a second-order Taylor + Gaussian-moment identity, not mutual information. The nearest published precedent is **Houska, Telen, Logist & Van Impe, "Self-reflective model predictive control", arXiv:[1610.03228](https://arxiv.org/abs/1610.03228)**, which propagates covariance forward and a matrix-valued adjoint backward to *"minimize a second order approximation of its own expected loss of control performance."* Same construction. So cite Houska et al. for the construction, and be precise that the missing piece is a *decision-theoretic* grounding (the genuine VOI literature — Andriotis et al. arXiv:[1912.12534](https://arxiv.org/abs/1912.12534); Kaelbling/Littman/Cassandra 1998, DOI [10.1016/S0004-3702(98)00023-X](https://doi.org/10.1016/S0004-3702(98)00023-X)) is a different object involving entropy.

**The structural asymmetry that should reshape §4.3.** At `d ≤ 512` a *dense* action-space Hessian is `512² = 262,144` floats ≈ 1 MB in FP32 — trivial to store. Every structured approximation in the curvature literature exists to solve a **dimensional** problem this repo does not have. The real cost is not storage but the `d` Hessian-vector products needed to build it. Consequences:

- The bottleneck is `O(d)` HVPs, not `O(d³)` inversion (`O(512³) ≈ 1.3×10⁸` flops is nothing on an L4).
- **Forward mode is the right AD mode here.** COMLN (Deleu et al., arXiv:[2203.01443](https://arxiv.org/abs/2203.01443)) shows memory *"does not scale with the length of the learning trajectory"* — exactly the `H ≥ 32` concern, and `d ≤ 512` makes `d` forward sweeps affordable.
- **Complex-step and finite differences should not be used.** They break on `relu`, `clamp`, `abs` — all present in this model. Double-backward is exact and already available.

**Two positive references for the *learned* diagonal head**, which is the design actually implemented (`allocators.py:214-230`):

- **Wu, Zhu, Wu, Wang & Ge, "Dissecting Hessian", arXiv:[2010.04261](https://arxiv.org/abs/2010.04261)** — Hessians across tasks share substantial common structure, including at initialisation. This is a published argument that a single factored or diagonal curvature transfers across instances, which is the justification for an input-conditioned learned head.
- **Gürbüzbalaban, Şimşekli & Zhu, "The Heavy-Tail Phenomenon in SGD", arXiv:[2006.04740](https://arxiv.org/abs/2006.04740)** — curvature eigenvalue spectra are heavy-tailed, the standard argument that a diagonal approximation can be adequate.

Against that, **Clarke & Hernández-Lobato, "Studying K-FAC Heuristics by Viewing Adam through a Second-Order Lens", ICML 2024**, arXiv:[2310.14963](https://arxiv.org/abs/2310.14963): *"K-FAC's adaptive heuristics are of variable standalone general effectiveness."* And TENGraD (Soori et al., arXiv:[2106.03947](https://arxiv.org/abs/2106.03947)) notes that *"none of the current approximate NGD approaches outperform or are comparable with the end-to-end wall-clock time of tuned SGD."* **There is no consensus to cite; any claim that second-order curvature improves allocation must rest on this repo's own held-out experiment.**

**MAML is not support for second-order adaptation.** Nichol, Achiam & Schulman, arXiv:[1803.02999](https://arxiv.org/abs/1803.02999) contains FOMAML, ANIL and Reptile and shows most of MAML's benefit survives truncation to first order; Raghu et al., arXiv:[1909.09157](https://arxiv.org/abs/1909.09157), conclude it is feature reuse. The empirical prior is that second-order buys little relative to cost.

**The methodological precedent for the diagonal form.** **Liu, Xie, Deng, Ge & Ye, "Stochastic Dimension-reduced Second-order Methods for Policy Optimization", arXiv:[2301.12174](https://arxiv.org/abs/2301.12174)** obtains second-order information at first-order cost via HVPs and a *projected two-dimensional trust-region subproblem* — structurally the same trick as evaluating `½ eᵀHe` on one vector instead of inverting `H`.

### 4.7 Three experiments, in priority order

1. **Validate the learned diagonal against a computed one.** One reverse pass with `create_graph=True` plus per-coordinate HVPs gives true `diag(∇²_z J)` for `d ≤ 50`. Report the correlation. **Until this number exists, no claim about the second-order term's value is supportable.** There is no ground-truth Hessian supervision anywhere in `allocators.py` — the head is trained only through the score losses — so at present the honest description is "a reparameterisation of a positive diagonal curvature that the downstream loss finds useful," not "a calibrated curvature estimate."
2. **The ablation that distinguishes curvature from a learned conditioning feature** (this is the decisive one, and it is cheap). Shuffle `diag_hessian` across batch elements, or replace it with a per-sample constant. If performance survives, the head is a conditioning feature and the second-order claim collapses to first-order. No equivalent ablation exists in the MAML literature.
3. **Add an LM floor** `F.softplus(x) + λ`, `λ > 0`. One line. It guarantees non-degenerate curvature and makes the low-rank + diagonal form of §4.4 legal, which requires `D ≻ 0`.

A negative result in (1) or (2) is publishable and should be reported as one. Research-integrity rule 3 applies.

---

## 5. Focus Area 3 — High-throughput streaming data pipelines

### 5.1 The cache format is the bottleneck, and it is a 1,462× cliff

`src/adjointrwm/data/cache.py:71` writes one `np.savez_compressed` per episode; `src/adjointrwm/data/windows.py:248` reads it back with `np.load(path)` and **copies every array** on each cache miss.

`.npz` is a ZIP archive. NumPy's source selects `zipfile.ZIP_DEFLATED` when `compress=True`, and `numpy.load` documents `mmap_mode` as the way to *"[a]ccess small fragments of large files without reading the entire file"* — which a ZIP container cannot offer. Every window access through an `.npz` re-inflates the member.

Measured this session with real files (`/tmp/opencode/dataloader_cost3.py`), 300-frame episode, 2 cameras × 257 tokens × 384 dims, float16:

| Strategy | ms/item | ms/batch (32) | effective steps/s @ 4 workers |
|---|---:|---:|---:|
| `.npz` compressed (**today's format**) | **356.38** | 11,404 | **0.35** |
| `.npy` + `np.load(mmap_mode='r')` | **0.24** | 7.8 | **513** |
| ratio | **1,462×** | | |

**Why it has stayed invisible.** The same test at the *current* ResNet-18 cache shape (`1 × 512` per camera) gives **2.12 ms/item** for `.npz`. The penalty scales with feature size — 168× worse at `grid=16` than at `grid=2`. The campaign has been running below the cliff for its entire life.

**Consequence.** Session 2 reached `P=32` on 5 episodes. Scaling to `P=257` on the full shard is a **host-CPU** problem first, not a GPU problem: at `B=32`, `.npz` costs 11.4 s of CPU per batch.

### 5.2 Two host-memory limits that bind before the GPU does

Computed from `results/data/droid_e3_1/e3_1_droid_500_manifest.json` (500 episodes, 148,216 frames, median 230):

| Setting | MiB/episode | `lru_cache(16)` × 1 worker | × 4 workers |
|---|---:|---:|---:|
| today's ResNet pooled | 0.4 | 0.03 GiB | 0.14 GiB |
| `grid=8` (65 tok/cam) | 21.9 | 0.45 GiB | 1.79 GiB |
| **`grid=16` (257 tok/cam)** | **112.9** | **1.76 GiB** | **7.06 GiB** |

`WindowDataset.__init__` sets `self._load = lru_cache(maxsize=16)` (`windows.py:235`), and **each DataLoader worker gets its own copy**. At `grid=16` with 4 workers that is **7.06 GiB of host RAM** — a large fraction of a Colab VM's RAM, before any model or page cache.

Whole-shard decoded feature volume at `grid=16`: **54.5 GiB**, versus **0.28 GiB** today — a 195× increase that must live on ephemeral local disk.

**Recommendation, in priority order:**

1. **Switch the cache to `.npy` + `mmap_mode='r'`.** Biggest single win in this brief, and no GPU is involved.
2. **Shrink or remove the `lru_cache`.** With mmap, the OS page cache does the job *once* and shares it across workers. `lru_cache(16)` becomes a liability rather than an optimisation.
3. **Budget `grid` against disk, not just VRAM.** At `grid=16` the shard is 54.5 GiB of ephemeral volume.

**One caution about `torch.from_numpy` on a memmap.** It is zero-copy — the tensor shares the ndarray's memory ([PyTorch docs](https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html)) — which means the storage is *pageable and file-backed*, so reads of non-resident pages fault synchronously on the critical path. `pin_memory=True` in the workers fixes this; note `training.py:356` already sets `pin_memory=(device.type == "cuda")`.

### 5.3 What Session 0 actually needed

The plan's Session 0 says to *"update `src/adjointrwm/data/windows.py` to enable multi-process worker pools"*. That change is **already released** — `TrainConfig.num_workers` exists and `train_job` wires `num_workers=cfg.num_workers`, `pin_memory`, and `persistent_workers` (`training.py:294`, `:354-357`). It is recorded in `CHANGELOG.md` (multi-worker prefetching support) and is in `HEAD`; no `src/` file is uncommitted.

The remaining gap is that several benchmark scripts bypass `train_job` and build their own loaders with no worker configuration: `src/adjointrwm/allocators.py:450`, `src/adjointrwm/training.py:170`, `scripts/benchmark_curvature_voi_allocator.py:307-308`, `scripts/train_e3_dynamics_pilot.py:292`. Session 0's real deliverable is a **real-data** profile of the actual `WindowDataset` over `/content/cache_e3_1`, not a synthetic microbenchmark.

### 5.4 Library guidance, with corrections

A subagent verified the following. Two widely-repeated premises are **wrong**:

| Claim | Status |
|---|---|
| "Colab has ~90 min idle timeout" | **Unverified.** Google states it *"does not publish these limits."* Only 12 h (free) / 24 h (Pro+) max lifetime is documented. Colab **Enterprise** documents 180 min idle shutdown |
| "Colab has ~100 GB ephemeral disk" | **Unverified.** No Google-published figure. Probe `df -h /content` at runtime |
| "Colab signals preemption" | **Unverified.** Only compute-unit-exhaustion termination is documented |
| "Zhang et al., An Audit of ML Engineering Practices" | **Does not exist** under that name. Substitute **Sculley et al. 2015**, *Hidden Technical Debt in ML Systems* |
| WebDataset sequential-vs-random = 3–10× | Author's own README claim, not independently benchmarked |

**Google's own advice validates this repo's architecture.** The Colab FAQ advises avoiding many small I/O reads and *"unarchive the data locally on the VM"* rather than reading from mounted Drive — exactly `colab_l4_operator_brief.md` rule 9.

**Do not reach for `torch.compile(mode="reduce-overhead")`.** NVIDIA's CUDA-graph guidance ([docs](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html)) states graphs *"only help when CPU kernel launch overhead is the bottleneck"* and are useless past ~95 % GPU utilisation. A world-model run is GPU-bound; use `mode="default"`.

**Never use `non_blocking=True` on D2H copies.** PyTorch's own tutorial states this *"will result in erroneous outputs"*. Use a pinned staging buffer plus an explicit event.

**L4 hardware facts** (NVIDIA datasheet rev. APR23), relevant to the bandwidth argument: 24 GB, **300 GB/s** memory bandwidth, **4 hardware JPEG decoders**, PCIe Gen4 x16 at 64 GB/s. The 300/64 ≈ 4.7× ratio means any step crossing PCIe is host-link-bound, so budget ≥5× more GPU time than transfer time before blaming the GPU.

### 5.5 Colab-native layout, if the shard outgrows one VM

At `grid=16` the shard is 54.5 GiB, close to the ephemeral volume ceiling. The format the robotics community converged on — HuggingFace **LeRobot v3** — solves precisely this: *"Many episodes per Parquet/MP4 file (v2 used one file per episode)"*, episode boundaries *"resolved through metadata, not filenames"*, with the stated principle *"decoupling storage from the user API"*.

The constraint for this repo is that LeRobot shards are episode-aggregated, which sits badly with research-integrity rule 5 (*split by episode before windowing*). It is also a migration of the DROID shard, not a refactor, so it is **not** recommended mid-campaign. Recorded here as the exit path if `grid=16` on 500 episodes proves infeasible on one VM.