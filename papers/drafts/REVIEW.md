# Claim check: `adjoint_guided_recursive_world_models_paper.md`

**Draft source:** Drive file `18XE3Wf8wvOAN8xKQUE9GOp7zVCFXiZf2`, written 2026-09-28 19:14, which is before the DROID-100 pilot ran.
**Rule used:** every claim must trace to a logged, reproducible result. If it doesn't, it gets cut or rewritten as a hypothesis.

| # | Claim in draft | Evidence available | Verdict | Action |
|---|---|---|---|---|
| 1 | "our causal adjoint-guided allocator systematically outperforms non-adjoint direct search baselines" (Abstract) | The DROID-100 pilot measured adjoint regret 0.0374 against critic 0.0383. The difference is −0.0009 with CI [−0.0028, +0.0004]. Both allocators lost to random. | **Contradicted** | Remove. Report the tie and the failure mode. |
| 2 | Stage 0: analytic co-state matches autograd to ≈1e-16 in float64 | Not in the repo. The Stage 0 notebooks are on Drive but haven't been reviewed. | Plausible but unverified | Keep once the Stage 0 notebook and its logs are committed. This is a correctness check, not a contribution. |
| 3 | The range coder reached 3.8400 bits/symbol, "outperforming the theoretical Shannon entropy threshold" of 3.8611 | No log. | **Wrong as stated** | A lossless code can't beat the entropy of the true source distribution. Beating an *estimated* entropy just means the estimate differs from the coder's model. Rephrase as "within X of the empirical entropy under model Y", or cut. |
| 4 | Selective invocation gate reduces latency to 0.27 ms per sample | The pilot measured the amortised adjoint at 0.28 ms p50. The gate was never invoked (0 % of test windows). | Misleading | Report the latency of the amortised path. Don't attribute it to a gate that never fired. |
| 5 | Embodied planning converged to trajectory cost 8.347e-5 with "100 % causal-taint isolation" (B=16, N=8, D=32) | `phase_n10_embodied_telemetry.json` holds one step of synthetic state. | **Unsupported** | Remove. Nothing embodied was evaluated. |
| 6 | Deployment advice: use FP64-capable A100/H100 for real-time use | The pilot ran in BF16 on an L4 at 0.6 GiB and reported no precision issues. | Unsupported | Remove, or test directly. |
| 7 | Theory §2: LQ system, Pontryagin co-state recursion `λ_k = Q(x_k−g) + Aᵀλ_{k+1}` | This is standard discrete-time optimal control. | Correct but generic | Keep as background, cite it, and state the discrete trace/objective precisely, as the plan does in §4.2. |

## What a paper could say now

At present, the defensible contribution is methodological. It consists of:

- a preregistered protocol for testing whether objective-sensitivity (co-state) signals improve refinement allocation over matched direct critics, and
- a first real-data pilot showing that the evaluation machinery works and exposes an allocator failure mode: collapse onto a subset of candidates under a train/test shift in gain distributions.

That is a workshop-paper or tech-report shape, not a main-track claim. A superiority claim needs the plan's Stage 0 → I.2 → I.3A gates, at least 8 seeds and a passing opportunity gate.
