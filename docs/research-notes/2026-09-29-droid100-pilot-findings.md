# DROID-100 production pilot: findings

**Run:** `droid100_adjoint_20260929T070629Z` · **Date:** 2026-09-29 · **Notebook:** `notebooks/01-production/AdjointRWM_Production_Pilot.ipynb`
**Hardware:** NVIDIA L4 (22 GiB), PyTorch 2.11.0+cu128, BF16 autocast · **Config SHA-256:** `433e7473…affa87` (checked against `config/run_config.json`)
**Artefacts:** `results/runs/droid100_adjoint_20260929T070629Z/`

**Status:** the data and dynamics gates pass; the allocation gate fails, so production status is **BLOCKED**. Task success was not measured, because no simulator or robot was attached.

---

## 1. What this run is

This is an offline, action-conditioned recursive world model trained on the official 100-episode DROID sample (`droid_100`, RLDS), with no synthetic fallback.

- **Data:** 80/10/10 episodes, split by episode *before* any windows were cut (5,910 / 796 / 835 windows). Each window has 8 context steps and a 4-step horizon at frame stride 2.
- **State and action:** the state is 14-D (cartesian position, gripper, joints) and the action is 7-D.
- **Visual target:** frozen ImageNet ResNet-18 embeddings from the exterior and wrist cameras (512-D each).
- **Model:** 27.36 M trainable parameters (d_model 512, 6 transformer layers).
- **Refinement candidates:** 4 candidates, each a learned latent perturbation `δ_k = 0.10·tanh(refiner(z, e_k))` with cost `c = 0.002·(1, 1, 1.5, 2)`.
- **Allocators:** three allocators pick **one** candidate per window.
  - **Adjoint:** first-order score `−⟨λ̂, δ_k⟩ − c_k` using an amortised co-state `λ̂`.
  - **Critic:** a direct marginal-gain critic given the same inputs.
  - **Hybrid:** a gate that chooses between the adjoint and critic paths.
- **Labels:** exact per-candidate gains come from a frozen copy of the dynamics model (the "teacher"), using a Gaussian-NLL + 0.25·visual-cosine objective.

## 2. Results

### 2.1 Dynamics: pass

| Horizon step | Model RMSE | Persistence RMSE | Model vs persistence |
|---|---:|---:|---:|
| 1 | 0.1334 | **0.1172** | −13.8 % (persistence better) |
| 2 | 0.1453 | 0.1958 | +25.8 % |
| 3 | 0.1631 | 0.2659 | +38.7 % |
| 4 | 0.1806 | 0.3258 | +44.6 % |
| **Mean (gate)** | **0.1556** | **0.2262** | **+31.2 %** (required ≥ 2 %) |

- The visual-embedding cosine on test is 0.769.
- **Caveat 1:** the gate averages over the horizon. At one step ahead, "nothing changes" is still better than the model. That is normal for smooth teleoperation at this frame rate, but it should be reported per horizon step.
- **Caveat 2:** refinement barely changes the state prediction. Base (no candidate) and full (all candidates) test RMSE differ by less than 0.3 % at every horizon step (e.g. 0.13302 vs 0.13341 at step 1).
- **Caveat 3:** this is one seed on 10 test episodes, so there is no confidence interval yet.

### 2.2 Allocation: fail

Headline numbers come from `artifacts/allocator_evaluation.json`. The episode-level bootstrap CIs are my re-analysis of `artifacts/allocation_traces.parquet`, stored in `artifacts/trace_summary.json` and produced by `scripts/analyze_allocation_traces.py`.

| Policy | Test regret (mean) | 95 % CI (episode bootstrap, 10 eps) | Top-1 vs oracle |
|---|---:|---|---:|
| Adjoint (amortised co-state) | 0.0374 | [0.0285, 0.0488] | 15.7 % |
| Direct critic | 0.0383 | [0.0291, 0.0512] | 15.8 % |
| Hybrid | 0.0383 (identical to critic) | — | 15.8 % |
| Random (one draw per window) | **0.0211** | not logged per window | ~25 % expected |
| Chance top-1 | — | — | 25.0 % |
| Best constant choice ("always c1") | not logged | — | **34.6 %** |

- **Adjoint minus critic:** −0.0009, 95 % CI [−0.0028, +0.0004]. This is a tie, so the run gives no evidence either way on H2.
- **Gate:** it was never invoked (maximum gate probability 0.24, threshold 0.5), so "hybrid" is just the critic.
- **Regret is heavy-tailed.** The median is 0.018, p90 is 0.081 and p99 is about 0.33. Per-episode mean regret ranges from 0.006 to 0.089, so a few episodes dominate the average.

### 2.3 Why the learned allocators lose to random

The trace file answers the question directly.

| Choice share on test | c0 | c1 | c2 | c3 |
|---|---:|---:|---:|---:|
| Oracle | 20.8 % | **34.6 %** | 13.7 % | **30.9 %** |
| Adjoint | 49.1 % | **0 %** | 50.9 % | **0 %** |
| Critic | 48.0 % | **0 %** | 52.0 % | **0 %** |

- **Candidates 1 and 3 are never chosen.** Both learned allocators ignore them entirely, although the oracle prefers one of them on **65.5 %** of test windows. The allocators have collapsed onto the pair {c0, c2} and agree with each other on 85.5 % of windows.
- **This explains below-chance top-1.** If you only ever pick from the two candidates the oracle likes least (34.5 % of oracle picks between them), you do worse than picking uniformly.
- **The problem is transfer, not missing opportunity.** In training (allocator logs, train batches, dropout on), regret was about 0.002–0.006 and rank correlation with exact gain was 0.2–0.5. On test, median regret is 0.018 and tail regret reaches 0.6.
  - Earlier I said "there's nothing to allocate". **That was wrong for the test split.** On test, the gain differences between candidates are 5–10× larger than the cost gap (0.002).
  - What *is* small is the opportunity on the **training** windows. There the dynamics model fits well, so refinements barely move the objective, and the allocator heads get a weak, possibly biased signal that does not carry over to test.

**Working hypothesis (not yet tested):**

1. The dynamics model's residual errors differ systematically between train and held-out episodes.
2. Which latent perturbation helps therefore depends on the episode.
3. The heads learned a preference for c0/c2 that holds on training windows.
4. With the objective in Gaussian-NLL units, test windows with larger errors produce much larger gain differences, and the learned preference does not hold there.

`notebooks/02-diagnostics/opportunity_audit.ipynb` tests this directly. It logs the full gain matrix on train, validation and test, and compares oracle shares and gain spreads between splits.

### 2.4 Design issues found while reading the code

These are independent of the result above and should be fixed in the next run.

1. **No hold option.** The allocator must always apply exactly one candidate, so "refine nothing" is not a legal choice. The research plan requires hold/stop semantics (§3.4, §5.7). If every candidate hurts on a window, regret is measured against the least-bad candidate, not against doing nothing.
2. **Training and evaluation semantics differ.** Stage 1 trains candidates under random subsets of size 0–4, but evaluation applies a single candidate. Candidate effects are therefore tuned to work *in combination*, then scored *alone*.
3. **The gate target is almost never positive.** The gate label is `critic_regret − adjoint_regret > 0.002`. With both regrets around 0.004 on train, that almost never happens, so the gate learns "never invoke". The target has to be scaled to the regret distribution, e.g. a sign-only label with a margin taken from validation.
4. **"Random" baseline is one draw.** `random_regret` is a single random draw per window. It should be the exact expectation, `mean_k(max gain − gain_k)`, and there should also be "always c_k" baselines. Without those, "beats random" is noisy and "beats best fixed" can't be checked at all. The plan's opportunity gate (§7.2B) needs the best-fixed comparison.
5. **The co-state is w.r.t. the latent only.** That's fine for a pilot, but the plan defines `λ` on the complete deployed state `x_k = (z, b, q, r, c, τ)`. Report it as a latent-state co-state.
6. **The rate ledger is not canonical.** It uses zlib byte counts, and the notebook already labels it "not the final canonical coder". Keep it out of any rate claim.

### 2.5 Systems

| Path | p50 latency | p99 latency |
|---|---:|---:|
| Direct critic | 0.24 ms | 0.27 ms |
| Amortised adjoint | 0.28 ms | 0.50 ms |
| Exact adjoint teacher (backward pass) | 15.3 ms | 17.8 ms |

- **Memory:** peak training memory is 0.60 GiB at microbatch 64. Throughput is about 22 steps/s regardless of batch size, so the run is most likely launch- or data-loading-bound, not memory-bound.
- **Hardware:** stay on the L4. An A100 is not justified by anything measured here.
- **Rate pilot:** 2.31 bits per candidate scalar (zlib; not canonical).

## 3. What the run does and doesn't support

- **Supports:** a 27 M-parameter recursive world model beats persistence on DROID-100 held-out episodes at horizon steps 2–4 (single seed).
- **Supports:** the checkpoint round-trip, split-disjointness and future-shift contracts all hold.
- **Does not support:** any statement that adjoint guidance helps or hurts relative to the direct critic. The two are tied, and both collapsed.
- **Does not support:** any statement about task success, rate savings or planning.
- **Contradicts:** the paper draft's claim that the allocator "systematically outperforms non-adjoint direct search baselines" (see `papers/drafts/REVIEW.md`).

## 4. Next steps (ordered; details in `docs/plans/roadmap.md`)

1. **Run the opportunity audit** on this checkpoint, with no retraining. It answers: best-fixed vs oracle headroom per split; how the exact co-state does in the same first-order score (a ceiling for any amortised estimator); how train and test gain distributions differ; and the effect cosines.
2. **Fix the evaluation semantics:** add a hold/no-op candidate; add exact expected-random and always-c_k baselines; log `exact_gain[N,K]` in the traces; and redefine the gate target.
3. **Fix the train/eval mismatch.** Either train candidates in the same single-choice regime used at evaluation, or evaluate subset allocation under a budget, which is closer to the plan's best-first allocator.
4. **Then rerun with ≥ 3 seeds**, keeping the same split and paired seeds. Report adjoint-minus-critic regret with an episode-cluster bootstrap CI. Promote only if the opportunity gate passes on validation.
