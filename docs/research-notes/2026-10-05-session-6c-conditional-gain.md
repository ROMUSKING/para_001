# Session 6C conditional-gain experiment: conditional benefit prediction vs matched learned singleton (diagnostic re-run)

**Date:** 2026-10-05 · **Status:** CLOSED — practical superiority not established at any
tested budget; norm/fixed control retained; no escalation. This note adopts the reviewed
closure language verbatim in §5.
**Artefacts:** `results/benchmarks/conditional_heads/benefit_heads_summary.json`,
`results/benchmarks/conditional_eval/conditional_eval_summary.json` (with per-window rows),
`results/benchmarks/conditional_protected/protected_manifest.json` + `.seal` (verified).
**Teacher backbone:** `/content/local_runs/runs/spatial_patches_20261003/best_spatial.pt`,
SHA-256 `cfbca4406f13df55…` (same prefix as Sessions 6A–6C runs; absolute J not compared
across sessions per DEV-20261004-02). Backbone smoke score −2.1425 (identity not claimed).
**Heads:** singleton BenefitRegressionHead (MSE on measured singleton gains) 1,579,501 params;
conditional head (same inputs + mean-selected-embedding + set size) 1,579,266 params
(capacity-matched via `matched_patch_critic_hidden`). 300 steps, student fraction 0.50/0.50
(capped), 79,200 label rollouts. Weights ephemeral (not committed); SHA-256 in heads summary.
**Protected set:** 150 VAL-split windows, episode-disjoint from diagnostic episodes, single
read logged, seal verifies locally. Prior 74-worker run (degenerate CIs, no per-window rows,
seal/weights lost with the VM) is superseded and must not be cited — this note replaces it.

## 1. What changed since the first run (codex review, all accepted)

Positional CI calls bound `confidence=seed=0` (degenerate intervals); now keyword-called.
`theta_gate` extracted as a pure tested function with an explicit absolute-margin fallback
for windows with R_single ≤ 1e-12. Training masks are quota-legal by construction (fixed
per-camera ∼U{0..8}; student score-ordered prefixes of per-camera greedy top-8). Evaluate
stores per-window Δ, both regrets, and sites. Cost gate is report-only (greedy conditional
scoring cannot pass strict no-higher-cost by construction). Repeat protected reads require
`--acknowledge-reread`. This re-run is diagnostic: all budgets reported without multiplicity
control; a confirmatory run would gate on k_cam=4 only.

## 2. Results (150 protected windows, 12 sites, seed 0, 2000 resamples)

| budget | Δ trim (cond−single) | Wilcoxon | site-CI₉₅ | θ one-sided 95% lower | sup 8% | non-inf 3% | harmed frac (p10 / p05) |
|---|---|---|---|---|---|---|---|
| k=2 | +0.000590 | p=0.0016 | [0.000134, 0.001369] | −0.1580 | NO | NO | 0.39 (−0.002205 / −0.002993) |
| k=4 (primary) | +0.000600 | p=8.3e-05 | [0.000447, 0.001022] | −0.0137 | NO | yes | 0.40 (−0.001246 / −0.001581) |
| k=8 | +0.000030 | p=0.69 | [−0.000068, 0.000471] | −0.2738 | NO | NO | 0.53 (−0.001306 / −0.001642) |

Label precision (reviewed): at k=4, non-inferiority is not superiority or equivalence —
the lower bound clears −3% but not +8%. At k=8 the result is "no detected difference"
(p=0.69 establishes no equivalence bound on its own). Small Wilcoxon p-values at k=2/k=4
establish an effect, not its size; the minimum-effect (θ) rule answers the practical
question and it fails. Harmed fractions describe distribution unevenness; the magnitudes
above are the prespecified tail guardrail output, retained alongside — not a separate
pass/fail invented here.

Regret scale (trimmed): singleton 0.00390/0.00350/0.00223 vs conditional 0.00327/0.00301/0.00222.
Conditional vs norm (means): +0.000359/+0.000428/+0.000024 (positive favours conditional);
vs uniform: +0.000323/−0.000039/−0.000049. Scoring calls: conditional 532 vs singleton 19 —
exactly 28× as many calls, which is not a latency, FLOP, or end-to-end cost ratio (call
contents and batching differ). Status kept separate per review: **primary quality gate
failed; cost criterion unfavourable, reported separately, not used as a retroactive veto.**
No optimisation of the 532-call path follows: retained decision value is insufficient to
justify that systems work. No θ-excluded windows (excl=0 at all budgets; fallback vacuous).

## 2b. θ estimand as implemented (reproducibility record, closure amendment 2026-10-05)

The reviewed and executed quantity is the window-wise construction with T = mean:

θ_window = mean_w(1 − r_cond,w / r_single,w), kept windows R_single > 1e-12,

NOT the ratio of aggregate trimmed regrets θ_aggregate = 1 − T(r_cond)/T(r_single).
Near-zero singleton regret produces extreme individual ratios under the window-wise form;
that is the documented reason the one-sided bound sits far below the mean-Δ signal, and it
is not grounds to select a more favourable formula now. Denominator handling: every other
window excluded to the absolute-margin fallback (n=0 here). Clustering: whole sites with
replacement, 2000 resamples, seed 0 (two-sided) / seed+1 (one-sided 5th percentile of
resampled means). Disclosure: Δ uses the trimmed mean while θ bootstraps means — the spec
said "agreed aggregation (trimmed mean)" but the implemented θ uses mean-of-ratios. This
inconsistency is recorded, not repaired post-hoc; a confirmatory run would freeze one
aggregation for both. The failed gate does not rule out an 8% benefit with certainty — it
means the evidence required to establish it was not supplied, which suffices for closure.

## 3. Reading

The mean-Δ and Wilcoxon signals at k=2/k=4 do not survive the frozen θ rule: per-window
ratios are heavy-tailed (small single-window denominators), so the clustered one-sided
lower bound is deeply negative even where the mean wins. The gate behaved as designed —
a mean driven by larger wins on ~60% of windows against losses on ~40% is not a robust
allocation improvement. At k=8 there is no detected difference. Combined with the
unfavourable call-count overhead and the mixed-at-best norm comparison, **no learned arm
demonstrates useful improvement over the norm/fixed control at acceptable cost**. Per the
frozen routing, this closes the learned-singleton/conditional formulation: retain the
norm/fixed control (an operational fallback among investigated options, not a universal
optimality claim), do not escalate capacity, hardware, training length, or encoder surgery.
A conditional win here would in any case have been a practical pipeline comparison, not
evidence that selected-set identities caused it. The context-blind attribution control
stays **deferred — it requires a separate scientific justification and is not required to
close this formulation**; it becomes relevant only if a future prospectively specified
experiment establishes a conditional advantage whose source matters. No co-state claim
attaches (direct benefit predictors only).

## 4. Supersession and exposure history of the 74-worker attempt

The 74-worker run is **invalid/superseded: excluded from all scientific summaries and
pooled analyses** (degenerate CIs, no per-window rows, seal/weights lost with the VM). It
supplies neither positive nor negative evidence. Exposure disclosure: its manifest file was
overwritten by the 75-worker download at the same path, so episode overlap between the two
protected sets cannot be established from preserved artefacts (both used seed 0 over
identically built caches; overlap is likely but unverified). Observed 74-worker output did
influence the redesign in one way — the degenerate intervals exposed the positional-arg CI
bug — but no model, threshold, budget, or stopping decision was tuned on protected
outcomes: heads were retrained from scratch under the codex-amended schedule, budgets were
spec-frozen, and the single 75-worker read postdates all selection. The one recorded read
of the new seal establishes access history for the new run only.

## 5. Closure record

| Status | Basis |
|---|---|
| Execution | Completed on 75-worker L4; session stopped, 0 assignments |
| Software checks | 573 pytest passed; `harness/check.py` 6/6 (reported verification, not evidence) |
| Evidence integrity | Final run has per-window artefacts, recorded verification, verified seal; 74-worker excluded |
| Practical superiority | Not established at any tested budget (small favourable Δ at k=2/k=4 retained visibly above) |
| Deployment decision | Retain the existing norm/fixed default |
| Co-state hypothesis | No new supporting claim from this experiment |

Programme synthesis (kinds of outcome, not equivalent failures): Session 5 —
withdrawn/unsupported, not a valid replication in either direction; VOI-vs-curvature —
retired as non-diagnostic; exact-vs-learned gradient gap — diagnostic opportunity, not
demonstrated deployability; tested Hessian correction — closed negative mechanism result
for that approximation; frozen-DINOv2 pixel masking — negative quality comparison, no
encoder savings demonstrated; conditional-vs-learned-singleton (75-worker) — small
favourable differences at lower budgets, practical-superiority gate not met, formulation
closed.

> **The protected 75-worker evaluation did not establish the prespecified practical
> superiority of conditional-gain selection over matched learned-singleton selection at any
> tested budget. Positive trimmed-regret differences at k=2 and k=4 did not satisfy the
> relative-improvement confidence-bound rule. The primary k=4 result met only the declared
> non-inferiority criterion. Conditional selection required 532 versus 19 scoring calls;
> this is reported as call-count overhead, not an end-to-end latency ratio. The 74-worker
> attempt is invalid and superseded. The current formulation is closed, the norm/fixed
> control remains the operational default, and no co-state advantage is claimed.**
