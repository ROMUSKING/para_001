# Session 6A diagnostic: the bottleneck is the λ̂ distillation, and the earlier headline was a sampling artefact

**Date:** 2026-10-04 · **Milestone:** Session 6A remediation · **Hardware:** NVIDIA L4 (22.03 GiB, peak 1.7 GiB)
**Artefact:** `results/benchmarks/spatial_bottleneck/spatial_bottleneck_summary.json`
**Runner:** `scripts/diagnose_spatial_selection_bottleneck.py`
**Supersedes part of:** [`2026-10-04-session-6a-spatial-selection.md`](2026-10-04-session-6a-spatial-selection.md)
**Audit addressed:** [`../audits/2026-10-04_session_6a_voi_scale_and_sampling_audit.md`](../audits/2026-10-04_session_6a_voi_scale_and_sampling_audit.md) remediation items 2 and 4

**Status: this note corrects a claim in the Session 6A note.** The earlier finding that belief-space
VOI "loses to `early_feature_norm` by −8.9 %" does not survive episode-stratified sampling. It was
substantially an artefact of a 3-episode evaluation set. What replaces it is narrower and points
somewhere else.

---

## 1. What was measured

192 held-out windows drawn **stratified over 50 test episodes and 12 sites** (was 3 episodes /
3 sites). On each window, five scorers are compared against the **exact** marginal gains
`J(z_S) − J(z_{S∪p})`, and each is also *used* to select at every budget, so ranking fidelity and
realised regret are measured on the same windows.

The backbone reproduced bit-for-bit across sessions (final selection score −2.1425 both times,
identical at every logged step), so these numbers sit on the same substrate as the Session 6A run.

## 2. Rank fidelity against exact marginal gains

Spearman ρ per window, averaged over 192 windows, against exact first-order marginal gains:

| scorer | ρ | regret @ k_cam=2 (trimmed) |
|---|---|---|
| `exact_gains` (the true gains themselves) | **+1.0000** | **0.000034** |
| `first_order_exact` (autograd cotangent, Tier 2) | **+0.9632** | **0.000082** |
| `delta_sigma_only` (epistemic term alone) | **−0.1834** | 0.005178 |
| `curvature_distilled` (λ̂ + Ĥ, deployable) | **+0.1151** | 0.003875 |
| `patch_norm` (raw DINOv2 patch L2 norm) | +0.0795 | 0.003916 |

**The additive form is not the ceiling.** Ranking by the *true* marginal gains reaches trimmed
regret 3.4e-05 — better than the autograd reference and ~110× better than the distilled scorer. So
"per-patch scoring is the wrong functional shape" (the hypothesis in the Session 6A note) is
**refuted as the limiting factor**. Score-then-top-k can express an excellent policy here. The
problem is that the scorers do not recover the true gains.

A sharper reason the earlier additivity argument should be dropped, raised by the rotating critic
(`cline`): `additivity_r2` regresses measured `J(S)` on `Σ_{p∈S} s_p` where `s_p` are the **policy's
own uncalibrated gains**, pooled across selections from many different policies. Gains are in
cotangent units and `J` is in nats, and the fitted slope is ≈ −4.7e-05 against an intercept of
≈ −1.59 (the mean `J`). `R² ≈ 0` is therefore close to *guaranteed* by the unit mismatch and the
pooling, regardless of the true interaction structure. It is not evidence about non-additivity.

One limit on the ceiling result, stated so it is not over-read: `exact_marginal_gains` is measured
at the **null** selection (`base_mask=None`), so this shows that even null-set first-order marginals
suffice for an excellent top-k policy. It does **not** establish that gains stay accurate at larger
selections, where interactions could still bite.

**The distillation is the bottleneck, and capacity is not the fix.** Going from the autograd
cotangent (ρ = 0.9632, regret 8.2e-05) to the distilled head (ρ = 0.1151, regret 3.9e-03) costs
~47× in regret. Widening the estimator does not help:

| head hidden | params | ρ(distilled) | regret @ k_cam=2 |
|---|---|---|---|
| 512 | 1,051,648 | 0.1020 | 0.003875 |
| 1024 | 1,577,472 | 0.1151 | 0.003766 |

Tripling the conditioning MLP width moves ρ by 0.013 and regret by 3 %. **The distilled head is
not capacity-limited.** That is sharper and more actionable than "distillation gap", and it rules
out the obvious first response.

## 3. The epistemic term is mildly *anti*-correlated — worse than inert

The audit established that the β sweep is inert because the term is 9.1e-06 of the curvature term.
Measured on its own, the term is not merely small, it points the **wrong way**: ρ = **−0.1834**,
and it has the **worst regret of every scorer at every budget** (0.005178 at k_cam=2 against
`patch_norm`'s 0.003916).

So the spec B2 control was not merely uninformative — the quantity it controls is negatively
associated with the true marginal gains at this operating point. **Scaling β up to give the term
leverage would make allocation worse, not better.** That settles the audit's open question about
whether to enlarge β: no.

Caveat, stated precisely: ρ here is measured against *first*-order marginal gains at the null
selection, while `ΔΣ` is a *second*-order quantity at that operating point, so a weak negative
association is not proof the concept is wrong. It is sufficient to reject "just raise β".

## 4. Correction: the Session 6A headline does not survive stratification

The stratified re-run of the full suite (1,200 windows, **50 episodes, 12 sites**, 3 seeds,
`results/benchmarks/spatial_selection_stratified/`) overturns more of the original result than the
diagnostic alone did. Criterion 1 advantage for belief-space VOI:

| budget | vs `uniform_grid` (3 ep) | vs `uniform_grid` (50 ep) | vs `early_feature_norm` (3 ep) | vs `early_feature_norm` (50 ep) |
|---|---|---|---|---|
| k=4 | +24.7 % | **+3.96 %** | −8.9 % | **−5.2 %** |
| k=8 | +5.6 % | **−2.2 %** | −0.8 % | **−2.9 %** |
| k=16 | +20.5 % | **+1.0 %** | −4.2 % | **−3.2 %** |

**Both** baseline comparisons were inflated by the 3-episode sample. On 50 episodes every
non-reference policy sits within about ±5 % of every other non-reference policy — trimmed-mean
regret at k=4, seed 0:

| policy | trimmed | site-clustered 95 % CI (12 sites) |
|---|---|---|
| `calibrated_greedy_oracle` | 0.00000 | — (defines regret) |
| `exact_costate_reference` | **0.00008** | [0.00009, 0.00018] |
| `early_feature_norm` | 0.00443 | [0.00446, 0.00543] |
| `belief_space_voi` / `second_order_curvature` / `early_cls_attention` | 0.00464 | [0.00468, 0.00565] |
| `direct_ranking_critic` | 0.00464 | [0.00465, 0.00575] |
| `stratified_random` | 0.00476 | [0.00469, 0.00578] |
| `uniform_grid` | 0.00485 | [0.00470, 0.00583] |
| `direct_critic_privileged` | 0.00502 | [0.00480, 0.00625] |

Every CI overlaps every other. The paired Wilcoxon agrees: VOI vs `direct_ranking_critic`
**p = 0.94** (no difference), and VOI vs `second_order_curvature` has **1,200 of 1,200 ties**,
p = 1.0 — the two policies pick *identical* patches.

Criterion 2 also flips: VOI was non-inferior to `direct_critic_privileged` on 3 episodes and is
**not** on 50 (it is nominally better, p = 0.0065, but outside the 3 % margin).

So the honest statement is not "geometric coverage dominates" and not "VOI loses to a norm
heuristic". It is: **at this operating point no deployable selector separates from any other,
including random stratification; the only policies that separate are the ones with autograd
access.** The pre-registered branch label was reached in both runs, but for different reasons and
with the effect sizes revised down by 5–20×.

Note the interaction with §3: `belief_space_voi` and `second_order_curvature` selecting identical
patches at 1,200/1,200 windows is the degenerate β sweep showing up in the selection itself, not
merely in the summary. The β values again return one identical regret (0.00465522) to eight
decimals.

**v2 rerun (same day, current code):** `results/benchmarks/spatial_selection_v2/` reproduces the
stratified numbers to the last digit (k=4 VOI 0.00466, +3.96%/−5.18%; 108,000 rows; 3 distinct
seeds) and is the first committed artefact in which the degeneracy guard fires:
`beta_sweep_degeneracy.degenerate: true` at all three budgets,
`exit_gate.branch: not_evaluated: degenerate beta sweep…`,
`downgraded_because: beta_sweep_degeneracy`, `tie_with_curvature: null`. The tie is now
reported as untested by the artefact itself, not just by this note.

## 5. Where the bottleneck actually is: the privilege ablation

The single "~110×" number above confounds three changes at once — distilled vs exact λ, presence
vs absence of the curvature term, and first-order vs second-order. The ablation changes **exactly one
factor per row**, which is what the rotating critic (`cline`) identified as the missing experiment.

192 stratified windows, budget k_cam=2. "overlap" is the fraction of the true top-2 patches the
scorer also selects; chance under a symmetric 2-of-16 per-camera budget is 0.125.

| scorer | what changed | ρ | regret | overlap (chance 0.125) |
|---|---|---:|---:|---:|
| `exact_gains` | the true marginal gains | +1.0000 | 3.4e-05 | 1.000 |
| `curvature_exact_costate` | exact λ, distilled Ĥ | +0.9771 | **2.3e-05** | 0.868 |
| `first_order_exact_costate` | exact λ, no Ĥ | +0.9771 | **2.2e-05** | 0.870 |
| `first_order_exact` | autograd cotangent | +0.9632 | 8.2e-05 | 0.844 |
| `curvature_distilled` | distilled λ̂, distilled Ĥ | +0.1151 | 3.875e-03 | 0.253 |
| `first_order_distilled` | distilled λ̂, **no Ĥ** | +0.1151 | 3.875e-03 | 0.253 |
| `patch_norm` | no learning at all | +0.0795 | 3.916e-03 | 0.236 |
| `delta_sigma_only` | epistemic term alone | −0.1834 | 5.178e-03 | 0.180 |

Three readings follow, and they are unambiguous:

1. **λ̂ is the entire bottleneck.** `curvature_distilled` and `first_order_distilled` are identical
   to four decimals (ρ 0.1151, regret 3.875e-03, overlap 0.253). **The curvature term contributes
   nothing** — dropping Ĥ entirely changes nothing. Whatever the distilled head is failing at, it
   is failing in the co-state *direction*.
2. **Ĥ is harmless when λ is exact.** `curvature_exact_costate` (ρ 0.9771, regret 2.3e-05) is as
   good as `first_order_exact_costate` (ρ 0.9771, regret 2.2e-05). So the diagonal-Hessian head is
   not the problem either, and the second-order functional form is not the problem.
3. **The functional form is not the problem.** Exact λ in either form reaches ~2e-05, 170× better
   than the distilled scorer, in the *same* additive surrogate and the *same* top-k selection.

The distillation gap is therefore not a "gap" in the general sense: it is one broken component. The
co-state hypothesis is not what failed here. And because rows 2–3 recover essentially all of the
achievable performance, **the ceiling for a deployable selector on this problem is ~2e-05 trimmed
regret**, which the current λ̂ misses by ~170×.

A methodological note worth carrying forward: **top-k overlap discriminates far better than mean
Spearman ρ.** `patch_norm` has ρ = 0.0795 — apparently useless as a ranker — yet selects 23.6 % of
the true top-2 against a 12.5 % chance level, twice chance. ρ over 32 patches is a global ordering
statistic and badly understates a scorer whose *selected set* is decent. Any future tuning against ρ
alone would optimise the wrong thing.

## 6. Next experiment

The ablation in §5 removes most of the candidate explanations, so the remaining work is narrow:

1. **Why does λ̂ fail?** It is trained with cosine alignment to the exact costate plus a curvature
   term, from the pooled latent `z` alone. Since the curvature term contributes *nothing*
   (§5, rows 5–6 are identical), the first thing to test is whether the cosine target is simply the
   wrong objective — a direction that is cos-correct on average need not rank correctly. Train λ̂
   directly on a **ranking** loss against exact marginal gains and compare overlap at fixed budget.
2. **Is `z` sufficient?** Per-patch utility is being predicted from one pooled vector per window
   that has already discarded *which* patch was which. Condition the head on the candidate patch's
   own embedding instead. This is the same hypothesis §5 leaves standing, since fixing λ̂ with the
   current input could still cap performance below the 2.2e-05 ceiling.
3. **Spec amendment required before any further VOI run** (critic P0, agreed): the squared-ΔΣ term
   is not commensurate with the curvature term by construction, and §3 shows it is mildly
   *anti*-correlated. Standardise both terms to unit inter-patch spread and re-preregister the β
   grid on that scale, or drop the term. Do **not** resolve it by enlarging β.

## 6.1 Norm controls (still open, CPU-only)

The critic is right that §4's `patch_norm` result is a statement about **DINOv2 patch norms plus a
patch-dropout-trained backbone**, not about learned allocation in general — `patch_norm` has almost
no rank fidelity yet ties the learned scorer. Three cheap controls on committed artefacts would
settle it, and none needs a GPU:

- top-2 overlap of `patch_norm` against **random** stratified selection, to confirm 0.236 is
  meaningfully above the 0.125 chance level rather than an artefact of the layout;
- `patch_norm` conditioned on grid position, to test whether it is simply picking centre or
  foreground cells;
- `patch_norm` against per-patch measurement noise, to test whether high-norm patches are simply
  the better-conditioned ones.

## 7. Addendum 2026-10-04 (later): WS1a/WS1b executed — the training objective is not the bottleneck

Artefact: `results/benchmarks/spatial_bottleneck_ws1b/spatial_bottleneck_summary.json`
(192 stratified windows, 50 episodes, 12 sites; backbone bit-identical, score −2.1425).
Runner: same `scripts/diagnose_spatial_selection_bottleneck.py` with `--ranking-margin-scales`.
Peer review: codex, recorded for milestone `Session-6B-resolution` (see `docs/plans/peer-critic-log.csv`).

Two extra co-state estimators rode along in the same `train_heads` loop — same head class, same
inputs, same optimiser settings, steps and data as production; only the objective differs:

| scorer @k_cam=2 | margin_scale | ρ | regret (trimmed) | overlap (chance 0.125) |
|---|---|---:|---:|---:|
| `curvature_distilled` (composite) | — | +0.1151 | 0.003875 | 0.253 |
| `curvature_cosine_only` (WS1a) | — | +0.0877 | 0.003915 | 0.253 |
| `curvature_ranking` (WS1b) | 1.0 | +0.0967 | 0.004168 | 0.249 |
| `curvature_ranking` (WS1b) | 1000.0 | +0.1370 | 0.003981 | 0.297 |
| `patch_norm` | — | +0.0795 | 0.003916 | 0.236 |

WS1a: dropping the curvature-alignment half makes ranking slightly *worse* (ρ 0.115→0.088, regret
unchanged). The cosine target is therefore not the culprit relative to the composite — if
anything the second half was mildly helping.

WS1b: direct ranking supervision does not beat the composite at either margin scale, and stays
~170× away from exact-λ on regret. The 1000× scale moves overlap 0.249→0.297 but regret only
0.00417→0.00398, still worse than composite. So the "ranking loss fixes λ̂" hypothesis fails
robustly to demanded-separation scale, and the margin scale (1.0 vs 1000.0) is on the record via
`ranking_margin_scale` per sweep row.

Per the codex review this is an **exploratory** result, not evidence about input sufficiency:
capacity, optimisation, the score parameterisation and the singleton-from-empty labels all remain
open. What it does rule out is the two cheapest objective-level explanations. The remaining
candidate from §6 is input sufficiency (pooled `z` has discarded which patch was which) — now the
leading hypothesis by elimination of the objective, but untested directly. The factorial
(objective × input) design in the resolution plan is the next step, not a conclusion drawn here.

## 8. Addendum 2026-10-04 (later): WS0 norm controls — confirmed above chance, position untestable

From the committed bottleneck JSON (n=192 windows each): `patch_norm` top-k overlap vs chance is
0.236 vs 0.125 (z≈6.6), 0.452 vs 0.25 (z≈13), 0.765 vs 0.50 (z≈21) at k_cam=2/4/8. Above chance is
real, not noise — but with ρ=0.08 and regret tied to the learned scorer, it is a weak retrieval
signal with no decision value. Do not cite it as norm usefulness.

Position-bias is **untestable from committed data**: both parquets carry exactly
`[window, k_per_camera, k_total, policy, regret, episode_id, site, seed]` — no patch indices,
norms or positions. Minimal extra logging for the next run: `selected_patch_ids` + `patch_row/col`
per window-policy. Until a frozen-DINOv2 control arm is committed, the norm result stays
backbone-confounded and non-evidential either way.

## 9. Addendum 2026-10-04 (later): WS2 factorial — per-patch conditioning does not help

Artefact: `results/benchmarks/spatial_bottleneck_ws2/spatial_bottleneck_summary.json`
(192 stratified windows, 50 episodes, 12 sites, seed 0; backbone bit-identical, −2.1425).
Design: `docs/plans/2026-10-04-session-6b-resolution-plan.md` §4 (copilot-reviewed, revised).
Peer review of the design: copilot, recorded for milestone `Session-6B-WS2-design`.

New head `PatchConditionedCostateEstimator`: shared per-patch MLP over
`[latent; patch_emb_p; budget; horizon]` (frame-mean raw tokens — no future targets),
parameter-count matched to `CurvatureCostateEstimator` (within 10%, tested), scored through
the per-patch twin of `curvature_scores` (broadcast identity tested). Cell C mirrors A's loss
term-for-term (cosine-of-mean + Hessian-only alignment; Δz detached, which leaves head
gradients identical while skipping backprop into the frozen backbone). Cell D is ranking
supervision on per-patch scores (margin 1.0 and 1000.0).

| cell | objective | input | ρ | regret@2 | overlap (chance .125) |
|---|---|---:|---:|---:|
| A `curvature_distilled` | composite | pooled z | +0.1151 | 0.003875 | 0.253 |
| B `curvature_ranking` | ranking | pooled z | +0.0967 / +0.1370 | 0.004168 / 0.003981 | 0.249 / 0.297 |
| C `conditioned_composite` | composite | per-patch | +0.0414 | 0.004231 | 0.257 |
| D `conditioned_ranking` | ranking | per-patch | +0.0633 / −0.0335 | 0.004199 / 0.005212 | 0.257 / 0.202 |

Within-objective contrasts (the pre-registered input test):
- **C−A: Δρ −0.074, Δregret +0.00036 (worse).** Conditioning hurts under the composite objective.
- **D−B: Δρ −0.034 / −0.17, Δregret +0.00003 / +0.00123 (tied-to-worse).** Conditioning adds
  nothing under ranking supervision; at margin 1000 it degrades (ρ negative, worst regret
  outside ΔΣ-only).

Per the frozen falsifier, **the pooled-input framing is rejected**: giving the head the
candidate's own embedding does not recover ranking under either objective. The remaining
candidates are optimisation, score parameterisation, and the singleton-from-empty labels —
explicitly not input sufficiency, which is now tested and failed rather than merely suspected.
Caveats: seed-0 screen only (no positive, so no 3-seed confirmation was triggered); training
curves were not recorded, so "conditioning unhelpful" and "conditioned heads under-optimised"
are not separated — but both C and D trained the full 300 steps under which A/B learned
ρ≈0.1, so equal-budget comparison stands. The margin-1000 D degradation reads as optimisation
instability at high demanded separation, not as signal.

## 10. Norm controls and pair panel: first read with systems precision (v1 records; v2 LANDED — see below)

**Norm (same windows, realised J, all budgets).** Postencoder beats preview everywhere:
k=2: 0.003731 vs 0.004593; k=4: 0.003461 vs 0.004436; k=8: 0.002518 vs 0.003112
(trimmed regret; ~19–22% gaps). Timing on L4: preview 0.561 ms/frame, DINOv2 re-encode
9.299 ms/frame (17×), top-k 0.515 ms, masked rollout 1.560 ms. Systems reading, per the
review: postencoder selection cannot recover encoding work already performed, but it CAN
save downstream work whenever T_downstream(M) − T_downstream(k) exceeds selection +
packing — a downstream-compute reduction claim, never an encoder/camera-acquisition one.
The 0.56/9.30 figures are named-arm timings (cached-token scoring vs streamed-frame
processing), not comparable full-pipeline measurements. And mask-on-cached-tokens does not
establish early-pruning quality — that needs the actual pre-encoder execution path.

**Pairs (v1 batch-mean records; per-window rerun in flight).** 200-pair exact manifest,
800 batch-cell entries: signed mean ε 1.23e-04, mean|ε| 4.36e-04; signed mean pair gain
3.74e-05, mean|gain| 8.36e-04. The "12×" headline is mean|ε| over the *signed* mean gain —
a denominator that cancels toward zero; against mean|gain| the ratio is 0.52. Both
denominators go on the record: interactions are large next to net gains, about half of
typical absolute gains. Same-budget (old, budget-mismatched 4-vs-2 form — superseded):
−0.0092 trimmed. The rerun replaces it with measured 2-patch joints on both sides plus
episode-level paired uncertainty.

**v2 landed** (`pair_interactions_v2/`, 32 windows, per-window rows committed): ranking
fidelity 0.9887 — summed singletons order measured pairs near-perfectly within windows;
residual interaction fraction 0.52 (half the spread is pair-specific, rarely flips order).
Same-budget measured-only subset (n=6) ties at +0.00001 (Wilcoxon/full stats in the Exp-2
note §3; headline −0.00224 is fallback-driven). Net revision of this section: additive
*ordering* is excellent; absolute magnitudes are not; the best-pair-vs-top-set decision
is unmeasured, not negative.

**Residual headroom H_k (existing artefacts only; recomputation command:**
`python -c` over `pair_interactions_v2` rows for k=2 plus the committed bottleneck/6A
summaries for k=4/8/16 — full command in WORKLOG 6H):**

| total k | oracle-singleton regret | learned-singleton regret | singleton gap | residual joint headroom |
|---|---|---|---|---|
| 4 | 0.000034 (exact gains, bottleneck 192win) | 0.004753 (direct_ranking, 6A 3-seed mean) | 0.00472 | — (pairs measured at k=2 only) |
| 8 | 0.000055 | 0.004319 | 0.00426 | — |
| 16 | 0.000038 | 0.002996 | 0.00296 | — |
| 2 | additive ordering takes measured-best pair on **32/32** windows | n/a (6A budgets start at k=4) | — | **0.000000** |

Different window sets per artefact (192 vs 1200 vs 32; same backbone, checkpoint, split —
stated, not pooled). No pair-to-larger-budget extrapolation. Reading: the learnable gap
is ~100× (singleton estimation); the joint residual at k=2 is zero on measured windows.
Conditional modelling must therefore justify itself as *learnability*, not as fixing a
demonstrated joint-decision deficit at k=2.

## 11. Frozen-DINOv2 control executed: pixel-masked early pruning (2026-10-05)

Artefact: `results/benchmarks/early_pruning/early_pruning_summary.json` (32 windows,
32 episodes, L4; backbone −2.1425 again). Runner: `scripts/diagnose_early_pruning.py`.

Same windows, same backbone, same budgets — only the token source differs:

| k_cam | full-image tokens | preview-pruned pixels → tokens |
|---|---|---|
| 2 | 0.004259 | 0.005113 |
| 4 | 0.003957 | 0.004877 |
| 8 | 0.002808 | 0.003464 |

Pruning pixels pre-encoder costs ~20–23% regret at every budget: dropping image regions
removes information the encoder would otherwise contextualize, and the surviving tokens
do not recover it. Timing (L4, preregistered): preview 0.62 ms/frame, DINOv2 re-encode
9.32 ms/frame (15×), top-k 0.54 ms. This is a quality effect with full token counts, not
a FLOP saving — true token-dropping needs DINOv2 surgery, explicitly out of scope. The
norm story is now complete on all three legs: postencoder norm wins on quality, preview
norm is cheaper but worse, and pre-encoder pixel pruning degrades quality ~20%.

## 12. Reproduce

```bash
python scripts/train_spatial_patch_dropout_backbone.py --cache-dir /content/spatial_cache_e3_1 \
    --drive-root /content/local_runs --run-id spatial_patches_20261003
python scripts/diagnose_spatial_selection_bottleneck.py --cache-dir /content/spatial_cache_e3_1 \
    --drive-root /content/local_runs --run-id spatial_patches_20261003 --windows 192
```

Peak VRAM 1.7 GiB on an L4; no hardware upgrade is implicated. Regret is against a greedy rollout
reference, not a proven optimum, so every regret above is a lower bound.