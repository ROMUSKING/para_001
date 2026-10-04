# Audit: Session 6A's "VOI ties curvature" result is degenerate, not scientific

**Date:** 2026-10-04
**Auditor:** OpenCode (`space-bunny-free`)
**Peer Critic:** Copilot CLI (rotating decision critic — see `docs/plans/peer-critic-log.csv`)
**Artefact under audit:** `results/benchmarks/spatial_selection/spatial_selection_summary.json`
and `results/benchmarks/spatial_selection/spatial_selection_per_window.parquet`
**Spec:** `docs/plans/2026-10-03-session-6a-spatial-selection-spec.md`
**Status:** `MECHANISM_UNTESTABLE_AT_MEASURED_SCALE` · `EVALUATION_NOT_EPISODE_DIVERSE` · `HEADLINE_NEGATIVE_STANDS`

---

## 1. Scope and headline

Session 6A ran on real data for the first time and reached a **pre-registered negative
branch**: `negative_result: geometric coverage dominates downstream sensitivity`. That headline
survives this audit and is discussed in §4.

The audit was triggered by one number in the report: `belief_space_voi` and
`second_order_curvature` have **identical** 10 % trimmed-mean regret to fifteen decimal places at
every budget, and the paired Wilcoxon between them returns `statistic 0.0, p = 1`. A perfect tie
between two policies the spec describes as *different functional families* (§2, item B2) is not a
scientific finding — it is the signature of one of them not being present in the computation.

This audit establishes **why**, measures it, and states what may and may not be claimed.

---

## 2. Finding 1 — the epistemic term is numerically inert (5 orders of magnitude)

### The observation

From the committed summary, the belief-space weight sweep (`beta_sensitivity`), all three
budgets, all five values including the sign-flip controls:

| budget | β=+0.0 | β=+0.5 | β=+1.0 | β=−0.5 | β=−1.0 |
|---|---|---|---|---|---|
| k=4 | 0.0071554184 | 0.0071554184 | 0.0071554184 | 0.0071554184 | 0.0071554184 |
| k=8 | 0.0065103307 | 0.0065103307 | 0.0065103307 | 0.0065103307 | 0.0065103307 |
| k=16 | 0.0046711432 | 0.0046711432 | 0.0046711432 | 0.0046711432 | 0.0046711432 |

Byte-identical. Not "close": identical.

### Cause, measured on the trained backbone

Reproduced on the run's own checkpoint
(`/content/local_runs/runs/spatial_patches_20261003/best_spatial.pt`, backbone SHA-256 prefix
`cfbca4406f13df55`), on held-out test windows:

| quantity | value |
|---|---|
| `delta_sigma` (predictive-variance reduction per patch) | absmean **7.68e-05**, absmax 1.29e-03 |
| `sigma_precision` (`E_t exp(-logvar)`) | absmean 2.20e+02 |
| curvature gains | absmean **1.46e-01**, absmax 1.87e-01 |
| epistemic gains (`β=0.5`) | absmean **1.33e-06**, absmax 3.49e-06 |
| **ratio, epistemic / curvature** | **9.14e-06** |
| spread across patches, curvature | 2.02e-02 |
| spread across patches, epistemic | 7.70e-07 |

The epistemic term is ~9e-06 of the curvature term. Even at the most extreme setting the sweep
offers, `β = ±1.0`, it is ~2e-05 of the quantity that actually orders the patches — far below
the 1e-16 resolution of float64, let alone the ~2e-02 inter-patch spread that determines a top-k
selection. **No β in the specified grid can change a single selection.**

### This is not a wiring bug

Checked and excluded, in this order:

1. `score_policy` passes `context.delta_sigma` and `context.sigma_precision` into
   `belief_space_voi_scores` (`src/adjointrwm/spatial_selection.py:1378`).
2. `selection_context` computes them via `epistemic_variance_reduction` (line 1307), which does
   the honest thing — one rollout per patch, reading `state_logvar_head`
   (`epistemic_variance_reduction`, line 692).
3. `belief_space_voi_scores` (line 655) forms `curvature + 0.5·β·Σ_d precision_d·ΔΣ_{p,d}²`, and
   the term is **squared**, which is where the magnitude goes.

The chain is correct. The problem is that `ΔΣ` is genuinely tiny: the trained variance head
barely moves when a single patch is added, and squaring a small number makes it smaller.

### What may be claimed

- **May:** at the trained operating point, adding one patch changes predicted state variance by
  ~1e-04, so a squared-variance VOI bonus is ~1e-06 while the curvature signal it is added to is
  ~1e-01. The specified functional form gives the epistemic term no leverage.
- **May:** the spec's B2 control (`β = 0` must reproduce curvature, negative β is a sign-flip
  control) is satisfied **trivially and therefore uninformatively**. It cannot distinguish "the
  epistemic term does not help" from "the epistemic term is not present".
- **May not:** that "belief-space VOI and second-order curvature are one functional family", or
  that "curvature captures the dominant spatial signal without epistemic covariance reduction".
  Both are statements about a comparison that this run cannot make.
- **Must not:** read `p = 1.0` as evidence of equivalence. It is evidence of no perturbation.

### Relation to the Session 5 audit

The Session 5 integrity audit found `delta_cov = effects.pow(2)`, which made VOI and curvature
literally the same expression up to a sign and a factor. This is a **different and milder**
defect: here `ΔΣ` is a genuine predictive-variance reduction from the model's own head, as B2
required, so the Session 5 mechanism criticism does not carry over. The defect is scale, not
identity. Both audits point the same way — Session 6A cannot adjudicate VOI against curvature —
but for different reasons, and conflating them would misstate both.

### What would make it testable

Not a larger β. Making β ~1e4 so the term could reorder patches would make it *dominate* the
curvature term, and the result would then be about β·ΔΣ² alone, not about a correction to
curvature. The honest options are:

1. Report ΔΣ on its own scale as a ranking signal (does predictive-variance reduction rank
   patches usefully *at all*, before being combined?), and
2. State the VOI term in units where it is commensurate with the curvature term, which requires
   a modelling decision about what ΔΣ should be relative to — not a sweep parameter.

That is a spec amendment, and it is **not** in scope for a run whose pre-registration is fixed.

---

## 3. Finding 2 — the evaluation covers 3 episodes from 3 sites, not the 14-site shard

The spec (and the Session 5 audit, remediation item 3) requires a **site-clustered** bootstrap
because stride-2 windows overlap and must not be resampled as if independent. The run computes
that statistic, but over the wrong number of clusters.

`--eval-windows 384` with `--batch-size 8` and `shuffle=False` takes the **first 48 batches** of
an ordered test loader. Window counts per episode, from the committed parquet:

| episode | site | windows |
|---|---|---|
| `150f9d8158d3de360dc66132` | BVL | 610 |
| `0d44548eb18346cc66c0e0d9` | ILIAD | 1600 |
| `0157ed6b7c5a7f1564726931` | IPRL | 1630 |

**3 episodes, 3 sites**, against a 50-episode / 14-site test split. The summary reports this
honestly — `"num_sites": 3` appears inside every `site_clustered_ci_95` — but a reader scanning
the report's tables would not see it.

Consequences:

- A cluster bootstrap over **3** clusters has essentially no coverage. The reported intervals are
  wide and unstable: `uniform_grid` at k=4 is `[0.0039, 0.0183]` on an estimate of 0.0108.
- Site-to-site variation, which is exactly what the shard's 14 laboratories were for and what
  Session 4 found decisive for `IRIS`, is **not sampled at all**.
- The three windows-per-episode counts (610/1600/1630) are not equal, so even within-episode
  stride overlap is unevenly weighted.

This is a runner limitation, not a scientific finding: taking a prefix of an ordered loader is
not a sample of episodes. It should be fixed by sampling windows **stratified by episode and
site** across the whole test split.

### What may be claimed

- **May:** the paired comparisons *within* the evaluated windows. Every policy saw the identical
  windows in the identical order, so the Wilcoxon tests and the paired regret differences are
  valid for these 3 episodes.
- **May not:** anything about generalisation across sites, laboratories, tasks or the 14-site
  shard. "Multi-site" would be false here — it is a 3-site result.

---

## 4. What survives: the headline negative result

The pre-registered negative branch stands, and it is the most important output of the run.

**Criterion 1** (≥8 % trimmed-mean-regret advantage over both `uniform_grid` and
`early_feature_norm`) is **not met at any budget**. Against `uniform_grid` the advantage is
+24.7 % (k=4), +5.6 % (k=8), +20.5 % (k=16) — inconsistent, and failing at k=8. Against
`early_feature_norm` it is **negative at every budget**: −8.9 % (k=4), −0.8 % (k=8), −4.2 %
(k=16).

So belief-space VOI **loses to taking the patches with the largest raw DINOv2 patch-token L2
norm**. Per-seed mean regret at k=4 makes the ordering unambiguous and shows the learned scorers
are seed-sensitive while the heuristic is not:

| seed | `belief_space_voi` | `early_feature_norm` | `uniform_grid` | `exact_costate_reference` |
|---|---|---|---|---|
| 0 | 0.00678 | 0.00713 | 0.01078 | 0.00013 |
| 1 | 0.00769 | 0.00713 | 0.01078 | 0.00013 |
| 2 | 0.00775 | 0.00713 | 0.01078 | 0.00013 |

The three seeds are genuine replicates: `duplicate_seed_check` reports 3 distinct head parameter
sets, and the backbone is a single shared checkpoint whose SHA-256 is recorded. This is the
Session 5 duplicate-seed failure mode explicitly checked and **passed** — the remediation worked.

**Criterion 2** (non-inferiority to `direct_critic_privileged` within 3 %) is **met at every
budget**. VOI beats the capacity-matched privileged critic, which is the comparator the spec
designated as decisive (B3). That is a real, if narrow, positive: a learned allocator using a
distilled co-state is not worse than one handed the co-state, curvature and `Δz` directly.

### The mechanistic explanation the run actually supports

The additive diagnostic is the strongest signal in the data and it is not about VOI at all:

| budget | additivity R² | submodularity violation rate |
|---|---|---|
| k=4 | 3.9e-05 | 11.5 % (133/1152) |
| k=8 | ~1e-04 | 22.5 % (606/2688) |
| k=16 | ~1e-04 | 39.8 % (2294/5760) |

`R² ≈ 0` means the additive surrogate `Σ_{p∈S} s_p` explains **none** of the variance in the
measured objective `J(z_S)` over 3,456 selections. Marginal value is essentially non-additive, and
the violation rate climbs with budget as the objective saturates. A per-patch scoring rule is
therefore structurally handicapped on this problem, and no amount of scorer sophistication
addresses that — which is exactly what beating `early_feature_norm` fails to do.

Against this, `exact_costate_reference` achieves trimmed-mean regret **0.00007** versus VOI's
0.00598 — roughly two orders of magnitude better. The useful signal exists and autograd finds
it; the ~99 % loss is entirely in the distillation to `λ̂`, not in the co-state idea. That is a
precise, actionable target, and it is consistent with the distillation gap being negative in
every budget (the distilled scorer is *worse* than the reference it approximates).

**Greedy-reference caveat, unchanged.** Regret is measured against a greedy rollout search, not a
proven optimum. The calibration at k_cam=2 reports greedy optimal on 50 % of 2 windows with mean
excess objective 1.10e-05 — a 2-window sample, far too small to bound the gap. Regret is
therefore a lower bound on true regret throughout.

---

## 5. Remediation

| # | Action | Blocking? |
|---|---|---|
| 1 | Do not cite Session 6A as evidence about value of information, or as a VOI-vs-curvature comparison. Cite it only for the negative allocation result and the additivity finding. | **yes** |
| 2 | Fix evaluation sampling: stratify evaluation windows by episode and site across the whole test split, so the site-clustered bootstrap has the number of clusters it assumes. Re-run before any generalisation claim. | **yes** |
| 3 | Decide explicitly whether the squared-ΔΣ VOI term is commensurable with the curvature term; if not, amend the spec and record the deviation. Do not fix it by enlarging β. | **yes, for any future VOI claim** |
| 4 | Report ΔΣ's standalone ranking quality (Spearman vs measured marginal gain) so the term is characterised even while it is too small to reorder a combined score. | no |
| 5 | Add a regression test that fails when the β sweep returns identical rows across β — that identity is a bug signature, not a result. | no |
| 6 | State the number of distinct evaluation episodes and sites in the report header, not only inside the CI objects. | no |

---

## 6. Peer critic review

Rotating critic: **Copilot CLI** (`--plan`, read-only tool set, write and shell denied), per the
Peer Critic Protocol in `AGENTS.md`. Review of the Session 6A prerequisite design; verdict
**partially adopted**. It raised five fixed-budget validity invariants, each verified in code
rather than accepted: a hard per-camera `topk` cap, an identical `[B, 32]` candidate pool across
all eight comparators, a true autograd `exact_costate_reference`, and capacity-matched critics.
Its cost-accounting point — that belief-space VOI costs one rollout per patch while the cotangent
scorers are O(1) — is real, already disclosed in the report, and deferred to Session 6C per spec
Q4. No P0 from that review changed the trainer design, and it did not engage this audit's
mechanistic findings, which were measured after it ran.