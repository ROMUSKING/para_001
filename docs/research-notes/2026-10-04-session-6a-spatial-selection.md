# Session 6A: fixed-budget spatial patch selection — a negative result, and a degenerate control

**Date:** 2026-10-04 · **Milestone:** Session 6A · **Hardware:** NVIDIA L4 (22.03 GiB)
**Spec:** [`../plans/2026-10-03-session-6a-spatial-selection-spec.md`](../plans/2026-10-03-session-6a-spatial-selection-spec.md)
**Artefacts:** `results/benchmarks/spatial_selection/` (summary JSON, report, 34,560-row per-window parquet)
**Audit:** [`../audits/2026-10-04_session_6a_voi_scale_and_sampling_audit.md`](../audits/2026-10-04_session_6a_voi_scale_and_sampling_audit.md)
**Deviation:** `prereg/deviation_log.yaml` DEV-20261004-02
**Run:** `spatial_patches_20261003` · backbone SHA-256 `cfbca4406f13df55…` · 3 seeds, all distinct

---

> **⚠ PARTIALLY SUPERSEDED (2026-10-04, later the same day).** The effect sizes in §3 are
> **badly inflated and must not be cited**. They came from a 3-episode evaluation set. A stratified
> re-run over **50 episodes / 12 sites** (`results/benchmarks/spatial_selection_stratified/`) gives
> VOI vs `uniform_grid` **+3.96 % / −2.21 % / +1.04 %** (was +24.7 / +5.6 / +20.5) and vs
> `early_feature_norm` **−5.2 % / −2.9 % / −3.2 %** (was −8.9 / −0.8 / −4.2) at k=4/8/16. Every
> deployable policy's site-clustered CI overlaps every other's, and VOI vs `direct_ranking_critic`
> gives p = 0.94 — no deployable selector separates from any other, including random
> stratification. §4's inference that non-additivity is the ceiling is **refuted**: ranking by the
> true marginal gains reaches trimmed regret 3.4e-05, so score-then-top-k *can* express an
> excellent policy. The real bottleneck is the λ̂ distillation (ρ 0.115 distilled vs 0.963
> autograd), and it is *not* capacity-limited.
> See [`2026-10-04-session-6a-bottleneck-diagnostic.md`](2026-10-04-session-6a-bottleneck-diagnostic.md).
> Everything below is retained as the record of what was measured at the time.

## 1. The question

At a fixed patch budget, does belief-space sensitivity pick more useful visual regions than a
competent direct gain predictor, curvature, and cheap token-selection heuristics?

Session 6A ran on real data for the first time. The previous artefact was `mode: synthetic`,
16 windows, random tensors.

## 2. Setup, and what had to be built first

Three prerequisites did not exist and were built for this run:

- **A patch-token feature cache.** The E3.1 cache stores only pooled 512-d ResNet18 embeddings, so
  it cannot supply the `P = 32` patch tokens selection needs. `scripts/build_spatial_patch_cache.py`
  re-extracts DINOv2 ViT-S/14 at a 4×4 grid for both cameras over all 500 episodes from
  `gs://gresearch/robotics`, carrying the E3.1 splits over verbatim (400 train / 50 val / 50 test,
  14 sites). It also records the **cached** episode length, because the E3.1 manifest leaves
  `length` null and `WindowDataset` then falls back to 100 steps against episodes up to 1,214.
- **The patch-dropout backbone (spec B6).** The runner refuses real mode without a
  `patch_dropout_trained` checkpoint and nothing in the repo wrote one. It is trained on **the
  objective the benchmark minimises**, through the same `apply_patch_mask` a deployed selector
  uses, with per-camera dropout over `k_cam ∈ {2,4,8,16}`. Selection is the lowest mean
  validation objective over the evaluated budgets against fixed masks; the no-dropout endpoint is
  tracked but not selected on. 1,500 steps, final selection score −2.1425.
- **The CLS-to-patch map** for the `early_cls_attention` comparator, plus `auxiliary_keys` on
  `WindowDataset` so a non-input per-frame array reaches the scorer sliced on the same `start` as
  `context_visual`.

## 3. Result: the pre-registered negative branch

`branch: negative_result: geometric coverage dominates downstream sensitivity`

Belief-space VOI's 10 % trimmed-mean regret, and its advantage over the two pre-registered
baselines (thresholds: ≥ 8 % over both):

| budget | VOI trimmed | vs `uniform_grid` | vs `early_feature_norm` | Criterion 1 |
|---|---|---|---|---|
| k=4 | 0.00716 | **+24.7 %** | **−8.9 %** | not met |
| k=8 | 0.00651 | **+5.6 %** | **−0.8 %** | not met |
| k=16 | 0.00467 | **+20.5 %** | **−4.2 %** | not met |

**Belief-space VOI loses to the patches with the largest raw DINOv2 patch-token L2 norm, at every
budget.** A learned allocator combining a distilled co-state, curvature and predictive-variance
reduction does not beat a one-line norm heuristic. Criterion 2 (non-inferiority to the
capacity-matched privileged critic, within 3 %) **is** met at every budget, so the distilled
co-state is at least not worse than being handed `λ̂, Ĥ, Δz` directly.

Per-seed mean regret at k=4 shows the learned scorers are seed-sensitive and the heuristic is not:

| seed | `belief_space_voi` | `early_feature_norm` | `uniform_grid` | `exact_costate_reference` |
|---|---|---|---|---|
| 0 | 0.00678 | 0.00713 | 0.01078 | 0.00013 |
| 1 | 0.00769 | 0.00713 | 0.01078 | 0.00013 |
| 2 | 0.00775 | 0.00713 | 0.01078 | 0.00013 |

These are **genuine replicates**: `duplicate_seed_check` reports 3 distinct head parameter sets,
and the shared backbone's SHA-256 is recorded. That is the Session 5 duplicate-seed failure mode
explicitly checked and passed.

## 4. Why: the objective is not additive, so per-patch scoring is the wrong shape

The strongest signal in the run is not about VOI:

| budget | additivity R² (3,456 selections) | submodularity violations |
|---|---|---|
| k=4 | 3.9e-05 | 11.5 % (133/1152) |
| k=8 | ~1e-04 | 22.5 % (606/2688) |
| k=16 | ~1e-04 | 39.8 % (2294/5760) |

`R² ≈ 0` means the additive surrogate `Σ_{p∈S} s_p` explains **none** of the variance in the
measured objective `J(z_S)`. Violations climb with budget as the objective saturates. A per-patch
scoring rule is structurally handicapped here, which is why scorer sophistication does not help.

Meanwhile `exact_costate_reference` reaches 0.00007 against VOI's 0.00598 — two orders of
magnitude better. **The signal exists and autograd finds it; essentially all of the loss is in
distilling `λ` to `λ̂`.** The distillation gap is negative in every budget, i.e. the distilled
scorer is *worse* than the reference it approximates. That is a specific, actionable target.

## 5. What this run cannot tell us

Recorded in full in the audit; stated here because it is easy to misread.

- **The VOI-vs-curvature comparison is void.** The β sweep returns byte-identical rows for
  β ∈ {0, ±0.5, ±1} at every budget, and the paired Wilcoxon gives `statistic 0.0, p = 1`.
  Measured on the trained backbone, the epistemic term is **9.1e-06** of the curvature term
  (1.33e-06 vs 1.46e-01), with inter-patch spread 7.7e-07 against 2.0e-02. No β in the grid can
  reorder a single selection. This is **scale, not identity** — `ΔΣ` is a genuine
  predictive-variance reduction from the model's own head, unlike Session 5's `effects²`. The
  tie is a degenerate control, not a finding, and `p = 1` is not evidence of equivalence.
- **Not episode-diverse.** `--eval-windows` took a prefix of an ordered loader: **3 test episodes
  from 3 sites** (BVL, ILIAD, IPRL), not 50 episodes across 14 sites. The site-clustered bootstrap
  therefore has 3 clusters and near-worthless coverage. Paired comparisons *within* those windows
  are valid; nothing here supports a claim about sites, laboratories or the 14-site shard.
- **Regret is against a greedy reference, not an optimum.** The k_cam=2 calibration reports greedy
  optimal on 50 % of **2** windows, mean excess 1.10e-05 — too small a sample to bound the gap.
  Regret is a lower bound throughout.
- **Absolute objective values are not comparable with Sessions 3–5** (DEV-20261004-02): Drive
  authorisation is human-only, so the state/action normaliser was refitted on the train split.

## 6. Hardware

Stayed on the L4. The run peaked at **1.7 GiB**; a single forward pass peaks at 0.09 GiB. An
earlier OOM in this same session requested 42.19 GiB and was a batching defect, not a capacity
limit — see the audit trail in `CHANGELOG.md` (2026-10-04 (bm)). G4-1 (profiler wait 21 % vs 15 %
limit) and G4-3 (peak 1,109 MiB = 5 % capacity) both still fail, so the conjunctive switchover
gate remains closed and DEV-20261003-01 stands.

## 7. Next

1. Fix evaluation sampling to stratify by episode and site across the whole test split, then
   re-run before any generalisation claim. **(blocking)**
2. Decide whether the squared-`ΔΣ` term is commensurable with the curvature term; amend the spec
   if not. Do not fix it by enlarging β — that makes the term dominate rather than correct.
   **(blocking for any future VOI claim)**
3. Attack the distillation gap: `exact_costate_reference` is 0.00007 and `λ̂` is 0.00598. The
   co-state signal is real; the estimator is the bottleneck.
4. Report ΔΣ's standalone ranking quality, so the term is characterised even while too small to
   reorder a combined score.