# Session 6B resolution plan (draft for peer review)

**Date:** 2026-10-04 · **Lead:** opencode · **Status:** APPROVED (codex review recorded 2026-10-04, milestone `Session-6B-resolution`) — WS0 and WS1 executed same day; WS2/WS3/WS4 open
**Supersedes nothing yet.** Governs how the five open items from the 2026-10-04 (final) WORKLOG entry are closed.

## 1. Findings being acted on (with sources)

- λ̂ is the sole bottleneck: `curvature_distilled` == `first_order_distilled` (ρ 0.1151, regret 3.875e-03, overlap 0.253); exact-λ variants reach ~2.2e-05. Source: `results/benchmarks/spatial_bottleneck/spatial_bottleneck_summary.json` production_point. Deployable ceiling ~2e-05, missed by ~170×.
- β sweep degenerate: identical regret to 8 decimals, 1200/1200 identical selections, Wilcoxon 1200 ties. ΔΣ alone ρ −0.183, worst regret. Never enlarge β.
- 50-episode stratified re-run revised all effect sizes down 5–20×; no deployable selector separates (VOI vs direct_ranking p=0.94). Source: `results/benchmarks/spatial_selection_stratified/`.
- `patch_norm` overlap 0.236 vs 0.125 chance (n=192) with ρ 0.08: weak retrieval, no decision value; backbone-confounded (dropout-trained, no frozen-DINOv2 control).
- Committed parquet has per-window regret only (no patch indices/norms/positions), so position-bias is currently untestable.

## 2. Workstreams (in order)

**WS0 — CPU-only norm controls first (no GPU, no new training).**
Compute from committed artefacts only: binomial overlap-vs-chance (done: z≈6.6/13/21 — real but small); document position-bias as untestable and specify minimal extra logging (`selected_patch_ids`, `patch_row/col`); decide DINOv2-vs-backbone with one frozen-DINOv2 rerun when GPU is next up. Output: short note + regression guard that identical-β sweeps are reported as degenerate, never as ties.

**WS1 — cosine-target ablation, then ranking comparison (revised per codex review 2026-10-04).**
The plan's first draft conflated "cosine is the wrong target" with "ranking supervision helps". Split it:
- **WS1a (isolates the suspect):** otherwise-identical head and training with ONLY the cosine term changed (keep curvature alignment, inputs, capacity, init policy, steps, data fixed). Same `curvature_scores(costate, hessian, delta_z)` path.
- **WS1b (separate objective-level comparison):** 4th head in `train_heads` (`scripts/benchmark_spatial_patch_selection.py:295`) optimised by existing `pairwise_margin_ranking_loss` (`src/adjointrwm/allocators.py:244`, margin scale must be specified — default 1.0 vs raw exact-gain differences) on `exact_marginal_gains` (`:275`), scored via `curvature_scores` on detached `delta_z`; evaluate top-k overlap AND regret on actual selected subsets in `scripts/diagnose_spatial_selection_bottleneck.py` (overlap alone is insufficient — rank labels are one-patch gains from the empty set and need not order multi-patch selections).
- WS1 is **exploratory**: a WS1 failure is NOT evidence for input insufficiency (capacity, optimisation, parameterisation, singleton labels could explain it). Touch list: `src/adjointrwm/spatial_selection.py` (+head class), benchmark script (+head+loss+return), diagnose script (+eval row), tests. Must NOT change: matched widths, split discipline (train only), existing 3 losses, label path.
- Primary endpoint (to be frozen before running): trimmed-mean regret at fixed budget vs current curvature head and direct critic, paired comparison, #seeds + episode/site aggregation + uncertainty procedure + inconclusive/tie/missing-score rules. The ~2e-05 exact-λ figure is a diagnostic reference, not a preregistered deployable threshold.

**WS2 — z-sufficiency, as factorial not fallback (revised per codex review).**
Cleanest design is a small factorial: current vs ranking objective × pooled-`z` vs candidate-conditioned input, matched capacity and identical splits. If resources force sequence: WS1 first as exploratory, then test input sufficiency before drawing any input conclusion. Falsifier needs a quantitative criterion (endpoint + tolerance, e.g. trimmed regret within X% of the exact-λ reference on the same windows/seeds) — "reaches the ceiling" is not yet specified and must be frozen before running.

**WS3 — spec amendment (before any further VOI run; revised per codex review).**
Options (A)/(B)/(C) are alternative scopes, NOT interchangeable ways to keep the original gate:
- (A) standardising changes score scale, effective β meaning, selections and gate outcome. Must define: per-window vs training-set spread, zero/tiny-spread handling, normalisation fitted on train/val only (never the confirmatory eval set), re-preregistered β decision rule on train/val with a pre-declared selection-flip check.
- (B) dropping VOI + removing its tie branch is a legitimate rescope but must be declared as a deviation; Criterion 1/2 thresholds for remaining selectors stay unchanged.
- (C) standalone ΔΣ characterisation cannot pass the combined VOI claim's gate. Record whichever is chosen in `prereg/deviation_log.yaml` with the amended gate text before any further VOI comparison runs.

**WS4 — provenance.**
Commit fitted normaliser hash + evaluation index list (episode/site/window ids + sampling code hash); report header states #episodes/#sites.

## 3. Hardware

Stay on L4. Peaks so far 1.7 GiB; no A100/H100/G4 justification. Colab rules: discover before provisioning, validate L4 + Drive, launch jobs first, `colab stop` immediately, 0 assignments at end.

## 4. WS2 factorial design (frozen 2026-10-04; copilot review recorded, revisions applied — cleared to run)

Question: is pooled `z` a sufficient input for per-patch utility, or has pooling discarded
which-patch-was-which? WS1 ruled out the two cheapest objective-level explanations (cosine-only
ρ 0.088; ranking ρ 0.097/0.137 at margin 1.0/1000.0; neither beats composite ρ 0.115), so the
input is now the leading hypothesis — still untested directly.

Cells (objective × input), all in one `train_heads` loop, same batches/steps/data, same scoring
path family. A/B already measured; C/D are new:

| cell | objective | input | head |
|---|---|---|---|
| A | composite (cosine + 0.5·curvature-align) | pooled `z` | `CurvatureCostateEstimator` (= current `curvature_distilled`) |
| B | ranking on own curvature scores | pooled `z` | existing `curvature_ranking` |
| C | composite | candidate-conditioned | NEW `PatchConditionedCostateEstimator` |
| D | ranking on own per-patch scores | candidate-conditioned | NEW, same class as C |

New head class (`src/adjointrwm/spatial_selection.py`, next to `PatchRankingCritic`):
per-patch shared MLP over `[latent; patch_emb_p; budget; horizon]` where `patch_emb_p` is the
frame-mean raw patch token (the same early features `PatchRankingCritic` sees — no future
targets, so Tier 1 status holds *as an information-boundary claim only*; the selector still
consumes every candidate's token, so any extraction-cost saving must account acquisition
separately) → `(λ̂_p, Ĥ_p)` per patch → new `patch_conditioned_curvature_scores`
(`-λ̂_p·Δz_p − ½Δz_pᵀĤ_pΔz_p`, all `[B,P,d]`).
Width set by a matched-width helper so *trainable params* ≈ `CurvatureCostateEstimator` (same
trick as `matched_patch_critic_hidden`, which stays untouched); exact counts reported, and a
test fails if the new head is not within 10% of the reference count. Reported strictly as
"parameter-count matched" — output DOF (`2d` per patch vs `2d` pooled) and function class
differ. Identity test: broadcast-λ̂ through the per-patch scorer must equal
`curvature_scores` exactly.

Cell C loss mirrors A term-for-term: `cosine(mean_p λ̂_p, exact_costate)` + `0.5·smoothL1` of
the *Hessian-only* piece vs gains (A regresses `(Ĥ·Δz²)` against gains, not the full score —
matching that form is what makes C-vs-A differ in input alone; the one deliberate difference
is detaching Δz, which leaves head gradients identical while skipping backprop into the frozen
backbone). C's per-patch costates stay underidentified (cosine constrains only the pooled mean
direction, P−1 DOF free), so C produces candidate-conditioned *scoring outputs*, not supervised
per-patch co-states — a C win needs follow-up attribution (λ̂_p vs Ĥ_p) before any mechanism
claim. Cell D loss: `pairwise_margin_ranking_loss` on per-patch scores vs gains,
margin_scale recorded and swept {1.0, 1000.0} for continuity with WS1b. Training branch gated
by `train_conditioned: bool = False` stark default-off (benchmark behavior byte-identical
unless requested); diagnose opts in via `--train-conditioned`.

Frozen endpoint: **regret is primary**, overlap is a mechanism diagnostic (they dissociate —
WS1b moved overlap 0.249→0.297 while regret barely moved). Primary: trimmed-mean regret at
k_cam ∈ {2,4,8} with a predeclared non-inferiority margin, paired per window, Wilcoxon +
site-clustered bootstrap, ≥192 stratified windows / 50ep / 12sites, seed-0 screen with 3-seed
confirmation pre-registered on a positive (staged design, stated as such — not an unqualified
confirmatory result). Overlap reported with its test, clustering, and multiplicity handling
across k declared up front. Contrasts: input effect **within** each objective (C−A and D−B)
plus the objective-by-input interaction — a diagonal D−A win alone is uninterpretable, and
A/B are retrained alongside C/D under the same seeds and data stream. Falsifier for the
framing: D ≈ B (conditioning adds nothing under ranking loss) → pooled input is not the cap;
look at optimisation/labels next. The ~2e-05 exact-λ figure is a diagnostic reference, never
a threshold. Inconclusive rules: ties/missing/non-finite scores reported, never dropped; a
C/D training failure is an optimisation result, not input evidence.

## 5. WS3 amendment (draft — peer review required before applying to the spec)

Status quo: §5 ties VOI-vs-curvature to a statistical tie, and B2 mandates the β grid. Both
are void because the control cannot move (audit Finding 1: term is 9.1e-06 of curvature;
1200/1200 identical selections; standalone ρ −0.183 with worst regret). Standardising (option
A) would give leverage to a term that points the wrong way — perverse, so A is rejected with
reasons. Standalone characterisation (option C) is already measured and fails its own bar.

Proposed amendment (option B), appended to the spec without rewriting §5 history:

> **Amendment 2026-10-04 (WS3):** the `belief_space_voi` policy and its β grid are retired.
> The VOI-vs-curvature comparison is declared *void* — the sweep was degenerate, so `p = 1`
> and the 1200/1200 selection identity are evidence of no perturbation, not of equivalence,
> and the tie branch must never be cited as "curvature captures the signal without VOI".
> Criteria 1/2 and the negative branch stand as fired for the deployable suite (no deployable
> selector separates, including random stratification). Any future VOI claim requires a fresh
> pre-registration with a commensurate term definition, not a re-weighting. Recorded as a
> deviation-log entry; the runner keeps the policy implemented for audit-trail purposes with
> the degeneracy guard (`beta_sweep_is_degenerate`) as its gate.

Deviation entry: new `DEV-20261004-03` citing the audit, the bottleneck note §§3/7, and this
review. No new pass/fail is declared by this amendment — that is the point: it removes an
untestable comparison rather than rewording one into a result.

## 6. Definition of done

`pytest -q` green, `harness/check.py` 6/6, per-window parquet + seed provenance committed, no number without a named source, negative results reported plainly.
