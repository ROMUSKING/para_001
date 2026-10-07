# Session 6C conditional-gain experiment (draft for peer review)

**Date:** 2026-10-05 · **Lead:** opencode · **Status:** DRAFT — do not implement or run
before peer review is recorded.
**Question:** does selected-set conditioning improve a *learned singleton* selector at
acceptable cost — not whether anything beats oracle correlation.

## 1. Frozen targets (decision benefits, not gradients)

- Singleton: b_p(I) = E[L(S_0) − L(S_0∪{p}) | I], S_0 = null selection (legal, verified),
  I = deployment-available inputs at the declared decision point.
- Conditional: b_p(I,S) = E[L(S) − L(S∪{p}) | I,S] for already-selected S.
- Labels come from measured no-grad rollouts (realised futures); futures never enter
  deployed inputs, mask-selection history, or scoring computation.

## 2. Arms (small, fixed)

| Arm | Role | Implementation status |
|---|---|---|
| Strongest fixed/norm control | Establish whether learning pays at all | Exists: `early_feature_norm` numbers committed (6A); preview arm committed (norm_controls) |
| Learned singleton-benefit predictor | Test whether additive information is predictable | NEW `BenefitRegressionHead`: same input family as the conditional arm, **MSE** on measured singleton gains (peer review: `PatchRankingCritic`'s pairwise-ranking target discards calibration and differs materially from b_p — its committed regrets stay a legacy reference only, never the singleton arm) |
| Matched conditional-benefit predictor | Test selected-set conditioning vs learned singleton | NEW: same input family + set embedding (mean of selected patch embeddings), one MSE head on measured conditional gains |
| Oracle-singleton C | Privileged ordering reference only | Exists (bottleneck diagnostic) |
| Best evaluated joint mask | Restricted hindsight opportunity reference | Exists (pair v2 rows) |

No new ranking-loss variants (review: one benefit-regression baseline; ranking only if
reviewed). No co-state advantage claim attaches to a conditional win (governed
direct-critic comparison preserved separately).

## 3. Training-set discipline (deployment-consistent)

Training selected sets must resemble deployment: declared set-generation policy mixing
fixed outcome-independent masks (bounds coverage) with training-time student-generated
selections (on-policy contexts). Never hindsight-optimal greedy chains alone (favourable
contexts the model's own decisions won't reproduce). Count label-generation rollouts and
sequential scoring calls in the cost ledger. Frozen mixture (peer review): 50% fixed
masks (seeded random subsets across budgets k_cam ∈ {2,4,8} and both cameras, committed
manifest) + at most 50% student-generated selections (greedy rollouts of the current
singleton head, reseeded per epoch from the frozen schedule); the student fraction is
hard-capped — any batch exceeding it is rejected before the optimiser step, and the
realised fractions are reported. Mixture fixed by budget/site/episode strata; no tuning
of the ratio after seeing results.

**Amendment 2026-10-05 (codex review, accepted):** fixed masks name per-camera counts ∼
Uniform{0..8} and student sets are score-ordered prefixes (length ∼ Uniform{0..16}) of
the current singleton head's per-camera greedy top-8. The previous global totals {0,2,4}
neither satisfied per-camera quotas nor covered the evaluation range (totals 4/8/16).
Every training mask is quota-legal by construction; the cover spans all deployment
prefix sizes at each evaluation budget. Student prefixes use singleton (not conditional)
scores — declared. The causal attribution of any win to selected-set *identities*
remains open: without the context-blind control (same rows/labels, identities
withheld), a win is a practical pipeline comparison, not evidence that identities
caused it. That control is deferred to a follow-up, not this run.

## 4. Endpoint and gates (frozen before running)

Primary: Δ_k = E[L(S^learned-singleton_k) − L(S^conditional_k)] on actual downstream
loss, per budget, with the agreed aggregation (trimmed mean), practical threshold (8%
superiority / 3% non-inferiority margins on θ with one-sided 95% bounds), site-clustered
uncertainty, and mean/upper-tail guardrails. Rank correlation stays diagnostic. Cost gate
explicit: superiority at no higher total selection-path cost, or a prospectively defined
trade-off — never waived quietly. Near-zero denominators use the absolute-margin rule.
Claims ladder: predictive allocation quality (testable now) vs downstream savings
(measured path only) vs encoder savings (out of scope).

**Amendment 2026-10-05 (codex review, accepted):** (a) θ excludes every window with
R_single ≤ 1e-12 (near-zero denominator or non-positive hindsight-reference regret);
those windows are reported under an absolute trimmed-mean-Δ fallback, never a ratio.
(b) This re-run is diagnostic: all budgets reported without multiplicity control; the
confirmatory run gates on k_cam=4 only. (c) Cost gate is report-only here — greedy
conditional selection charges one head forward per pick and cannot pass strict
no-higher-cost by construction; no quality-cost trade-off is claimed. Upper-tail
guardrail (p10/p05 of Δ, harmed fraction) is reported per budget. (d) Repeat protected
reads require `--acknowledge-reread` with a recorded reason; silent re-reads refuse.

## 5. Protected evaluation (frozen before model selection; read-once protocol)

Fresh episode-disjoint windows (manifest committed, results sealed until all model
selection and stopping rules complete); candidate identities, budgets, comparator rule,
selection procedure and stopping rule frozen alongside. The 32 diagnostic windows stay
development-only. More episodes, not adjacent windows, if data must grow.
Read-once enforcement (peer review): the protected manifest (episode/window ids +
checksum) is written before training starts; a run log appends every read of protected
results with timestamp and purpose; any use of protected windows — including threshold
tuning, budget or stopping-rule selection, candidate pruning, or cost-tradeoff decisions —
invalidates the protection and must be declared as a deviation. No diagnostic-window
information (including the 32 mechanism windows or Exp-1 held-out sets) may influence
model selection, full stop.
