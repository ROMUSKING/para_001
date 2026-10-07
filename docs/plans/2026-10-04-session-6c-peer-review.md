# Session 6C learnability audit: peer review

**Review date:** 2026-10-04 · **Reviewer:** ChatGPT (external, delivered via user message)
**Basis:** the Session 6C draft (`docs/plans/2026-10-04-session-6c-learnability-audit-plan.md`);
prior user-supplied README for historical context only.
**Review status:** CHANGES_REQUESTED · **Execution approval:** NOT_GRANTED
**Experiments executed for this review:** NONE · **Repository modified by reviewer:** NO

> **Decision.** Retain the bounded learnability audit and the retired status of the Session 6A
> VOI-versus-curvature comparison. Do not broaden the architecture search. The draft needs
> specification changes before execution: complete the decision-time information contract, make
> the fitting gate executable, separate derivative fidelity from decision quality, and
> distinguish diagnostic progression from confirmatory superiority.

## Required changes R1–R10 (summary; full text below)

- **R1 (§0):** distinguish actual future actions from available plans; define permitted
  information I_t through an explicit input manifest (tensor, source, timestamp, stage, cost);
  extend the contract to the entire scoring pipeline.
- **R2 (§0):** add hypothesis H-target (realised-error co-state may carry little predictable
  conditional-mean signal); include a pure squared-error diagnostic; do not assume composite /
  cosine / ranking losses share its population optimum.
- **R3 (preflight before Exp 1):** freeze the label contract (z def, reductions, batch
  invariance incl. the 1/B factor, finite differences, provenance hashes).
- **R4 (§1):** replace undefined "fit" with a numerical protocol (E_rec ≤ 0.01 gate, energy
  threshold, absolute tolerance); add stored-label injection + pure reconstruction controls;
  fix the ridge interpretation (constant predictor, train-only scaling, held-out).
- **R5 (§§1–2):** report three layers separately (reconstruction / score fidelity / outcome);
  rename to relative vector reconstruction error; NA handling; evaluate at budgets {2,4,8};
  keep curvature identical across λ comparisons.
- **R6 (§2):** exact input manifests per condition; privileged = y or residual, never λ/scores;
  architecture matching; narrow diagnostic-split language.
- **R7 (§3):** keep first-order projection and curvature distinct; d = Dᵀλ with fixed
  candidate identities; basis stability; D availability/cost.
- **R8 (§4):** run the interaction panel after preflight, not only after distillation succeeds;
  freeze exact pair manifest + strata; anchor-S0 semantics; empty-mask legality; hindsight
  greedy labeling; same-budget loss consequence.
- **R9 (§5):** name the two norm controls separately
  (`raw_gradient_energy_preview` vs `dinov2_token_norm_postencoder`); DINOv2 tensor spec;
  selection-stage freezing; full cost boundary + L4 timing preregistration.
- **R10 (§§6–7):** diagnostic vs confirmatory statistics (θ/κ, superiority/non-inferiority/
  equivalence rules, comparator rule, pairing, epsilon policy, mean/upper-tail guardrails);
  revised execution order; strengthened definition of done
  (EXPERIMENT_EXECUTED / INTEGRITY_VALID / DIAGNOSTIC_RESULT / SCIENTIFIC_CLAIM).

Suggested review-register entry (not committed by the reviewer; recorded by the lead in
`docs/plans/peer-review-register.yaml`):

```yaml
review:
  date: '2026-10-04'
  milestone: session_6c
  reviewer: ChatGPT
  decision: CHANGES_REQUESTED
  execution_approved: false
  experiments_run_by_reviewer: false
  repository_code_verified_by_reviewer: false
  required_revisions: [R1, R2, R3, R4, R5, R6, R7, R8, R9, R10]
  retired_comparisons_reopened: []
```

Source and scope notes (reviewer's): experiment descriptions and historical references come
from the supplied draft and README; mathematical implications, thresholds, routing, and
required tests are the reviewer's recommendations, not reported outcomes. External checks used
official PyTorch MSELoss / autograd.grad docs and the official DINOv2 vision_transformer.py
(2026-10-04); served docs are not proof of installed dependency versions — pin and inspect
the actual environment.

---

## Full review text (verbatim as delivered)

### Updated direction (context supplied with the review)

Investigate whether the useful co-state signal is learnable — not whether the retired VOI
result can be rescued. Do not proceed as though a validated belief-space VOI advantage awaits
distillation. There is a negative deployable-selector result alongside a potentially useful
exact-gradient diagnostic. The next milestone is a bounded investigation of teacher
information, target representation, optimisation, and joint-allocation labels, to determine
whether the exact-gradient advantage can become a useful deployable method — or whether a
simpler direct or feature-based selector is the appropriate outcome.

Key consequences stated in the review:

- Session 5 is unsupported (bit-identical seeds; sign flips): withdraw the robustness
  interpretation; reproducing aggregation arithmetic does not restore independent replication.
- VOI and curvature were not distinct mechanisms (`delta_cov = effects²`; identical
  1200/1200 selections): keep the comparison retired; a new VOI claim needs genuinely
  distinct predictions and a fresh protocol.
- Session 6A shows no deployable-selector separation (VOI vs direct critic p = 0.94) with
  three now-verified distinct seeds: no demonstrated practical advantage to distil.
- Exact-λ (~2e-05) vs learned-λ (0.003875) with ρ 0.963 vs 0.115: investigate the
  exact-to-learned transition; do not assume the exact reference shares the student's
  information. Stop repeating generic capacity/ranking/input sweeps; test the assumptions
  beneath the training problem.
- A high p-value is not proof of equivalence; retiring this VOI implementation does not
  disprove value-of-information reasoning generally.

### 1. Section 0: complete the information contract — and sharpen the hypothesis

Separate planned actions from subsequently executed actions: a candidate action sequence
genuinely available at selection time is potentially legitimate deployment context; actions
subsequently executed and recorded are privileged unless prior availability is demonstrated.
Do not make logged future actions deployable by relabelling them "planned". Define the
permitted information I_t through an explicit input manifest: tensor, source, timestamp,
preprocessing, computation cost. Extend the contract to the entire scoring pipeline (pooled
latent, candidate effects Δz_m, curvature terms, mask construction, normalisation, previews,
rescue triggers). Say whether the pooled latent comes from the previous frame, a current
inexpensive preview, or the current full encoder — only work actually skipped counts as saved.

Add hypothesis H-target: under squared realised error
J(z,y) = ½(f_H(z,a)−y)ᵀW(f_H(z,a)−y), λ = D_z f_H(z,a)ᵀW(f_H(z,a)−y). When actions,
prediction, Jacobian and fixed W are determined by deployment information I,
E[λ|I] = D_z f_HᵀW(f_H − E[y|I]); when the predictor equals the conditional mean, the
conditional-mean realised-error gradient is zero. This is a mathematical implication under
stated assumptions — not evidence of calibration, irreducibility, or impossibility. It shows
why a strong hindsight reference can coexist with a weak distillation target. The squared
expectation is the squared-error regression target; do not assume composite, cosine, or
ranking losses share its population minimiser — include a pure squared-error diagnostic.

### 2. Section 1: make "can be fitted" an executable test

Freeze optimiser, learning rate, regularisation, precision, maximum updates,
stopping/checkpoint rule, and fit tolerances before execution; keep held-out episodes out of
those choices. Proposed engineering fit criterion (to ratify): E_rec =
sqrt(Σ‖λ̂−λ‖²/Σ‖λ‖²) ≤ 0.01 on the tiny training set for each of the three initialisations —
a numerical sanity criterion, not a research effect-size threshold. Declare a separate
absolute-error rule when total target energy is negligible. Add stored-label injection
(exact λ straight through scorer/evaluator — must reproduce the exact-λ result) and a pure
reconstruction control (same small network, unregularised λ-reconstruction loss only).
Interpretation table: pure reconstruction fails on train → implementation/optimisation/labels;
pure fits but composite does not → composite constraints; both fit but held-out fails →
information/generalisation. Rename ridge as linear diagnostic (constant predictor,
train-only scaling, fixed regularisation, held-out); record one teacher with three student
initialisations, not three independent replications.

### 3. Sections 1–2: separate reconstruction, score fidelity, and actual allocation quality

Report three layers: (1) derivative reconstruction λ̂ vs λ; (2) score fidelity — identical
scorer with λ̂ vs λ, all else fixed; (3) outcome quality — actual downstream losses of
selected masks. Rename ‖λ̂−λ‖/‖λ‖ as relative vector reconstruction error (magnitude-specific
error is a separate quantity with a declared denominator policy). Undefined metrics are NA,
not zero. Evaluate overlap/agreement at budgets {2,4,8} with per-camera mapping declared.
Keep the curvature term and candidate effects identical when changing only λ. Experiment 2's
"legitimate context" and "privileged information" must become exact array manifests;
privileged targets as y or residuals, never λ or exact scores; architecture matched; narrow
"only privileged generalises" language to the diagnostic split unless a larger protected
assessment is specified.

### 4. Section 3: the projection argument is correct, but only for the stated scoring terms

For g_m = −λᵀΔz_m, orthogonal components are irrelevant — but the second-order term needs
its own accounting: evaluate the first-order component separately, then restore an identical
curvature contribution across arms. Practical target d = Dᵀλ with D = [Δz_1…Δz_M], indexed by
fixed candidate identities (avoids per-window basis rotation/sign instability of Qᵀλ).
Declare whether D is decision-time available and charge for obtaining it. Terminology: gains-
only models are direct benefit predictors, never co-state estimators.

### 5. Section 4: run a small interaction audit earlier

Run the frozen interaction panel once labels and rollout evaluation are validated — do not
wait for distillation success. Freeze the exact window subset and pair set (496 unordered
pairs among 32 patches; replace "~200" with an exact manifest plus within/camera-cross/
distance strata). Freeze retain-vs-remove semantics, the derivative reference state, and
effect signs. Verify empty-mask legality, else anchor S0 with G_S0(S) = L(S0) − L(S0∪S).
Label hindsight greedy chains as diagnostic references, not deployable policies. Demonstrate
practical consequence through actual same-budget mask losses.

### 6. Section 5: retain both cheap controls, but distinguish identities and stages

Keep `raw_gradient_energy_preview` (image-gradient heuristic with declared
acquisition/decode/resize cost) separate from `dinov2_token_norm_postencoder` (post-encoder
quality reference with exact tensor/normalisation specified; DINOv2 exposes
`x_norm_patchtokens` and `x_prenorm` — declare which, report dispersion/ties). Discarding
cached final tokens ≠ pruning before contextual attention (survivors already contain omitted-
patch context); freeze the selection stage and what it saves, and charge full path cost to
learned and non-learned policies under one boundary. Preregister L4 warmup, batch size,
timing blocks, synchronisation, repetitions, primary cost metric, transfers/packing; batch-one
latency separate from throughput; cache-assisted claims only with refresh cost declared.

### 7. Sections 6–7: distinguish diagnostic progression from scientific superiority

Define θ = 1 − R_A/R_C on frozen trimmed-regret estimands: superiority = one-sided 95% lower
bound ≥ 0.08; non-inferiority = lower bound > −0.03; equivalence = interval inside
[−0.03,+0.03] (procedure declared in advance — a 95% two-sided interval is the conservative
choice). Cost κ = T_A/T_C with one-sided 95% upper bound ≤ 1 for literal "no higher cost" (or
a preregistered tolerance, never post-hoc). Predeclare comparator rule, trimming, weighting,
cluster resampling, init handling, repeated-budget handling, pairing; repeated timings ≠
independent trainings. No silent epsilon in near-zero denominators — predeclared absolute
margin or undefined endpoint. Keep mean and upper-tail regret as guardrails. Statuses:
EXPERIMENT_EXECUTED / INTEGRITY_VALID / DIAGNOSTIC_RESULT / SCIENTIFIC_CLAIM. Record what the
sample does not establish (new-site generalisation, uncertainty floor, task performance,
VOI-vs-curvature advantage).

### Revised execution order

1. Record this peer review and commit the revised specification and manifests.
2. Validate checkpoint identity, per-example gradient semantics, batch invariance,
   label-to-evaluator identity, and information boundaries.
3. Run Experiment 1 on the fixed tiny real dataset.
4. Run the cheap norm control and small actual-rollout interaction panel once the shared
   preflight passes (no distillation success required).
5. Run the frozen Experiment 2 when its fitting prerequisite is satisfied.
6. Freeze and run only the justified Experiment 3 or expanded Experiment 4 branch.
7. Confirmatory thresholds only on an untouched assessment, if warranted.

Branches must reflect the decision table (score fidelity can route directly to the
interaction audit; composite fit failure can justify loss-path repair — no forced linear
sequence). Definition of done adds: explicit required-test list; separated seed/teacher/init
identities; exact input/target contracts and cost boundary; immutable manifests and hashes;
raw labels/score components/mask identities/losses/fit traces; the four statuses; and a
record of what the sample does not establish.
