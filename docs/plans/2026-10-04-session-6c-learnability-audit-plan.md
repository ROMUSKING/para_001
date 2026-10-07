# Session 6C learnability audit plan

**Date:** 2026-10-04 (revised after peer review) · **Lead:** opencode · **Status:**
REVIEW REVISIONS RECORDED — Exp 1 executed same day (note
`2026-10-04-session-6c-experiment-1-fitting.md`, artefact
`results/benchmarks/costate_fitting/`). Peer review ChatGPT CHANGES_REQUESTED
(`docs/plans/2026-10-04-session-6c-peer-review.md`, register
`docs/plans/peer-review-register.yaml`); Exps 2–4 and the norm control still require their
frozen details before running. Preflight gaps outstanding: finite-difference label checks,
batch-invariance verification (injection control passed; covers indexing/normalisation only).
**Second review** (`2026-10-04-session-6c-peer-review-2.md`, register entry
`session_6c_exp1_routing`): Exp-1 gate failed everywhere → **no Exp-2 transition**; next is
the bounded package (finite differences, batch invariance, unregularised fit, scalar
calibration, manifests) plus a recorded routing decision. The Exp-1 sections below stand as
the frozen design that was executed; the routing correction lives in the Exp-1 note §2.
**Governs:** whether the exact-gradient advantage can become deployable, or whether a simpler
selector is the outcome. Supersedes the WS2-follow-up ordering with a bounded audit.
**Retired and staying retired:** the Session 6A VOI-vs-curvature comparison (spec §6,
DEV-20261004-03). Nothing here re-opens it; a new VOI claim still needs fresh pre-registration.

## 0. Teacher–student information contract (frozen; checked in code, not assumed)

The exact teacher is `cotangent_bundle` (autograd `dJ/dz`); the student is
`CurvatureCostateEstimator` (or ablations) on decision-time inputs.

| Information | Teacher (exact λ₀) | Student (distilled λ̂) |
|---|---|---|
| Future targets y (t+1…t+4 states + visuals) | **yes** — λ(s,y) differentiates the realised error | **no** |
| Genuinely available action plans (generated before selection) | case-by-case (see manifest) | case-by-case |
| Subsequently executed/logged actions | **privileged** unless availability at decision time is demonstrated — logged future actions must not become deployable by relabelling them "planned" | **no** |
| Goal | n/a (no goal conditioning in this task) | n/a |
| Horizon | yes, fixed H=4 | yes, scalar |
| Full-image / patch features | yes, via the model | pooled latent only (WS2 tested per-patch conditioning: failed) |
| Acquisition outcomes | n/a | n/a |

**Permitted information I_t (explicit manifest; per deployed input: tensor, source,
timestamp, computation stage, access cost):** context latent `z` (pooled adapter output over
context frames — state whether from previous frame, cheap preview, or current full encoder;
only work actually skipped counts as saved), budget/horizon scalars, candidate perturbations
Δz_m (one encode per patch — charged), per-patch raw tokens where a head consumes them. The
manifest covers the **whole scoring pipeline**, not just the head: pooling, candidate effects,
Hessian/curvature terms, mask construction, normalisation, previews, rescue triggers. A head
receiving only a pooled latent does not prove its complete selection process avoids
unavailable or already-expensive information.

**Hypothesis H-target (mathematical implication under stated assumptions, not evidence):**
under squared realised error, E[λ|I] = D_z f_HᵀW(f_H − E[y|I]); when the predictor equals the
conditional mean, the conditional-mean realised-error gradient is zero. A strong hindsight
reference can therefore coexist with a weak distillation target. The squared expectation is
the **squared-error** regression target only — do not assume composite, cosine, or ranking
losses share its population minimiser. The audit carries a pure squared-error diagnostic
alongside the production composite loss wherever H-target is invoked.

## 0.5. Preflight: label contract and invariants (frozen; must pass before Exp 1)

Freeze: the definition of z (coordinate ordering, normalisation), objective weights, horizon
reduction, state/visual reductions, action conditioning, stochasticity sources, and the
differentiation boundary (frozen/stop-gradient encoders per repo rule 5). Require the
per-example λ to agree within declared tolerances when evaluated **alone, in a batch, under
batch permutation, and under different chunk sizes** — if J_batch = mean_i J_i,
differentiation w.r.t. z_i introduces a 1/B factor, so the label generator must implement
the intended per-example gradient explicitly (per-example loss, declared averaging over
feature/horizon dims; a mean↔sum change without declared dims changes the objective).
Check selected directional finite differences of the actual production objective on real
diagnostic windows (plus directional-curvature checks where curvature scores). Record:
requested checkpoint, resolved artefact SHA-256, loaded teacher identity, objective hash,
normalisation hash, input manifest, label-generation config. One teacher with three student
initialisations is **not** three independent teachers. Missing checkpoint is a hard failure
(Session 5 lesson).

## 1. Experiment 1 — can the training problem be fitted at all? (fully frozen)

**Setup:** one frozen teacher checkpoint; declared H=4 and production objective; 16 fixed
train windows from 4 train-split episodes + 16 held-out windows from 4 *different*
train-split episodes (episode ids committed; held-out kept out of all tuning choices); no
augmentation; no intentional randomness beyond 3 init seeds. Frozen optimiser/learning
rate/regularisation/precision/max-updates/stopping rule (values in the runner's
`--help` and echoed into the summary).
**Students:** (a) ridge regression (linear diagnostic) **plus a training-mean constant
predictor**, train-only scaling, fixed regularisation rule — fitting 16 high-dimensional
windows proves nothing about generalisable linear signal; (b) production
`CurvatureCostateEstimator` + composite loss, hidden 256; (c) **pure reconstruction
control**: same small network, unregularised λ-reconstruction loss only (diagnostic control,
not a benchmark candidate); (d) **stored-label injection**: exact λ straight through scorer
and evaluator — must reproduce the exact-λ result, else indexing/normalisation/conversion is
broken.
**Fit gate (engineering sanity, not a research threshold; ratify before running):**
E_rec = sqrt(Σ‖λ̂−λ‖²/Σ‖λ‖²) ≤ 0.01 on the tiny train set for each init seed, with a
predeclared total-target-energy floor below which a separate absolute tolerance applies.
**Metrics, three layers reported separately:**
1. derivative reconstruction — relative *vector* reconstruction error ‖λ̂−λ‖/‖λ‖ (magnitude-only
   error |‖λ̂‖−‖λ‖|/‖λ‖ is a separate quantity with a declared denominator policy);
2. score fidelity — identical scorer with λ̂ vs λ, curvature term and candidate effects held
   fixed; Spearman + top-k overlap vs exact gains;
3. outcome quality — actual downstream losses of selected masks.
Undefined Spearman is **NA, never zero**; near-zero targets/constant scores/ties get declared
handling (directional error on negligible vectors flagged separately). Overlap/agreement at
**all deployed budgets** k_cam ∈ {2,4,8} with per-camera→total mapping declared (patches are
kept, not removed).
**Decision table:**

| Observation | Interpretation → next action |
|---|---|
| Pure reconstruction fails on train | Implementation/optimisation/labels. Stop broad sweeps. |
| Pure fits, composite does not | Composite-objective constraints or competing terms — loss-path repair. |
| Both fit, held-out fails | Generalisation/information/target-variability (→ Exp 2). |
| Gradient error large but decisions accurate | Full-gradient reconstruction is the wrong target (→ Exp 3). |
| Reconstruction improves, decisions do not | Score conversion, near-tie handling, joint labels (→ Exp 4). |

Tiny-set fitting rules out basic failures only — memorisation of noisy targets proves no
sufficiency.

## 2. Experiment 2 — teacher–student information gap (runs on Exp-1 pass; manifests frozen)

**Question:** is the useful exact-gradient information predictable from permitted inputs?
Exact per-condition manifests (arrays + preprocessing, frozen before running):
(i) deployment-available inputs (the actual problem); (ii) + only contexts established as
known at decision time, with equal access for direct and co-state models; (iii) explicitly
privileged context — unavailable realised future states/visuals (as **y itself or a declared
y-residual**, never λ or exact scores: those belong to the label-injection control, not to
an information experiment), and logged future actions where applicable. Match or explicitly
control architecture/parameter differences across conditions; record per-episode, per-init
performance; fit scaling only on training data. Same four Exp-1 metric families, paired
differences per condition. The 16 held-out windows stay diagnostic: "only the privileged
condition improved on this split" is the permitted language — a general "only privileged
inputs generalise" needs a separately specified, larger episode-disjoint assessment.

## 3. Experiment 3 — decision-relevant target (gated sketch; frozen after Exp-2 decision)

For g_m = −λᵀΔz_m, λ-components orthogonal to all Δz_m are irrelevant — for the first-order
term. The second-order term needs separate accounting: evaluate λ-only scores
diagnostically, then restore an **identical** curvature contribution across arms. Practical
target: d = Dᵀλ with D = [Δz_1…Δz_M], indexed by fixed candidate identities (stable across
windows — unlike per-window bases Q whose signs/rotations make raw Qᵀλ coefficients
incomparable; use candidate-direction products, a globally fixed basis, or explicitly
aligned coordinates). Declare whether D is decision-time available and charge for obtaining
it. **Terminology (hard):** projected targets must remain explicitly derived from
∂J/∂state; gains-only models are direct benefit predictors, never co-state estimators.
Matched deployment information; label and inference costs charged. A smaller target may win
without establishing any adjoint advantage.

## 4. Experiment 4 — singleton vs joint labels (runs after preflight passes; NOT gated on distillation success)

A singleton-to-set mismatch can make all label-fitting work uninformative, so this panel
runs once labels and rollout evaluation are validated. Frozen manifest: exact window subset;
exact pair list — 496 unordered pairs exist among 32 patches; the panel uses an exact,
predeclared count with within-camera, cross-camera and spatial-distance strata (no "~200").
Frozen semantics: S **retains** patches (gains measured from the null/empty selection, as in
`exact_marginal_gains`); derivative reference state declared; effect signs declared.
**Empty-mask legality verified** (masked rollout with all patches dropped must be a legal
model input); else anchor S0 with G_S0(S) = L(S0) − L(S0∪S). Report interactions relative
to relevant gains/score gaps (small ε can still flip rankings when score gaps are small),
and demonstrate practical consequence through **actual same-budget mask losses**, not the
surrogate. Hindsight greedy chains are diagnostic references, never deployable policies; a
learned conditional selector is compared against the same fixed/direct baselines with its
extra scoring calls charged.

## 5. Norm controls (run before any larger training; two named arms)

- `raw_gradient_energy_preview`: grayscale image-gradient energy on a 4×4 grid from streamed
  raw frames — no encoder — with declared acquisition/decode/resize cost.
- `dinov2_token_norm_postencoder`: post-encoder quality reference; declare the exact tensor
  (`x_norm_patchtokens` vs `x_prenorm` per the DINOv2 implementation), report score
  dispersion and ties. Never conflate the two arms.
Selection-stage freezing: discarding cached final tokens ≠ pruning before contextual
attention (survivors already contain omitted-patch context); state the stage and what it
saves. Same stratified windows; decision quality (regret via the frozen backbone) **and**
full selection-path cost (preview + features + selection + packing + remaining execution),
timed on L4 with preregistered warmup, batch size, timing blocks, synchronisation,
repetitions, transfers/packing, hardware/software identity; batch-one latency separate from
throughput. Cache-assisted readings allowed only with refresh cost declared.

## 6. Decision rules

Frozen trimmed-regret estimands R_A (candidate) and R_C (comparator); θ = 1 − R_A/R_C;
cost ratio κ = T_A/T_C. **Superiority** (practical): one-sided 95% lower bound for θ ≥ 0.08.
**Non-inferiority**: lower bound > −0.03. **Equivalence**: preregistered interval inside
[−0.03,+0.03] (procedure declared in advance; 95% two-sided is the conservative choice).
These are three different claims — a point estimate above 8% with a lower bound near zero
establishes nothing. Literal "no higher cost": one-sided 95% upper bound for κ ≤ 1, or a
preregistered tolerance — never post-hoc. Predeclare comparator rule, trimming fraction per
tail, window/episode/site weighting, cluster resampling, init handling, repeated-budget
handling, pairing (inits, timings, overlapping windows are distinct variation sources, never
pooled as independent). Near-zero comparator regret: predeclared absolute-margin rule or
relative endpoint undefined — no silent epsilon. Mean and upper-tail regret stay as
guardrails so trimming cannot hide expensive failures. Prediction vs planning claims stay
separate: task success is unmeasured.

| Finding | Direction |
|---|---|
| Tiny-set fitting fails | Repair implementation/optimisation. Stop broad sweeps. |
| Only privileged inputs generalise (protected assessment) | Expected benefit under available information; privileged teacher stays an upper reference. |
| Projected targets beat full reconstruction | Decision-relevant sensitivity prediction, direct-benefit comparator retained. |
| Singleton accurate, set selection fails | Conditional/interaction-aware allocation. |
| Cheap norm/direct matches learned co-state | Simpler selector absent a separately tested transfer/sample-efficiency edge. |
| Deployable co-state wins at measured total cost | Freeze; transfer across goals, horizons, sites before expanding domains. |

## 7. Prospective amendment: Exp-2 without a passed gate (peer-reviewed, kilo — revised before running)

Exp-1's gate failed (best E_rec 0.22 pure), so the frozen rule's Exp-2 path does not fire.
This amendment licenses a **bounded linear-probe Exp-2 only**, under these limits (review
§5 path 2 — described as this path, never as the gate path). Peer review accepted with
four corrections applied below (recorded milestone `Session-6B-Exp2-amendment`); the
original draft's "future actions are legitimate context" premise is **withdrawn** as
contradicted by §0's own table.

- Scope: closed-form linear probes (float64 gelsd, rcond 1e-12) on the same 16+16 windows;
  no iterative training, no hyperparameters, no selection on held-out. Directional evidence
  only: no gate passes, no superiority claims, no "only privileged generalises" beyond
  "on this diagnostic split".
- **Equal capacity (review attack 2):** raw column counts (~513 vs ~3600 on 16 rows)
  confound information with nullspace size — every condition is rank-deficient and C's
  nullspace contains the targets. All probe conditions go through one **seeded Gaussian
  random projection to d = 12** (projection seed recorded; Johnson–Lindenstrauss capacity
  equaliser, not feature learning — data-independent). Rank, residual and gate reported on
  the projected problems.
- **Falsifier, inverted (review attack 3):** a C held-out win is the **null expectation**
  (mechanical), not evidence. Informative outcomes: C ≈ A within the noise band (privilege
  adds nothing even mechanically → targets carry no linear signal at all), or the
  condition-D pattern below. "Directional evidence" never leaks into superiority language.
- Falsifier (original, retained): if the privileged condition does not beat deployment
  inputs on held-out by more than the seed/noise band, the information-gap framing gains
  nothing and the next look is optimisation/labels.

Manifests (exact arrays, frozen):
- Condition A (deployment): X = [latent z (512), 1], projected to d=12. The actual
  deployable problem.
- Condition B_ctx (legitimate context only): X = [z, budget (1), horizon (1), 1],
  projected to d=12. These scalars are the only decision-time context beyond z whose
  availability is established — the production head already consumes them, so this is a
  strictly-honest legitimate arm.
- Condition B_log (logged actions, **privileged — explicit §0 violation, diagnostic only**):
  X = [z, future_actions flattened (28), 1], projected to d=12. Logged executed actions
  are outcomes, not query context; this arm exists to measure how much realised-action
  information would help, not as a deployable candidate.
- Condition C (privileged targets, diagnostic only): X = [z, target_state flattened (56),
  target_visual flattened, 1], projected to d=12. Never λ or exact scores (those are the
  label-injection control).
- Condition D (true extrapolation): weights fit on **full, unprojected** C columns, then
  evaluated with privileged blocks **zero-filled** at test (deployment features only).
  D ≈ A-held-out means the C fit depended on privileged columns (gap confirmed
  directionally); D ≈ C-held-out would mean transfer with deployment features (gap denied).
- Targets Y: stored exact co-states in all conditions. Metrics: the Exp-1 four-metric set
  (reconstruction + fidelity + outcome at k_cam ∈ {2,4,8}) per condition per split, plus
  rank/singular values/residual per fit. Solver identical everywhere — the only variation
  is the input columns (then projected to equal width).

## 8. Definition of done

`pytest -q` green (explicit required-test list — unexpected skips never count as success);
`harness/check.py` 6/6; per-window parquet + requested-seed/teacher/init identities +
input/target contracts + cost boundary committed; every number traced to a committed file;
negative results plainly reported. Required artefacts for recomputation: raw labels, score
components, mask identities, realised downstream losses, fit traces. Four statuses tracked
separately: EXPERIMENT_EXECUTED / INTEGRITY_VALID / DIAGNOSTIC_RESULT / SCIENTIFIC_CLAIM.
Record what the sample does not establish: new-site generalisation, uncertainty floor,
task/planning performance, VOI-vs-curvature advantage. Stay on L4.
