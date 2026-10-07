# Session 6C Exp-2 + pairs: follow-up review — narrow the conclusions

**Review date:** 2026-10-04 · **Reviewer:** ChatGPT (external, delivered via user message)
**Basis:** reported Exp-2/pair/norm results, notes and plan as reported by the lead.
**Experiments executed for this review:** NONE · **Repository modified by reviewer:** NO
**Overall:** prioritise joint-allocation labels, but narrow the Exp-2 conclusion. The
mismatch between singleton scoring and joint selection is the lead — not a demonstrated
absence of transferable privileged information.

## Required items (lead's mapping)

1. **D-row reconciliation:** D as implemented is a *separate* full-space fit, not C's
   weights — relabel it as such (training row 0.000/+1.00 belongs to that object), add
   fitted-weight hashes + train-prediction identity, specify the zero-fill stage, confirm
   near-zero output via prediction norms (rel error 1.0 alone also fits λ̂=2λ), NaN policy
   for constant-score vectors.
2. **Narrow the numerical conclusion** to: no held-out advantage over zero for projected
   linear predictors; does not distinguish absent signal from data/representation/
   calibration/model-class limits.
3. **Pair reanalysis from existing records:** per-window demeaned epsilon (ranking
   fidelity of summed singletons), decision consequence with full contracts for −0.0092
   (loss vs regret, best-of-200 restricted hindsight, singleton inclusion, per-camera
   constraints, episode-level paired uncertainty). Magnitude ratio alone ≠ uninformative
   singletons (common-shift counterexample).
4. **Conditional-gain experiment next** (after a mechanism check: exact second-order
   ε ≈ −δ_pᵀHδ_q vs measured joints; distill only if it succeeds). Small comparison:
   fixed/norm reference, singleton top-k, direct conditional, derivative-derived
   conditional if justified. More episodes, not more adjacent windows, if data must grow.
5. **Norm systems precision:** downstream-compute reduction vs encoder/camera savings;
   named arms with timing boundaries; mask-on-cached-tokens ≠ early pruning. Complete the
   pending write-up with these distinctions.
6. **Work package order:** reporting fixes → existing-artifact analysis → mechanism check
   → bounded learning experiment → systems assessment. Most of items 1–2 use existing
   records, no broad L4 campaign. Adopt the replacement session conclusion.

Direction unchanged: joint-allocation labels. Nothing reopens retired VOI.

---

## Full review text (verbatim as delivered)

# Review decision: retain the implementation work, but correct the experimental routing

**The reported engineering checks are green; Experiment 1's reconstruction gate is not. Proceeding directly to Experiment 2 as though that gate passed would contradict the frozen decision rule.**

The README corrections, recorded review, added controls, and explicit negative results address substantive parts of the previous review. I have not inspected the uncommitted changes or independently run the 542 tests, so those remain **reported verification**, not verification performed here.

The next step should be **one bounded numerical and calibration investigation**, followed by a recorded decision about Experiment 2—not another broad training sweep.

## 1. What the reported results establish

| Reported observation | Supported interpretation | Interpretation not yet supported |
|---|---|---|
| Composite training direction error ≈0.02 and ranking correlation ≈0.96 | Strong training agreement on those particular metrics | Accurate co-state reconstruction or an optimisation problem that has been resolved |
| Pure-MSE training vector error 0.22–0.42 | Some reconstruction was learned | Passing the declared $E_{\mathrm{rec}}\leq0.01$ gate |
| Every fit gate is `False` | The specified tiny-set fitting criterion was not met | An automatic transition through the "train fits, held-out fails" branch |
| Held-out correlations are approximately zero | The reported ranking agreement did not transfer to this diagnostic split | Information insufficiency identified as the cause |
| Finite-difference and batch-invariance checks remain open | Teacher-label correctness and scale consistency still need their specified checks | The label pipeline has completed its scientific preflight |

**There is a training-to-held-out deterioration. But its cause remains unresolved because training reconstruction itself failed and the derivative preflight is incomplete.**

## 2. High training rank does not substitute for the reconstruction gate

Suppose $\hat\lambda=25\lambda$: direction error 0, relative error 24. Rankings preserved
under positive scaling for first-order scores; magnitude error can still change decisions
once costs or curvature terms enter. "Optimisation moves" cannot replace the declared fit
criterion. Clarify what reported ρ measures: same scorer with learned, exact, zero, and
training-mean co-state, holding everything else fixed.

The "Smooth L1 instability" explanation remains a hypothesis: SmoothL1's derivative alone
does not explain magnitude growth. Required wording: the composite head exhibits a large
relative reconstruction error despite strong training directional agreement; cause
unresolved. Inspect absolute norms first.

## 3. Run three bounded diagnostics before another substantial experiment

**A. Derivative preflight** (individual/batch/permuted/chunks; 1/B convention; directional
finite differences with absolute + scaled discrepancies; repair-then-regenerate on failure;
post-hoc pass recorded with exact scope).
**B. Unregularised linear fit** (float64, rank/singular values/residual/rcond; full-row-rank
interpolation = finite-sample fitability only; must not retroactively pass production).
**C. Scalar calibration** (train-fitted a*; both recorded; constant predictor published;
held-out errors exceed the zero reference 1.0).

## 4. Check the new "random gains can't agree" test

Random subsets agree by chance (1/6 for M=4,k=2); stochastic tests must assert
distributions, not impossible universals. Held-out agreement 0.000 needs overlap + regret
context — zero exact matches are not independently decisive.

## 5. Experiment 2 can remain next—but not through an unpassed gate

Two paths: follow the gate, or amend routing deliberately with limits stated (never
described as the gate path). Manifests can be prepared now. Norm control + interaction
panel proceed after shared rollout/scoring checks pass.

## 6. Recommended next work package

1. Correct Experiment 1 interpretation; preserve the failed fit gate.
2. Complete finite differences, batch/chunk invariance, label alignment.
3. Run unregularised fitting and scalar-calibration controls.
4. Finish norm-control and interaction-panel implementation; freeze Experiment 2 manifests.
5. Record the routing decision.
Record base commit **and content hash** of the executed snapshot including untracked files.
Do not keep tuning against the 16 held-out windows while calling them untouched.

## Suggested replacement conclusion for the research note

> Experiment 1 completed, but no reported model passed the preregistered
> reconstruction-fit gate. The composite head achieved high training directional and
> score-ranking agreement while retaining a large relative reconstruction error; held-out
> ranking correlations were approximately zero on the diagnostic split. These observations
> do not yet isolate information insufficiency or rule out optimisation, scaling,
> representation, or label-pipeline problems. Finite-difference and batch-invariance
> preflight remain pending. The next action is a bounded numerical/calibration
> investigation and completion of the Experiment 2 input manifests; any departure from the
> existing progression rule will be recorded prospectively.

---

# Direction: prioritise joint-allocation labels, but narrow the Exp-2 conclusion
# (second message, verbatim as delivered)

The strongest new research lead is the mismatch between singleton scoring and joint mask
selection—not a demonstrated absence of transferable privileged information.

The reported preflight results close specific concerns about gradient scaling and numerical
agreement. The reviewed amendment also addresses the original action-information violation
and gross dimensionality mismatch. However, condition D's table needs reconciliation, and
the interpretation of both the privilege and pair experiments is stronger than the
supplied results support.

This is a review of your reported results. I have not inspected the new artifacts or
independently rerun them.

## 1. What should now be treated as closed—and what remains open

| Reported result | Supported conclusion |
|---|---|
| Batch invariance to 9×10⁻⁸; alone/batched ratio 1.0000001 | The tested label-generation path does not exhibit the suspected 1/B scaling error. |
| Finite-difference agreement at ε=0.01 | Numerical agreement was demonstrated for the tested perturbations and settings. Preserve those settings and tolerances in the artifact. |
| Logged future actions quarantined as privileged | The revised protocol no longer presents those outcomes as legitimate deployment context. |
| All conditions projected to d=12 | The nominal fitted input dimension is controlled. This does not, by itself, equalise retained information or numerical conditioning. |
| Poor held-out reconstruction in A and C | Neither tested predictor beats the zero-gradient reference on the reported reconstruction endpoint. |
| Large non-additivity and a reported best-pair advantage | Joint-mask label adequacy deserves direct investigation. The magnitude ratio alone does not establish that singleton rankings are uninformative. |

Do not repeat the completed preflight merely because the scientific result is negative.
Preserve its scope, resolve the reporting issues below, and move to the decision-level
question.

## 2. Exp-2 supports "no successful reconstruction," not "privilege adds nothing transferable"

Condition D's training row contradicts its stated construction: D defined as C's weights
with privileged inputs zero-filled at test time implies Ŷ_D,train = Ŷ_C,train, so training
metrics must match C's 0.168/+0.97, not 0.000/+1.00. Reconcile via fitted-weight hash,
projection/preprocessing hash, target identity, train-prediction identity; if D is strictly
test-time ablation, label its training row "same fit as C". Zero-filling does not establish
exclusive dependence: for λ̂_C = Ax + Bz + b, finding the ablated prediction ≈0 on held-out
examples does not prove A=0. Raw zero vs standardised zero differ (zero in transformed
coordinates = training mean); a joint projection can mix deployment and privileged inputs.
Specify where ablation occurs, retain fitted transforms, confirm near-zero output via
prediction norms (rel error 1.0 alone also fits λ̂=2λ), NaN policy for constant vectors.
Narrow the numerical conclusion: A and C worse than zero on held-out; D equals (not beats)
that reference; C lower held-out error than A (2.92 vs 4.12) with higher correlation
(+0.24 vs −0.26) — not a useful deployable method, but not a clean privilege-changes-nothing
either. Small sample, projection and estimator restrictions prevent that attribution.

Recommended conclusion: under the reviewed dimension-controlled linear diagnostic, neither
predictor achieved held-out reconstruction better than zero; no transferable-privilege
finding; close this diagnostic without claiming the broader hypothesis falsified.

## 3. The pair result is the most useful lead—but analyse ranking-relevant interactions

Mean|ε|/mean G ≈ 12 shows non-additivity but does not establish uninformative singleton
rankings (counterexample: ε_{i,p,q} = c_i constant within a window leaves pair ranking =
summed-singleton ranking). Reanalyse existing records: per-window demeaned epsilon,
ranking fidelity of summed singletons vs measured pair gains, decision consequence
(actual loss difference singleton-chosen vs best evaluated legal pair), residual variation
after removing the common shift. Inspect the denominator (signed near-zero means inflate
ratios); report signed/absolute/distribution separately; check pooling/mask/count
rescaling. The −0.0092 result is more decision-relevant than the 12× ratio — specify: loss
vs regret vs normalised quantity; best-of-200 restricted hindsight scope; singleton-pair
inclusion; identical per-camera constraints; episode-level paired distribution and
uncertainty. It establishes potential improvement only to those contracts' extent — not
that the better pair is predictable from deployment inputs.

## 4. The next experiment should test conditional gains

Bounded conditional-allocation experiment with target g(p|I,S) = E[L(S)−L(S∪{p})|I]:
offline mechanism check first (exact second-order ε ≈ −δ_pᵀHδ_q vs measured joints with
common expansion point, appropriate masks, accurate local approximation — distill only on
success), then direct conditional-gain vs derivative-derived conditional predictors with
matched information/supervision/cost. Keep hindsight references labelled. Small comparison:
fixed/norm reference, singleton top-k, direct conditional, derivative-derived conditional
if justified. More episodes, not adjacent windows, for data growth.

## 5. The norm result needs a more precise systems conclusion

Post-encoder selection cannot recover encoding work but CAN save downstream work when the
saving exceeds selection overhead (T_downstream(M) − T_downstream(k) > T_selector +
T_packing): a downstream-compute reduction claim, not encoder/camera savings. Attach arm
names and timing boundaries to 0.56 and 9.30 ms/frame (cached-token scoring vs
streamed-frame processing are not comparable full-pipeline measurements). Mask-on-cached-
tokens ≠ early pruning (needs that execution path evaluated). Complete the pending
norm/pair write-up with these distinctions, not just aggregate headlines.

## 6. Recommended next work package

1. Reporting (reconcile D, document zero-fill, publish omitted B result or non-execution).
2. Existing-artifact analysis (common vs pair-specific interactions; −0.0092 endpoint +
   uncertainty).
3. Mechanism check (derivative approximations vs same-budget mask losses).
4. Bounded learning experiment (direct conditional vs justified derivative-derived).
5. Systems assessment (quality improvement vs downstream savings vs encoder savings).
Most of 1–2 needs existing records only — no broad L4 campaign. Adopt the replacement
session conclusion. Proceed toward joint-allocation labels.

Bottom line: the project invalidated a misleading positive, established a better-controlled
negative, localised a potentially useful teacher–student gap — a more credible position
than the stale README suggested. Next question: can decision-relevant sensitivity be
predicted from deployment-available information, with suitable joint-allocation labels,
cheaply enough to beat a strong direct or feature-based selector?
