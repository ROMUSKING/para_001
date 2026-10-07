# Peer review: Session 6C conditional-allocator analysis fix (codex, 2026-10-05)

**Lead:** opencode · **Critic:** codex (gpt-6-luna, `codex exec --sandbox read-only`) ·
**Verdict:** CHANGES_REQUESTED · **Disposition:** all four findings accepted and implemented
before the re-run (see spec amendments 2026-10-05 and `scripts/train_conditional_allocator.py`).
Reviewer path: rotation returned agy (429 no credits), cline (free model retired), kilo
(timed out mid-read); codex reviewed. Recorded in `docs/plans/peer-critic-log.csv` and
`docs/plans/peer-review-register.yaml`.

## Findings

1. **The one-sided bound is the right form, but `theta_gate` does not fully implement the frozen rule.** For a higher-is-better per-window effect, testing whether the site-clustered bootstrap’s 5th percentile clears the 8% or −3% threshold is a conventional one-sided 95% rule. The keyword fix also avoids the earlier `confidence=0` bug. But the implementation keeps only windows with `single > 1e-12`, while its docstring says to exclude denominators with `|single| ≤ 1e-12`. More importantly, it says those windows use an absolute-margin fallback, but implements no such fallback. Since the “exact” reference is a hindsight top-k *singleton-gain* ordering, not a guaranteed joint optimum, `single_regret` can be zero or negative; the ratio may then be undefined or misleading. Define and implement the fallback and clarify the regret reference before treating the booleans as gate results. Also, per-budget bounds do not by themselves control error across the three tested budgets.

2. **The current code stores the needed rows; the existing evaluation artefact does not.** On a fresh evaluation, the summary stores per-window delta, both regrets, and site labels, which is enough to recompute θ for each budget. But the existing `conditional_eval_summary.json` has no per-window arrays or θ results; its clustered intervals have `confidence: 0` and identical endpoints. That artefact cannot be repaired into a gate result from its contents. A recomputed bound should also record the seed and resample count explicitly so its finite-sample result is reproducible.

3. **The context-blind marginal head is needed for a selected-set-information claim, but not for a practical pipeline comparison.** The conditional head trains on gains from varied selected sets; the singleton head trains only on empty-set gains. A win therefore mixes the effect of exposing selected-set information with differences in training rows and labels. A head trained on the same rows and labels while withholding selected-patch identities would better isolate that information’s contribution; retaining set size would let the comparison separate cardinality effects from identity effects. Without this control, describe any result as an end-to-end comparison of the two specified pipelines, not as evidence that selected-set identities caused the gain.

4. **Several protocol gaps could make a reported win hard to interpret.**
   - Training sets do not match the frozen deployment distribution: the schedule uses total sizes `{0,2,4}`, and runtime student sets use global top-`0/2/4`; evaluation selects per-camera budgets `{2,4,8}`. Fixed subsets also do not enforce per-camera quotas. This conflicts with the spec’s budget- and camera-based mixture.
   - The code logs scoring-call counts but does not enforce the cost gate or measure the promised upper-tail guardrail. A quality win alone cannot establish the frozen practical gate; the artefact itself notes many more conditional calls than singleton calls.
   - `_read_protected` logs and reseals each read but accepts another read afterward. That records repeat access; it does not enforce “read-once.”

## Verdict: CHANGES_REQUESTED

The bootstrap-call fix is sound, and the new source can retain rows for reanalysis. The present artefact does not support re-deriving the gate, and the ratio fallback, training-distribution mismatch, and missing cost/upper-tail checks need resolution before interpreting a conditional win under the frozen spec.

## Dispositions (lead)

1. Accepted: docstring fixed to the implemented keep-rule, absolute-Δ fallback reported on excluded windows, re-run diagnostic with k_cam=4 named as the confirmatory primary.
2. Accepted: re-ran; fresh artefact stores per-window Δ/regrets/sites plus seed/resample counts; superseded artefact must not be cited.
3. Accepted: this run framed as a practical pipeline comparison; context-blind control deferred to a follow-up (never run).
4. Accepted: quota-legal training cover (fixed per-camera U{0..8}, student prefixes of per-camera greedy top-8); cost gate report-only with tail guardrail (p10/p05, harmed fraction); `--acknowledge-reread` required for repeat reads.
