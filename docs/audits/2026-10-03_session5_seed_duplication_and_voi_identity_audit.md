# Audit: Session 5 Seed Duplication, Belief-Space VOI Functional Identity, and Untestable Primary Endpoint

**Date:** 2026-10-03
**Auditor:** OpenCode (`space-bunny-free`)
**Peer Critic:** Copilot CLI (rotating decision critic — see `docs/plans/peer-critic-log.csv`)
**Artefact Under Audit:** `results/benchmarks/robustness_horizon/robustness_horizon_summary.json` and `scripts/benchmark_robustness_horizon_allocator.py`
**Companion Note:** [`2026-10-03_session_5_loso_and_cost_reconciliation_audit.md`](2026-10-03_session_5_loso_and_cost_reconciliation_audit.md)
**Subject Note:** [`../research-notes/2026-10-03-session-5-robustness-cost-sensitivity.md`](../research-notes/2026-10-03-session-5-robustness-cost-sensitivity.md)
**Status:** `PRIMARY_ENDPOINT_UNSUPPORTED` · `SEED_COUNT_INVALID` · `MECHANISM_CLAIM_CONTRADICTED`
**Reproduce:** `python scripts/audit_session5_evidence_integrity.py`

---

## 1. Scope and Method

The Session 5 reconciliation audit (`2026-10-03_session_5_loso_and_cost_reconciliation_audit.md`) checked that the Leave-One-Site-Out (LOSO) aggregation was arithmetically correct. It did not check whether the underlying runs were what the note says they were. This audit closes that gap.

Scope: every quantitative and mechanistic claim in the Session 5 research note, re-derived from the committed summary JSON. Nothing in the note was taken on trust. Per `AGENTS.md` research-integrity rule 2, every number below is produced by `scripts/audit_session5_evidence_integrity.py` from files already committed here, so the whole audit is re-runnable:

```bash
python scripts/audit_session5_evidence_integrity.py          # human-readable
python scripts/audit_session5_evidence_integrity.py --json   # machine-readable
```

**Scope limit of that script, stated plainly.** It is a focused *evidence-integrity* check, not a re-derivation of the note. It covers the seed fingerprint, per-seed endpoint signs, the scorer functional identity, the metric rank inversion, the availability of difference-level inference, and the `±` convention. It does **not** re-derive the LOSO table, the bounding identity, or the §2A/§2B/§2D means — those are reproduced by the companion script `scripts/reconcile_session5_evidence.py`, and both were run for this audit.

**The arithmetic in the note is correct.** All four of its tables — the 5-regime cost sweep, the $H \in \{2, 4\}$ horizon sweep, the 12-site LOSO panel, and the tail/spread panel — reproduce from `robustness_horizon_summary.json` to the last quoted digit, including the bounding identity $\sum_s w_{-s}\bar r_{-s} = \bar r$ and its $3.23 \times 10^{-9}$ residual. The reconciliation work was done properly. The findings below are about what the numbers can support, not about whether they were computed correctly.

---

## 2. Finding 1 — "3 random seeds" is two seeds, one of them evaluated twice

**Classification:** `SEED_COUNT_INVALID`

The note's §1 states *"4,154 unique held-out test windows evaluated across 3 random seeds, yielding 12,462 total evaluations."* The artefact records three seed entries (`seed: 0, 1, 2`), so the count is arithmetically consistent. The three entries are not three independent replicates, however.

Comparing every scalar in the artefact across seeds, split by kind:

| Comparison | Performance scalars (regret, gain, CI, win rate, counts) | Wall-clock scalars |
|---|---|---|
| seed 0 vs seed 2 | **2,359 / 2,359 identical** | 0 / 14 identical (all 14 differ) |
| seed 0 vs seed 1 | 288 / 2,359 identical | — |

Every performance scalar — regret, gain, bootstrap CI, win rate, per-site and LOSO values, window counts — is **bit-for-bit identical** between seeds 0 and 2. The 14 scalars that differ are, without exception, wall-clock timing fields (`mean_latency_ms_per_window`, `throughput_hz`), which vary between any two runs of identical work. Restricting to the eight policies' reported metrics (7 panels × 8 policies = 56 policy records, 12 scalar fields each = **672/672 identical**), the match is exact.

Two independent evaluations of a trained network on a fixed test set with seed-derived heads cannot agree to the last floating-point bit across 672 reported scalars unless they computed the same thing twice.

### Mechanism — established as the only silent path, not as proven provenance

`scripts/benchmark_robustness_horizon_allocator.py:561-565`:

```python
model_path = run_dir / f"seed_{seed}" / "best.pt"
if not model_path.exists():
    model_path = run_dir / "best.pt"
if not model_path.exists():
    model_path = run_dir / "seed_0" / "best.pt"   # <-- added in commit 2f5507d
```

This is the **only** code path by which seed 2 can silently evaluate another seed's teacher: `seed_2/best.pt` is tried first, then a shared `run_dir/best.pt`, then `seed_0/best.pt`. In all three branches `load_state_dict` overwrites whatever `torch.manual_seed(6000 + seed * 100)` drew at line 558, so the differing RNG seed does not affect the teacher; the allocator heads then retrain to the same solution on the same data, which is why the *performance* metrics also match bit-for-bit and not merely approximately. The fallback prints no warning when it succeeds — the `[warning] Checkpoint not found` string sits in the final `else`, unreachable once the seed-0 fallback succeeds — so the run log showed a clean three-seed sweep.

**Provenance limit, stated honestly:** the committed files do not record *which* checkpoint each seed loaded. Checkpoints live on Google Drive, not in this repository, and the artefact does not carry a resolved path or hash. So whether `seed_2/best.pt` was absent, or `run_dir/best.pt` happened to be a copy of seed 0's weights, cannot be settled from what is committed. Confirming the mechanism needs the run log or the `docs/DRIVE_INVENTORY.csv` hashes. **This limit does not weaken Finding 1:** whatever the route, seeds 0 and 2 are the same evaluation and cannot be counted as two replicates.

### Impact on the primary endpoint

With one genuine replicate, the sign of the headline result is not established. VOI advantage over the direct critic, per seed:

| Panel | seed 0 (= seed 2) | seed 1 | recorded-seed mean as reported |
|---|---:|---:|---:|
| cost `zero` | +37.00 % | **−1.12 %** | +27.21 % |
| cost `uniform` | +36.21 % | **−6.83 %** | +25.04 % |
| cost `default` | +36.14 % | **−38.29 %** | +16.98 % |
| cost `latency_weighted` | +36.34 % | **−12.68 %** | +23.72 % |
| cost `high_penalty` | +3.01 % | **−0.40 %** | +2.30 % |
| horizon $H=2$ | +59.69 % | +5.99 % | +56.05 % |
| horizon $H=4$ | +36.14 % | **−38.29 %** | +16.98 % |

**6 of 7 panels flip sign** between the two distinct teachers ($H=2$ is the exception, where both are positive). Every LOSO subset inherits this: the per-seed LOSO advantage is positive for seeds 0 and 2 and **negative for seed 1 on all 12 site deletions** (−7.76 % to −65.65 %). The note's claim that the advantage is "strictly positive across all 12 deletion subsets (+4.20 % to +19.78 %)" is true of the recorded-seed mean and false of the independent replicate.

Against the capacity-matched critic the picture is the same shape: seed 0 is −31.47 % under `high_penalty` where seed 1 is +1.12 %.

Because two of three entries are duplicates, the reported mean is a two-seed mean weighted 2:1 toward the duplicated seed, and every `±` in the note is computed over a multiset containing a repeat.

### A second, separate inference problem

Independently of the seed duplication, the uncertainty in the note is not clustered correctly. `compute_bootstrap_ci` (`benchmark_robustness_horizon_allocator.py:331-337`) resamples **individual windows**, but the windows are generated with a stride of 2 (context 8, horizon 4), so adjacent windows overlap in their inputs and targets. Resampling them as if independent understates the true sampling variance. The Session 6A specification already anticipates this by mandating a *site-clustered* bootstrap; Session 5 used neither. This is recorded as remediation item 3 and is not counted among the three headline findings.

---

## 3. Finding 2 — `belief_space_voi` and `second_order_curvature` are one functional family

**Classification:** `MECHANISM_CLAIM_CONTRADICTED`

The note presents belief-space VOI and second-order curvature as two competing allocators, and §1.5 attributes VOI's profile to a distinct mechanism. They are one functional family evaluated at two points. The repository's own Session 6A specification already states this (`docs/plans/2026-10-03-session-6a-spatial-selection-spec.md:38`): *"In Session 4/5, `delta_cov = effects.pow(2)`, reducing VOI and curvature to the same functional family up to a sign flip."* The Session 5 note never says so.

### Verification

`scripts/benchmark_robustness_horizon_allocator.py:318` sets `delta_cov = effects.pow(2)`. With `src/adjointrwm/allocators.py:181-211`, writing $Q = \sum_d H_d\,\mathrm{effects}_d^2$:

$$\text{second\_order\_curvature}: \quad s = \underbrace{-\langle \lambda, \mathrm{effects}\rangle}_{\text{first}} - 0.50\,Q - c$$
$$\text{belief\_space\_voi}: \quad s = \underbrace{-\langle \lambda, \mathrm{effects}\rangle}_{\text{first}} + 0.25\,Q - c \qquad (\beta = 0.5)$$

Identical terms. The quadratic changes sign and halves in weight. The script confirms this numerically: `belief_space_voi_scores(..., uncert_weight=-1.0)` reproduces `second_order_curvature_scores` exactly, and `uncert_weight=0.0` reproduces the first-order scorer **built from the same $\lambda$**. (That is a scorer-level identity, not a policy-level one: the runner's separate `first_order` policy draws $\hat\lambda$ from a different estimator, `heads["costate"]`, while both `second_order_curvature` and `belief_space_voi` use `heads["curvature"]`.) There is no epistemic term to switch off — `delta_cov` carries no prediction-variance semantics whatsoever.

### Two consequences

**The "VOI beats curvature" comparison is a hyperparameter statement.** The measured gap is small and consistent in direction (+5.62 % / +9.55 % / +5.62 % per seed), which is what one expects from two points in a one-parameter family rather than two different mechanisms. It cannot support a claim about value of information.

**The note's stated failure mechanism is wrong.** §2A Takeaway 3 explains the `high_penalty` result as: *"Because belief-space VOI incorporates an epistemic uncertainty bonus ($+\beta\,\mathrm{Tr}(\Delta\Sigma)$), it over-invests in expensive exploration when the gross benefit does not justify the penalty."* There is no uncertainty bonus in the scorer. The term is the **negated curvature** term; because its sign is flipped, it *adds* score to expensive joint refinement, which is the whole reason `high_penalty` inverts the ranking. The correct description is: the sign-flipped quadratic over-rewards high-curvature candidates, and a $5\times$ acquisition penalty on joint refinement flips that into a loss. That is a real and reportable sensitivity, but it is curvature-sign sensitivity, not a VOI phenomenon, and the note should say so.

### Terminology

Under `AGENTS.md` rule 4 and research plan §15.2, nothing may be called a co-state or adjoint unless it is supervised by or derived from $\partial J/\partial \text{state}$. Two precise statements, which are easy to conflate:

- `belief_space_voi` **does** contain a co-state term. Its score includes $-\langle\lambda, \mathrm{effects}\rangle$, and the runner trains that $\hat\lambda$ through `CurvatureCostateEstimator` against autograd derivatives of the declared objective. So it would be wrong to say the policy has "no co-state in it".
- What it does **not** contain is the epistemic/uncertainty term its name advertises. `delta_cov = effects.pow(2)` is the squared state perturbation, not a prediction-variance reduction.

The note avoids calling this policy a "co-state", which is correct. The audit records the distinction because the policy name implies a mechanism the formula does not deliver, and an earlier draft of this audit conflated the two.

---

## 4. Finding 3 — The headline metric and the count metric order the policies almost oppositely

**Classification:** `PRIMARY_ENDPOINT_UNSUPPORTED`

`win_rate_vs_critic` is already computed (`benchmark_robustness_horizon_allocator.py:407`) and appears in the note's own Table D. It is never reconciled with the headline. Ranking all eight policies on both (lower mean regret is better; higher win rate is better):

| Policy | mean regret | 10 % trimmed | std | win % | rank by mean | rank by win |
|---|---:|---:|---:|---:|:--:|:--:|
| `exact_costate` | 0.00813 | 0.00151 | 0.02521 | 66.4 % | 1 | 1 |
| `belief_space_voi` | 0.10899 | 0.07426 | 0.64384 | **20.6 %** | 2 | **6** |
| `second_order_curvature` | 0.11764 | 0.07381 | 0.85044 | 25.5 % | 3 | 5 |
| `direct_critic_curv_matched` | 0.12878 | 0.07557 | 1.66612 | 3.6 % | 4 | 7 |
| `direct_critic` | 0.13129 | 0.07603 | 1.66535 | 0.0 % | 5 | 8 |
| `first_order` | 0.20069 | 0.06162 | 3.24359 | 35.4 % | 6 | 4 |
| `normalized_first_order` | 0.31577 | 0.05940 | 10.34503 | 45.3 % | 7 | 3 |
| `always_mode0` (refusal) | **1.17650** | 0.05552 | 28.22032 | **63.2 %** | **8** | **2** |

Only **1 of 8** policies holds the same rank on both metrics. Two readings matter:

- Belief-space VOI beats the direct critic on **20.6 % of windows**. The critic beats VOI on 79.4 %.
- Do-nothing refusal beats the direct critic on **63.2 % of windows** while carrying 10.8× VOI's mean regret, and has the *best* 10 % trimmed mean of any learned policy.

So the entire reported advantage is a statement about mean regret being dominated by a rare tail (`always_mode0` std = 28.2 against VOI's 0.64). On every order-robust statistic the note reports — 10 % trimmed mean and median — VOI and curvature are within 0.6 % and 0.1 % of each other, and refusal is ahead of both.

§2D of the note gestures at this (*"its primary mechanism is preventing catastrophic, expensive misallocations"*), which is honest. But the Executive Summary and the Exit Gate present `+16.98 %` and `+8.29 % to +24.42 %` as robustness without stating that the same policies are beaten by the comparator on most windows, or that refusal wins on the count metric. A reader taking the exit gate at face value would conclude a result the data does not support.

### The endpoint is not testable from what is committed

- The summary JSON contains **no** difference-level CI or significance test. Searching for `diff_ci`, `paired`, `p_value`, `pvalue`, `bootstrap_diff`, `t_stat`, `wilcoxon`, `sign_test` returns nothing.
- The only uncertainty present is each policy's CI on its *own* mean regret. Against an effect size of **0.02230**, the `direct_critic` own-mean CI half-width is **0.04832** (2.2× the effect) and VOI's is **0.01864** (0.8× the effect). Per-policy CIs are the wrong instrument for a paired comparison and are far too wide to support the claim on their own.
- Only two files are committed for this benchmark: `robustness_horizon_report.md` and `robustness_horizon_summary.json`. **No per-window regret data.** The paired Wilcoxon signed-rank test that the Session 6A specification (§4.3) mandates for the *next* session cannot be applied retroactively, because the per-window regrets needed to run it were never committed.

### What is and is not recoverable

Two different things must be separated, because the note's phrasing conflated them.

**Recoverable.** The aggregate point differences are exactly reproducible from the committed means. Every percentage in this audit and in the corrected note can be recomputed by hand from the summary JSON, and `scripts/audit_session5_evidence_integrity.py` does so.

**Not recoverable.** What cannot be recovered is *paired uncertainty*: a CI, standard error or p-value for the **difference** between two policies on the same windows. Recovering it requires the per-window regret of each policy, which was not committed. So the correct statement is narrow: **the paired inferential comparison for Session 5 cannot be computed from what is committed**, and would require a re-run. It is not true that the aggregate point estimates are in doubt — those are reproducible.

Two further limits on any uncertainty statement here, both inherited from the run:

- The existing per-policy bootstrap resamples individual windows, but stride-2 windows with context 8 and horizon 4 **overlap**, so that CI understates sampling variance.
- The 4,154 windows are reused across both teachers. Two evaluations of the same windows are repeated measurements, not 8,308 independent observations, and the 12,462 / 8,308 counts should never be read as sample sizes.

This is the generalisable lesson: committing only aggregates makes the paired comparison permanently untestable, and thereby makes headline percentages unfalsifiable to anyone who was not present.

---

## 5. Finding 4 — Undeclared dispersion convention (minor)

Table B's `±` values are the **population** standard deviation (`ddof=0`) over the three seed entries, not the sample standard deviation. Verified against the note's `0.05679 ± 0.06643` for $H=2$ `exact_costate`: `ddof=0` gives 0.06643, `ddof=1` gives 0.08136. Population std understates cross-seed spread by 18 %.

The convention is undeclared. It is also moot in substance: with two of three seeds duplicated there is one independent replicate and no meaningful cross-seed dispersion to report under either convention. Recorded for completeness and because it will recur once the seed bug is fixed.

---

## 6. What survives this audit

Stated plainly, because most of Session 5 does survive and the note was honest about the hard parts:

1. **All arithmetic.** Every table reproduces exactly. The LOSO reconciliation, the bounding identity, and the trimmed-mean/variance-reductions corrections are correct and were done carefully.
2. **The cost-regime sweep as a sweep.** Five regimes, heads retrained and actions reselected per regime, with the `high_penalty` failure reported rather than dropped. Reporting a regime where the method loses is a mark of a real experiment.
3. **The `high_penalty` boundary itself.** VOI losing 22.73 % against the matched critic there is a genuine, correctly-computed negative result. Only its *stated mechanism* is wrong (Finding 2).
4. **The omission-sensitivity caveat.** The note already refuses to call the LOSO panel unseen-site generalisation, correctly. That discipline should be preserved.
5. **Throughput.** ~28.4 kHz measured, consistent with the Session 4 figure and above the 7.6 kHz gate.
6. **Session 6A's design.** The specification already mandates everything Findings 2 and 3 demand: site-clustered bootstrap, paired Wilcoxon, a true epistemic $\Delta\Sigma$ from the model's own `state_logvar_head`, an explicit $\beta \in \{0, 0.5, 1\}$ sweep with a $-\beta$ sign-flip control, and pre-registered tie *and* negative-result branches. The uncommitted `src/adjointrwm/spatial_selection.py` implements the corrected scorer — `curvature + 0.5\beta\sum_d \sigma^2_d\,(\Delta\Sigma_{p,d})^2`, with real variance semantics and $\beta = 0$ collapsing exactly onto `curvature_scores`, which `tests/test_spatial_selection.py` asserts. Finding 2 is a Session 4/5 defect that Session 6A has already fixed at the implementation level.

**Net effect:** the Session 5 *measurements* stand. The Session 5 *claims* — three seeds, a VOI mechanism distinct from curvature, and a PASS exit gate — do not.

---

## 7. Remediation

Ordered. Items 1–3 are prerequisites for any further Session 5 citation.

| # | Action | Owner | Blocking? |
|---|---|---|---|
| 1 | Replace the silent `seed_0` fallback in `benchmark_robustness_horizon_allocator.py:561-565` with a hard error; record the **resolved checkpoint path and its SHA-256** per seed in the summary JSON, so provenance is verifiable from the artefact alone | next implementer | **yes** |
| 2 | Re-run Session 5 with three genuinely distinct teacher checkpoints | next implementer | **yes** |
| 3 | Commit per-window per-policy regrets (`.parquet`); compute a paired CI and a **clustered** bootstrap on the *difference*, resampling episodes or sites rather than overlapping stride-2 windows | next implementer | **yes** |
| 4 | Apply the corrections in `docs/research-notes/2026-10-03-session-5-robustness-cost-sensitivity.md` | done in this audit's working tree | — |
| 5 | Add a regression test asserting that a benchmark summary with bit-identical **performance** metrics across seed entries is rejected by the artefact validator (compare keyed paths; treat wall-clock fields separately) | next implementer | no |
| 6 | Sweep $\beta$ explicitly (including $\beta = -1$, which is exactly `second_order_curvature`, and $\beta = 0$) so the functional identity is visible in the results rather than hidden in a default | next implementer | no |
| 7 | Declare the `±` convention in every results table, and relabel within-window `std_regret` as population SD — `reg.std()` is `ddof=0` | next implementer | no |
| 8 | Never quote the 12,462 / 8,308 evaluation counts as sample sizes; they are repeated measurements of 4,154 reused, overlapping windows | next implementer | no |

Until items 1–3 are done, Session 5 must not be cited as evidence for the co-state hypothesis, and no paper or README claim may rest on its Exit Gate.

---

## 7.1 Peer Critic Review

Rotating critic: **Copilot CLI** (`--plan`, read-only tool set, write and shell denied), per the Peer Critic Protocol in `AGENTS.md`. Verdict: **SOUND WITH CORRECTIONS**. The critic independently confirmed the $\beta = -1$ algebra and the 6-of-7 sign-flip count, and raised seven points. All seven were accepted and applied:

| # | Severity | Critic's point | Resolution |
|---|---|---|---|
| 1 | P1 | Identical metrics prove duplicated *outputs*, not which checkpoint was loaded; the runner tries `run_dir/best.pt` first, and the audit asserted the fallback as established fact | Accepted. §2 now separates "only silent code path" from "proven provenance", states that checkpoints live on Drive and the artefact records no path or hash, and adds hashing to remediation item 1. Finding 1 is unchanged: the seeds are the same evaluation either way. |
| 2 | P1 | The reproduction script labelled seeds "independent" whenever *any* scalar differed, so it returned the opposite verdict from the audit; it also dropped JSON paths and used `zip` without length checks | Accepted. `check_seed_duplication` now compares keyed paths, asserts matching schemas, classifies wall-clock fields separately, and bases the verdict on performance scalars only. It reports 2,359/2,359 identical performance scalars, 14/14 differing timing scalars, and prints `DUPLICATE`. |
| 3 | P1 | "8,308 independent window evaluations" is wrong: the same windows are reused and stride-2 windows overlap; "untestable" should be narrowed to the paired inferential comparison | Accepted. §4 now separates what is recoverable (aggregate point differences) from what is not (paired uncertainty), and the note no longer says "independent". The overlapping-window bootstrap problem, which the audit had missed, is recorded as a second inference limit and folded into remediation item 3. |
| 4 | P1 | "no co-state in it" is false — the score includes $-\langle\lambda,\mathrm{effects}\rangle$ with $\lambda$ trained against autograd derivatives; and $\beta=0$ gives the shared-$\lambda$ first-order scorer, not the runner's separate `first_order` policy | Accepted, and this was a real error. §3 now states both facts precisely, distinguishes the scorer-level identity from the policy-level one, and the earlier conflation is acknowledged in place. |
| 5 | P1 | "Its conclusions hold" was too broad about the prior LOSO audit, and `CHANGELOG.md` still advertised Session 5 as passing | Accepted. §8 now marks the prior audit as partly superseded (arithmetic survives, interpretation does not) and states the exact scope. A correction line was appended to `CHANGELOG.md`. |
| 6 | P2 | "5 cost regimes × 2 horizons × 8 policies" is not a cross-product; and "reproduce every number below" overstates the script | Accepted. Corrected to 7 panels × 8 policies = 56 records × 12 fields = 672. §1 now states the script's scope limit and points to `scripts/reconcile_session5_evidence.py` for the LOSO and mean tables. |
| 7 | P2 | The "Sample Std Dev" column is `reg.std()`, which is population SD | Accepted. Relabelled, and added to remediation item 7. |

Rejected: nothing. The critic found no defect in the three findings themselves.

---

## 8. Relation to Prior Audits

- `2026-10-03_session_5_loso_and_cost_reconciliation_audit.md` — arithmetic reconciliation. **Partly superseded, not wholesale.** Its *arithmetic* conclusions hold and its reconciliation script still reproduces the LOSO table and the bounding identity. Its *interpretive* conclusions do not survive: it accepted the three-seed framing, and it carried the same statistical-superiority and epistemic-over-exploration readings that this audit retracts. Cite it for the arithmetic only.
- `2026-09-30_b2_probe_seed0_dynamics_gate_audit.md` and `2026-09-30_dynamics_parity_audit.md` — these already failed runs for "one seed" and single-split evidence. This audit shows the same class of defect recurring in Session 5 under a different mechanism, which is the argument for remediation items 1 and 3.
- `2026-10-03_repo_code_gen_verdict_audit.md` — a hardcoded verdict string. Same family of problem (an artefact that does not say what it did), opposite direction: that one overclaimed in the summary while this one underclaimed in the log.

This audit adds a new file and rewrites nothing. Per `docs/audits/README.md`, audits are append-only; the research note is not an audit file, so correcting it in place is permitted and is what `AGENTS.md` rule 3 requires when a published claim turns out to be wrong.
