# From Oracle Opportunity to Deployable Allocation: A Gated Empirical Study of Learned Resource Selection

> Draft manuscript (B track). Every sentence with an empirical claim cites a
> register claim id from `docs/plans/evidence-register.yaml`; sentences without
> one are definitions, methods, or explicitly marked hypotheses. Status: sections
> 1–2 drafted; 3–6 pending.

## 1. Research question and evidence rules

We study whether sensitivity of a declared objective to model state (a co-state,
∂J/∂state) helps a world model allocate computation, sensing, and context better
than a direct marginal-gain critic matched on information, supervision,
parameters, coded rate, and measured compute. A method counts as a co-state
method only if it is supervised by or derived from ∂J/∂state; saliency or
attention scores do not qualify.

Evidence tiers used throughout: proof-check, microbenchmark, exploratory
diagnostic, protected-validation result, confirmatory test. Each claim carries
integrity (verified/partial/unavailable/withdrawn), verdict
(positive/negative/inconclusive/non-diagnostic/void), and lifecycle
(active/completed/closed/withdrawn/superseded) states, recorded in the evidence
register. A withdrawn prerequisite invalidates dependent active claims; missing
artefacts render claims unavailable, never silently dropped.

## 2. Experimental map

| Branch | Intervention and domain | Reason opened | Reason closed or current state |
|---|---|---|---|
| B2.1/B2.2 adaptive sensing, DROID-100 | Amortised co-state vs matched direct critic for camera gating | Real-data H2 test with diagnostic headroom | Mixed: co-state advantage at benchmark strength with an under-realised-critic caveat [`B2.1-adaptive-sensing`]; gap closed at 2500 steps [`S6C-lambda-bottleneck` context] |
| S6A spatial patch selection | Per-patch co-state/curvature/VOI scoring vs norm/random | Fixed-budget allocation on DINOv2 tokens | Negative: no deployable selector separates; VOI comparison void [`S6A-spatial-negative`] |
| 6C learnability (WS0–WS2, Hessian, conditional) | Distillation variants, cross-Hessian correction, conditional vs singleton | Isolate why exact gradients (regret ~2e-05) beat distilled ones (~170× worse) | Closed: tested variants fail; Hessian premise fails; conditional θ gate fails [`S6C-lambda-bottleneck`, `S6C-hessian-closed`, `S6C-conditional-closed`] |
| B2 critic readiness probes | Convergence/capacity/floor probes of the direct critic | Settle whether B2's critic was realised | `UNDER_REALIZED_DUE_TO_COMPUTE_CONSTRAINTS` (protocol bin); rescue stopped [`B2-critic-readiness`] |
| D2 QA context (Rung-0, bridge, R2, Rung-1, confirmatory) | Paragraph selection: fixed, lexical, learned rankers; frozen-reader answers | Licence-cleared corpus with exact support labels; retrieval opportunity G1 0.93 | Closed: confirmatory GATE FAILs (only recall-k4 passes; EM negative both budgets); frozen formulation closed, H2 unchanged |
| D5 cluster admission | Job selection under concurrency budget, Borg 2011 | Licence-cleared traces with exact hindsight | Closed: G1 0.006, oracle≈fixed; pools dwarf budgets |
| D4 simulation | Learned co-state vs direct critic, priced compute | Exact answers known, CPU-cheap | Direct-sufficient at real prices; teacher-only at lookup price [`D4-1-critics`] |
| Programme H2 | Synthesis | Gate decision required by roadmap | INCONCLUSIVE, teacher-only hypothesis [`G-H2-programme`] |

## 3. Findings by failure mechanism

Premise failure: the tested joint-latent additivity premise fails (pair-additivity
residual median 1.235, 0% of pairs below 10%), so the cross-Hessian correction is
negligible (predicted |ε| 9.8e-08 vs measured 1.6e-03) and corrected/uncorrected
arms are identical [`S6C-hessian-closed`]. The belief-space VOI term is
incommensurate with the curvature term (degenerate β sweep, identical selections
on 1200/1200 windows) and retired [`S6A-spatial-negative`].

Optimisation/readiness failure: the B2 direct critic collapses to full
observation (B2.1) with regret near the worst static policy across checkpoints
(identical to it only to rounding at step 2,500) (B2.2); floor and convergence
probes fail; the arm is
`UNDER_REALIZED_DUE_TO_COMPUTE_CONSTRAINTS`, a protocol bin, not a causal claim
[`B2-critic-readiness`]. No H2 inference follows in either direction.

Insufficient opportunity: D5 admission shows oracle≈fixed at all budgets
(G1 = 0.0058) with informative value granularity (priority entropy 2.16 bits) —
knowing outcomes buys nothing at the tested budgets [`D5-cluster admission row
pending register entry`].

Failed practical gates: conditional selection misses the θ-superiority bar at
all budgets (one-sided lower bounds −0.158/−0.014/−0.274 vs +8%) at 28× scoring
cost [`S6C-conditional-closed`]; the D2 reader bridge fails its sanity leg
(BM25 loses to longest-first, p=0.0047 at k=4) [`D2QA-bridge-v2`]; the D2 Rung-1
and confirmatory gates fail on answer legs (§5).

Unresolved (not negative): H2 as a whole [`G-H2-programme`]; the exact-gradient
diagnostics (exact-λ regret ~2e-05 vs distilled ~170× worse), which show value
exists without showing it is deployable [`S6C-lambda-bottleneck`].

Withdrawn/void comparisons (methodology illustrations only, never evidence):
Session 5's primary endpoint (duplicate seeds) and VOI-vs-curvature tie
[`S5-robustness`]; the 74-worker conditional attempt (degenerate intervals, lost
artefacts).

## 4. Oracle versus deployable performance

Each privileged reference and what deployment cannot see: exact-λ (autograd
access to realised objectives); greedy joint masks (measured outcomes);
supports-first paragraph sets (gold support labels); hindsight top-k singleton
orderings (realised gains); per-instance exact co-state tables (D4 lookup
price, 2.1–50× the arm's own compute). A gap between these references and a
deployable policy measures the price of hindsight. As an interpretation (not a
result): in the tested variants the gap survived distillation, capacity,
ranking supervision, and conditioning — consistent with, but not proof of, the
absence of a learnable deficit.

## 5. D2 case study

The frozen D2 QA-context formulation failed the preregistered confirmatory gate
in both the primary and sensitivity analyses. Recall at k=4 was the only passing
endpoint. Exact-match effects were negative at both evaluated budgets, and the
exploratory one-percentage-point answer improvement did not replicate. Under the
predeclared decision rules, the frozen formulation was closed. The passing
recall endpoint does not support a partial answer-quality or deployment claim.

| leg (Holm-6, one-sided lower edges) | primary, n=1417, 439 clusters | sensitivity, n=282, 200 clusters |
|---|---|---|
| recall learned−BM25 k=4 | +0.0578, low +0.0351, p=2.4e-14, PASS | +0.0455, low +0.0095, p=0.0040, FAIL |
| recall learned−BM25 k=8 | +0.0389, low +0.0177, p=2.6e-06, FAIL | +0.0077, low −0.0287, FAIL |
| recall learned−longest k=4 | +0.2695, low +0.2258, PASS | +0.3032, low +0.2472, PASS |
| recall learned−longest k=8 | +0.2086, low +0.1672, PASS | +0.2145, low +0.1520, PASS |
| EM learned−BM25 k=4 | −0.0049, low −0.0241, FAIL | −0.0035, low −0.0399, FAIL |
| EM learned−BM25 k=8 | −0.0183, low −0.0379, FAIL | −0.0177, low −0.0479, FAIL |

Sequence without reinterpretation: exploratory recall wins with EM +1.3pp/+1.0pp
(bridge/Rung-1) → frozen confirmatory protocol (1,417 unused ids, clustered
inference, blinded operations) → EM negative both budgets → adjudicated FAIL →
closure. A rejected code proposal (one-sided p in the degradation direction)
was excluded as a hypothesis change with documented rationale, not explored as
an explanation. H2 remains INCONCLUSIVE regardless of this outcome: D2 never
compared a co-state against a matched direct critic.

## 6. Limitations and reusable methodology

No confirmatory co-state advantage has been established anywhere in this
programme. Privileged-reference performance does not establish deployable value.
Closed formulations do not exhaust their broader domains (spatial allocation,
QA context, cluster scheduling beyond the tested budgets). All D2/Rung
p-values on reused ids are nominal; the confirmatory analysis is the only preregistered confirmatory inference and it failed.

Reusable: gated Rung-0 opportunity protocol with frozen thresholds; per-budget
Holm families with one-sided bounds; component-clustered inference for composed
benchmarks; outcome-blind operations (opaque progress, sealed outputs, single
unblinding); cost ledgers that refuse to render call counts as latency;
checkpoint contracts with tensor fingerprints; tripwire tests that fail on
reintroduced shortcuts. The defensible methodological observation is that the
protocol prevented an exploratory +1pp answer improvement from becoming an
unsupported positive claim. That observation does not establish the framework
as statistically optimal or superior to other research practices.

## Appendix A. Claim-to-artefact matrix

| Manuscript statement | Claim id | Analysis definition | Permitted interpretation |
|---|---|---|---|
| H2 INCONCLUSIVE, teacher-only hypothesis | `G-H2-programme` | synthesis, no new runs | no co-state advantage established; exact-gradient value is diagnostic |
| S6A negative; VOI void | `S6A-spatial-negative` | trimmed-mean regret ratios, site-clustered bootstrap | no deployable selector separates; VOI comparison untested, not tied |
| λ̂ bottleneck ~170× | `S6C-lambda-bottleneck` | Spearman vs exact gains, per-window | tested variants failed; optimisation/target/information open |
| Hessian branch negative | `S6C-hessian-closed` | predicted vs measured ε, per-window | tested approximation only; no general Hessian claim |
| Conditional θ fail, 28× calls | `S6C-conditional-closed` | per-window θ ratios, one-sided 95% lower bounds; call counts, not latency | formulation closed; norm/fixed retained |
| B2 critic UNDER-REALIZED | `B2-critic-readiness` | rung-stratified regret, floor/convergence/capacity probes | protocol bin, not causal; rescue stopped |
| D2 confirmatory FAIL (table §5) | `D2-confirm-closed` | Holm-6 clustered, §5 | formulation closed; H2 unchanged |
| D5 G1 0.006 fail | `D5-rung0-closed` | area(fixed−oracle)/area(fixed), blocked split | formulation closed; no budget rescue |
| B2.1 positive as caveated (B2.2 registered separately) | `B2.1-adaptive-sensing` + `B2.2-allocator-optimization` | episode-clustered CI | benchmark strength with under-realised-critic caveat |
| Reimplemented-rival forecasting comparison (DROID proprio RMSE under this protocol) | `B1-rivals` | cluster bootstrap, fairness contract | implementation comparison under this protocol; nothing about H2 or published systems |
| Excluded runs (S5, 74-worker) | `S5-robustness` + audit trail | integrity audit | methodology illustrations only |
| 637 tests / 6-6 harness | reproducibility record | `pytest -q`, `harness/check.py` at push `10e9ad2` | software verification, not scientific efficacy |
