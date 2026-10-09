# D2 confirmatory: frozen ranker answer-quality on remaining ids (FROZEN)

**Status:** FROZEN for peer review — no reader run, no outcome inspection until
reviewed and approved. This is confirmation, not optimisation: ranker weights,
features, reader, prompt, budgets, and comparator set are all inherited frozen.
**Question (directive A):** does the frozen logistic context ranker preserve
answer quality within the inherited −2pp margin on previously unused evaluation
examples?

## 1. Pool (exposure manifest `results/benchmarks/d2_confirm/exposure_manifest.json`)

- 1,417 remaining dev-answerable questions (2,417 total − 1,000 Rung-0-evaluated;
  tuning-2,000 from the train file disjoint by id; frozen-300 ⊂ used-1000).
- Of the 1,417: 282 component-clean (no sub-question in 14,589 train-file
  components or used-1000 components), 1,135 shared. Union cluster structure:
  439 clusters over the remaining (max 35); 200 clusters over the clean subset
  (max 6) — the sensitivity analysis is clustered too.
- **Estimand (review):** question-level performance conditional on prior
  component exposure — NOT a clean component-generalisation claim. Overlap with
  fitting or prior-evaluation components scopes the claim; it does not invalidate
  the estimate. The manifest carries ids, cluster assignments, input SHA-256,
  code rev, and selection rule (no outcomes).

## 2. Power (paired, discordance-based, from development evidence)

Development discordance (bridge v2, n=300): learned-only 15, BM25-only 11–12
(~9%; Wilson 95% intervals on the discordant rate: 6.0%–12.4% at k=4, 6.3%–12.8% at k=8 — carried as
uncertainty, not a point assumption). Scenarios at true diff +1pp, margin −2pp,
one-sided α=0.025. **Deff values are illustrative sensitivity cases, not
established bounds:**

| design | discordance | se | pass needs d̂> | power at true +1pp |
|---|---|---|---|---|
| n=282 clean, Deff=1 | 6.0–12.8% | 0.0146–0.0213 | +0.0086–+0.0217 | 29–54% |
| n=1417, Deff=1 | 6.0–12.8% | 0.0065–0.0095 | −0.0072–−0.0014 | 88–99.6% |
| n=1417, Deff=2 | 6.0–12.8% | 0.0092–0.0134 | −0.0020–+0.0063 | 61–90% |
| n=1417, Deff=4 | 6.0–12.8% | 0.0130–0.0190 | +0.0055–+0.0172 | 35–64% |

No sample size is claimed sufficient unconditionally: if the clustered lower
bounds are too wide, the verdict is the **third conclusion** (unresolved), not a
fail-and-retry. No batch extension, no margin relaxation, no id recycling.

## 3. Frozen method

Ranker weights/coefs, 6 features, BM25/longest comparators, oracle reference,
Phi-4-mini-instruct fp16 greedy, v2 prompt/template bytes, k=4,8, EM+aliases,
token-F1 diagnostic — all inherited unchanged. Sets generated for all 1,417
before any reader call; set manifest hashed; then reader run; then join.
Reader calls: 1417 × 3 sets (learned/BM25/longest) × 2 budgets = 8,502 + smoke.

**Outcome-blind operations (review):** progress logs show completion counts and
failure lines only — no per-arm or per-id correctness, no decoded outputs, no
EM summaries until the single unblinding point after all calls reconcile.
Retries: up to 3 attempts per call, then marked missing (missingness reported by
arm/budget; no refill, no substitution). Immutable append-only run log. One
unblinding: EM computed once, all arms together, after call reconciliation.

## 4. Frozen analysis (one analysis, Holm family as Rung-1)

Recall superiority learned-vs-BM25 (k=4,8; +5pp) + recall vs longest lower
bounds (−2pp) + EM vs BM25 lower bounds (−2pp), one-sided Wilcoxon /
Holm-adjusted Wald lower edges, component-clustered SEs for the primary (cluster
bootstrap over the 439 union groups; disjoint sensitivity clustered over its
200 groups). Exploratory
labels removed ONLY for this preregistered analysis; everything else stays
exploratory.

## 5. Predeclared conclusions (directive table)

- All quality + resource gates pass → bounded offline practical result for this
  reader/budget/population. H2 unchanged.
- Unacceptable degradation or resource failure → close this formulation.
- Unresolved intervals or integrity shortfall → stop under cap; distinct from
  demonstrated inferiority. **Final confirmation attempt for the frozen D2
  formulation under the current budget; re-entry needs new evidence or premise.**
