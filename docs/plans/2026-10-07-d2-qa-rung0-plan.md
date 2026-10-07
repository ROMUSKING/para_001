# D2-QA: paragraph-selection domain card and Rung-0 opportunity gate (MuSiQue)

**Written:** 2026-10-07, before any data file is downloaded · **Status:** DRAFT for peer
review — no downloads, no implementation until reviewed. **Roadmap:** D2-QA (new,
Tier P) · **Precedence:** comprehensive plan §7.2B-1 (opportunity gate); D1-0 plan
(structural template); licence register `musique` row (`adopt`, CC BY 4.0).

## 1. Question and scope

**Question.** For multi-hop QA over paragraph sets, is there room for adaptive
*context* allocation: does an allocation that knows the supporting paragraphs (a
privileged oracle) beat the best fixed allocation by at least 15 % of the normalised
area at equal paragraph budget, and how much of that room can a deployable policy
that sees only the question keep?

**This is a Rung-0 gate.** Non-learned policies only; no statement about H2 (no
learned critic, no co-state). No reader model is run: the endpoint is support recall
(§2), not answer accuracy. A G1 pass is **permission to pursue a reader-aware Rung-1
design, not proof that paragraph allocation improves QA** — the recall-to-accuracy
link is unverified, and a policy could raise recall without helping (or while
harming) a robust reader. The result is exploratory and uses validation questions
only; **the test file is never downloaded.**

## 2. Domain card

| # | Requirement | D2-QA |
|---|---|---|
| 1 | Structured state | A question with a paragraph set: 20 paragraphs per question (MuSiQue full). Hierarchy *question → paragraph*; the paragraph is the allocatable unit. |
| 2 | Downstream objective | `J`, lower is better: **support miss rate** `J_w(S) = 1 − |S ∩ P_w| / |P_w|`, `P_w` the supporting-paragraph id set from the dataset's decomposition labels (privileged). Budgets below |P_w| cannot reach 0 — reported, not hidden. |
| 3 | Legal actions | `include(p)` (kind `sample`); `stop` (leave budget unused). Order within S is irrelevant to J. |
| 4 | Constrained resource | Paragraph slots (ledger R): exactly `k` paragraphs per question. Policy computation not charged (see §9 of D1-0 pattern). |
| 5 | Measurable effects | Exact for every allocation: J is computed from labels, no estimator. Oracle is exact up to ties (supports first, then arbitrary fill). `DomainSpec.oracle_support = "exact"`. |
| 6 | Baselines | Fixed: question-order first-k, longest-paragraphs first, round-robin over source documents, random-expected. Dynamic deployable (question text only): BM25 ranks, TF-IDF cosine ranks. Oracle: supports-first, privileged. **Learned rankers are Rung 1, not built here.** |
| 7 | Privilege | P0 (at decision): question text + paragraph texts. P3 (evaluation/oracle only): supporting-paragraph ids, answers, decompositions. |
| 8 | Data | MuSiQue full + ans Drive files (owner README links; CC BY 4.0, register `musique`, `adopt`). Downloaded to ephemeral cache, kept out of git, pinned by SHA-256 in the run manifest. Attribution: StonyBrookNLP MuSiQue, CC-BY-4.0, in note + manifest. |
| 9 | Opportunity | The gate of §4. |
| 10 | Regime | Acquisition of unknown information (which paragraphs matter). A pure relevance score without question conditioning cannot work; whether BM25 suffices is the empirical question. |

## 3. Splits and budgets (frozen)

- **Files:** `musique_full_v1.0` (train/dev/test) + `musique_ans_v1.0` (answerable subset). **Canonical registry first:** load both files' question ids, deduplicate (ans overlaps full), freeze one id→split assignment (train→tuning, dev→validation; test ids listed but the test file is never downloaded). Shuffle with seed 0 once over the registry, then cap within split (tuning ≤ 2000, validation ≤ 1000 in registry order). The frozen id lists are committed in the run manifest — the cap is a declared rule, not a knob.
- **Sample caps (frozen):** tuning ≤ 2000 questions, validation ≤ 1000 (first-N in file order after a frozen shuffle with seed 0 — no selection on content). Caps bound CPU BM25 cost, not a tuning knob.
- **Budgets:** `k ∈ {1, 2, 4, 6, 8, 10, 15, 20}` paragraphs. `k = 20` observes everything.
- **Best fixed allocation** = the fixed policy with the lowest mean normalised miss area on **tuning** (chosen there, applied to validation).

## 4. Gate (frozen; mirrors D1-0 §7.2B-1)

- **G0 correctness:** every J in [0, 1]; oracle J ≡ 0 wherever k ≥ |P_w|; deployable policies never read P3 fields (input-boundary test: shuffling support ids must not move any deployable selection).
- **G1 opportunity:** relative headroom = area(J_fixed − J_oracle)/area(J_fixed) ≥ **0.15** on validation, trapezoidal over k/20.
- **G2 deployable share:** best deployable keeps ≥ **0.50** of the headroom (area(J_fixed − J_dep)/area(J_fixed − J_oracle)). Failing G2 with G1 passing means: redesign candidates, not a learned-rescue claim.
- Statistics: per-question pairing; gate on point estimates with the same clustered treatment as prior notes only if G1 passes by >5 points, else reported as exploratory.

## 5. What this does not licence

No H2 claim; no reader-quality claim; no token-cost claim (paragraph slots ≠ tokens — paragraph lengths vary; a token-priced variant is Rung-1 work). BM25/TF-IDF need no model download (scikit-learn OK; `rank-bm25` only if vendored and pinned).
