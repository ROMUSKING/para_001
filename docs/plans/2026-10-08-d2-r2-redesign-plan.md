# R2: bounded D2 candidate redesign (CPU-first, frozen)

**Status:** DRAFT for peer review — no implementation until reviewed.
**Predecessor:** Rung-0 G1 pass / G2 fail (BM25 keeps 26%); bridge v2 FAIL.
**Question:** can a small frozen family of deployment-available candidate generators
beat BM25's recall at the same budgets, before any learned selector?
**Budget:** at most three new constructions (below), frozen. Failure closes this
bounded study; no wider sweep, no larger model.

## 1. Frozen candidates (deployment inputs: question + visible texts/titles only)

1. **`entity_expanded`:** BM25 with the question plus visible entity spans
   (`[A-Z][a-z]+` sequences and quoted spans from the question, ×2 term weight).
   Provenance of ×2 (review): a round-number field-boost convention, frozen
   a priori without any tuning run — not an optimized value. Overlapping spans
   merged (each distinct span counted once); repeats capped so no term exceeds
   ×2 total weight. Hypothesis: visible entity links recover evidence missed by
   raw question matching.
2. **`incremental_bm25`:** round 1 = BM25 top-1; round 2 = BM25 with the question
   plus the top-5 TF-IDF terms of the round-1 pick absent from the question;
   duplicate suppression; exactly 2 rounds. Provenance (review): the minimal
   multi-hop shape — one follow-up round, five terms ≈ one sentence of new
   vocabulary — frozen a priori, not tuned. Extraction frozen: TF-IDF corpus =
   the question's own 20 paragraphs, repo `tokenize`, no stopword list, ties to
   the lower paragraph index (same conventions as the Rung-0 rankers).
   Hypothesis: first evidence reveals second-hop terms.
3. **`title_bundle`:** group paragraphs by title; rank groups by BM25(question,
   concatenated group text); take whole groups in rank order, skipping any group
   that does not fit the remaining slots (budget strictly enforced; undershoot
   recorded). Hypothesis: connected evidence jointly beats isolated picks.
   Anti-gaming (review): recall is reported alongside **support precision**
   (|S∩P|/|S|) and token cost, so whole-group grabbing that collapses precision
   is visible, not rewarded.

Gold supports, answers, and decompositions never enter construction (boundary
tests pin this). Candidate universe stays the supplied 20 paragraphs (no open
retrieval — expansion would change the problem).

## 2. Controls and budgets

Retained: `bm25`, `tfidf`, `longest_first` (validation-selected fixed), `first_k`,
`doc_round_robin`, `random_expected`, `oracle` (privileged), plus `no_context`
(miss 1.0 floor) and `full_context` (k=20 ceiling).
Paragraph budgets k ∈ {2, 4, 6, 8, 10} (Rung-0 continuity). Token accounting with
the Phi-4-mini-instruct tokenizer over `[title] text` + `\n` separators: per-arm
token distributions reported; secondary token-capped arm at the tuning-median
oracle-k8 token cost, filled in rank order.

## 3. Metrics and gate (frozen)

Per question: miss rate, **complete-support recovery** (1 if all supports
selected, else 0 — fraction reported), **support precision** (|S∩P|/|S|),
per-hop bucket (2/3/4-hop id prefix), selected token count. Endpoint: mean recall
(1 − miss) at k=4 and k=8 with one-sided paired Wilcoxon (improvement direction)
+ Holm across the full six-test family (3 candidates × 2 budgets).
**Improvement bar:** a new candidate beats BM25 by ≥5pp absolute recall at k=4 or
k=8 with Holm-adjusted p < 0.05 — significance and practical bar are separate
requirements. Passing candidates are reported, not deployed; the study
passes if ≥1 does, and closes otherwise. No learned rescue on any outcome.
