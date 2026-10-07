# D2-QA reader bridge: NON-DIAGNOSTIC (reader does not answer)

**Date:** 2026-10-07 · **Status:** executed (L4, SmolLM3-3B, 300 q × 3 sets × 2 budgets
+ smoke); result is non-diagnostic, not a FAIL of the bridge hypothesis.
**Artefact:** `results/benchmarks/d2_qa_reader_bridge/reader_bridge_summary.json`
(per-question EM/F1/texts). Session stopped, 0 assignments.

## 1. What happened

EM = 0.000 on all arms at both budgets (paired p = 1.0, vacuous). Inspection shows
why: the reader does not answer questions under the frozen prompt — it continues the
context (echoes paragraph text, emits bracketed entity lists). F1 is floor-level
(oracle 0.09 vs BM25 0.04–0.05: directionally consistent, uninterpretable in absolute
terms). The smoke probe passed 0/20 because its malformedness definition (empty /
verbatim echo / runaway) did not test answer-likeness — a protocol hole, recorded.

## 2. Reading

The gate cannot be adjudicated: EM differences under a non-answering reader measure
prompt compliance failure, not context use. This is **not** evidence that recall
fails to transfer, and not evidence that it does. Per the plan's FAIL scoping (as
narrowed in review), nothing closes beyond this reader+prompt configuration — and
even that narrow closure is suspended, since the endpoint never engaged.

## 3. Route forward (requires new review, not a rerun)

A revised protocol needs: (a) a prompt validated for answer-likeness on a frozen
smoke set *before* the endpoint runs (e.g. standard QA instruction format, checked
against answer shape not just non-emptiness); (b) possibly an instruction-tuned
reader from the cleared list (Phi-4-mini-instruct) with fresh justification — not a
swap chase. No prompt iteration against the endpoint.
