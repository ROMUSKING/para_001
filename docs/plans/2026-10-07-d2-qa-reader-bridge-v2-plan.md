# D2-QA reader bridge v2 (revised protocol after non-diagnostic v1)

**Status:** DRAFT for peer review — no provisioning until reviewed.
**Predecessor:** v1 (`2026-10-07-d2-qa-reader-bridge.md`) — EM 0 everywhere because the
base model continued context instead of answering; smoke lacked answer-likeness.
**Change basis (pre-endpoint, principled):** v1 used SmolLM3-3B, a base (non-
instruction-tuned) model, with an ad-hoc prompt. v2 uses Phi-4-mini-instruct
(MIT, register `adopt`, 7.67 GB — fits L4): instruction tuning is the relevant
property for following an answer instruction, selected for that property plus
licence clearance, not for any observed score. No model may be swapped again on
the basis of v2 outcomes.

## 1. Frozen v2 differences (everything else identical to v1 plan)

- **Reader:** `microsoft/Phi-4-mini-instruct`, fp16, greedy, 32-token cap.
- **Prompt:** the model's chat template with system instruction
  (“Answer the question using only the context. Reply with the answer alone, no
  explanation.”) + user block (question + titled paragraphs). Template bytes frozen
  by hash in the artefact (same mechanism as v1).
- **Smoke (answer-likeness, frozen):** questions 20–39 of the frozen 300 (fresh set,
  disjoint from v1's smoke 0–19), oracle sets at k=8 (carried frozen from v1).
  Malformed = empty, verbatim question echo, uncapped runaway, **or structural
  non-answer: more than one sentence or any “?” in the output** (legitimate answers
  are short noun phrases; the v1 failure emitted multi-sentence continuations; the
  ≥12-word-span idea was dropped on review since extractive answers legitimately
  quote 8–15 word spans). Stop if **≥6/20** malformed (binomial: P(X≥6|n=20,
  p=0.10)≈0.011 < 0.05 against a 10% background malformed rate).
- **Endpoints, budgets, sample, gates, cost:** unchanged from v1 (per-budget EM
  oracle−BM25 with Holm, +3pp bar with power disclosure, n=300 frozen ids).

## 2. What v2 can and cannot conclude

- PASS licences the design of a separately gated Rung-1 study (unchanged meaning).
- FAIL (adjudicated EM, reader answering): closes the tested configuration only.
- NON-DIAGNOSTIC again (reader not answering despite instruction tuning + template):
  auto-converts to FAIL for this bridge hypothesis (falsifiability cap, reviewed) —
  no third attempt, no new reader class under this hypothesis. The run metadata
  records the v1→v2 switch rationale (base→instruction-tuned property + licence)
  so the change can never read as score-chasing.
