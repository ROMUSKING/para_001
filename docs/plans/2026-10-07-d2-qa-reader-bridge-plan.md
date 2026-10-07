# D2-QA reader bridge (GPU): does support recall move answer accuracy?

**Status:** DRAFT for peer review — no provisioning, no downloads until reviewed.
**Predecessor:** Rung-0 (`2026-10-07-d2-qa-rung0.md`, G1 pass on recall / G2 fail).
**Question it answers:** the explicitly deferred recall→accuracy link. A positive
result licences a reader-aware Rung-1; a negative result closes the QA-context
direction without further spend.

## 1. Design (frozen reader, frozen sets, inference only)

- **Reader (frozen):** SmolLM3-3B (`HuggingFaceTB/SmolLM3-3B`, Apache-2.0, 6.15 GB,
  register `adopt`) — fits L4 with headroom. Fallback if it fails to load: Phi-4-mini
  (7.67 GB, MIT). One reader only; no comparison between readers, no finetuning.
- **Paragraph sets (frozen, from the committed Rung-0 artefact
  `results/benchmarks/d2_qa_rung0/`):** oracle supports-first vs BM25 top-k vs
  longest-first, at k ∈ {4, 8} (the budgets where Rung-0 headroom concentrates).
  Same validation questions (frozen id list). No new selection, no tuning.
- **Prompt (frozen):** system + `Question:` + `Context:` (selected paragraphs in
  rank order, titled) + `Answer in a few words:`. Greedy decoding, 32-token cap.
- **Metrics:** exact match (primary; accepted if it matches the answer or any entry
  in the dataset's `answer_aliases`, after case/whitespace normalisation) + token-F1
  (diagnostic), paired per question. Endpoints are **per budget** (k=4 and k=8
  separately — 3 sets × 2 budgets = 1800 reader calls, not 900): EM(oracle) − EM(BM25)
  with paired Wilcoxon + clustered bootstrap over questions, Holm correction across
  the two budget tests. Practical bar frozen at +3 EM points (no relative-ratio gate
  on small denominators).
- **Power disclosure (review):** with n=300 paired binary outcomes, power depends on
  the discordant-pair rate, which is unknown before running: at 25% discordance the
  SE of the paired difference is ≈2.5pp, so +3pp is a practical bar, not a powered
  detection threshold. The bar and the significance test are separate requirements;
  the CI around the effect is reported either way.
- **Ceiling interpretation (review):** supports-first uses privileged labels — its
  gain shows this reader *can* benefit from these paragraphs, not that a deployable
  selector reproduces it. Result is protocol-specific (one reader, one prompt,
  greedy decoding, extractive-style short answers).
- **Sample:** 300 validation questions (first-300 of the frozen Rung-0 validation id
  list — subset declared before running, not chosen on outcomes).

## 2. Gates (frozen)

- **Bridge PASS:** oracle sets beat BM25 sets by ≥ +3 EM points (paired p < 0.05,
  Holm-adjusted) at either budget AND BM25 beats longest-first (sanity that the
  reader responds to relevance at all). Licences **the design of a separately
  gated Rung-1 study** (learned ranker vs BM25 on held-out questions with answer
  endpoints) — not training, not a learned-benefit claim.
- **Bridge FAIL:** anything else. Scoped to this reader and prompt: closes nothing
  beyond the tested configuration; no model swap without a new reviewed plan
  (swapping readers until one works is tuning on the endpoint).

## 3. Cost and provenance (prospective)

- Single L4 session; 1800 reader calls (300 q × 3 sets × 2 budgets) + smoke probe + trial overhead; stop on
  completion, then `colab sessions` verifies 0 assignments.
- Artefacts: per-question EM/F1 rows + summary JSON with reader id, quant setting
  (fp16), prompt hash, set provenance (Rung-0 artefact hashes), model SHA.
- No training, no gradient updates, no Drive dependency (model + data from HF).

## 4. Non-goals

No co-state, no critic, no H2 content. No token-pricing claims. No reader comparison.
If SmolLM3 fails the smoke probe below, the run stops and the design returns for
review instead of silently swapping readers. **Smoke probe (frozen, reviewer-
specified):** 20 questions (first-20 of the frozen 300), oracle sets at k=8.
Malformed = empty output OR verbatim echo of the question OR no EOS within the
32-token cap. Stop if ≥5/20 malformed — a reader-usability floor for this frozen
protocol (below it, EM differences measure instruction-following failure, not
context use), not a tuned threshold.
