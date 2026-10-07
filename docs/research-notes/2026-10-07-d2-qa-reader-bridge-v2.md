# D2-QA reader bridge v2: gate FAILs, primary contrast strongly positive

**Date:** 2026-10-07 · **Status:** executed (L4, Phi-4-mini-instruct, 300 q × 3 sets ×
2 budgets, smoke 1/20); **Bridge FAIL under the frozen rule** (copilot-adjudicated).
**Artefact:** `results/benchmarks/d2_qa_reader_bridge_v2/reader_bridge_summary.json`
(per-question EM/F1/texts). Session stopped, 0 assignments.

## 1. Results (EM counts /300)

| k | oracle | BM25 | longest | oracle−BM25 (Holm) | BM25−longest (Holm) |
|---|---|---|---|---|---|
| 4 | 119 | 29 | 46 | +0.300, p=4.6e-29 ✓ | −0.057, p=0.0047 ✗ (BM25 worse) |
| 8 | 101 | 41 | 46 | +0.200, p=5.7e-17 ✓ | −0.017, p=0.42 ✗ |

## 2. Adjudication (accepted from review)

The frozen PASS required both legs; the sanity leg (BM25 beats longest) fails —
significantly reversed at k=4. **Bridge FAIL as specified.** The oracle−BM25
contrast (+30pp/+20pp) is recorded as a positive descriptive result, not a PASS
with an exception; oracle−longest is exploratory, not confirmatory. No Rung-1
licence is granted by this run.

## 3. Reading

Support recall moves answers decisively for this reader (+30pp/+20pp) — the bridge
question itself answers YES in descriptive terms. But the deployable comparator
the plan named (BM25) loses to an even simpler fixed policy (longest-first) on the
same endpoint. Any follow-up needs a new reviewed plan retaining BM25 **and**
longest-first among deployable baselines, with longest-first's edge (suggestive,
not established) tested prospectively — not a silent comparator swap.
