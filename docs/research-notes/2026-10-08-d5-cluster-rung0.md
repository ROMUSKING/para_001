# D5 cluster admission Rung-0: G1 fails informatively, formulation closed

**Date:** 2026-10-08 · **Status:** executed (Borg 2011 slice, parts 170–239, 87,516
jobs, 44+45 windows); **G1 = 0.0058 (FAIL), G2 negative (report-only)**.
Formulation closed per reviewed routing: no budget amendment, no learned rescue.
**Artefact:** `results/benchmarks/d5_cluster_rung0/d5_cluster_rung0_summary.json`.
**Peer review:** kilo (plan, 4 findings) + copilot (amendment + outcome; routing
adopted). CPU-only; test/usage tables never fetched.

## 1. Results

- **G−1 (tuning-eligible):** entropy 2.16 bits, top-1 mass 32%, n=23,943 — the value
  metric has granularity; not degenerate.
- **G1 headroom = 0.0058 (FAIL ≥ 0.15).** Oracle 0.9002 vs fixed 0.9055 on
  validation (tuning 0.9139 vs 0.9282). Per-budget oracle miss 0.996 (k=1) → 0.863
  (k=32), tracking fixed within ~0.01 throughout.
- **Pools dwarf budgets:** median pool 1067 tuning / 675 validation vs k ≤ 32
  (k/pool ≈ 3–5%). Whether the failure is structural (budgets too small) or
  substantive (outcomes unpredictable) cannot be distinguished from this
  artefact — and per the reviewed routing, a budget change now would be
  rescue-tuning, not evidence. It would need an independent prospective rationale
  (deployment capacity), a frozen gate, and a fresh untouched holdout.

## 2. Reading

Knowing realised outcomes buys essentially nothing at these budgets: completion
is not usefully predictable from submit-time fields on this slice. The plan's
split amendment (censor-first median split) is recorded as implemented, not as
originally written. Branch: closed.
