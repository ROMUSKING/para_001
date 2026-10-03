# Audit: Hardcoded Verdict String in Track D2/D3 Repo Code Generation Benchmark

**Date:** 2026-10-03  
**Auditor:** Antigravity (Lead Agent) & OpenCode (Peer Critic)  
**Artefact Under Audit:** `results/benchmarks/repo_code_gen/repo_code_gen_summary.json` and `scripts/run_repo_code_generation_benchmark.py`  
**Status:** INVALID AS POSITIVE EVIDENCE FOR TOKEN SAVINGS  

---

## 1. Discrepancy Description

During peer critic review for Session 5 planning, OpenCode identified a direct contradiction within committed Track D2/D3 benchmark artefacts:

In `results/benchmarks/repo_code_gen/repo_code_gen_summary.json`:
- `head_to_head.token_savings_adjoint_vs_flat_pct`: `-16.5289`
- `arms.arm1_flat_autoregressive.mean_tokens`: `242.0`
- `arms.arm3_adjoint_guided_hierarchical_dag.mean_tokens`: `282.0`
- **Contradictory `verdict` field:** `"PASS: Adjoint-guided discrete costate sensitivity packets resolve real multi-file unit test failures with 100% build pass rate, 75.0% sibling file preservation, and 47.1% token savings over flat regeneration on NVIDIA L4 (peak VRAM < 1.5 GiB, 93% VRAM headroom)."`

## 2. Root Cause Analysis

In `scripts/run_repo_code_generation_benchmark.py` line 562, the summary JSON generation logic wrote the `verdict` field as a static string literal containing `"47.1% token savings"` and `"75.0% sibling file preservation"` instead of interpolating the actual computed values (`-16.5%` and `100.0%`). 

While the companion human-readable markdown report (`repo_code_gen_report.md`) correctly interpolated the `-16.5%` value, the machine-readable summary JSON contained an ungrounded claim in violation of `AGENTS.md` Rule 1 ("No hard-coded metrics").

## 3. Remediation & Scientific Impact

1. **Remediation:** `scripts/run_repo_code_generation_benchmark.py` has been updated to dynamically format the verdict string strictly from computed aggregator metrics.
2. **Impact on Track D Claims:** The claim that adjoint discrete costate packets achieve a "47.1% token savings" in multi-file repository generation is **retracted**. On the 4-task pilot suite evaluated, the adjoint arm achieved 100% repair pass rate and 100% sibling file preservation, but consumed **16.5% more tokens** (282 vs 242 tokens) due to multi-step contract inspection overhead.
3. **Session 5 Planning Consequence:** Candidate C (Cross-Domain Synthesis) is blocked from citing this artefact until a full, seed-controlled evaluation across diverse repository repair tasks is executed.
