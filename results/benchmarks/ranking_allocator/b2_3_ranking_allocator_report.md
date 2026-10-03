# Milestone B2.3 Ranking Allocator Optimization Benchmark

**Date:** 2026-10-02T23:11:22Z · **Run ID:** `droid100_adjoint_v2_5seeds_20261001T080821Z` · **Seeds:** 3
**Max Training Steps:** 1000

## 1. Test Regret Comparison Across Ranking Loss Variants

| Allocation Policy / Loss | Mean Test Regret | Std Regret | Gap to Oracle Floor (0.03007) | Advantage vs CE |
|---|:---:|:---:|:---:|:---:|
| **Exact Autograd Oracle** | `0.03308` | `±0.00559` | `0.00000` | — |
| **Refusal Baseline (`always_mode0`)** | `0.12646` | `±0.01385` | `+0.09338` | — |
| `costate_ce` | **`0.13781`** | `±0.02928` | `+0.10473` | **`+0.00000`** |
| `costate_margin` | **`0.13858`** | `±0.03092` | `+0.10550` | **`+0.00076`** |
| `costate_listwise` | **`0.14257`** | `±0.03216` | `+0.10949` | **`+0.00476`** |
| `costate_hybrid` | **`0.13963`** | `±0.03016` | `+0.10655` | **`+0.00181`** |
