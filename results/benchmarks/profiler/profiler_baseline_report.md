# Session 0 Profiler Baseline Report

**Date (UTC):** 2026-10-03T08:43:27.622148+00:00 · **Device:** cuda · **GPU:** NVIDIA L4 · **VRAM:** 22.03 GiB · **torch:** 2.11.0+cu130 · **Batch size:** 32

> SYNTHETIC throughput microbenchmark: batches are on-the-fly random tensors used only to measure pipeline timing and VRAM scaling. These numbers are not model-quality evidence and must not be cited as held-out gains.

## Per-config results

| Representation | num_workers | Step median (ms) | Step p95 (ms) | DataLoader wait % | GPU kernel exec % | Peak alloc (MiB) | Peak reserved (MiB) |
|---|---|---|---|---|---|---|---|
| pooled | 0 | 3.26 | 3.41 | 55.9 | 35.0 | 37.5 | 270.0 |
| pooled | 2 | 2.20 | 5.17 | 24.1 | 50.5 | 37.5 | 270.0 |
| pooled | 4 | 2.27 | 2.87 | 21.0 | 48.3 | 37.5 | 270.0 |
| spatial | 0 | 158.85 | 160.63 | 95.5 | 9.2 | 213.8 | 270.0 |
| spatial | 2 | 86.58 | 157.72 | 92.0 | 16.9 | 213.8 | 270.0 |
| spatial | 4 | 7.19 | 151.66 | 83.9 | 29.3 | 213.8 | 270.0 |

## Gate G4-1 reading (profiler saturation)

Best pooled config: num_workers=2, step median 2.20 ms, dataloader wait 24.1%, GPU kernel exec 50.5%.
Verdict: **NOT GPU-saturated (G4-1 not met)** — G4-1 requires GPU kernel execution to dominate step time with dataloader wait < 15%.

## Chrome trace

Full Kineto trace for (pooled, num_workers=0): `chrome_trace_pooled_nw0.json` (if present; omitted from git if > 5 MB per repo policy).
