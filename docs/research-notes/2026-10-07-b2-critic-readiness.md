# B2 direct-critic readiness verdict (§1893 probes)

**Date:** 2026-10-07 · **Status:** RECORDED — peer-reviewed (codex, CHANGES_REQUESTED, all
accepted; verdict wording below is the reviewer's, adopted verbatim).
**Artefacts:** `results/benchmarks/critic_probes_{w10,w05,w20,cont}/summary.json`
(5 seeds each, all finite; per-seed partials; 60/60/60/125-ledger entries).
**Lineage:** same Drive cache (100/100 episodes sha-verified), same 5 teachers, same
seeds/loader order as B2.2; critic-only reruns reproduce B2.2's critic trajectory
(step-300/1000 values byte-identical). Costate numbers absent by design.

## 1. Results vs the frozen gates

- **P0 floor: FAILS on the low-cost rung.** Low rung (oracle modes 0–1): adaptive gain
  negative on all 5 seeds (−0.09 to −0.16); critic numerically worse than uncertainty on
  all 5 (significant on 3/5, p 1e-4..1e-10; n.s. on 2/5). High rung passes on all 5
  (vs random p≤1e-32; vs uncertainty p 0.03..1e-17; adaptive gain hugely positive).
  Rungs stay separate per plan — the high pass does not offset the low fail. Caveat
  (reviewer): panel Wilcoxons are per-window without episode clustering — treat
  p-values cautiously; the consistent negative adaptive gain stands regardless.
- **P1 convergence: NOT demonstrated.** Final-20% val-surrogate moves −4.5%..+3.9%,
  exceeding the 1% threshold on all seeds (mixed signs). The 2500→5000 comparison
  (−19%..+24% per seed, mean +3% worse) is schedule-confounded (5000-horizon cosine
  vs 2500-horizon) — confounded evidence, not a valid continuation estimate.
- **P2 capacity: no ≥2% gain at 2× width** (0.5× 0.32492 / 1× 0.32787 / 2× 0.33234;
  2× worse by 1.4%). 4× reference not triggered. This does not establish broad
  saturation (P1 failed; 4×/2× condition untested).

## 2. Verdict

“P0 fails on the low-cost rung; P1 convergence is not demonstrated, and its continuation
comparison is schedule-confounded; P2 shows no ≥2% gain at 2× width. Under the frozen
readiness rule, classify the critic as `UNDER_REALIZED_DUE_TO_COMPUTE_CONSTRAINTS` and
leave H2 inconclusive. This is a protocol classification, not evidence that compute is
the cause. Stop further rescue at the declared cap unless separately approved.”

Negative low-rung adaptive gain motivates an objective-side hypothesis but does not
prove it. Cost: 4 probe runs ≈ 2× B2.2-allocator envelope as declared; session stopped,
0 assignments. Balance 341.23.
