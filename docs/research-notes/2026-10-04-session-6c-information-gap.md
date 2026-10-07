# Session 6C Experiment 2: information-gap linear probes

**Date:** 2026-10-04 · **Milestone:** Session 6C (learnability audit, Exp 2 bounded form)
**Hardware:** NVIDIA L4 · **Artefact:** `results/benchmarks/information_gap/information_gap_summary.json`
**Runner:** `scripts/diagnose_information_gap.py` · **Design:** plan §7 as peer-reviewed
(kilo, milestone `Session-6B-Exp2-amendment`) — equal-capacity projection, reclassified
conditions, inverted falsifier, condition D.
**Status:** directional evidence only; no gates pass here; no superiority claims.
**Teacher:** `best_spatial.pt` SHA `cfbca4406f13df55…`; same 16+16 windows as Exp 1
(4+4 train episodes, ids committed); closed-form linear probes, no hyperparameters.

## 1. Result: nothing transfers, including privilege

| condition | fit vec | fit ρ | hold vec | hold ρ | hold agree@2 | gate |
|---|---|---:|---:|---:|---:|:--:|
| D (full-C weights, privileged zero-filled at test) | 0.0000 | +1.0000 | 0.9957 | −0.0300 | 0.125 | True |
| A (deployment, d=12) | 0.0690 | +0.9906 | 4.1202 | −0.2556 | 0.000 | False |
| B_ctx (legitimate, d=12) | 0.0639 | +0.9952 | 2.8398 | +0.1752 | 0.0625 | False |
| B_log (logged actions, d=12) | 0.0666 | +0.9910 | 3.1318 | +0.1746 | 0.0625 | False |
| C (privileged targets, d=12) | 0.1675 | +0.9652 | 2.9221 | +0.2424 | 0.0625 | False |

All fits train well (vec ≤ 0.17, ρ ≥ 0.96); all fail held-out catastrophically
(vec 2.8–4.1, all worse than the zero reference 1.0; agreement 0–0.125, chance-level).
Calibration a* = 1.0 everywhere — scale is not the issue.

## 2. Reading (frozen falsifiers, no leakage into superiority)

- **Inverted falsifier:** C (2.92) vs A (4.12) — both catastrophic, neither near usable.
  Privilege adds nothing transferable even mechanically: the gap framing gains nothing
  from this probe. The next look is optimisation/labels, per the retained falsifier.
- **Condition D:** full-C exact train fit (vec 0, gate True) collapses to 0.996 with
  privileged columns zeroed — the C fit depended **entirely** on privileged columns.
  Combined with C-projected failing on held-out *with* its columns present, no column
  subset transfers: the mapping itself does not generalise across these episodes.
- **B_ctx ≈ B_log ≈ C on held-out** (2.84/3.13/2.92): legitimate context adds nothing over
  deployment either. Everything sits in the noise band of a shared failure.
- **Limitations stated plainly:** (i) the d=12 random projection may destroy the sparse
  deterministic C structure that full columns preserve — equal capacity was bought by
  potentially discarding the very signal being tested; (ii) 16+16 windows cannot separate
  "no transferable signal" from "too little data even for privileged linear"; (iii) linear
  probes only — a nonlinear information advantage would not appear here.

## 3. Addendum: v2 rerun reconciles D and reframes the pair question

Rerun with current code (hashes `diagnose_information_gap.py 9de66950…`,
`diagnose_pair_interactions.py 6e33a768…`, verified identical pre-launch).

- **D is a separate object, proven:** D weight hash `9ba6dd18…` ≠ C hash `84c45b46…`.
  D held-out prediction norm 6.27e-03 vs target norm 1.17e-01 (5%) — near-zero output
  confirmed via norms, not inferred from rel error 1.0. Training row 0.000/+1.00 belongs
  to that full-space fit. The review's reconciliation is satisfied as specified.
- **Pair ranking fidelity is 0.9887 (n=32 windows):** summed singleton gains order
  measured pair gains near-perfectly within windows — the additive surrogate preserves
  ORDER even where absolute ε is large. Residual interaction fraction 0.52: half the
  spread is pair-specific, but it rarely flips order.
- **Same-budget decision unresolved:** measured-on-both-sides subset (n=6) ties at
  +0.00001; the headline −0.00224 (Wilcoxon p=2.0e-05, site CI [−0.00309, −0.00144])
  comes entirely from the 26 windows using the additive fallback. Per the review's
  contracts this cannot support a decision claim — the manifest must include likely-top
  sets or evaluate top-k sets directly.
- Net: singleton sums rank pairs excellently; whether the best pair beats the
  top-singleton set at equal budget is unmeasured (n=6 tie). The conditional-gain
  experiment stays motivated, re-aimed: beat additive *ordering*, not repair broken
  ranking.

## 4. Reproduce

Substrate rebuild (cache → backbone, score −2.1425 again) then:

```bash
python scripts/diagnose_information_gap.py --cache-dir /content/spatial_cache_e3_1 \
    --drive-root /content/local_runs --run-id spatial_patches_20261003 \
    --output-dir results/benchmarks/information_gap
```

L4 retained. Regret references are greedy-rollout lower bounds throughout.
