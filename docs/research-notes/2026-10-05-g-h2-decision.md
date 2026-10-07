# G-H2 decision record: co-state vs direct marginal-gain critic (programme synthesis)

**Date:** 2026-10-05 · **Status:** RECORDED at diagnostic strength — peer-reviewed (kilo,
CHANGES_REQUESTED, all six findings accepted; prior draft class TEACHER-ONLY withdrawn).
**Lead:** opencode · **Scope:** synthesis of committed artefacts only; no new runs.
**Programme class: INCONCLUSIVE (direct baseline under-realised; no confirmatory test ever
ran). Leading hypothesis: teacher-only value.** See §4 for why these are two different
statements and what each opens.

## 1. Procedural fact constraining the verdict

The comprehensive plan's formal confirmatory H2 (preregistered δ/w, alpha-spending,
8–32 paired seeds) was never executed. Every H2-relevant result below is a 3–5-seed
benchmark or diagnostic. The per-domain mapping (§3) is the load-bearing part of this
record; the single programme class (§4) is a resource-allocation judgement at diagnostic
strength, not a gate verdict.

## 2. Evidence by domain (committed artefacts only)

| Domain | Adjoint-relevant result | Realisation / audit status |
|---|---|---|
| DROID-100 adaptive sensing (B2.1) | Adjoint − critic = −0.00808, 95% CI [−0.01481, −0.00047], 5/5 seeds | Critic collapsed to full observation — under-realised baseline suspected |
| DROID-100 allocator optimisation (B2.2) | Co-state advantage −0.03032 → −0.05937 at 2500 steps (p<0.0001); gap closed/reversed | Critic regret 0.18960→0.18998→0.18956, identical to worst static `always_mode3` (0.18959): collapse signature, not saturation; critic *loss* curve unreported. Under the plan's collapse trigger the direct arm is UNDER-REALISED — the B2 advantage is a benchmark signal, not gate-capable evidence |
| S4 curvature/VOI | Gate PASS reported (VOI 0.06354, −22.84% vs critic) | **Excluded from this record.** The S5 audit's remediation items 1–3 are BLOCKING (seed duplication with sign flips across teachers; tail-dominated headline; VOI≡curvature family identity) and forbid citation as co-state evidence. An earlier draft reused the regret number as a "policy number"; the reviewer showed Findings 1 and 3 reach the number too. Excluded in both directions. |
| S5 robustness | Primary endpoint unsupported (2/3 seeds duplicated) | Excluded in both directions |
| S6A spatial selection (1,200 windows, 50ep/12 sites, 3 seeds) | No deployable selector separates (incl. distilled adjoint scorers); VOI comparison void, retired | Supports direct-sufficient in this domain |
| 6C learnability (192 windows + 150 protected) | λ̂ misses exact-λ by ~170×; conditional formulation closed (θ gate fails, 28× calls) | Exact-λ (~2e-05) works — value exists, distillation fails |
| D4 simulation (CPU) | Co-state critic ≈ direct critic (1.02–1.10 m4, 0.96–1.04 m64) | Frozen cells: **m4 teacher-only value** (~8% at hypothetical lookup price), **m64 inconclusive**. Neither is direct-sufficient at gate strength; the programme borrows no H2 claim from D4 |

## 3. Per-domain mapping

- **DROID-100 adaptive sensing: co-state advantage at benchmark strength; H2 INCONCLUSIVE
  (direct arm under-realised).** The B2.2 numbers stay visible — they are the strongest
  H2-adjacent signal in the programme — but without the §1893 convergence/capacity probes
  they cannot carry even a diagnostic deployable label. (Prior draft called this
  "deployable at benchmark strength"; withdrawn as internally inconsistent with its own
  §5 probe requirement.)
- **Spatial patch allocation: direct-sufficient.** No learned (incl. adjoint-featured)
  selector beats norm/fixed/random; exact gradients work but do not distil.
- **D4 simulation: m4 teacher-only value at a hypothetical price; m64 inconclusive.**

## 4. Programme class: INCONCLUSIVE — with teacher-only value as the stated hypothesis

"INCONCLUSIVE would erase the distillation failures" was a false dichotomy (withdrawn):
the plan's INCONCLUSIVE class is precisely what records diagnostic-strength failures
without a confirmatory test. The hypothesis it carries forward: exact co-state guidance
works everywhere affordable (B2 exact 0.00026; 6A exact ~2e-05; D4 lookup-price gains)
while no amortised adjoint allocator has beaten a *realised* direct critic anywhere.

What INCONCLUSIVE opens (roadmap letter): bounded direct rescue (≤20% extra compute —
here, the §1893 critic convergence/capacity probes on the B2 setup) and seed expansion;
no scale-up; N4 stays closed; production P5+ stays gated; paper track B waits on a
verdict it does not yet have. If the probes realise the critic and the B2 advantage
survives, the class upgrades through §5; if the critic saturates far below the co-state
head, that is itself the teacher-only evidence.

## 5. Upgrade path (not executed): confirmatory B2-sensing H2

A preregistered one-sided superiority test on frozen B2.2-style dynamics with a
saturation-probed direct critic (convergence + capacity probes per plan §1893), frozen
δ/w/alpha, episode-clustered inference. Immediate bounded precursor: run the §1893
probes (≤20% extra compute, no new architecture). Only a passed confirmatory test — not
more spatial or D4 runs — can move the programme class to deployable.

## 6. Deviations logged here (reviewer: advise if a DEV entry is required)

(a) G-H2 recorded at diagnostic strength although the plan specifies a confirmatory test.
(b) Single programme-level class assigned despite domain-split evidence; §3 is load-bearing.
(c) Critic realisation resolved as under-realised on the regret-collapse trigger without
the critic loss curve the readiness rule prefers — a judgement call, flagged so a future
probe can overturn it with data. (d) Draft cited the S4 regret number despite the audit's
blocking remediation; corrected to full exclusion before recording.
