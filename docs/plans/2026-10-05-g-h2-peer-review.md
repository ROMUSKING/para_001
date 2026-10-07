# Peer review: G-H2 decision draft (kilo, 2026-10-05)

**Lead:** opencode · **Critic:** kilo (plan agent, kilo-auto/free) ·
**Verdict:** CHANGES_REQUESTED · **Disposition:** all six findings accepted; draft
rewrote the class to INCONCLUSIVE, excluded S4, restored the D4 m64 cell, relabelled B2.2.
Recorded in `docs/plans/peer-critic-log.csv`. Reviewer path: agy 429-exhausted (twice),
cline free model retired; kilo reviewed (an earlier kilo dispatch timed out mid-read and
was not used). Raw event stream: session workspace `/tmp/opencode/kilo_gh2_review.jsonl`
(session `ses_ef263bee3ffe1K5uXKVk32zFY4`, not committed).

## Findings (verbatim substance)

1. **B2.2's critic flatline does NOT clearly satisfy the readiness rule.** Flatline at
   regret 0.18956 = worst static `always_mode3` (0.18959) is the collapse signature;
   the note reports regret flatness, not the critic's loss curve. The plan is
   internally split on realised vs under-realised; the note picked the favourable
   reading without citing either provision, while its own §5 demands the probes.
2. **TEACHER-ONLY is over-strong; INCONCLUSIVE with teacher-only as hypothesis.**
   No confirmatory test ran; the strongest domain is contested by Finding 1; the
   roadmap's D4-1 precedent records two classes per cell, so no single global class
   is forced.
3. **S4 policy-number reuse does not survive the audit; exclude.** Seed duplication
   (sign flips across teachers), tail-dominated headline, and the audit's explicit
   BLOCKING prohibition on citation — undodged by the "policy number" framing and an
   undeclared deviation.
4. **Domain-split incoherent as drafted:** §3 claimed deployable for B2.2 while §4
   assigned programme-level teacher-only (whose consequence is keep-the-critic).
5. **D4's two-cell verdict collapsed:** m64 = inconclusive dropped; synthesis claimed
   direct-sufficient where the artefact says inconclusive.
6. **Three undeclared deviations:** S4 citation; favourable critic-realisation
   resolution; D4 verdict collapse.

**What would clear it:** (1) exclude/reframe S4; (2) loss curve or §1893 probes, then
relabel B2.2; (3) record m64; (4) INCONCLUSIVE (or TEACHER-ONLY explicitly as
diagnostic-strength judgement); (5) log all deviations. No new runs except the probes.
