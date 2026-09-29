---
name: claim-check
description: Check every empirical claim in a paper draft, README, research note or status update against the committed evidence, and mark each one supported, unsupported, contradicted or wrong. Use before sharing or submitting any write-up, or when asked whether a claim is justified.
---

# Claim check

## Procedure

1. **Extract the claims.** List every sentence that asserts a result, comparison, number, capability or hardware need. Qualitative framing ("we propose…") is out of scope.
2. **Find the evidence for each claim**, in this order:
   1. `results/runs/*/`: acceptance reports, evaluations, trace summaries.
   2. `docs/research-notes/`.
   3. Anything under `docs/audits/` or `results/legacy/`. Evidence found only here is **not** support; it usually means the claim is unsupported.
3. **Classify each claim:**
   - **Supported:** the number matches a committed artefact, and the comparison has matched baselines and a CI.
   - **Plausible but unverified:** a correctness check exists somewhere but isn't committed.
   - **Unsupported:** no artefact.
   - **Contradicted:** an artefact shows otherwise.
   - **Wrong as stated:** logically or mathematically impossible, e.g. a lossless code beating the source entropy.
4. **Write the result as a table** with columns `# | claim (quoted) | evidence (path) | verdict | action`. Use `papers/drafts/REVIEW.md` as the template. For a paper, write to `papers/drafts/REVIEW.md` or a dated sibling; for anything else, write inline in your reply.
5. **Suggest a defensible rewrite** for each claim that isn't supported. Prefer "we observe X on run Y (CI …)" over removing the claim altogether.

## Watch for

- Precision without a CI, or a single seed described as "consistently" or "systematically".
- A gate that never fired being credited with a latency or benefit.
- Results from synthetic, legacy or smoke-test runs presented as real-data results.
- "Co-state" used for something that isn't ∂J/∂state (research plan §15.2, "Claims to avoid").
- Hardware claims (FP64 need, A100 need) that aren't backed by a profiler.
