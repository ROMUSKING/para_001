# B2 direct-critic readiness probes (§1893): plan (DRAFT for peer review)

**Date:** 2026-10-05 · **Lead:** opencode · **Status:** DRAFT — no GPU until review recorded
AND the data path (§4) is resolved. No closed branch reopens.
**Goal:** decide whether the B2 direct critic is realised (converged + saturated) or
`UNDER_REALIZED_DUE_TO_COMPUTE_CONSTRAINTS`, using only the frozen B2 setup
(`scripts/run_allocator_optimization_benchmark.py`, config committed in
`results/runs/droid100_adjoint_v2_5seeds_20261001T080821Z/config/`).

## 1. Probe definitions (plan §1893 + tuning §1–5, adapted to 4-mode sensing)

Oriented validation endpoint: mean SmoothL1+CE critic surrogate AND validation regret,
evaluated on the script's `val_dataset` (built but never evaluated — code change:
evaluate val alongside test at checkpoints; no change to training semantics).
- **P0 floor:** critic beats `random_expected` and `uncertainty` on validation regret and
  shows positive adaptive gain (`always_mode0 − critic > 0`). B2.2 test numbers fail the
  uncertainty leg (0.18956 vs 0.18322); the probe re-tests on validation.
- **Amendment 2026-10-05 (copilot review, accepted):** the plan's floor rule requires two
  *adjacent budgets*. B2's counterpart is two adjacent cost rungs: the critic-vs-random
  and critic-vs-uncertainty comparisons are computed separately within low-cost
  (oracle-choice modes 0–1) and high-cost (modes 2–3) validation strata, each with a
  frozen paired test. A single pooled readout is a monitor, not the floor gate.
- **Val-discipline (copilot review, accepted):** all thresholds in §1 are frozen in this
  plan before any val read, so val reads are decision variables under a frozen rule —
  but no *further* selection (widths, checkpoints, rescue direction) may use val beyond
  the frozen P0/P1/P2 logic. Any additional val use needs a new reviewed amendment.
- **P1 convergence:** all 5 seeds finite; validation surrogate changes <1% relative over
  the final 20% of steps; 2×-step continuation (2500→5000, critic only) improves <2%.
- **P2 capacity:** 0.5×/1×/2× `DirectCritic` hidden width (via `matched_critic_hidden`
  scaling), 5 paired seeds, 2500 steps, critic-only (co-state head not retrained).
  If 2× improves the validation surrogate ≥2%, the envelope grows and the 4× diagnostic
  reference runs; otherwise saturation is recorded without it.
- **Readiness:** realised iff floor met, or (P1+P2 show saturation AND 4×/2× reference
  improves <2%). Else `UNDER_REALIZED_DUE_TO_COMPUTE_CONSTRAINTS`; H2 stays inconclusive.

## 2. Code changes (small, reviewed before run)

`scripts/run_allocator_optimization_benchmark.py`: `--critic-width-scale` (float, default
1.0); `--max-steps` reuse for 5000-continuation; per-200-step logging of train/val
critic surrogate to a jsonl ledger (the B2.2 loss curve exists only in lost stdout);
val-split policy evaluation at each checkpoint; per-seed finiteness tracking. Tests:
width-scaling parameter counts; ledger schema; val/test split disjointness.

## 3. Compute cap (prospective)

Critic-only, 5 seeds throughout. P2 widths (0.5×, 2×) at 2500 steps + P1 continuation to
5000 on the 1× critic: total ≈ 5×(2500+2500+2500+2500) = 50k head-steps vs B2.2's
2-heads×5×2500 = 25k head-steps of allocator training. **Framing correction (copilot
review, accepted):** this 2×-of-B2.2-allocator cap is a practical single-probe budget,
not a derivation of the plan's 20%-of-Tier-C rescue cap — Tier-C hours were never
quantified, so an exact mapping is impossible. Declared prospectively as: probe
allocator compute ≤ 2× the B2.2 allocator cost, L4 only, single session, stop on
completion or first hard failure. The 4× reference runs only if P2 triggers it, and only after the cap is
re-declared for it — it is not smuggled inside the 2× envelope.
**Implementation (2026-10-05):** `--critic-only` skips costate training/eval (separate
optimisers ⇒ identical critic trajectory); all probe runs use it, roughly halving
per-run cost against the cap arithmetic above. The 1× rerun reproduces B2.2's critic
trajectory exactly (same seeds/loader order) while adding the missing loss ledger and
val panels.
**Same-lineage note (copilot review, accepted with rebuttal):** the B2.2 loop gives each
head its own optimiser/scheduler with separate zero_grad/step on a shared batch stream
— no gradient interaction — so a critic-only rerun under identical seeds and loader
order reproduces the critic's exact training trajectory. Labelled same-lineage
diagnostic (same teachers, data, seeds); option (b) remains a different scientific object.

## 4. Data path (BLOCKER — resolve before provisioning)

The script reads the DROID-100 cache and 5 frozen teachers from Drive
(`drive_root/runs/<run_id>/{cache,jobs}`), unreachable from a headless CLI VM (Drive is
human-only; prior note). Committed locally: config, manifests, summaries, metrics —
no arrays, no checkpoints. Options: (a) user stages the Drive run folder (cache + 5×
`jobs/seed_*/dynamics/best.pt`) onto the VM; (b) full TFDS rebuild + 5-seed teacher
retrain (new benchmark lineage, not probes of B2 — changes the scientific meaning, needs
separate review). Recommended: (a); do not provision until staged.

## 5. No-go conditions

Critic realises AND B2 advantage survives → confirmatory-test proposal. Critic realises
and advantage vanishes → direct-sufficient in B2; programme class stands. Critic still
under-realised at cap → UNDER_REALIZED recorded, no further rescue without new approval.
