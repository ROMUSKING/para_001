# D4-1: a learned direct critic against a co-state critic on D4 (cells `m4` and `m64`)

**Written:** 2026-09-30, before any D4-1 critic was evaluated on a tuning or validation instance · **Redesigned:** 2026-09-30 after a smoke run on train and tuning instances, before any validation instance was generated (§10) · **Decision:** Roman, 2026-09-30 ("proceed with all practical CPU experiments"), on the rule that D4-3 froze · **Roadmap:** D4-1 · **Follows:** [D4-3](d4-3-plan.md) and its [note](../research-notes/2026-09-30-d4-3-varying-goal.md), which reopened D4-1 in `m4` and `m64` · **Precedence:** the governing protocol §5.5 (value-model factorial, information-equivalent direct baseline) and §5.5.1 (direct-baseline realisation), scaled down to NumPy on CPU; the cross-domain plan §5 (D4).

## 1. Question and scope

**Question.** D4-3 showed that a tabulated co-state weight helps a goal-blind cheap estimator when the goal varies (11 to 17 % less compute in `m4`, 8 to 18 % in `m64`) but that the exact table costs more to build per instance than it saves. H2's premise is that a *learned, amortised* co-state can deliver that weight for a new goal at a small price. D4-1 asks the H2-shaped question on the one domain where the co-state is known exactly: **for a goal the critic has not seen, does a value head that receives co-state features from a learned co-state estimator allocate refinement with less total compute than an information-equivalent direct value head that does not?**

**What it is.** An exploratory, validation-only design study (tuning and validation families; the test family is never generated) with learned critics in NumPy on CPU. A positive result is a statement about an analytic linear system whose co-state is linear in the goal. It is **not** evidence about real data or about H2 there (roadmap: "do not use D4 as evidence for H2 on real data").

**What it is not.** It does not test the exact table as a deployable arm (that is `cheap_adjoint`, D4-3, kept as a reference), and it does not price training in the gated comparison (training cost is reported and a break-even count is given).

## 2. Instances and data

- **Cells:** `m4` and `m64`, the two D4-3 candidates (same systems, pulses and initial states as D4-2 and D4-3). `base`, `depth7` and `depth11` are not run: R3v failed there in D4-3.
- **Families:** *train* (seed 4004, 30 instances) for critic training; *tuning* (seed 3003, 10) for `θ` and width; *validation* (seed 1001, 20) for the frozen rules. Tuning and validation instances are **D4-3's** (`highdim.varying_goal_instances`, one random unit goal per instance). **The validation instances have been read before, for the non-learned arms of D4-3; no learned critic has been evaluated on any of them.** The test family (seed 2002) stays unread and `cell_instances` refuses it.
- **Training rows.** The critics see meshes visited by three collection policies (`uniform_pass`, `residual` at `θ` = 0.85, `adjoint` at `θ` = 0.85, as in D4-2) on the train instances, with the train instance's own goal. For every visited state (at most 40 passes) up to 64 refinable intervals are sampled **once per cell** (collection seed 0, shared by all seeds); per seed, four fresh random unit goals per state are drawn from a stream seeded by the run seed, and each sampled interval takes one of them. At most 60,000 rows per cell are kept (a seeded random subset). A row is `(nine scalar interval features, the goal, the forcing discrepancy d and the solution-difference estimate e of the interval, the time of its right node)` with two labels (§3). **Goals are not a scarce resource** (thousands of distinct goals per cell); what is limited is capacity and row count.

## 3. Critics: components, inputs, labels, arms

**Labels (privileged, training only):**
- **Gain label (every value head):** `y = log10(|Λ_{j+1}(c)ᵀ τ_j| + 1e-15)` with the exact local error `τ_j` and the **discrete** co-state of the current mesh for goal `c` (the term of the objective's sum `Σ|Λᵀτ|`, as in D4-2). For the heads this is a **scalar measurement of the objective's own decrease** (what refining the interval would be worth), the analogue of a measured marginal gain.
- **Co-state label (estimator only):** the continuous co-state at the interval's right node, `Λ(t_b; c) = exp(Aᵀ(T − t_b)) c` (∂J/∂state of the underlying problem; a function of `(t_b, c)` only), from the system's propagator tensor, checked against `highdim.costate_table` to 1e-10.

**Co-state estimator `ŵ(t_b, c)`.** An MLP with two tanh layers of width 32 on `[t_b / T, c]` (standardised), output scaled by the mean training `‖Λ‖`. Trained on its own, before any value head, to minimise the relative error `mean ‖ŵ − Λ‖² / ‖Λ‖²` (Adam, learning rate 3e-3, batch 256, at most 200 epochs, patience 20, best epoch by the same error on held-out train instances). It is evaluated independently of the heads (governing §5.5). **Estimator floor (frozen):** relative error against the true co-state on the held-out train instances of at most 0.5; a (cell, seed) below the floor is flagged *estimator under-realised*.

**Value head.** An MLP with two tanh layers of width `H`, trained by `highdim.train_scorer` on the gain label (Adam, learning rate 3e-3, batch 1024, at most 40 epochs, patience 6, validation = 20 % of the *train instances*, best epoch by validation RMSE in `log10`). Inputs: the nine scalar interval features of D4-2, the goal `c`, the unit directions of `d_j` and `e_j` (`FEATURE_DIM + 3m` numbers, standardised); **the co-state arms add three co-state features** `[log10|ŵᵀd_j|, log10|ŵᵀe_j|, log10‖ŵ‖]`. Everything is P0 except the labels.

| Arm | Head inputs | Co-state source | What it is |
|---|---|---|---|
| **D** `direct` | P0 inputs | none | the information-equivalent direct marginal-gain critic (governing §5.5 "Direct") |
| **C** `costate_critic` | P0 inputs + 3 co-state features | the trained estimator | the deployable adjoint critic ("Full adjoint" with an amortised co-state) |
| **R** `costate_randomised` | as C | an estimator trained on co-state labels **permuted across rows** (seeded; also for its early stopping) | control: same architecture, scale and marginals, alignment with the objective broken (governing §5.5 "Randomized co-state") |
| **T** `teacher_feature` | as C | the exact co-state `Λ(t_b; c)` | privileged diagnostic ceiling for this head, **never a deployment result** (governing §5.5 "Frozen teacher feature") |

Information equivalence holds: D, C, R and T receive the same P0 inputs and the same gain labels; only C, R and T receive co-state-derived inputs, and only the estimators of C and R receive the co-state tensors as supervision. Parameter matching is by **width adjustment** (`critics.match_direct_hidden`): D's width is the integer that brings its parameter count closest to the co-state head's plus the estimator's; R and T use C's width. D therefore has at least as many parameters as C in total and is never starved of inputs.

**References (from D4-3's machinery, on the same instances):** `uniform_pass` (no scoring), `cheap` (goal-blind, no learning), `cheap_adjoint` (the exact table taken as given, hypothetical for varying goals), `adjoint_free` (hypothetical ceiling).

## 4. Prices

Every critic is priced in FLOPs converted to CN steps (`highdim.flops_step`), counted from its code path (`critics.flops_critic_per_interval`): the cheap-estimator features (as `flops_amortised_per_interval`), the unit directions and standardisation of the remaining inputs, the head, and for the co-state arms the estimator (when there is one) and the three co-state features. The notebook checks `flops_step` against an operation count and the tests check the critic formula against a count from the weights. Two scenarios, both from the same traces (`decision_scales`):

- **Real price (primary, gated).**
- **Lookup price (reported, hypothetical):** every critic, including D, is charged what `cheap_adjoint` is (features plus two dot products). It separates *ranking quality* from the price of a neural network. **It is labelled hypothetical wherever it appears.** T is reported at the lookup price only (its exact table has a per-instance setup that D4-3 priced at 880 to 14,075 CN steps).

Expected consequence, stated before any validation run: a CN step costs about 170 FLOPs in `m4` and about 16,500 in `m64`; D4-3's `cheap_adjoint` costs 0.60 and 0.08 of step doubling's price there (`results/runs/d4_3_varying_goal_20260930T084725Z/reports/acceptance_report.json`). A learned critic that takes the goal and two error directions as inputs has `FEATURE_DIM + 3m` inputs and costs several times that. **It will very probably fail the real-price gate in both cells.** The lookup-price analysis is what the comparison is informative for.

## 5. Training, sizes and the direct-baseline realisation protocol

- **Seeds:** five paired seeds, `0` to `4`. A seed fixes the goal draws, the row subsample, the validation split, every initial weight and every batch order; **the four arms of a seed see identical rows, split and head batch order.**
- **Width `H` per cell (scaled §5.5.1):** grid `{8, 16, 32}` for the co-state head; D's width is matched by parameters. With seed 0 only, train D and C at each `H`, run the tuning instances at `θ` = 0.95 and record the geometric-mean compute over tuning instances and targets **at the lookup price** (ranking quality, so that price does not shrink the direct baseline; log-log). `H*` is the smallest `H` whose **D** is within 2 % of the best D over the grid; C, R and T use the same `H*` (D's width matched). **Capacity probe:** if the best D is at `H` = 32, train D at `H` = 64 (matched C at 64); if D improves by at least 2 %, the envelope is raised to `H` = 64 for every arm, the probe stops there, and the cell is reported as *direct baseline not shown saturated* (no probe at 128), so a tie or a co-state win there is inconclusive.
- **`θ`** per (cell, arm, seed) on the tuning family from `{0.5, 0.7, 0.85, 0.95, 0.98, 0.995}`, at the real price for the gated comparison and at the lookup price for the hypothetical one (two separate tunings, each arm at its own best `θ`); a minimum on the grid's edge is flagged.
- **Diagnostics reported, not gated:** for each head the validation RMSE and within-state Spearman correlation with the gain label; for each estimator the relative error against the true co-state on held-out train instances and on its own (training) labels; parameter counts; FLOPs per interval and price in steps; training FLOPs in CN-step units and the break-even instance count against `uniform_pass` (`highdim.break_even_instances`).

## 6. Protocol

1. Generate train and tuning instances. Correctness checks before any training (recorded in `tests/correctness.json`): the propagator tensor equals `costate_table` for a goal; the exact-co-state features of T equal the terms of the `cheap_adjoint` score; the estimator's backpropagation equals finite differences; the discrete co-state for a goal equals `domain.costate` on an instance carrying that goal; `flops_step` equals the operations counted from `CrankNicolson.steps`; the test family is refused.
2. Collect training states once per cell; per seed, draw goals and labels; train the estimators (C, R), then the heads (D, C, R, T). Width selection and capacity probe as in §5 fix `H*`.
3. Tune `θ` per (cell, arm, seed) on the tuning family. **Freeze** `H*`, every `θ`, the rules and the comparison list, and write their SHA-256 (`reports/frozen_before_validation.json`).
4. Only then generate the validation family and run every arm and reference (real and lookup price, three interpolation methods from the same traces).

## 7. Frozen rules

Every comparison is the mean over instances of `log(compute_a / compute_b)` per target and seed, with the interval from a **two-stage bootstrap** (resample seeds with replacement, then instances within each drawn seed; 10,000 resamples, generator seed 0). "Beats" means the 95 % interval lies below 0 at **two adjacent targets** (10 %, 3 %, 1 % of the initial-to-finest gap; primary method log-log). Compute not reached within twice the finest grid counts as the cap and censored counts are reported. A rule is evaluated at the **real price** unless its name ends in `-L` (lookup price, hypothetical).

- **R0 (a learned critic pays):** C or D beats `uniform_pass`. If neither does, the cell's answer is "no learned critic pays at the real price".
- **RC1 / RC1-L (the H2-shaped test):** C beats D.
- **RC2 / RC2-L (direct utility sufficient):** at each of the three targets the 95 % interval of `log(C / D)` lies inside `±ln 1.05`. The 5 % margin is declared here from D4-3's scale (the exact table itself gave 8 to 18 % over `cheap`); it is not tuned.
- **RC3 / RC3-L (attribution):** C beats R. Without it an RC1 win is not attributed to the co-state.
- **RT-L (teacher ceiling):** T beats D at the lookup price: the exact co-state, given at lookup price, helps a learned head.
- **Estimator floor:** met in a (cell, seed) if the held-out relative error is at most 0.5. RC1 needs it in at least four of five seeds; otherwise C is *estimator under-realised*.
- **Reported, not gated:** every arm against `cheap`, `cheap_adjoint` and `adjoint_free`; the **retained fraction** of the headroom between `cheap` and `cheap_adjoint` for each critic (`highdim.retained_headroom`); the same rules under the semilog and staircase pricings; each seed alone; a t-interval over the five seed means as a check on the bootstrap.

**Exit classes (governing §0A.3), per cell.**

| Outcome | Class |
|---|---|
| R0, RC1 and RC3 hold at the real price, estimator floor met | deployable adjoint contribution, on this analytic domain only |
| RC1-L and RC3-L hold but not at the real price, estimator floor met | value conditional on a cheap critic (basis/domain-dependent effect); says nothing about a real critic's price |
| RT-L holds but C does not beat D (estimator under-realised, or RC1-L fails with the floor met) | teacher-only value: the exact co-state helps, a learned estimator does not deliver it |
| RC2-L holds and RT-L does not | direct utility sufficient: co-state features add nothing to a learned head here |
| RC1 or RC1-L holds but RC3 or RC3-L fails | effect not attributable to co-state alignment; no mechanism claim |
| the width probe did not show D saturated, or intervals are too wide for RC1 and RC2 | inconclusive because the direct baseline or the precision is insufficient |

**Decision rules.** A cell in class *deployable adjoint contribution* → write a confirmatory plan (test family, frozen before it is generated) and open D4 as a mechanism study; nothing follows for real data. *Direct utility sufficient* in both cells → record "direct utility sufficient" for D4 (roadmap stop-loss). Any other class → record it plainly; the note says what a follow-up would need and does not run it.

## 8. Artefacts

`src/adjointrwm/domains/critics.py` (propagator tensor, rows, estimator, heads, composite critic, policy, prices, statistics) and `tests/test_critics.py`; `scripts/d4_1_estimator_probe.py` (the design-time probe of §10); `notebooks/04-domains/d4_1_learned_critics.ipynb` (CPU); run directory `results/runs/d4_1_learned_critics_<UTC>/` imported with a README and a clean-worktree reproduction (model weights are small and are committed as JSON if under the size limit, otherwise their SHA-256); `scripts/d4_1_tables.py`; a research note.

## 9. Risks and limits

- **A generic estimator may not learn the co-state operator.** `Λ(t; c) = Φ(t) c` is a matrix-valued function of time; a small MLP on `[t, c]` must learn it from rows (§10 records what it reached on synthetic rows). If it does not, C has no usable co-state features and the cell falls in *teacher-only value* or *inconclusive*; a structured cotangent head (linear in `c`) would need per-interval work of order `m²` and is not tried.
- **Price.** The critics' inputs grow with `m`; at the real price they are several times `cheap_adjoint`'s. The real-price gate is expected to fail; the lookup-price rules are hypothetical.
- **Continuous versus discrete co-state.** The estimator's label is the continuous co-state; the gain label uses the mesh's discrete one. They differ on coarse meshes, and the same approximation sits in `cheap_adjoint`.
- **Linear in the goal.** `Λ(t; c)` is linear in `c`, so a critic that has learned the propagator generalises to every goal. A real system's co-state is not linear in anything the critic sees.
- **One system per cell**; `m` is confounded with the system draw (as in D4-2 and D4-3). Twenty validation instances and five seeds give wide intervals on effects of a few percent; the equivalence margin may not be resolvable.
- **Small networks and short head training** (at most 40 epochs; estimator at most 200). A direct baseline that needs much more capacity is reported as not shown saturated.
- **Training is not priced in the gated comparison.** With few instances the break-even count may exceed the number of instances one would ever solve; it is reported.
- **Validation reuse.** The validation instances were read in D4-3, which chose these cells; a positive result here is therefore design-level and needs the test family before it is stated as confirmatory.

## 10. Changes after this plan was written

- **2026-09-30, before any validation instance was generated (redesign of §3, §4, §5, §7).** The first version of this plan had four arms built on a *bilinear head* (`score = log10(|ŵᵀd| + |ŵᵀe|) + δ`, with `ŵ` and `δ` produced by one trunk; arms `direct_flat`, `direct_bilinear`, `costate_critic` with a joint scalar-plus-vector loss, and `costate_randomised`). A timing smoke of the first implementation on **train and tuning instances only** (30 train instances, three tuning instances at `θ` = 0.95, seed 0, hidden width 16; scratch script and logs not kept, numbers not evidence) showed that the bilinear arms fit the gain label worse than the flat network at equal parameters (validation RMSE in log10 units 0.73 to 0.80 in `m4` and 0.76 to 0.77 in `m64`, against 0.46 and 0.67 for the flat network; within-state Spearman correlation 0.32 to 0.37 and 0.22 to 0.23, against 0.69 and 0.36) and that the joint loss distorted the co-state estimate (relative error 6.5 in `m4` and 3.5 in `m64` for the co-state arm, against 1 for predicting zero). The co-state arms were therefore handicapped by their architecture in a way unrelated to the co-state, which would make a "direct utility sufficient" verdict uninformative. The design was replaced by the governing protocol's structure: the estimator is trained alone on the co-state label, and the value head receives the direct inputs **plus** co-state features, so the co-state arm has strictly more information than the direct arm. The smoke also measured prices: the critics cost about 10.7 CN steps per interval in `m4` and 0.78 in `m64`, against 1.19 and 0.163 for `cheap_adjoint` (this is the reason for §4's expectation that the real-price gate fails).
- **2026-09-30, before any validation instance was generated (estimator budget).** A probe on synthetic rows (`scripts/d4_1_estimator_probe.py`: right-node index and unit goal drawn at random, label the exact propagator; no allocation policy, no tuning or validation instance) showed that 40 epochs at batch 1024 do not train the estimator at all (relative error near 1 at width 16 and 32 in `m4`). The estimator budget was therefore fixed at 200 epochs, batch 256, width 32 (§3); the probe's output (`python scripts/d4_1_estimator_probe.py m4 m64`, relative error against the exact co-state on held-out train instances; 48,000 synthetic rows, seed 0):

  | Cell | Width | Epochs | Batch | Validation error | Training error |
  |---|---:|---:|---:|---:|---:|
  | `m4` | 32 | 40 | 1024 | 1.000 | 1.001 |
  | `m4` | 32 | 200 | 256 | 0.366 | 0.356 |
  | `m4` | 64 | 200 | 256 | 0.340 | 0.332 |
  | `m64` | 32 | 40 | 1024 | 0.986 | 0.982 |
  | `m64` | 32 | 200 | 256 | 0.657 | 0.652 |
  | `m64` | 64 | 200 | 256 | 0.605 | 0.602 |

  The frozen estimator floor (0.5) is therefore expected to be met in `m4` and **not** in `m64` with a generic MLP at this budget; the plan keeps the floor as written and treats an `m64` miss as a result (*estimator under-realised*), not as a reason to change the estimator after the fact.
- **Changes after the validation family was read for a critic:** none so far.
