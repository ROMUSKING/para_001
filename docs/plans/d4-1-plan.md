# D4-1: a learned direct critic against a co-state critic on D4 (cells `m4` and `m64`)

**Written:** 2026-09-30, before any critic was trained and before any D4-1 code existed · **Decision:** Roman, 2026-09-30 ("proceed with all practical CPU experiments"), on the rule that D4-3 froze · **Roadmap:** D4-1 · **Follows:** [D4-3](d4-3-plan.md) and its [note](../research-notes/2026-09-30-d4-3-varying-goal.md), which reopened D4-1 in `m4` and `m64` · **Precedence:** the governing protocol §5.5 (value-model factorial, information-equivalent direct baseline) and §5.5.1 (direct-baseline realisation), scaled down to NumPy on CPU; the cross-domain plan §5 (D4).

## 1. Question and scope

**Question.** D4-3 showed that a tabulated co-state weight helps a goal-blind cheap estimator when the goal varies (11 to 17 % less compute in `m4`, 8 to 18 % in `m64`) but that the exact table costs more to build per instance than it saves. H2's premise is that a *learned, amortised* co-state can deliver that weight for a new goal at a small price. D4-1 asks the H2-shaped question on the one domain where the co-state is known exactly: **for a goal the critic has not seen, does a critic supervised by the co-state (∂J/∂state) allocate refinement with less total compute than a matched critic that learns the same allocation score directly from gain labels?**

**What it is.** An exploratory, validation-only design study (tuning and validation families; the test family is never generated) with learned critics in NumPy on CPU. A positive result is a statement about an analytic linear system whose co-state is linear in the goal. It is **not** evidence about real data or about H2 there (roadmap: "do not use D4 as evidence for H2 on real data").

**What it is not.** It does not test the exact table (that is `cheap_adjoint`, D4-3, kept as a *teacher* reference), and it does not price training in the gated comparison (training cost is reported and a break-even count is given).

## 2. Instances and data

- **Cells:** `m4` and `m64`, the two D4-3 candidates (same systems, pulses and initial states as D4-2 and D4-3). `base`, `depth7` and `depth11` are not run: R3v failed there in D4-3.
- **Families:** *train* (seed 4004, 30 instances) for critic training; *tuning* (seed 3003, 10) for `θ`, width and the direct-arm choice; *validation* (seed 1001, 20) for the frozen rules. Tuning and validation instances are **D4-3's** (`highdim.varying_goal_instances`, one random unit goal per instance). **The validation instances have been read before, for the non-learned arms of D4-3; no learned critic has been evaluated on any of them.** The test family (seed 2002) stays unread and `cell_instances` refuses it.
- **Training rows.** The critics see meshes visited by three collection policies (`uniform_pass`, `residual` at `θ` = 0.85, `adjoint` at `θ` = 0.85, as in D4-2) on the train instances, with the train instance's own goal. For every visited state (at most 40 passes) up to 64 refinable intervals are sampled **once per cell** (collection seed 0, shared by all seeds); per seed, four fresh random unit goals per state are drawn from a stream seeded by the run seed, and each sampled interval takes one of them. A row is `(interval features, direction and size of the two cheap error vectors, position, goal)` with two labels (§3). At most 60,000 rows per cell are kept (random subsample per seed). **Goals are therefore not a scarce resource** (thousands of distinct goals per cell); what is limited is critic capacity and row count.

## 3. Critics: inputs, labels, arms

**Inputs (identical for every arm):** the nine scalar interval features of D4-2 (`highdim.scorer_features`: step size, position, three forcing norms, forcing discrepancy norm, solution change, solution size, solution-difference norm), the goal `c` (`m` numbers), and the unit directions of the forcing discrepancy `d_j` and of the solution-difference estimate `e_j` (`2m` numbers). Everything is P0: no exact solution, no co-state, no step-doubling. Inputs are standardised with training-row statistics.

**Labels (privileged, training only):**
- **Gain label (every arm):** `y = log10(|Λ_{j+1}(c)ᵀ τ_j| + 1e-15)` with the exact local error `τ_j` and the **discrete** co-state of the current mesh for goal `c` (the term of the objective's sum `Σ|Λᵀτ|`, as in D4-2). For the direct arms this is a **scalar measurement of the objective's own decrease** (what refining the interval would be worth), the analogue of a measured marginal gain; D and B are never given `Λ` as a tensor, an input or an auxiliary target, which is what makes them direct. (D4-2's scorer used the same label, but as an amortised *adjoint* scorer because its score was trained to reproduce `Λᵀτ` for a fixed goal; here the distinction is made by what else is supervised.)
- **Co-state label (co-state arms only):** the continuous co-state at the interval's right node, `Λ(t_b; c) = exp(Aᵀ(T − t_b)) c` (∂J/∂state of the underlying problem; a function of `(t_b, c)` only). Computed from the system's propagator tensor, checked against `highdim.costate_table` to 1e-10.

**Architecture (shared trunk):** an MLP with two tanh hidden layers of width `H`. Two output forms:
- *flat* (D): one output, the predicted `log10` score;
- *bilinear* (B, C, R): `m + 1` outputs, a co-state estimate `ŵ` (scaled by a fixed constant, the mean training `‖Λ‖`) and a scalar correction `δ`; the predicted score is `log10(|ŵᵀd_j| + |ŵᵀe_j| + 1e-15) + δ`. With `ŵ = Λ` and `δ = 0` this is exactly the `cheap_adjoint` score (the teacher).

| Arm | Output form | Loss | What it isolates |
|---|---|---|---|
| **D** `direct_flat` | flat | scalar gain loss | the plain direct marginal-gain critic (governing §5.5 "Direct") |
| **B** `direct_bilinear` | bilinear | scalar gain loss only | the adjoint's *architecture* with **no** co-state supervision: the direct baseline's rescue variant |
| **C** `costate_critic` | bilinear | scalar gain loss + vector co-state loss | the deployable adjoint critic ("Full adjoint" with an amortised co-state) |
| **R** `costate_randomised` | bilinear | as C, with the co-state labels **permuted across rows** (seeded) | control: same architecture, scale and marginals, alignment with the objective broken (governing §5.5 "Randomized co-state") |

Losses: scalar gain loss = mean squared error of the predicted `log10` score against `y`, divided by the variance of `y`; vector loss = mean over rows of `‖ŵ − Λ‖² / ‖Λ‖²`. C and R minimise their sum with weight 1 on each (declared here, not tuned). Information equivalence holds: D, B, C and R receive the same inputs and the same gain labels; **C and R alone receive the co-state tensors** as supervision. Parameter matching is by **width adjustment**: B, C and R share one architecture (width `H`); D's width is the integer that brings its parameter count closest to theirs.

**References (from D4-3's machinery, on the same instances):** `uniform_pass` (no scoring), `cheap` (goal-blind, no learning), `cheap_adjoint` (the exact table taken as given: the **teacher**, hypothetical for varying goals), `adjoint_free` (hypothetical ceiling).

## 4. Prices

Every critic is priced in FLOPs converted to CN steps (`highdim.flops_step`), counted from its code path: the cheap-estimator features (`flops_cheap_per_interval`), the standardisation and logarithms as in `flops_amortised_per_interval`, the trunk (`flops_mlp_forward` with the arm's dimensions), and for the bilinear arms the two dot products, absolute values and the `10^δ` factor. The notebook checks the formulas against operation counts before any training. Two scenarios, both from the same traces (`decision_scales`):

- **Real price (primary, gated).**
- **Lookup price (reported, hypothetical):** each critic is charged as much per interval as `cheap_adjoint` (features plus two dot products), i.e. the price of an ideally amortised critic. It separates *ranking quality* from the price of a neural network. **It is labelled hypothetical wherever it appears.**

Expected consequence, stated before any run: in `m4` a CN step costs about 170 FLOPs and a hidden-16 network about a thousand, so D4-2's amortised scorer was already 2.24 times step doubling's price there; a learned critic will very probably fail the real-price gate in `m4`. The lookup-price analysis is what `m4` is informative for. In `m64` the network is a small fraction of a step.

## 5. Training, sizes and the direct-baseline realisation protocol

- **Optimiser (frozen):** Adam, learning rate 3e-3, batch 1024, at most 40 epochs, patience 6, 20 % of the *train instances* held out for validation (never a subset of a training instance's rows), best epoch by the **scalar validation RMSE in `log10` units for every arm** (one common surrogate).
- **Seeds:** five paired seeds, `0` to `4`. A seed fixes the goal draws, the row subsample, the validation split, the weight initialisation and the batch order (the meshes and intervals are collected once per cell and shared); **the four arms of a seed see identical rows, split, batch order and (for B, C, R) initial weights.**
- **Width `H` per cell (scaled §5.5.1):** grid `{8, 16, 32}`. With seed 0 only, train D and B at each `H` and measure the geometric-mean compute over tuning instances and the three targets at `θ` = 0.95 (real price, log-log). `H*` is the smallest `H` whose **better** of D and B is within 2 % of the best over the grid. All arms and seeds use `H*`. **Capacity probe:** if the best is at `H` = 32, train D and B at `H` = 64 (seed 0); if either improves the tuning compute by at least 2 %, the envelope is raised to `H` = 64 for every arm (the co-state arms get the same budget), and the probe is repeated at `H` = 128 only if the notebook's time cap allows (otherwise the cell is reported as *direct baseline not shown saturated*, and a tie or a co-state win there is inconclusive).
- **Direct baseline for the primary comparison:** the better of D and B by tuning-family geometric-mean compute at their own tuned `θ` (per cell, frozen before validation). The direct baseline is an adversary: the co-state critic is never compared with the weaker of the two.
- **`θ`** per (cell, arm, seed) on the tuning family from `{0.5, 0.7, 0.85, 0.95, 0.98, 0.995}`; a minimum on the grid's edge is flagged.
- **Diagnostics reported, not gated:** validation RMSE and Spearman correlation of the predicted score with the gain label within states; for C and R the relative co-state error; parameter counts; FLOPs per interval and price in steps; training FLOPs in CN-step units and the break-even instance count against `uniform_pass` (`highdim.break_even_instances`).

## 6. Protocol

1. Generate train and tuning instances. Correctness checks before any training (recorded in `tests/correctness.json`): the propagator tensor equals `costate_table` for a goal; the bilinear score with `ŵ = Λ`, `δ = 0` equals the `cheap_adjoint` score; the hand-written backpropagation equals finite differences for D and for the bilinear head; the FLOP formulas equal the counted operations; the discrete co-state for a goal equals `domain.costate` on an instance carrying that goal; the test family is refused.
2. Collect training states once per cell; per seed, draw goals and labels; train D, B, C, R for seed 0 at each width in the grid (D and B only for the width probe), fix `H*`, train all arms for all seeds.
3. Tune `θ` per (cell, arm, seed) on the tuning family; choose the direct baseline per cell. **Freeze** `H*`, every `θ`, the direct-baseline choice, the rules and the comparison list, and write their SHA-256 (`reports/frozen_before_validation.json`).
4. Only then generate the validation family and run every arm and reference (real and lookup price, three interpolation methods from the same traces).

## 7. Frozen rules

Every comparison is the mean over instances of `log(compute_a / compute_b)` per target and seed, with the interval from a **two-stage bootstrap** (resample seeds with replacement, then instances within each drawn seed; 10,000 resamples, generator seed 0). "Beats" means the 95 % interval lies below 0 at **two adjacent targets** (10 %, 3 %, 1 % of the initial-to-finest gap; primary method log-log). Compute not reached within twice the finest grid counts as the cap and censored counts are reported. All rules are evaluated at the **real price** unless marked otherwise.

- **R0 (a learned critic pays):** C or the direct baseline beats `uniform_pass`. If neither does, the cell's answer is "no learned critic pays at the real price".
- **RC1 (the H2-shaped test):** C beats the direct baseline.
- **RC2 (direct utility sufficient):** at each of the three targets the 95 % interval of `log(C / direct baseline)` lies inside `±ln 1.05`. The 5 % margin is declared here from D4-3's scale (the exact table itself gave 8 to 18 % over `cheap`); it is not tuned.
- **RC3 (attribution):** C beats R. Without it an RC1 win is not attributed to the co-state.
- **Reported, not gated:** C against B (co-state supervision at identical architecture) and B against D (architecture); every arm against `cheap`, `cheap_adjoint` and `adjoint_free`; the **retained fraction** of the headroom between `cheap` and `cheap_adjoint` for each critic (`highdim.retained_headroom`); the same rules at the lookup price and under the semilog and staircase pricings; each seed alone; a t-interval over the five seed means as a check on the bootstrap.

**Exit classes (governing §0A.3), per cell.**

| Outcome | Class |
|---|---|
| R0, RC1 and RC3 hold at the real price | deployable adjoint contribution, on this analytic domain only |
| RC1 and RC3 hold at the lookup price but not at the real price | value conditional on a cheap critic (basis/domain-dependent effect); says nothing about a real critic's price |
| RC2 holds (real price, and lookup price if the real-price R0 fails) | direct utility sufficient |
| `cheap_adjoint` beats every learned critic and none beats `uniform_pass` | teacher-only value |
| RC1 holds but RC3 fails | effect not attributable to co-state alignment; no mechanism claim |
| the width probe did not show the direct baseline saturated, or intervals are too wide for RC1 and RC2 | inconclusive because the direct baseline or the precision is insufficient |

**Decision rules.** A cell in class *deployable adjoint contribution* → write a confirmatory plan (test family, frozen before it is generated) and open D4 as a mechanism study; nothing follows for real data. *Direct utility sufficient* in both cells → record "direct utility sufficient" for D4 (roadmap stop-loss). Any other class → record it plainly; the note says what a follow-up would need and does not run it.

## 8. Artefacts

`src/adjointrwm/domains/critics.py` (propagator tensor, row builder, the critic model with hand-written backpropagation, training, policy, prices, `θ` tuning helpers) and `tests/test_critics.py`; `notebooks/04-domains/d4_1_learned_critics.ipynb` (CPU); run directory `results/runs/d4_1_learned_critics_<UTC>/` imported with a README and a clean-worktree reproduction (model weights are small and are committed as JSON if under the size limit); `scripts/d4_1_tables.py`; a research note.

## 9. Risks and limits

- **A structural head against a flat network.** The bilinear head encodes the `cheap_adjoint` form; a flat network must learn a product. Arm B keeps that architecture for the direct baseline, and the direct baseline is the better of D and B, so a co-state win cannot come from architecture alone. It may still come from the *estimator's* form (`|Λᵀd| + |Λᵀe|`), which D lacks; that is a property of the estimator, reported through B against D.
- **Continuous versus discrete co-state.** The co-state label is the continuous one; the gain label uses the mesh's discrete one. They differ on coarse meshes, and the same approximation sits in `cheap_adjoint`.
- **Linear in the goal.** `Λ(t; c)` is linear in `c`, so a critic that has learned the propagator generalises to every goal. A real system's co-state is not linear in anything the critic sees.
- **One system per cell**; `m` is confounded with the system draw (as in D4-2 and D4-3). Twenty validation instances and five seeds give wide intervals on effects of a few percent; the equivalence margin may not be resolvable.
- **Small networks and short training** (at most 40 epochs, width at most 32 unless the capacity probe raises it). A direct baseline that needs much more capacity is reported as not shown saturated.
- **Training is not priced in the gated comparison.** With few instances the break-even count may exceed the number of instances one would ever solve; it is reported.
- **Validation reuse.** The validation instances were read in D4-3, which chose these cells; a positive result here is therefore design-level and needs the test family before it is stated as confirmatory.

## 10. Changes after this plan was written

_None yet. Each change will be dated here, and any change made after the validation family was read for a critic will be marked as such._
