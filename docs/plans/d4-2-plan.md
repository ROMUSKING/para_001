# Plan: D4-2, FLOP-priced scoring on a higher-dimensional system

**Written:** 2026-09-29 · **Status:** plan, before any code or run · **Owner:** Roman · **Track:** D ([`cross-domain-plan.md`](cross-domain-plan.md) §9) · **Follows:** [D4-0b note](../research-notes/2026-09-29-d4-0b-where-adaptivity-pays.md) · **Roadmap:** [§2.1](roadmap.md)

## 1. The question

D4-0b showed that with exact step-doubling scoring (2n or 3n CN steps per pass) adaptive refinement does not pay on `smooth`, pays only in the most localised family, and that the co-state weighting beats the cheap residual score only when scoring is re-priced to a quarter of its real cost or less. **D4-2 asks what scoring actually costs when it is done cheaply, and whether adaptive refinement pays at that price.**

Two cheap ways to score are tested: (a) a **hand-written indicator** from the forcing (no learning, no solves), with and without the co-state weight, and (b) a **learned amortised scorer** that imitates the exact co-state-weighted local error from features that are already computed or cheap to compute. Both are priced in floating-point operations, in the same units as the CN steps, so their price is a *measured ratio*, not an assumed scale. The dimension `m` of the system is the lever: a CN step costs about `4 m²` operations, while a scorer on a fixed feature vector costs roughly the same at any `m`, so the price ratio should fall as `m` grows (a hypothesis to test, not a result).

D4-2 does **not** compare a co-state-featured critic with a direct critic (that is D4-1) and says nothing about H2.

## 2. What is new since D4-0b

| Change | Why |
|---|---|
| Systems of dimension `m` = 4, 32, 64 (dense, stable, a fixed `A` and goal `c` per system; forcing pulses and initial state vary per instance) | Solve cost grows with `m`; the D4-0b system had `m` = 3 |
| One factor varied at a time from a base cell (D4-0b changed pulse width, amplitude and depth together) | Hypothesis 1 of the D4-0b note |
| Scoring priced in FLOPs (converted to CN steps), not in step-doubling steps | Measured price of a cheap scorer |
| Compute-to-target by **log-log** interpolation between passes for every policy (primary); semilog interpolation and the plain staircase as sensitivity | The D4-0b note found the staircase flatters adaptive policies and the semilog interpolation biases against uniform by up to 6.1 % |
| Wider marking-fraction grid `θ` ∈ {0.5, 0.7, 0.85, 0.95, 0.98, 0.995} | All nine tuned `θ` sat at the old grid's top (0.95) |
| A separate scorer-training family (seed 4004) besides the tuning (3003) and validation (1001) families | The scorer is learned, so training and tuning must be disjoint from validation, and from each other |

## 3. Systems and cells

`make_system(m, seed)`: `m/2` damped oscillators (frequencies `2π·U(1,4)`, damping ratio `U(0.02, 0.3)`) rotated by a random orthogonal matrix so `A` is dense; goal `c` is a random unit vector. **The goal is fixed per system**, so the co-state weight is a fixed function of time that a learned scorer can absorb; this is a deliberate simplification that D4-1 removes by varying `c` per instance. Instances vary in 2–4 Gaussian forcing pulses (time `U(0.05, 0.95)`, width in the cell's range, amplitude `±U(5, 40)` times the cell's scale, random unit direction) and `y0 ~ N(0, I)`. System seed is `7000 + m`. Every instance is generated from frozen seeds and is **not real data**.

| Cell | `m` | Pulse width | Amplitude scale | Depth (finest grid) |
|---|---:|---|---:|---:|
| `base` | 32 | 0.001–0.004 | 10 | 9 (8,192 intervals) |
| `m4` | 4 | as base | as base | as base |
| `m64` | 64 | as base | as base | as base |
| `wide` | 32 | 0.01–0.04 | as base | as base |
| `amp1` | 32 | as base | 1 | as base |
| `depth7` | 32 | as base | as base | 7 (2,048) |
| `depth11` | 32 | as base | as base | 11 (32,768) |

The initial mesh has 16 intervals. Families: scorer training (seed 4004, 30 instances per cell), tuning (3003, 10), validation (1001, 20). **The test family (2002) is never generated.**

## 4. Arms (all deployable, all pass-based with Dörfler marking except `uniform_pass`)

| Arm | Score | Scoring price per pass (CN-step units) |
|---|---|---|
| `uniform_pass` | refine every longest interval | 0 |
| `residual` | step-doubling `‖τ̂‖` | `2n` (as D4-0b) |
| `goal_local` | `|cᵀτ̂|` | `2n` |
| `adjoint` | `|Λᵀτ̂|`, co-state by an on-mesh backward sweep | `3n` |
| `indicator` | forcing quadrature discrepancy `‖d_j‖`, `d_j = h/6 (b_a + 4 b_m + b_b) − h/2 (b_a + b_b)`; **diagnostic only** (it cannot see the homogeneous error) | FLOPs / FLOPs per step |
| `cheap` | `‖d_j‖ + ‖e_j‖`, `e_j` the solution-difference estimate `(h³/2)` times the third divided difference of the numerical solution over four nodes | FLOPs / FLOPs per step |
| `cheap_adjoint` | `|Λ(t_b)ᵀ d_j| + |Λ(t_b)ᵀ e_j|` with a tabulated continuous co-state | FLOPs / FLOPs per step; the table is a one-off per-system cost, reported separately |
| `amortised` | MLP prediction of `|Λ_{j+1}ᵀ τ_j|` from local features (no co-state input) | FLOPs / FLOPs per step |
| `adjoint` at scale 0 (**reference, hypothetical**) | as `adjoint` with free scoring | 0; the ceiling that `amortised` tries to reach |

`n` is the number of intervals before the pass. The forcing-only `indicator` ignores the homogeneous part of the truncation error and is expected to stall; it is kept as a diagnostic, and `cheap` adds the solution-difference estimate so that the non-learned baseline is competent.

## 5. Ledger (FLOPs)

- **CN step:** `4 m²` (two matrix–vector products with cached operators) `+ 3 m` (combine) `+ 2 P (2 m + 8)` (two forcing evaluations for `P` pulses). The CN-step unit of D4-0 and D4-0b is kept, so every earlier number stays comparable; a scoring price is `FLOPs / flops_per_step`.
- **Indicator:** per interval, two forcing evaluations (mid-point and one node; nodes are shared by neighbours), `8 m + 1` for the discrepancy vector and its norm.
- **Solution-difference estimate:** `16 m + 1` per interval; `cheap` is the indicator plus this plus one addition.
- **Tabulated co-state weight:** `2 m` per dot product (two for `cheap_adjoint`, charged on top of the norms, so conservative). One-off setup per system: `n_fine` backward matrix–vector products.
- **Amortised scorer:** the indicator, three forcing norms (`6 m`), two solution norms (`5 m`), the difference estimate, eight logarithms, standardisation, and the network `2 (d_in H₁ + H₁ H₂ + H₂)` plus biases and tanh.
- **One-off costs** are reported separately in CN steps: the tabulated co-state (`n_fine` backward products, per system) and the scorer's training (`3 ×` forward FLOPs `×` samples `×` epochs actually run; generating the exact labels needs reference solutions and is **not** charged, which understates the scorer's one-off cost). The **break-even** number of instances is the one-off cost divided by the mean compute the arm saves per instance against `uniform_pass` at the tightest target; it is reported only where that saving is positive.
- Wall-clock timings of a step and of the scorer are printed as a sanity check and are **not** a claim (CPU sandbox).

## 6. Learned scorer

Inputs (9): `log(h/T)`, `t_mid/T`, `log ‖b‖` at the two ends and the middle of the interval, `log` of the indicator, `log(‖y_b − y_a‖/h)`, `log ‖y_a‖` and `log ‖e_j‖` (all with a small floor), standardised on the training set. No co-state enters as an input. Target: `log10(|Λ_{j+1}ᵀ τ_j| + 1e-15)` with the exact co-state and the exact local error (privileged, used as supervision only). Network: 9 → h → h → 1 with `h` ∈ {8, 16, 32} chosen per cell on the tuning family (at a fixed `θ = 0.95`; the price of a wider network is part of the trade-off), tanh, Adam (learning rate 3e-3), mean squared error, at most 200 epochs, early stopping (patience 20) on a held-out part of the training instances (split by instance). Trained on the meshes visited by `uniform_pass`, `residual`-marking and `adjoint`-marking at `θ = 0.85` on the training family. Deployment uses NumPy only, with the weights stored as JSON in the run folder.

Because the goal is fixed per system, the target depends on `t` only through a fixed function, and the network sees `t_mid`. **This scorer is an amortised adjoint scorer (supervised by the adjoint-weighted error), not a direct critic.** Naming it a co-state scorer is licensed by AGENTS.md rule 4 because its supervision is derived from `∂J/∂state`; a direct critic trained on the realised gain is D4-1.

## 7. Protocol and freeze points

1. **Design commit (this plan) and code commit with tests**, before any run.
2. **Correctness checks** (in the notebook, before results): FLOP formulas against counted operations on a tiny instance; tabulated co-state against the discrete co-state; indicator is zero for zero forcing; batch equals sequential; error representation on batch grids; reference converged.
3. **Scorer training** on the training family only; **`θ` tuning** on the tuning family only, per (cell, arm). Boundary optima are flagged.
4. **Freeze** the tuned `θ`, the trained weights and the decision rules by writing their hashes to the run before the validation family is generated.
5. **Validation** (20 instances per cell), at the real price. Reported: compute-to-target at 10 %, 3 % and 1 % of the initial-to-finest gap, geometric-mean ratios with instance-bootstrap CIs (10,000 resamples), log-log primary; semilog and staircase as sensitivity.
6. Clean-checkout re-execution, import into `results/runs/`, research note, claim check.

## 8. Frozen decision rules

Each "beats" means the CI of `mean(log(compute_a / compute_b))` lies below 0 at **two adjacent targets** (log-log primary).

- **R1, an amortised scorer pays:** `amortised` beats `uniform_pass`, at its FLOP price.
- **R3, the co-state weighting helps a cheap estimator:** `cheap_adjoint` beats `cheap`, at their FLOP prices. (This is a non-learned matched pair: the same estimator with and without the co-state weight.)
- **D4-1 candidate cell:** R1 and R3 both hold. A candidate cell fixes the regime for a D4-1 design, which is gated separately and reads the test family once.
- **Reported, not gated:** R2, the retained headroom `(log u − log a) / (log u − log f)` where `u`, `a`, `f` are the compute of `uniform_pass`, `amortised` and `adjoint` at scale 0 at the same target; the price ratio of `amortised` to the step-doubling scores; the training cost and break-even instances; the comparison of `amortised` with `cheap` and `cheap_adjoint`; the forcing-only `indicator` as a diagnostic.

## 9. What D4-2 can and cannot support

**Can:** the FLOP price of cheap scorers on this analytic family and how it changes with `m`; whether adaptive refinement at that price beats uniform refinement in each cell; whether the co-state weighting helps a cheap estimator; how much of the exact co-state's headroom an amortised scorer keeps.

**Cannot:** H2 (no direct critic, no learned co-state-featured critic); any claim about real data, other domains, or hardware (FLOPs are counted, not measured); any test-family result; the comparison with a direct critic when the goal varies per instance (D4-1).

## 10. Risks and stop-losses

| Risk | Signal | Response |
|---|---|---|
| The scorer does not generalise across instances | Validation-like held-out error on the scorer's own held-out instances is high; `amortised` is worse than `indicator` | Report it; do not tune on validation; a richer feature set is a new design |
| Every tuned `θ` is at the top of the new grid again | `selected_at_grid_edge` | Report the boundary optimum plainly; do not extend the grid after reading validation |
| No cell satisfies R1 | R1 false everywhere | D4-1 stays closed; record as a D4 negative result |
| The forcing indicator cannot see pulses narrower than the initial mesh | `indicator` needs more compute than uniform in narrow cells | A finding, reported; not a reason to add information the deployable policy lacks |
| Run time | The notebook exceeds a CPU session | Reduce instances per cell only in the design commit, before running |

## 11. Deliverables

`src/adjointrwm/domains/highdim.py` (systems, FLOP ledger, indicator and tabulated-co-state arms, amortised scorer and its training) with tests; `notebooks/04-domains/d4_2_flop_priced_scoring.ipynb` (CPU); an imported run; a research note; roadmap, plan, README, changelog and worklog updates.

## 12. Changes after this plan was written

Each change is dated and stated here; any change after the validation family was read would be called out as such.

- **2026-09-29, before any code ran:** the CN-step combine term is `3 m`, not `5 m`: the code adds two vectors, scales one and adds two (`3 m`); the plan's `5 m` overcounted.
- **2026-09-29, after a dry run on the `base` cell's train and tuning families only (validation and test never generated):** (1) the forcing-only `indicator` never reached even the 10 % target (it ignores the homogeneous error), so `cheap` (indicator plus a solution-difference estimate) and `cheap_adjoint` were added as the competent non-learned arms, R3 now uses that pair, and `indicator` is a diagnostic; `indicator_adjoint` stays in the code but is not an arm. (2) The scorer gets a ninth input, the log of the difference estimate, so that it sees what `cheap` sees. (3) A 32-unit hidden layer cost more in scoring than it saved, so the hidden size is chosen per cell from {8, 16, 32} on the tuning family, and the training defaults became learning rate 3e-3, at most 200 epochs, patience 20. These choices came from tuning-family compute only.
- **2026-09-29, before the real run:** a smoke test of the notebook with a tiny configuration (cells `m4` and `depth7`; 6, 3 and 4 instances; two `θ` values; hidden sizes 8 and 16; 12 to 15 passes) executed the validation stage for the first four validation instances of those two cells and printed the median compute of each arm, so **those numbers were seen**. They came from a different configuration than the real run, and no design choice was made or changed because of them. The smoke test found and fixed two notebook bugs (a hard-coded reference to the `m64` cell, and the wall-clock probe's key name). The real run regenerates everything with the full configuration.
- **2026-09-29, before the real run:** the run manifest is `config/instances_manifest.json` (seeds, counts and SHA-256 hashes) plus `config/instances.parquet` and `config/systems.parquet` (full parameters), not one pretty-printed JSON listing every instance, which would add tens of thousands of lines to a diff. Existing runs are unchanged.
- **2026-09-29 to 09-30, launch of the real run (no design change):** the first launch was started inside a tool call with a 600 s limit and was killed during the scorer-training stage (four of the seven scorers had been written), before the tuning freeze and before the validation family was generated. No validation number was seen from it. It was relaunched detached with unchanged code and configuration; that run (`d4_2_flop_scoring_20260929T233556Z`, package commit `b139a33`, clean tree) is the one imported. The test family was not generated in either launch.
