# Plan: generalising the allocator across domains (Track D)

**Written:** 2026-09-29 · **Status:** D4 is the first domain (decided by Roman, 2026-09-29); the others are deferred (§1); D4-0 is run and imported (see the [note](../research-notes/2026-09-29-d4-0-adaptive-time-stepping.md)): the equal-compute condition failed, so D4-1 is closed; D4-0b (validation-only design study) is run and imported (see its [note](../research-notes/2026-09-29-d4-0b-where-adaptivity-pays.md)): no candidate regime at the real scoring price, two at hypothetical prices, so D4-1 stays closed at the real ledger; D4-2 (FLOP-priced scoring, seven cells) is run and imported (see its [note](../research-notes/2026-09-30-d4-2-flop-priced-scoring.md)): a learned scorer beats uniform refinement in 3 of 7 cells and the co-state weight helps a cheap estimator in 1, never in the same cell, so D4-1 stays closed; D1-0 (sensor streams on SMD) is run and imported (see its [note](../research-notes/2026-09-30-d1-0-sensor-opportunity.md)): the opportunity gate passes on a fidelity proxy that does not track the labelled endpoint; D4-3 (varying goal, seven cells) is run and imported (see its [note](../research-notes/2026-09-30-d4-3-varying-goal.md)): the co-state weight helps a goal-blind cheap estimator and adaptivity pays in `m4` and `m64`, so **D4-1 reopens in those two cells**, but only with the co-state table taken as given; D1-0b (native forecasting loss on SMD) is run and imported (see its [note](../research-notes/2026-09-30-d1-0b-forecast-sensing.md)): no deployable policy keeps the headroom, so D1-1 is not designed; D4-1 (a learned direct critic against a learned co-state critic in `m4` and `m64`) is run and imported (see its [note](../research-notes/2026-09-30-d4-1-learned-critics.md)): no learned critic pays at the real price and the co-state critic does not beat the direct critic · **Owner:** Roman
**Source:** Roman's domain-portfolio brief (chat, 2026-09-29), reconciled here with the governing plan · **Code:** `src/adjointrwm/domains/` · **Notebook:** [`04-domains/d4_adaptive_time_stepping.ipynb`](../../notebooks/04-domains/d4_adaptive_time_stepping.ipynb)

## 0. Thesis

The architecture is not fundamentally a signal filter. Its general role is to **allocate information, computation, sensing, communication or search effort to the parts of a structured problem that most affect a declared future objective**:

- the co-state estimates how sensitive the future cost is to each part of the state;
- the allocator spends precision or bandwidth only where that sensitivity, combined with a feasible candidate effect, buys a worthwhile improvement.

The design choices that transfer across domains are these:

- directional candidate effects, not raw gradient magnitude;
- a matched direct critic;
- causal teacher/deployment separation;
- typed external ports;
- expected value of information;
- complete rate and compute accounting.

**This plan does not add new theory.** The governing plan already contains the formal machinery:

| Brief concept | Governing plan |
|---|---|
| Domain-neutral allocation expression | §4.4 net score `S(u) = ĝ(u) − β_R ΔR_u − β_C ΔC_u − β_L ΔL_u − β_Q ΔQ_u − κ σ_g(u)` |
| Legal actions (refine, sample, query, join, stop, …) | §4.1 action set; §3.10.4 join lifecycle |
| Sensitivity is not value of information | §4.5 VOI and the `S_query` score |
| A domain's structure | §3.10.2 `DomainSpec` / `HierarchySpec`; §3.10.3 `JoinSpec` |
| Universal metrics | §2.3 regret, `AURC`, absolute adaptive gain, fraction of oracle advantage (with `δ_oracle_min`) |
| Negative-transfer test | §2.3.1 `TransferMacro`, `TransferWorst ≥ −m_domain` |
| Cross-domain phase | N6 (weeks 21–42) and `HJoinBench` (§5.9.1) |

The brief's own allocation and metric formulas did not survive pasting into the chat, so the governing plan's definitions are used throughout.

What this plan adds:

- a domain admission checklist;
- the portfolio, mapped to plan families and tiers;
- one code interface shared by every domain;
- a first exact-adjoint reference domain;
- a sequence that respects the existing gates.

## 1. Precedence and the one real conflict

- Nothing here changes a gate. A **confirmatory** cross-domain claim goes only through N6: a positive `TransferMacro`, `TransferWorst ≥ −m_domain`, and no privilege or exact-correspondence dependency.
- Real-domain adapters (time series, text, graphs, event logs) are **Tier P** under §5.9.1. They "cannot replace the exact synthetic join benchmark".
- Language enters "only through a frozen typed boundary with tokenization, encoder, context, latency, message, and rate costs" (N6). The LLM-context domain therefore uses a frozen LLM as its evaluator, and token counts go into the rate ledger.
- In every domain, the primary comparison stays **adjoint vs a realised direct critic** (§2.3, §5.5.1). Beating a heuristic is not the claim.
- **Conflict, resolved.** The brief ranks LLM context compression as the first major cross-domain experiment. The governing plan orders analytic correctness (N1), then matched joins (N2), then cross-domain work (N6).
  - **Decision (Roman, 2026-09-29): D4 first; "the rest subject to future research into permissively licenced content and testing".**
  - Read as: D1 (sensor streams), D2 (LLM context), D3 (graph), ERP and every Tier 2–3 domain wait until a licence survey has found permissively licensed data or models and a small test has shown they are usable. Nothing here is built for them until then.
  - The survey should record, per candidate source: licence text, redistribution terms, attribution requirements, and whether models trained on it can be released.
  - No amendment to N6 is made; no cross-domain claim is made before the N6 gate.

## 2. Domain admission checklist

A domain enters the programme only with a filled card covering every row below. The field names follow `DomainSpec`.

| # | Requirement | What to write down | Plan field |
|---|---|---|---|
| 1 | Structured state | Sequence, spatial hierarchy, graph, document tree, multiresolution field or entity set; how the hierarchy is built | `hierarchy_builder`, `node_schema` |
| 2 | Downstream objective | Lower-is-better `J`: prediction error, task success, control cost, retrieval accuracy, simulation error, loss | `native_endpoint` |
| 3 | Legal actions | Refine, sample, query, retain, compress, drop, simulate, expand, route, join; **hold and stop always** | `candidate_api` |
| 4 | Constrained resource | Tokens, bits, compute, latency, energy, memory, sensor bandwidth, human time, each mapped to the R/C/L/Q ledgers (§3.6–3.7) | `adapter_budget`, cost units |
| 5 | Measurable effects | Exact effect, or a counterfactual estimate of downstream benefit; oracle support `exact`, `bounded`, `approximate` or `none` | `candidate_api` |
| 6 | Baselines | Fixed, heuristic, uncertainty-only, direct critic, oracle | §13.2 of the production plan |
| 7 | Privilege | What is deployable (P0) and what is a label or target (P3) | `privilege_schema` |
| 8 | Data | Source, licence and provenance. Proprietary data never goes into git | manifest |
| 9 | Opportunity | The oracle beats the best fixed allocation by ≥ 15 % of the normalised area on validation; if not, redesign the candidates | §7.2B-1 |
| 10 | Regime | Does the key action refine known state (the first-order co-state score applies) or acquire unknown information (VOI, §4.5; a co-state alone is not enough)? | §4.4 / §4.5 |

## 3. Portfolio

Tiers follow the brief. The plan family is the §3.10.1 representation domain. "Regime" refers to row 10 of the checklist.

| Domain (brief) | Plan family | Main resources | Regime | Oracle | Tier / earliest phase | Notes |
|---|---|---|---|---|---|---|
| A. LLM context, memory, retrieval | event/token; language through a frozen boundary | tokens (R), LLM calls (C), latency (L) | keep, summarise, drop are refinements; retrieval is VOI | exhaustive on small contexts | 1 / Tier P pilot after D4-0 | exact token ledger; inspectable outputs |
| B. Code-repository context | relational + token | tokens, tool calls | joins (imports, calls, tests) | bounded; the test-suite verifier is expensive | 1 / Tier P | localisation accuracy is a cheaper proxy endpoint |
| C. Multivariate time series | temporal | samples, bits, energy | sampling is VOI; retention is a refinement | bounded | 1 / Tier P | closest to the original signal-filtering idea; the legacy filtering benchmark was audited as non-evidential, so start fresh |
| D. Image and video | spatial + temporal | bits, patches, compute | refinement | bounded | 2 | Track E (DROID) already covers part of it |
| E. Active sensing, fusion | typed ports | sensor power, latency | VOI (Track X) | enumeration or Monte Carlo | 2 | the domain where sensitivity ≠ VOI matters most |
| F. Scientific simulation | spatial / temporal | unknowns, solver steps | refinement (goal-oriented, adjoint-weighted) | exact on small cases | **1 / now (D4)** | cleanest test of the formal adjoint |
| G. Database query planning | relational | latency, memory | discrete graph actions | exact on small queries | 2 | gives "hierarchical join" a literal meaning |
| H. Network routing | relational / event | bandwidth, energy | mixed | bounded | 2–3 | |
| I. Multi-agent communication | peer ports (`request`) | bits, delay | VOI | bounded | 3 | decentralised credit assignment |
| J. Cybersecurity triage | event + graph | storage, analyst time (Q) | mixed | bounded | 2 | keep attack chains whole while compressing noise |
| K. ERP operations | event + graph | review time, queries | mixed | bounded | 2 | proprietary data (§8) |
| L. Healthcare | temporal / spatial | clinician time, device energy | mixed | expensive | 3 | advisory only; never autonomous |
| M. Finance | temporal / event | compute, latency | mixed | hard (non-stationary) | 3 | causal validation is difficult |
| N. Cloud scheduling, adaptive inference | event | compute, latency, energy | selective invocation (N4) | bounded | 2 | natural extension of the gate |
| O. Search and ranking | relational | compute, latency | mixed | bounded | 2 | |
| P. Genomics, molecules | relational / spatial hierarchy | lab cost, compute | mixed | expensive | 3 | counterfactual labels are costly |

## 4. What is shared and what is domain-specific

The pipeline is the same in every domain:

```
domain adapter -> canonical hierarchical state -> shared candidate/effect interface
               -> shared or partly shared adjoint allocator (plus the direct critic)
               -> domain-specific objective and action executor
```

**Shared:** hierarchy representation, budget representation, candidate schema, co-state estimator backbone, candidate-effect interface, direct-critic interface, uncertainty calibration, selective invocation, ledgers and trace format, stopping policy.

**Domain-specific:** observation encoder, action semantics, objective, safety rules, cost units, oracle or verifier, output renderer.

**The brief's `AllocationDomain` interface as implemented** in `adjointrwm.domains.AllocationDomain`:

| Brief method | Implementation | Why it differs |
|---|---|---|
| `encode_observation` + `build_hierarchy` | `initial_state(instance)` | One call builds the state the allocator acts on |
| `legal_candidates` | `legal_candidates(state)` | Hold and stop are generic, so domains don't list them |
| `predict_candidate_effect` | a **policy/scorer**, not the domain; optional exact effects via domain helpers | The effect model is what the adjoint and critic arms learn, so it cannot be part of the environment |
| `execute_candidate` | `apply(state, candidate)` | |
| `objective` | `objective(state, instance)` | May read targets (privileged). Deployable scorers never call it |
| `resource_cost` | `Candidate.cost` (R, C, L, Q) | Costs sit on the candidate and are totalled by the runner's ledger |
| `oracle` | generic `one_step_oracle` / `exhaustive_oracle` in `adjointrwm.domains.runner` | Written once, using `apply` and `objective` |
| `native_metrics` | `native_metrics(state, instance)` | |
| — | `observation(instance)` | The P0 view that deployable scorers receive |

**Three levels of generalisation**, each with its own test (§6):

1. **Architectural:** the same formal design works in another domain. Tested by specialists.
2. **Allocator transfer:** some allocator parameters transfer after the domain adapters are replaced. Tested by a shared backbone and by leave-one-domain-out.
3. **Zero-shot weight transfer:** a strong hypothesis. It is never assumed.

## 5. The four initial domains

### D4: adaptive time stepping for a linear dynamical system (implemented reference domain)

- **Why this and not mesh refinement.** In 1-D Poisson with linear elements, the dual-weighted-residual weight `z − I_h z` depends only on the local goal density. The adjoint allocator and a goal-local heuristic then coincide, so the benchmark cannot show what the co-state adds.

  In time stepping, a local error made early is carried forward, and damped, amplified or rotated, by the dynamics. The discrete co-state `λ_j = M_jᵀ λ_{j+1}` is exactly the recursion of comprehensive plan §4.2. For a linear system the error representation `c ᵀ(y(T) − y_N) = Σ_j λ_jᵀ τ_j` is exact, which makes a strong correctness test.
- **State:** a time grid over `[0, T]` built by dyadic bisection of an initial uniform grid, plus the Crank–Nicolson solution on it.
- **Candidates:** `refine(k)`, which bisects interval `k`, up to a maximum depth. Hold and stop are generic.
- **Objective:** the cancellation-free goal-oriented error `Σ_j |Λ_{j+1}ᵀ τ_{j+1}|`, an exact upper bound on the QoI error `|cᵀ(y(T) − y_N)|`, which is reported as a secondary metric. Lower is better.
  - Why not the QoI error itself: with signed local errors, a greedy oracle on `|cᵀ e_N|` won by finding lucky cancellations, not by allocating well (seen while building D4-0). The bound is the standard adaptivity target of dual-weighted-residual methods. `objective_kind='abs_error'` keeps the signed version available for sensitivity runs.
  - The exact solution comes from matrix-exponential propagation with 10-point Gauss–Legendre quadrature of the forcing on the finest grid.
- **Ledger:** rate = stored states (one per new node). Compute = CN steps.
  - `refine(j)` is charged the steps that must be re-solved, from the refined interval to the end (`n + 1 − j` for `n` intervals), because an early correction changes the whole downstream solution.
  - Scoring is charged at every decision: 2 steps per interval for the step-doubling estimate, plus 1 per interval for the backward co-state sweep.
  - Privileged references are not charged and are excluded from equal-compute comparisons.
  - **History:** the first execution charged 1 step per refinement. The audit that found this is described in the research note and the run README.
- **Policies in D4-0 (no learning):**

  | Policy | Uses | Deployable? |
  |---|---|---|
  | uniform | the longest interval first | yes |
  | random | uniform choice | yes |
  | residual | step-doubling local-error norm `‖τ̂_k‖` (goal-agnostic) | yes |
  | goal-local | `|cᵀ τ̂_k|` (goal projection, no propagation) | yes |
  | adjoint | `λ_kᵀ τ̂_k`, the discrete co-state computed from the known model | yes |
  | exact-τ adjoint | `λ_kᵀ τ_k` with the true local error | no: diagnostic |
  | one-step oracle | exact re-solve for every candidate | no |
  | exhaustive oracle | tiny cases only | no |

  The residual, goal-local and adjoint policies all use the same `τ̂`, so they differ only in how they weight it.
- **Instances:** a parametric family generated in the repo, like `HJoinBench-0`: damped or oscillatory dynamics, possibly one growing mode, Gaussian forcing pulses, a goal vector, and frozen seeds. It is an analytic benchmark and is never presented as real data.
- **Two comparisons, both reported.**
  - *Equal refinement count* (`AURC` over `B = 0..48`): the declared rate-budget opportunity gate and the weighting comparison.
  - *Equal total compute* (fixed levels 1000–8000 CN steps, frozen before the test family was read): each deployable policy against plain uniform refinement. Uniform refinement spends nothing on scoring, so it gets many more refinements.
  - Preview on the validation family (no test data): uniform refinement was ahead of every adaptive policy at equal compute, so **D4-1 opens only if the co-state policy beats uniform at equal compute on validation (CI below 0 at two adjacent levels)**. The plan's rate-budget gate alone is not enough, because comprehensive plan §2.3 requires matched `C_total`.
  - If the equal-compute check fails, the next step is a compute-efficient scoring scheme (mark many intervals per pass, incremental estimates, or a learned amortised scorer) validated at equal compute, not the learned-vs-critic comparison.
  - **D4-0b outcome** (validation only, [note](../research-notes/2026-09-29-d4-0b-where-adaptivity-pays.md)): pass-based marking does not make adaptive refinement pay at the real scoring price on the `smooth` family. It pays at the real price only in the most localised family (`sharper`), and there the residual score does at least as well as the co-state score. The frozen rule (co-state beats uniform, residual and goal-local at two adjacent targets) is met in 2 of 9 cells, both `sharp` at hypothetical scoring prices (×0.25 and ×0), where the co-state needs 9–29 % less compute than the residual score in point estimate. So D4-1 does not open at the real ledger; it may open for `sharp` once a learned scorer's price is *measured* at ×0.25 or less (D4-2).
  - **D4-2 outcome** (validation only, [note](../research-notes/2026-09-30-d4-2-flop-priced-scoring.md)): with scoring priced in FLOPs, a small learned scorer costs 0.245 of step-doubling scoring at `m` = 32 and 0.133 at `m` = 64 (2.24 at `m` = 4), and adaptive refinement beats uniform refinement in the three `m` = 32 cells with narrow pulses. A hand-built cheap estimator with no learning does at least as well in every cell, and the co-state weight helps it only at `m` = 64 (about 10 %). The frozen rules R1 (amortised beats uniform) and R3 (co-state weight helps the cheap estimator) hold together in no cell, so D4-1 stays closed. The goal is fixed per system in D4-2, so the note supports no statement about a varying goal.
  - **D4-3 outcome** (validation only, [note](../research-notes/2026-09-30-d4-3-varying-goal.md)): with a random unit goal per instance (same systems, pulses and initial states as D4-2), the tabulated co-state weight lets the goal-blind `cheap` estimator reach the targets with 11 to 17 % less compute in `m4` and 8 to 18 % less in `m64` (rule R3v), and the goal-aware arm beats uniform refinement in 5 of 7 cells (R4v); both hold in `m4` and `m64`, which are the D4-1 candidate cells. **The table was taken as given.** Building it for each instance's own goal costs 880 to 14,075 CN steps, more than the 341 to 741 steps the goal-aware arm needs to reach the 1 % target, so with the exact table charged per instance the arm needs 2.1 to 50 times `cheap`'s compute. With a goal per instance the co-state is worth having only if it is amortised across goals, and that is the question D4-1 asks.
  - **D4-1 outcome** (validation only, [note](../research-notes/2026-09-30-d4-1-learned-critics.md)): a co-state estimator trained on the co-state label and value heads trained on the gain label (direct; co-state features from the estimator; a randomised-label control; the exact co-state as a privileged feature), five paired seeds, information-equivalent inputs and width-matched by parameters. At the real price no learned critic beats uniform refinement in either cell (6.1 to 10.3 times its compute in `m4`, 1.5 to 1.8 in `m64`), as the plan expected from the networks' price. The co-state critic does not beat the direct critic (ratio 1.02 to 1.10 in `m4`, 0.96 to 1.04 in `m64`) and equals its randomised control. At a hypothetical lookup price a direct critic in `m4` needs 0.67 of uniform's compute at the 3 % target (keeping 0.66 of the headroom between `cheap` and the exact table), and the exact co-state as a feature helps the head by about 8 %; the learned estimator (error 0.43 against the exact co-state in `m4`, 0.67 in `m64`) does not deliver that. Exit classes: `m4` teacher-only value, `m64` inconclusive. D4 is not evidence for H2.
- **Caveat.** Here the discrete co-state is cheap and exact, so the adjoint arm is "Mode A" (computed), not amortised. The H2-relevant question is D4-1: a learned direct critic against a co-state-featured critic, both learned (plan §5.5 factorial).

### D2: LLM context tree (next; Tier P)

- **State:** a document or conversation tree of spans, sections and messages.
- **Candidates:** `KEEP_RAW`, `SUMMARIZE`, `DROP`, `EXPAND`, `RETRIEVE`, `JOIN(evidence)`, hold and stop.
- **Objective:** answer loss of a **frozen** open-weights LLM given the assembled context. The tokens used go in the R ledger; LLM calls and summariser calls in C.
- **Oracle:** exhaustive over subsets for small contexts, which makes exact regret possible.
- **Co-state:** the gradient of the answer loss with respect to span embeddings or soft-inclusion weights, taken through the frozen LLM. It is labelled a co-state only if it comes from ∂J/∂state (AGENTS.md rule 4); attention scores do not qualify.
- **Before building:**
  - choose the frozen model: it must fit an L4 in BF16, and its licence must be recorded;
  - choose QA dataset(s) with verified licences;
  - write the privilege schema: answers are P3.

### D1: multivariate sensor stream (Tier P)

- **State:** a channel × time hierarchy.
- **Candidates:** raise the sampling rate on an interval, open a channel, retain a band, query history, hold, stop.
- **Objective:** forecast or anomaly-detection loss at a downstream decision.
- **Regime:** sampling is VOI (§4.5), so this domain tests "sensitivity ≠ VOI".
- **Before building:** a public operational dataset with labels and a licence.
- **D1-0 outcome** (validation machines only, [note](../research-notes/2026-09-30-d1-0-sensor-opportunity.md)): on the Server Machine Dataset, opening a channel is allocated at the start of a 60-minute window after a snapshot of all channels; the privileged oracle beats the best fixed allocation by 0.920 [0.886, 0.949] of the normalised area (gate ≥ 0.15) and non-learned dynamic policies keep 0.42 to 0.49 of it, but the loss is fidelity to a frozen detector whose labelled F1 does not improve with sensing (0.259 hold, 0.192 full). "Sensitivity ≠ VOI" is not decided (no separation between sensitivity times volatility and either factor). D1-1 is not designed on this loss.
- **D1-0b outcome** (validation machines only, [note](../research-notes/2026-09-30-d1-0b-forecast-sensing.md)): with the loss changed to the native endpoint (a frozen per-machine ridge forecaster's 5-minute-ahead error), holding instead of observing raises the forecast error by 30 % on average over the validation machines (0.298 [0.023, 0.743]; one machine is at −0.174). The privileged oracle keeps almost all of it with a handful of channels (headroom over the best fixed allocation 0.875 [0.295, 1.347], an unstable ratio), but **neither deployable dynamic policy keeps any of the headroom** (retained −0.24 [−5.54, 1.28] and −0.10 [−5.04, 1.18]) and the best static policy beats both. The value-of-information comparison is inconclusive (every interval includes 0), and opening a few channels can make the forecast much worse than holding all of them. With a linear forecaster the sensitivity is a fixed matrix row, so a state-dependent co-state advantage cannot arise here; it would need a nonlinear downstream model. The note recommends not designing D1-1 on this domain.

### D3: graph domain (Tier P). **Which graph domain is open question 2.**

- **Default: code-repository context.**
  - Hierarchy: repository → module → function → block.
  - Joins: imports, calls and tests.
  - Objective: localisation accuracy (cheap), then patch/test success (expensive).
- **Alternative: database query planning.** It has exact oracles for small queries and a literal join structure.

## 6. Experimental ladder

Each rung opens only when the previous one passes. Every rung reports the primary comparison: adjoint vs realised direct critic.

| Rung | What | Gate to pass |
|---|---|---|
| 0. Correctness and opportunity (per domain) | Exact checks where available (D4: error representation, co-state vs finite differences); non-learned policies across budgets | Correctness tests pass; the opportunity gate (checklist row 9, uniform vs best-known curve) passes on validation instances |
| 1. Specialist (per domain) | Learned direct critic vs learned adjoint-featured critic; 5 paired seeds; `AURC` primary | The critic meets the §5.5.1 realisation floor; the result is classified as a §0A.3 exit class |
| 2. Shared allocator, separate adapters | One allocator backbone across ≥ 3 specialists that passed rung 1 | Per-domain non-inferiority to its specialist within `m_domain` |
| 3. Leave one domain out | Train on three domains, adapt minimally, test the fourth | `TransferMacro > 0` and `TransferWorst ≥ −m_domain` (§2.3.1) |
| 4. Hierarchy stress | Deeper trees, new branching factors, unseen budgets, candidate counts and join structures | Reported, not promoted |
| 5. Confirmatory cross-domain | Only inside N6, after the N2 exit class is recorded | The N6 exit gate |

**Negative transfer is always reported.** Every shared model must stay non-inferior to its own specialist on each domain. An aggregate gain must not hide one domain getting worse.

## 7. Universal metrics (implemented in `adjointrwm.domains.metrics`)

All metrics are lower-is-better, and each domain also keeps its native metric.

- **Oracle regret at budget `B`:** `J_method(B) − J_oracle(B)`.
  - When the exact oracle is intractable, the reference is the **best-known curve**: the pointwise minimum over every evaluated policy, privileged ones included.
  - The greedy one-step oracle alone is not enough: it can be beaten, which makes regret negative. The best-known curve upper-bounds `J*`, so regret against it is non-negative and a lower bound on true regret.
  - Reports list the policy set, because the best-known curve depends on it.
  - On tiny instances, `exhaustive_oracle_curve` measures the gap to the true oracle.
- **`AURC`:** the trapezoidal area under the regret curve over the frozen budget range.
- **Absolute adaptive gain:** `J_fixed(B) − J_method(B)`.
- **Fraction of oracle advantage:** `(J_fixed − J_method) / (J_fixed − J_oracle)`. It is reported as undefined when `J_fixed − J_oracle ≤ δ_oracle_min`, per plan §2.3.
- **Opportunity over budgets:** `∫(J_fixed − J_oracle) ≥ 0.15 ∫ J_fixed`, integrated over the budget range.
- **Transfer:** `TransferMacro = mean_d Δ_d` and `TransferWorst = min_d Δ_d`, where `Δ_d` is oriented so positive is better.
- **Secondary:** quality per bit, per ms and per joule; candidate ranking correlation and sign accuracy; calibration of predicted gain; fraction of expensive invocations avoided; p50/p95/p99 latency; peak memory; robustness to missing inputs.

## 8. Governance

- **Over-generalisation.** "The architecture works in domain X" requires X's specialist to pass rung 1. A Tier P result never rescues a failed gate elsewhere.
- **High-stakes domains.** Healthcare stays advisory, with conservative fallback. Finance and security results are exploratory until causal validation is designed.
- **Proprietary and business data (ERP, K).**
  - It never enters git.
  - No synthetic stand-in counts as evidence (AGENTS.md rule 1).
  - Connectors to business systems do not live in this repository (AGENTS.md, Scope).
- **Licences.** Every dataset and every frozen model gets a licence record before the first run (operator brief).

## 9. Milestones

| ID | Deliverable | Done when | Status (2026-09-29) |
|---|---|---|---|
| D0 | This plan; `adjointrwm.domains` (interface, runner, metrics); D4 domain with exact tests | `harness/check.py` passes | Done (this change) |
| D4-0 | `04-domains/d4_adaptive_time_stepping.ipynb`: correctness, rate-budget opportunity, weighting comparison, equal-compute check | Run directory imported; note written, including any failed gate | Done: correctness ✅, rate-budget opportunity ✅, equal-compute payoff ❌ (see the research note) |
| D4-0b | Pass-based (batch) allocation and a localisation sweep, validated at equal compute on validation only | Run imported; note written | Done: correctness ✅; 0 of 3 families pay for the co-state at the real price; 2 of 9 cells are candidates, both at hypothetical prices (sharp ×0.25, ×0) ([note](../research-notes/2026-09-29-d4-0b-where-adaptivity-pays.md)) |
| D4-2 | Price a learned, amortised scorer at its measured cost in the D4 ledger; higher-dimensional system; one factor varied at a time; wider `θ` grid (tuning family only) | Regime and price frozen before validation is read; run imported; note written | Done: correctness ✅; R1 in 3 of 7 cells, R3 in 1, none has both; a non-learned cheap estimator matches or beats the learned scorer in every cell ([plan](d4-2-plan.md), [note](../research-notes/2026-09-30-d4-2-flop-priced-scoring.md)) |
| D4-3 | Goal varying per instance; the non-learned pair `cheap` vs `cheap_adjoint` on validation, rule frozen before validation is generated | Regime and rule frozen first; note written | Done: correctness ✅; R3v in 2 of 7 cells (`m4`, `m64`), R4v in 5 of 7, both in `m4` and `m64`; with the exact table charged per instance the goal-aware arm needs 2.1 to 50 times `cheap`'s compute ([plan](d4-3-plan.md), [note](../research-notes/2026-09-30-d4-3-varying-goal.md)) |
| D4-1 | Learned direct critic vs co-state-featured critic on D4, 5 seeds, cells `m4` and `m64` | Rung-1 gate; §0A.3 exit class | Done: correctness ✅; R0 ❌ in both cells; co-state critic vs direct critic ❌ at both prices, no better than its randomised control; exact co-state feature helps the head ~8 % in `m4` at a hypothetical lookup price; exit class `m4` teacher-only value, `m64` inconclusive; test family unread ([plan](d4-1-plan.md), [note](../research-notes/2026-09-30-d4-1-learned-critics.md)) |
| DL | Licence survey of permissively licensed data and models for D1–D3 and Tier 2 domains | Per-source licence, redistribution and release terms recorded; a small usability test per candidate | **Pass 2 done** (network access widened): 59 sources, 17 `adopt`, 12 `adopt_with_conditions`, 5 `avoid`, 25 `unverified`. Licence candidates: D2 Qwen3-8B (Apache-2.0, licence file read) with MuSiQue and Qasper; D1 SMD, UCI electricity, Monash records; D3 code-repository context (query planning is closed by IMDb's non-commercial terms). Open gaps: QuALITY annotations, TriviaQA, TPC ([pass 2](../licences/survey-2026-09-29-pass2.md), [register](../licences/register.csv)) |
| D1-0 | Sensor-stream domain card, adapter and Rung-0 opportunity gate on SMD (validation machines only) | Run imported; note written | Done: correctness ✅; G1 ✅ robust (relative headroom 0.920 [0.886, 0.949]); non-learned deployable policies keep 0.42 to 0.49 of the headroom; the labelled F1 does not improve with sensing, so the proxy loss does not track the task ([plan](d1-0-plan.md), [note](../research-notes/2026-09-30-d1-0-sensor-opportunity.md)) |
| D1-0b | Redesign D1's objective: a frozen per-machine ridge forecaster's native 5-minute-ahead error; tuning machines first | Regime and rule frozen first; note written | Done: correctness ✅; G0 ✅ on tuning (0.522) and by the point rule on validation, not robust (0.298 [0.023, 0.743]); G1 passes by the frozen rule, unstable ratio (0.875 [0.295, 1.347]); neither deployable dynamic policy keeps any headroom (G2); D1-1 not designed ([plan](d1-0b-plan.md), [note](../research-notes/2026-09-30-d1-0b-forecast-sensing.md)) |
| D2-0, D3-0 | LLM-context and graph domain cards and adapters | Rung-0 gate | Candidates confirmed by Roman 2026-09-30 ("proceed as recommended"); **architecture codified:** dependency-aware DAGs, hot-swappable QLoRA adapters ($r=16$), discrete adjoint sensitivity packets ([plan](d2_d3_hierarchical_adjoint_plan.md)) |
| DX | Shared allocator, leave-one-domain-out, negative transfer | Rungs 2–3; confirmatory only in N6 | Waits for ≥ 3 specialists and the N2 exit class |

## 10. Decisions and open questions

**Decided (Roman, 2026-09-29):** D4 first. The rest is subject to future research into permissively licensed content and testing. This closes the former questions on the first experiment, D3, D2's model and data, ERP data, and amending N6: none of them is acted on until the licence survey (DL) reports.

**Decided (Roman, 2026-09-29, later):** licence survey first, then the rest.

**Decided (Roman, 2026-09-30):** "proceed as recommended": the survey's candidates are confirmed (D1 SMD; D2 Qwen3-8B with MuSiQue and Qasper; D3 code-repository context), D4-1 stayed closed at that point, and the real-data path (E1.1, then B2) stays on the critical path. D1-0 has been run (see §9).

**Decided (Roman, 2026-09-30, later):** redesign D1's objective (D1-0b), run the varying-goal check (D4-3), and proceed with all practical CPU experiments, flagging when to switch to Colab and on what hardware. D1-0b and D4-3 have been run; D4-3 reopened D4-1 in `m4` and `m64` (see §9).

**Open:**

1. ~~Should the licence survey (DL) be run next, or after D4-2 / D4-1?~~ Decided by Roman (2026-09-29): licence survey first, then the rest. Pass 1 and pass 2 are done.
2. ~~Should sensor streams (D1) also wait for DL?~~ They did; DL is done, SMD is licence-cleared and D1-0 has run.
3. ~~Should D1's objective be redesigned (D1-0b)?~~ Decided (Roman, 2026-09-30) and run with a forecasting objective. Result: no deployable policy keeps the headroom; the D1-0b note recommends not designing D1-1. Still open: should D1 be deprioritised for H2, and should the effort move to D2-0 and D3-0 (D2-0 needs an L4)?
4. ~~Should the varying-goal D4 check (D4-3) be run?~~ Decided (Roman, 2026-09-30) and run; it reopened D4-1 in `m4` and `m64`. D4-1 was planned, frozen and run (2026-09-30): no learned critic pays at the real price and the co-state critic does not beat the direct critic.
