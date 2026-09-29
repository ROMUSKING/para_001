# Plan: generalising the allocator across domains (Track D)

**Written:** 2026-09-29 · **Status:** proposed; one reference domain implemented (D4), nothing run yet · **Owner:** Roman
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
- **Conflict.** The brief ranks LLM context compression as the first major cross-domain experiment. The governing plan orders analytic correctness (N1), then matched joins (N2), then cross-domain work (N6).
  - **Proposed resolution:** D4, an exact-adjoint simulation, comes first as the correctness anchor.
  - D2, LLM context, is the first real-domain pilot. It is Tier P and exploratory, and may start after D4-0.
  - No cross-domain claim is made before the N6 gate.
  - Open question 1 asks Roman to confirm.

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
- **Ledger:** rate = stored states (one per new node); compute = solver steps, including the extra steps used to estimate local error; the co-state sweep is charged as backward steps.
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
| D4-0 | `04-domains/d4_adaptive_time_stepping.ipynb`: correctness and opportunity on the frozen instance family | Run directory imported; note written, including a failed opportunity gate if that happens | Notebook ready, not run (CPU only, no GPU needed) |
| D4-1 | Learned direct critic vs co-state-featured critic on D4, 5 seeds | Rung-1 gate; §0A.3 exit class | Waits for D4-0 |
| D2-0 | LLM-context domain card, frozen evaluator, exact token ledger, small-context oracle | Rung-0 gate | Waits for open question 3 |
| D1-0, D3-0 | Sensor-stream and graph domain cards and adapters | Rung-0 gate | Waits for data and licence choices |
| DX | Shared allocator, leave-one-domain-out, negative transfer | Rungs 2–3; confirmatory only in N6 | Waits for ≥ 3 specialists and the N2 exit class |

## 10. Open questions for Roman

1. **First experiment.** D4, the exact-adjoint simulation (assumed; consistent with the governing plan), or D2, LLM context (the brief's first pick)?
2. **D3.** Code-repository context (assumed) or database query planning?
3. **D2.** Which frozen LLM and which QA dataset(s)? They need an L4 fit and a recorded licence.
4. **ERP.** Is there a de-identified dataset licensed for research? Otherwise ERP stays out of research runs.
5. **Governing plan.** Should N6 be amended to name these domains? That needs your explicit instruction and a deviation-log entry.
