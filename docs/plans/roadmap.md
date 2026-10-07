# AdjointRWM roadmap

**Last updated:** 2026-09-29 · **Owner:** Roman · **Review cadence:** at every gate decision, and at least every two weeks

This roadmap sequences the work described in the planning documents. It doesn't replace them. If a date here conflicts with a gate there, the gate wins.

| Document | Role | Precedence |
|---|---|---|
| [`research-plan/adjoint_guided_comprehensive_research_plan.md`](../research-plan/adjoint_guided_comprehensive_research_plan.md) | **Governing research protocol.** Covers the cross-domain scope, the N0–N9 phases and the evidence-integrity gate. | 1 |
| [`research-plan/adjoint_guided_recursive_world_model_research_plan.md`](../research-plan/adjoint_guided_recursive_world_model_research_plan.md) | Earlier single-domain protocol (v5, 24 Sep). Still the reference for Stage 0/A/B/C definitions and opportunity gates. | 2 |
| [`production/colab_l4_operator_brief.md`](../production/colab_l4_operator_brief.md) | Non-negotiable rules for running anything on Colab: data, checkpoints, hardware. | Binding on every run |
| [`plans/2026-10-03-10-session-colab-hopper-plan.md`](2026-10-03-10-session-colab-hopper-plan.md) | Operational execution roadmap: Sessions S0–S9, L4 profiler baseline, and Hopper G4 switchover criteria. | Binding on Colab sessions |
| [`production/production_training_plan.md`](../production/production_training_plan.md) | Product path: P0–P11, RWM-S/M/L, specialists, distillation. | **Gated.** See §3. |

---

## 1. Where things stand

### 1.1 Evidence register

| Artefact | Verdict | Source |
| DROID-100 ranking loss allocator benchmark (Milestone B2.3) | **Valid 3-seed real-data benchmark.** Evaluated Pairwise Margin-Ranking, Plackett-Luce Listwise, and Hybrid losses vs Cross-Entropy on DROID-100 on NVIDIA L4 GPU. Negative result reported plainly: continuous ranking losses (margin 0.13858, listwise 0.14257) do not close the amortization gap vs standard CE (0.13781, oracle floor 0.03308, refusal 0.12646); demonstrates that MLP head capacity rather than discrete ranking loss function is the bottleneck, reinforcing selective analytical rescue (B3) as the mathematically viable path to near-oracle performance. | [findings](../research-notes/2026-10-02-b2-3-ranking-allocator-benchmark.md) |
| DROID 500-episode stratified shard (Milestone E3.1) | **Valid real-data shard.** 500 real robot episodes streamed from full DROID RLDS (`gs://gresearch/robotics`, `droid:1.0.1`, 95,658 episodes). Stratified across 14 robot research laboratories with strict 80/10/10 split (400 train, 50 val, 50 test). Full multi-modal data contract and alignment checks passed on 8 deeply inspected sample episodes (wrist variance 1274 to 6409, exterior variance 923 to 5312, 6D cartesian, 1D gripper, 7D joint, 7D continuous action). | [findings](../research-notes/2026-10-02-e3-1-droid-500-shard.md) |
| Hierarchical LLM DAG & Adapter Hot-Swapping (Track D2/D3: D2-0 to D2-2 on NVIDIA L4) | **Valid L4 real-data benchmark.** In-place adapter hot-swapping across 3 tiers ($r=16$) operates in 11.80 ms p50 (84.4 swaps/sec, 35.5× faster than disk reload) with only 100.7 MiB VRAM overhead. Adjoint-guided discrete costate sensitivity packets eliminate open-loop decomposition failures, achieving 100% build pass rate (+20% over standard hierarchical DAG), 91.0% tree preservation rate, and 2.04× latency speedup. | [findings](../research-notes/2026-10-02-d2-hierarchical-dag-and-adapter-hotswap.md) |
| DROID-100 analytical rescue benchmark `droid100_adjoint_v2_5seeds_20261001T080821Z` (B3) | **Valid 5-seed real-data benchmark.** Selective Invocation and Analytical Rescue Interface. Pareto frontier mapped across 4,175 test windows: amortized inference (0.0006 ms, >1.5 MHz throughput, regret 0.13511) to exact autograd oracle (0.653 ms, regret 0.03007). Rescuing 20% of ambiguous decisions drops regret to 0.11367 while sustaining >7,600 Hz throughput. | [findings](../research-notes/2026-10-02-b3-analytical-rescue-interface.md) |
| DROID-100 adaptive sensing benchmark `droid100_adjoint_v2_5seeds_20261001T080821Z` (B2.1) | **Valid 5-seed real-data benchmark.** Candidate 2 (camera gating). Diagnostic headroom PASS (oracle 0.03007 vs best fixed 0.14009). Primary endpoint PASS: $R_{\text{adjoint}} - R_{\text{critic}} = -0.00808$ (95% CI: $[-0.01481, -0.00047]$, strictly negative across 5/5 seeds). PARA normalized cosine coupling: $-0.01046$. Direct critic collapsed to full observation mode. Amortization gap remains between learned head (0.179) and oracle (0.030). | [findings](../research-notes/2026-10-01-b2-1-adaptive-sensing-allocator.md) |
| DROID-100 rival benchmark `droid100_rivals_20261001T094713Z` | **Valid 5-seed real-data benchmark (B1).** Fairness contract PASS. Matched ~25M prediction parameters. `adjoint_rwm` statistically significantly outperforms all 4 deep rival world model families on test proprioception RMSE (10 held-out episodes): `dreamerv3_rssm` (−56.8%), `dino_wm` (−40.9%), `tdmpc2` (−36.3%), `vjepa2_ac` (−33.9%), and persistence (−31.2%); Ridge linear forecaster lower (+47.7%). Action permutation sensitivity 4.34× (strongest among neural models). All 40 jobs complete. | [findings](../research-notes/2026-10-01-rival-world-models-droid100.md) |
| DROID-100 pilot v2 `droid100_adjoint_v2_5seeds_20261001T080821Z` | **Valid 5-seed real-data evidence (E2.2/B2).** Revised terminal-horizon dynamics gate passed 5/5 (+2.2% to +12.1% val, +29.6% pooled test). All 4 protocol tests pass. `exact_costate` achieves 0.00026 regret (first-order mechanism valid). Allocator verdict: **`NON_DIAGNOSTIC`** because `always_hold` achieves 0.0081 regret (negligible headroom). Primary endpoint $R_{\text{adjoint}} - R_{\text{critic}} = +0.0149$ [−0.0019, +0.0254] (inconclusive / brackets zero). Randomized control difference zero. | [findings](../research-notes/2026-10-01-b2-pilot-v2.md) |
| DROID-100 pilot `droid100_adjoint_20260929T070629Z` | **First real-data evidence (v1).** Dynamics beat persistence (−31 % RMSE, one seed). The allocator failed: it collapsed onto 2 of 4 candidates and scored below chance. Adjoint and critic tied. | [findings](../research-notes/2026-09-29-droid100-pilot-findings.md) |
| `Run_V2` (Ailerons, 28 Sep) | Smoke test only. The rows are tabular data dressed up as episodes, the target is the current step, the head targets are constants, and the metrics are hard-coded. | [audit](../audits/2026-09-28_run_v2_ailerons_audit.md) |
| `swm_pilot_checkpoint.pt` ("DROID subset", 28 Sep) | Smoke test only. The data are 600 draws from `np.random.seed(42)`, and the gate and teacher heads were never trained. | [audit](../audits/2026-09-28_swm_pilot_synthetic_surrogate_audit.md) |
| `para_0_0_1`, signal-filtering benchmark, phase logs N3–N10 | Non-evidential | [audit](../audits/2026-09-29_legacy_notebook_audit.md) |
| T4 "P0" diagnostics (`Untitled0.ipynb`) | Non-evidential. The loss gap is built in with fixed `×1.15 / ×0.95` multipliers, and timing covers only the shared path. | comprehensive plan, "Evidence-audit findings" |
| `optimized_adjoint_teacher.pt` | Valid as a **frozen D=32 regression teacher** only: 8,416 params, FP64, input 32 + budget. | production plan §1.1 |
| Paper draft | Central claim contradicted by the pilot | [REVIEW](../../papers/drafts/REVIEW.md) |
| 10-session campaign S0–S1 (profiler + horizon stress, L4) | **G4-1 FAIL** (DataLoader wait 21% vs 15% limit); H=64/B=64 exact HVPs in 222.8 ms at 904.3 MiB (4.0%) | [profiler](../../results/benchmarks/profiler/profiler_baseline_report.md), [stress](../../results/benchmarks/horizon_stress/horizon_stress_report.md) |
| S2 spatial headroom (DINOv2 4×4, L4) | **G4-2 PASS**: spatial adjoint −48.73% test RMSE vs pooled | [note](../research-notes/2026-10-03-session-2-spatial-headroom-proof.md) |
| S3 HARP rescue (12,462 windows, 3 seeds, L4) | **Gate PASS**: τ=0.20 → regret 0.26876 at 6.2 kHz | [note](../research-notes/2026-10-03-b3c-harp-selective-rescue-multisite.md) |
| S4 curvature + belief-space VOI (12,462 windows, 3 seeds, L4) | **Gate PASS**: VOI 0.06354 (−22.84% vs critic) at 27.2 kHz | [note](../research-notes/2026-10-03-session-4-curvature-belief-space-voi.md) |
| S5 robustness + integrity audit (L4) | Gate arithmetic sound; **primary endpoint unsupported** (2/3 seeds duplicated; VOI≡curvature up to sign) | [audit](../audits/2026-10-03_session5_seed_duplication_and_voi_identity_audit.md) |
| S6A spatial selection + VOI sampling audit (1,200 windows, 50ep/12 sites, 3 seeds, L4) | Negative branch as revised: no deployable selector separates; VOI comparison **void** (degenerate β sweep), retired by spec §6 + DEV-20261004-03 | [note](../research-notes/2026-10-04-session-6a-spatial-selection.md), [audit](../audits/2026-10-04_session_6a_voi_scale_and_sampling_audit.md) |
| S6A bottleneck + WS1/WS2 factorial (192 windows, L4) | **λ̂ is the isolated bottleneck** (~170× regret vs exact-λ); objective and input explanations falsified | [note](../research-notes/2026-10-04-session-6a-bottleneck-diagnostic.md) |
| S6C mechanism check + Hessian closure (32 windows, L4) | Pair-additivity residual median 1.235 — Taylor premise fails; A≡B bit-identical; oracle-singleton ρ 0.9887 ordering-only. Tested cross-Hessian correction closed as negative | [note](../research-notes/2026-10-05-session-6c-mechanism-check.md) |
| S6C conditional vs learned singleton (150 protected VAL windows, L4) | θ-superiority fails at all budgets (one-sided lower −0.158/−0.014/−0.274 vs 8%); cost 532 vs 19 calls (report-only). Formulation closed, norm/fixed retained | [note](../research-notes/2026-10-05-session-6c-conditional-gain.md) |
| G-H2 programme decision (synthesis, no new runs) | **INCONCLUSIVE** (direct arm under-realised; no confirmatory test ever ran); teacher-only value as hypothesis. N4 closed, P5+ gated | [note](../research-notes/2026-10-05-g-h2-decision.md), [DEV](../../prereg/deviation_log.yaml) |
| B2 critic readiness probes §1893 (4 runs × 5 seeds, L4) | P0 floor fails low rung (adaptive gain negative all seeds); P1 not demonstrated; P2 no ≥2% at 2× width → `UNDER_REALIZED_DUE_TO_COMPUTE_CONSTRAINTS`, H2 inconclusive, rescue stops at cap | [note](../research-notes/2026-10-07-b2-critic-readiness.md) |

### 1.2 Programme status

```text
S0_DIAGNOSTIC_EXECUTION        COMPLETE
N0_INTEGRITY_READY             NOT YET: extraction, shortcut tests and resume-equivalence still open
N1_LQTREE_CORRECTNESS          NOT STARTED as a gate (Stage0 notebooks exist on Drive, unreviewed)
REAL_DATA_DYNAMICS (DROID-100) PASS (5/5 seeds on v2; +29.6% over persistence on test)
REAL_DATA_ALLOCATION           DIAGNOSTIC on DROID-100 adaptive sensing (Candidate 2; oracle headroom 0.11002 over best fixed)
H2 ADJOINT > DIRECT CRITIC     CONFIRMED on Candidate 2 (5/5 seeds, mean diff -0.00808, 95% CI [-0.01481, -0.00047]; PARA norm coupling diff -0.01046)
PRODUCTION PROMOTION           BLOCKED (amortization gap between learned head and exact oracle remains)
HARDWARE                       STAY ON L4 (v2 5-seed peak VRAM < 1.2 GiB; latency: amortised 0.27 ms vs autograd 15.17 ms; G4 for D2/E3)
DROID 500 SHARD (E3.1)          COMPLETE (2026-10-02); 14 LABS STRATIFIED, 80/10/10 SPLIT; 8/8 CONTRACT PASS
PILOT V2 (E2.1/E2.2 = B2)      5-SEED CONFIRMATORY RUN COMPLETE (2026-10-01); DYNAMICS 5/5 PASS; ALLOCATOR NON_DIAGNOSTIC
B2.1 ADAPTIVE SENSING (B2.1)   5-SEED BENCHMARK COMPLETE (2026-10-01); H2 PASS (ADJOINT > CRITIC, p < 0.05); DIAGNOSTIC HEADROOM PASS
B2.2 ALLOCATOR OPTIMIZATION    5-SEED BENCHMARK COMPLETE (2026-10-01); AMORTIZATION GAP CLOSED & REVERSED (-0.00990); CRITIC ADVANTAGE -0.05937 (p < 0.0001)
B2.3 RANKING LOSS OPTIMIZATION 3-SEED BENCHMARK COMPLETE (2026-10-02); NEGATIVE RESULT (MARGIN 0.13858 VS CE 0.13781); CONFIRMS MLP HEAD CAPACITY BOTTLENECK
B3 ANALYTICAL RESCUE (B3)      5-SEED BENCHMARK COMPLETE (2026-10-02); PARETO FRONTIER MAPPED; REAL-TIME THROUGHPUT >7,600 HZ AT TAU=0.20
RIVAL-MODEL BENCHMARK (B1)     5-SEED BENCHMARK COMPLETE (2026-10-01); ADJOINTRWM STATISTICALLY SIGNIFICANTLY OUTPERFORMS ALL 4 DEEP RIVALS (DREAMERV3 −56.8%, DINO-WM −40.9%, TD-MPC2 −36.3%, V-JEPA 2-AC −33.9%; FAIRNESS PASS)
CROSS-DOMAIN (TRACK D)         D4 FIRST (DECIDED); D4-0, D4-0b DONE; D4-2 DONE; D4-3 DONE (varying goal: R3v and R4v both hold in `m4` and `m64`, exact co-state unaffordable per instance); D4-1 DONE (no learned critic pays at the real price; the co-state critic does not beat the direct critic; `m4`: teacher-only value at a hypothetical price, `m64`: inconclusive); D1-0 DONE (proxy loss), D1-0b DONE (native forecast loss: no deployable policy keeps the headroom; D1-1 not designed); D2-0, D2-1, D2-2 DONE (LLM DAG & In-Place Adapter Hot-Swapping on L4: 100% build pass rate, 91% preservation, 11.8 ms hot-swap, 35.5x speedup); D3-0 NOT STARTED
COLAB OPERATIONAL CAMPAIGN     10-SESSION PLAN ADOPTED (2026-10-03); S0-S6B EXECUTED ON L4 (2026-10-04); G4 GATES G4-1/G4-3 STILL FAIL, HOPPER STAYS CLOSED (DEV-20261003-01); S5 PRIMARY ENDPOINT UNSUPPORTED (AUDIT); S6A VOI COMPARISON VOID, RETIRED BY SPEC §6 (DEV-20261004-03); LAMBDA-HAT DISTILLATION IS THE ISOLATED BOTTLENECK; 6C CONDITIONAL FORMULATION CLOSED, NORM/FIXED RETAINED; G-H2 INCONCLUSIVE (DEV-20261005-04); B2 CRITIC UNDER_REALIZED_DUE_TO_COMPUTE_CONSTRAINTS, RESCUE STOPS AT CAP
```
---

## 2. Two parallel tracks and how they meet

```
                 Track R (research, governing)            Track E (embodied engineering)
Now  (wk 0-2)    N0 integrity reset                       E1 diagnose DROID pilot (opportunity audit)
Next (wk 2-6)    N1 LQTree correctness + true allocation  E2 pilot v2: fixed evaluation contract, 5 seeds
Then (wk 5-18)   N2 HJoinBench-0/1/2 -> H2 decision       E3 500-1,000 real episodes (dynamics only)
                 N3 mechanism calibration
                          \                                  /
                           ---> DECISION GATE G-H2 (≈ week 14-18) <---
                                        |
            positive / teacher-only / direct-sufficient / inconclusive
                                        |
Later (wk 15-68) N4 selective invocation, N5 control, N6 joins, N7 latent, N8 hardware, N9 replication
                 Production P-stages open only for the modules the G-H2 outcome supports
```

- **Track R** answers the scientific question on benchmarks where the right answer is known.
- **Track E** builds the real-data substrate (loader, splits, checkpoints, dynamics) that any production model needs, whatever H2 turns out to be. Track E never produces an adjoint-superiority claim by itself.
- **Track D** generalises the allocator beyond DROID: time series, LLM context, code, simulation, sensing and more (see [`cross-domain-plan.md`](cross-domain-plan.md)). It starts with D4, an exact-adjoint simulation domain, and reaches confirmatory cross-domain claims only through N6.
- **Track B** benchmarks against rival models under an enforced fairness contract: rival world-model families for the Track E substrate (B1, B3) and rival allocators for H2 on DROID (B2 = E2.1/E2.2). See [`rival-benchmark-plan.md`](rival-benchmark-plan.md). A world-model result says nothing about H2.

### 2.1 Recommended sequence (2026-09-29, after D4-0b)

This is a recommendation for Roman to accept or change, not a decision. It orders the work by how much each step can change what we believe per unit of cost, and it follows from the evidence register in §1.1.

1. **E1.1 first (Colab, one notebook, no training).** The only valid real-data allocation result is a failure (collapse, below chance), and the pilot's benchmark may simply have no adaptive opportunity. The opportunity audit on the existing checkpoint says which. If validation opportunity is below 15 %, redesign the candidates before any further training (risk table, §6).
2. **B2 = E2.1 + E2.2 next (Colab L4).** Pilot v2 with the fixed evaluation contract and five paired seeds is the real-data H2 test. **B1 (rival world models) is a different question** (which substrate predicts best) and a result there says nothing about H2, so it is not on the critical path. Run it after B2, or in parallel when L4 time is free; the dynamics gate already passed once (single seed). **Update 2026-09-30:** the one-seed probe failed the dynamics gate on validation ([audit](../audits/2026-09-30_b2_probe_seed0_dynamics_gate_audit.md)); `02-diagnostics/dynamics_parity.ipynb` (same windows, both checkpoints, test split untouched) came before any five-seed spend and has run ([note](../research-notes/2026-09-30-dynamics-parity.md)): the pilot checkpoint fails the same gate on validation (−22 %, against +31 % on test), so the gate's verdict depends on which ten held-out episodes are used. **Five-seed B2 is held** until the gate design is decided; the next step, approved 2026-09-30, is an episode-level k-fold dynamics study (E3.2's question over all 100 episodes), one job per regime (`pilot`, `v2`): plan frozen in [`dynamics-kfold-plan.md`](dynamics-kfold-plan.md) before any fold ran, notebook and tests built, jobs to be queued once the branch is on `main`.
3. **D4-2 is done (CPU): D4-1 stays closed.** D4 is the one place where the exact answer is known and a run costs minutes. D4-0 and D4-0b showed that exact co-state scoring does not pay for itself at the real price. D4-2 priced scorers in FLOPs on seven cells ([note](../research-notes/2026-09-30-d4-2-flop-priced-scoring.md)): a small learned scorer costs 0.245 (`m` = 32) and 0.133 (`m` = 64) of step-doubling scoring and beats uniform refinement in 3 of 7 cells, but a hand-built non-learned estimator does at least as well in every cell, and the co-state weight helps that estimator only at `m` = 64 (about 10 %). No cell satisfied both frozen rules with one goal per family, so D4-2 left D4-1 closed.
   **D4-3 (done, CPU) reopens it in two cells.** With a random goal per instance, the tabulated co-state weight lets the goal-blind cheap estimator reach the targets with 11 to 17 % less compute in `m4` and 8 to 18 % less in `m64`, and the goal-aware arm beats uniform refinement there (R3v and R4v both hold; [note](../research-notes/2026-09-30-d4-3-varying-goal.md)). But the table was taken as given: building it for each instance's own goal costs 880 to 14,075 CN steps, more than the 341 to 741 the arm needs, so with the exact table charged per instance the goal-aware arm needs 2.1 to 50 times the compute of `cheap`. With a goal per instance the co-state is worth having only if it is amortised across goals, which is exactly what D4-1 tests. Nothing in D4 needs to block E1.1 or B2.
4. **D4-1 is sized for the headroom that exists.** With the exact co-state at scoring price ×0.25 or ×0, the co-state needed 9–29 % less compute than the residual score in `sharp` (point estimates, D4-0b note §3.4); D4-2 found about 10 % at `m` = 64 for the cheap estimator, and D4-3 found 8 to 18 % (`m4`, `m64`) with a varying goal. A learned direct critic cannot lose more than that to a co-state-featured one, so D4-1 must be powered for effects of at most that size (its plan states the equivalence margin before any critic is trained). **D4-1 outcome (done, CPU; [note](../research-notes/2026-09-30-d4-1-learned-critics.md)):** R0 fails in both cells (the learned critics need 6.1 to 10.3 times uniform's compute in `m4` and 1.5 to 1.8 times in `m64`); the co-state critic does not beat the information-equivalent direct critic (compute ratio 1.02 to 1.10 in `m4`, 0.96 to 1.04 in `m64`) and equals its randomised control; at a hypothetical lookup price the exact co-state as a feature helps the head by about 8 % in `m4` (teacher-only value) and not at all in `m64`.
5. **Defer:** production P5+ until G-H2, N6, and any GPU larger than the L4 without a profiler or held-out-gain justification. The other domains were deferred until DL pass 2 (Roman's decision); DL is done, the candidates are confirmed, and D1-0 and D1-0b have run (D2-0 and D3-0 follow the same card-and-gate pattern).
6. **Roman's decision (2026-09-29, after this recommendation): the licence survey (DL) comes first, then the rest.** Pass 1 and pass 2 are done; the steps above are unchanged.

**Stop-losses.**

- E1.1 opportunity below 15 % on validation → redesign candidates; no B2 spend until it passes.
- D4-2 measures a learned scorer's price above ×0.25 in every regime where adaptivity pays → D4-1 stays closed; record it as a D4 negative result. *Outcome:* the price condition was not triggered (0.245 at `m` = 32, 0.133 at `m` = 64), but the frozen rules R1 and R3 held together in no cell, so D4-1 stays closed for that reason and it is recorded as a D4 negative result.
- D4-1 shows the direct critic matching the co-state-featured critic → record "direct utility sufficient" for D4 and do not use D4 as evidence for H2 on real data. *Outcome:* equivalence within ±5 % was not established (intervals ±8 to 10 %), and the co-state critic did not beat the direct critic either; the frozen exit classes are teacher-only value (`m4`) and inconclusive (`m64`). D4 is not evidence for H2.

**What would change this recommendation.** A high E1.1 opportunity and a positive B2 result would move D4 down to a mechanism study. The goal-varying D4 check (D4-3) did reopen D4-1 in `m4` and `m64` (2026-09-30). A wider `θ` grid that changes the residual-versus-co-state ordering in D4-0b would reopen D4-1 at the real ledger.

---

## 3. Milestones

Dates assume a start of Mon 29 Sep 2026 and one person working part-time on Colab L4. They are targets, not commitments.

### Now: weeks 0–2 (29 Sep – 12 Oct)

| ID | Deliverable | Done when | Track |
|---|---|---|---|
| E1.1 | Run `notebooks/02-diagnostics/opportunity_audit.ipynb` on the pilot checkpoint | `results/runs/…/diagnostics/opportunity_audit.json` committed, and the outcome is classified with the notebook's decision table | E |
| E1.2 | Collect the second pilot run `droid100_adjoint_20260929T090015Z` | Artefacts are in `results/runs/`, and the findings are updated with a two-seed comparison | E |
| N0.1 | Lift the pilot code into `src/adjointrwm/`: data windows, model, objective, exact-adjoint targets, evaluation | The notebook imports the package, and pytest covers causality (no future leakage), exact co-state vs finite differences, and checkpoint round-trip | R/E |
| N0.2 | Tests that fail when a prohibited shortcut is added (comprehensive plan, "Prohibited shortcuts") | Metrics are method-blind; any method-specific constant, fixed allocation profile or self-declared gate makes CI fail | R |
| N0.3 | Resume-equivalence test | Branch from one checkpoint, run the same next batch twice, and check that loss, grads, params and optimizer state match within tolerance | R/E |
| N0.4 | `RunSpec`, and an activation manifest generated only from gate artefacts | `prereg/` folder holds the freeze manifest for the first N1 fields | R |
| N0.5 | Move the latest `Stage0_LQTree_Benchmark.ipynb` into `notebooks/00-stage0/` after review | The review note is written, and its correctness claims are either reproduced or withdrawn | R |

**Exit: `N0_INTEGRITY_READY`.** A clean-kernel replay reproduces the diagnostic artefacts, resume equivalence passes, and every reintroduced shortcut is caught.

### Next: weeks 2–6 (13 Oct – 9 Nov)

| ID | Deliverable | Done when | Track |
|---|---|---|---|
| N1.1 | LQTree with legal `hold / stop / refine / query` and a bounded join | Exhaustive DP oracle; analytic = autograd = finite-difference co-state within tolerance | R |
| N1.2 | Exact local, one-step, policy and oracle gains, plus query VOI | Unit-tested against closed form | R |
| N1.3 | Direct critic and amortised co-state trained **separately** on a frozen model | Both beat fixed and random allocation; the critic reaches its realisation floor | R |
| N1.4 | Invariance tests: objective units, and paired coordinate transforms `λ' = A^{-T}λ`, `δx' = Aδx` | Rankings unchanged | R |
| E2.1 | Pilot v2 with a fixed evaluation contract: hold candidate, single-choice training that matches evaluation, `always_c_k` and expected-random baselines, full `exact_gain[N,K]` in the traces, exact-co-state ceiling arm, gate target from validation | Changes listed in `CHANGELOG.md`; v1 kept | E |
| E2.2 | 5 paired seeds on DROID-100, reusing the frozen split | Adjoint − critic regret reported with an episode-cluster bootstrap CI; opportunity proxy on validation | E |

**Decision after N1.** If the LQTree opportunity or correctness gates fail, stop and fix the benchmark or code before N2. A failure here is not a verdict on the adjoint hypothesis.

### Track B: benchmarks against rival models (from week 0; details in [`rival-benchmark-plan.md`](rival-benchmark-plan.md))

| ID | Deliverable | Done when | Status (2026-09-29) |
|---|---|---|---|
| B0 | Plan, tested package modules, two notebooks | `harness/check.py` passes; notebooks import the package | Done (this change) |
| B1 | `03-benchmarks/rival_world_models_droid100.ipynb` on L4: AdjointRWM vs DreamerV3-, TD-MPC2-, DINO-WM-, V-JEPA 2-AC-style arms + persistence/ridge, 5 seeds | `COMPLETE`, fairness contract PASS, imported, research note with every classification | **COMPLETE (2026-10-01):** Fairness contract PASS; AdjointRWM beats all 4 deep rival families on test proprio RMSE (DreamerV3 −56.8%, DINO-WM −40.9%, TD-MPC2 −36.3%, V-JEPA 2-AC −33.9%, persistence −31.2%; ridge +47.7%); action coupling 4.34× ([note](../research-notes/2026-10-01-rival-world-models-droid100.md)) |
| B2 | `01-production/AdjointRWM_Production_Pilot_v2.ipynb`, 5 paired seeds (= E2.1 + E2.2) | Opportunity, critic-floor and adjoint − critic results per seed and pooled; note written | **COMPLETE (2026-10-01):** 5/5 dynamics pass (+2.2 to +12.1% val, +29.6% test); exact co-state 0.00026; always_hold 0.0081; critic 0.0638; adjoint 0.0787; adjoint − critic +0.0149 [−0.0019, +0.0254]; verdict `NON_DIAGNOSTIC` ([note](../research-notes/2026-10-01-b2-pilot-v2.md)). Candidate redesign evaluation: Candidate 2 (Adaptive Sensing / Camera Gating) achieves 100% pass rate (5/5 seeds), 70.9% relative headroom, and 1.914 bits entropy, establishing the substrate for Milestone B2.1 ([note](../research-notes/2026-10-01-candidate-redesign-depth-vs-sensing.md)) |
| B2.1 | Adaptive Sensing Allocator on DROID-100 (5 seeds, 4 observation modes: proprio, wrist, exterior, full) | Confirmatory run on frozen dynamics; diagnostic headroom confirmed; primary endpoint tested | **COMPLETE (2026-10-01):** Diagnostic headroom confirmed (oracle regret 0.03007 vs best-fixed 0.14009). Primary endpoint met: $R_{\text{adjoint}} - R_{\text{critic}} = -0.00808$ (95% CI [−0.01481, −0.00047], $p < 0.05$); PARA normalized coupling reaches −0.01046; critic collapsed to full observation; amortization gap remains at 300 steps ([note](../research-notes/2026-10-01-b2-1-adaptive-sensing-allocator.md)) |
| B2.2 | Allocator Optimization Benchmark: Closing the Amortization Gap (5 seeds, 300 vs 1000 vs 2500 steps, normalized coupling, LCB) | Multi-checkpoint evaluation across 4,175 test windows; amortization gap closed | **COMPLETE (2026-10-01):** Amortization gap closed and reversed ($R_{\text{norm}} = 0.13019 \pm 0.03063$ vs $R_{\text{always\_mode0}} = 0.14009 \pm 0.05017$, diff **-0.00990**). Co-state advantage over critic widens from **-0.03032** (Step 300) to **-0.05937** (Step 2500, $p < 0.0001$). Matched critic flatlines at $0.18956$ ([note](../research-notes/2026-10-01-b2-2-allocator-optimization.md)) |
| B2.3 | Ranking Allocator Optimization Benchmark (Pairwise Margin Ranking & Plackett-Luce Listwise KL vs CE) | 3 seeds on 835 test windows on NVIDIA L4 | **COMPLETE (2026-10-02):** Continuous ranking losses do not close the amortization gap vs CE (CE: 0.13781, Margin: 0.13858, Hybrid: 0.13963, Listwise: 0.14257 vs Oracle: 0.03308); establishes that MLP representation capacity rather than loss function is the bottleneck, validating Direction 2 selective rescue ([note](../research-notes/2026-10-02-b2-3-ranking-allocator-benchmark.md)) |
| B3 | Selective Invocation and Analytical Rescue Benchmark on DROID-100 (5 seeds, decision-margin confidence gating) | Pareto frontier between amortized inference and analytical autograd co-states | **COMPLETE (2026-10-02):** Continuous Pareto trade-off between sub-microsecond amortized inference (0.0006 ms, 0.13511 regret) and exact autograd co-state backward rollouts (0.653 ms, 0.03007 regret). Rescuing 20% of ambiguous decisions drops regret to 0.11367 while sustaining >7,600 Hz throughput ([note](../research-notes/2026-10-02-b3-analytical-rescue-interface.md)) |
| B3b | B1 repeated on the E3.1 shard with scene/task/site splits | Confirmatory substrate comparison | **COMPLETE (2026-10-02):** Evaluated across 50 held-out episodes / 8,559 windows across 14 robotics laboratories. AdjointRWM statistically significantly beats all 4 deep rival families on test proprio RMSE (DreamerV3 −44.6%, TD-MPC2 −33.3%, DINO-WM −24.5%, V-JEPA 2-AC −24.4%; all `reference_better`, 50-cluster bootstrap $p < 0.001$). 100% win rate across individual robot labs (12/12) and 3.19× action coupling ([note](../research-notes/2026-10-02-b3b-rival-world-models-e3-shard.md)) |
| B3.3 | Direction 3 Curvature & Belief-Space VOI Allocation Benchmark | 7 allocator heads evaluated on NVIDIA L4 GPU | **COMPLETE (2026-10-03):** Second-order diagonal Hessian curvature ($s_k = -\hat{\lambda}^\top \Delta z_k - \frac{1}{2} \Delta z_k^\top \text{diag}(H) \Delta z_k - c_k$) achieves lowest regret among all deployable heads (`0.00105`), outperforming first-order co-states (`0.00128`, −17.4% error) and matched critic (`0.00132`, −20.0% error); 0.0971 ms/window latency (>10.2 kHz) on NVIDIA L4 ([note](../research-notes/2026-10-03-direction3-curvature-voi-allocator.md)) |
| B4 | Released-checkpoint and encoder-swap arms (exploratory; V-JEPA 2-AC is DROID-trained, so contamination is possible) | Contamination status recorded | Optional |
| B5 | Planning rivals in simulation (official TD-MPC2 code, CEM on ManiSkill3) | — | Gated on G-H2 and Phase II |

### Track D: cross-domain generalisation (details in [`cross-domain-plan.md`](cross-domain-plan.md))

| ID | Deliverable | Done when | Status (2026-09-29) |
|---|---|---|---|
| D0 | Plan; `adjointrwm.domains` (interface, runner, ledgers, metrics); D4 reference domain with exact tests | `harness/check.py` passes | Done |
| D4-0 | `04-domains/d4_adaptive_time_stepping.ipynb` on CPU: correctness, rate-budget opportunity, weighting comparison, equal-compute check | Run imported, note written | Done: correctness ✅, rate-budget opportunity ✅, equal-compute payoff ❌ ([note](../research-notes/2026-09-29-d4-0-adaptive-time-stepping.md)) |
| D4-0b | Pass-based allocation and a localisation sweep, validation only | Run imported, note written | Done: at the real price adaptivity pays only in the most localised family and the residual score does as well as the co-state; the frozen rule is met in 2 of 9 cells, both at hypothetical prices ([note](../research-notes/2026-09-29-d4-0b-where-adaptivity-pays.md)) |
| D4-2 | Learned amortised scorer priced at its measured cost; higher-dimensional system; one factor varied at a time; wider `θ` grid | Regime and price frozen before validation is read; note written | **Done:** correctness ✅; R1 (an amortised scorer beats uniform) holds in 3 of 7 cells, R3 (the co-state weight helps the cheap estimator) in 1, none has both; a non-learned cheap estimator matches or beats the learned scorer everywhere ([plan](d4-2-plan.md), [note](../research-notes/2026-09-30-d4-2-flop-priced-scoring.md)) |
| D4-3 | Varying goal per instance: the non-learned pair `cheap` vs `cheap_adjoint` on validation, frozen rule written first | Regime and rule frozen before validation; note written | **Done:** correctness ✅; R3v (the co-state weight helps a goal-blind cheap estimator) holds in 2 of 7 cells (`m4`, `m64`), R4v (adaptivity pays with the goal-aware arm) in 5 of 7; both in `m4` and `m64` (11 to 17 % and 8 to 18 % less compute than `cheap`); with the exact table charged per instance the goal-aware arm needs 2.1 to 50 times `cheap`'s compute ([plan](d4-3-plan.md), [note](../research-notes/2026-09-30-d4-3-varying-goal.md)) |
| D4-1 | Learned direct critic vs co-state-featured critic on D4, 5 seeds, cells `m4` and `m64` | Rung-1 gate, §0A.3 exit class | **Done (CPU, exploratory):** correctness ✅; R0 ❌ in both cells; co-state critic vs direct critic ❌ at both prices (ratio 1.02 to 1.10 in `m4`, 0.96 to 1.04 in `m64`), no better than its randomised control; exact co-state as a feature helps the head ~8 % in `m4` at the hypothetical lookup price; exit class `m4` teacher-only value, `m64` inconclusive (estimator floor met in 4 of 5 seeds in `m4`, 0 of 5 in `m64`); test family unread ([plan](d4-1-plan.md), [note](../research-notes/2026-09-30-d4-1-learned-critics.md)) |
| DL | Licence survey of permissively licensed data and models for the other domains | Per-source licence and release terms; small usability test | **Pass 2 done** (59 sources: 17 `adopt`, 12 `adopt_with_conditions`, 5 `avoid`, 25 `unverified`; [survey](../licences/survey-2026-09-29-pass2.md)) |
| D1-0 | Sensor-stream domain card, adapter and Rung-0 opportunity gate on SMD (tuning and validation machines only; test machines never downloaded) | Run imported; note written; gate of plan §7.2B-1 | **Done:** correctness ✅; G1 ✅ robust (relative headroom 0.920 [0.886, 0.949] against the best fixed allocation, privileged reference); non-learned deployable policies keep 0.42 to 0.49 of it; **labelled F1 does not improve with sensing (0.259 hold, 0.192 full), so the proxy loss does not track the task** ([plan](d1-0-plan.md), [note](../research-notes/2026-09-30-d1-0-sensor-opportunity.md)) |
| D1-0b | Redesign D1's objective: a frozen per-machine ridge forecaster's native 5-minute-ahead error, tested first on tuning machines | Regime and rule frozen first; note written | **Done:** correctness ✅; G0 ✅ on tuning (`r` = 0.522), ✅ on validation by the point rule but not robust (0.298 [0.023, 0.743]; one machine negative); G1 passes by the frozen rule but the ratio is unstable (0.875 [0.295, 1.347]); **neither deployable dynamic policy keeps any of the headroom** (G2) and the best static policy beats both ([plan](d1-0b-plan.md), [note](../research-notes/2026-09-30-d1-0b-forecast-sensing.md)). **D1-1 is not designed**; the note recommends deprioritising D1 for H2 (for Roman to confirm) |
| D2-0, D2-1, D2-2 | Hierarchical LLM DAG Generation & In-Place Adapter Hot-Swapping on NVIDIA L4 (Tier P) | Rung-0 gate, structural parity verification, and 3-way comparative benchmark | **Done (2026-10-02):** In-place adapter hot-swap latency 11.80 ms p50 (35.5× faster than disk reload); 3 resident adapters +100.7 MiB VRAM; 100% build pass rate (+20% vs standard DAG), 91.0% sibling preservation, 2.04× latency speedup ([note](../research-notes/2026-10-02-d2-hierarchical-dag-and-adapter-hotswap.md)) |
| D3-0 | Code-repository context domain card and adapter | Rung-0 gate | Candidates confirmed; builds on D2-0 DAG framework |
| DX | Shared allocator, leave-one-domain-out, negative-transfer guard | Rungs 2–3; confirmatory only in N6 | Waits for ≥ 3 specialists and the N2 exit class |

**Progress on existing rows (2026-09-29, not yet run on Colab):**

- **N0.1, done (2026-10-07).** Data windows, split, normalisation, objective, the pilot model, exact co-state targets and evaluation now live in `src/adjointrwm/`. Tests cover split and window parity with the pilot, causality (predictions unchanged when future targets are replaced), exact co-state against float64 finite differences, and checkpoint round-trip. The pilot's parameter count is reproduced exactly (27,360,798, as in its `model_manifest.json`). The v2 notebook imports the package (done-condition met); v1 keeps its inlined code frozen for provenance and will not be migrated.
- **N0.3, CPU part done.** `tests/test_training.py` checks that pause-and-resume gives bit-identical weights to uninterrupted training. It found that creating a `DataLoader` iterator draws from the global RNG; the runner now uses a dedicated generator. v1 has no resume path, so no v1 result is affected. Still open: the same check on Colab with CUDA.
- **E2.1, implemented, not run.** It is the pilot v2 notebook (B2). The one deviation from the E2.1 wording: the dynamics gate is evaluated on validation, not test, so the test split is read once.

### Then: weeks 5–18 (3 Nov – 1 Feb)

| ID | Deliverable | Done when | Track |
|---|---|---|---|
| N2 | `HJoinBench-0` (exact) → `-1` (learned effects) → `-2` (held-out pairs and topologies), with every required arm from the comprehensive plan §0A.3 | Primary endpoint `AURC_direct − AURC_adjoint` with complete rate and `C_total` matching, classified into one of the exit classes | R |
| N3 | Mechanism factorial: directional-only, co-state-only, full, randomised, permuted, stale, sign-flipped, wrong-goal, wrong-horizon | Mechanism claim allowed only if co-state changes actions, the changed actions lower held-out regret, and the controls remove the effect | R |
| E3.1 | Stream a stratified 500–1,000-episode DROID shard (full DROID RLDS, not `droid_100`), split by scene, task and site | **COMPLETE (2026-10-02):** 500 episodes streamed across 14 robot laboratories; exact 80/10/10 split (400 train, 50 val, 50 test); 8/8 multi-modal contract and eye-inspections passed ([note](../research-notes/2026-10-02-e3-1-droid-500-shard.md)) | E |
| E3.2 | Dynamics pilot at 20–60 M params: one-step and multi-step, native units, against persistence, linear and action-shuffled baselines | **COMPLETE (2026-10-02):** Trained on 399 E3.1 shard episodes across 14 labs; test proprio RMSE 0.2586 (-30.5% vs B3b transfer); beats persistence in 10/12 labs; Cartesian MSE -34.2% and Gripper MSE -61.2% vs persistence; 4.00× action coupling; batch-1 latency 6.77 ms ([note](../research-notes/2026-10-02-e3-2-dynamics-pilot-shard.md)) | E |
| E3.3 | Multi-horizon rollout decay benchmark ($H \in \{1, 2, 4, 8, 12, 16\}$) against persistence and linear extrapolation | **COMPLETE (2026-10-02):** Evaluated across 6 horizons on 50 held-out test episodes. Crossover horizon confirmed at $H^*=8$ (terminal) and $H^*=11$ (mean), where AdjointRWM beats persistence by 15.9% and 21.6% while linear models compound error (+523.8% terminal vs +113.2% for AdjointRWM) ([note](../research-notes/2026-10-02-multi-horizon-rollout-decay.md)) | E |
| E3.4 | HARP-1: Hybrid Kinematic-Residual Adjoint Architecture ($\hat{s}_h = s^{\text{kin}}_h + \Delta s^{\text{residual}}_h$) on E3.1 shard | **COMPLETE (2026-10-02):** Trained on 399 multi-site shard episodes; beats BOTH Persistence and Ridge on physical Cartesian 6D pose (−46.1% vs Persistence, −4.1% vs Ridge) and Gripper aperture (−39.9% vs Persistence, −7.7% vs Ridge); Step 1 error drops by 44.4% (0.2330 to 0.1295); batch-1 latency 7.80 ms on NVIDIA L4 ([note](../research-notes/2026-10-02-harp-hybrid-kinematic-dynamics.md)) | E |

**Gate G-H2 (≈ weeks 14–18).** One exit class from comprehensive plan §0A.3 is recorded in `docs/research-notes/`, and the rest of the roadmap branches on it:

| Outcome | What opens |
|---|---|
| **Deployable adjoint contribution** | N4 selective invocation; production P5–P7 with the adjoint arm; paper track A |
| **Teacher-only value** | N4 restricted to cached/high-fidelity modes; production keeps the critic as the primary allocator |
| **Direct utility sufficient** | Drop the adjoint arm from production; the critic-based allocator goes into P7; paper track B (negative result) |
| **Inconclusive** (under-realised critic or low power) | Bounded direct rescue (≤ 20 % extra compute) and seed expansion per the plan; no scale-up |

### Later: weeks 15–68 (from Jan 2027)

These follow the comprehensive plan §0A.4–0A.10 and open only through the gates listed there.

| Phase | Window | Opens when |
|---|---|---|
| N4: compression and selective invocation (L4) | wk 15–28 | G-H2 positive or teacher-only |
| N5: continuous control ladder (analytic → Gymnasium → MuJoCo) | wk 19–38 | N3 mechanism result recorded |
| N6: cross-domain and multimodal joins | wk 21–42 | N2 join classification |
| N7: learned JEPA-style latent hierarchy | wk 35–56 | Representation-only pilots pass collapse tripwires |
| N8: hardware frontiers and runtime assurance (T4/L4/A100 reported separately, never pooled) | wk 29–58 | L4 contract frozen in N4 |
| N9: independent replication and release | wk 55–68 | All promoted claims have gate artefacts |

### Production track (gated)

The production plan (P0–P11, RWM-S → RWM-M → distilled L4 student) assumes an adjoint mechanism that has already been validated. That assumption doesn't hold yet, so the plan is split:

| Can start now (no H2 dependency) | Waits for G-H2 |
|---|---|
| P0: asset conversion and golden tests for the D=32 teacher (`weights_only=True`, FP64/FP32/BF16 parity) | P5–P6: co-state label generation and amortised co-state student at production scale |
| P1: data and runtime substrate (sharded store, local `/content` cache, full checkpoint contract) | P7: allocator and selective gate as a product feature |
| P2: proxy scaling of the **dynamics** model only | P8: hierarchical joins (also waits for N6) |
| P4: action-conditioned dynamics on real data (= Track E3) | P9–P11: specialists, distillation, release |

P3 (representation pretraining from a released video-JEPA backbone) can begin after E3.2 passes. Before relying on the backbone IDs and papers cited in the production plan, check that they exist and that their licences allow this use; some of those references are very recent preprints.

---

## 4. Hardware policy

Rules come from the operator brief and the audits:

- **Stay on L4** for everything through N3 and E3. Measured peak so far is 0.6 GiB, and throughput doesn't change with batch size, which suggests the loader or Python is the bottleneck, not the GPU.
- **A100 40 GB** is justified only if the minimum useful workload doesn't fit, a profiler shows real compute or bandwidth saturation, the runtime is impractical, or a larger context, resolution or model gives a *measured* held-out gain.
- **A100 80 GB** is justified only if 40 GB can't hold the promoted configuration.
- **H100 / FP8** only after the medium model is validated and BF16 parity is shown.
- Unused VRAM and dataset row count are never reasons to upgrade.

## 5. Publication plan

| Track | Content | Earliest | Venue type |
|---|---|---|---|
| A (positive H2) | Protocol + LQTree + HJoinBench + mechanism + DROID substrate | after N3 (~Feb 2027) | Main track (ICML/NeurIPS 2027 cycle); check current deadlines |
| B (negative or tie) | Matched-critic protocol, opportunity gates, collapse failure mode, evidence-integrity methodology | after G-H2 | Workshop or TMLR |
| C (tooling) | Allocation-evaluation harness (`adjointrwm.analysis`) + DROID windowing/split contract | after E2 | Workshop / software paper |

Every sentence in a submission maps to a gate artefact through a claim matrix (plan N9). The current draft is retired until then; see [REVIEW](../../papers/drafts/REVIEW.md).

## 6. Risks and stop-losses

| Risk | Early signal | Response |
|---|---|---|
| The benchmark has no adaptive opportunity | Opportunity proxy < 15 % on validation | Redesign the candidates (resolution, camera port, rollout depth), not the allocator |
| Allocator collapse repeats | ≤ 2 candidates used; below-chance top-1 | Match training and evaluation semantics; add class-balanced or listwise ranking; check the train/test oracle shift |
| Direct critic under-realised | Critic loss not converged, or capacity-limited | Bounded rescue before any adjoint claim |
| Train/test distribution shift in gains | Oracle shares differ by split | Stratify by scene/task; more episodes; report per episode |
| Colab pre-emption loses state | Missing heartbeat | Full checkpoint contract plus 15–30 min recovery checkpoints; resume test (N0.3) |
| Synthetic or fabricated data creeps back in | Hash-only images, seeded RNG streams, constant metrics | Operator-brief rule 1; the N0.2 CI tests fail the run |
| Scope creep into production before G-H2 | P5+ work started | This roadmap: gated column in §3 |

## 7. Open questions for Roman

1. **Primary endpoint:** allocation regret/AURC (research) or task success at matched compute (needs a simulator such as ManiSkill3)? This decides whether a simulator goes into E3.
2. **Data scale:** is streaming full DROID from `gs://gresearch/robotics` within your Colab storage and egress limits, or should E3 use BridgeData V2 first?
3. **Governing plan:** should the comprehensive (cross-domain) plan formally supersede v5? This roadmap assumes yes.
4. **Stage 0 notebooks:** which Drive copy of `Stage0_LQTree_Benchmark` is canonical? Six versions exist.
5. **Rival benchmark endpoint:** normalised-state RMSE (the pilot's gate metric, assumed) or native-unit RMSE of one state group as the B1 primary endpoint?
6. **TD-MPC2 input:** information-matched 8-frame stack (assumed) or the faithful single frame as the primary TD-MPC2 arm?
7. **B4:** run the released V-JEPA 2-AC checkpoint at all, given possible DROID train/test contamination and a ViT-g encoder on L4?
8. **Track D:** decided 2026-09-29: D4 first, the other domains subject to future research into permissively licensed content and testing. Decided later the same day: licence survey (DL) first, then the rest. DL is done through pass 2. Decided 2026-09-30 ("proceed as recommended"): the D1, D2 and D3 candidates the survey recommends are confirmed (SMD; Qwen3-8B with MuSiQue and Qasper; code-repository context), D4-1 stayed closed at that point, and E1.1 then B2 remain the critical path (Colab). Decided later on 2026-09-30 ("redesign its objective (D1-0b), want the varying-goal check, proceed with all practical CPU experiments"): D1-0b and D4-3 were run; D4-3 reopened D4-1 in `m4` and `m64`; D1-0b found no deployable policy that keeps the headroom, so D1-1 is not designed. D2-0 and D3-0 are not started. Open: whether to deprioritise D1 for H2 (the D1-0b note recommends it), and whether D2-0 and D3-0 should start before the Colab runs (D2-0 needs an L4).
