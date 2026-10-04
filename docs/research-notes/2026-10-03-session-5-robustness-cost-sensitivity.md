# Session 5 Research Note: Regret Robustness, Cost-Model Sensitivity & Multi-Horizon Generalization

**Date:** 2026-10-03 (Reconciled post-Audit; **corrected post evidence-integrity audit**)
**Status:** ⚠️ **MEASUREMENTS STAND · PRIMARY ENDPOINT UNSUPPORTED · EXIT GATE NOT PASSED AS STATED**
**Authors:** Antigravity (Lead Agent) & OpenCode (`space-bunny-free`, Peer Critic)
**Target Milestone:** Session 5 (10-Session Operational Research Campaign)
**Hardware:** Google Colab NVIDIA L4 GPU (22.03 GiB VRAM, Session `l4-worker`)
**Pinned Commit:** runner at `2cd0faf`; corrected at `8facdf1`
**Audit Documents:** [`2026-10-03_session_5_loso_and_cost_reconciliation_audit.md`](../audits/2026-10-03_session_5_loso_and_cost_reconciliation_audit.md) (arithmetic) · [`2026-10-03_session5_seed_duplication_and_voi_identity_audit.md`](../audits/2026-10-03_session5_seed_duplication_and_voi_identity_audit.md) (evidence integrity)
**Artifacts:** [`robustness_horizon_summary.json`](../../results/benchmarks/robustness_horizon/robustness_horizon_summary.json), [`robustness_horizon_report.md`](../../results/benchmarks/robustness_horizon/robustness_horizon_report.md)
**Reproduce the integrity checks below:** `python scripts/audit_session5_evidence_integrity.py` (seed fingerprint, per-seed signs, scorer identity, metric inversion, inference availability). The §2 means and the LOSO/bounding-identity tables come from `python scripts/reconcile_session5_evidence.py`. Both were run for the audit.

---

> ### ⚠️ Read this before citing anything in this note
>
> A second audit (`2026-10-03_session5_seed_duplication_and_voi_identity_audit.md`) found three defects that this note did not report. All arithmetic here is correct; the **interpretation** was not.
>
> 1. **There are two seeds, not three.** Seed 2 reproduced seed 0's evaluation exactly — all 2,359 performance scalars are bit-identical; the only 14 that differ are wall-clock timings. The runner contains one silent code path that can do this (`benchmark_robustness_horizon_allocator.py:561-565`, a `seed_0` checkpoint fallback added in commit `2f5507d`), but the committed files do not record which checkpoint each seed loaded, so the *mechanism* is not proven — only the *duplication* is. Either way the sign of the headline result flips on the one independent seed in **6 of 7 panels**.
> 2. **`belief_space_voi` and `second_order_curvature` are the same functional family** (`delta_cov = effects.pow(2)`), differing only in the sign and a factor of two on the quadratic term. There is no epistemic term. The stated failure mechanism in §2A is therefore wrong, and no claim about value of information follows.
> 3. **The headline metric and the count metric rank the policies almost oppositely** (1 of 8 policies agrees). VOI beats the direct critic on 20.6 % of windows. No paired test or CI on the difference exists, and no per-window data was committed, so the endpoint is not testable from these artefacts.
>
> **Session 5 must not be cited as evidence for the co-state hypothesis, and the Exit Gate in §4 is not a pass.** What survives is listed in §5.

---

## 1. Executive Summary

Session 5 executed an adversarial sensitivity and robustness benchmark of second-order curvature and belief-space Value-of-Information (VOI) allocation on the real multi-site robotics test set (DROID E3.1 stratified shard across 12 robotics laboratories, $N = 4{,}154$ unique held-out test windows). The artefact records three seed entries, but **only two are independent**: seed 2 reproduced seed 0's evaluation exactly — all 2,359 performance scalars (regret, gain, CI, win rate, per-site, LOSO, counts) are bit-identical, and the only 14 that differ are wall-clock timing fields. The effective sample is therefore **2 teachers** applied to the same 4,154 windows.

⚠️ **On evaluation counts:** 4,154 × 2 = 8,308 (or × 3 = 12,462) is *not* a sample size. The same windows are reused across teachers, and they are generated with a stride of 2 (context 8, horizon 4), so adjacent windows overlap. These are repeated measurements of 4,154 overlapping windows. Do not read them as independent observations.

Findings, after both audits:

1. **Arithmetic reconciliation holds.** The LOSO multi-seed aggregation and the pooling identity ($\min_s \bar r_{-s} \le \bar r \le \max_s \bar r_{-s}$) were verified to machine precision across all seeds. Every table in this note reproduces exactly from the committed JSON. This work was correct and is retained.
2. **The primary endpoint is a point estimate whose sign is not stable.** Belief-space VOI improves mean regret over the capacity-matched critic by +8.29 % to +24.42 % in the 2-seed mean, but the independent replicate (seed 1) is **negative in 2 of 4 moderate regimes** and **−38.29 %** against the direct critic under the default regime. There is no CI or paired test on any difference.
3. **Cost robustness is conditional, and the mechanism is curvature-sign sensitivity, not value of information.** VOI is 22.73 % worse than the matched critic under severe dual-camera penalties. Because `delta_cov = effects.pow(2)`, the "epistemic" term is the *negated curvature* term; it over-rewards high-curvature candidates, and a $5\times$ acquisition penalty on joint refinement flips that into a loss.
4. **Multi-horizon behaviour is mixed.** The 2-seed mean gives +56.05 % at $H=2$ and +16.98 % at $H=4$, but the per-seed split is +59.69 % / +5.99 % and +36.14 % / −38.29 % respectively.
5. **Omission sensitivity is positive on average and negative for the independent replicate in every subset.** The 2-seed mean advantage spans +4.20 % to +19.78 %, but seed 1 is negative on **all 12** site deletions (−7.76 % to −65.65 %).
6. **The advantage is a heavy-tail mean-regret effect, not a per-window win.** VOI beats the direct critic on 20.6 % of windows; do-nothing refusal beats it on 63.2 %. Mean regret and win rate rank the eight policies in near-opposite order (1 of 8 agree).
7. **Track D2/D3 Deprecation:** cross-domain code-generation claims remain superseded and non-evidentiary (`2026-10-03_repo_code_gen_verdict_audit.md`).

**Bottom line:** Session 5 produced a real, correctly-computed cost-regime sweep and a real negative result at high penalty. It did not produce evidence that belief-space VOI beats a direct critic, because the comparison rests on two teachers that disagree in sign, between two scorers that are the same functional family, on a metric whose per-window ordering is reversed.

---

## 2. Key Empirical Findings

### A. Cost-Model Sensitivity Sweep ($H=4$, 4,154 Unique Windows × 2 Independent Seeds)

Allocators choose among 4 named modes: Mode 0 = Hold (0.0 cost), Mode 1 = Visual-only refinement ($c_1$), Mode 2 = State-only refinement ($c_2$), Mode 3 = Joint visual + state refinement ($c_3$). Heads were retrained and actions reselected under each cost regime.

| Cost Regime | Modality Cost Vector $[c_0, c_1, c_2, c_3]$ | Direct Critic Regret | Matched Critic Regret | Belief-Space VOI Regret | VOI vs Direct Critic (%) | VOI vs Matched Critic (%) |
|---|---|---|---|---|---|---|
| **Zero Cost** | `[0.0, 0.0, 0.0, 0.0]` | 0.13008 | 0.10324 | **0.09469** | **+27.21%** | **+8.29%** |
| **Uniform** | `[0.0, 1.0, 1.0, 1.0] * 0.002` | 0.12953 | 0.12846 | **0.09710** | **+25.04%** | **+24.42%** |
| **Default** | `[0.0, 1.0, 1.0, 2.0] * 0.002` | 0.13129 | 0.12878 | **0.10899** | **+16.98%** | **+15.36%** |
| **Latency-Weighted** | `[0.0, 0.8, 0.8, 1.8] * 0.002` | 0.13077 | 0.12856 | **0.09975** | **+23.72%** | **+22.41%** |
| **High Penalty** | `[0.0, 1.0, 1.0, 5.0] * 0.002` | 0.16289 | **0.12967** | 0.15915 | **+2.30%** | **−22.73%** |

⚠️ **These percentages are 2-seed means that weight the duplicated `seed 0` entry 2:1, and the sign is not stable.** Per-seed VOI advantage over the direct critic:

| Cost Regime | seed 0 (= seed 2) | seed 1 (independent) |
|---|---:|---:|
| Zero Cost | +37.00% | **−1.12%** |
| Uniform | +36.21% | **−6.83%** |
| Default | +36.14% | **−38.29%** |
| Latency-Weighted | +36.34% | **−12.68%** |
| High Penalty | +3.01% | **−0.40%** |

Against the *matched* critic, seed 0 is **−31.47%** under `high_penalty` where seed 1 is +1.12% — the opposite ordering. No CI or paired test on any of these differences exists; see §5.

**Key Takeaways:**
1. **Zero-Cost Baseline:** At zero acquisition cost, the 2-seed mean shows VOI reducing regret by +8.29 % over the capacity-matched critic and +27.21 % over the direct critic. The intent — that the effect is not purely an artefact of acquisition penalties — is reasonable but **not established**: seed 1 is −1.12 % against the direct critic here.
2. **Conditional Robustness:** Across moderate cost variations, the 2-seed mean keeps a +15 % to +24 % advantage over matched critics, but **3 of 4 moderate regimes are negative for the independent replicate.** Robustness is not established.
3. **Failure Mode Under High Acquisition Penalty (corrected):** When joint refinement is penalised at $5\times$ base cost ($c_3 = 0.010$), belief-space VOI is 22.73 % worse than the capacity-matched critic. **The earlier explanation in this note was wrong.** There is no epistemic uncertainty bonus in this scorer: `scripts/benchmark_robustness_horizon_allocator.py:318` sets `delta_cov = effects.pow(2)`, so with $\beta = 0.5$ the scorers are
   $$\text{second\_order\_curvature}: \ s = -\langle\lambda, \text{effects}\rangle - 0.50\,Q - c, \qquad \text{belief\_space\_voi}: \ s = -\langle\lambda, \text{effects}\rangle + 0.25\,Q - c$$
   with $Q = \sum_d H_d\,\text{effects}_d^2$. The two differ only in the sign and a factor of two on the quadratic term; `uncert_weight = -1.0` maps one exactly onto the other. Because VOI's quadratic is *sign-flipped*, it **adds** score to expensive, high-curvature candidates, which is precisely what a $5\times$ joint-refinement penalty punishes. The correct description is **curvature-sign sensitivity**, not uncertainty-driven over-exploration — and no claim about value of information follows from it.

---

### B. Multi-Horizon Generalization (Default Cost, 2-Seed Mean)

⚠️ **`±` is the population standard deviation (`ddof=0`) over the three seed entries, two of which are the same evaluation.** Sample std (`ddof=1`) would be 18 % larger (e.g. $H=2$ `exact_costate`: 0.06643 vs 0.08136). The convention is undeclared in the original table. With one independent replicate there is no meaningful cross-seed dispersion to report under either convention.

| Horizon $H$ | Autograd Reference | Refusal (`always_mode0`) | Direct Critic | Matched Critic | Second-Order Curvature | Belief-Space VOI | VOI vs Critic Advantage |
|---|---|---|---|---|---|---|---|
| **$H=2$** | 0.05679 ± 0.06643 | 0.88536 ± 0.17924 | 0.37324 ± 0.21029 | 0.07024 ± 0.00512 | 0.26054 ± 0.13240 | **0.16404 ± 0.06557** | **+56.05%** |
| **$H=4$** | 0.00813 ± 0.00166 | 1.17650 ± 0.26122 | 0.13129 ± 0.02115 | 0.12878 ± 0.01806 | 0.11764 ± 0.02643 | **0.10899 ± 0.02207** | **+16.98%** |

Per-seed split of the VOI advantage over the direct critic:

| Horizon | seed 0 (= seed 2) | seed 1 (independent) |
|---|---:|---:|
| $H=2$ | +59.69% | +5.99% |
| $H=4$ | +36.14% | **−38.29%** |

**Key Takeaways:**
1. At short horizon ($H=2$), standard direct critics exhibit severe cross-seed estimation variance in the 2-seed mean ($0.37324 \pm 0.21029$). Second-order curvature stabilizes value estimation ($0.26054$), while belief-space VOI cuts regret to $0.16404$ (+56.05 % over standard critic). **The seed split is +59.69 % / +5.99 %, so the size of the short-horizon gain is not pinned down.**
2. **Variance vs Standard Deviation (terminology corrected):** At $H=2$, the cross-seed spread of the mean falls from 0.21029 to 0.06557 — a **68.8 % reduction in cross-seed standard deviation**, corresponding to a **90.3 % variance reduction**. This ratio is unaffected by the ddof choice. Within-sample standard deviation falls from 9.8260 to 4.2246 (57.0 % std reduction; 81.5 % variance reduction).
3. The capacity-matched critic achieves strong performance at $H=2$ ($0.07024$) due to aggressive refusal, highlighting that horizon-specific capacity interacts strongly with value estimation. **Note this is a large advantage over every deployed allocator at $H=2$ and is not addressed by the VOI comparison.**

---

### C. Leave-One-Site-Out (LOSO) Omission Sensitivity ($H=4$, Default Cost, 2-Seed Mean)

*Methodological Note:* This panel measures **omission sensitivity** (re-aggregating evaluation metrics when excluding records belonging to laboratory $s$), not unseen-site out-of-domain generalization. Total unique test windows: $N = 4,154$ across $K = 12$ robotics laboratories.

| Dropped Site | Site Windows $n_s$ | Omission Weight $w_{-s}$ | Remaining Critic Regret | Remaining VOI Regret | VOI Advantage (%) |
|---|---|---|---|---|---|
| **All 12 Sites (Full Sample)** | **4,154** | **1.000000** | **0.13129** | **0.10899** | **+16.98%** |
| `Drop AUTOLab` | 512 (12.3%) | 0.079704 | 0.14207 | 0.11654 | +17.97% |
| `Drop BVL` | 183 (4.4%) | 0.086904 | 0.13489 | 0.11176 | +17.14% |
| `Drop ILIAD` | 1,024 (24.7%) | 0.068499 | 0.11459 | 0.09238 | +19.38% |
| `Drop IPRL` | 376 (9.1%) | 0.082680 | 0.13298 | 0.10668 | +19.78% |
| `Drop IRIS` | 500 (12.0%) | 0.079967 | **0.10546** | **0.10103** | **+4.20%** |
| `Drop PennPAL` | 185 (4.5%) | 0.086860 | 0.13448 | 0.11125 | +17.27% |
| `Drop RAD` | 144 (3.5%) | 0.087758 | 0.13034 | 0.10757 | +17.46% |
| `Drop RAIL` | 76 (1.8%) | 0.089246 | 0.13295 | 0.10980 | +17.41% |
| `Drop REAL` | 96 (2.3%) | 0.088808 | 0.13349 | 0.11007 | +17.55% |
| `Drop RPL` | 197 (4.7%) | 0.086598 | 0.13444 | 0.11231 | +16.47% |
| `Drop TRI` | 781 (18.8%) | 0.073817 | 0.14417 | 0.11593 | +19.59% |
| `Drop WEIRD` | 80 (1.9%) | 0.089158 | 0.13281 | 0.10994 | +17.21% |

**Key Takeaways:**
1. Under the recorded-seed aggregation, the mathematical bounding identity holds exactly:
   $$\min_s \bar{r}_{-s} = 0.09238 \le 0.10899 \le 0.11654 = \max_s \bar{r}_{-s}$$
   $$\sum_{s=1}^{12} w_{-s} \bar{r}_{-s} = 0.108992 \quad (\text{difference: } 3.23 \times 10^{-9})$$
2. **Impact of IRIS:** Omitting laboratory `IRIS` reduces the direct critic regret from $0.13129$ to $0.10546$, compressing the VOI advantage from **+16.98 % down to +4.20 %**.
3. **Robustness conclusion (downgraded):** the earlier claim that the advantage is "strictly positive across all 12 deletion subsets" holds for the 2-seed mean, and the earlier flawed figures (+9.83 % / +36.2 %–+44.7 %) remain retracted. However, the **independent replicate (seed 1) is negative on all 12 subsets** (−7.76 % to −65.65 %). The advantage is not robust to the choice of teacher.

**Per-seed LOSO advantage (VOI vs direct critic):**

| Dropped Site | seed 0 (= seed 2) | seed 1 (independent) |
|---|---:|---:|
| `AUTOLab` | +37.78% | **−40.79%** |
| `BVL` | +36.56% | **−39.26%** |
| `ILIAD` | +44.74% | **−65.65%** |
| `IPRL` | +39.06% | **−36.88%** |
| `IRIS` | +9.83% | **−7.76%** |
| `PennPAL` | +36.89% | **−39.68%** |
| `RAD` | +37.31% | **−40.11%** |
| `RAIL` | +36.64% | **−38.32%** |
| `REAL` | +36.77% | **−38.04%** |
| `RPL` | +36.23% | **−40.89%** |
| `TRI` | +39.71% | **−42.15%** |
| `WEIRD` | +36.33% | **−38.45%** |

---

### D. Tail-Insensitive vs Spread Analysis (Default Cost, H=4)

⚠️ **This table is where the headline claim lives or dies. Mean regret and win rate rank the eight policies in near-opposite order (1 of 8 agree).**

| Policy | Mean Regret | Mean Rank | Median Regret | 10% Trimmed Mean | Sample Std Dev | Win Rate vs Critic | Win Rank |
|---|---:|:--:|---:|---:|---:|---:|:--:|
| `exact_costate` (Reference) | 0.00813 ± 0.00166 | 1 | 0.00000 | 0.00151 | 0.02521 | 66.4% | 1 |
| `belief_space_voi` | **0.10899 ± 0.02207** | 2 | 0.05635 | 0.07426 | **0.64384** | **20.6%** | **6** |
| `second_order_curvature` | 0.11764 ± 0.02643 | 3 | **0.05563** | **0.07381** | 0.85044 | 25.5% | 5 |
| `direct_critic_curv_matched` | 0.12878 ± 0.01806 | 4 | 0.05996 | 0.07557 | 1.66612 | 3.6% | 7 |
| `direct_critic` | 0.13129 ± 0.02115 | 5 | 0.06009 | 0.07603 | 1.66535 | 0.0% | 8 |
| `first_order` | 0.20069 ± 0.14618 | 6 | 0.04093 | 0.06162 | 3.24359 | 35.4% | 4 |
| `normalized_first_order` | 0.31577 ± 0.08034 | 7 | 0.04110 | 0.05940 | 10.34503 | 45.3% | 3 |
| `always_mode0` (Refusal) | 1.17650 ± 0.26122 | **8** | 0.01130 | 0.05552 | 28.22032 | **63.2%** | **2** |

**Key Takeaways:**
1. Second-order curvature achieves marginally better central summaries (median $0.05563$ vs $0.05635$, trimmed mean $0.07381$ vs $0.07426$) — differences of 0.1 % and 0.6 %.
2. Belief-space VOI achieves a better mean ($0.10899$ vs $0.11764$, 7.35 %) and a 24.3 % lower standard deviation ($0.64384$ vs $0.85044$). **Given Finding 2, this is the difference between two points in one functional family, not between two mechanisms.**
3. **Mechanism (restated precisely):** the mean-regret advantage is driven by preventing catastrophic, expensive misallocations in high-uncertainty tail episodes, not by improving ordinary decisions. The order-robust statistics say so directly — on median and 10 % trimmed mean, VOI and curvature are indistinguishable, and **refusal (`always_mode0`) beats both on the 10 % trimmed mean**.
4. **⚠️ The inversion that the original note did not state:** belief-space VOI beats the direct critic on only **20.6 %** of held-out windows — the critic wins 79.4 %. Do-nothing refusal beats the direct critic on **63.2 %** of windows while carrying 10.8× VOI's mean regret. Mean regret is dominated by a rare tail (`always_mode0` std = 28.2 against VOI's 0.64), so "VOI beats the critic by +16.98 %" and "the critic beats VOI on 79 % of windows" are both true and are statements about different things. **Any claim of the form "VOI allocates better than the critic" must state which metric it means.**

**Why this cannot be resolved from the committed artefacts.** The effect size is 0.02230 (critic − VOI mean regret). The only uncertainty present is each policy's bootstrap CI on its *own* mean: `direct_critic` half-width **0.04832** (2.2× the effect), VOI **0.01864** (0.8×). There is **no** CI or test on the difference. Critically, only `robustness_horizon_summary.json` and `robustness_horizon_report.md` are committed — **no per-window regret data** — so the paired Wilcoxon signed-rank test that Session 6A mandates cannot be applied retroactively. A paired comparison here is permanently untestable without a re-run.

---

## 3. Compute Discipline & Audit Compliance

- **Runtime Profile:** Executed on single NVIDIA L4 GPU (22.03 GiB VRAM) on Google Colab.
- **Compute Discipline:** All jobs ran sequentially without idle time. VM was terminated immediately via `colab stop -s l4-worker` upon completion (verified 0 active sessions).
- **Integrity Compliance:**
  - Audited and remediated hardcoded verdict string in `scripts/run_repo_code_generation_benchmark.py` ([`2026-10-03_repo_code_gen_verdict_audit.md`](../audits/2026-10-03_repo_code_gen_verdict_audit.md)).
  - Audited and resolved LOSO multi-seed aggregation bug ([`2026-10-03_session_5_loso_and_cost_reconciliation_audit.md`](../audits/2026-10-03_session_5_loso_and_cost_reconciliation_audit.md)).
  - Audited seed duplication, VOI/curvature functional identity and endpoint testability; this note corrected accordingly ([`2026-10-03_session5_seed_duplication_and_voi_identity_audit.md`](../audits/2026-10-03_session5_seed_duplication_and_voi_identity_audit.md)).
  - Documented L4 hardware retention deviation in `prereg/deviation_log.yaml`.
  - Formally marked Track D2/D3 cross-domain claims as superseded pending raw trace revalidation.

---

## 4. Exit Gate & Roadmap Evolution

- **Session 5 Exit Gate:** ❌ **NOT PASSED — DOWNGRADED TO INCONCLUSIVE.**

  The gate as previously written read "PASS (Conditional Robustness Confirmed)". That verdict is retracted. It rested on three claims that do not survive audit:

  | Original gate claim | Status after audit |
  |---|---|
  | "confirmed across 4 cost regimes (+8.29 % to +24.42 %)" | **Point estimate only.** Negative for the independent replicate in 3 of 4 moderate regimes; no CI or paired test exists. |
  | "across horizons $H \in \{2, 4\}$" | **Not established.** $H=4$ is −38.29 % on the independent replicate. |
  | "across all 12 site deletion subsets (+4.20 % to +19.78 %)" | **False for the independent replicate**, which is negative on all 12. |
  | "fails under severe penalties due to uncertainty-driven over-refinement" | **Mechanism retracted.** It is curvature-sign sensitivity; there is no uncertainty term. |

  The only component of the core hypothesis that survives as a *measured, correctly-computed* result is the negative one: belief-space VOI is 22.73 % worse than the capacity-matched critic under `high_penalty`. That is a genuine boundary condition and it is worth keeping.

  **Gate re-entry criteria** (all required before Session 5 may be cited as evidence for the co-state hypothesis):
  1. Three genuinely distinct teacher checkpoints; no fallback permitted.
  2. Per-window per-policy regrets committed, with a paired CI and site-clustered bootstrap on the *difference*.
  3. A pre-specified primary metric — this note used mean regret as the headline while the win rate reverses the ordering, so the metric must be fixed in advance.
  4. $\beta$ swept explicitly, including $\beta = -1$ (which is exactly `second_order_curvature`) and $\beta = 0$ (exactly `first_order`), so the functional identity is visible in results rather than hidden in a default.

- **Sequential Experimental Roadmap Update:**
  Per the user's directive, spatial allocation and policy distillation are decoupled:
  - **Stage 5.1 (Complete):** Reconcile evidence on CPU, audit bounds, narrow claims, and mark D2/D3 as superseded.
  - **Stage 5.2 (Complete):** Evidence-integrity audit of Session 5; seed duplication, functional identity and endpoint testability established; this note corrected and the exit gate downgraded. Session 6A is **not** blocked, because it does not depend on Session 5's endpoint — it is a separate fixed-budget spatial selection study whose spec already mandates every control Session 5 lacked.
  - **Session 6A (Next):** Fixed-Budget Spatial Patch Selection. The uncommitted `src/adjointrwm/spatial_selection.py` already replaces `delta_cov = effects.pow(2)` with a true epistemic $\Delta\Sigma$ from the model's own `state_logvar_head`, and `tests/test_spatial_selection.py` asserts $\beta = 0$ collapses exactly onto `curvature_scores`.
  - **Session 6B (Subsequent):** Policy Distillation & Decision-Focused Learning.
  - **Session 6C (Follow-up):** Systems & Hardware Latency Validation (measure end-to-end savings on L4).

---

## 5. What Survives, and What Must Not Be Cited

**Survives (measured, reproducible via `scripts/audit_session5_evidence_integrity.py`):**

1. All arithmetic in §§2A–2D, the LOSO bounding identity, and the std-vs-variance terminology corrections.
2. The 5-regime cost sweep as a sweep: heads retrained and actions reselected per regime, with the losing regime reported rather than dropped.
3. The `high_penalty` negative result (VOI 22.73 % worse than the matched critic) — correct number, corrected mechanism.
4. Throughput ≈ 28.4 kHz, consistent with Session 4 and above the 7.6 kHz gate.
5. The refusal baseline's pathology: mean regret 1.1765 with std 28.2, i.e. mean regret on this task is tail-dominated and must not be read as typical-window performance.
6. The omission-sensitivity caveat (this is not unseen-site generalisation), which was correct in the original note.
7. Throughput and the L4 compute-discipline record.

**Must not be cited:**

1. Session 5 as evidence that belief-space VOI outperforms a direct critic.
2. Any statement of the form "3 seeds" or "12,462 evaluations" as independent evidence.
3. Any claim that belief-space VOI and second-order curvature are distinct mechanisms, or that VOI carries an epistemic uncertainty term, as implemented in `src/adjointrwm/allocators.py`.
4. The Session 5 Exit Gate as a PASS.
5. Any mean-regret advantage quoted without its win rate and trimmed mean alongside.
