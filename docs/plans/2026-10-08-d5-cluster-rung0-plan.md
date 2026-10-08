# D5 cluster admission: job-selection domain card and Rung-0 opportunity gate

**Written:** 2026-10-08, after downloading 20 slice parts (10 job_events + 10
task_events, ~15MB) and inspecting the v2.1 positional schema. Test data (usage
tables) never fetched. **Status:** DRAFT for peer review.
**Roadmap:** new domain D5 (Tier P) · **Licence:** CC-BY-4.0 Google clusterdata
(register `google_cluster_data`, `adopt`; bucket world-readable).
**Hardware:** CPU-only. No GPU is needed for a Rung-0 (non-learned policies on
event tables); GPU comes only if a later gate requires it.

## 1. Question and scope

**Question.** For Borg job admission under a concurrency budget, is there room
for adaptive selection: does an allocation that knows realised job outcomes (a
privileged oracle) beat the best fixed admission rule by at least 15% of the
normalised area at equal budget, and how much room can a deployable rule using
only submit-time fields keep?

**Rung-0 gate.** Non-learned policies only; no H2 statement. Windows pool rows
from all slice parts sorted by trace time (parts are hash-sharded, not
time-sharded). **No usage tables, no L4, no reader.**

## 2. Domain card

| # | Requirement | D5 |
|---|---|---|
| 1 | Structured state | Trace-time windows of length W=3600s tiled over the slice span; candidate pool per window = jobs with a SUBMIT event in [t0, t0+W). Timestamps are microseconds (v2.1 schema); W and H below are converted accordingly. |
| 2 | Downstream objective | `J`, lower is better: **priority-weighted miss rate** `J_w(S) = 1 − V(S)/V(pool)`, where `V(X)` = Σ priority(j) over jobs j∈X that COMPLETE, priority(j) = max task priority at submit, COMPLETE = last job event in [submit, submit+H] is FINISH(4), H=21600s (6h). Windows whose pool has zero completable value are dropped (empty denominator, disclosed counts). **Censoring (review):** drop windows with trace_end − window_end < H, where trace_end = max event timestamp in the slice (computed at runtime, frozen rule). **Zero-task jobs** (no task rows: priority undefined) are dropped from pools (disclosed count), never imputed. **Missing-info rows:** any job with a non-empty missing-info event is dropped (strict; disclosed count). Outcomes are therefore exact on the retained set. |
| 3 | Legal actions | `admit(j)`; exactly k admissions per window; `stop` unused (exact-k). |
| 4 | Constrained resource | Admission slots (ledger R): k ∈ {1, 2, 4, 8, 16, 32} per window. Policy computation not charged. |
| 5 | Measurable effects | Exact: outcomes from events, no estimator. Oracle = top-k by realised V (ties → lower job id). Exact up to ties. |
| 6 | Baselines | Fixed: FIFO (earliest submit), highest-priority-first, most-tasks-first. Dynamic deployable (submit-time fields only): priority×tasks, earliest-submit-among-high-priority (priority≥9 then FIFO). Oracle privileged. **Learned policies are Rung 1.** |
| 7 | Privilege | P0: submit time, priority, task count, scheduling class, user hash. P3: all later events (completions, fails, kills). |
| 8 | Data | `clusterdata-2011-2` slice: job_events + task_events parts 00000–00009 (pinned by per-part SHA-256 in the run manifest; kept out of git). CC-BY-4.0 with attribution in note + manifest. |
| 9 | Opportunity | The gate of §4. |
| 10 | Regime | Selection among known arrivals with unknown outcomes (hindsight value, not VOI acquisition — the arrivals are visible, their outcomes are not). |

## 3. Splits and budgets (frozen)

Windows ordered by trace time over the slice span [Tmin, Tmax]. **Censoring
first (amendment 2026-10-08, reviewed):** drop windows with trace_end −
window_end < H (trace_end = max event timestamp in the slice, computed at
runtime); then split the survivors at their median start (tuning earlier,
validation later — forward time direction preserved); drop the two windows
straddling the split only when ≥4 eligible windows survive. Rationale (review):
adjacent-hour interleaving leaks diurnal regime; the original full-span
midpoint left validation entirely censored on short slices. Within a block,
windows remain correlated — reported, not corrected; the fixed-policy choice on
tuning is a single low-dimensional selection.
Cap: first 400 tuning / 200 validation windows with non-empty completable pools
(frozen counts, disclosed drops). Best fixed = lowest mean normalised area on tuning. Budgets above.

## 4. Gate (frozen; D1-0 §7.2B-1 pattern)

- **G−1 priority diagnostic (pre-gate, review):** Shannon entropy + top-1 mass of
  the priority distribution over completable tuning jobs. If top-1 mass > 70% or
  H < 1.5 bits, the value metric has no granularity: record G1 as uninformative
  and close the study (redesign, never silent redefinition of value).
- **Area (review):** per window, trapezoidal area of J(k) over k/kmax on the
  frozen grid; J is already normalised (miss rate in [0,1]). Machine/window
  means: mean over windows (equal weight). G1 = area(fixed)−area(oracle) over
  area(fixed).

- **G0 correctness:** every J in [0, 1]; oracle J ≡ 0 wherever the pool's
  completable value fits in k; deployable rules never read P3 (boundary test:
  shuffling outcome fields must not move selections).
- **G1 opportunity:** relative headroom area(fixed−oracle)/area(fixed) ≥ 0.15.
- **G2 deployable share:** best deployable keeps ≥ 0.50 of the headroom, else
  redesign candidates (no learned rescue). Floor (review): if headroom < 0.10,
  G2 is uninterpretable — reported only, no pass/fail.
