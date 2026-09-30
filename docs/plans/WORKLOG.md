# Work log

Append-only handoff log shared by every agent and human working in this repo. Newest entries go **at the top**. Read the latest entries before starting; add one before ending a session.

Each entry records:

- what changed;
- what was verified, and how (commands and results);
- what's open or blocked;
- what the next agent should do first.

Reference roadmap IDs (`docs/plans/roadmap.md`). Keep each entry under about 15 lines.

```
## YYYY-MM-DD HH:MM (tz) · <agent/tool> · <roadmap IDs>
- Changed:
- Verified:
- Open:
- Next:
```

---

## 2026-09-30 17:30 (BST) · Claude Code (web) · D4-1, Colab worker

- **Changed:** `reproduction.md` added to `results/runs/d4_1_learned_critics_20260930T122312Z/`; the Colab worker now exits when the inbox has been empty for 5 minutes and its notebook flushes Drive and releases the runtime (`colab_jobs.idle_limit_seconds`, `scripts/colab_worker.py --max-idle-minutes`, `WORKER_REF`; 3 new tests; CHANGELOG (q), handoff §5).
- **Verified:** the clean-worktree re-execution of D4-1 (`d4_1_learned_critics_20260930T154712Z`, 2197 s) matches: 25 of 32 files byte-identical, including all eight critic files and every validation table; the other seven differ only in a run id or a timestamp; the freeze hash is identical. `pytest` 311 passed and `python harness/check.py --base origin/main` 7/7 (with the worker tests).
- **Open:** the worker change is on the branch, not on `main`: to use it, open `colab_worker.ipynb` from the branch and set `WORKER_REF = 'ccr-88589394-9zz9zq'`, or merge. The queue still holds `1-ops-smoke-006.json` and `2-b2-probe-seed0-r2.json` (commit `ef7a75b`) until Roman starts the worker.
- **Next:** read `results/ops-smoke-006` and `results/b2-probe-seed0-r2` after the worker runs.

---

## 2026-09-30 17:10 (BST) · Claude Code (web) · D4-1, Colab queue, B2 probe

- **Changed:** D4-1 plan (frozen, redesigned once before any validation read), `adjointrwm.domains.critics` with 26 tests, notebook, `scripts/d4_1_tables.py` and `d4_1_estimator_probe.py`; imported run `d4_1_learned_critics_20260930T122312Z` (with README) and note `2026-09-30-d4-1-learned-critics.md`; roadmap, cross-domain plan, README, notebooks README, plan §10. Colab: `ops_smoke` now probes the data stack; the TFDS notebooks pin `tensorflow-metadata<1.18`; `load_droid` fails loudly (3 tests).
- **D4-1 result (from the run files, via `scripts/d4_1_tables.py`):** R0 fails in both cells (critics 6.1 to 10.3 times uniform's compute in `m4`, 1.5 to 1.8 in `m64`); co-state critic over direct critic 1.02 to 1.10 (`m4`) and 0.96 to 1.04 (`m64`), no interval entirely below 1, equal to the randomised control; at the hypothetical lookup price the exact co-state feature helps the head by about 8 % in `m4` (teacher-only value); `m64` inconclusive (estimator error 0.655 to 0.684, floor never met). D4 is not evidence for H2. Recommendation: do not spend the test family; put the effort on E1.1/B2.
- **Colab:** worker on an L4 ran `ops-smoke-001` (ok) and `b2-probe-seed0` (**failed in 20 s at the data load**: protobuf 5.29.6 with `tensorflow-metadata` 1.21.0 generated for protobuf 6.31.1, TFDS swallows the import error and exposes no `load`; reproduced locally, fixed by the pin). The worker stopped polling twice after `pip install --upgrade` jobs (12:35 and 12:53 UTC). **Queued in Drive `jobs/inbox`, waiting for a worker restart:** `1-ops-smoke-006.json` (checks the fix) and `2-b2-probe-seed0-r2.json` (pilot v2, `seeds=[0]`, 8 h), both pinned to commit `ef7a75b` of branch `ccr-88589394-9zz9zq`.
- **Verified:** `python harness/check.py --base origin/main` 7/7 before the D4-1 import; the table script on the imported folder prints exactly the tables in the note; the reproduction of D4-1 was still running when this entry was written (result to be added as `reproduction.md` in the run folder).
- **Open:** Roman restarts the worker; then read `results/ops-smoke-006` (top lines of `run_summary.md`) and `results/b2-probe-seed0-r2` (`acceptance_report.json`: `opportunity_gate_validation` and the dynamics gate for seed 0). A failed gate is a result. D2-0 needs a plan and an L4 job; D3-0 needs Roman's choice of a licence-cleared repository set and localisation benchmark.
- **Next:** add `reproduction.md` to the D4-1 run when the clean-worktree re-execution ends; then E1.1/B2 depending on the probe.

---

## 2026-09-30 13:10 (BST) · Claude Code (web) · D4-3, D1-0b, D4-1, Colab queue

- **Decisions (Roman, 2026-09-30):** "redesign its objective (D1-0b), want the varying-goal check, proceed with all practical CPU experiments and flag when to switch to Colab"; the repo was made public so the Colab worker can clone it (the token-support patch was declined and is not committed).
- **Changed:** imported runs `d4_3_varying_goal_20260930T084725Z` and `d1_0b_forecast_sensing_20260930T085746Z` (with READMEs and clean-worktree reproductions); research notes `2026-09-30-d4-3-varying-goal.md` and `2026-09-30-d1-0b-forecast-sensing.md`; roadmap, cross-domain plan, README (also repaired misplaced D4-0b/D4-2 links), notebooks README, plans §9/§10, CHANGELOG (n); **D4-1 plan written** (`docs/plans/d4-1-plan.md`), nothing trained yet.
- **Results (from the run files, via `scripts/d4_3_tables.py` and `scripts/d1_0b_tables.py`):** D4-3: R3v and R4v hold together in `m4` and `m64` only (co-state weight saves 11 to 17 % and 8 to 18 % of `cheap`'s compute) and with the table charged per instance the goal-aware arm needs 2.1 to 50 times `cheap`'s compute. D1-0b: G0 passes on tuning (0.522) and by the point rule on validation, not robustly (0.298 [0.023, 0.743]); G1 passes with an unstable ratio (0.875 [0.295, 1.347]); neither deployable dynamic policy keeps any headroom, and the best static policy beats them.
- **Verified:** `pytest -q` 279 passed before this entry's docs; the table scripts run on the imported folders and print exactly what the notes quote; reproductions: D4-3 11 of 18 files byte-identical, D1-0b 7 of 15 byte-identical, every other difference is a run id, a timestamp or a timing (READMEs).
- **Open:** D4-1 implementation (critics module with hand-written backprop, tests, notebook, timing smoke on train/tuning only, freeze, run). Colab: the worker needs to be re-run from `main` now that the repo is public; first job `ops_smoke`, then the one-seed pilot v2 probe. Not started: D2-0, D3-0 (D2-0 needs an L4).
- **Next:** build `src/adjointrwm/domains/critics.py` per the D4-1 plan §3-§6; do not read the validation family for any critic until `reports/frozen_before_validation.json` exists.

---

## 2026-09-30 07:37 (BST) · Claude Code (web) · D1-0

- **Decision (Roman, 2026-09-30):** "proceed as recommended": D4-1 stays closed, the D1/D2/D3 candidates are confirmed, E1.1 then B2 stay on the critical path (Colab; not runnable here).
- **Changed:** D1-0 plan, `adjointrwm.domains.sensor` with 26 tests, `scripts/fetch_smd.py`, `scripts/d1_0_tables.py`, notebook `d1_0_sensor_opportunity.ipynb`; imported run `d1_0_sensor_opportunity_20260930T062900Z` (with README); research note; roadmap, cross-domain plan, README, notebooks README, CHANGELOG.
- **Result (from the run files, via `scripts/d1_0_tables.py`):** G1 passes robustly (headroom 0.920 [0.886, 0.949]); non-learned dynamic policies keep 0.42 to 0.49 of it and cannot be separated; **labelled F1 does not improve with sensing (0.259 hold, 0.192 full)**, so the fidelity proxy does not track the task. Validation machines only; the 14 test machines were never downloaded.
- **Verified:** `pytest -q`: 204 passed; `python harness/check.py --base origin/main`; clean-worktree re-execution at the same commit is byte-identical except run id, timestamps, timings and hashes that cover them. Two smoke runs on tuning machines preceded the run and led to the oracle change (plan §10); no validation file existed before the freeze.
- **Open:** D1-0b (redesign D1's objective) vs D2-0/D3-0 vs D4-3: Roman's decision. E1.1 and B2 need Colab (GPU, DROID, Drive).
- **Next:** if Roman wants D1 kept, run the cheap tuning-machine diagnostic (does any simple detector's labelled F1 improve with sensing?) before any design. Otherwise start D2-0 or D3-0 with the card-and-gate pattern, checking the loss against the native endpoint first.

---

## 2026-09-30 01:12 (BST) · Claude Code (web) · D4-2, D4-1

- **Changed:** imported run `d4_2_flop_scoring_20260929T233556Z` (with README), wrote `docs/research-notes/2026-09-30-d4-2-flop-priced-scoring.md`, added `scripts/d4_2_tables.py`; updated roadmap (D4-2 done, D4-1 stays closed, D4-3 proposed), cross-domain plan, README, notebooks README, plan §12, CHANGELOG.
- **Result (from the run files, via `scripts/d4_2_tables.py`):** R1 in 3 of 7 cells, R3 in 1, none has both; a non-learned cheap estimator matches or beats the learned scorer in every cell; the co-state weight helps it only at `m` = 64 (about 10 %). Validation only, fixed goal per system, test family never generated.
- **Verified:** `pytest -q`: 178 passed; `python harness/check.py --base origin/main`: 7/7. The note's tables and numbers were checked programmatically against the run files.
- **Reproduced:** a clean-worktree re-execution at the same commit (`d4_2_flop_scoring_20260930T000230Z`, not imported) gave byte-identical scorers, `theta_tuning.csv`, `hidden_size_selection.csv`, `validation_summary.json`, `validation_work_precision.parquet` and manifests; the other files differ only in run id, a timestamp or a timing (`results/runs/d4_2_flop_scoring_20260929T233556Z/reproduction.md`, one repetition, same machine).
- **Open:** nothing blocks. The scratch worktree `wt_d42` was removed.
- **Next:** Roman decides on D4-3 (varying goal) versus returning to E1.1 / B2 and the licence-cleared domain cards (D1-0 SMD; D2-0 Qwen3-8B with MuSiQue or Qasper; D3 code-repository context).

---

## 2026-09-29 22:53 (BST) · Claude Code (web) · DL

- **Decisions (Roman):** add `.gitattributes` but do not compact `instances.json`; network access is on; proceed according to plan.
- **Changed:**
  - Added `.gitattributes` (`results/runs/**` as `linguist-generated`).
  - Licence survey pass 2: register 59 rows, 124 evidence URLs, 14 probes; `scripts/licence_survey.py` now reads JSON licence fields, HTML catalogue markup, Hugging Face gating, commits and parameter counts, probes Hugging Face datasets through the datasets server and runs a licence census over the repositories behind SWE-bench Verified. New note `docs/licences/survey-2026-09-29-pass2.md`; plan, roadmap, README and changelog updated.
- **Verified:** `pytest -q`: 151 passed. The note's model table (11 rows: parameters and BF16 weight sizes), verdict counts, evidence counts and the census were checked programmatically against `evidence.jsonl`, `register.csv` and `probes.jsonl`.
- **Result:** 17 `adopt`, 12 `adopt_with_conditions`, 5 `avoid`, 25 `unverified`. Licence candidates: D2 Qwen3-8B (Apache-2.0, licence file read) with MuSiQue and Qasper; D1 SMD, UCI electricity, Monash records; D3 code-repository context (JOB's IMDb data is non-commercial; STATS-CEB has no licence). Owner statements conflict for TriviaQA and are silent for LongBench v1. Datasets server returned 501 for LongBench v2 (recorded as a failed probe).
- **Open:** Roman to confirm the D1, D2 and D3 candidates before any domain card is written; QuALITY annotation licence, TriviaQA, TPC terms and the licence files of the checkpoints that had none in their listing; the Hugging Face file CDN is still unreachable (no dataset file was downloaded from Hugging Face).
- **Next:** D4-2 (CPU): learned amortised scorer priced at its measured cost, higher-dimensional system, one factor varied at a time, wider `θ` grid on the tuning family only; freeze regime and price before reading validation. E1.1 and B2 still need Colab.

---

## 2026-09-29 20:40 (BST) · Claude Code (web) · DL

- **Decision (Roman):** licence survey first, then the rest.
- **Changed:**
  - Added `docs/licences/` (register of 53 sources, fetched evidence, structure-only probes, survey note), `scripts/licence_survey.py` and `tests/test_licences.py`. Updated the cross-domain plan, roadmap, README, docs index, AGENTS map and changelog.
  - Answered the diff-size question: 33,561 of 43,388 lines added on the branch against `main` were committed run artefacts (mostly pretty-printed `instances.json`), but only 2.84 MB in 76 files.
- **Verified:**
  - Every row that claims a read points at a URL recorded with HTTP status 200 (`tests/test_licences.py`); the Llama clauses and the QuALITY licence counts were re-checked against the fetched text and data.
  - `pytest -q`: 143 passed. `python harness/check.py --base origin/main`: 7/7.
- **Result:** 4 `adopt`, 10 `adopt_with_conditions`, 3 `avoid`, 36 `unverified`. Hugging Face, UCI, Zenodo, PhysioNet, Kaggle, TPC and others are blocked by the sandbox network policy, so no checkpoint card and no non-GitHub dataset page was read.
- **Open:** pass 2 needs those hosts reachable (Environment settings, Network access) or a Colab run of `python scripts/licence_survey.py snapshot` and `probe`. QuALITY's annotation licence is unstated. D3's graph domain is undecided.
- **Next:** unblock the hosts, re-run the snapshot, update the 14 `unread` rows by hand, then decide D2-0 and D1-0 candidates. D4-2 (CPU) does not depend on this.

---

## 2026-09-29 19:20 (BST) · Claude Code (web) · D4-0b, D4-2, recommendation

- **Changed:**
  - Added pass-based (batch) allocation, work-precision evaluation, scoring-price what-ifs and a three-family localisation sweep to `adjointrwm.domains` (32 domain tests), the D4-0b notebook, `scripts/d4_0b_tables.py` and `scripts/d4_0b_interpolation.py`.
  - Imported `results/runs/d4_0b_adaptivity_20260929T174347Z/` (result) and `…T173428Z/` (first execution, superseded). Wrote `docs/research-notes/2026-09-29-d4-0b-where-adaptivity-pays.md`.
  - Added roadmap §2.1 (recommended sequence, stop-losses) and the D4-2 milestone; updated the cross-domain plan, README and notebooks index.
- **Verified:**
  - Second run reproduced from a clean worktree at the same commit: eight result files byte-identical, the rest differ only in the run id (run README).
  - The 14 imported files match the source directory by SHA-256; the config hash re-derives.
  - Every number in the note's tables and prose was checked against `scripts/d4_0b_tables.py` output; the run's own report is reproduced by the rule code inside the script (asserted).
  - `pytest -q`: 135 passed.
- **Result:** no candidate regime at the real scoring price; the frozen rule is met only for `sharp` at hypothetical prices ×0.25 and ×0. All tuned `θ` are at the grid edge (0.95). Post-hoc interpolation weakens the `sharper` co-state result and adds `smooth` ×0 as a marginal cell that the interpolation bias (≤ 6.1 %) cannot exclude.
- **Open:** D4-2 design (measured price of a learned scorer, higher-dimensional system, wider `θ` grid on the tuning family only). E1.1 and B2 are still waiting to be run on Colab. Roman's open questions: DL timing, whether D1 waits for DL.
- **Next:** run E1.1 on Colab; start D4-2 on CPU. Do not read the test family (seed 2002) until a D4-2 or D4-1 design is frozen.

---

## 2026-09-29 17:40 (BST) · Claude Code (web) · D4-0, DL

- **Decision recorded (Roman):** D4 first; the other domains wait for research into permissively licensed content and testing. Added milestone DL (licence survey) and D4-0b to the plan and roadmap.
- **Changed:**
  - Fixed the D4 action ledger: a refinement is charged the re-solve it causes. The first D4 execution had charged 1 step (never committed).
  - Added equal-compute evaluation (`run_policy(compute_budget=…)`, `evaluate_at_compute`, `compute_level_summary`) and a stricter D4-1 opening rule.
  - Ran and imported D4-0: `results/runs/d4_time_stepping_20260929T162518Z/` (main) and `…T162914Z/` (exploratory signed-error variant).
  - Wrote `docs/research-notes/2026-09-29-d4-0-adaptive-time-stepping.md`.
- **Result:** correctness ✅, rate-budget opportunity ✅, equal-compute payoff ❌. Co-state weighting wins at equal refinement count; plain uniform refinement wins at equal total compute, on both splits and under both objectives. D4-1 stays closed.
- **Verified:**
  - `pytest`: 123 passed.
  - All hashes typed into the run READMEs were checked against the files.
  - A clean re-execution of the same commit (a `git worktree`) reproduced every result file byte for byte.
  - A programmatic claim check of the note against the committed files found three rounding typos, now fixed.
  - The refinement-count artefacts are byte-identical to the first execution's.
- **Open:**
  - Whether to run DL next or after D4-0b, and whether D1 also waits for DL (roadmap Q8).
  - D4-0b design: mark many intervals per scoring pass, incremental estimates, or a learned amortised scorer.
- **Next:** D4-0b, validated at equal compute on validation with the test family unread. Carry the lesson to pilot v2: charge scoring costs and compare at matched compute.

## 2026-09-29 13:50 (BST) · Claude Code (web) · D0, D4-0

- **Changed:** added Track D, cross-domain generalisation, from Roman's domain-portfolio brief:
  - the plan, `docs/plans/cross-domain-plan.md`;
  - `src/adjointrwm/domains/`: interface, runner, ledgers, metrics and D4;
  - the D4-0 notebook, `notebooks/04-domains/d4_adaptive_time_stepping.ipynb`;
  - updates to the roadmap, READMEs and `AGENTS.md`.
- **Design findings while building D4:**
  - 1-D Poisson mesh refinement cannot tell adjoint weighting from goal-local weighting (the weight `z − I_h z` is local), so D4 uses time stepping instead.
  - The signed QoI error rewards lucky cancellations, so the objective is the bound `Σ|Λᵀτ|`.
  - The greedy one-step oracle can be beaten, so regret uses a best-known curve.
- **Verified:**
  - `pytest`: 119 passed. `harness/check.py --base origin/main` passes 7/7.
  - The D4 error representation holds to about 1e-16 relative error, and the reference matches a closed form to 1e-11.
  - The D4-0 notebook runs end to end on CPU (smoke config; numbers not recorded).
  - The previous push's CI passed.
- **Open:**
  - D4-0 has not been run at full size; it takes a few CPU minutes.
  - Roadmap open questions 8–12 need Roman's decision (first domain, D3 choice, D2 model and data, ERP data, N6 amendment).
- **Next:** run D4-0 and import it. If the opportunity gate passes, design D4-1 (learned direct critic vs co-state-featured critic).

## 2026-09-29 13:25 (BST) · Claude Code (web) · B0, E2.1, E2.2, N0.1, N0.3

- **Changed:** added Track B, the benchmark against rival models. It has four parts:
  - the plan, `docs/plans/rival-benchmark-plan.md`;
  - two notebooks: `03-benchmarks/rival_world_models_droid100.ipynb` (B1) and `01-production/AdjointRWM_Production_Pilot_v2.ipynb` (B2 = E2.1/E2.2);
  - the tested modules they import: `data`, `eval`, `models`, `training`, `allocators`, `features`, `benchmark` and `io`;
  - updates to the roadmap, READMEs, CI (CPU torch) and `pyproject`.
- **Verified:**
  - `pytest`: 103 passed (56 passed and 3 modules skipped without torch). `python harness/check.py` passes.
  - The lifted pilot model has 27,360,798 parameters, matching `model_manifest.json`. The stage-1 loss equals a verbatim copy of the pilot's.
  - The exact co-state matches float64 finite differences.
  - Pause-and-resume gives bit-identical weights. This test found a `DataLoader` RNG drift, now fixed.
  - Both notebooks ran end to end on CPU with stubbed Colab/TFDS and random fixture episodes (code paths only, not evidence), including a run resumed over 5 sessions.
  - Rival constants come from the official GitHub configs. arXiv was blocked from the sandbox.
- **Open:**
  - Nothing has run on real data or a GPU.
  - Until this branch is merged, set `REPO_REF` in both notebooks to the branch name.
  - The split-parity check needs the pilot's `data_manifest.json` on Drive; it is listed in `DRIVE_INVENTORY.csv`.
  - Roadmap open questions 5–7 (endpoint, TD-MPC2 input, B4) need Roman's decision.
  - The CUDA resume check is still to do.
- **Next:**
  1. E1.1: the opportunity audit on the v1 checkpoint. It is cheap and informs the B2 reading.
  2. B2: run pilot v2.
  3. B1: run the rival benchmark on L4.
  4. Import each run with `/import-run`, then write the note with `/research-note`.

## 2026-09-29 11:30 (BST) · Claude (claude.ai agent) · harness

- **Changed:** added the cross-tool agent harness:
  - `AGENTS.md` (root plus `notebooks/` and `results/`), the single source of truth;
  - `.agents/skills/` with 5 portable skills;
  - `harness/sync.py`, which generates `CLAUDE.md` imports, `.claude/skills`, `.agents/workflows` and `.opencode/commands`;
  - `harness/check.py`, the definition of done;
  - the `protect_paths` hook, `.claude/settings.json`, `opencode.json` and `.gemini/settings.json`;
  - a CI job.
- **Verified:**
  - `python harness/check.py --base origin/main` passes 7/7, with 14 tests.
  - The hook blocks edits to existing run artefacts, dated audits and generated files, and allows new files.
  - Negative tests: hand-edits, stale skills and a modified run README are all caught.
  - Per-tool paths were checked against the docs; see the matrix in `harness/README.md`. Mid-build, `.agent/*` and `GEMINI.md` copies were removed as redundant, and the OpenCode rule order was fixed (last match wins).
- **Open:** Codex project-level `.codex/config.toml` is unverified (only matters once MCP servers exist). The MCP server list is empty.
- **Next:** roadmap E1.1 (run the opportunity audit notebook in Colab), then N0.1 (move the pilot code into `src/`).

## 2026-09-29 10:44 (BST) · Claude (claude.ai agent) · docs

- **Changed:** imported the comprehensive plan, the production docs and two audits; rewrote the roadmap, README and `.gitignore`.
- **Verified:** links resolve and 8/8 tests pass.
- **Next:** E1.1 and E1.2 (import the second pilot run `droid100_adjoint_20260929T090015Z`).
