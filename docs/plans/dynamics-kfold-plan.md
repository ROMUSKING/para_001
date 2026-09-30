# Episode-level k-fold dynamics study (plan, frozen before any fold is trained)

**Status: frozen 2026-09-30, before the notebook exists and before any fold has been trained or scored.** Changes after the first result is read go in §9 with a date and a reason, and a change after a result has been read is labelled as such. Roadmap: E3.2's question ("beats persistence on held-out episodes") asked over all 100 DROID-100 episodes; it also feeds the B2 gate decision (roadmap §2.1, item 2).

## 1. Why

The dynamics parity run (`docs/research-notes/2026-09-30-dynamics-parity.md`) showed that one and the same checkpoint passes the 2 % dynamics gate on the pilot's test split (+31.2 %) and fails it on the validation split (−21.9 %), ten held-out episodes each, and that pilot v2's seed-0 checkpoint fails it on validation (−12.7 %). The pass or fail of a single ten-episode split is therefore not a reliable statement about held-out episodes. This study asks the question once, over every episode: **does the model beat persistence on episodes it has not seen, and how often would a random ten-episode split say so?**

## 2. Design

- **Data.** The feature cache of the pilot v2 probe run (`droid100_adjoint_v2_20260930T165409Z`): 100 valid DROID-100 episodes, frozen ResNet-18 embeddings, frame stride 2, windows of 8 context steps and a 4-step horizon at stride 2, the pilot's layout. No download, no new data.
- **Folds.** `k = 5`. An episode's fold is its rank by ascending SHA-256 of the episode id (the rule `adjointrwm.data.windows.split_order_key` the pilot's split uses), modulo 5: 20 episodes per fold, every episode held out exactly once. Split by episode before any window is cut.
- **Per held-out fold.** The other 80 episodes form the pool. In the same hash order, the first 10 pool episodes are the **inner validation** episodes (checkpoint selection only) and the remaining 70 are the training episodes. The input normaliser is fitted on the 70 training episodes. **The 20 held-out episodes are never used for training, selection or normalisation.** (The pilots trained on 80 episodes; here 70, which can only disadvantage the model; see §7.)
- **Regimes** (two jobs, one regime each, same folds): `pilot`: `mask_mode='subset'`, `prediction_mode='full'`; `v2`: `mask_mode='single_choice'`, `prediction_mode='base'` (each regime's own gate mode is its primary mode). Everything else is identical to pilot v2's configuration: model `d_model` 512, 6 layers, 8 heads, feed-forward 2048, dropout 0.10, 4 candidates with costs (1, 1, 1.5, 2), `rate_beta` 0.002; 1500 steps, batch 64, learning rate 3e-4, weight decay 0.05, warmup 100, gradient clip 1.0, evaluation every 100 steps, BF16 autocast; training by `adjointrwm.training.train_job`. Training seed for fold `f` is `20260928 + 1 + f`. These are the pilot's training code path as implemented in `src/`, not the pilot notebook's own loop.
- **Anchor unit (one per regime, trained first).** Train on the pilot's split (its 80 training episodes, its 10 validation episodes for selection, seed `20260928`, exactly as the probe's seed 0). Compare its best validation score with the stored number: `v2`: `dynamics_gate.json` of the probe run, seed 0, `model_rmse` 0.2124711301360882 (mode `base`); `pilot`: the validation `full` RMSE stored in the pilot checkpoint's `best_metrics` (the mean of `full_rmse_by_horizon`). Tolerances, fixed now as policy values and not measured: **0.01 absolute for `v2`** (same code path, same seed, same data) and **0.02 absolute for `pilot`** (different training loop; the pilot's stored number is itself weighted differently from ours, by up to 0.0036 in the parity run). The anchor unit scores only its own validation windows; the pilot's test episodes are not read. **If an anchor is outside its tolerance the report says `ANCHOR_FAILED`, the tables are still written, and no classification or reading is made.**
- **Scoring of a held-out fold.** For the checkpoint selected by inner validation (**primary**) and for the final-step weights (secondary, no selection; scored from the in-memory model right after training because `train_job` deletes `latest.pt`), in the regime's own mode (primary) and the other mode (secondary): per held-out episode, the window count and, for each of the 4 horizon steps, the sum over windows of the state-dimension-mean squared error, for the model and for persistence (the last observed state repeated). BF16 autocast.
- **Estimand.** Pool all 100 held-out episodes (each scored by the model that did not see it): horizon-mean RMSE of the model `M`, of persistence `P`, and the **relative improvement `R = (P − M) / P`**, in normalised units. Each fold's model uses its own training-fold normaliser for its own held-out episodes; persistence is scored in those same units per fold, then pooled as summed squared errors.
- **Uncertainty.** Episode-cluster bootstrap: 10,000 resamples of the 100 episodes with replacement, seed 0, `R` recomputed from the resampled summed squared errors and counts; 95 % percentile interval. The same for each horizon step (descriptive).

## 3. Decision rule (frozen)

With the gate's margin `m = 0.02`, for the primary estimate of each regime:

| Class | Condition | Reading |
|---|---|---|
| `BEATS` | lower bound of the 95 % interval of `R` > `m` | the model beats persistence by at least the gate margin on held-out episodes |
| `WORSE_THAN_PERSISTENCE` | upper bound < 0 | the model is worse than persistence on held-out episodes |
| `FAILS` | upper bound < `m` and not `WORSE_THAN_PERSISTENCE` | the model does not reach the gate margin (it may be slightly better than persistence) |
| `INCONCLUSIVE` | otherwise | the interval contains `m` |

These classes describe the regime as trained here; they are not a verdict on H2.

## 4. Descriptive outputs (no decision attached)

- **Gate-pass rate of a random ten-episode split:** 10,000 random subsets of 10 of the 100 held-out episodes (without replacement, seed 0); the fraction whose pooled `R` ≥ `m`, and the 5 %, 50 % and 95 % quantiles of `R` across subsets. This bears directly on how a single ten-episode validation gate (B2) behaves for this model.
- `R` per fold (20 episodes each); the fraction of the 100 episodes on which the model's own horizon-mean RMSE is below persistence's; `R` by horizon step; the secondary (final-step, other-mode) versions of the primary numbers.
- **Between regimes** (after both jobs, `scripts/dynamics_kfold_compare.py`): the paired difference `R(v2) − R(pilot)` over the same episodes with a paired episode-cluster bootstrap interval (10,000 resamples, seed 0). Labelled `V2_BETTER`, `PILOT_BETTER` or `NOT_DISTINGUISHED` by whether the 95 % interval excludes 0. The folds and episodes are identical across regimes.
- Per-episode persistence RMSE and model RMSE are saved (`artifacts/episode_errors.csv`), so that the near-static-motion hypothesis H-B of the parity note can be tested offline without another run.

## 5. What each outcome would change (recommendations, not automatic)

- **`BEATS` in a regime:** the model generalises to unseen episodes at the gate margin; the parity failure is then mostly a property of the ten-episode split, and the gate-pass rate of §4 says how often B2's single-split gate would block a good model. The B2 gate design is then a statistical-power decision for Roman.
- **`FAILS` or `WORSE_THAN_PERSISTENCE` in both regimes:** the dynamics model does not beat persistence on held-out DROID-100 episodes under this training recipe; five-seed B2 is moot until dynamics changes (Track E, roadmap E3.x).
- **`INCONCLUSIVE`:** the data cannot separate the model from persistence at the margin; more episodes or a different recipe would be needed. No rerun with changed settings to reach a class.
- **`ANCHOR_FAILED`:** the training pipeline in this notebook does not reproduce a stored number; the fold results are not interpreted, and the gap itself (run-to-run or pipeline) becomes the finding.

## 6. Cost and logistics

- Per regime: 6 trainings (5 folds + the anchor unit) at about 496 s each on the L4 (the probe's measured 1500-step job, `DONE.json` `session_seconds` 496.09), plus scoring and the cache restore (1 min 4 s in the parity run): about 55 minutes. Two jobs: about 110 minutes of L4. Time limit per job 2 hours (allow-list maximum 3).
- L4 only. Checkpoints stay on the VM's local disk and are deleted after each unit; their SHA-256 values are recorded. Only small files go to Drive (per-unit results, per-episode errors, logs, report). Resumable at unit granularity (`RESUME_RUN_ID`).
- **Use of the pilot's test episodes.** The folds cover all 100 episodes, so the pilot's ten test episodes are each held out once and are training or inner-validation data for the other folds. The anchor unit does not read them. These k-fold models are diagnostics: no allocator, critic or H2 quantity is computed from them and B2 does not reuse them, so the test split keeps its role in B2 (read once, by B2's own models). The pilot itself had already read those ten episodes for its dynamics gate.

## 7. Limits (stated before the result)

- **One training seed per fold.** Episode-sampling variance is in the interval; training-seed variance is not.
- **70 training episodes instead of 80** (the inner validation takes 10); this is pessimistic for the model relative to the pilots. The anchor unit uses the pilots' 80.
- DROID-100 only; normalised units; horizon-mean RMSE, a point summary that hides the step-1 persistence advantage (reported by horizon).
- Overlapping windows within an episode are dependent; the episode-cluster bootstrap accounts for that, plain window counts would not.
- The pilot regime is the pilot's loss and prediction mode in `src/`, not the pilot notebook's own loop or sampler.
- Nothing here tests H2 or the allocator.

## 8. Files

`src/adjointrwm/eval/dynamics_parity.py` (k-fold helpers), `tests/test_dynamics_kfold.py`, `notebooks/02-diagnostics/dynamics_kfold.ipynb` (allow-listed, L4, overrides `regimes` and `resume_run_id`), `tests/test_dynamics_kfold_notebook.py` (CPU smoke test on tiny random fixtures), `scripts/dynamics_kfold_compare.py` and `tests/test_dynamics_kfold_compare.py`.

## 9. Dated changes

- **2026-09-30 (implementation note, before any fold ran):** while building the notebook the stored anchors, including the pilot checkpoint's hash check, were moved to the start of the run so that a wrong file fails in seconds and not after the trainings; the design is unchanged. The notebook keeps every frozen value of §2 to §4 as a literal in its configuration cell, and a test checks that they are still there.
- **2026-09-30 (correction, before any fold ran):** §6 said "the test split of the pilot is never built". That is wrong for a k-fold over all 100 episodes (the pilot's test episodes are held out once and used for training in the other folds) and is replaced by the paragraph "Use of the pilot's test episodes" in §6. The design, the anchors, the decision rule and the outputs are unchanged. The report field `test_split_read` of the parity notebook has no counterpart here; the report records `kfold_uses_all_100_episodes: true` and `anchor_reads_pilot_test_episodes: false` instead.
- **2026-09-30 (wording only, before any fold ran):** §2 named the ordering function as `adjointrwm.data.split_order_key`; it lives in `adjointrwm.data.windows` and is not exported from `adjointrwm.data`. The rule itself is unchanged.
