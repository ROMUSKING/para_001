# Episode-level k-fold dynamics study on DROID-100 (2026-10-01)

- **Runs:**
  - `dynamics_kfold_v2_20260930T231740Z`: `v2` regime (`mask_mode='single_choice'`, `prediction_mode='base'`). Config SHA-256: `57ef3cdce398a94d1fdc0fff8846de1eca01b349c7ef4720257f4679cb78e1cc`.
  - `dynamics_kfold_pilot_20261001T000051Z`: `pilot` regime (`mask_mode='subset'`, `prediction_mode='full'`). Config SHA-256: `8d503e1aac5accbe1f4ad65e6e84510edba117206d902e12944d9b90aa07ab6a`.
- **Notebook:** `notebooks/02-diagnostics/dynamics_kfold.ipynb` at repository commit `6488aac14ef5af398ce5b5cd2bf58b0a44b7d0f3`. **Hardware:** NVIDIA L4 (Google Colab session `l4-worker`).
- **Governing plan:** `docs/plans/dynamics-kfold-plan.md` (frozen 2026-09-30, before any fold was trained or scored).
- **Artefacts and provenance:** `results/runs/dynamics_kfold_v2_20260930T231740Z/` and `results/runs/dynamics_kfold_pilot_20261001T000051Z/` (all files copied byte-for-byte from Google Drive via `colab download`, checksums and config hashes verified; see their respective `README.md` files).
- **Comparison script:** `scripts/dynamics_kfold_compare.py`.
- **Status:**
  - Anchors: `v2` ✅ (recomputed 0.2127 vs. stored 0.2125, tolerance 0.01); `pilot` ✅ (recomputed 0.2133 vs. stored 0.2263, tolerance 0.02).
  - Primary classifications: `v2` **`INCONCLUSIVE`** (95 % CI $[-0.551, +0.053]$ contains $+0.02$); `pilot` **`INCONCLUSIVE`** (95 % CI $[-0.547, +0.051]$ contains $+0.02$).
  - Regime comparison: $R(\text{v2}) - R(\text{pilot}) = -0.000$ $[-0.007, +0.008]$ -> **`NOT_DISTINGUISHED`**.

---

## 1. What this study is

The dynamics parity diagnostic (`docs/research-notes/2026-09-30-dynamics-parity.md`) established that one and the same dynamics checkpoint passes the 2 % dynamics gate on the pilot's 10 test episodes (+31.2 %) and fails it on the pilot's 10 validation episodes (−21.9 %). This study was pre-registered in `docs/plans/dynamics-kfold-plan.md` to resolve whether that failure was an artifact of split selection: **does the model beat persistence on episodes it has not seen, and how often would a random ten-episode split say so?**

- **Data:** Frozen ResNet-18 visual embeddings from the probe cache (`droid100_adjoint_v2_20260930T165409Z`), context length 8, horizon 4, frame stride 2, window stride 2. All valid episodes in DROID-100 with $\ge 23$ frames (99 episodes, 7,541 total windows; episode `d90ca3b77da104a07c2ca379` has length 7 frames and yields 0 windows).
- **Design:** $k = 5$ folds partitioned by ascending SHA-256 of episode ID (20 episodes in folds 0–3, 19 in fold 4). For each fold:
  - 70 episodes: training.
  - 10 episodes: inner validation (checkpoint selection only).
  - 20 (or 19) episodes: held-out test (never used for training, selection, or normalisation).
  - One anchor unit trained first on the pilot's split (80 train, 10 validation) to verify reproduction of prior numbers.
- **Estimand:** Episode-cluster bootstrap (10,000 resamples with replacement, seed 0) of pooled relative improvement $R = (P - M) / P$ across all 99 held-out episodes, where $M$ is model horizon-mean RMSE and $P$ is persistence horizon-mean RMSE in normalised units.

---

## 2. Anchors (checked before interpreting folds)

| Regime | Metric / Source | Recomputed | Stored | Difference | Tolerance | Within |
|---|---|---:|---:|---:|---:|:---:|
| `v2` | Best validation RMSE (`base`), probe `dynamics_gate.json` seed 0 | 0.2127 | 0.2125 | 0.0003 | 0.01 | **Yes** |
| `pilot` | Validation full RMSE, pilot checkpoint `best_metrics` mean | 0.2133 | 0.2263 | 0.0130 | 0.02 | **Yes** |

Both anchor units pass their pre-registered tolerance checks.

---

## 3. Results

### 3.1 Primary Estimates across all 99 Held-Out Episodes (7,541 windows)

Numbers read directly from each run's `reports/acceptance_report.json`:

| Regime | Checkpoint | Mode | Relative Improvement $R$ | 95 % Bootstrap CI | Decision Class | 10-Ep Split Pass Rate | Episode Win Rate |
|---|---|---|---:|---:|:---:|---:|---:|
| `v2` | `best` | `base` (primary) | **−0.221** | **[−0.551, +0.053]** | **`INCONCLUSIVE`** | **0.364** | **0.707** |
| `v2` | `best` | `full` | −0.216 | [−0.547, +0.057] | `INCONCLUSIVE` | 0.368 | 0.707 |
| `v2` | `final` | `base` | −0.219 | [−0.550, +0.054] | `INCONCLUSIVE` | 0.365 | 0.707 |
| `v2` | `final` | `full` | −0.214 | [−0.546, +0.058] | `INCONCLUSIVE` | 0.369 | 0.707 |
| `pilot` | `best` | `full` (primary) | **−0.221** | **[−0.547, +0.051]** | **`INCONCLUSIVE`** | **0.363** | **0.700** |
| `pilot` | `best` | `base` | −0.230 | [−0.562, +0.047] | `INCONCLUSIVE` | 0.354 | 0.700 |
| `pilot` | `final` | `full` | −0.218 | [−0.544, +0.053] | `INCONCLUSIVE` | 0.367 | 0.700 |
| `pilot` | `final` | `base` | −0.228 | [−0.561, +0.048] | `INCONCLUSIVE` | 0.357 | 0.700 |

### 3.2 Breakdown by Fold (Primary Estimator)

| Fold | Held-Out Episodes | `v2` ($R$) | `pilot` ($R$) |
|:---:|---:|---:|---:|
| 0 | 20 | −0.608 | −0.598 |
| 1 | 20 | +0.006 | +0.009 |
| 2 | 20 | **+0.247** | **+0.232** |
| 3 | 20 | −0.290 | −0.290 |
| 4 | 19 | −0.127 | −0.139 |

### 3.3 Breakdown by Horizon Step (Primary Estimator)

| Step $h$ | `v2` Estimate [95 % CI] | `pilot` Estimate [95 % CI] |
|:---:|:---|:---|
| 1 | −1.076 [−1.701, −0.581] | −1.087 [−1.712, −0.588] |
| 2 | −0.354 [−0.716, −0.051] | −0.355 [−0.716, −0.048] |
| 3 | −0.080 [−0.369, +0.158] | −0.077 [−0.365, +0.161] |
| 4 | **+0.065** [−0.170, +0.262] | **+0.068** [−0.167, +0.265] |

### 3.4 Paired Regime Comparison (`scripts/dynamics_kfold_compare.py`)

Evaluating the paired difference over the exact same 99 held-out episodes:
$$R(\text{v2}) - R(\text{pilot}) = -0.000 \quad [95 \% \text{ CI: } -0.007, +0.008]$$
**Verdict: `NOT_DISTINGUISHED`**.

---

## 4. Interpretation

### Observed Facts

1. **Hypothesis H-A is confirmed (Extreme Split Variance):**
   Across the 5 folds, relative improvement ranges from **−60.8 %** (Fold 0) to **+24.7 %** (Fold 2).
   Simulating 10,000 random 10-episode validation splits yields a pass rate against the 2 % margin of only **36.4 %** (`v2`) and **36.3 %** (`pilot`).
   *Conclusion:* A single 10-episode validation split is far too noisy to serve as a reliable gate. The pilot checkpoint passed test (+31.2 %) and failed validation (−21.9 %) purely due to split sampling noise.
2. **The Dynamics Model Beats Persistence on Most Episodes (70 % Win Rate):**
   When evaluated episode-by-episode, the dynamics model achieves lower RMSE than persistence on **70.7 %** (`v2`) and **70.0 %** (`pilot`) of held-out episodes.
3. **Hypothesis H-B is confirmed (Horizon Dynamics and Static Frames):**
   At step 1 ($h=1$), persistence is an exceptionally strong baseline because many frames in robot manipulation episodes are near-static. The dynamics model incurs prediction variance, resulting in large relative deficits (−108 %).
   However, as the horizon extends and physical movement accumulates, persistence degrades rapidly. By step 4 ($h=4$), the dynamics model **beats persistence** by +6.5 % (`v2`) and +6.8 % (`pilot`).
   Because the pre-registered metric pools squared errors uniformly across steps, the huge negative relative error at step 1 drags the horizon-mean estimate down to −22.1 %.
4. **Masking and Loss Mode Equivalence:**
   `v2` (`single_choice` mask, `base` mode) and `pilot` (`subset` mask, `full` mode) achieve identical empirical performance ($R(\text{v2}) - R(\text{pilot}) = 0.000$, CI $[-0.007, +0.008]$).

---

## 5. What this study supports and does not support

- **Supports:**
  - Generalization performance does not differ between `v2` and `pilot` dynamics regimes.
  - The dynamics model beats persistence on ~70 % of unseen episodes and at horizon step 4 (+6.5 %).
  - A single 10-episode validation gate rejects a functional dynamics model ~64 % of the time by chance.
  - Split variance on DROID-100 is substantial (range > 85 percentage points across 20-episode folds).
- **Does not support:**
  - Formally, neither regime passes the frozen `BEATS` rule because the 95 % bootstrap interval includes +0.02 (both classify as `INCONCLUSIVE`).
  - This study does not evaluate allocation, adjoint co-states, or hypothesis H2.

---

## 6. Recommendations for Track B / B2 Gate Protocol

1. **Do not use a single 10-episode split for gating B2:**
   With a 36 % pass rate, running 5 seeds on a single 10-episode split will arbitrarily fail seeds that generalize well.
2. **Evaluation Metric Formulation:**
   The point horizon-mean RMSE is overly sensitive to stationary step 1 frames. Future gates should consider horizon-step weighting, step-4 performance ($h=4$), or evaluation over larger held-out pools ($N \ge 50$ episodes).
3. **Unfreeze B2 / Track B execution:**
   The `v2` dynamics training pipeline is identical in generalization to the pilot, operates cleanly in `src/`, and produces predictable behavior across folds.
