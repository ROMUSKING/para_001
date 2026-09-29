# Plan: training and benchmarking against rival models (Track B)

**Written:** 2026-09-29 · **Status:** proposed; nothing in this plan has been run yet · **Owner:** Roman
**Roadmap rows:** B0–B5 (new), and it implements E2.1/E2.2 and part of N0.1/N0.3 · **Notebooks:** [`03-benchmarks/rival_world_models_droid100.ipynb`](../../notebooks/03-benchmarks/rival_world_models_droid100.ipynb), [`01-production/AdjointRWM_Production_Pilot_v2.ipynb`](../../notebooks/01-production/AdjointRWM_Production_Pilot_v2.ipynb)

This plan does not change the governing protocol. It adds a benchmark track that the protocol and the production plan already ask for (production plan §3.1 and §13.2; comprehensive plan §5.5 and §5.5.1), and says how to run it on Colab L4 without breaking the research-integrity rules.

---

## 1. Two different "rival" questions, kept apart

| Benchmark | Question | What a result can support | What it can never support |
|---|---|---|---|
| **B-WM: rival world models** | On the same DROID data, split, inputs and budget, does the AdjointRWM dynamics model predict the future as well as published world-model families? | "The dynamics substrate is (or is not) competitive with re-implemented rival families at ~27 M parameters on DROID-100." | Anything about H2. A better predictor says nothing about whether the co-state allocates better than the direct critic. |
| **B-AL: rival allocators** | On a frozen dynamics model, does the adjoint allocator beat the direct critic and every other required allocator? | Evidence on H2 *for DROID refinement candidates* (Track E). | A confirmatory H2 verdict, which still goes through N1/N2 (LQTree, HJoinBench) and gate G-H2. |

B-WM is new. B-AL is roadmap E2.1/E2.2 (pilot v2), with the allocator arms that the plan requires and that pilot v1 didn't have.

## 2. Rival world models

All rivals are **re-implemented in PyTorch inside `src/adjointrwm/models/`** at a matched parameter budget, on the same frozen features. They are labelled "-style" everywhere: we do not run the official training code, so no result may be reported as the performance of the official method.

Sources were checked on 2026-09-29 against the official repositories (arXiv itself was not reachable from the agent sandbox, so paper details come from the official code and configs).

| Arm (`--arm`) | Rival | Official code, licence | Taken from the official config/code | Our deviations |
|---|---|---|---|---|
| `adjoint_rwm` | Reference: pilot v1 architecture | this repo | `notebooks/01-production/AdjointRWM_Production_Pilot.ipynb`, lifted verbatim | none; parameter names are identical so pilot checkpoints load strictly |
| `dreamerv3_rssm` | DreamerV3 (Hafner et al., arXiv:2301.04104) | [danijar/dreamerv3](https://github.com/danijar/dreamerv3), MIT | `configs.yaml`: `stoch: 32`, `unimix: 0.01`, `free_nats: 1.0`, loss scales `dyn: 1.0, rep: 0.1, rec: 1.0`; size presets scale `deter = 8·hidden`, `classes = hidden/16` (e.g. `size25m`: deter 3072, hidden 384, classes 24); `rssm.py`: block-GRU core, prior from deter, posterior from deter + tokens, KL with stop-gradient on the other side and a free-nats floor | world-model part only (no reward, actor or critic); symlog on vector inputs but decoder heads use the shared prediction objective (§4) instead of symlog-MSE; open-loop evaluation uses the mean of 8 prior samples |
| `tdmpc2` | TD-MPC2 (Hansen et al., arXiv:2310.16828) | [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2), MIT | `config.yaml`: `simnorm_dim: 8`, `num_enc_layers: 2`, `horizon: 3`, `rho: 0.5`, `consistency_coef: 20`, `lr: 3e-4`, `enc_lr_scale: 0.3`, `grad_clip_norm: 20`; `layers.py`: SimNorm, NormedLinear (Linear → LayerNorm → Mish); `tdmpc2.py`: latent rollout with `rho**t`-weighted MSE to encoder targets computed under `no_grad` | reward/value/policy heads replaced by state and visual read-out heads (offline data has no reward); the encoder sees the 8-step context by frame stacking so it gets the same information as the other arms (single-frame is an ablation, §8); width scaled as `latent = mlp = w`, `enc = w/2` (our rule, not an official preset) |
| `dino_wm` | DINO-WM (Zhou et al., arXiv:2411.04983) | [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm), MIT | `conf/encoder/dino.yaml`: `dinov2_vits14`, `x_norm_patchtokens`; `conf/predictor/vit.yaml`: depth 6, heads 16, mlp_dim 2048, dropout 0.1; `models/vit.py`: block-causal frame mask, vit-pytorch attention with 64-d heads; `conf/train.yaml`: action and proprio embeddings (dim 10 each) concatenated to every visual token | patch tokens average-pooled to a 2×2 grid plus CLS per camera (10 tokens per frame) so 100 episodes fit in RAM; no VQ-VAE pixel decoder (pixels are not scored); state and visual read-out heads |
| `vjepa2_ac` | V-JEPA 2-AC (Assran et al., arXiv:2506.09985) | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2), mostly MIT (some files Apache-2.0) | `configs/train/vitg16/droid-256px-8f.yaml`: predictor depth 24, 16 heads, `auto_steps: 2` (rollout loss), `loss_exp: 1.0` (L1), `normalize_reps: true`, 4 fps | the *predictor recipe* only, at matched size (depth 24, narrow width) on our frozen features; the released ViT-g encoder and 24×1024 predictor are not used in the primary comparison (see B4) |

**Visual encoders** (frozen, a declared factor, never mixed within one comparison):

- `resnet18` — torchvision ImageNet ResNet-18 global features, 512-d per camera. This is exactly the pilot's input and target.
- `dinov2_vits14` — [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2), code and weights Apache-2.0, loaded with `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`; CLS plus a 2×2 average-pooled patch grid, 384-d.

**Classical baselines** (required by roadmap E3.2 and the operator brief): persistence, a ridge-regression linear forecaster (λ chosen on validation), and action-shuffled controls for every arm.

**Released-checkpoint rivals are exploratory only.** V-JEPA 2-AC's released checkpoint (`torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')`) was post-trained on DROID. `droid_100` is a subset of DROID, so its test episodes may be in that model's training data, and we can't rule that out without the episode list. It also predicts in its own feature space, not proprio state. It therefore enters only as B4, labelled `contaminated: possible`, and never as a primary comparison.

## 3. Fairness contract (frozen before the first B1 run)

`adjointrwm.benchmark.check_fairness` enforces items 1–5 in code; the notebook stops if any fails.

1. **Same data.** `droid_100`, frame stride 2, the pilot's episode split (the SHA-256 ordering rule, reproduced by `adjointrwm.data.episode_split` and asserted against the pilot's `data_manifest.json` on Drive when it is present), windows of 8 context and 4 future steps at stride 2, normaliser from train episodes only.
2. **Same inputs.** Every arm gets the 8-step context (state, previous action, visual features of the declared encoder) and the 4 future actions. Nothing else. A test checks that every arm's prediction is unchanged when the future targets in the batch are replaced.
3. **Same targets and one scoring function.** All arms predict future proprio state at t+1…t+4 and the frozen ResNet-18 embedding. Metrics come from `adjointrwm.eval.prediction` using predictions only; the arm name is a label and never enters a computation (a test checks this).
4. **Matched parameters.** Prediction-path trainable parameters within ±10 % of the reference arm's prediction path, by width search (`adjointrwm.models.match_width`). The reference is the pilot model without its allocator heads; the pilot's full count is 27,360,798 (`results/runs/droid100_adjoint_20260929T070629Z/config/model_manifest.json`). Frozen encoders are excluded and reported separately.
5. **Matched optimisation budget.** Same optimiser steps, same batch size, AdamW with warmup-cosine, BF16 autocast. Recipe-specific values that the official configs fix (TD-MPC2's encoder LR scale and clip norm) are kept and recorded.
6. **Equal tuning budget.** Each arm gets the same validation-only learning-rate search (default 3 values × 1 seed × 30 % of the steps). Test data is not touched until every arm is frozen. This mirrors the equal initial search of comprehensive plan §5.5.1.
7. **Paired seeds.** 5 seeds, the same seed list for every arm (the seed sets initialisation and the batch order). A failed or diverged seed is reported as failed, never replaced.
8. **One hardware stratum.** All arms in one B1 run use the same GPU type. Timings from different GPU types are never pooled.

## 4. Endpoints and analysis

- **Shared training term.** Every arm's loss includes the pilot's prediction objective on its own predictions (Gaussian NLL on normalised state + 0.25 × (1 − cosine) on the ResNet-18 target), plus its own recipe term: KL for RSSM, latent consistency for TD-MPC2, feature prediction for the two JEPA-style arms.
- **Primary endpoint (B-WM).** Test RMSE of the normalised proprio state, averaged over the 4 horizon steps. This is the pilot's dynamics-gate metric. It is reported per horizon step too, because persistence wins at step 1 in the pilot.
- **Secondary endpoints.** Native-unit RMSE per state group (cartesian position, gripper, joints), visual cosine, action-shuffle sensitivity (how much error rises when future actions are permuted across windows), ±1σ/±2σ interval coverage, latent effective rank (collapse tripwire), parameters, training throughput, peak VRAM, and p50/p95/p99 inference latency.
- **Comparison.** Per seed, the paired relative difference `(RMSE_ref − RMSE_rival) / RMSE_rival` on identical test windows, with a two-level bootstrap that resamples test episodes and seeds.
- **Classification** (margin `m = 0.02`, the pilot's `baseline_improvement_margin`):
  - `reference_better` if the 95 % CI upper bound is below 0;
  - `rival_better` if the lower bound is above 0;
  - `equivalent_within_margin` if the CI lies inside (−m, +m);
  - `inconclusive` otherwise.
- **Power caveat.** DROID-100 has 10 test episodes (`tests/data_audit.json`), so wide CIs and `inconclusive` outcomes are likely. B1 is a pilot; B3 on the 500–1,000-episode shard is the confirmatory substrate comparison.

## 5. Allocation rivals (B-AL = pilot v2)

The pilot v2 notebook fixes the contract problems in [findings §2.4](../research-notes/2026-09-29-droid100-pilot-findings.md):

| v1 problem | v2 change |
|---|---|
| No hold option | Candidate 0 is `hold` (zero effect, zero cost). Regret is measured against the best of hold and the refinements. |
| Train/eval semantics differ | Stage 1 trains single-choice refinement (hold or exactly one candidate, uniform), matching evaluation. The v1 subset regime stays available as an option. |
| "Random" was one draw | Exact expected-random regret, plus `always_hold` and `always_c_k` for every k. |
| Traces lacked gains | Traces hold the full `exact_gain[N, K+1]`, so every audit can be recomputed offline. |
| Gate label almost never positive | Gate label is the sign of `critic_regret − adjoint_regret` (ties get zero weight), with the positive-class weight estimated on validation. |
| Critic and co-state trained in one loss | Trained separately, with the same batches, steps and optimiser, and parameter-matched heads. |

**Allocator arms** (all score the same K+1 candidates):

| Arm | Deployable? | Role |
|---|---|---|
| `critic` (direct marginal-gain critic) | yes | The principal rival. Same latent, effects and costs; no co-state. |
| `adjoint` (amortised co-state, first-order score) | yes | The proposal. |
| `adjoint_randomized` (co-state permuted across the batch) | yes | Control: does alignment with ∂J/∂state matter, or just dimensionality? |
| `uncertainty` (reduction in the model's own predicted variance) | yes | Required baseline (production plan §13.2). |
| `hybrid` (gate between adjoint and critic) | yes | Selective invocation. |
| `always_hold`, `always_c_k`, `random_expected` | yes | Fixed and random baselines. |
| `exact_costate` (autograd ∂J/∂z, first-order score) | **no**, uses targets | Ceiling for any amortised co-state. Diagnostic only. |
| `oracle` | **no** | Defines regret. |

**Gates for B-AL:**

1. **Opportunity.** `adjointrwm.analysis.opportunity_audit` on validation must pass (per-window headroom over the best fixed choice ≥ 15 % of the headroom over random). If it fails, the H2 comparison on this benchmark is non-diagnostic; redesign the candidates, not the allocator (roadmap §6).
2. **Critic realisation floor.** The critic must beat `random_expected` and `uncertainty` on validation. Otherwise H2 is `UNDER_REALIZED` and inconclusive (comprehensive plan §5.5.1).
3. **Primary endpoint.** Test regret `adjoint − critic`, with an episode-cluster bootstrap 95 % CI per seed and pooled over 5 paired seeds.
4. **Controls.** `adjoint_randomized` must not reproduce any adjoint advantage.

## 6. Milestones

| ID | Deliverable | Done when |
|---|---|---|
| **B0** | This plan; tested `data`, `eval`, `models`, `training`, `allocators` modules; both notebooks | `python harness/check.py` passes; notebooks import the package; no outputs committed |
| **B1** | Run `rival_world_models_droid100.ipynb` on L4: 5 arms × 5 seeds + classical baselines | Run directory has `COMPLETE`, `fairness_contract.json` PASS, `benchmark_summary.json`, and raw per-window error tables; imported with `/import-run`; research note written with every classification, including losses and ties |
| **B2** | Run `AdjointRWM_Production_Pilot_v2.ipynb` for 5 paired seeds (= E2.1 + E2.2) | Opportunity, critic-floor and primary-endpoint results recorded for every seed; note written |
| **B3** | Repeat B1 on the E3.1 500–1,000-episode shard, split by scene/task/site | Confirmatory substrate comparison with the same frozen contract |
| **B4** (exploratory) | Frozen released-checkpoint arms (V-JEPA 2-AC, DINOv2 encoder swap for every arm) | Contamination status recorded; never a primary comparison |
| **B5** (gated) | Planning rivals in simulation (official TD-MPC2 code, CEM planners on ManiSkill3 per comprehensive plan §7.5) | Opens only after G-H2 and the Phase II unlock |

**Cost.** Only the reference arm has been timed: 21.5 training steps/s at batch 64 on L4 (`results/runs/droid100_adjoint_20260929T070629Z/benchmarks/microbatch_training_benchmark.csv`). At that rate its 5 seeds × 1,500 steps plus a 3 × 450-step LR search take about 7,500 + 1,350 steps ≈ 7 minutes of pure optimisation. The other arms have not been timed. The RSSM steps through 12 frames sequentially, so expect it to be slower. Data extraction and feature caching dominated the pilot's wall-clock time. B1 therefore caches features once per run and shares them across arms. Every (arm, seed) job is resumable, so the benchmark can span several Colab sessions.

**Drive storage.** A finished job keeps only `best.pt`; its recovery checkpoint is deleted, and LR-search trials keep no weights (their hashes stay in `DONE.json`). At about 25 M FP32 parameters, one kept checkpoint is roughly 100 MB. B1 keeps 25 of them (5 arms × 5 seeds), about 2.5 GB. B2 keeps 5 dynamics checkpoints plus small allocator heads. While a job runs, its recovery checkpoint also holds the AdamW state, about three times the model size.

**Verification so far.** Both notebooks were dry-run end to end on CPU, with Colab, Drive, TFDS and torchvision stubbed and small random fixture episodes standing in for DROID. The dry run included a run split over five sessions that resumed each time. This checks code paths only; its numbers are meaningless and are not recorded anywhere. Nothing has run on real data yet.

## 7. What each B1 outcome lets us say

| Outcome vs a rival | Allowed statement | Consequence |
|---|---|---|
| `reference_better` or `equivalent_within_margin` | "The AdjointRWM dynamics model is competitive with a re-implemented `<rival>`-style model on DROID-100 under the matched contract (n = 10 test episodes, 5 seeds)." | Keep the substrate for E3. |
| `rival_better` | "A re-implemented `<rival>`-style model predicts better." Report it plainly. | The co-state is ∂J/∂state for any differentiable model, so B-AL can run on the rival's latent. Adopting the better substrate for E3 is a design decision, not a failure of H2. |
| `inconclusive` | "No detectable difference at this sample size." | B3 decides. |

Never claim "state of the art": the rivals are re-implementations at 27 M parameters, not the published systems at their published scale.

## 8. Ablations and sensitivity (report, don't promote)

- TD-MPC2 with a single frame (the faithful Markov encoder) vs the 8-frame stack.
- DINOv2 vs ResNet-18 input for the reference arm, to separate encoder effects from architecture effects.
- DreamerV3 evaluated with the prior mode instead of the 8-sample mean.
- AdjointRWM evaluated with the base (no refinement) rollout instead of all candidates. In the pilot, base and full differed by less than 0.3 % (findings §2.1).

## 9. Risks

| Risk | Mitigation |
|---|---|
| A re-implementation is unfaithful, so a rival looks weak | "-style" naming; every deviation listed in §2; key constants copied from the official configs; an equal tuning budget; results reported as re-implementations |
| Tuning asymmetry | Same LR grid size and steps for all arms, validation only, all trials logged |
| Underpowered test split (10 episodes) | Two-level bootstrap; classification allows `inconclusive`; B3 confirms |
| Released checkpoints contaminated by DROID training | Kept out of primary comparisons (B4) |
| Colab pre-emption mid-benchmark | Per-job recovery checkpoints with the full contract; `RESUME_RUN_ID` re-enters the same immutable run directory |
| Scope creep into "our model is SOTA" claims | §7 wording; `claim-check` skill before any write-up |

## 10. Open questions for Roman

1. **Primary endpoint for B-WM:** normalised-state RMSE (pilot gate, unit-free) or native-unit RMSE of one state group (e.g. cartesian position in metres)? The plan assumes the former and reports the latter.
2. **TD-MPC2 input:** information-matched 8-frame stack (assumed) or the faithful single frame as primary?
3. **B4:** is the V-JEPA 2-AC released checkpoint worth running at all, given the contamination risk and a ViT-g encoder on L4?
