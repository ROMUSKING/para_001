# AdjointRWM Run_V2 Audit and Colab Execution Directive

## Decision

```text
RUN_V2_PIPELINE_EXECUTION: PARTIAL_PASS
REAL_TABULAR_SOURCE: PASS
VALID_SEQUENTIAL_WORLD_MODEL_DATA: FAIL
WORLD_MODEL_OBJECTIVE: FAIL
FULL_CHECKPOINT_RESUMABILITY: FAIL
PRODUCTION_QUALITY: NOT_ESTABLISHED
A100_PROMOTION: DEFERRED
CURRENT_HARDWARE: REMAIN_ON_L4
```

The run is a useful L4 code-path and gradient-flow smoke test. It is not valid evidence of a trained adjoint world model, and neither A100 40 GB nor A100 80 GB is currently justified.

## Direct archive findings

- Archive: `AdjointRWM_Production-20260928T213925Z-1-001.zip`
- Archive size: `508,250,118` bytes
- Archive SHA-256: `7c2bd647f61c8cb4b29975fcaa4de642ff3f797e43da41bf2c4e7e9d1800edd8`
- ZIP entries: `47`
- `Run_V2/logs/` is empty.
- The archive contains two 20-file dataset copies, two manifests, two production checkpoints, and one lineage file.
- The root and `Run_V2` state/action arrays are numerically identical. `Run_V2` renamed and repackaged the same first 600 Ailerons rows; it did not add new examples.

### Checkpoints

| Path | Step | Stored loss | Parameters | SHA-256 |
|---|---:|---:|---:|---|
| `AdjointRWM_Production/checkpoints/swm_production_checkpoint.pt` | 200 | 0.06993459165096283 | 23,794,978 | `6261d2d1c97ee2d3d9360ac09a172dc15a58881cc6ae956652a2b43bde13d22a` |
| `AdjointRWM_Production/Run_V2/checkpoints/swm_production_checkpoint.pt` | 100 | 0.0202 | 23,794,978 | `8488c65e06004fa5afa08aac6b65e1ffb16468484251ce8aad0b9813b494ba79` |

Both contain finite FP32 model tensors. The `Run_V2` checkpoint is the relevant versioned run.

## Scientific and engineering defects

### 1. Ailerons was converted into invented episodes

The code takes the first 600 tabular rows, groups every 30 rows, fabricates 10 Hz timestamps, labels the first seven input columns as actions and the next ten as states, and ignores the dataset target. The result is not a demonstrated physical trajectory or DROID episode.

### 2. The training target is the same time step, not the future

The training loop passes the same `states` and `actions` tensors as both input and target. No `t -> t+1` shift is applied. The future head is therefore trained to match a learned transform of the current state rather than to predict a future state.

### 3. The target transforms are trainable moving targets

`target_state_projection` and `target_action_projection` are optimized jointly with the model. There is no frozen or stop-gradient target encoder. The model and target can co-adapt toward a low loss without learning a useful transition law.

### 4. The co-state, critic, gate and teacher objectives are not semantically valid

- `co_state` is trained against a learned projection of current input columns labelled as actions, not an adjoint or directional gain.
- `critic_head` is trained toward zero, not a marginal-gain or return target.
- `gate_head` is trained toward constant `0.5`, not a selective-invocation decision.
- `teacher_projection` is trained toward zero. The frozen research teacher is not loaded or queried.

A direct probe of the `Run_V2` checkpoint found gate output mean approximately `0.498` with standard deviation approximately `0.003`, confirming that the gate learned the imposed constant rather than a useful switching policy.

### 5. The reported hardware-promotion metrics are constants

The A100 report hard-codes:

```python
eval_mse = 0.5936
projected_regret = 0.0475
task_success_rate = 94.1
```

They are not calculated from `Run_V2`. The reported regret and task success therefore cannot support any hardware or model-quality decision.

### 6. The checkpoint is not a complete resumable training state

The model has 40 parameter tensors, while the optimizer stores state for 48 tensors. The extra eight are the two target projection modules. Their optimizer moments are saved, but their weights are not included in `model_state_dict` and are not otherwise checkpointed.

The checkpoint also lacks the scheduler, AMP/scaler state, sampler/dataloader position, split manifest, data/config/source hashes, environment lock and best held-out metrics. The interruption test only verifies model-weight reload and RNG assignment; it does not run the next batch in two branches and compare the resulting loss and update.

### 7. The archive lacks the claimed production evidence package

No `metrics.jsonl`, TensorBoard events, validation predictions, allocation traces, profiler trace, memory summary, run configuration, environment manifest, source manifest, split manifest, run summary or acceptance report is present.

## L4 versus A100 decision

### Remain on L4 now

The current forward-only dummy-input sweep peaks at about `1.76 GB` VRAM at batch 64, leaving most of the L4 unused. The 23.8M-parameter state/action model runs quickly. The limiting factors are invalid data semantics, invalid targets and absent held-out evaluation—not compute or memory.

### A100 40 GB gate

Move to A100 40 GB only after a valid real sequential/visual pilot and when at least one condition is measured:

1. the minimum useful BF16 workload cannot fit on L4 after sensible microbatching and activation checkpointing;
2. actual end-to-end training time for the promoted run is operationally impractical and an A100 benchmark shows a material speed-up;
3. the actual training profiler, including backward and optimizer steps, shows sustained compute or memory-bandwidth saturation rather than data-loading delay;
4. longer context, more cameras, higher resolution or a larger model produces a demonstrated held-out gain;
5. full imagined-adjoint or teacher-label generation dominates runtime.

### A100 80 GB gate

Choose 80 GB only when the promoted configuration cannot fit within 40 GB with a useful microbatch, or when multi-camera high-resolution sequences, long rollout horizons, simultaneous teacher/student execution or label generation require more than roughly the 40 GB envelope. Do not select 80 GB merely for speed when 40 GB fits.

## Required next run

### Track choice

Choose one target and name it honestly:

- **Embodied world-model track — recommended:** use publisher-sourced DROID, BridgeData V2, RoboNet or another real episodic dataset with actual observations, actions, states and boundaries.
- **Ailerons specialist track:** treat Ailerons as static supervised regression. Use the published target as the aileron-control target. Do not call arbitrary rows episodes, actions, robot states, DROID data or a world model.

Do not mix the two tracks.

### Embodied world-model execution order

1. Acquire and verify real episodes. Preserve original episode IDs, timestamps, image payloads, actions, states, instructions, provenance and licence evidence.
2. Split by whole episode, scene/task/site where available, before window creation.
3. Define windows as `context[t-L+1:t]`, actions `a[t:t+H-1]`, and targets `state/latent[t+1:t+H]`.
4. Use a frozen or EMA/stop-gradient target encoder. Never jointly optimize an unconstrained target projection with the predictor.
5. Load `optimized_adjoint_teacher.pt` read-only and verify its actual 32-dimensional input/output contract before distillation.
6. Generate co-state/directional labels from a declared objective; do not use action vectors as co-state labels.
7. Train the direct critic on counterfactual marginal gain or return labels.
8. Train the selective gate on measured net benefit of adjoint invocation after incremental latency/compute cost.
9. Evaluate native-unit one-step and multi-step prediction against persistence and linear dynamics.
10. Evaluate adjoint allocation against the direct critic using candidate rank, regret and AURC, plus randomized/wrong-goal/wrong-horizon controls.
11. Profile the actual BF16 training step, including data decode, forward, backward, optimizer, checkpoint and evaluation.
12. Reconsider hardware only after the valid L4 pilot report is complete.

## Copy-paste directive for the Colab agent

```text
STOP. Do not switch to A100 and do not continue training from the Run_V2
checkpoint as a production world model.

CLASSIFICATION
RUN_V2_PIPELINE_EXECUTION = PARTIAL_PASS
VALID_SEQUENTIAL_WORLD_MODEL_DATA = FAIL
WORLD_MODEL_OBJECTIVE = FAIL
FULL_CHECKPOINT_RESUMABILITY = FAIL
PRODUCTION_QUALITY = NOT_ESTABLISHED
A100_PROMOTION = DEFERRED
CURRENT_HARDWARE = L4

WHY
1. The run uses OpenML Ailerons rows, not DROID trajectories.
2. It invents 30-row episodes and 10 Hz timestamps.
3. It labels arbitrary input columns as actions and states and ignores the true
   dataset target.
4. It predicts a learned projection of the current state, not t+1 or a rollout.
5. The co-state target is a learned projection of current input columns, not an
   adjoint label.
6. The critic is trained to zero, the gate to 0.5 and the teacher projection to
   zero; the frozen teacher is not used.
7. The A100 report uses hard-coded MSE, regret and success values.
8. Target-projection optimizer states are stored but target-projection weights
   are absent from the checkpoint, so full training resume is not reproducible.
9. Run_V2/logs is empty and the required evaluation artifacts are absent.

IMMEDIATE ACTIONS
A. Preserve Run_V2 read-only as `tabular_pipeline_smoke_only` and record the
   archive/checkpoint hashes.
B. Do not overwrite either checkpoint.
C. Choose exactly one track:
   - EMBODIED WORLD MODEL: acquire real publisher-sourced episodic robot data;
     or
   - AILERONS REGRESSOR: use Ailerons as static supervised regression with its
     real target and remove all DROID/world-model claims.
D. For the recommended embodied track, remain on L4 and execute a 20-100 real
   episode smoke run first.
E. Build targets using t+1 and multi-step future states/latents.
F. Use a frozen or stop-gradient target encoder.
G. Load the frozen 32-D adjoint teacher and generate valid teacher targets.
H. Train critic, co-state and gate against their real decision targets.
I. Save all trainable modules, optimizer/scheduler/scaler, sampler position,
   RNG states, split/data/config/source hashes and best metrics.
J. Verify resume by branching from one checkpoint, processing the identical next
   batch twice, and comparing loss, gradients, parameters and optimizer state.
K. Produce held-out native-unit prediction, rollout, critic-vs-adjoint ranking,
   regret/AURC, calibration and actual training profiler results.

HARDWARE RULE
Remain on L4. Recommend A100 40 GB only if the valid real workload cannot fit,
actual end-to-end runtime is impractical, profiling demonstrates a material
compute/bandwidth bottleneck, or a larger context/resolution/model yields a
measured held-out gain. Recommend A100 80 GB only if 40 GB cannot fit the
minimum promoted configuration or full teacher/student/high-resolution rollout
training genuinely requires it. Validation failure alone is not a GPU-upgrade
criterion.
```
