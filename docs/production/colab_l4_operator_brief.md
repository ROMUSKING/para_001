# Colab L4 Production-Training Operator Brief

## Mission

Train a useful production model derived from the validated adjoint-guided world-model system. The first production target is a **specialist embodied world model** trained on real robot trajectories. It must learn temporal latent dynamics, action-conditioned prediction, co-state estimation, candidate ranking, resource allocation, and uncertainty. It must not be reduced to unrelated tabular regression.

The validated `optimized_adjoint_teacher.pt` checkpoint is a small `D=32` co-state teacher and regression oracle. Preserve it. Do not overwrite it and do not mislabel it as the new production backbone.

## Non-negotiable operating rules

1. **Never fabricate missing data.** If a required file or dataset is absent, stop with a precise error and list the paths checked. Do not create random features, duplicate one telemetry record, or silently substitute synthetic data.
2. **Never claim a run succeeded unless the referenced files exist.** Print and hash every saved checkpoint, metrics file, artifact and dataset manifest.
3. **Do not optimize for VRAM occupancy.** Select batch size from measured throughput, gradient noise, validation quality and stability. Unused VRAM is acceptable.
4. **Do not treat training loss alone as usefulness.** Report held-out temporal prediction, multi-step rollout error, action-conditioned error, candidate-ranking quality, calibration, allocation regret, planning performance and systems cost.
5. **Do not train a generic MLP on unrelated static rows.** Ailerons may be retained only as a loader or tabular smoke test; it is not production training for this world model.
6. **Do not use global float64 for production training.** Use BF16 autocast on the L4, FP32 for sensitive accumulations and normalization, and FP64 only for sampled adjoint/oracle audits.
7. **No data leakage.** Split by trajectory, scene, task and collection location before generating windows. Never random-split adjacent frames from the same trajectory across train and validation.
8. **All experiment state must be reproducible.** Save configuration, environment, data split, RNG state, model, optimizer, scheduler, scaler, sampler position and source snapshot.
9. **Train from local Colab storage, persist to Drive.** Stage active shards under `/content`; write compact, atomic checkpoints to Drive. Do not perform high-frequency small-file training I/O through mounted Drive.
10. **Keep the direct critic.** The critic is a production fallback, comparison arm and possible control variate. It is not deleted because the adjoint method performed better in research.
11. **Discover sessions before provisioning.** Always run `colab sessions` to inspect active assignments before launching new compute. Attach to existing sessions rather than provisioning duplicate concurrent GPU VMs.
12. **Validate remote hardware before execution.** Run an explicit hardware probe (`torch.cuda.get_device_name(0)`, VRAM capacity, CUDA status). Verify that the runtime accelerator matches the job requirements (L4 standard, 22–24 GiB VRAM; G4/A100 require explicit profiler justification) and verify Drive mount access (`/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production`).
13. **Start jobs first when compute is active.** Never burn billable compute credits while an accelerator sits idle reading documents, reviewing PDFs, or planning offline. Launch queued remote jobs immediately, and conduct reading or analysis in parallel while remote execution progresses.
14. **Immediate session teardown.** Stop the session (`colab stop -s <session>`) immediately upon job completion or failure. Never leave billable GPU runtimes running unattended.

## Recommended first production target

### Model type

A small-to-medium action-conditioned latent world model with:

- visual observation adapter;
- proprioception/state adapter;
- action adapter;
- shared temporal latent backbone;
- recursive coarse-to-fine latent hierarchy;
- future-latent predictor;
- co-state estimator;
- direct marginal-gain critic;
- candidate-effect head;
- uncertainty/calibration head;
- rate/compute cost head;
- selective-invocation gate;
- analytical rescue interface.

### Initial L4 envelope

Start with a model that is large enough to learn useful temporal structure but small enough to iterate rapidly:

- trainable parameters: approximately 20–60 million;
- latent width: 384 or 512;
- temporal blocks: 6–8;
- attention heads: 6–8;
- context: 8–16 time steps;
- initial image resolution: 128×128 or 160×160;
- cameras: one exterior plus wrist view initially;
- state/action tokens: always included;
- hierarchy depths: 3 initially;
- precision: BF16 autocast;
- optimizer states: FP32;
- gradient clipping: enabled and logged;
- activation checkpointing: only after profiling establishes a memory need.

Use a permissively licensed pretrained visual encoder when its model and training-data terms are documented. Freeze it during the first dynamics run, then unfreeze only the last blocks after the temporal and allocation heads become stable.

### Teacher integration

The existing teacher accepts 32 state features and a budget value. Integrate it through a learned projection:

```text
production latent -> 32-dimensional teacher projection -> frozen teacher
                 \-> production co-state head
```

Use the frozen teacher for:

- initialization checks;
- distillation on compatible state features;
- sampled directional-consistency audits;
- regression tests;
- fallback calibration.

Do not force all production latents into the old `D=32` state. The projection is an interface, not the production representation bottleneck.

## Real-data programme

### Primary dataset

Use **DROID** as the first real embodied dataset because it includes temporally aligned camera observations, robot actions, language instructions and robot state across diverse scenes and tasks.

Do not attempt to copy the full corpus into Google Drive. Use deterministic shards:

1. **Loader smoke test:** 20–100 episodes.
2. **Pipeline pilot:** 500–1,000 episodes.
3. **L4 specialist training:** 2,000–5,000 episodes, subject to local-disk and session constraints.
4. **A100/H100 scale-up:** larger stratified corpus after the L4 model and data pipeline pass all gates.

Create a dataset manifest containing:

- publisher and dataset version;
- source identifiers;
- license text and attribution requirements;
- download date;
- episode IDs and hashes;
- task, scene and site distributions;
- excluded/corrupt episodes;
- exact train/validation/test split;
- preprocessing and augmentation hash.

### Secondary datasets

After the DROID pipeline is stable, consider:

- BridgeData V2 for embodiment and environment diversity;
- a permitted ManiSkill trajectory subset for controlled simulation transfer;
- domain-specific real telemetry only when it contains temporal transitions, actions or interventions relevant to the target task.

Every secondary dataset requires its own license and provenance record. Do not assume that a code repository license governs the data or assets.

### What not to do

- Do not concatenate unrelated tabular datasets merely to increase row count.
- Do not mix domains until each domain has a native held-out metric and sampling weight.
- Do not use adjacent-window random splitting.
- Do not report MSE without inverse-transforming to native units and comparing against persistence, linear dynamics and direct-critic baselines.

## Google Drive and local runtime layout

Use this persistent root:

```text
/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production/
```

Each run receives an immutable run ID:

```text
AdjointRWM_Production/
  datasets/
    manifests/
    shard_indexes/
  runs/
    <run_id>/
      config/
        run_config.yaml
        data_manifest.json
        environment.json
        source_manifest.json
      checkpoints/
        latest.pt
        best_validation.pt
        best_planning.pt
        step_<N>.pt
      logs/
        metrics.jsonl
        events.out.tfevents...
        console.log
        warnings.jsonl
      artifacts/
        validation_predictions.parquet
        allocation_traces.parquet
        calibration.json
        profiler_trace.json
        memory_summary.txt
        plots/
      samples/
      reports/
        run_summary.md
        acceptance_report.json
      COMPLETE
```

Use local runtime paths for active work:

```text
/content/adjoint_rwm_cache/<run_id>/
/content/adjoint_rwm_work/<run_id>/
```

Copy or stream a bounded shard to local disk at run start. Save checkpoints first to a local temporary file, calculate its SHA-256 hash, then copy to a temporary Drive filename and atomically rename it. Keep no more than the best checkpoints plus a rotating recovery window.

## Checkpoint contract

Every checkpoint must contain:

```python
{
    "schema_version": 1,
    "run_id": run_id,
    "global_step": global_step,
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "critic_state_dict": critic.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict() if scaler else None,
    "sampler_state": sampler_state,
    "rng_state": {
        "python": ...,
        "numpy": ...,
        "torch_cpu": ...,
        "torch_cuda": ...,
    },
    "data_manifest_hash": ...,
    "config_hash": ...,
    "source_hash": ...,
    "best_metrics": ...,
}
```

A forced-interruption test must confirm that a resumed run reproduces the next validation result within the frozen tolerance.

## Training curriculum

### Stage 0 — Audit and data verification

Before training:

- identify the exact L4 model and available VRAM;
- mount Drive;
- verify the production root;
- hash the teacher checkpoint;
- inspect the checkpoint architecture and tensor dtypes;
- validate the real dataset source, license and episode schema;
- create trajectory-level splits;
- decode and visualize at least eight episodes;
- verify state/action/image timestamp alignment;
- run one batch through every model head;
- write `PRETRAINING_AUDIT.json`.

No missing-file fallback is permitted.

### Stage 1 — Data and model smoke test

Run 50–200 optimizer steps on 20–100 real episodes.

Pass criteria:

- finite losses and gradients;
- no target leakage;
- dataloader utilization is acceptable;
- local cache works;
- checkpoint/resume works;
- saved outputs can be reloaded in a clean runtime;
- critic, adjoint and random/no-adjoint controls execute.

This stage produces no model-quality claim.

### Stage 2 — Representation and dynamics pilot

Train on 500–1,000 episodes.

Primary objectives:

- masked future-latent prediction;
- action-conditioned one-step prediction;
- multi-step latent rollout;
- cross-scale consistency;
- anti-collapse regularization;
- uncertainty calibration.

Track latent variance and effective rank. Abort and preserve evidence when collapse thresholds persist.

### Stage 3 — Co-state and allocator training

Generate or compute teacher targets only where the objective and state contract are valid. Train:

- co-state vector or candidate projection;
- directional product;
- sign and ranking targets;
- direct marginal-gain critic;
- candidate-effect model;
- uncertainty and cost heads.

Evaluate against:

- direct critic;
- directional-only model;
- randomized co-state;
- wrong-goal co-state;
- wrong-horizon co-state;
- fixed allocation;
- full-compute allocation.

### Stage 4 — Specialist L4 training

Train the 20–60M parameter model on 2,000–5,000 stratified episodes.

Use:

- BF16 autocast;
- FP32 loss reductions and normalizers;
- AdamW;
- warmup followed by cosine or a validated schedule;
- gradient clipping;
- exponential moving average only if the representation recipe requires it;
- early stopping on a composite gate, not raw training loss.

Promotion requires improvement in temporal prediction and allocation/planning utility without violating rate, latency, calibration or safety limits.

### Stage 5 — Scale decision

Run controlled scaling at widths and depths such as:

```text
small -> medium -> large-L4-limit
```

Tune on the small model and transfer the recipe to larger variants where the parameterization supports it. Compare validation-quality gain per GPU-hour. Do not scale solely because memory remains unused.

## Batch-size policy

Do not start at 512 or 1,024 by assumption.

1. Benchmark microbatches such as `8, 16, 32, 64` using the real sequence and image shapes.
2. Warm up the GPU before timing.
3. Record examples/second, tokens/second, step latency, peak allocated and reserved memory, dataloader wait, GPU utilization and validation behaviour.
4. Select the smallest batch within approximately 95% of peak throughput unless gradient-noise measurements justify a larger global batch.
5. Use gradient accumulation when the desired global batch is larger than the best-throughput microbatch.
6. Re-tune or scale the learning rate when global batch changes.
7. Re-run the sweep when resolution, context, camera count or model width changes.

VRAM usage is not a performance metric. A useful model can be compute-bound with substantial free memory.

## L4 precision and performance policy

- Default production training: `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`.
- Keep optimizer state and selected reductions in FP32.
- Use FP64 only for analytic co-state checks and small audit batches.
- Test `model.compile()` only after eager-mode correctness passes.
- Record compile time separately from steady-state training time.
- Use `torch.profiler` with warm-up and active windows.
- Apply activation checkpointing only to the blocks responsible for a measured memory bottleneck.
- Bucket variable-length sequences to control padding and recompilation.

## When to move from L4 to A100 40 GB, A100 80 GB or H100

### Remain on L4 when

- the selected specialist model fits with at least a reasonable recovery margin;
- the minimum useful sequence, resolution, camera count and microbatch fit;
- dataloader throughput is not the dominant bottleneck;
- a full pilot can finish within the available Colab sessions;
- BF16 quality matches the audit precision within tolerance.

### Switch to A100 40 GB when

- the validated architecture cannot fit the minimum useful workload on L4 after BF16, sensible checkpointing and shape bucketing;
- peak memory repeatedly exceeds the frozen L4 threshold;
- backward or attention kernels are memory-bandwidth bound and profiling predicts material wall-clock savings;
- longer sequences, more cameras or a larger teacher materially improve held-out utility;
- full imagined adjoint sweeps are required frequently.

### Switch to A100 80 GB when

- 40 GB cannot support the promoted model and minimum microbatch;
- high-resolution multi-camera sequences or long rollouts are required;
- teacher-label generation or multi-head training needs substantially larger activation memory;
- checkpointing overhead on 40 GB erases the expected savings.

### Switch to H100 when

- the medium model is already scientifically and operationally validated;
- the workload is dominated by large transformer or attention kernels;
- FP8 or H100-specific mixed-precision experiments have BF16 parity checks;
- the expected time saving exceeds migration and retuning cost.

Dataset row count alone is never a hardware-switch trigger. Datasets reside primarily in CPU/local storage; GPU memory is driven by model state, optimizer state, activations, sequence shape, camera count, precision and microbatch.

### A CUDA OOM is a code diagnosis before it is a hardware question (added 2026-10-04)

An out-of-memory error on a Colab GPU is, in this project's experience, almost always a defect in
how the work is batched rather than a limit of the card. Session 6A requested a single
**42.19 GiB** allocation on the 22.03 GiB L4, which looks like exactly the evidence rule 7
demands — until it is traced. The frame was `objective_at_masks` inside an exhaustive subset
search that built all 14,400 masks for every window in one call, outside `torch.no_grad()`. A
single forward pass of that model peaks at **0.09 GiB**, measured with
`torch.cuda.reset_peak_memory_stats()`.

After chunking over the *(sample, candidate)* product and wrapping the measurement in `no_grad`,
the same science runs in **1.7 GiB**. Provisioning an A100 80 GB at that point would have burned
credits to conceal a three-line fix and produced a run whose memory footprint said nothing about
the hardware.

Before recommending any accelerator, work this order:

1. Read the **requested** allocation size. Tens of GiB in one request means one enormous tensor,
   not a model that needs the memory.
2. Read the **traceback frame** that built it, not just the final allocator line.
3. Check for a whole split materialised on the GPU (`[move_to_device(b) for b in DataLoader(...)]`).
   Keep batches on the host, upload per use, set `num_workers>0` for `.npz`/`.parquet` shards.
4. Check whether a combinatorial enumeration is batched in one call, and chunk over the product
   of axes rather than one axis.
5. Check for missing `torch.no_grad()` around measurement code; retained graph nodes accumulate
   across chunks.
6. Measure the true single-pass peak. Small peak plus a huge OOM request means the excess is the
   algorithm's batching.

Only after all six is this a hardware question, and then it needs a committed profiler trace
showing saturation, per the conjunctive G4 gates in
`docs/plans/2026-10-03-10-session-colab-hopper-plan.md`.

### Remote-runtime caveats that cost real time (added 2026-10-04)

- **`colab drivemount` cannot be automated.** Authorisation is interactive and human-only, so a
  headless agent on a fresh runtime has no Drive. Either stage everything to `/content` and pass an
  explicit output root, or ask the user to mount Drive first. Do not discover this after the
  pipeline is written.
- **A browser-spawned runtime is unreachable.** A `[?] <assignment_id>` entry from the Colab web
  UI cannot be adopted by `colab new` (it provisions a *second* VM) nor addressed by
  `-s <assignment_id>`. Ask the user to release it, or provision your own and say which to stop.
- **Long jobs must be detached.** `colab exec` runs in the shared kernel, so a foreground job
  blocks every later command; use `setsid nohup … &` and poll a log file. See the `colab-cli`
  skill for the exact incantation.

## Required metrics

### Model quality

- one-step latent prediction;
- multi-step rollout error by horizon;
- state and action prediction in native units;
- held-out scene/task/site performance;
- latent variance and effective rank;
- uncertainty calibration;
- downstream planning success or return.

### Allocation quality

- candidate sign accuracy;
- candidate rank correlation;
- oracle regret or bounded proxy;
- AURC across budgets;
- quality per actual transmitted bit;
- quality per measured millisecond;
- critic-versus-adjoint paired difference.

### Systems

- step latency after warm-up;
- examples and frames per second;
- dataloader wait fraction;
- forward, backward and optimizer time;
- peak allocated and reserved VRAM;
- host RAM and local disk usage;
- compile and recompilation count;
- checkpoint duration;
- resume integrity;
- p50, p95 and p99 inference latency.

## Stop conditions

Stop the run rather than improvising when:

- the real dataset cannot be located or licensed;
- state/action/image alignment fails;
- train/validation leakage is detected;
- NaN/Inf persists after one documented recovery attempt;
- latent collapse persists beyond the frozen threshold;
- checkpoints cannot be reloaded;
- Drive writes fail integrity checks;
- the critic is under-trained relative to the adjoint arm;
- the selected metric is unrelated to the production objective;
- a supposed improvement disappears after native-unit or task-level evaluation.

## Immediate next execution order

1. Preserve the current notebook and teacher weights as read-only research artifacts.
2. Create a new production notebook rather than appending more cells to `Untitled2.ipynb`.
3. Create the persistent Drive hierarchy and immutable run manifest.
4. Download or stream a bounded DROID shard using the publisher's schema.
5. Validate the data and trajectory-level split.
6. Implement the 20–60M parameter specialist architecture.
7. Run the real-data smoke test and checkpoint-resume test.
8. Profile microbatches on L4.
9. Train the 500–1,000 episode pilot.
10. Review the promotion report before the 2,000–5,000 episode specialist run.
11. Trigger an A100 recommendation only from the measured memory/time/quality criteria above.

## Directive to paste into the Colab chatbot

```text
You are operating the production-training notebook for the Adjoint-Guided Recursive Rate-Distortion World Model.

Your goal is to produce a useful real-data specialist model on the NVIDIA L4, not merely to complete a training loop or maximize VRAM use.

NON-NEGOTIABLE RULES
1. Never fabricate, duplicate, randomly generate or silently substitute training data when a file or dataset is missing. Stop and report the exact missing dependency and paths checked.
2. Never claim that a file, checkpoint, log, dataset or metric exists without verifying it on disk and printing its path, byte size and SHA-256 hash.
3. Do not use Ailerons or another unrelated static tabular regression dataset as production training for this embodied world model. Such data may be used only as an explicitly labelled loader smoke test.
4. Use real temporally aligned observations, actions and robot states. Start with a deterministic, licensed DROID subset. Record source, version, license, attribution, episode IDs, hashes and splits.
5. Split by full trajectory and, where metadata allows, by scene/task/site. Never split adjacent windows from one episode across train and validation.
6. Preserve `/content/drive/MyDrive/Colab Notebooks/Phase_N3_Logs/optimized_adjoint_teacher.pt` as a read-only teacher/regression artifact. Do not overwrite it.
7. Build a production specialist with visual, state, action, temporal-dynamics, co-state, direct-critic, candidate-effect, uncertainty and resource-cost modules. Do not substitute a generic regression MLP.
8. Use BF16 autocast on L4, FP32 optimizer states and sensitive reductions, and FP64 only for small adjoint/oracle audits. Remove global float64 from the production training path.
9. Select batch size by a measured microbatch sweep on the real workload. Do not choose 512 or 1024 just to occupy memory. Use gradient accumulation for the desired global batch.
10. Keep the direct critic fully trained and information-equivalent. It remains a comparison arm, fallback and control variate.
11. Train from `/content` local storage. Save compact, atomic checkpoints, logs and artifacts under `/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production/runs/<run_id>/`.
12. Every checkpoint must include model, critic, optimizer, scheduler, scaler, sampler position, RNG states, epoch, global step, config hash, data-manifest hash, source hash and best metrics.
13. Save `metrics.jsonl`, TensorBoard events, profiler traces, allocation traces, validation predictions, environment manifest, data/license manifest, plots and a Markdown run summary.
14. Do not recommend A100 based on dataset row count or unused L4 memory. Recommend A100 only when the validated minimum workload does not fit, measured throughput makes the run impractical, longer context/resolution/camera count materially improves held-out quality, or frequent full adjoint sweeps are a demonstrated bottleneck.
15. Stop on missing real data, data leakage, persistent NaN/Inf, latent collapse, checkpoint corruption, Drive integrity failure or objective mismatch. Do not bypass a failed gate.

EXECUTE IN THIS ORDER
A. Audit the current runtime, L4 GPU, Drive mount, teacher checkpoint, package versions and available local disk.
B. Create the immutable run directory and manifests.
C. Acquire and validate a small real DROID shard; display sample episodes and verify timestamp alignment.
D. Build the production specialist architecture and count trainable parameters.
E. Run 50–200 real-data smoke-test steps and verify checkpoint/resume from a clean model instance.
F. Profile microbatches 8/16/32/64 with BF16 and select the throughput/quality operating point.
G. Train a 500–1,000 episode pilot with proper held-out trajectory splits.
H. Produce a promotion report covering prediction, rollout, co-state ranking, critic comparison, allocation quality, calibration, VRAM, throughput and resume integrity.
I. Proceed to 2,000–5,000 episodes only if the promotion gates pass.

Before training, print a concise execution plan and the exact dataset, model, split, batch, precision, checkpoint and metric contracts. Then execute. Do not replace a failed real-data step with synthetic data.
```
