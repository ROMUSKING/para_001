# Colab handoff: what cannot run in the sandbox, in what order, on which hardware

**Written:** 2026-09-30 · **Applies to:** everything in the roadmap that needs a GPU, DROID data, a Drive mount or a model download. Everything else in Track D is already run on CPU in the sandbox (D4-0 to D4-3, D1-0, D1-0b, DL).

**Hardware policy (from the [operator brief](../production/colab_l4_operator_brief.md) and the [roadmap](roadmap.md) §hardware):** stay on the **NVIDIA L4 (24 GB)** for everything through N3 and E3. The measured peak so far is 0.6 GiB (pilot, `results/runs/droid100_adjoint_20260929T070629Z/`), so memory is not the constraint. An A100 40 GB is justified only if the minimum useful workload does not fit, a profiler shows real compute or bandwidth saturation, or the run is impractical; unused VRAM and dataset size are never reasons. **The pilot v2 notebook refuses any GPU that is not an L4, A100 or H100** (`GPU_NAME` check in its section 1), so a T4 will not start it.

## 1. Switch to Colab now for these

| Order | Item | Notebook | Hardware | What it needs | What to send back | What it decides |
|---|---|---|---|---|---|---|
| 1 | **B2-probe**: pilot v2 with **one seed** | `notebooks/01-production/AdjointRWM_Production_Pilot_v2.ipynb`, set `seeds` to `(0,)` in the config cell | **L4** | Colab GPU runtime, Drive mounted, DROID-100 through TFDS (the pilot v1 used the same source), a few GB of Drive for the feature cache | the run directory (`/import-run`) | Gate 1 of the pilot, the **opportunity gate on validation** (the roadmap's E1.1 stop-loss: below 15 % means redesign the candidates before any more training). Read `opportunity_gate_validation` and the dynamics gate for seed 0 first. |
| 2 | **B2**: pilot v2 with the five planned seeds | the same notebook, `seeds = (0, 1, 2, 3, 4)` | **L4** | as above; the run is resumable across sessions (`RESUME_RUN_ID`, `MAX_STEPS_PER_SESSION`) | the run directory | The real-data **H2 test** (adjoint versus a matched direct critic, five paired seeds) with the gates in plan §5. Launch only if the probe's opportunity gate passes. |
| 3 | **B1**: rival world-model benchmark | `notebooks/03-benchmarks/rival_world_models_droid100.ipynb` | **L4** | as above; about 2.5 GB of Drive for the 25 kept checkpoints (5 arms × 5 seeds); resumable | the run directory | Which substrate predicts best. It says nothing about H2 and is not on the critical path: run it after B2, or when L4 time is free. |

**Why a one-seed probe instead of E1.1.** The roadmap's first step, `opportunity_audit.ipynb` (E1.1), needs the kernel state of pilot v1 (it says: run the pilot through section 5, then paste its cells in the same runtime), which means re-running the pilot's data extraction and training first. Pilot v2 computes the same opportunity quantity on validation for every seed as its gate 1 (`opportunity_ok` in section 6). A one-seed v2 run therefore answers E1.1's question with one run instead of two. E1.1 stays available if you want the three other diagnostics in it (exact-co-state score against random, train-versus-test gain distributions, candidate-effect geometry). This replaces the roadmap's E1.1-first order with an equivalent, cheaper probe; say if you prefer the original order. The roadmap's stop-loss ("no B2 spend until it passes") is respected in spirit: the probe is one seed, and the full five-seed run waits for its gate.

**Two things to check when you start.** (1) Whether the per-run feature cache is reused between the probe and the full run: the notebook restores cached episodes with hash verification, but I have not verified that a second run ID finds the first run's cache, so budget for the extraction to repeat. (2) The time: only the reference arm of B1 has been timed (21.5 training steps per second at batch 64 on the L4; about 7 minutes of pure optimisation for five seeds of 1,500 steps plus the learning-rate search), and data extraction dominated the pilot's wall clock. Nothing in pilot v2 or B1 has run on real data yet, so treat any duration estimate as unmeasured.

## 2. Will need Colab later (not built yet)

| Item | Why a GPU | Recommended hardware | Note |
|---|---|---|---|
| **D2-0**, LLM context tree (Qwen3-8B, MuSiQue, Qasper) | The frozen LLM scores every context. | **L4.** Qwen3-8B in BF16 has 16.38 GB of weights (`docs/licences/survey-2026-09-29-pass2.md` §2.1) and fits the L4's 24 GB for the opportunity gate (no gradients) at contexts of a few thousand tokens and batch size 1. **Qwen3-4B (8.04 GB) for any run that takes gradients** through the frozen model (the co-state with respect to span embeddings) or needs longer contexts. | Move to an A100 40 GB only after a measured out-of-memory with gradient checkpointing and shorter contexts, per the brief's rule. A CPU run of the 4B model is possible for a handful of examples but far too slow for an exhaustive oracle. |
| **D3-0**, code-repository context | Only if a neural retriever or a code LLM scores candidates. | A CPU first pass (import-graph and BM25 baselines) is feasible in the sandbox; a GPU is needed only for a learned scorer. | Needs a licence-cleared repository set and Roman's choice of the localisation benchmark (the SWE-bench dataset itself is `unverified` in the register). |
| **Rung-1 critics on real data** (if B2 is positive) | Training. | L4 first; the pilot's 25 M-parameter model is far below the L4's limits. | A100 only with a profiler trace or a measured held-out gain that justifies it. |

## 3. Stays on the CPU in the sandbox

D4 (time stepping, with and without a goal per instance), D1 on SMD, licence survey passes, table scripts, the harness. A learned-critic comparison on D4 (D4-1) is a NumPy job of minutes to tens of minutes if it is ever opened.

## 4. How to bring a Colab run back

1. Finish the run so `COMPLETE` exists; copy the run directory from Drive to the sandbox (or attach it), excluding weights and the executed notebook if they are large (the repository rules: nothing over about 5 MB, no `.pt`, `.npz` or checkpoints; record Drive IDs and SHA-256 in `docs/DRIVE_INVENTORY.csv`).
2. Use the `import-run` skill (`/import-run`) to copy it into `results/runs/<run_id>/` with a README, hashes and a trace summary, then `audit-run` if the run is meant to count as evidence, then `research-note`.
3. A failed gate is a result: do not rerun with different settings to get a pass, and report failed seeds instead of replacing them.
