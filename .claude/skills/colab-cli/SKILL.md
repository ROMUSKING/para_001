---
name: colab-cli
description: Interact with Google Colab runtimes via the official colab CLI tool to provision GPU/TPU/CPU VMs, execute scripts and notebooks remotely, transfer files, inspect logs, and manage compute lifecycle. Use when offloading compute, running remote training or benchmarks on Colab, or inspecting active Colab sessions.
---

# Colab CLI

**Goal:** manage and execute workloads on Google Colab remote runtimes from the command line using the official `google-colab-cli` tool (`colab`).

## Mental Model

- **Session = rented VM + Jupyter kernel:** `colab new` provisions a billable remote VM; `colab stop` releases it. Sessions stay alive as long as the kernel is active.
- **Kernel state persists across commands in a session:** `colab exec` reattaches to the existing kernel without restarting it. Imports, variables, and defined functions survive between calls. Resetting state requires `colab restart-kernel` or `colab stop`.
- **Default working directory:** `/content` on the remote runtime. Always prefer absolute `/content/...` paths.
- **Primary execution path:** Use `colab-cli` to send jobs, execute notebooks (`colab exec -s <session> -f <notebook>`), and monitor progress directly. Worker scripts (`scripts/colab_worker.py`, `notebooks/05-ops/colab_worker.ipynb`) and the Drive job queue (`jobs/inbox/`) are preserved as **fallback** mechanisms when direct CLI access is not configured.
- **Fire-and-forget:** each CLI invocation connects, executes the requested action, and exits.

## Authentication & Setup

The global flag is `--auth=<adc|oauth2>` and must precede subcommands (e.g. `colab --auth=adc new -s job`).

### ADC (Application Default Credentials) - Recommended for Agents

Run headless authentication with the required Google OAuth scopes:
```bash
gcloud auth application-default login \
  --scopes=openid,\
https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
```
*Note:* all four scopes are necessary: `userinfo.email` for Colab backend identity, `colaboratory` for Colab API features, and `openid` + `cloud-platform` as required by gcloud.

### OAuth2 (Browser Flow)

```bash
colab --auth=oauth2 sessions
```
Prompts for browser consent on first use; caches token in `~/.config/colab-cli/token.json`.

### Verification & Session Discovery

Always check authentication and discover existing server assignments before provisioning new compute:
```bash
colab sessions
colab whoami
```
*Interpreting `colab sessions`:*
- `[name] <assignment_id> | Hardware: L4 ...`: Session tracked locally under `name`. Reattach directly via `-s name`.
- `[?] <assignment_id> | Hardware: L4 ...`: Remote assignment active under the user's account (e.g. provisioned from Colab web UI or previous CLI session) but untracked in local `sessions.json`.
- When an untracked assignment is already running, run `colab new -s <name> --gpu <TYPE>` to attach to and name that active assignment rather than creating duplicate VMs.

### Interactive SSH (measured 2026-10-05)

`colab ssh` opens an interactive shell on a CLI-tracked session (`colab ssh -s NAME`,
ProxyCommand mode for IDE/scp integration — see `colab ssh --help`). Measured boundary:
`exec` and `ssh` against a browser-provisioned session fail with `not found` even though
`sessions` lists it as `[?]` — browser-owned runtimes are CLI-unreachable by design, so
browser sessions execute via notebooks/pasted commands, never via CLI dispatch.

### CLI → browser handoff (help-verified 2026-10-05, live attachment untested)

The reverse direction has a built-in command — no adoption script needed:

```bash
colab sessions            # confirm the CLI session still exists (stop destroys it)
colab url -s NAME         # print the attach URL (or --open on a desktop)
colab log -s NAME -o session_history.ipynb   # export recorded CLI history separately
```

The URL opens an empty scratch notebook attached to the existing VM (dual `?dbu=` +
`#datalabBackendUrl=` signals; do not hand-edit it). It does not reconstruct CLI-run
cells — use `colab log` for history. Verify attachment with the marker test before
doing anything stateful: set a UUID marker via `colab exec`, then read it back from a
browser cell along with hostname/pid; matching values prove shared kernel. Then
`colab sessions` to confirm no second endpoint appeared. Never use `colab new`,
`restart-kernel`, or `stop` as connection-repair steps (they allocate/reset/terminate).
Do not click the ordinary Connect button if the scratch notebook looks disconnected —
that allocates a separate CPU runtime (upstream issue #24).

Measured 2026-10-06: attach FAILED against a live CLI L4 session. Correct unencoded
dual-signal URL used verbatim; browser landed disconnected (a pasted variant with the
`#datalabBackendUrl` value percent-encoded was also tried and rejected — the fragment
must stay unencoded). Installed CLI is 0.7.4 and already emits the current format, so
format staleness is unlikely; cause is unverified (account/session policy suspected).
Session stopped after the failure to end idle burn. Retry only with a live session and
the marker test; until one succeeds, browser-owned execution (notebook/nbconvert) stays
the working path for Drive-mounted work.

## Session Lifecycle & Compute Discipline

Running GPU runtimes burn billable compute credits every second. All agents must enforce strict compute discipline:

1. **Discover Before Provisioning:** Always run `colab sessions` before launching new compute. If an active session or compatible assignment exists, attach to it. Never provision duplicate concurrent GPU sessions.
2. **Execution Priority (Start Jobs First):** When a Colab session is provisioned and waiting, **start the remote jobs immediately**. Never keep remote compute idling while reading review documents, analyzing PDFs, or planning locally. Conduct document analysis and planning in parallel while the remote GPU job executes.
3. **Immediate Teardown (Zero Idle Burn):** Stop the session immediately upon job completion or unrecoverable error (`colab stop -s <session_id>`). Verify with `colab sessions` that 0 active billable assignments remain.

## Hardware Validation Protocol

Before kicking off training or benchmark jobs, probe and validate the remote hardware to ensure it matches the task specification and adheres to the repository Hardware Policy:

### 1. Probe GPU and Memory
Execute a hardware check one-liner:
```bash
echo "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('VRAM (GiB):', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0)" | colab exec -s <session>
```

### 2. Verify Hardware Policy Compliance
- **Standard Baseline (NVIDIA L4, 22–24 GiB VRAM):** Use for all primary training, rival benchmarking, and multi-horizon rollouts.
- **Diagnostic / Smoke Tests (NVIDIA T4 or CPU):** Use for simple script validation or single-batch tests.
- **Gated Accelerators (A100, H100, G4):** Strictly forbidden without prior profiler evidence or measured held-out gain justification (`docs/production/colab_l4_operator_brief.md`).

### 3. Verify Filesystem and Storage Staging
- High-throughput training must stage data on local Colab storage under `/content` (e.g. `/content/cache/` or `/content/data/`).
- Persistent run outputs, manifests, and checkpoints must verify access to `/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production`.

## Common Workflows

### 1. Provision or Attach to a Session

Always assign an explicit session name via `-s <name>` to prevent ambiguous random hex IDs:
```bash
# GPU runtime (options: L4, T4, G4, A100, H100)
colab new -s l4-worker --gpu L4

# CPU runtime
colab new -s cpu-worker
```
*Notice:* GPU/TPU availability depends on account tier and quota. If an accelerator request fails (400), fall back to `--gpu T4` or CPU.

**`colab new` cannot adopt an untracked assignment.** A `[?] <assignment_id>` entry is a
runtime started outside the CLI (typically from the Colab web UI). Re-running `colab new -s
<name> --gpu <type>` **provisions a second VM instead of adopting it** — verified twice on
2026-10-04 — and `-s <assignment_id>` fails with *not found*. A browser-spawned runtime is
therefore unreachable headlessly: either ask the user to release it, or provision your own and
tell them explicitly which session to stop so credits are not burned twice. Check
`colab sessions` before and after provisioning.

### 2. Execute Code

- **Run a local script remotely:**
  ```bash
  colab exec -s my-session -f scripts/train.py
  ```
- **Stream piped code:**
  ```bash
  echo "import torch; print(torch.cuda.is_available())" | colab exec -s my-session
  ```
- **Execute a notebook:**
  ```bash
  colab exec -s my-session -f notebooks/01-production/smoke_test.ipynb
  ```
  Writes output to `<basename>_output.ipynb`.
- **Save remote image/plot outputs:**
  ```bash
  colab exec -s my-session -f plot.py --output-image results/loss.png
  ```

### 3. Ephemeral Jobs (`colab run`)

For one-off runs that create a VM, execute a script, and terminate automatically on exit:
```bash
colab run --gpu T4 -s temp-job scripts/run_eval.py --arg1 val
```
- Exit codes propagate (non-zero if script raises an unhandled error).
- Stderr contains CLI telemetry; stdout contains the script's output.

### 4. File Transfers & Remote Management

- **Upload files or data:**
  ```bash
  colab upload -s my-session local_file.tar.gz /content/
  ```
- **Download outputs or checkpoints:**
  ```bash
  colab download -s my-session /content/checkpoint.pt ./checkpoints/
  ```
- **List remote files:**
  ```bash
  colab ls -s my-session /content/
  ```
- **Install packages on the remote VM:**
  ```bash
  colab install -s my-session torch torchvision
  colab install -s my-session -r requirements.txt
  ```

### 5. Monitoring & Teardown

- **Check status and hardware utilization:**
  ```bash
  colab status -s my-session
  ```
- **View event logs:**
  ```bash
  colab log -s my-session -n 30
  ```
- **Export session log as notebook or markdown:**
  ```bash
  colab log -s my-session -o session_summary.ipynb
  ```
- **Get web UI connection URL:**
  ```bash
  colab url -s my-session
  ```
- **Clean up (Mandatory):**
  ```bash
  colab stop -s my-session
  ```
  Always release compute to prevent burning compute units.

## Agent Guardrails

- **Do NOT run interactive commands in automated loops:** `colab repl`, `colab console`, `colab auth`, and `colab drivemount` expect an interactive TTY and can hang headless agents. Use non-interactive `colab exec` or batch pipes instead.
- **`colab drivemount` cannot be used headlessly.** Drive authorisation needs a human. On a fresh runtime a headless agent must either avoid Drive entirely (stage to `/content`, pass an explicit output root) or ask the user to mount it first. Budget for this before promising a pipeline that writes checkpoints to Drive.
- **Long jobs must be launched detached.** `colab exec` runs in the *shared* kernel, so a foreground job blocks every later command until it finishes — a 10-minute extraction makes every subsequent `colab exec` appear to hang. Launch detached and poll the log:
  ```bash
  colab exec -s my-session --timeout 60 <<'PY' 2>&1 | tail -5
  import subprocess
  open('/content/launch.sh','w').write(
      'cd /content/para_001 && exec python scripts/train.py > /content/logs/run.log 2>&1\n')
  print(subprocess.run(
      ['bash','-c','setsid nohup bash /content/launch.sh >/dev/null 2>&1 </dev/null & disown; echo launched'],
      capture_output=True, text=True, timeout=20).stdout.strip())
  PY
  ```
  `setsid` plus `nohup` plus `</dev/null` matters: without them the job dies with the exec's shell. Poll with a short `colab exec` that only reads the log and `pgrep`s the pattern.
- **`colab exec --timeout` defaults to 30 seconds** (`colab exec -h`). Pass `--timeout` explicitly for anything slower, and remember your *shell* timeout is separate and usually shorter — a 120 s shell timeout can cut off a healthy remote job.
- **Recover a wedged kernel with `colab restart-kernel`.** Symptoms: `RuntimeError: Connection was lost.` from `colab exec`, `colab status` stuck at `Status: BUSY (exec(stdin))`, and `colab log` showing no new entries. `/content` survives a kernel restart, so cached data and detached jobs' outputs are usually still there — restart, then check before relaunching.
- **`--lengths-only`-style repair beats re-extraction.** When a long extraction was interrupted, prefer rebuilding derived metadata from what is already on disk over re-running the GPU work.
- **Isolate concurrent runs:** use `--config <path>` to specify a dedicated session state file for isolated agent runs (e.g. `--config /tmp/colab_run_1.json`).
- **Research integrity in this repository:** GPU training runs for AdjointRWM occur in Colab. When a Colab run completes, use the `import-run` skill to pull small verification artefacts into `results/runs/<run_id>/` without modifying committed history.

## Diagnosing CUDA OOM on Colab

An out-of-memory error on a Colab GPU is **overwhelmingly a code defect, not a hardware limit**, and
provisioning a bigger GPU to accommodate it hides the defect and burns credits. Work through this
order before considering any hardware change (`AGENTS.md` rule 7,
`docs/production/colab_l4_operator_brief.md`):

1. **Read the requested size.** `Tried to allocate 42.19 GiB` on a 22 GiB card is a single
   enormous tensor, not a model that genuinely needs the memory.
2. **Check the traceback for the frame**, not just the last line. The frame tells you which stage
   built the tensor (here `objective_at_masks` inside an exhaustive search).
3. **Is anything materialising a whole split onto the GPU?** `[move_to_device(b) for b in
   DataLoader(...)]` holds every sample at once. Keep batches on the host, upload per use, and set
   `num_workers>0` because `.npz`/`.parquet` shards are IO-bound.
4. **Is a large combinatorial enumeration batched in one call?** Chunk over the product of the
   axes, not one axis alone — masking builds one context per *(sample, candidate)* pair.
5. **Is it running outside `torch.no_grad()`?** Retained graph nodes across many chunks accumulate.
   Measurement code that never differentiates should be under `no_grad`.
6. **Measure the true peak** with `torch.cuda.reset_peak_memory_stats()` around a single forward
   pass. If that is small (here 0.09 GiB) while the run OOMs at 42 GiB, the excess is the
   algorithm's batching, and it is fixable in code.

Only after all six is it a hardware question — and then it needs a profiler trace showing
saturation, committed as evidence, per the plan's conjunctive G4 gates.
