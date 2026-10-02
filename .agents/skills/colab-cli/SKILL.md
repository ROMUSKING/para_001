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
- **Isolate concurrent runs:** use `--config <path>` to specify a dedicated session state file for isolated agent runs (e.g. `--config /tmp/colab_run_1.json`).
- **Research integrity in this repository:** GPU training runs for AdjointRWM occur in Colab. When a Colab run completes, use the `import-run` skill to pull small verification artefacts into `results/runs/<run_id>/` without modifying committed history.
