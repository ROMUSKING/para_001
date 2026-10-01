---
name: opencode-delegate
description: Delegate autonomous coding tasks, prompt workflows, and script executions to OpenCode (opencode CLI v2). Use when delegating prompts or background tasks to OpenCode, orchestrating multi-agent collaboration, running headless coding runs, or inspecting OpenCode sessions.
---

# OpenCode Delegate

**Goal:** delegate coding tasks, prompt workflows, and script executions to OpenCode using the `opencode` CLI (v2 architecture), managing sessions and verifying results.

## Prerequisites

Verify OpenCode installation and available models:
```bash
opencode --version
opencode models
opencode auth list
```

## Architecture & Modes

OpenCode v2 introduces a client-server architecture:
- **Default mode:** connects to the shared background service (`opencode service`).
- **`--standalone` mode:** runs with an isolated, private server instance. Recommended for background tasks, automated scripts, and test runs to avoid interference with interactive user sessions.
- **Output formats:**
  - `--format default`: human-readable output streamed directly to stdout.
  - `--format json`: streaming JSON Lines events (`step_start`, `text`, tool executions, `sessionID`).

## Core Delegation Commands

### 1. Basic Headless Task Delegation

Send an instruction directly to OpenCode:
```bash
opencode run -m opencode/space-bunny-free "Refactor function foo in src/adjointrwm/utils.py to handle None inputs"
```

### 2. Standalone Execution with Auto-Approval

For automated delegations where manual terminal interaction is not possible:
```bash
opencode run --standalone --auto -m <provider/model> "Run pytest and summarize failures"
```
*Note:* `--auto` automatically approves permissions that are not explicitly denied in `opencode.json`.

### 3. Attaching Context Files

Attach relevant files directly to the prompt:
```bash
opencode run -f src/adjointrwm/allocator.py -f tests/test_allocator.py "Add test coverage for edge case in allocator"
```

### 4. Structured JSON Output

Capture structured JSON lines for automated parsing:
```bash
opencode run --format json -m <provider/model> "Analyze src/adjointrwm/metrics.py"
```
Each line contains a JSON object with `type` (`step_start`, `text`, etc.) and `sessionID` (formatted as `ses_<id>`).

## Multi-Turn Sessions

- **Continue the last session:**
  ```bash
  opencode run -c "Now run pytest to verify the changes"
  ```
- **Continue a specific session:**
  Session IDs must start with `ses_`:
  ```bash
  opencode run -s ses_f0b507ab2ffewM1H2sj13Va4BI "Check edge cases for negative inputs"
  ```
- **Fork an existing session:**
  Branch off a session to test an alternative approach without modifying original session state:
  ```bash
  opencode run -c --fork "Try an alternative algorithm using binary search"
  ```

## Session Lifecycle & Auditing

- **Export full session details (cost, token usage, outcome, steps):**
  ```bash
  opencode session export <ses_id>
  ```
- **Delete completed session:**
  ```bash
  opencode session delete <ses_id>
  ```
- *Avoid interactive TUI:* do not run `opencode session list` without non-interactive redirects or scripts, as it launches an interactive curses-style terminal picker.

## Permissions & Repository Rules

- OpenCode evaluates permissions configured in `opencode.json` (`permission.bash`).
- The **last matching rule wins**.
- In this repository, destructive git operations (`git push --force`, `git reset --hard`) and file wipes (`rm -rf *`) are strictly denied.
- Operations requiring human sign-off (`git commit`, `git push`, `pip install`) prompt with `ask` unless explicitly managed.

## Procedure for Delegating Agents

1. **Scope the task clearly:** specify target files, expected behaviors, and constraints.
2. **Select model & mode:** choose an available model from `opencode models` and use `--standalone` if isolation is needed.
3. **Dispatch command:** execute via `opencode run [--standalone] [--auto] [-m <model>] [-f <file>] "<prompt>"`.
4. **Inspect output:** check exit code and review text output or exported session JSON.
5. **Verify repository invariants:** always run `python harness/check.py` after OpenCode modifies files to ensure tests pass, links resolve, and no immutable files were altered.
