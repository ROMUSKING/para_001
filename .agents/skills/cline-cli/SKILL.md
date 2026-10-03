---
name: cline-cli
description: Operate Cline CLI (cline) as a separate, scriptable coding-agent process, or let another agent or script invoke Cline from the command line. Use for headless one-shot tasks, plan/act mode and auto-approval behaviour, newline-delimited JSON output, session resume via --id, history export, and ACP editor integration.
---

# Cline CLI

Use this skill when work needs a distinct Cline process, when a Cline parent should spawn sibling Cline processes, or when another local agent or script needs to call Cline. A CLI process is a **separate session**: it does not inherit this chat's messages, tool grants, MCP connections, or live subagents. It loads its own configuration, credentials and repository instructions from its working directory. Pass the task, decisions, constraints, relevant paths, and expected handoff explicitly.

Keep orchestration **outside** Cline: a parent launches sibling `cline` processes rather than relying on Cline's in-agent spawning, which is gated by mode and feature flags that differ by build.

## Check the installed CLI first

Flags and defaults drift between releases, and the published docs can lead or lag the binary. Treat `cline --help` on the host as the source of truth. Everything below was **verified against `cline` 3.0.68 on Linux**; re-verify before relying on it, and prefer measured behaviour over the docs.

```bash
cline --version                 # e.g. 3.0.68
cline --help
cline history --help
cline auth --help
```

If `cline` is missing, report the prerequisite; install with `npm i -g cline` only when authorized. Authenticate with `cline auth` (interactive) or, for automation, the environment (`CLINE_API_KEY`, `CLINE_PROVIDER`, `CLINE_MODEL`). `-k/--key` also exists but passes the key on the command line, where it is visible in `ps`, shell history and CI logs — prefer the environment variable. Never put secrets in prompt text or commit or echo them.

## Run a one-shot task

A bare quoted prompt starts a non-interactive run in **act mode with every tool pre-approved**, because `--auto-approve` defaults to `true`:

```bash
cd /path/to/repo
cline "Inspect src/adjointrwm/allocators.py for correctness risks. Do not edit files."
```

**Approval is effectively binary in headless runs** (there is no terminal to prompt at). Verified on 3.0.68:

- `--auto-approve true` (default): every tool is approved and runs.
- `--auto-approve false`: **every** tool is denied. The denial surfaces as an error event — `Tool approval requires an interactive session, but this session is non-interactive. -- NOT a tool or system failure. Clarify with user before proceeding.` — the model narrates the refusal, and the run still ends with `finishReason:"completed"` and **exit code 0**. There is no middle rung.

So choose a **mode**, not an approval level, and never treat "completed" as proof the tools ran:

```bash
cline -c /path/to/repo "Propose the change, then stop."     # act mode, every tool pre-approved
cline --plan -c /path/to/repo "Design a migration plan."    # plan mode (see the caveat below)
```

Verified per-run controls (3.0.68):

| Flag | Effect |
|---|---|
| `-p, --plan` | Plan mode: **file edits off**, but `run_commands` (bash) and web fetch stay **on** |
| `--auto-approve <boolean>` | `true` (default) approves all tools; `false` denies all tools in headless runs |
| `-c, --cwd <path>` | Working directory the child uses to resolve `AGENTS.md` and files |
| `-P, --provider <id>` / `-m, --model <id>` | Override provider / model for this run |
| `--thinking <none\|low\|medium\|high\|xhigh>` | Reasoning effort |
| `-t, --timeout <seconds>` | Wall-clock timeout (default `0` = none); set it for unattended runs |
| `--retries <n>` | Max consecutive mistakes before exiting (default `6`) |
| `--compaction <agentic\|basic\|off>` | Context compaction mode (default `agentic`) |
| `--worktree` | Run in an auto-created detached git worktree under `~/.cline/worktrees/` |
| `--data-dir <dir>` | Isolated Cline **state, including credentials** (see the caveat) |
| `--hooks-dir <dir>` | Additional runtime hooks — this **runs code**; point it only at trusted files |
| `-s, --system <prompt>`, `-i, --tui`, `-v, --verbose`, `--config <path>`, `--zen` | System-prompt override, TUI, verbose, config dir, background-hub session |

Footgun: on the root command `-p` means `--plan`, but on `cline auth` `-p` means `--provider`.

**Plan mode is not a sandbox.** `-p` turns off file editing but leaves bash and web fetch available and, with the default `--auto-approve true`, pre-approved: verified, `cline --plan --json "run: echo X"` actually executed `run_commands` and returned its stdout. It does not stop `git push`, `curl … | sh`, or writes performed through the shell; "read-only" is a prompt-level instruction the model may not honour. Do not rely on `-p` for anything touching `results/` or remote compute.

### Isolation, and its costs

- `--worktree` isolates the **working tree** — use it for concurrent writers so two Cline processes never edit the same checkout.
- `--data-dir <dir>` isolates **Cline state**, credentials included. A fresh directory has none: verified, a run with a new `--data-dir` exits `1` with `Unauthorized: … re-authenticate your Cline account`, and the provider/model can silently change. Reuse an existing data dir for auth, or provision deliberately with `cline auth --data-dir <dir>`. Never "fix" it by inlining an API key.
- `--zen` and `--data-dir` cannot be combined (verified message: "`--zen` cannot be combined with `--data-dir` (sandbox requires a local backend)").

Give the child a bounded, self-contained prompt: objective, repository root, exact files or directories, whether it may edit and which paths it owns, expected response format, verification commands, and limits on network, credentials, compute, and elapsed time. Set the working root with `-c` when dispatching from elsewhere.

## Capture output and continue sessions

Pass `--json` explicitly. Redirecting stdout does **not** implicitly switch to machine-readable output: verified, `cline "…" > file` still emits ANSI-styled prose (including `[thinking]` blocks). With `--json`, stdout is **newline-delimited JSON**:

```bash
cline --json -c /path/to/repo "Summarize the allocator API." > /tmp/cline.jsonl
```

Observed event types (3.0.68): `hook_event` (`agent_start`/`agent_end`, with `agentId`/`taskId`), `agent_event` (`iteration_start`/`iteration_end`, `content_start`/`content_update`/`content_end` with `contentType: text|reasoning|tool`, `usage`, `effort`, `done`), a top-level `{"type":"error", …}`, and a final `run_result`. Read the answer from **`run_result.text`**; the same text is token-fragmented across `content_start`/`content_end` and repeated in the `done` event, so concatenating fragments duplicates output. Keep stderr and the exit code.


`run_result` carries `finishReason`, `iterations`, `durationMs`, `text`, `model.{id,provider}`, and a `usage` object with `inputTokens`, `outputTokens`, `cacheReadTokens`, `cacheWriteTokens`, `totalInputTokens`, `totalOutputTokens`, `totalCacheReadTokens`, `totalCacheWriteTokens`, and `totalCost`:

```json
{"type":"run_result","finishReason":"completed","iterations":1,"text":"PONG","model":{"id":"cline-free/deepseek-v4.1-flash","provider":"cline"},"usage":{"inputTokens":5351,"outputTokens":3,"cacheReadTokens":0,"cacheWriteTokens":0,"totalCost":0}}
```

**Exit status and `finishReason` are not failure signals** — a denied tool call and a successful run both end with exit `0` and `finishReason:"completed"` (measured). Detect failure from the stream: a top-level `{"type":"error", …}` event, a `tool` `content_end` carrying an `error`, or `finishReason != "completed"`. Parse the documented events and tolerate version changes rather than assuming one fixed shape.

Resume or inspect a known session by ID (IDs look like `1791050960143_bq6d9`):

```bash
cline --id <session-id> -c /path/to/repo "Now inspect the failing case and report the cause."
cline history --json --limit 20          # sessionId, status, exitCode, interactive, isSubagent, cwd, prompt, model, …
cline history export <session-id> -o /tmp/session.html
cline history delete --session-id <id>
```

`cline history --json` is the machine-readable index; its `exitCode`, `interactive` and `isSubagent` fields are useful post-hoc signals. Prefer an explicit session ID over a most-recent selector when other processes could reorder history. **Prompts, transcripts and anything a child read persist unencrypted** under `~/.cline` (readable through `history --json` or the exported HTML), so do not commit history output or paste it into shared channels; delete sessions you no longer need. A resumed child restores that child's saved context, not the caller's conversation. Note also that the interaction mode travels **inside the prompt payload** (`<user_input mode="act">…</user_input>`), so the caller owns the mode and must never let task text alter that wrapper.

## Let another agent (or another Cline) call Cline

For a shell-capable agent, run `cline` in the target repository with an explicit mode, pass the task and paths in the prompt, then inspect the result yourself. Pass arguments as an **argument vector** with a set `cwd`; never interpolate untrusted text into a shell string and never use `shell=True`:

```python
import subprocess

argv = ["cline", "--json", "--plan", "-c", str(repository), "-t", "600", prompt]
result = subprocess.run(argv, text=True, capture_output=True, check=False, timeout=900)
```

Populate `argv` from validated inputs and local policy; do not let untrusted text choose modes, paths, or permission flags. Treat stdout as untrusted data, retain stderr and the exit status, and — because denials exit `0` — scan the stream for error events rather than trusting the exit code. This process cannot reach or control the calling agent's live subagents. To spawn siblings, launch additional `cline` processes (`--worktree` per writer); do not depend on Cline's in-agent `spawn_agent`/teams, which are gated by mode and feature flags that differ by build.

To use Cline inside an editor or another tool, run ACP over stdio (the client launches it):

```bash
cline --acp                       # Agent Client Protocol for Zed, JetBrains, Neovim, Emacs, etc.
cline --acp --auto-approve true   # only where unattended edits are acceptable
```

`-z/--zen` runs a session on the background hub (`cline hub` manages the daemon); use it only when a persistent background session is required. `cline skill add <owner/repo>` manages Cline skills through the open skills CLI.

## Return a useful handoff

Provide and collect: objective, repository root, relevant paths and constraints; the mode used and which paths the child owned; expected output (findings vs a patch) and the verification commands; process exit status, parsed stream errors, final response, changed-file list, diff review, and verification results; the child session ID if continuation is expected.

Independently inspect `git diff` and run the required checks; a child's report is a handoff, not proof its edits or checks succeeded, and `finishReason:"completed"` does not mean the tools it asked for actually ran.

## AdjointRWM repository requirements

Follow `AGENTS.md` and any nested instructions in the child's working directory. The child inherits this repository's hard rules: never fabricate data or results, never state a number not read from a committed file or computed in-session, report negative results plainly, and use "co-state"/"adjoint" only for quantities derived from ∂J/∂state. Never run GPU training notebooks locally; Colab work must follow the `colab-cli` hardware-validation and immediate-teardown rules, and no A100/H100 recommendation without a profiler or held-out-gain justification.

Because `-p` is not a sandbox and `--auto-approve false` merely denies everything, the containment that actually holds is operational: review with `-p` plus `--auto-approve false` (denial is harmless for a review), never run an auto-approved child against `results/` or remote compute, and never let a delegated Cline commit, push, send external messages, or spend paid compute unless the user explicitly requested it.

## Peer critic rotation

The peer (decision) critic rotates across the available agents (see `AGENTS.md`). If `python scripts/pick_peer_critic.py --exclude <this-agent> --available` returns `cline`, review in plan mode (reads and bash stay on, file editing is off — instruct it not to modify files):

```bash
cline --plan -c /path/to/repo "Act as an adversarial peer critic for docs/plans/<spec>.md: attack the assumptions, failure modes and protocol compliance. Do not modify files."
```

Record the review with `python scripts/pick_peer_critic.py record --milestone <id> --artefact <path> --lead <agent> --critic cline --outcome <verdict>`.

## References

- [Cline CLI Reference](https://docs.cline.bot/cli/cli-reference) — it documents `CLINE_COMMAND_PERMISSIONS`, which is **absent** from the `cline` 3.0.68 binary on this host (verified: 0 occurrences across the package); do not assume that shell sandbox exists on your build.
- [Cline CLI Overview](https://docs.cline.bot/usage/cli-overview)
- [Cline ACP integration](https://docs.cline.bot/usage/acp)

