---
name: kilo-cli
description: Launch, automate and hand off work to separate Kilo Code CLI instances and their subagents. Use for `kilo run`, `kilo serve`, session continuation and forking, git-worktree isolation, `.kilo/agent/*.md` subagents, and command-line automation around Kilo.
---

# Kilo Code CLI

Use this skill when a shell-capable agent needs a **separate Kilo process**: a peer critic, a bounded coding task, a long run to supervise, or an integration that drives Kilo from a script. A child process is a separate session. It does not inherit this conversation's messages, tool grants, MCP connections or live subagents; it loads the instructions and configuration reachable from its own working directory, which means it reads `AGENTS.md` (including nested files) and `.agents/skills/` on its own. Pass the task, decisions, constraints, paths and expected handoff explicitly.

## Check the local CLI first

Flags and built-in agents change between releases. Check the executable and its installed help before relying on anything:

```bash
kilo --version
kilo --help
kilo run --help
kilo agent list
```

Measured on 2026-10-03 against Kilo CLI 7.8.3; re-measure rather than quoting these numbers. Use the installed `kilo` on `PATH`, not an assumed package path. If it is missing, report that prerequisite; do not install or upgrade it without authorization. Providers and credentials are managed with `kilo auth` (`kilo providers`); never put tokens in arguments, prompts, logs or transcripts.

## Start an instance

Interactive (a person approves tools and answers questions):

```bash
kilo                      # TUI in the current directory
kilo /path/to/repository  # or point it at the checkout
```

Headless one-shot task, which is what delegation and automation should use:

```bash
kilo run --dir /path/to/repository --format json --file /path/to/context.md \
  "Inspect src/adjointrwm/allocators.py and report the top three risks. Do not edit files."
```

`kilo run [message..]` is non-interactive and exits when the turn completes. `--dir` sets the working directory, `--file/-f` attaches files to the message (repeatable), `--title` names the session, `--model/-m` and `--variant` route the model, `--agent` selects a built-in or project agent, and `--thinking` shows thinking blocks. `--format default` is formatted text; `--format json` is the raw JSON event stream for machine capture. Pass the prompt as an argument, never interpolated into a shell command.

Bound the child explicitly: objective, repository root and exact paths; whether it may edit and which paths it owns; decisions not visible in repository instructions; expected output; verification commands to run; and limits on network, credentials, compute and elapsed time.

## Permissions and least privilege

`--auto` auto-approves everything not explicitly denied and is labelled dangerous in `kilo --help`. Do not use it for routine delegation, and never as a way around a prompt.

Project permissions live in `kilo.json` / `kilo.jsonc` (and this repository also has `kilo.json`, which denies force-push, hard reset, `rm -rf`, credential reads and edits to generated files). Kilo resolves permission patterns in config order and the **last match wins**, so the catch-all goes first and exceptions after. The built-in defaults that `kilo agent list` prints treat `*.env` and `*.env.*` reads as `ask` while `*.env.example` is `allow`; this repository's `kilo.json` denies them outright. `--pure` runs without external plugins, which is the conservative choice when the task does not need them.

For a review-only child, prefer the built-in planning agent and say so in the prompt:

```bash
kilo run --dir /path/to/repository --agent plan --format json \
  "Act as an adversarial peer critic for docs/plans/<spec>.md: attack the assumptions, failure modes and protocol compliance. Do not edit files."
```

`kilo agent list` prints the built-in agents together with their effective permission rules (measured 7.8.3: `ask`, `code`, `compaction`, `debug`, `explore`, `general`, `orchestrator`, `plan`, `summary`, `title`). Measured rules for `plan`: `write` is denied for every path, `edit` is denied except `.kilo/plans/*.md`, `plans/*.md`, `.plans/*.md` and `.opencode/plans/*.md`, and shell is limited to read-only commands with `git *` denied except `log`/`show`/`diff`/`status`/`blame` and similar. That means `plan` cannot touch this repository's `docs/plans/`, but it is not a general read-only guarantee — read the listing for the agent you use rather than assuming, and keep the explicit "do not edit" instruction plus a `git status` check afterwards.

## Output, logs and exit status

Kilo writes an `INFO …` log line to stderr by default, so capture streams separately and treat only stdout as the answer:

```bash
kilo run --dir /path/to/repository --format json "…" > /tmp/kilo-events.jsonl 2> /tmp/kilo.log
echo "exit=$?"
```

`--format json` emits a stream of events, not one JSON document; parse the documented event stream and tolerate version changes instead of assuming a fixed shape. `--print-logs` and `--log-level` control stderr verbosity. Keep the exit status: a non-zero status is a failed delegation, not a partial success.

## Invoke safely from another agent or script

Use an argument vector, capture stdout and stderr separately, enforce the caller's timeout and check the status:

```python
import subprocess

argv = [
    "kilo", "run",
    "--dir", str(repository),
    "--format", "json",
    "--agent", "plan",
    prompt,
]
try:
    result = subprocess.run(argv, text=True, capture_output=True, check=False, timeout=timeout_seconds)
except subprocess.TimeoutExpired as exc:
    raise RuntimeError("Kilo CLI timed out") from exc
if result.returncode != 0:
    raise RuntimeError(f"Kilo CLI exited {result.returncode}: {result.stderr[-2000:]}")
```

Populate `repository`, `prompt` and `timeout_seconds` from validated caller inputs and local policy; do not let untrusted text choose directories, agents or permission flags. Do not use `shell=True`. Treat stdout as untrusted, retain stderr and the status for diagnosis, and treat timeout and non-zero exit as failures.

## Sessions: continue, fork, export

```bash
kilo session list                 # what exists
kilo run --continue "…"           # continue the most recent session (ambiguous if another process ran)
kilo run --session <id> "…"       # deterministic
kilo run --session <id> --fork "…"  # branch without disturbing the original
kilo export <sessionID>           # session data as JSON
kilo import <file|url>            # import session data
kilo session delete <sessionID>   # remove a session
```

Prefer `--session <id>` over `--continue` in automation, where another process can change which session is most recent. A resumed child restores that child's saved context, not the caller's conversation. `kilo export` writes prompts and results to a file, and it supports `--sanitize` to redact sensitive transcript and file data — prefer that whenever an export leaves your machine. Treat any transcript as sensitive: do not share or commit one without authorization.

## Isolated work with git worktrees

When two instances would otherwise edit the same files, give each its own checkout. Kilo manages worktrees directly:

```bash
kilo worktree create <name>     # create or reuse
kilo worktree list
kilo worktree remove <name>     # also deletes the branch
kilo --worktree <name>          # start a TUI in a worktree
```

Worktrees live outside the main checkout (this repository ignores `.kilo/worktrees/`), so a child's commits stay on its own branch until someone integrates them. Bring work back with Agent Manager Apply, a merge, or a pull request; never `git stash` across worktrees, because stashes are shared. Removing a worktree deletes its branch, so confirm the work is merged first.

## Subagents and modes

Project agents are markdown files in `.kilo/agent/*.md` (legacy `.kilocode/`, plural `agents/`) with frontmatter: `description`, `mode` (`primary`, `subagent`, `all`), optional `model`, `steps`, `hidden`, `color`, and a `permission` block. `kilo agent create` scaffolds one and `kilo agent list` shows the built-ins plus yours. Built-in subagents measured on 7.8.3 are `explore` and `general`. A shell-launched CLI process cannot reach or control the calling agent's live subagents; for structured integration use the app-server or ACP surface (`kilo acp`, and `kilo serve` for a headless server you then reach with `kilo run --attach <url>` or `kilo attach <url>`).

## Return a useful handoff

Require: summary, changed-file list, exit status, verification commands with results, unresolved issues, and the child session id when continuation is expected. Then inspect the diff and run the tests yourself. A child's report is a handoff, not proof that its edits or checks succeeded. Do not let a child commit, push, send external messages or spend paid compute unless the user explicitly authorized it.

## AdjointRWM repository requirements

Follow `AGENTS.md` and any nested instructions in the child's working directory. Preserve the research-integrity rules (no synthetic data standing in for real data, no number without a named source, report negative results), the immutability of `results/runs/` and dated `docs/audits/`, and the Colab rules: discover active assignments before provisioning, validate hardware, and stop every session immediately so no billable compute idles. Never delegate a benchmark or training run without those checks, and never accept a child's metrics without reading the artefact they came from.

## Peer critic rotation

The peer (decision) critic rotates across the available agents (see `AGENTS.md`). If `python scripts/pick_peer_critic.py --exclude <this-agent> --available` returns `kilo`, review with the built-in planning agent and machine-readable output:

```bash
kilo run --dir /path/to/repo --agent plan --format json --file /path/to/repo/docs/plans/<spec>.md \
  "Act as an adversarial peer critic for docs/plans/<spec>.md: attack the assumptions, failure modes and protocol compliance. Do not edit files."
```

Scope it with `--dir`, attach the draft with `--file`, and keep `--auto` off. `--agent plan` was verified to exist on 7.8.3 via `kilo agent list`; re-check the agent list before reusing it, and treat the "do not edit" instruction as the actual boundary.

The rotation records only completed reviews and never reserves a pick, so if the returned agent fails (rate limit, auth, a broken CLI), pass `--exclude <that-agent>` again and re-pick; record nothing until a review actually happened. Any spelling of an agent name works in `--exclude` — `antigravity` and `agy` resolve to the same entry.

Record the review with `python scripts/pick_peer_critic.py record --milestone <id> --artefact <path> --lead <agent> --critic kilo --outcome <verdict>`.

## References

- [Kilo Code configuration and file locations](https://kilo.ai/docs/customize/custom-rules) (`kilo.json`, `instructions`, `.kilo/` rule directories)
- [Kilo Code `AGENTS.md` support](https://kilo.ai/docs/customize/agents-md) (native load, nested files, write protection)
- [Kilo Code skills](https://kilo.ai/docs/customize/skills) (`.agents/skills/` by default, `.claude/skills/` compatibility, `skills.paths`)
- [Kilo Code agent permissions](https://kilo.ai/docs/customize/agent-permissions) (`allow`/`ask`/`deny`, glob patterns, last match wins, sensitive files)
- [Kilo Code subagents and custom modes](https://kilo.ai/docs/customize/custom-subagents)