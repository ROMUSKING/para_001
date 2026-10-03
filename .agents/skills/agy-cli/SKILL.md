---
name: agy-cli
description: Operate Google Antigravity CLI (agy) as a separate command-line agent, or let another local agent invoke it. Use for headless prompts, machine-readable output, and conversation continuation.
---

# Antigravity CLI (`agy`)

Use this skill when another local agent or automation needs to call Antigravity as its own process. The process uses its own conversation, CLI settings, and cached authentication. It does not inherit this agent's chat history, tool connections, or active subagents. It does share the filesystem at the chosen working directory.

## Check the CLI and authentication

Check the installed version and available flags instead of assuming another machine has the same release:

```bash
agy --version
agy --help
agy agents
agy models
```

Headless mode uses cached credentials. Complete sign-in by launching `agy` interactively once; there is no `agy auth` or `agy login` subcommand. Do not inspect or copy the local credential store. An unauthenticated headless run exits with an authentication error.

## Launch a one-shot task

Start the process in the target repository root so it loads the applicable `AGENTS.md` instructions and sees the intended files. A one-shot prompt runs with `-p` and exits:

```bash
cd /path/to/repo
agy --output-format json --print-timeout 10m \
  -p "Review src/adjointrwm/allocators.py for correctness risks. Do not edit files."
```

For another agent or program, invoke `agy` with an argument vector and set the process `cwd`; do not build a shell command by interpolating prompt text. For example:

```python
result = subprocess.run(
    ["agy", "--output-format", "json", "--print-timeout", "10m", "-p", prompt],
    cwd=repo_root,
    capture_output=True,
    text=True,
    check=False,
)
```

Pass the task, relevant paths, constraints, allowed write scope, and required checks explicitly. Use `agy --mode=plan -p "..."` for a read-only plan or review. Use `--mode=accept-edits` only when the task authorizes file changes. The CLI follows its configured permission policy; headless actions that require unavailable approval may be denied while the run continues. Avoid `--dangerously-skip-permissions`. `--sandbox` enables terminal sandbox restrictions, and does not replace checking the active policy.

## Capture and continue

With `--output-format json`, stdout contains a single result object including `conversation_id`, `status`, `response`, and sometimes `error` and usage metadata; diagnostics and permission notices go to stderr. Check the process exit code, JSON `status`, and stderr. `status: SUCCESS` alone does not prove every requested tool action was allowed or completed.

Use `stream-json` when you need live tool/subagent events or usage details:

```bash
agy --output-format stream-json --print-timeout 10m \
  -p "Inspect the current diff and return prioritized findings."
```

Save the returned `conversation_id` to continue that exact conversation:

```bash
agy --output-format json --conversation <CONVERSATION_ID> \
  -p "Address the second finding and report the changed files."
```

Headless runs are otherwise stateless. `--continue`/`-c` chooses the most recent conversation, so prefer an explicit ID in automation or concurrent work. Set a finite `--print-timeout` in unattended workflows.

For multiple dependent turns in one long-running process, use both `--input-format stream-json` and `--output-format stream-json`. Send one JSON line per prompt, such as `{"event":"user","message":{"content":"Review the changed function."}}`; read events through that turn's `result`, then send the next prompt. Close stdin when finished. Do not send `-p` in this mode.

## Return a useful handoff

Collect and report the process exit code, parsed status, final response, stderr notices, conversation ID, changed files, and verification commands/results. Inspect `git diff` yourself and run the requested checks independently; an agent's report is not proof that a change or check succeeded. Use the CLI's native subagent features when already operating inside Antigravity; this skill is for other agents that need a separate `agy` process.

Follow the repository's `AGENTS.md` and relevant skills. In this research repo, preserve the research-integrity rules, never treat unverified outputs as evidence, and do not run GPU training locally. Colab work must follow `colab-cli` hardware validation and teardown rules. Do not authorize commits, pushes, or external actions unless the user explicitly requested them.

## Peer critic rotation

The peer (decision) critic rotates across the available agents (see `AGENTS.md`). If `python scripts/pick_peer_critic.py --exclude <this-agent> --available` returns `agy`, review in plan mode:

```bash
agy --mode=plan --output-format json --print-timeout 10m \
  -p "Act as an adversarial peer critic for docs/plans/<spec>.md: attack the assumptions, failure modes and protocol compliance. Do not edit files."
```

Record the review with `python scripts/pick_peer_critic.py record --milestone <id> --artefact <path> --lead <agent> --critic agy --outcome <verdict>`.

## References

- [Antigravity CLI headless mode](https://www.antigravity.google/docs/cli/headless/)
- [Antigravity CLI best practices](https://antigravity.google/docs/cli/best-practices/)
- [Antigravity CLI execution modes](https://antigravity.google/docs/cli/modes)
