---
name: codex-cli
description: Operate Codex as a separate command-line agent, or make Codex available to another local agent or tool. Use for codex exec, session continuation, CLI automation, and Codex integrations.
---

# Codex CLI

Use this skill when work needs a distinct Codex CLI process or when another local agent needs to invoke Codex. A CLI process is a separate session: it does not inherit this chat's messages, tool permissions, MCP connections, or live subagents. It loads its own configuration and repository instructions from the working directory.

## Start a separate Codex instance

Run from the target repository and set the working root explicitly when dispatching from elsewhere:

```bash
codex --version
codex login status
codex exec -C /path/to/repo "Inspect the allocator API and report risks. Do not edit files."
```

`codex exec` is non-interactive. It reads repository context from the selected working directory, including applicable `AGENTS.md` files. Name the exact files or directories the agent should inspect in the prompt; it can read them from the workspace. Give it a bounded task, expected deliverable, write boundary, and verification request. Pass necessary decisions, constraints, and prior findings explicitly; do not assume it can see this session.

Use the least access required. `codex exec` defaults to a read-only sandbox. For an authorized coding task, allow workspace edits explicitly:

```bash
codex exec --sandbox workspace-write -C /path/to/repo \
  "Implement the scoped fix in src/example.py and its focused test. Report changed files and verification."
```

Do not use `--dangerously-bypass-approvals-and-sandbox` for routine delegation. If concurrent work could overlap, use `--worktree` or make the delegated task read-only; coordinate before letting two instances edit the same files. Authorization for an outer task does not silently authorize commits, pushes, external messages, paid compute, or other gated actions.

## Capture, continue, and inspect runs

Use JSONL when another process needs progress events or final output in a parseable form. `--json` sends events to stdout; retain stderr and the process exit code as well:

```bash
codex exec --json -C /path/to/repo "Review the current diff for correctness issues."
codex exec -o /tmp/codex-final.txt -C /path/to/repo "Summarize the current implementation."
```

Record the session/thread ID from the `thread.started` event if you need deterministic continuation. Continue or fork with:

```bash
codex exec resume <SESSION_ID> "Now inspect the failing case and report the cause."
codex exec fork <SESSION_ID> "Explore an alternative implementation without changing the original session."
```

`codex exec resume --last` selects the latest run for the current working directory; use a session ID when the target must be unambiguous. Check the installed CLI's `codex exec --help` and subcommand help when syntax or flags may have changed. Avoid `--ephemeral` if the session must be resumed or audited later.

## Let another agent call Codex

For a shell-capable agent, the supported simple interface is to run `codex exec` in the target repository, using read-only mode for review and `--sandbox workspace-write` only for authorized file changes. Have the caller pass the task and relevant paths in the prompt, capture the process exit code and final message, then inspect `git diff` and run the requested verification itself. Treat Codex's report as a handoff, not proof that edits or checks succeeded. When implementing a caller, pass arguments as an argument vector rather than interpolating untrusted prompt text into a shell command.

For an application that needs a persistent, structured integration, prefer the Codex SDK. Use `codex app-server` only when the integration needs its lower-level session, approval, and streamed-event protocol; the app-server protocol is experimental. The former `codex mcp-server` command and `codex-mcp-server` binary have been removed. Do not build new integrations around them. Check the current official documentation before maintaining an existing integration.

If already running inside a Codex-hosted agent session, use that session's available delegation tools for its subagents. A shell-launched CLI is a separate agent session, not a way to access or control the host's internal subagents.

## Return a useful handoff

When coordinating a second instance, provide and collect:

- task objective, repository root, relevant paths, and constraints;
- whether it may edit, and which paths it owns;
- expected output, including whether to return findings or a patch;
- commands it should run and any limits on network, credentials, or compute;
- process exit status, final response, changed-file list, diff review, and verification results.

Follow the repository's `AGENTS.md` and applicable skills. In this repository, preserve the research-integrity and Colab compute rules; never delegate benchmark execution without the required hardware/session checks. Report unverified claims as unverified and do not let a delegated agent commit or push unless the user explicitly requested it.

## Peer critic rotation

The peer (decision) critic rotates across the available agents (see `AGENTS.md`). If `python scripts/pick_peer_critic.py --exclude <this-agent> --available` returns `codex`, use it as the reviewer with an explicit read-only sandbox:

```bash
codex exec --sandbox read-only -C /path/to/repo \
  "Act as an adversarial peer critic for docs/plans/<spec>.md: attack the assumptions, failure modes and protocol compliance. Do not edit files."
```

Record the review with `python scripts/pick_peer_critic.py record --milestone <id> --artefact <path> --lead <agent> --critic codex --outcome <verdict>`.

## References

- [Codex CLI non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)
- [Codex SDK](https://learn.chatgpt.com/docs/codex-sdk)
- [Codex app-server](https://learn.chatgpt.com/docs/app-server)
- [Codex MCP server removal](https://learn.chatgpt.com/docs/mcp-server)
