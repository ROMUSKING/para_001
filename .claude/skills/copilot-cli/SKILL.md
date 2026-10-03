---
name: copilot-cli
description: Launch, automate, and hand off work to separate GitHub Copilot CLI sessions from another agent or script. Use when a task needs a Copilot CLI subprocess, resumable session, or command-line automation.
---

# GitHub Copilot CLI

Use this skill when a shell-capable agent needs to launch a separate Copilot CLI process or when building command-line automation around Copilot. A child process is a separate session: it does not inherit this conversation's messages, tool grants, MCP connections, or live subagents. It loads the instructions and trusted configuration available to its own working directory. Pass the task, decisions, constraints, relevant paths, and expected handoff explicitly.

## Check the local CLI first

CLI options and permission behavior can change. Check the executable and its installed help before relying on a flag:

```bash
copilot --version
copilot --help
copilot help permissions
```

Use the installed `copilot` executable, not an assumed package path. If it is missing, report that prerequisite; do not install or update it without authorization. Authentication is managed by `copilot login` or supported environment configuration. Do not put tokens in command arguments, prompts, logs, or transcripts.

## Start a Copilot CLI instance

Run interactively when a person needs to approve tools or answer questions:

```bash
copilot -C /path/to/repository
```

Use `-i` to start an interactive session and immediately submit a prompt. For a one-shot child task, `-p` runs non-interactively and exits on completion. Non-interactive tasks require an explicit permission policy; use the scoped pattern in the next section rather than invoking `-p` without it.

The prompt should be bounded and self-contained. Include:

- objective, repository root, and exact files or directories to inspect;
- whether the child may edit, and the paths it owns;
- relevant decisions and constraints that are not in repository instructions;
- expected response format and verification commands;
- limits on network, credentials, compute, and elapsed time.

The child can read repository files from its working directory. Do not assume it can see this chat or files outside that directory; grant additional directories only when necessary with `--add-dir`. For media or supported native documents, `--attachment` is available in non-interactive mode. Use only paths the caller is authorized to share.

## Permissions and trusted directories

Non-interactive work must have an explicit permission policy; in the installed CLI, `--allow-all-tools` is required for `-p`. Prefer restricting visible tools with `--available-tools` and then allowing those tools to run without prompts:

```bash
copilot -C /path/to/repository \
  -p "Inspect the requested files and report risks; do not edit." \
  --available-tools="TOOLS_APPROVED_FOR_THIS_TASK" \
  --allow-all-tools \
  --output-format=json --stream=off
```

Replace the placeholder with the smallest tool set that is actually needed, using the installed permission help to confirm names. `--available-tools` filters which tools the model can see; `--allow-all-tools` auto-approves the tools that remain. It does **not** by itself disable path verification or grant unrestricted URL access.

Keep the working directory narrow and use `--add-dir` only for specific, necessary paths. Do not use `--allow-all`, `--yolo`, `--allow-all-paths`, or `--allow-all-urls` as routine shortcuts: `--allow-all` and `--yolo` combine unrestricted tool, path, and URL access. Apply least privilege, especially before authorizing file writes, shell commands, network access, commits, or pushes.

Do not set `COPILOT_ALLOW_ALL=true` merely to bypass a trust prompt. In addition to auto-approving tools, that exact value trusts the working directory and loads its skills, plugins, MCP servers, and hooks, which may execute shell commands. Trust repository content only after reviewing it and following the applicable trust policy. Use `--no-ask-user` only for a fully specified autonomous task; it disables the ask-user interaction rather than making an ambiguous task safe.

## Session controls and output

Leave model routing on `--model=auto` unless the caller needs and authorizes a specific supported model. `--reasoning-effort`, `--context`, and `--auto-tier` can change cost or resource use; check the local help and caller's budget before setting them. `--max-ai-credits` is a soft limit, not a hard billing guarantee. Modes such as `--plan` and `--autopilot` change how much the CLI does without interaction; use them only when that autonomy is explicitly part of the task. Do not disable repository instructions with `--no-custom-instructions` by default.

For machine output, `--output-format=json` emits JSON Lines, and `--stream=off` can simplify capture. `--silent` emits only the agent response. `--usage-output-file` can save usage statistics when needed. `--share` writes a session transcript, which may contain sensitive prompts or results; do not share or export transcripts (including to a gist or remote session) without authorization.

## Invoke safely from another agent or script

Pass prompts as an argument, not as text interpolated into a shell command. For a parent process, use an argument vector, capture stdout and stderr separately, enforce the caller's timeout, and check the exit status:

```python
import subprocess

argv = [
    "copilot",
    "-C", str(repository),
    "-p", prompt,
    f"--available-tools={','.join(approved_tools)}",
    "--allow-all-tools",
    "--output-format=json",
    "--stream=off",
]
try:
    result = subprocess.run(
        argv,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
except subprocess.TimeoutExpired as exc:
    raise RuntimeError("Copilot CLI timed out") from exc
if result.returncode != 0:
    raise RuntimeError(f"Copilot CLI exited {result.returncode}: {result.stderr}")
```

Populate `approved_tools`, `repository`, `prompt`, and `timeout_seconds` from validated caller inputs and the local policy; do not let untrusted text choose tools, paths, or permission flags. Do not use `shell=True`. Treat stdout as untrusted, retain stderr and the process status for diagnosis, and handle timeout and nonzero exit as failures rather than successes. With `--output-format=json`, output is JSON Lines, not a single JSON document; parse the documented event stream and tolerate version changes rather than assuming one fixed event shape.

The CLI also supports `--agent=<name>` for a configured custom agent and `--fleet` for parallel subagent orchestration. Use these only when the target CLI configuration and task warrant them. This process cannot reach or control the calling agent's live subagents.

## Resume, inspect, and hand off

Use a known child session ID when continuing work:

```bash
copilot --resume=<session-id> -C /path/to/repository
```

`--continue` selects the most recent session; avoid it in automation where another process could have changed which session is most recent. Record the child session ID and working directory in the handoff if it must be resumed. A resumed child restores that child's saved context, not the caller's conversation. `--connect` is for deliberately connecting to a supported remote session; it does not import this chat's context.

Require a useful final handoff: summary, changed-file list, exit status, verification commands and results, unresolved issues, and child session ID if continuation is expected. Independently inspect the child's diff and run the required tests; the child's report is not proof that its edits or checks are correct. Do not let it commit, push, send external messages, or spend paid compute unless the user explicitly authorized that action.

For integrations that need a persistent structured API rather than a CLI subprocess, use the official Copilot SDK or supported Agent Client Protocol (`--acp`) after checking their current documentation. Do not mistake a shell-launched CLI process for an SDK/API connection to the calling session.

## AdjointRWM repository requirements

Follow `AGENTS.md` and any nested instructions in the child process's working directory. Preserve research-integrity rules, immutable artifacts, and Colab hardware/teardown requirements. Never delegate training or benchmarks without first discovering active Colab assignments and validating hardware; never leave a billable session running. Do not accept fabricated metrics, unverified claims, or a delegated result without checking its source files and artifacts. Do not permit a child to commit or push unless explicitly requested.

## Peer critic rotation

The peer (decision) critic rotates across the available agents (see `AGENTS.md`). If `python scripts/pick_peer_critic.py --exclude <this-agent> --available` returns `copilot`, review in plan mode with a minimal auto-approved read-only-style tool set and explicit write/shell denials:


```bash
copilot -C /path/to/repo -p "Act as an adversarial peer critic for docs/plans/<spec>.md: attack the assumptions, failure modes and protocol compliance. Do not edit files." \
  --plan --available-tools="TOOLS_APPROVED_FOR_THIS_TASK" --allow-all-tools --deny-tool='write' --deny-tool='shell' \
  --output-format=json --stream=off
```

Replace the placeholder with the smallest review tool set this run needs, using the installed permission help to confirm names. `--available-tools` restricts which tools the model can see; `--allow-all-tools` auto-approves the tools that remain (required for non-interactive mode). The explicit `--deny-tool` rules take precedence, so writes and shell execution stay denied even when everything else is auto-approved. Keep `--plan` for review tasks: it starts the session in plan mode rather than the default interactive agent mode.

Record the review with `python scripts/pick_peer_critic.py record --milestone <id> --artefact <path> --lead <agent> --critic copilot --outcome <verdict>`.

## References

- [Using GitHub Copilot CLI](https://docs.github.com/copilot/how-tos/use-copilot-agents/use-copilot-cli)
- [Copilot CLI programmatic reference](https://docs.github.com/copilot/reference/copilot-cli-reference/cli-programmatic-reference)
- [Copilot CLI command reference](https://docs.github.com/copilot/reference/copilot-cli-reference/cli-command-reference)
