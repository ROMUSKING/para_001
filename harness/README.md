# Agent harness

This is a single harness shared by every coding agent that works in this repo: **Claude Code, OpenAI Codex, Google Antigravity (IDE and `agy` CLI), OpenCode, Gemini CLI and Cursor**, plus anything else that reads `AGENTS.md` or Agent Skills.

## Design

```
                SOURCES OF TRUTH (edit these)
   AGENTS.md  notebooks/AGENTS.md  results/AGENTS.md    .agents/skills/<name>/SKILL.md    harness/mcp.json
        │                                                       │                            │
        │ read natively by Codex, Antigravity, OpenCode,        │ read natively by Codex,     │ (empty today)
        │ Cursor; Gemini CLI via .gemini/settings.json          │ Antigravity, OpenCode,      │
        │                                                       │ Gemini CLI                  │
        ▼                     harness/sync.py                   ▼                            ▼
   CLAUDE.md (@AGENTS.md import)            .claude/skills/  (mirror)           .mcp.json, .gemini/settings.json,
   notebooks/CLAUDE.md, results/CLAUDE.md   .agents/workflows/  (Antigravity)   opencode.json, .codex/config.toml
                                            .opencode/commands/ (OpenCode)
                                                        │
                            harness/check.py  (definition of done, run locally and in CI)
                            harness/hooks/protect_paths.py  (pre-edit guard)
```

**Principles:**

1. **One source, thin adapters.** Instructions live only in `AGENTS.md`, and procedures only in `.agents/skills/`. Copies exist only where a tool can't read the source directly (today, just Claude Code's `CLAUDE.md` and `.claude/skills/`). Adapter paths are data (`manifest.json`), not code, so adding a tool means adding a manifest entry.
2. **Keep the always-loaded context small; load the rest on demand.** `AGENTS.md` is short and points to docs rather than copying them. Long procedures are skills, whose bodies load only when used. Nested `AGENTS.md` files carry directory-specific rules.
3. **A machine-checkable definition of done.** A task is done when `python harness/check.py` passes. That runs:
   - the tests;
   - notebook validity, with no outputs;
   - a check that generated files are in sync with their sources;
   - skill-format lint;
   - a link check;
   - size and secret scans;
   - the immutability check.

   CI runs the same command, so every agent is held to the same bar.
4. **Guardrails are deterministic code, not prose.**
   - `protect_paths.py` blocks edits to immutable run artefacts, append-only audits and generated files. Claude Code runs it as a PreToolUse hook, and every tool is held to the same rules in CI.
   - Permission rules deny force-push, hard reset and `rm -rf`, and ask before commit, push or installing packages. These live in `.claude/settings.json` and `opencode.json`.
5. **State outlives the session.** `docs/plans/WORKLOG.md` is an append-only handoff log, and roadmap milestone IDs name the work. Any agent, in any tool, can pick up where another stopped.
6. **Domain rules come first.** The research-integrity rules in `AGENTS.md` are hard constraints: no fabricated data, no number without a source file, report failures plainly. They exist because earlier Colab chat runs broke exactly these (see `docs/audits/`).

## Commands

```bash
python harness/sync.py           # regenerate adapters after editing AGENTS.md / .agents/skills / mcp.json
python harness/sync.py --check   # drift check (used by harness/check.py)
python harness/sync.py --list    # list generated files
python harness/check.py          # full definition of done
python harness/check.py --fast --base origin/main   # quick pre-commit incl. immutability
```

## Tool support matrix

These paths were verified on 2026-09-29 against each tool's docs, except where the last column says otherwise.

| Tool | Instructions | Skills | Slash commands | Permissions / hooks | MCP (when `harness/mcp.json` is non-empty) |
|---|---|---|---|---|---|
| **Claude Code** | `CLAUDE.md` → `@AGENTS.md` (Claude Code doesn't read `AGENTS.md` natively) | `.claude/skills/` (generated mirror) | skills are `/name` | `.claude/settings.json`: allow/ask/deny + PreToolUse guard | `.mcp.json` |
| **Codex** (CLI/IDE/cloud) | `AGENTS.md`, nested, native | `.agents/skills/`, native (cwd → repo root) | `$skill-name` / skills UI | Codex sandbox/approval defaults; CI enforces the rest | `.codex/config.toml` block. *Unverified:* the docs only describe `~/.codex/config.toml`, so copy the block there if the project file is ignored. |
| **Antigravity** (IDE, `agy` CLI, 2.0) | `AGENTS.md` / `GEMINI.md` / `.agents/rules/*.md`, directory-scoped, native | `.agents/skills/`, native (`.agent/skills` is legacy) | `.agents/workflows/` (generated shims) | CI | user-level MCP config (not generated) |
| **OpenCode** | `AGENTS.md`, native | `.agents/skills/`, native (also reads `.claude/skills/`, `.opencode/skills/`) | `.opencode/commands/` (generated shims) | `opencode.json` `permission.bash`, where the **last matching rule wins** | `opencode.json` `mcp` (merged) |
| **Gemini CLI** | `AGENTS.md` via `.gemini/settings.json` `context.fileName` | `.agents/skills/`, native (takes precedence over `.gemini/skills/`) | skills | CI | `.gemini/settings.json` `mcpServers` (merged) |
| **Cursor** and others following the AGENTS.md standard | `AGENTS.md`, native | read `.agents/skills/` or the SKILL.md directly | — | CI | — |

**Duplicate skills in OpenCode.** OpenCode reads both `.agents/skills/` and the `.claude/skills/` mirror. The two copies are identical, and `sync.py --check` keeps them that way, so a duplicate listing is harmless. If it becomes a nuisance, check OpenCode's current docs for a setting that disables `.claude/` compatibility. Don't delete the mirror, because Claude Code needs it.

**Why no `GEMINI.md`?** Antigravity reads both `AGENTS.md` and `GEMINI.md`, so a generated `GEMINI.md` copy would load the same instructions twice. Gemini CLI is pointed at `AGENTS.md` through `.gemini/settings.json` instead.

## Extending

- **Add a skill.** Create `.agents/skills/<name>/SKILL.md`. The frontmatter needs `name` (equal to the folder name, lowercase-hyphenated, ≤ 64 chars) and `description` (≤ 1024 chars, saying *what* the skill does and *when* to use it). Then run `harness/sync.py` and add a row to the skills table in `AGENTS.md`.
- **Add an MCP server.** Add it to `harness/mcp.json` using env-var *names* for secrets, then run `harness/sync.py`.
- **Add a tool.** If it reads `AGENTS.md` and `.agents/skills/`, nothing is needed beyond a row in the matrix above. Otherwise add an entry to `manifest.json`: an `instruction_targets` entry with `import` or `copy` mode, or a `skills_targets` or `command_targets` entry. Then run sync and add a test to `tests/test_harness.py`.
- **Add a guardrail.** Put deterministic checks in `harness/check.py` (so every tool gets them via CI) and, where the tool supports it, in a pre-edit hook.

## Sources

- AGENTS.md standard and per-CLI context files: [inventivehq: CLAUDE.md vs AGENTS.md vs GEMINI.md](https://inventivehq.com/blog/claude-md-vs-agents-md-vs-gemini-md)
- Claude Code skills (locations, commands merged into skills): [code.claude.com/docs/en/skills](https://code.claude.com/docs/en/skills)
- Codex skills (`.agents/skills`, cwd → repo root): [developers.openai.com/codex/skills](https://developers.openai.com/codex/skills)
- Antigravity skills (`.agents/skills`, legacy `.agent/skills`): [antigravity.google/docs/skills](https://antigravity.google/docs/skills/)
- Antigravity rules and workflows (`AGENTS.md`/`GEMINI.md`/`.agents/rules`, `.agents/workflows`): [antigravity.google/docs/rules](https://antigravity.google/docs/rules)
- Antigravity CLI best practices: [antigravity.google/docs/cli/best-practices](https://antigravity.google/docs/cli/best-practices/)
- OpenCode skills, commands and permissions: [opencode.ai/docs/skills](https://opencode.ai/docs/skills/), [/commands](https://opencode.ai/docs/commands/), [/permissions](https://opencode.ai/docs/permissions/)
- Gemini CLI skills (`.agents/skills` alias): [geminicli.com/docs/cli/skills](https://geminicli.com/docs/cli/skills/)
