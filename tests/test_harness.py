"""Tests for the cross-tool agent harness (harness/sync.py, harness/hooks/protect_paths.py)."""

import fnmatch
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sync = _load("harness_sync", "harness/sync.py")
guard = _load("harness_guard", "harness/hooks/protect_paths.py")


def test_generated_files_are_in_sync():
    result = subprocess.run([sys.executable, "harness/sync.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


def test_every_agents_md_gets_a_claude_md_import():
    files, _ = sync.plan(json.loads((ROOT / "harness/manifest.json").read_text()))
    for src in json.loads((ROOT / "harness/manifest.json").read_text())["instruction_sources"]:
        folder = str(Path(src).parent)
        prefix = "" if folder == "." else folder + "/"
        assert f"{prefix}CLAUDE.md" in files
        assert files[f"{prefix}CLAUDE.md"].rstrip().splitlines()[2] == "@AGENTS.md"


def test_every_skill_is_mirrored_and_has_command_shims():
    files, _ = sync.plan(json.loads((ROOT / "harness/manifest.json").read_text()))
    skills = [p.parent.name for p in (ROOT / ".agents/skills").glob("*/SKILL.md")]
    assert skills
    for name in skills:
        assert f".claude/skills/{name}/SKILL.md" in files
        assert f".agents/workflows/{name}.md" in files
        assert f".opencode/commands/{name}.md" in files


def test_frontmatter_parser():
    meta = sync.frontmatter("---\nname: demo\ndescription: Does a thing.\n---\n\nbody")
    assert meta == {"name": "demo", "description": "Does a thing."}


def test_mcp_rendering_for_each_format(tmp_path):
    servers = {
        "local": {"command": "uvx", "args": ["some-server"], "env": {"TOKEN": "${TOKEN}"}},
        "remote": {"url": "https://example.com/mcp"},
    }
    claude = json.loads(sync.render_mcp("claude", servers, tmp_path / "a.json"))
    assert claude["mcpServers"]["local"]["command"] == "uvx"
    assert claude["mcpServers"]["remote"]["type"] == "http"
    opencode = json.loads(sync.render_mcp("opencode-merge", servers, tmp_path / "b.json"))
    assert opencode["mcp"]["local"]["command"] == ["uvx", "some-server"]
    toml = sync.render_mcp("codex-toml", servers, tmp_path / "c.toml")
    assert "[mcp_servers.local]" in toml and 'url = "https://example.com/mcp"' in toml
    # Re-rendering over an existing block replaces it rather than appending a second copy.
    (tmp_path / "c.toml").write_text('model = "x"\n\n' + toml)
    again = sync.render_mcp("codex-toml", servers, tmp_path / "c.toml")
    assert again.count("[mcp_servers.local]") == 1 and again.startswith('model = "x"')


def test_guard_blocks_immutable_and_generated_paths_only():
    run_readme = ROOT / "results/runs/droid100_adjoint_20260929T070629Z/README.md"
    assert guard.verdict(str(run_readme)) is not None
    assert guard.verdict(str(ROOT / "results/runs/some_new_run/README.md")) is None
    assert guard.verdict(str(ROOT / "CLAUDE.md")) is not None
    assert guard.verdict(str(ROOT / ".claude/skills/audit-run/SKILL.md")) is not None
    assert guard.verdict(str(ROOT / "AGENTS.md")) is None
    assert guard.verdict(str(ROOT / ".agents/skills/audit-run/SKILL.md")) is None
    assert guard.verdict(str(ROOT / ".agents/workflows/audit-run.md")) is not None
    assert guard.verdict(str(ROOT / "docs/audits/2026-09-28_run_v2_ailerons_audit.md")) is not None
    assert guard.verdict("/tmp/outside_repo.txt") is None


def _globs(config_name: str, tool_key: str, *actions: str) -> list[str]:
    config = json.loads((ROOT / config_name).read_text(encoding="utf-8"))
    return [pat for pat, action in config["permission"][tool_key].items() if action in actions]


def _effective(config_name: str, tool_key: str, target: str) -> str | None:
    """The action that actually applies, resolving patterns the way Kilo does: config order, last match wins."""
    rules = json.loads((ROOT / config_name).read_text(encoding="utf-8"))["permission"][tool_key]
    action = None
    for pattern, candidate in rules.items():
        if fnmatch.fnmatch(target, pattern):
            action = candidate
    return action


MUST_STAY_EDITABLE = (
    "AGENTS.md",
    "kilo.json",
    "opencode.json",
    "harness/sync.py",
    "harness/manifest.json",
    ".agents/skills/audit-run/SKILL.md",
    "src/adjointrwm/training.py",
    "results/runs/some_new_run/README.md",
    "results/benchmarks/some_new_benchmark/summary.json",
    "docs/audits/2026-10-03_some_new_audit.md",
)


def test_kilo_json_denies_every_path_the_harness_generates():
    # Kilo has no pre-edit hook, so the deny globs are its copy of protect_paths.py. The
    # expected set is derived from the manifest, so a new sync target cannot slip past.
    # Existence-dependent rules (imported runs, dated audits) are deliberately absent:
    # a glob cannot tell a new artefact from an immutable one.
    generated, _ = sync.plan(json.loads((ROOT / "harness/manifest.json").read_text()))
    assert generated
    for tool_key in ("edit", "write"):
        denied = _globs("kilo.json", tool_key, "deny")
        for rel_path in sorted(generated):
            assert any(fnmatch.fnmatch(rel_path, pat) for pat in denied), f"kilo.json {tool_key} allows {rel_path}"
        for rel_path in MUST_STAY_EDITABLE:
            assert not any(fnmatch.fnmatch(rel_path, pat) for pat in denied), f"kilo.json {tool_key} blocks {rel_path}"


def test_kilo_json_denies_the_same_paths_as_the_pre_edit_guard():
    # protect_paths.py is the reference; kilo.json must not be weaker on generated files.
    denied = _globs("kilo.json", "edit", "deny")
    for rel_path in ("CLAUDE.md", "notebooks/CLAUDE.md", "results/CLAUDE.md",
                     ".claude/skills/audit-run/SKILL.md", ".agents/workflows/audit-run.md",
                     ".opencode/commands/audit-run.md"):
        assert guard.verdict(str(ROOT / rel_path)) is not None
        assert any(fnmatch.fnmatch(rel_path, pat) for pat in denied), f"kilo.json allows {rel_path}"


def test_kilo_json_denies_credential_reads_at_any_depth():
    # Patterns are anchored at both ends, so a leading "*" is what makes them match nested paths.
    for rel_path in (".env", "apps/web/.env", "config/credentials.json",
                     "deploy/service_account.json", "keys/client_secret.json",
                     "home/.netrc", "gcloud/application_default_credentials.json"):
        assert _effective("kilo.json", "read", rel_path) == "deny", f"kilo.json allows reading {rel_path}"
    # .env.example is documentation, so the later allow rule has to win over "*.env.*".
    for rel_path in (".env.example", "docs/env.md", "src/adjointrwm/training.py"):
        assert _effective("kilo.json", "read", rel_path) == "allow", f"kilo.json blocks reading {rel_path}"


def test_permission_files_agree_on_bash_guardrails():
    # Kilo also loads ./opencode.json as a legacy config path, so the two blocks must agree.
    # Order is part of the contract: both tools resolve permission patterns last-match-wins,
    # so a reordered deny would degrade to ask while this assertion still passed.
    assert _globs("kilo.json", "bash", "ask", "deny") == _globs("opencode.json", "bash", "ask", "deny")


def test_bash_guardrails_are_not_bypassed_by_reordering_flags():
    # "git -C <repo> push --force" is the natural form inside an Agent Manager worktree,
    # so a prefix-anchored "git push --force*" deny would never match it.
    denied = _globs("opencode.json", "bash", "deny")
    for command in ("git push --force", "git -C /repo push --force", "git -C /repo push --force-with-lease origin main",
                    "git -C /repo push origin +main:main", "git -C /repo reset --hard", "rm -rf /", "rm -r -f /"):
        assert any(fnmatch.fnmatch(command, pat) for pat in denied), f"nothing denies: {command}"
    for command in ("git status", "git -C /repo log --oneline", "pytest -q"):
        assert not any(fnmatch.fnmatch(command, pat) for pat in denied), f"denies a safe command: {command}"
