"""Tests for the cross-tool agent harness (harness/sync.py, harness/hooks/protect_paths.py)."""

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
