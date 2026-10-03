"""Tests for the peer-critic rotation (scripts/pick_peer_critic.py).

The rotation is what keeps the Peer Critic Protocol in AGENTS.md from always
consulting the same agent; these tests pin the least-recently-used ordering, the
lead/excluded and installed-only filters, and the append-only log round-trip.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pick_peer_critic", ROOT / "scripts/pick_peer_critic.py")
ppc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ppc)


def _row(critic, date="2026-10-03", **extra):
    row = {"date": date, "milestone": "M", "artefact": "a.md", "lead": "claude", "critic": critic, "outcome": "accepted"}
    row.update(extra)
    return row


def test_first_pick_is_canonical_when_log_is_empty():
    assert ppc.choose([]) == "opencode"


def test_rotation_moves_past_a_reviewed_critic():
    assert ppc.choose([_row("opencode")]) == "codex"


def test_never_used_critics_come_before_previously_used_ones():
    rows = [_row("opencode", "2026-10-01"), _row("codex", "2026-09-30")]
    assert ppc.choose(rows) == "agy"  # never used, ahead of the two used ones


def test_earliest_reviewed_takes_the_next_turn_once_all_have_reviewed():
    rows = [_row(c) for c in ppc.CRITICS]  # one review each, in canonical order
    assert ppc.choose(rows) == "opencode"  # its review sits earliest in the log


def test_same_day_reviews_do_not_stall_the_rotation():
    # The log only has day granularity; rotation keys on row position, not date,
    # so two reviews on one day still alternate rather than sticking on one agent.
    critics = ["opencode", "codex"]
    day = "2026-10-03"
    rows = [_row("opencode", day)]
    assert ppc.choose(rows, critics=critics) == "codex"
    rows.append(_row("codex", day))
    assert ppc.choose(rows, critics=critics) == "opencode"
    rows.append(_row("opencode", day))
    assert ppc.choose(rows, critics=critics) == "codex"


def test_exclude_skips_the_lead():
    assert ppc.choose([], exclude=["opencode"]) == "codex"
    assert ppc.choose([], exclude=["codex"]) == "opencode"


def test_available_filter_respects_which():
    # `which` resolves a path and `probe` is handed that path, so the probe runs the
    # executable that was actually found rather than resolving PATH a second time.
    only_copilot = lambda bin_name: f"/usr/bin/{bin_name}" if bin_name == "copilot" else None
    yes = lambda path: True
    assert ppc.choose([], available=True, which=only_copilot, probe=yes) == "copilot"
    assert ppc.choose([], exclude=["copilot"], available=True, which=only_copilot, probe=yes) is None


def test_available_filter_rejects_a_cli_that_is_on_path_but_broken():
    # Measured 2026-10-03: `cline` is on PATH and `cline --version` exits 1 with
    # "Could not find the Cline CLI binary for your platform". A rotation that trusts
    # `which` alone hands the review to a CLI that cannot run it.
    launcher_on_path = lambda bin_name: f"/usr/bin/{bin_name}"
    assert ppc.choose([], available=True, which=launcher_on_path, probe=lambda p: False) is None
    # one broken candidate is skipped, the next available one is still returned
    picked = ppc.choose([], available=True, which=launcher_on_path, probe=lambda p: not p.endswith("/opencode"))
    assert picked == "codex"


def test_runs_reports_a_non_executable_cli_as_unavailable():
    assert ppc.runs("this-binary-does-not-exist") is False


def test_installed_distinguishes_missing_from_broken():
    # strings, not booleans: "broken" (present but will not run) must not collapse into "missing"
    on_path = lambda bin_name: None if bin_name == "copilot" else f"/usr/bin/{bin_name}"
    state = ppc.installed(which=on_path, probe=lambda p: not p.endswith("/cline"))
    assert state["copilot"] == "missing"
    assert state["cline"] == "broken"
    assert state["codex"] == "installed"


def test_exclude_accepts_any_spelling_of_an_agent():
    # The log records every lead as `antigravity` while the pool key is `agy`; without
    # normalisation `--exclude antigravity` would let Antigravity review its own work.
    assert ppc.canonical("antigravity") == "agy"
    assert ppc.canonical("AGY") == "agy"
    assert ppc.choose([], exclude=["antigravity"]) != "agy"
    assert ppc.choose([], exclude=["claude", "antigravity"], available=False) == "opencode"


def test_a_backfilled_alias_critic_still_counts_toward_the_rotation():
    rows = [_row("antigravity"), _row("opencode"), _row("codex")]
    assert ppc.last_index(rows)["agy"] == 0  # the alias resolves to the pool key
    assert ppc.choose(rows) not in {"agy", "opencode", "codex"}  # picks a never-reviewed agent


def test_path_entries_covers_version_manager_install_dirs(tmp_path, monkeypatch):
    # Measured 2026-10-03: `copilot` and `cline` live in ~/.nvm/versions/node/*/bin,
    # which a non-interactive shell that skipped ~/.bashrc never puts on PATH.
    home = tmp_path / "home"
    node_bin = home / ".nvm" / "versions" / "node" / "v24.21.0" / "bin"
    node_bin.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "")
    entries = ppc.path_entries()
    assert str(node_bin) in entries
    assert str(home / ".local" / "bin") not in entries  # only directories that exist


def test_which_runs_skips_a_broken_launcher_that_shadows_a_working_install(tmp_path, monkeypatch):
    shadow, real = tmp_path / "shadow", tmp_path / "real"
    shadow.mkdir()
    real.mkdir()
    for directory in (shadow, real):
        launcher = directory / "agentcli"
        launcher.write_text("#!/bin/sh\nexit 1\n")
        launcher.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", f"{shadow}{os.pathsep}{real}")
    working = lambda path: str(path).startswith(str(real))
    assert ppc.which_runs("agentcli", probe=working) == str(real / "agentcli")
    # when nothing runs the first candidate is returned, so callers report "broken", not "missing"
    assert ppc.which_runs("agentcli", probe=lambda p: False) == str(shadow / "agentcli")


def test_every_critic_has_a_skill_and_is_named_in_agents_md():
    # The rotation is only as wide as the pool: a critic with no skill, or one the
    # Peer Critic Protocol never mentions, silently never gets used.
    agents_md = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for critic, (_, skill) in ppc.CRITICS.items():
        assert (ROOT / ".agents" / "skills" / skill / "SKILL.md").exists(), f"{critic} has no {skill} skill"
        assert f"`{skill}`" in agents_md, f"AGENTS.md never mentions {skill}"


def test_the_pool_includes_kilo():
    assert "kilo" in ppc.CRITICS
    assert ppc.CRITICS["kilo"] == ("kilo", "kilo-cli")


def test_none_when_every_candidate_is_excluded():
    assert ppc.choose([], exclude=list(ppc.CRITICS)) is None


def test_missing_log_reads_as_empty(tmp_path):
    assert ppc.read_rows(tmp_path / "absent.csv") == []


def test_append_and_read_roundtrip(tmp_path):
    log = tmp_path / "log.csv"
    ppc.append_row(log, _row("codex", "2026-10-03"))
    rows = ppc.read_rows(log)
    expected = _row("codex", "2026-10-03")
    expected["source"] = ""
    assert rows == [expected]
    assert ppc.last_used(rows) == {"codex": "2026-10-03"}
    assert ppc.last_index(rows) == {"codex": 0}
    assert log.read_text().splitlines()[0] == ",".join(ppc.FIELDS)  # header written once


def test_last_used_keeps_the_latest_date_per_critic():
    rows = [_row("opencode", "2026-09-01"), _row("opencode", "2026-10-03")]
    assert ppc.last_used(rows) == {"opencode": "2026-10-03"}


def test_last_index_tracks_the_most_recent_row():
    rows = [_row("opencode"), _row("codex"), _row("opencode")]
    assert ppc.last_index(rows) == {"opencode": 2, "codex": 1}


def test_cli_prints_the_pick(tmp_path, capsys):
    log = tmp_path / "log.csv"
    ppc.append_row(log, _row("opencode"))
    assert ppc.main(["--log", str(log)]) == 0
    assert capsys.readouterr().out.strip() == "codex"


def test_cli_record_appends_a_row(tmp_path, capsys):
    log = tmp_path / "log.csv"
    code = ppc.main(["--log", str(log), "record", "--milestone", "M2", "--artefact", "docs/plans/x.md",
                     "--lead", "cline", "--critic", "codex", "--outcome", "accepted", "--date", "2026-10-03"])
    assert code == 0
    assert ppc.read_rows(log) == [
        {"date": "2026-10-03", "milestone": "M2", "artefact": "docs/plans/x.md", "lead": "cline",
         "critic": "codex", "outcome": "accepted", "source": "recorded"}
    ]
    assert "recorded codex" in capsys.readouterr().out


def test_cli_record_rejects_an_unknown_critic(tmp_path):
    code = ppc.main(["--log", str(tmp_path / "log.csv"), "record", "--milestone", "M",
                     "--artefact", "a", "--lead", "cline", "--critic", "nobody"])
    assert code == 2


def test_cli_record_rejects_a_bad_date(tmp_path):
    code = ppc.main(["--log", str(tmp_path / "log.csv"), "record", "--milestone", "M",
                     "--artefact", "a", "--lead", "cline", "--critic", "codex", "--date", "tomorrow"])
    assert code == 2


def test_an_empty_existing_log_behaves_like_a_fresh_one(tmp_path):
    log = tmp_path / "log.csv"
    log.write_text("")
    assert ppc.read_rows(log) == []
    code = ppc.main(["--log", str(log), "record", "--milestone", "M", "--artefact", "a",
                     "--lead", "cline", "--critic", "codex"])
    assert code == 0
    assert log.read_text().splitlines()[0] == ",".join(ppc.FIELDS)
    assert len(ppc.read_rows(log)) == 1


def test_a_malformed_log_is_refused_not_guessed(tmp_path, capsys):
    log = tmp_path / "log.csv"
    log.write_text("wrong,header\nfoo,bar\n")
    assert ppc.read_rows(log) == []
    assert "expected header" in capsys.readouterr().err


def test_committed_log_is_well_formed():
    rows = ppc.read_rows(ppc.DEFAULT_LOG)
    assert rows, "the rotation log should be seeded so the imbalance is corrected"
    for row in rows:
        assert set(row) == set(ppc.FIELDS)
        assert row["critic"] in ppc.CRITICS
