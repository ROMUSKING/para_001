"""Tests for the peer-critic rotation (scripts/pick_peer_critic.py).

The rotation is what keeps the Peer Critic Protocol in AGENTS.md from always
consulting the same agent; these tests pin the least-recently-used ordering, the
lead/excluded and installed-only filters, and the append-only log round-trip.
"""

from __future__ import annotations

import importlib.util
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
    only_copilot = lambda bin_name: bin_name == "copilot"
    assert ppc.choose([], available=True, which=only_copilot) == "copilot"
    assert ppc.choose([], exclude=["copilot"], available=True, which=only_copilot) is None


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
