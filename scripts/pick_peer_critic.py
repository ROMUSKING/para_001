#!/usr/bin/env python3
"""Pick the next peer (decision) critic by least-recently-used rotation.

`AGENTS.md` requires an adversarial peer critic for major planning and design
decisions, but the reviewer must **rotate** across the available coding agents so
that no single agent always reviews. This tool is that rotation: it reads the
append-only log at `docs/plans/peer-critic-log.csv` and returns the critic that
has gone longest without reviewing - never-used agents first, then the oldest
last review - excluding the lead agent and, optionally, agents whose CLI is not
installed on this host.

Standard library only, so it runs in any agent sandbox.

Usage:
    python scripts/pick_peer_critic.py                      # print the next critic
    python scripts/pick_peer_critic.py --exclude cline      # never pick the lead
    python scripts/pick_peer_critic.py --available          # only installed CLIs
    python scripts/pick_peer_critic.py --list               # show rotation state
    python scripts/pick_peer_critic.py record --milestone B2.3 \\
        --artefact docs/plans/x.md --lead claude --critic codex --outcome accepted

Exit code is 0 with the chosen critic id on stdout, or 1 (nothing available) or 2
(bad arguments) with a message on stderr.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "docs" / "plans" / "peer-critic-log.csv"
# `source` is provenance metadata only; it never influences the rotation.
FIELDS = ["date", "milestone", "artefact", "lead", "critic", "outcome", "source"]
REQUIRED = ["date", "milestone", "artefact", "lead", "critic"]
PROBE_TIMEOUT = 10  # seconds; every CLI installed here answers --version in under a second

# Canonical order (used to break ties) -> (CLI executable, skill that documents it).
# Add an agent here and to the Skills table in AGENTS.md to widen the rotation.
CRITICS: dict[str, tuple[str, str]] = {
    "opencode": ("opencode", "opencode-delegate"),
    "codex": ("codex", "codex-cli"),
    "agy": ("agy", "agy-cli"),
    "copilot": ("copilot", "copilot-cli"),
    "cline": ("cline", "cline-cli"),
    "kilo": ("kilo", "kilo-cli"),
}

# Agents are named inconsistently across the repository and the log: AGENTS.md says
# "Antigravity (`agy`)" and every recorded lead is `antigravity`, while the pool key is
# `agy`. Without normalisation `--exclude antigravity` hands Antigravity its own review,
# which the protocol forbids. Map every spelling to the pool key, for both sides.
ALIASES: dict[str, str] = {
    "antigravity": "agy",
    "gemini": "copilot",  # only if a host ever aliases them; harmless otherwise
}


def canonical(agent: str) -> str:
    agent = (agent or "").strip().lower()
    return ALIASES.get(agent, agent)


def _fail(message: str) -> None:
    print(f"pick_peer_critic: {message}", file=sys.stderr)


def read_rows(path: Path) -> list[dict]:
    """Return the non-empty rows of the rotation log ([] if it does not exist)."""
    if not path.exists():
        return []
    if path.stat().st_size == 0:
        return []  # an empty file is a fresh log, not a malformed one
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != FIELDS:
                _fail(f"{path}: expected header {','.join(FIELDS)}, found "
                      f"{','.join(reader.fieldnames or [])}; refusing to guess")
                return []
            rows = []
            for lineno, row in enumerate(reader, start=2):
                if not any((v or "").strip() for v in row.values()):
                    continue
                clean = {k: (row.get(k) or "").strip() for k in FIELDS}
                clean["critic"] = canonical(clean["critic"])  # so `antigravity` counts as `agy`
                rows.append(clean)
            return rows
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        _fail(f"{path}: cannot parse rotation log ({exc}); refusing to guess")
        return []


def last_index(rows: list[dict]) -> dict[str, int]:
    """Map critic id -> the position of its most recent review in the log.

    Position, not date, drives the rotation: the log has day granularity, so
    reviews recorded on the same day share a date and a date-keyed rotation could
    keep selecting the same agent. Row order is strictly increasing, so it always
    advances. Ids are canonicalised, so an `antigravity` row counts as `agy`.
    """
    seen: dict[str, int] = {}
    for i, row in enumerate(rows):
        critic = canonical(row.get("critic") or "")
        if critic:
            seen[critic] = i
    return seen


def last_used(rows: list[dict]) -> dict[str, str]:
    """Map critic id -> the latest date it reviewed (display only)."""
    seen: dict[str, str] = {}
    for row in rows:
        critic = canonical(row.get("critic") or "")
        if critic:
            seen[critic] = max(seen.get(critic, ""), (row.get("date") or "").strip())
    return seen


def path_entries() -> list[str]:
    """PATH for availability checks: the caller's PATH plus common version-manager dirs.

    Agent CLIs are often installed somewhere the calling process never sourced - the
    nvm-managed node CLIs (`copilot`, `cline`) on this host live in
    `~/.nvm/versions/node/*/bin`, and a non-interactive shell that skipped `~/.bashrc`
    cannot see them. Reporting those agents as missing would silently narrow the
    rotation, so the usual install locations are searched too. Existing directories only.
    """
    entries = [e for e in os.environ.get("PATH", "").split(os.pathsep) if e]
    seen = set(entries)
    extras = [str(Path.home() / ".local" / "bin")]
    nvm = Path.home() / ".nvm" / "versions" / "node"
    if nvm.is_dir():
        extras += [str(p / "bin") for p in sorted(nvm.iterdir()) if p.is_dir()]
    for extra in extras:
        if extra not in seen and Path(extra).is_dir():
            seen.add(extra)
            entries.append(extra)
    return entries


def runs(binary: str, timeout: int = PROBE_TIMEOUT) -> bool:
    """True when the CLI actually executes.

    `shutil.which` alone is not availability: a launcher can sit on PATH while the
    real binary is missing (measured 2026-10-03: `cline --version` exits 1 with
    "Could not find the Cline CLI binary for your platform"). Dispatching a review
    to such a CLI wastes the lead's turn, so `--available` probes instead.

    `stdin` is closed so a CLI that asks a first-run question cannot block until the
    timeout, and `binary` is a resolved path, so the probe runs exactly the executable
    that was found.
    """
    try:
        done = subprocess.run(
            [binary, "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def which_runs(binary: str, probe=runs) -> str | None:
    """First executable named `binary` on the augmented PATH that actually runs.

    A launcher earlier in PATH can shadow a working install (measured 2026-10-03: the
    Windows npm shim `cline` came before the working nvm `cline` and exited 1), so
    candidates are probed in order and the first one that runs wins. When none run, the
    first candidate is returned so callers can report `broken` rather than `missing`.
    """
    first = None
    for entry in path_entries():
        candidate = os.path.join(entry, binary)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            if first is None:
                first = candidate
            if probe(candidate):
                return candidate
    return first


def choose(rows, exclude=(), available=False, which=which_runs, critics=CRITICS, probe=runs):
    """Return the next critic id, or None if every candidate is excluded/absent.

    Order: never-reviewed critics first (canonical order), then the critic whose
    most recent review sits earliest in the log, so the pairing always rotates.
    `available=True` drops critics whose CLI is absent or does not run. `exclude`
    accepts any spelling of an agent (see ALIASES).
    """
    order = {c: i for i, c in enumerate(critics)}
    seen = last_index(rows)
    excluded = {canonical(e) for e in exclude}
    candidates = [c for c in critics if c not in excluded]
    if available:
        candidates = [c for c in candidates
                      if (found := which(critics[c][0])) and probe(found)]
    if not candidates:
        return None
    # ``-1`` for never-reviewed sorts before any index; then the earliest index wins.
    return min(candidates, key=lambda c: (seen.get(c, -1), order[c]))


def installed(which=which_runs, critics=CRITICS, probe=runs) -> dict[str, str]:
    """Map critic id -> "installed" | "broken" | "missing".

    Returns strings, not booleans: "broken" (on PATH but will not run) is the case that
    matters most and must not collapse into "missing".
    """
    state = {}
    for critic in critics:
        found = which(critics[critic][0])
        state[critic] = "missing" if not found else ("installed" if probe(found) else "broken")
    return state


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            if path.stat().st_size == 0:  # new file, or an empty placeholder
                csv.DictWriter(fh, fieldnames=FIELDS).writeheader()
            csv.DictWriter(fh, fieldnames=FIELDS).writerow({k: row.get(k, "") for k in FIELDS})
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _display(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="rotation log path")
    parser.add_argument("--exclude", action="append", default=[], metavar="AGENT",
                        help="agent that must not be picked (e.g. the lead); repeatable")
    parser.add_argument("--available", action="store_true", help="only pick agents whose CLI is installed")
    parser.add_argument("--list", action="store_true", help="show the rotation state and exit")
    sub = parser.add_subparsers(dest="cmd")
    record = sub.add_parser("record", help="append a completed review to the rotation log")
    record.add_argument("--milestone", required=True)
    record.add_argument("--artefact", required=True, help="plan/spec/note path that was reviewed")
    record.add_argument("--lead", required=True, help="agent that owned the work")
    record.add_argument("--critic", required=True, help=f"one of: {', '.join(CRITICS)}")
    record.add_argument("--outcome", default="", help="e.g. accepted / partially adopted")
    record.add_argument("--date", default="", help="ISO date (default: today)")
    record.add_argument("--source", default="recorded",
                        help="provenance: 'recorded' for a review run now, 'reconstructed' for a backfill from a committed artefact")

    args = parser.parse_args(argv)
    rows = read_rows(args.log)

    if args.cmd == "record":
        if args.critic not in CRITICS:
            print(f"unknown critic {args.critic!r}; known critics: {', '.join(CRITICS)}", file=sys.stderr)
            return 2
        try:
            when = args.date or dt.date.today().isoformat()
            dt.date.fromisoformat(when)
        except ValueError:
            print(f"invalid --date {args.date!r}; expected ISO YYYY-MM-DD", file=sys.stderr)
            return 2
        row = {
            "date": when,
            "milestone": args.milestone.strip(),
            "artefact": args.artefact.strip(),
            "lead": canonical(args.lead),
            "critic": canonical(args.critic),
            "outcome": (args.outcome or "").strip(),
            "source": (args.source or "").strip(),
        }
        missing = [key for key in REQUIRED if not row[key]]
        if missing:
            print(f"missing required fields: {', '.join(missing)}", file=sys.stderr)
            return 2
        append_row(args.log, row)
        print(f"recorded {row['critic']} ({row['date']}) for {row['milestone']} -> {_display(args.log)}")
        return 0

    if args.list:
        seen = last_used(rows)
        present = installed()
        nxt = choose(rows, args.exclude, args.available)
        print(f"log: {_display(args.log)} ({len(rows)} recorded reviews)")
        print(f"pool: exactly these {len(CRITICS)} agents ({', '.join(CRITICS)})")
        print("available means an executable named after the agent was found on PATH (plus "
              "version-manager dirs) and its --version exits 0; nothing about credentials or config")
        for critic in CRITICS:
            mark = "*" if critic == nxt else " "
            print(f"{mark} {critic:<8} {present[critic]:<10} last={seen.get(critic) or 'never':<10}  "
                  f"skill={CRITICS[critic][1]}")
        print("rotation applies to recorded reviews; pick does not reserve, and an unrecorded review changes nothing")
        unfiltered = choose(rows, args.exclude)
        if nxt != unfiltered:
            print(f"note: rotation would pick {unfiltered}, but it is not available on this host; "
                  f"--available picks {nxt}")
        return 0

    pick = choose(rows, args.exclude, args.available)
    if not pick:
        print("no available peer critic (all excluded or no CLI installed)", file=sys.stderr)
        return 1
    print(pick)
    return 0


if __name__ == "__main__":
    sys.exit(main())

