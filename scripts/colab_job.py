#!/usr/bin/env python3
"""Write, check and list Colab jobs (see docs/plans/colab-handoff.md §5 and adjointrwm.colab_jobs).

    python scripts/colab_job.py allowlist
    python scripts/colab_job.py make --notebook notebooks/05-ops/ops_smoke.ipynb --commit <40-hex sha> [--set seeds=[0]] [--timeout-hours 0.25]
    python scripts/colab_job.py validate job.json

``make`` prints a validated job as JSON (the exact bytes to put in ``<drive>/jobs/inbox/<job_id>.json``) and its SHA-256 on stderr.
The allow-list is read from ``--allowlist`` (default: the working tree's ``notebooks/05-ops/allowlist.json``); the worker itself always uses
the one on ``origin/main``, so a job that is valid here can still be rejected there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adjointrwm import colab_jobs as cj  # noqa: E402


def load_allowlist(path: Path) -> dict:
    return cj.check_allowlist(json.loads(path.read_text()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--allowlist", type=Path, default=ROOT / cj.ALLOWLIST_RELPATH)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("allowlist", help="print the allowed notebooks, hardware, time limits and overrides")
    make = sub.add_parser("make", help="print a validated job")
    make.add_argument("--notebook", required=True)
    make.add_argument("--commit", required=True, help="full 40-character SHA")
    make.add_argument("--set", action="append", default=[], metavar="NAME=JSON", help="an override, e.g. seeds=[0] or resume_run_id=\"run_id\"")
    make.add_argument("--timeout-hours", type=float)
    make.add_argument("--job-id")
    make.add_argument("--note", default="")
    make.add_argument("--submitted-by", default="claude-code")
    check = sub.add_parser("validate", help="validate a job file")
    check.add_argument("file", type=Path)
    args = parser.parse_args()
    allow = load_allowlist(args.allowlist)
    if args.command == "allowlist":
        for path, entry in allow["notebooks"].items():
            names = ", ".join(name + " (" + spec["kind"] + ")" for name, spec in entry.get("overrides", {}).items()) or "none"
            print(f"{path}\n  gpu: {entry.get('gpu') or 'none required'} | default {entry.get('default_hours', 2)} h, max {entry['max_hours']} h | overrides: {names}")
        return 0
    if args.command == "make":
        overrides = {}
        for item in args.set:
            name, _, raw = item.partition("=")
            overrides[name] = json.loads(raw)
        try:
            spec = cj.make_job(args.notebook, args.commit, allow, overrides=overrides, timeout_hours=args.timeout_hours, job_id=args.job_id, note=args.note,
                               submitted_by=args.submitted_by)
        except cj.JobError as error:
            print(f"rejected: {error}", file=sys.stderr)
            return 1
        print(json.dumps(spec, indent=1, sort_keys=True))
        print(f"job_sha256 {cj.job_hash(spec)}", file=sys.stderr)
        return 0
    spec = json.loads(args.file.read_text())
    errors = cj.validate_job(spec, allow)
    print("ok" if not errors else "rejected: " + "; ".join(errors))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
