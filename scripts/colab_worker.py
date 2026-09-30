#!/usr/bin/env python3
"""Run the Colab job worker (see docs/plans/colab-handoff.md §5 and adjointrwm.colab_jobs).

    python scripts/colab_worker.py --drive-root "/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production" \\
        --repo-dir /content/para_001 --repo-url https://github.com/ROMUSKING/para_001

The worker polls ``<drive-root>/jobs/inbox`` for job files, validates each against the allow-list committed on ``origin/main``
(``notebooks/05-ops/allowlist.json``), and runs allow-listed notebooks from a pinned commit in ``--repo-dir``. Run it from a checkout
that is **not** ``--repo-dir``: jobs check out other commits there, and the worker's own files must not change underneath it.
While a job runs it prints a progress line every ``--progress-minutes`` (default 2): elapsed time and the newest file the job wrote.
It exits by itself when the inbox has been empty for ``--max-idle-minutes`` (default 5). Stop it earlier with Colab's interrupt or by creating
``<drive-root>/jobs/STOP``. The worker notebook then flushes Drive and releases the Colab runtime.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adjointrwm import colab_jobs as cj  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drive-root", type=Path, default=Path(cj.DEFAULT_DRIVE_ROOT), help="folder that holds jobs/ and runs/")
    parser.add_argument("--repo-dir", type=Path, default=Path("/content/para_001"), help="checkout that jobs run in (its origin must be --repo-url)")
    parser.add_argument("--repo-url", default="https://github.com/ROMUSKING/para_001")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-idle-minutes", type=float, default=None,
                        help=f"exit after the inbox has been empty this long (default {cj.DEFAULT_IDLE_MINUTES:g}; 0 exits as soon as the inbox is empty)")
    parser.add_argument("--max-idle-hours", type=float, default=None, help="the same in hours (kept for old callers; --max-idle-minutes wins)")
    parser.add_argument("--progress-minutes", type=float, default=cj.DEFAULT_PROGRESS_SECONDS / 60,
                        help="while a job runs, print a progress line (elapsed time, newest file the job wrote) this often; 0 turns it off")
    parser.add_argument("--max-session-hours", type=float, default=10.0, help="do not start a job that would outlast this budget")
    parser.add_argument("--once", action="store_true", help="process the jobs that are waiting, then exit")
    parser.add_argument("--dry-run", action="store_true", help="validate waiting jobs and report what would run; run nothing")
    args = parser.parse_args()
    here = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except OSError:
        commit = None
    queue = cj.JobQueue(args.drive_root / "jobs")
    print(f"worker commit {commit} | jobs {queue.root} | runs {args.drive_root / 'runs'} | repo {args.repo_dir}", flush=True)
    results = cj.run_worker(queue, args.repo_dir, args.repo_url, args.drive_root / "runs", poll_seconds=args.poll_seconds,
                            max_idle_seconds=cj.idle_limit_seconds(args.max_idle_minutes, args.max_idle_hours), max_session_seconds=args.max_session_hours * 3600,
                            once=args.once, dry_run=args.dry_run, progress_seconds=args.progress_minutes * 60, worker_commit=commit,
                            log=lambda line: print(line, flush=True))
    print(f"{len(results)} job(s) handled", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
