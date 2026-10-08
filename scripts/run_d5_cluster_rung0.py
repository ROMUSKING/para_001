#!/usr/bin/env python3
"""scripts/run_d5_cluster_rung0.py — D5 cluster-admission Rung-0 gate (CPU-only).

Implements docs/plans/2026-10-08-d5-cluster-rung0-plan.md: G-1 priority
diagnostic, blocked temporal split, exact-k curves, G0/G1/G2 adjudication.
Test/usage tables never fetched; only the listed event parts are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "src"))

import numpy as np  # noqa: E402

from adjointrwm.domains import cluster_admit as ca  # noqa: E402


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="D5 cluster admission Rung-0")
    parser.add_argument("--parts-dir", required=True,
                        help="directory with job_events/task_events part-*.csv.gz")
    parser.add_argument("--n-job-parts", type=int, default=10)
    parser.add_argument("--n-task-parts", type=int, default=10)
    parser.add_argument("--part-offset", type=int, default=0,
                        help="first part index (slice blocks by time order)")
    parser.add_argument("--fast", action="store_true",
                        help="tuple rows instead of dicts (required beyond ~1M rows)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tune-cap", type=int, default=400)
    parser.add_argument("--val-cap", type=int, default=200)
    args = parser.parse_args(argv)

    parts = Path(args.parts_dir)
    job_all = sorted((parts / "job_events").glob("part-*.csv.gz"))
    task_all = sorted((parts / "task_events").glob("part-*.csv.gz"))
    job_parts = job_all[args.part_offset:args.part_offset + args.n_job_parts]
    task_parts = task_all[args.part_offset:args.part_offset + args.n_task_parts]
    if not job_parts or not task_parts:
        raise SystemExit("no event parts found under %s" % parts)
    events = ca.read_events([str(p) for p in job_parts] + [str(p) for p in task_parts],
                            fast=args.fast)
    print(f"rows: {len(events['jobs'])} job, {len(events['tasks'])} task", flush=True)

    built = ca.build_job_table(events)
    table = built["jobs"]
    print(f"jobs: {len(table)} kept, dropped={built['dropped']}", flush=True)

    # G-1 priority diagnostic on tuning-eligible jobs (before windowing).
    diag = ca.priority_diagnostic(table)
    print("G-1 priority: %s" % diag, flush=True)

    times = [j["submit"] for j in table.values()]
    t_min, t_max = min(times), max(times)
    trace_end = max(int(ca._col(r, "time", ca.JT)) for r in events["jobs"])
    windows = ca.tile_windows(table, t_min, t_max, trace_end, args.tune_cap, args.val_cap)
    print("windows: tune=%d val=%d dropped=%s t_split=%d" % (
        len(windows["tuning"]), len(windows["validation"]), windows["dropped"],
        windows["t_split"]), flush=True)
    # Review diagnostics: pool-size distribution per split + G-1 on the
    # tuning-eligible subset (jobs submitted before t_split), not the whole table.
    import numpy as _np

    def _pool_sizes(ws):
        return [len(w["pool"]) for w in ws]

    tune_sizes = _pool_sizes(windows["tuning"])
    val_sizes = _pool_sizes(windows["validation"])
    pool_stats = {
        split: {"n": len(v), "median": float(_np.median(v)) if v else 0.0,
                "min": int(min(v)) if v else 0, "max": int(max(v)) if v else 0}
        for split, v in (("tuning", tune_sizes), ("validation", val_sizes))}
    tune_table = {jid: j for jid, j in table.items()
                  if j["submit"] < windows["t_split"]}
    tune_diag = ca.priority_diagnostic(tune_table)
    print("pool sizes: %s" % pool_stats, flush=True)
    print("tuning-eligible G-1: %s" % tune_diag, flush=True)

    tune_curves = ca.window_curves(windows["tuning"], table)
    tune_areas = {p: ca.normalized_area(c) for p, c in tune_curves.items()}
    best_fixed = min(ca.FIXED_POLICIES, key=lambda p: tune_areas[p])
    val_curves = ca.window_curves(windows["validation"], table)
    val_areas = {p: ca.normalized_area(c) for p, c in val_curves.items()}
    g1 = ca.headroom(val_curves[best_fixed], val_curves["oracle"])
    dep = {p: val_areas[p] for p in ca.POLICIES if p not in ca.FIXED_POLICIES}
    best_dep = min(dep, key=lambda p: dep[p])
    denom = val_areas[best_fixed] - val_areas["oracle"]
    g2 = (val_areas[best_fixed] - val_areas[best_dep]) / denom if denom > 0 else float("nan")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "benchmark": "d5_cluster_rung0",
        "mode": "real",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "parts_dir": str(parts),
            "job_parts": [p.name for p in job_parts],
            "task_parts": [p.name for p in task_parts],
            "parts_sha256": {p.name: sha256_file(str(p)) for p in job_parts + task_parts},
        },
        "n_windows": {"tuning": len(windows["tuning"]),
                      "validation": len(windows["validation"])},
        "pool_sizes": pool_stats,
        "tuning_eligible_Gminus1": tune_diag,
        "dropped": {"jobs": built["dropped"], "windows": windows["dropped"]},
        "Gminus1_priority": diag,
        "Gminus1_degenerate": diag["degenerate"],
        "budgets": list(ca.BUDGETS),
        "tune_areas": tune_areas,
        "best_fixed_on_tuning": best_fixed,
        "val_areas": val_areas,
        "val_curves": val_curves,
        "G1_headroom": g1,
        "G1_pass_015": bool(g1 >= 0.15),
        "best_deployable": best_dep,
        "G2_share": g2,
        "G2_report_only": bool(g1 < 0.10),
        "G2_pass_050": bool(g1 >= 0.10 and g2 >= 0.50),
        "caveats": [
            "No usage tables: value is priority-weighted completion, not CPU-hours.",
            "Test/usage data never fetched; Rung-0 non-learned policies only.",
        ],
    }
    (output_dir / "d5_cluster_rung0_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"G-1 degenerate={diag['degenerate']} | best_fixed={best_fixed} "
          f"area={tune_areas[best_fixed]:.4f}")
    print(f"val: fixed={val_areas[best_fixed]:.4f} oracle={val_areas['oracle']:.4f} "
          f"dep={best_dep} ({dep[best_dep]:.4f})")
    print(f"G1={g1:.4f} pass={summary['G1_pass_015']} | "
          f"G2={g2:.4f} report_only={summary['G2_report_only']} pass={summary['G2_pass_050']}")
    print(f"Saved summary to {output_dir / 'd5_cluster_rung0_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
