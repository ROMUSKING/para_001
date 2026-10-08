"""D5 cluster admission Rung-0: Borg job selection under a concurrency budget.

CPU-only, non-learned policies. Implements
docs/plans/2026-10-08-d5-cluster-rung0-plan.md (kilo-reviewed). Event-type codes
follow the v2.1 schema: 0 SUBMIT, 1 SCHEDULE, 2 EVICT, 3 FAIL, 4 FINISH, 5 KILL,
6 LOST, 7 UPDATE_PENDING, 8 UPDATE_RUNNING.
"""

from __future__ import annotations

import gzip
import math
from typing import Mapping, Sequence

import numpy as np

FINISH = 4
#: Trace timestamps are MICROSECONDS (v2.1 schema). All horizons/windows below
#: are in microseconds; misreading them as seconds silently empties every pool.
HORIZON_S = 21600 * 1_000_000
WINDOW_S = 3600 * 1_000_000
BUDGETS = (1, 2, 4, 8, 16, 32)

JOB_COLS = ("time", "missing", "job_id", "etype", "user", "sched_class",
            "job_name", "logical_name")
TASK_COLS = ("time", "missing", "job_id", "task_idx", "machine", "etype", "user",
             "sched_class", "priority", "cpu_req", "mem_req", "disk", "constraint")

# Positional indexes for the fast tuple path (same order as the *_COLS above).
JT, JM, JJ, JE, JS = 0, 1, 2, 3, 5
TT, TM, TJ, TK, TE, TP = 0, 1, 2, 3, 5, 8


def read_events(paths: Sequence[str], fast: bool = False) -> dict:
    """Parse v2.1 CSV parts (no headers) into job/task rows.

    ``fast=True`` returns rows as plain tuples (positional, see index constants)
    instead of dicts — required beyond ~1M rows. Consumers must use the
    :func:`_col` accessor, which accepts both forms.
    """
    jobs, tasks = [], []
    for path in paths:
        opener = gzip.open if path.endswith(".gz") else open
        is_task = "task_events" in path
        with opener(path, "rt") as handle:
            for line in handle:
                fields = line.rstrip("\n").split(",")
                if is_task:
                    if len(fields) < 13:
                        continue
                    tasks.append(tuple(fields) if fast else dict(zip(TASK_COLS, fields)))
                else:
                    if len(fields) < 8:
                        continue
                    jobs.append(tuple(fields) if fast else dict(zip(JOB_COLS, fields)))
    return {"jobs": jobs, "tasks": tasks, "fast": fast}


def _col(row, name: str, idx: int):
    """Column accessor accepting dict rows (tests/fixtures) or fast tuples."""
    if isinstance(row, dict):
        return row[name]
    return row[idx]


def build_job_table(events: dict) -> dict:
    """One record per submitted job: submit time, priority, task count, outcome.

    Drops (with counts): jobs with any missing-info event; jobs with no task
    rows (priority undefined). COMPLETE = last job event in [submit, submit+H]
    is FINISH. Priority = max task-SUBMIT priority; task count = distinct task
    indices ever seen for the job.
    """
    by_job: dict = {}
    for row in events["jobs"]:
        by_job.setdefault(_col(row, "job_id", JJ), []).append(row)
    tasks_by_job: dict = {}
    for row in events["tasks"]:
        tasks_by_job.setdefault(_col(row, "job_id", TJ), []).append(row)
    task_prio: dict = {}
    task_seen: dict = {}
    for jid, trows in tasks_by_job.items():
        seen = set()
        for row in trows:
            seen.add(_col(row, "task_idx", TK))
            if _col(row, "etype", TE) == "0":
                key = (jid, _col(row, "task_idx", TK))
                raw_prio = _col(row, "priority", TP)
                p = int(raw_prio) if raw_prio not in ("", None) else None
                if p is not None:
                    prev = task_prio.get(key)
                    task_prio[key] = p if prev is None else max(prev, p)
        task_seen[jid] = seen
    table, dropped = {}, {"missing_info": 0, "zero_task": 0}
    for jid, rows in by_job.items():
        submits = [int(_col(r, "time", JT)) for r in rows if _col(r, "etype", JE) == "0"]
        if not submits:
            continue
        submit = min(submits)
        if any(_col(r, "missing", JM) for r in rows):
            dropped["missing_info"] += 1
            continue
        seen = task_seen.get(jid, set())
        if not seen:
            dropped["zero_task"] += 1
            continue
        prios = [task_prio[k] for k in {(jid, t) for t in seen} if k in task_prio]
        if not prios:
            dropped["zero_task"] += 1
            continue
        pairs = sorted((int(_col(r, "time", JT)), int(_col(r, "etype", JE))) for r in rows
                       if submit <= int(_col(r, "time", JT)) <= submit + HORIZON_S)
        complete = bool(pairs) and pairs[-1][1] == FINISH
        table[jid] = {
            "submit": submit,
            "priority": max(prios),
            "n_tasks": len(seen),
            "sched_class": next((_col(r, "sched_class", JS) for r in rows
                                 if _col(r, "etype", JE) == "0"), ""),
            "complete": complete,
            "value": max(prios) if complete else 0,
        }
    return {"jobs": table, "dropped": dropped}


def tile_windows(table: dict, t_min: int, t_max: int, trace_end: int,
                 tune_cap: int = 400, val_cap: int = 200) -> dict:
    """Tile [t_min, t_max] into hourly windows; pool = jobs submitted in window.

    Drops: windows with zero completable pool value (empty denominator);
    windows with trace_end − window_end < HORIZON_S (censoring); the two windows
    straddling the midpoint timestamp (gap). Split: t0 < Tmid → tuning.
    Caps: first-N per split (frozen counts).
    """
    t_mid = (t_min + t_max) // 2
    starts = list(range(t_min, t_max, WINDOW_S))
    # Censoring first: drop windows too close to the trace end, then split the
    # survivors at their median start (amendment 2026-10-08: splitting the full
    # span left the validation block entirely inside the censored zone on short
    # slices). Forward time direction preserved (tuning earlier, validation later).
    eligible = [s for s in starts if trace_end - (s + WINDOW_S) >= HORIZON_S]
    if not eligible:
        return {"tuning": [], "validation": [], "dropped": {"empty_pool": 0,
                "censored": len(starts), "gap": 0}, "t_mid": t_mid, "trace_end": trace_end}
    mid_pos = len(eligible) // 2
    t_split = eligible[mid_pos]
    # Gap separates the blocks; with fewer than 4 eligible windows the gap would
    # eat the sample, so it applies only when there is enough span to spare.
    gap = {eligible[mid_pos - 1], eligible[mid_pos]} if len(eligible) >= 4 else set()
    windows, dropped = [], {"empty_pool": 0, "censored": len(starts) - len(eligible),
                            "gap": 0}
    for t0 in eligible:
        t1 = t0 + WINDOW_S
        if t0 in gap:
            dropped["gap"] += 1
            continue
        pool = [jid for jid, j in table.items() if t0 <= j["submit"] < t1]
        if sum(table[j]["value"] for j in pool) <= 0:
            dropped["empty_pool"] += 1
            continue
        windows.append({"t0": t0, "pool": pool,
                        "split": "tuning" if t0 < t_split else "validation"})
    tune = [w for w in windows if w["split"] == "tuning"][:tune_cap]
    val = [w for w in windows if w["split"] == "validation"][:val_cap]
    return {"tuning": tune, "validation": val, "dropped": dropped,
            "t_split": t_split, "trace_end": trace_end}


def _ranked(pool: list, table: dict, key) -> list:
    return sorted(pool, key=lambda j: (key(table[j]), int(j)))


def select(policy: str, pool: list, table: dict, k: int) -> list:
    """Exact-k admission; ties toward the lower job id (frozen)."""
    if policy == "fifo":
        order = _ranked(pool, table, lambda r: r["submit"])
    elif policy == "highest_priority":
        order = _ranked(pool, table, lambda r: -r["priority"])
    elif policy == "most_tasks":
        order = _ranked(pool, table, lambda r: -r["n_tasks"])
    elif policy == "priority_x_tasks":
        order = _ranked(pool, table, lambda r: -(r["priority"] * r["n_tasks"]))
    elif policy == "highprio_fifo":
        order = _ranked(pool, table, lambda r: (0 if r["priority"] >= 9 else 1,
                                                r["submit"]))
    elif policy == "oracle":
        order = _ranked(pool, table, lambda r: -r["value"])
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return order[:k]


def miss_rate(selected: Sequence, table: dict, pool: list) -> float:
    """Priority-weighted miss rate of an admitted set (lower is better)."""
    total = sum(table[j]["value"] for j in pool)
    got = sum(table[j]["value"] for j in selected)
    return 1.0 - got / total


def random_expected_miss(table: dict, pool: list, k: int) -> float:
    """Exact expected miss of uniform random admission (no simulation)."""
    n = len(pool)
    if n == 0:
        return float("nan")
    return 1.0 - min(k, n) / n


POLICIES = ("fifo", "highest_priority", "most_tasks", "priority_x_tasks",
            "highprio_fifo")
FIXED_POLICIES = ("fifo", "highest_priority", "most_tasks")


def window_curves(windows: Sequence[Mapping], table: dict,
                  budgets: Sequence[int] = BUDGETS) -> dict:
    """Mean miss rate per policy per budget + exact oracle curve."""
    curves = {}
    for name in (*POLICIES, "oracle"):
        means = []
        for k in budgets:
            vals = [miss_rate(select(name, w["pool"], table, k), table, w["pool"])
                    for w in windows]
            means.append(float(np.mean(vals)))
        curves[name] = means
    curves["random_expected"] = [
        float(np.mean([random_expected_miss(table, w["pool"], k) for w in windows]))
        for k in budgets]
    return curves


def normalized_area(curve: Sequence[float], budgets: Sequence[int] = BUDGETS) -> float:
    """Trapezoidal area of J(k) over k/kmax (J already normalised)."""
    xs = np.asarray(budgets, dtype=float) / max(budgets)
    return float(np.trapezoid(np.asarray(curve, dtype=float), xs))


def headroom(fixed_curve: Sequence[float], ref_curve: Sequence[float],
             budgets: Sequence[int] = BUDGETS) -> float:
    """Relative headroom area(fixed − ref)/area(fixed)."""
    denom = normalized_area(fixed_curve, budgets)
    if denom <= 0:
        return float("nan")
    return (denom - normalized_area(ref_curve, budgets)) / denom


def priority_diagnostic(table: dict) -> dict:
    """G−1: entropy + top-1 mass of priority over completable jobs."""
    prios = [j["priority"] for j in table.values() if j["value"] > 0]
    if not prios:
        return {"entropy_bits": 0.0, "top1_mass": 0.0, "n": 0, "degenerate": True}
    values, counts = np.unique(prios, return_counts=True)
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    top1 = float(probs.max())
    return {"entropy_bits": entropy, "top1_mass": top1, "n": len(prios),
            "degenerate": bool(top1 > 0.70 or entropy < 1.5)}
