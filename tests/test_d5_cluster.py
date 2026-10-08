"""D5 cluster admission tests: parsing, labels, policies, gates, tripwires."""

import numpy as np
import pytest

from adjointrwm.analysis.allocation import assert_predictions_ignore_futures
from adjointrwm.domains import cluster_admit as ca


def _events():
    # Two jobs: j1 (prio 9, completes), j2 (prio 1, fails). One zero-task job j3.
    jobs = [
        {"time": "0", "missing": "", "job_id": "1", "etype": "0", "user": "u",
         "sched_class": "2", "job_name": "a", "logical_name": "a"},
        {"time": "10", "missing": "", "job_id": "1", "etype": "4", "user": "u",
         "sched_class": "2", "job_name": "a", "logical_name": "a"},
        {"time": "5", "missing": "", "job_id": "2", "etype": "0", "user": "u",
         "sched_class": "0", "job_name": "b", "logical_name": "b"},
        {"time": "12", "missing": "", "job_id": "2", "etype": "3", "user": "u",
         "sched_class": "0", "job_name": "b", "logical_name": "b"},
        {"time": "7", "missing": "", "job_id": "3", "etype": "0", "user": "u",
         "sched_class": "1", "job_name": "c", "logical_name": "c"},
    ]
    tasks = [
        {"time": "0", "missing": "", "job_id": "1", "task_idx": "0", "machine": "m",
         "etype": "0", "user": "u", "sched_class": "2", "priority": "9",
         "cpu_req": "1", "mem_req": "1", "disk": "1", "constraint": ""},
        {"time": "5", "missing": "", "job_id": "2", "task_idx": "0", "machine": "m",
         "etype": "0", "user": "u", "sched_class": "0", "priority": "1",
         "cpu_req": "1", "mem_req": "1", "disk": "1", "constraint": ""},
        {"time": "5", "missing": "", "job_id": "2", "task_idx": "1", "machine": "m",
         "etype": "0", "user": "u", "sched_class": "0", "priority": "1",
         "cpu_req": "1", "mem_req": "1", "disk": "1", "constraint": ""},
    ]
    return {"jobs": jobs, "tasks": tasks}


def test_job_table_labels_and_drops():
    table = ca.build_job_table(_events())
    assert table["jobs"]["1"]["complete"] is True
    assert table["jobs"]["1"]["value"] == 9
    assert table["jobs"]["2"]["complete"] is False
    assert table["jobs"]["2"]["value"] == 0
    assert table["jobs"]["2"]["n_tasks"] == 2
    assert "3" not in table["jobs"]
    assert table["dropped"]["zero_task"] == 1


def test_missing_info_drops_job():
    events = _events()
    events["jobs"][0] = {**events["jobs"][0], "missing": "1"}
    table = ca.build_job_table(events)
    assert "1" not in table["jobs"]
    assert table["dropped"]["missing_info"] == 1


def test_oracle_beats_fifo_on_fixture():
    table = ca.build_job_table(_events())["jobs"]
    pool = ["1", "2"]
    assert ca.miss_rate(ca.select("oracle", pool, table, 1), table, pool) == 0.0
    # FIFO takes the earliest submit (job 1, completes); most_tasks takes job 2 (fails).
    assert ca.miss_rate(ca.select("fifo", pool, table, 1), table, pool) == 0.0
    assert ca.miss_rate(ca.select("most_tasks", pool, table, 1), table, pool) == 1.0
    assert ca.random_expected_miss(table, pool, 1) == pytest.approx(0.5)


def test_priority_diagnostic_flags_concentration():
    table = {str(i): {"priority": 0, "value": 1} for i in range(10)}
    table["x"] = {"priority": 9, "value": 1}
    diag = ca.priority_diagnostic(table)
    assert diag["degenerate"] is True
    varied = {str(i): {"priority": i % 12, "value": 1} for i in range(24)}
    assert ca.priority_diagnostic(varied)["degenerate"] is False


def test_deployable_selections_ignore_outcomes():
    """Boundary: shuffling P3 outcome fields must not move deployable selections."""
    table = ca.build_job_table(_events())["jobs"]
    pool = ["1", "2"]
    base = {"pool": pool, "table": table}
    for name in ca.POLICIES:
        sel = lambda trial, _n=name: ca.select(
            _n, trial["pool"], trial["table"], 1)
        assert_predictions_ignore_futures(
            sel, base,
            [{"flipped": True}, {"flipped": False, "complete": {"1": False, "2": True}}])
    _ = name



def test_tile_windows_drops_censored_before_splitting():
    """Amendment 2026-10-08: censoring first, then median split of survivors."""
    H = ca.HORIZON_S
    W = ca.WINDOW_S
    # Jobs submit in hour 0 and hour 1; trace ends early in hour 3.
    table = {
        "1": {"submit": 0, "priority": 9, "n_tasks": 1, "sched_class": "2",
              "complete": True, "value": 9},
        "2": {"submit": W, "priority": 9, "n_tasks": 1, "sched_class": "2",
              "complete": True, "value": 9},
    }
    trace_end = 27000000000  # hour-0 eligible, hour-1 censored (W=3.6e9, H=2.16e10)
    out = ca.tile_windows(table, 0, 4 * W, trace_end, tune_cap=10, val_cap=10)
    kept = [w["t0"] for w in out["tuning"] + out["validation"]]
    assert 0 in kept and W not in kept
    assert out["dropped"]["censored"] >= 1


def test_fast_tuples_match_dict_rows():
    """Fast tuple path and dict path build identical tables."""
    from adjointrwm.domains.cluster_admit import JOB_COLS, TASK_COLS, build_job_table

    events = _events()
    as_tuples = {
        "jobs": [tuple(r[c] for c in JOB_COLS) for r in events["jobs"]],
        "tasks": [tuple(r[c] for c in TASK_COLS) for r in events["tasks"]],
    }
    assert build_job_table(as_tuples) == build_job_table(events)
