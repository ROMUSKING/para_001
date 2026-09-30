"""Tests for the Colab job queue (adjointrwm.colab_jobs): validation, notebook preparation, the queue and the worker loop.

The end-to-end test runs a tiny notebook through a real git checkout and nbclient; everything else uses fakes. No Colab is needed.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_notebook

from adjointrwm import colab_jobs as cj

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
TINY = "notebooks/05-ops/tiny.ipynb"
GPU_NB = "notebooks/05-ops/gpu.ipynb"

ALLOW = {
    "version": 1, "max_timeout_hours": 4,
    "notebooks": {
        TINY: {"gpu": [], "default_hours": 0.1, "max_hours": 0.2, "overrides": {
            "seeds": {"kind": "int_list", "prefix": "seeds: tuple = "},
            "arms": {"kind": "choice_list", "prefix": "arms: tuple = ", "choices": ["a", "b", "c"]},
            "resume_run_id": {"kind": "run_id", "prefix": "RESUME_RUN_ID = "},
            "max_steps": {"kind": "int_or_null", "prefix": "MAX_STEPS = "}}},
        GPU_NB: {"gpu": ["L4", "A100"], "default_hours": 1, "max_hours": 3, "overrides": {}},
    },
}


def job(**changes):
    spec = {"schema": 1, "job_id": "probe-001", "notebook": TINY, "commit": SHA}
    spec.update(changes)
    return spec


# ---- allow-list and validation -----------------------------------------------------------------------------------------

def test_the_committed_allowlist_is_well_formed_and_every_override_prefix_matches_one_line_in_its_notebook():
    allow = cj.check_allowlist(json.loads((ROOT / cj.ALLOWLIST_RELPATH).read_text()))
    for path, entry in allow["notebooks"].items():
        assert (ROOT / path).is_file(), path
        if not entry.get("overrides"):
            continue
        nb = nbformat.read(ROOT / path, 4)
        samples = {"int_list": [0], "run_id": "abc_123", "int_or_null": 5, "choice_list": [entry["overrides"][k].get("choices", ["x"])[0] for k in entry["overrides"]][:1]}
        cj.apply_overrides(nb, {k: (samples["choice_list"] if v["kind"] == "choice_list" else samples[v["kind"]]) for k, v in entry["overrides"].items()}, entry)
        assert cj.pin_repo_ref(nb, SHA) >= 1


@pytest.mark.parametrize("bad", [{}, {"version": 2, "notebooks": {TINY: {}}}, {"version": 1, "notebooks": {}},
                                 {"version": 1, "notebooks": {"../x.ipynb": {"max_hours": 1}}},
                                 {"version": 1, "notebooks": {TINY: {"max_hours": 99}}},
                                 {"version": 1, "notebooks": {TINY: {"max_hours": 1, "overrides": {"x": {"kind": "eval", "prefix": "x = "}}}}},
                                 {"version": 1, "notebooks": {TINY: {"max_hours": 1, "overrides": {"x": {"kind": "choice_list", "prefix": "x = "}}}}}])
def test_malformed_allowlists_are_rejected(bad):
    with pytest.raises(cj.JobError):
        cj.check_allowlist(bad)


def test_a_good_job_validates_with_and_without_optional_fields():
    assert cj.validate_job(job(), ALLOW) == []
    full = job(overrides={"seeds": [0, 1], "arms": ["a", "c"], "resume_run_id": "run_2026-01", "max_steps": None}, timeout_hours=0.2, note="x" * 500,
               submitted_by="me", submitted_utc="2026-09-30T00:00:00+00:00")
    assert cj.validate_job(full, ALLOW) == []


@pytest.mark.parametrize("spec, fragment", [
    ("not a dict", "JSON object"),
    (job(extra=1), "unknown keys"),
    ({"schema": 1, "job_id": "abc"}, "missing keys"),
    (job(schema=2), "schema"),
    (job(job_id="../../x"), "job_id"),
    (job(job_id="ab"), "job_id"),
    (job(commit="abc123"), "40-character"),
    (job(commit="A" * 40), "40-character"),
    (job(notebook="notebooks/other.ipynb"), "not in the allow-list"),
    (job(notebook="notebooks/05-ops/../../evil.ipynb"), "not in the allow-list"),
    (job(notebook="/etc/passwd"), "not in the allow-list"),
    (job(overrides={"unknown": 1}), "not allowed"),
    (job(overrides=[1]), "overrides must be an object"),
    (job(overrides={"seeds": []}), "distinct integers"),
    (job(overrides={"seeds": [1, 1]}), "distinct integers"),
    (job(overrides={"seeds": [True]}), "distinct integers"),
    (job(overrides={"seeds": "(0,)"}), "distinct integers"),
    (job(overrides={"arms": ["z"]}), "distinct values"),
    (job(overrides={"arms": ["a", "a"]}), "distinct values"),
    (job(overrides={"resume_run_id": "a; rm -rf /"}), "run id"),
    (job(overrides={"resume_run_id": "x'"}), "run id"),
    (job(overrides={"max_steps": 0}), "null or an integer"),
    (job(overrides={"max_steps": "5"}), "null or an integer"),
    (job(timeout_hours=0), "timeout_hours"),
    (job(timeout_hours=5), "timeout_hours"),
    (job(timeout_hours="1"), "timeout_hours"),
    (job(note="x" * 501), "note"),
    (job(submitted_by=3), "submitted_by"),
])
def test_bad_jobs_are_rejected_with_a_reason(spec, fragment):
    errors = cj.validate_job(spec, ALLOW)
    assert errors and any(fragment in e for e in errors), errors
    with pytest.raises(cj.JobError):
        cj.check_job(spec, ALLOW)


def test_job_hash_is_canonical_and_content_sensitive():
    a = job(overrides={"seeds": [0]})
    assert cj.job_hash(a) == cj.job_hash(dict(reversed(list(a.items())))) and len(cj.job_hash(a)) == 64
    assert cj.job_hash(a) != cj.job_hash(job(overrides={"seeds": [1]})) and cj.job_hash(a) != cj.job_hash(job(commit="b" * 40))


def test_make_job_builds_a_validated_spec_with_a_default_id():
    spec = cj.make_job(TINY, SHA, ALLOW, overrides={"seeds": [0]}, timeout_hours=0.1, note="probe", now=lambda: "2026-09-30T06:29:00+00:00")
    assert spec["job_id"] == "tiny-20260930T062900Z" and spec["overrides"] == {"seeds": [0]} and spec["note"] == "probe" and cj.validate_job(spec, ALLOW) == []
    with pytest.raises(cj.JobError):
        cj.make_job(TINY, "short", ALLOW)
    with pytest.raises(cj.JobError):
        cj.make_job("notebooks/nope.ipynb", SHA, ALLOW)
    assert cj.make_job(TINY, SHA, ALLOW, job_id="my-job")["job_id"] == "my-job"


# ---- preparing the executed copy ----------------------------------------------------------------------------------------

def tiny_nb():
    return new_notebook(cells=[
        new_code_cell("REPO_REF = 'main'  # branch, tag or commit SHA\nprint('setup')"),
        new_code_cell("class C:\n    seeds: tuple = (0, 1, 2, 3, 4)\n    arms: tuple = ('a', 'b')\nRESUME_RUN_ID = None  # e.g. 'x'\nMAX_STEPS = None  # steps\nprint(C.seeds)")])


def test_render_override_covers_every_kind():
    assert cj.render_override("int_list", [0]) == "(0,)" and cj.render_override("int_list", [0, 2]) == "(0, 2)"
    assert cj.render_override("choice_list", ["a"]) == "('a',)" and cj.render_override("choice_list", ["a", "b"]) == "('a', 'b')"
    assert cj.render_override("run_id", "r_1") == "'r_1'" and cj.render_override("int_or_null", None) == "None" and cj.render_override("int_or_null", 7) == "7"


def test_apply_overrides_replaces_exactly_the_named_lines_and_keeps_indentation():
    nb = tiny_nb()
    cj.apply_overrides(nb, {"seeds": [3], "arms": ["c", "a"], "resume_run_id": "run_9", "max_steps": 100}, ALLOW["notebooks"][TINY])
    lines = nb.cells[1].source.split("\n")
    assert lines[1] == "    seeds: tuple = (3,)" and lines[2] == "    arms: tuple = ('c', 'a')"
    assert lines[3] == "RESUME_RUN_ID = 'run_9'" and lines[4] == "MAX_STEPS = 100" and lines[5] == "print(C.seeds)"
    assert nb.cells[0].source.startswith("REPO_REF = 'main'")


def test_apply_overrides_fails_unless_exactly_one_line_matches():
    entry = ALLOW["notebooks"][TINY]
    missing = new_notebook(cells=[new_code_cell("x = 1")])
    with pytest.raises(cj.JobError, match="found 0"):
        cj.apply_overrides(missing, {"seeds": [0]}, entry)
    twice = new_notebook(cells=[new_code_cell("seeds: tuple = (0,)\nseeds: tuple = (1,)")])
    with pytest.raises(cj.JobError, match="found 2"):
        cj.apply_overrides(twice, {"seeds": [0]}, entry)
    in_markdown = new_notebook(cells=[nbformat.v4.new_markdown_cell("seeds: tuple = (0,)")])
    with pytest.raises(cj.JobError, match="found 0"):
        cj.apply_overrides(in_markdown, {"seeds": [0]}, entry)


def test_pin_repo_ref_sets_the_commit_on_every_repo_ref_line():
    nb = tiny_nb()
    assert cj.pin_repo_ref(nb, SHA) == 1 and nb.cells[0].source.split("\n")[0] == f"REPO_REF = '{SHA}'"
    assert cj.pin_repo_ref(new_notebook(cells=[new_code_cell("x = 1")]), SHA) == 0


def test_prepare_notebook_returns_a_patched_copy_with_the_prelude_first_and_leaves_the_file_alone(tmp_path):
    nb = tiny_nb()
    nb.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="old")]
    nb.cells[0].execution_count = 3
    path = tmp_path / "tiny.ipynb"
    nbformat.write(nb, path)
    before = path.read_bytes()
    prepared = cj.prepare_notebook(path, job(overrides={"seeds": [0]}), ALLOW["notebooks"][TINY])
    assert path.read_bytes() == before
    assert prepared.cells[0].source == cj.PRELUDE and len(prepared.cells) == len(nb.cells) + 1
    assert all(not c.outputs and c.execution_count is None for c in prepared.cells if c.cell_type == "code")
    assert "REPO_REF = '" + SHA + "'" in prepared.cells[1].source and "seeds: tuple = (0,)" in prepared.cells[2].source
    compile(cj.PRELUDE, "prelude", "exec")                                                         # valid Python, and harmless without google.colab
    exec(cj.PRELUDE, {})                                                                           # noqa: S102


# ---- the queue ----------------------------------------------------------------------------------------------------------------

def put(queue, name, spec):
    (queue.dirs["inbox"] / name).write_text(json.dumps(spec) if not isinstance(spec, str) else spec)
    return queue.dirs["inbox"] / name


def test_queue_orders_jobs_by_name_ignores_hidden_and_other_files_and_moves_them(tmp_path):
    q = cj.JobQueue(tmp_path / "jobs")
    q.ensure()
    assert set(os.listdir(q.root)) == set(cj.QUEUE_DIRS)
    for name in ("20260930-b.json", "20260930-a.json", ".hidden.json", "notes.txt", "x.json.part"):
        (q.dirs["inbox"] / name).write_text("{}")
    assert [p.name for p in q.pending()] == ["20260930-a.json", "20260930-b.json"]
    moved = q.move(q.pending()[0], "running")
    assert moved.parent == q.dirs["running"] and [p.name for p in q.pending()] == ["20260930-b.json"]
    assert not q.stop_requested()
    (q.root / "STOP").write_text("")
    assert q.stop_requested()
    q.heartbeat({"state": "idle"})
    assert json.loads((q.root / "worker_status.json").read_text()) == {"state": "idle"}


def test_read_job_reports_oversized_and_invalid_files(tmp_path):
    big = tmp_path / "big.json"
    big.write_text(" " * (cj.MAX_JOB_BYTES + 1))
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    ok = tmp_path / "ok.json"
    ok.write_text('{"a": 1}')
    assert cj.read_job(big)[0] is None and "bytes" in cj.read_job(big)[1][0]
    assert cj.read_job(bad)[0] is None and "unreadable" in cj.read_job(bad)[1][0]
    assert cj.read_job(ok) == ({"a": 1}, []) and cj.read_job(tmp_path / "missing.json")[0] is None


def outcome_runner(calls, status="ok"):
    def run(spec, entry, results):
        calls.append(spec["job_id"])
        results.mkdir(parents=True, exist_ok=True)
        return cj.RunOutcome(status, "done", [{"run_id": "r1", "complete": True, "bundle": [], "files": 1}], "executed_notebook.ipynb")
    return run


def test_process_job_runs_a_valid_job_writes_its_result_and_moves_the_file(tmp_path):
    q = cj.JobQueue(tmp_path)
    q.ensure()
    path = put(q, "j1.json", job(job_id="j1-ok"))
    calls = []
    result = cj.process_job(q, path, ALLOW, outcome_runner(calls), gpu_name_fn=lambda: "", now=lambda: "T", info={"python": "3"})
    assert calls == ["j1-ok"] and result["status"] == "ok" and result["runs"][0]["run_id"] == "r1" and result["job_sha256"] == cj.job_hash(job(job_id="j1-ok"))
    saved = json.loads((q.results_dir("j1-ok") / "result.json").read_text())
    assert saved == json.loads(json.dumps(result)) and saved["worker"]["python"] == "3" and saved["commit"] == SHA
    assert not path.exists() and (q.dirs["done"] / "j1.json").exists() and not list(q.dirs["running"].iterdir())


@pytest.mark.parametrize("content, reason", [
    (job(job_id="j2-bad", notebook="notebooks/evil.ipynb"), "not in the allow-list"),
    (job(job_id="j2-bad", overrides={"seeds": "x"}), "distinct integers"),
    ("{not json", "unreadable"),
    ("[1, 2]", "JSON object"),
])
def test_process_job_rejects_invalid_jobs_without_running_anything(tmp_path, content, reason):
    q = cj.JobQueue(tmp_path)
    q.ensure()
    path = put(q, "j2.json", content)
    calls = []
    result = cj.process_job(q, path, ALLOW, outcome_runner(calls), gpu_name_fn=lambda: "")
    assert calls == [] and result["status"] == "rejected" and reason in result["reason"]
    assert (q.dirs["done"] / "j2.json").exists() and (q.results_dir(result["job_id"]) / "result.json").exists()


def test_process_job_refuses_the_wrong_hardware_and_accepts_the_right_one(tmp_path):
    q = cj.JobQueue(tmp_path)
    q.ensure()
    calls = []
    refused = cj.process_job(q, put(q, "g1.json", job(job_id="g1-t4", notebook=GPU_NB)), ALLOW, outcome_runner(calls), gpu_name_fn=lambda: "Tesla T4")
    none = cj.process_job(q, put(q, "g2.json", job(job_id="g2-none", notebook=GPU_NB)), ALLOW, outcome_runner(calls), gpu_name_fn=lambda: "")
    ok = cj.process_job(q, put(q, "g3.json", job(job_id="g3-l4", notebook=GPU_NB)), ALLOW, outcome_runner(calls), gpu_name_fn=lambda: "NVIDIA L4")
    assert refused["status"] == "refused_hardware" and "Tesla T4" in refused["reason"] and none["status"] == "refused_hardware" and "none" in none["reason"]
    assert ok["status"] == "ok" and ok["worker"]["gpu_name"] == "NVIDIA L4" and calls == ["g3-l4"]


def test_process_job_survives_a_runner_that_raises(tmp_path):
    q = cj.JobQueue(tmp_path)
    q.ensure()

    def boom(spec, entry, results):
        raise RuntimeError("kernel died")

    result = cj.process_job(q, put(q, "b.json", job(job_id="boom-1")), ALLOW, boom, gpu_name_fn=lambda: "")
    assert result["status"] == "failed" and "kernel died" in result["reason"] and "RuntimeError" in result["traceback_tail"]
    assert (q.dirs["done"] / "b.json").exists()


def test_collect_runs_reports_new_runs_bundles_small_summaries_and_hashes_files(tmp_path):
    runs = tmp_path / "runs"
    (runs / "old_run").mkdir(parents=True)
    before = cj.list_runs(runs)
    new = runs / "new_run"
    (new / "reports").mkdir(parents=True)
    (new / "reports" / "acceptance_report.json").write_text('{"ok": true}')
    (new / "reports" / "run_summary.md").write_text("x" * (cj.BUNDLE_MAX_BYTES + 1))
    (new / "COMPLETE").write_text("now")
    (runs / "half_run").mkdir()
    results = tmp_path / "results" / "j"
    infos = {i["run_id"]: i for i in cj.collect_runs(runs, before, results)}
    assert set(infos) == {"new_run", "half_run"} and infos["new_run"]["complete"] and not infos["half_run"]["complete"]
    assert infos["new_run"]["bundle"] == ["reports/acceptance_report.json"] and (results / "bundle/new_run/reports/acceptance_report.json").read_text() == '{"ok": true}'
    assert not (results / "bundle/new_run/reports/run_summary.md").exists()
    manifest = json.loads((results / "bundle/new_run/manifest.json").read_text())
    assert manifest["COMPLETE"]["bytes"] == 3 and len(manifest["COMPLETE"]["sha256"]) == 64 and "reports/run_summary.md" in manifest
    assert cj.list_runs(tmp_path / "nope") == set()


# ---- the worker loop (fakes) ------------------------------------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def worker(tmp_path, **kwargs):
    q = cj.JobQueue(tmp_path / "jobs")
    q.ensure()
    clock = Clock()
    defaults = dict(poll_seconds=10, max_idle_seconds=100, max_session_seconds=10 * 3600, allowlist_loader=lambda: ALLOW, sleep=clock.sleep, clock=clock,
                    log=lambda line: None, gpu_name_fn=lambda: "")
    defaults.update(kwargs)
    return q, clock, defaults


def test_worker_processes_jobs_in_order_then_idles_out_and_reports_stopped(tmp_path):
    q, clock, kw = worker(tmp_path)
    put(q, "02-b.json", job(job_id="second"))
    put(q, "01-a.json", job(job_id="first"))
    calls = []
    results = cj.run_worker(q, tmp_path / "repo", "url", tmp_path / "runs", runner=outcome_runner(calls), worker_commit="w" * 40, **kw)
    assert calls == ["first", "second"] and [r["status"] for r in results] == ["ok", "ok"]
    status = json.loads((q.root / "worker_status.json").read_text())
    assert status["state"] == "stopped" and status["jobs_done"] == 2 and status["worker_commit"] == "w" * 40 and clock.now >= 100
    assert status["settings"] == {"max_idle_seconds": 100, "max_stall_seconds": 1200.0, "progress_seconds": 120.0, "poll_seconds": 10, "once": False}   # the limits the worker actually runs with


def test_idle_limit_helper_prefers_minutes_then_hours_then_the_immediate_exit_default():
    assert cj.DEFAULT_IDLE_MINUTES == 0.0 and cj.idle_limit_seconds() == 0.0
    assert cj.idle_limit_seconds(minutes=2) == 120.0 and cj.idle_limit_seconds(hours=6) == 21600.0
    assert cj.idle_limit_seconds(minutes=0, hours=6) == 0.0 and cj.idle_limit_seconds(minutes=1.5) == 90.0
    with pytest.raises(ValueError, match="negative"):
        cj.idle_limit_seconds(minutes=-1)
    import inspect

    assert inspect.signature(cj.run_worker).parameters["max_idle_seconds"].default == 0.0


def test_worker_with_a_zero_idle_limit_exits_within_one_poll_of_its_last_job(tmp_path):
    q, clock, kw = worker(tmp_path, max_idle_seconds=0)
    put(q, "01.json", job(job_id="only"))
    calls = []
    results = cj.run_worker(q, tmp_path / "repo", "url", tmp_path / "runs", runner=outcome_runner(calls), **kw)
    assert calls == ["only"] and len(results) == 1 and clock.now <= kw["poll_seconds"]
    assert json.loads((q.root / "worker_status.json").read_text())["state"] == "stopped"


def test_worker_notebook_and_script_agree_on_the_idle_exit_and_release_the_runtime():
    import ast

    nb = json.loads((Path(__file__).resolve().parents[1] / "notebooks/05-ops/colab_worker.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    ast.parse(code)
    assert "--max-idle-minutes" in code and "MAX_IDLE_MINUTES = 0" in code and "DISCONNECT_WHEN_DONE = True" in code
    assert code.index("drive.flush_and_unmount()") < code.index("runtime.unassign()")            # Drive is flushed before the VM is released
    assert "except KeyboardInterrupt" in code                                                    # an interrupt keeps the runtime
    helped = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/colab_worker.py"), "--help"], capture_output=True, text=True)
    assert helped.returncode == 0 and "--max-idle-minutes" in helped.stdout


def wait_until(condition, seconds=5.0):
    import time

    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def test_format_duration_and_newest_run_activity(tmp_path):
    assert cj.format_duration(0) == "0m00s" and cj.format_duration(75) == "1m15s" and cj.format_duration(3725) == "1h02m05s" and cj.format_duration(-3) == "0m00s"
    runs = tmp_path / "runs"
    (runs / "old_run").mkdir(parents=True)
    (runs / "old_run" / "x.txt").write_text("x")
    before = cj.list_runs(runs)
    assert cj.newest_run_activity(runs, before) is None                       # no new run directory yet
    (runs / "new_run" / "dynamics").mkdir(parents=True)
    (runs / "new_run" / "empty_dir").mkdir()
    older, newer = runs / "new_run" / "a.json", runs / "new_run" / "dynamics" / "latest.pt"
    older.write_text("{}")
    newer.write_text("w")
    os.utime(older, (1000.0, 1000.0))
    os.utime(newer, (1060.0, 1060.0))
    (runs / "old_run" / "y.txt").write_text("not counted: the run existed before the job")
    activity = cj.newest_run_activity(runs, before, now=lambda: 1100.0)
    assert activity == {"run_id": "new_run", "file": "dynamics/latest.pt", "age_seconds": 40.0, "files": 2, "mtime": 1060.0}
    assert cj.newest_run_activity(runs, before, now=lambda: 900.0)["age_seconds"] == 0.0         # a clock that is behind a file is clamped, not negative
    assert cj.newest_run_activity(tmp_path / "nope", set()) is None


def test_progress_ticker_ticks_while_the_block_runs_survives_a_failing_tick_and_stops_at_exit():
    ticks, errors = [], []

    def tick():
        ticks.append(1)
        if len(ticks) == 2:
            raise RuntimeError("drive hiccup")

    with cj.ProgressTicker(0.01, tick, on_error=errors.append):
        assert wait_until(lambda: len(ticks) >= 4)
    settled = len(ticks)
    assert wait_until(lambda: True) and len(ticks) == settled                               # stopped at exit
    assert errors == ["progress report failed: RuntimeError: drive hiccup"]
    off = []
    with cj.ProgressTicker(0, lambda: off.append(1)) as ticker:
        assert ticker._thread is None
    assert off == []


def test_a_running_job_reports_elapsed_time_and_its_newest_file_to_the_log_and_worker_status(tmp_path):
    lines = []
    q, clock, kw = worker(tmp_path, once=True, log=lines.append, progress_seconds=0.02)
    put(q, "01.json", job(job_id="slow-job"))
    runs, seen = tmp_path / "runs", {}

    def slow(spec, entry, results):
        (runs / "run_a" / "dynamics").mkdir(parents=True)
        (runs / "run_a" / "dynamics" / "latest.pt").write_text("w")
        assert wait_until(lambda: any("still running" in line for line in lines))
        seen.update(json.loads((q.root / "worker_status.json").read_text()))
        results.mkdir(parents=True, exist_ok=True)
        return cj.RunOutcome("ok", "done", [], None)

    results = cj.run_worker(q, tmp_path / "repo", "url", runs, runner=slow, **kw)
    progress = [line for line in lines if "still running" in line]
    assert progress and "01.json: still running" in progress[0] and "newest file run_a/dynamics/latest.pt written" in progress[0]
    assert seen["state"] == "running 01.json" and seen["newest_run_activity"]["file"] == "dynamics/latest.pt" and "job_elapsed_seconds" in seen
    assert [r["status"] for r in results] == ["ok"] and json.loads((q.root / "worker_status.json").read_text())["state"] == "stopped"
    assert "newest_run_activity" not in json.loads((q.root / "worker_status.json").read_text())


def test_progress_reporting_is_off_when_the_interval_is_zero_and_the_cli_exposes_it():
    import inspect

    assert cj.DEFAULT_PROGRESS_SECONDS == 120.0 and inspect.signature(cj.run_worker).parameters["progress_seconds"].default == 120.0
    helped = subprocess.run([sys.executable, str(ROOT / "scripts/colab_worker.py"), "--help"], capture_output=True, text=True)
    assert helped.returncode == 0 and "--progress-minutes" in helped.stdout


def test_activity_watch_counts_new_files_and_gpu_load_as_life_and_nothing_else():
    watch = cj.ActivityWatch(0.0, 600.0)
    assert not watch.stalled(600.0) and watch.stalled(600.1) and watch.idle_seconds(700.0) == 700.0
    watch.observe(300.0, newest_mtime=250.0)                                        # a first file counts
    assert watch.last_active == 300.0
    watch.observe(400.0, newest_mtime=250.0)                                        # the same file again does not
    watch.observe(410.0, newest_mtime=200.0)                                        # an older one does not
    assert watch.last_active == 300.0 and not watch.stalled(900.0) and watch.stalled(900.1)
    watch.observe(500.0, gpu_percent=cj.GPU_ACTIVE_PERCENT - 0.1)                   # an idle GPU does not
    watch.observe(510.0, gpu_percent=None)                                          # no nvidia-smi does not
    assert watch.last_active == 300.0
    watch.observe(520.0, gpu_percent=cj.GPU_ACTIVE_PERCENT)
    assert watch.last_active == 520.0
    watch.observe(530.0, newest_mtime=260.0)                                        # a newer file does
    assert watch.last_active == 530.0
    off = cj.ActivityWatch(0.0, 0)
    assert not off.stalled(10**9)


def test_gpu_utilization_takes_the_busiest_gpu_and_is_none_without_nvidia_smi(monkeypatch):
    class Done:
        def __init__(self, out, code=0):
            self.stdout, self.returncode = out, code

    monkeypatch.setattr(cj.subprocess, "run", lambda *a, **k: Done("3\n87\n"))
    assert cj.gpu_utilization() == 87.0
    monkeypatch.setattr(cj.subprocess, "run", lambda *a, **k: Done("", 9))
    assert cj.gpu_utilization() is None

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(cj.subprocess, "run", missing)
    assert cj.gpu_utilization() is None


def tiny_origin(tmp_path, body):
    origin = tmp_path / "origin"
    (origin / "notebooks" / "05-ops").mkdir(parents=True)
    allow = {"version": 1, "max_timeout_hours": 1, "notebooks": {TINY: {"gpu": [], "default_hours": 0.05, "max_hours": 0.1, "overrides": {}}}}
    (origin / cj.ALLOWLIST_RELPATH).write_text(json.dumps(allow))
    nbformat.write(new_notebook(cells=[new_code_cell(body)]), origin / TINY)
    sh(origin, "git", "init", "-q", "-b", "main")
    sh(origin, "git", "add", "-A")
    sh(origin, "git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    return origin, sh(origin, "git", "rev-parse", "HEAD"), allow


def test_an_idle_job_is_killed_and_reported_as_stalled_with_its_partial_output_kept(tmp_path, monkeypatch):
    pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    import time

    body = ("import os, time\nfrom pathlib import Path\nrun = Path(os.environ['TINY_RUNS']) / 'quiet_run'\nrun.mkdir(parents=True)\n"
            "(run / 'first.txt').write_text('started')\nprint('started', flush=True)\ntime.sleep(300)\n(run / 'never.txt').write_text('late')")
    origin, commit, allow = tiny_origin(tmp_path, body)
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("TINY_RUNS", str(runs))
    spec = cj.make_job(TINY, commit, allow, job_id="quiet-job", timeout_hours=0.05)
    results = tmp_path / "results"
    began = time.monotonic()
    outcome = cj.run_job_notebook(spec, allow["notebooks"][TINY], results, repo_dir=tmp_path / "work", repo_url=str(origin), runs_root=runs,
                                  stall_seconds=4, sample_seconds=0.2, scan_seconds=0.2, gpu_fn=lambda: None)
    assert outcome.status == "stalled" and "no new file" in outcome.reason and "GPU" in outcome.reason
    assert time.monotonic() - began < 120                                              # far below the 300 s the notebook would have slept
    assert (runs / "quiet_run" / "first.txt").exists() and not (runs / "quiet_run" / "never.txt").exists()
    assert [r["run_id"] for r in outcome.runs] == ["quiet_run"] and not outcome.runs[0]["complete"]
    executed = nbformat.read(results / "executed_notebook.ipynb", 4)
    assert any("started" in o.get("text", "") for c in executed.cells for o in c.get("outputs", []))


def test_the_worker_stops_after_a_stalled_job_and_leaves_the_rest_of_the_inbox(tmp_path):
    q, clock, kw = worker(tmp_path)
    put(q, "01.json", job(job_id="quiet"))
    put(q, "02.json", job(job_id="later"))
    calls = []

    def stalls(spec, entry, results):
        calls.append(spec["job_id"])
        results.mkdir(parents=True, exist_ok=True)
        return cj.RunOutcome("stalled", "no sign of life", [], None)

    results = cj.run_worker(q, tmp_path / "repo", "url", tmp_path / "runs", runner=stalls, **kw)
    assert calls == ["quiet"] and [r["status"] for r in results] == ["stalled"]
    assert [p.name for p in q.pending()] == ["02.json"]
    assert json.loads((q.root / "worker_status.json").read_text())["state"] == "stopped"
    assert json.loads((q.results_dir("quiet") / "result.json").read_text())["status"] == "stalled"


def test_the_script_ignores_the_legacy_idle_hours_flag_so_an_old_notebook_cannot_keep_the_worker_polling(tmp_path):
    drive = tmp_path / "drive"
    done = subprocess.run([sys.executable, str(ROOT / "scripts/colab_worker.py"), "--drive-root", str(drive), "--repo-dir", str(tmp_path / "repo"), "--once", "--dry-run",
                           "--max-idle-hours", "6"], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "--max-idle-hours 6 is ignored" in done.stdout and "idle limit is 0 min" in done.stdout and "0 job(s) handled" in done.stdout
    assert "limits: idle 0 min (0: exit as soon as the inbox is empty) | stall 20 min (0: off) | progress every 2 min | session budget 10 h" in done.stdout
    assert json.loads((drive / "jobs" / "worker_status.json").read_text())["settings"]["max_idle_seconds"] == 0.0


def test_stall_watch_is_on_by_default_and_the_cli_and_notebook_expose_it():
    import inspect

    assert cj.DEFAULT_STALL_MINUTES == 20.0 and inspect.signature(cj.run_worker).parameters["max_stall_seconds"].default == 1200.0
    assert inspect.signature(cj.run_job_notebook).parameters["stall_seconds"].default == 0.0          # direct callers opt in
    helped = subprocess.run([sys.executable, str(ROOT / "scripts/colab_worker.py"), "--help"], capture_output=True, text=True)
    assert helped.returncode == 0 and "--max-stall-minutes" in helped.stdout
    nb = json.loads((ROOT / "notebooks/05-ops/colab_worker.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "MAX_STALL_MINUTES = 20" in code and "--max-stall-minutes" in code


def test_worker_once_stops_when_the_inbox_is_empty_and_honours_a_stop_file(tmp_path):
    q, clock, kw = worker(tmp_path, once=True)
    assert cj.run_worker(q, tmp_path / "repo", "url", tmp_path / "runs", runner=outcome_runner([]), **kw) == [] and clock.now == 0
    q2, _, kw2 = worker(tmp_path / "other")
    put(q2, "01.json", job(job_id="never"))
    (q2.root / "STOP").write_text("")
    calls = []
    assert cj.run_worker(q2, tmp_path / "repo", "url", tmp_path / "runs", runner=outcome_runner(calls), **kw2) == [] and calls == [] and q2.pending()


def test_worker_leaves_a_job_that_would_exceed_the_session_budget_in_the_inbox(tmp_path):
    q, _, kw = worker(tmp_path, max_session_seconds=3600)
    put(q, "01.json", job(job_id="too-long", timeout_hours=0.2))        # 720 s fits
    put(q, "02.json", job(job_id="fits"))
    kw["max_session_seconds"] = 600                                   # now neither fits: 0.2 h = 720 s > 600 s
    calls = []
    assert cj.run_worker(q, tmp_path / "repo", "url", tmp_path / "runs", runner=outcome_runner(calls), **kw) == [] and calls == [] and len(q.pending()) == 2


def test_worker_dry_run_validates_and_reports_without_running(tmp_path):
    q, _, kw = worker(tmp_path, dry_run=True, once=True)
    put(q, "01.json", job(job_id="fine"))
    put(q, "02.json", job(job_id="nope", notebook="notebooks/x.ipynb"))
    calls = []
    results = cj.run_worker(q, tmp_path / "repo", "url", tmp_path / "runs", runner=outcome_runner(calls), **kw)
    assert calls == [] and [r["status"] for r in results] == ["would_run", "rejected"] and not q.pending() and not list(q.dirs["results"].iterdir())


# ---- end to end: a real git checkout and a real kernel ----------------------------------------------------------------------

def sh(cwd, *args):
    done = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def test_end_to_end_with_a_real_repository_and_kernel(tmp_path, monkeypatch):
    pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    origin = tmp_path / "origin"
    (origin / "notebooks" / "05-ops").mkdir(parents=True)
    allow = {"version": 1, "max_timeout_hours": 1, "notebooks": {TINY: {"gpu": [], "default_hours": 0.05, "max_hours": 0.1, "overrides": {
        "seeds": {"kind": "int_list", "prefix": "seeds: tuple = "}}}}}
    (origin / cj.ALLOWLIST_RELPATH).write_text(json.dumps(allow))
    body = ("import os, json\nfrom pathlib import Path\nREPO_REF = 'main'\nseeds: tuple = (0, 1, 2)\nrun = Path(os.environ['TINY_RUNS']) / 'tiny_run'\n"
            "(run / 'reports').mkdir(parents=True)\n(run / 'reports' / 'acceptance_report.json').write_text(json.dumps({'seeds': list(seeds), 'ref': REPO_REF}))\n"
            "(run / 'COMPLETE').write_text('done')\nprint('seeds', seeds)")
    nbformat.write(new_notebook(cells=[new_code_cell(body)]), origin / TINY)
    sh(origin, "git", "init", "-q", "-b", "main")
    sh(origin, "git", "add", "-A")
    sh(origin, "git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    commit = sh(origin, "git", "rev-parse", "HEAD")

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("TINY_RUNS", str(runs))
    queue = cj.JobQueue(tmp_path / "jobs")
    queue.ensure()
    put(queue, "01.json", cj.make_job(TINY, commit, allow, job_id="e2e-ok", overrides={"seeds": [5]}, timeout_hours=0.05))
    put(queue, "02.json", cj.make_job(TINY, "f" * 40, allow, job_id="e2e-missing-commit"))
    put(queue, "03.json", job(job_id="e2e-not-allowed", notebook="notebooks/05-ops/other.ipynb", commit=commit))
    results = cj.run_worker(queue, tmp_path / "work", str(origin), runs, once=True, poll_seconds=0, log=lambda line: None, gpu_name_fn=lambda: "")
    by_id = {r["job_id"]: r for r in results}
    assert by_id["e2e-ok"]["status"] == "ok", by_id["e2e-ok"]
    assert by_id["e2e-missing-commit"]["status"] == "failed" and "cat-file" in by_id["e2e-missing-commit"]["reason"]
    assert by_id["e2e-not-allowed"]["status"] == "rejected"
    ok = by_id["e2e-ok"]
    assert [r["run_id"] for r in ok["runs"]] == ["tiny_run"] and ok["runs"][0]["complete"]
    report = json.loads((queue.results_dir("e2e-ok") / "bundle/tiny_run/reports/acceptance_report.json").read_text())
    assert report == {"seeds": [5], "ref": commit}                                             # the override and the pinned commit reached the kernel
    executed = nbformat.read(queue.results_dir("e2e-ok") / "executed_notebook.ipynb", 4)
    assert executed.cells[0].source == cj.PRELUDE and "seeds (5,)" in executed.cells[1].outputs[0]["text"]
    assert sh(tmp_path / "work", "git", "rev-parse", "HEAD") != "" and not queue.pending()
    assert json.loads((queue.results_dir("e2e-ok") / "result.json").read_text())["job_sha256"]
