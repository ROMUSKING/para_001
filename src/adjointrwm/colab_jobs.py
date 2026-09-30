"""A job queue for running allow-listed notebooks in Colab from a Drive folder (docs/plans/colab-handoff.md §5).

A *worker* (``scripts/colab_worker.py``, started from ``notebooks/05-ops/colab_worker.ipynb`` in a Colab runtime with Drive
mounted) polls ``<drive>/jobs/inbox/`` for small JSON job files. A job names a notebook, a full 40-hex commit of the repository
and a few whitelisted overrides; it carries no code, no environment and no secrets. The worker validates the job against an
allow-list **read from origin/main**, refuses the wrong hardware, checks the commit out, executes the notebook from that commit
in a fresh kernel with nbclient, and writes ``result.json`` (plus an executed copy and a small bundle of each new run's summary
files) to ``<drive>/jobs/results/<job_id>/``.

The module needs only the standard library; ``nbformat`` and ``nbclient`` are imported where used.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

SCHEMA = 1
ALLOWLIST_RELPATH = "notebooks/05-ops/allowlist.json"
TRUSTED_REF = "origin/main"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production"
DEFAULT_IDLE_MINUTES = 0.0      # a worker with an empty inbox exits after this long; 0 (the default) means "exit as soon as the inbox is empty", so queue every job before starting it
DEFAULT_STALL_MINUTES = 20.0    # a running job that writes no file and shows no GPU activity for this long is killed and the worker exits; 0 turns the watch off
GPU_ACTIVE_PERCENT = 5.0        # GPU utilisation at or above this counts as a sign of life
DEFAULT_PROGRESS_SECONDS = 120.0  # a running job prints one progress line (and refreshes worker_status.json) this often; 0 turns it off
QUEUE_DIRS = ("inbox", "running", "done", "results")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,63}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,80}$")
JOB_KEYS = {"schema", "job_id", "notebook", "commit", "overrides", "timeout_hours", "submitted_utc", "submitted_by", "note"}
REQUIRED_KEYS = {"schema", "job_id", "notebook", "commit"}
MAX_JOB_BYTES = 64_000
MAX_NOTE = 500
BUNDLE_FILES = ("reports/acceptance_report.json", "reports/run_summary.md", "reports/frozen_before_validation.json",
                "tests/correctness.json", "config/run_config.json")
BUNDLE_MAX_BYTES = 300_000
EXECUTED_MAX_BYTES = 50_000_000
HASH_MAX_BYTES = 50_000_000
OVERRIDE_KINDS = ("int_list", "choice_list", "run_id", "int_or_null")

# Injected as the first cell of the executed copy (never committed). A job runs in a fresh kernel of the Colab VM: importing
# google.colab makes ``'google.colab' in sys.modules`` true so the notebook takes its Colab branch, and the worker has already
# mounted Drive, so the notebook's own mount call becomes a no-op. Outside Colab the import fails and nothing happens.
PRELUDE = '''# colab_jobs prelude (added by the worker; not part of the committed notebook)
try:
    import google.colab  # noqa: F401
    import google.colab.drive as _colab_drive
    _colab_drive.mount = lambda *args, **kwargs: print('Drive is mounted by the worker')
except ImportError:
    pass
'''


class JobError(ValueError):
    """A job or allow-list that must not be run."""


def idle_limit_seconds(minutes: float | None = None, hours: float | None = None, default_minutes: float = DEFAULT_IDLE_MINUTES) -> float:
    """The idle limit of a worker in seconds: ``minutes`` if given, else ``hours``, else the default (0: exit as soon as the inbox is empty). Negative values are refused."""
    value = minutes * 60.0 if minutes is not None else hours * 3600.0 if hours is not None else default_minutes * 60.0
    if value < 0:
        raise ValueError("the idle limit cannot be negative")
    return float(value)


def utcnow() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def job_hash(spec: Mapping) -> str:
    return hashlib.sha256(canonical_json(dict(spec)).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


# ---- the allow-list and job validation ----------------------------------------------------------------------------

def check_allowlist(allow: Mapping) -> Mapping:
    """Shape check of ``notebooks/05-ops/allowlist.json``; raises :class:`JobError`."""
    if not isinstance(allow, Mapping) or allow.get("version") != 1 or not isinstance(allow.get("notebooks"), Mapping) or not allow["notebooks"]:
        raise JobError("allow-list must be an object with version 1 and a non-empty 'notebooks' mapping")
    for path, entry in allow["notebooks"].items():
        if not _notebook_path_ok(path):
            raise JobError(f"allow-list notebook path {path!r} is not a notebooks/… .ipynb path")
        for g in entry.get("gpu", []):
            if not isinstance(g, str) or not g:
                raise JobError(f"{path}: gpu entries must be non-empty strings")
        if not 0 < float(entry.get("max_hours", 0)) <= float(allow.get("max_timeout_hours", 12)):
            raise JobError(f"{path}: max_hours must be in (0, max_timeout_hours]")
        for name, spec in entry.get("overrides", {}).items():
            if spec.get("kind") not in OVERRIDE_KINDS or not isinstance(spec.get("prefix"), str) or not spec["prefix"]:
                raise JobError(f"{path}: override {name!r} needs a known 'kind' and a 'prefix'")
            if spec["kind"] == "choice_list" and not spec.get("choices"):
                raise JobError(f"{path}: override {name!r} of kind choice_list needs 'choices'")
    return allow


def _notebook_path_ok(path) -> bool:
    if not isinstance(path, str) or not path.startswith("notebooks/") or not path.endswith(".ipynb") or "\\" in path:
        return False
    return ".." not in Path(path).parts and not Path(path).is_absolute() and len(path) <= 200


def validate_override(name: str, spec: Mapping, value) -> list[str]:
    kind = spec["kind"]
    if kind == "int_list":
        if not isinstance(value, list) or not 1 <= len(value) <= 20 or not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 10**6 for v in value) \
                or len(set(value)) != len(value):
            return [f"override {name!r}: expected 1 to 20 distinct integers in [0, 1000000]"]
    elif kind == "choice_list":
        if not isinstance(value, list) or not value or len(set(map(str, value))) != len(value) or not all(isinstance(v, str) and v in spec["choices"] for v in value):
            return [f"override {name!r}: expected distinct values from {sorted(spec['choices'])}"]
    elif kind == "run_id":
        if not isinstance(value, str) or not RUN_ID_RE.match(value):
            return [f"override {name!r}: expected a run id of 3 to 80 characters from [A-Za-z0-9_-]"]
    elif kind == "int_or_null":
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 10**7):
            return [f"override {name!r}: expected null or an integer in [1, 10000000]"]
    return []


def validate_job(spec, allow: Mapping) -> list[str]:
    """All reasons a job spec must be rejected (empty list: acceptable)."""
    if not isinstance(spec, Mapping):
        return ["a job must be a JSON object"]
    errors = []
    unknown = set(spec) - JOB_KEYS
    if unknown:
        errors.append(f"unknown keys {sorted(unknown)}")
    missing = REQUIRED_KEYS - set(spec)
    if missing:
        errors.append(f"missing keys {sorted(missing)}")
    if errors:
        return errors
    if spec["schema"] != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if not isinstance(spec["job_id"], str) or not JOB_ID_RE.match(spec["job_id"]):
        errors.append("job_id must be 3 to 64 characters from [A-Za-z0-9_.-] starting with a letter or digit")
    if not isinstance(spec["commit"], str) or not SHA_RE.match(spec["commit"]):
        errors.append("commit must be a full 40-character lowercase hex SHA (jobs are pinned for reproducibility)")
    notebook = spec["notebook"]
    if not _notebook_path_ok(notebook) or notebook not in allow["notebooks"]:
        return errors + [f"notebook {notebook!r} is not in the allow-list"]
    entry = allow["notebooks"][notebook]
    overrides = spec.get("overrides", {})
    if not isinstance(overrides, Mapping):
        errors.append("overrides must be an object")
    else:
        for name, value in overrides.items():
            if name not in entry.get("overrides", {}):
                errors.append(f"override {name!r} is not allowed for {notebook}")
            else:
                errors += validate_override(name, entry["overrides"][name], value)
    hours = spec.get("timeout_hours", entry.get("default_hours", 2))
    if not isinstance(hours, (int, float)) or isinstance(hours, bool) or not 0 < hours <= float(entry["max_hours"]):
        errors.append(f"timeout_hours must be in (0, {entry['max_hours']}] for {notebook}")
    for key, limit in (("note", MAX_NOTE), ("submitted_by", 80), ("submitted_utc", 40)):
        if key in spec and (not isinstance(spec[key], str) or len(spec[key]) > limit):
            errors.append(f"{key} must be a string of at most {limit} characters")
    return errors


def check_job(spec, allow: Mapping) -> None:
    errors = validate_job(spec, allow)
    if errors:
        raise JobError("; ".join(errors))


def make_job(notebook: str, commit: str, allow: Mapping, *, overrides: Mapping | None = None, timeout_hours: float | None = None, job_id: str | None = None,
             note: str = "", submitted_by: str = "claude-code", now: Callable[[], str] = utcnow) -> dict:
    """A validated job spec. ``job_id`` defaults to the notebook's stem plus a UTC timestamp."""
    stamp = now()
    if job_id is None:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", Path(notebook).stem).strip("-").lower()[:40]
        try:
            compact = datetime.datetime.fromisoformat(stamp).astimezone(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        except ValueError:
            compact = re.sub(r"[^0-9]", "", stamp)[:14]
        job_id = f"{slug}-{compact}"
    spec: dict = {"schema": SCHEMA, "job_id": job_id, "notebook": notebook, "commit": commit, "submitted_utc": stamp, "submitted_by": submitted_by}
    if overrides:
        spec["overrides"] = dict(overrides)
    if timeout_hours is not None:
        spec["timeout_hours"] = timeout_hours
    if note:
        spec["note"] = note
    check_job(spec, allow)
    return spec


# ---- preparing the notebook that is actually executed --------------------------------------------------------------

def render_override(kind: str, value) -> str:
    if kind == "int_list":
        return "(" + ", ".join(str(v) for v in value) + ("," if len(value) == 1 else "") + ")"
    if kind == "choice_list":
        return "(" + ", ".join(repr(v) for v in value) + ("," if len(value) == 1 else "") + ")"
    if kind == "run_id":
        return repr(value)
    return "None" if value is None else str(value)


def _code_lines(nb):
    for cell in nb.cells:
        if cell.cell_type == "code":
            yield cell


def apply_overrides(nb, overrides: Mapping, entry: Mapping) -> None:
    """Replace, in place, the one line of a code cell that starts with each override's prefix; anything else is an error."""
    for name, value in overrides.items():
        spec = entry["overrides"][name]
        pattern = re.compile(r"^(\s*)" + re.escape(spec["prefix"]) + r".*$")
        hits = [(cell, i) for cell in _code_lines(nb) for i, line in enumerate(cell.source.split("\n")) if pattern.match(line)]
        if len(hits) != 1:
            raise JobError(f"override {name!r}: expected exactly one line starting with {spec['prefix']!r}, found {len(hits)}")
        cell, index = hits[0]
        lines = cell.source.split("\n")
        indent = pattern.match(lines[index]).group(1)
        lines[index] = f"{indent}{spec['prefix']}{render_override(spec['kind'], value)}"
        cell.source = "\n".join(lines)


def pin_repo_ref(nb, commit: str) -> int:
    """Set every ``REPO_REF = '…'`` line to the job's commit; returns how many lines were changed."""
    pattern = re.compile(r"^(\s*)REPO_REF = '[^']*'.*$")
    changed = 0
    for cell in _code_lines(nb):
        lines = cell.source.split("\n")
        for i, line in enumerate(lines):
            if pattern.match(line):
                lines[i] = f"{pattern.match(line).group(1)}REPO_REF = '{commit}'"
                changed += 1
        cell.source = "\n".join(lines)
    return changed


def prepare_notebook(path: Path, spec: Mapping, entry: Mapping):
    """The notebook at ``path`` with overrides applied, ``REPO_REF`` pinned, outputs cleared and the prelude first (a copy; the file is untouched)."""
    import nbformat
    from nbformat.v4 import new_code_cell

    nb = copy.deepcopy(nbformat.read(path, 4))
    for cell in _code_lines(nb):
        cell.outputs = []
        cell.execution_count = None
    apply_overrides(nb, spec.get("overrides", {}), entry)
    pin_repo_ref(nb, spec["commit"])
    nb.cells.insert(0, new_code_cell(PRELUDE))
    return nb


# ---- git -----------------------------------------------------------------------------------------------------------

def git(repo_dir: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo_dir), *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise JobError(f"git {' '.join(args)} failed: {(done.stderr or done.stdout).strip()[-400:]}")
    return done.stdout


def ensure_checkout(repo_dir: Path, repo_url: str, commit: str | None = None) -> str:
    """Clone or fetch ``repo_url`` into ``repo_dir`` (whose origin must be ``repo_url``) and, if ``commit`` is given, check it out detached.
    Returns the checked-out HEAD. Untracked files are left alone; tracked files are forced to the commit."""
    repo_dir = Path(repo_dir)
    if not (repo_dir / ".git").exists():
        if repo_dir.exists() and any(repo_dir.iterdir()):
            raise JobError(f"{repo_dir} exists and is not a git checkout")
        done = subprocess.run(["git", "clone", "--quiet", repo_url, str(repo_dir)], capture_output=True, text=True)
        if done.returncode != 0:
            raise JobError(f"git clone failed: {done.stderr.strip()[-400:]}")
    origin = git(repo_dir, "remote", "get-url", "origin").strip()
    if origin != repo_url:
        raise JobError(f"{repo_dir} has origin {origin!r}, not {repo_url!r}")
    git(repo_dir, "fetch", "--quiet", "origin", "+refs/heads/*:refs/remotes/origin/*")
    if commit is not None:
        if not SHA_RE.match(commit):
            raise JobError("commit must be a full 40-character SHA")
        git(repo_dir, "cat-file", "-e", f"{commit}^{{commit}}")
        git(repo_dir, "checkout", "--quiet", "--detach", "--force", commit)
    return git(repo_dir, "rev-parse", "HEAD").strip()


def trusted_allowlist(repo_dir: Path, ref: str = TRUSTED_REF) -> Mapping:
    """The allow-list as committed on ``origin/main`` (so a job or a feature branch cannot widen it)."""
    try:
        return check_allowlist(json.loads(git(repo_dir, "show", f"{ref}:{ALLOWLIST_RELPATH}")))
    except json.JSONDecodeError as error:
        raise JobError(f"the allow-list on {ref} is not valid JSON: {error}") from error


# ---- the queue ------------------------------------------------------------------------------------------------------

@dataclass
class RunOutcome:
    status: str                      # ok, failed, timeout
    reason: str = ""
    runs: list = field(default_factory=list)
    executed_notebook: str | None = None


class JobQueue:
    """``<root>/inbox`` (jobs waiting), ``running``, ``done`` (job files after processing), ``results/<job_id>/`` and ``worker_status.json``."""

    def __init__(self, root):
        self.root = Path(root)
        self.dirs = {name: self.root / name for name in QUEUE_DIRS}

    def ensure(self) -> None:
        for path in self.dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    def pending(self) -> list[Path]:
        return sorted((p for p in self.dirs["inbox"].glob("*.json") if not p.name.startswith(".")), key=lambda p: (p.name, p.stat().st_mtime))

    def move(self, path: Path, where: str) -> Path:
        dest = self.dirs[where] / path.name
        os.replace(path, dest)
        return dest

    def results_dir(self, job_id: str) -> Path:
        return self.dirs["results"] / job_id

    def heartbeat(self, status: Mapping) -> None:
        write_json(self.root / "worker_status.json", dict(status))

    def stop_requested(self) -> bool:
        return (self.root / "STOP").exists()


def read_job(path: Path) -> tuple[dict | None, list[str]]:
    try:
        size = path.stat().st_size
        if size > MAX_JOB_BYTES:
            return None, [f"job file is {size} bytes (limit {MAX_JOB_BYTES})"]
        spec = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"unreadable job file: {error}"]
    return spec, []


def gpu_name() -> str:
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return done.stdout.strip().splitlines()[0] if done.returncode == 0 and done.stdout.strip() else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def gpu_utilization() -> float | None:
    """The highest GPU utilisation in percent right now, or ``None`` when there is no ``nvidia-smi`` or it fails."""
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        values = [float(line) for line in done.stdout.split() if line.replace(".", "", 1).isdigit()]
        return max(values) if done.returncode == 0 and values else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def worker_info(worker_commit: str | None = None) -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "gpu_name": gpu_name(), "worker_commit": worker_commit}
    try:
        import torch  # noqa: PLC0415

        info["torch"] = torch.__version__
    except ImportError:
        info["torch"] = None
    return info


def process_job(queue: JobQueue, path: Path, allow: Mapping, runner: Callable[[Mapping, Mapping, Path], RunOutcome], *,
                gpu_name_fn: Callable[[], str] = gpu_name, now: Callable[[], str] = utcnow, info: Mapping | None = None) -> dict:
    """Validate, hardware-check and run one job file, write its ``result.json``, and move the job file to ``done``. Always returns the result."""
    started = now()
    spec, errors = read_job(path)
    if spec is not None:
        errors = validate_job(spec, allow)
    job_id = spec.get("job_id") if isinstance(spec, dict) and isinstance(spec.get("job_id"), str) and JOB_ID_RE.match(spec["job_id"]) else path.stem
    results = queue.results_dir(re.sub(r"[^A-Za-z0-9_.-]", "_", job_id)[:64] or "unnamed")
    result: dict = {"schema": SCHEMA, "job_id": job_id, "job_file": path.name, "started_utc": started, "worker": dict(info or {})}
    if isinstance(spec, dict):
        result.update({"job_sha256": job_hash(spec), "notebook": spec.get("notebook"), "commit": spec.get("commit"), "overrides": spec.get("overrides", {})})
    if errors:
        result.update({"status": "rejected", "reason": "; ".join(errors)})
    else:
        entry = allow["notebooks"][spec["notebook"]]
        gpus = entry.get("gpu", [])
        found = gpu_name_fn()
        result["worker"]["gpu_name"] = found
        if gpus and not any(g in found for g in gpus):
            result.update({"status": "refused_hardware", "reason": f"needs a GPU matching {gpus}; this runtime has {found or 'none'}"})
        else:
            path = queue.move(path, "running")
            try:
                outcome = runner(spec, entry, results)
                result.update({"status": outcome.status, "reason": outcome.reason, "runs": outcome.runs, "executed_notebook": outcome.executed_notebook})
            except Exception as error:  # the worker must survive any job
                result.update({"status": "failed", "reason": f"{type(error).__name__}: {error}"[:2000], "traceback_tail": traceback.format_exc()[-3000:]})
    result["finished_utc"] = now()
    write_json(results / "result.json", result)
    try:
        queue.move(path, "done")
    except FileNotFoundError:
        pass
    return result


# ---- running a notebook ---------------------------------------------------------------------------------------------------

def list_runs(runs_root: Path) -> set[str]:
    return {p.name for p in Path(runs_root).iterdir() if p.is_dir()} if Path(runs_root).is_dir() else set()


def file_manifest(run_dir: Path, limit: int = 2000) -> dict:
    out = {}
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and len(out) < limit:
            size = p.stat().st_size
            out[p.relative_to(run_dir).as_posix()] = {"bytes": size, "sha256": sha256_file(p) if size <= HASH_MAX_BYTES else "skipped (larger than the hash limit)"}
    return out


def collect_runs(runs_root: Path, before: set[str], results_dir: Path) -> list[dict]:
    """New run directories since ``before``: completeness, a hashed file manifest, and a small bundle of summary files copied next to the result."""
    infos = []
    for run_id in sorted(list_runs(runs_root) - before):
        run_dir = Path(runs_root) / run_id
        bundle = []
        for rel in BUNDLE_FILES:
            src = run_dir / rel
            if src.is_file() and src.stat().st_size <= BUNDLE_MAX_BYTES:
                dest = results_dir / "bundle" / run_id / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
                bundle.append(rel)
        manifest = file_manifest(run_dir)
        infos.append({"run_id": run_id, "complete": (run_dir / "COMPLETE").exists(), "bundle": bundle, "files": len(manifest)})
        write_json(results_dir / "bundle" / run_id / "manifest.json", manifest)
    return infos


def run_job_notebook(spec: Mapping, entry: Mapping, results_dir: Path, *, repo_dir: Path, repo_url: str, runs_root: Path, stall_seconds: float = 0.0,
                     sample_seconds: float = 15.0, scan_seconds: float = 60.0, gpu_fn: Callable[[], float | None] = gpu_utilization,
                     clock: Callable[[], float] = time.time) -> RunOutcome:
    """Check out the job's commit, execute the prepared notebook with nbclient in a fresh kernel, and collect what it wrote.

    With ``stall_seconds > 0`` a watcher thread kills the kernel when the job has shown no sign of life for that long (no new file in its run
    directories, GPU idle; see ``ActivityWatch``) and the outcome is ``stalled``. The partial run directory is kept."""
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError
    import nbformat

    head = ensure_checkout(repo_dir, repo_url, spec["commit"])
    notebook_path = Path(repo_dir) / spec["notebook"]
    results_dir.mkdir(parents=True, exist_ok=True)
    result_nb = prepare_notebook(notebook_path, spec, entry)
    hours = float(spec.get("timeout_hours", entry.get("default_hours", 2)))
    before = list_runs(runs_root)
    client = NotebookClient(result_nb, timeout=int(hours * 3600), interrupt_on_timeout=True, startup_timeout=180, kernel_name="python3",
                            resources={"metadata": {"path": str(notebook_path.parent)}})
    status, reason = "ok", f"executed at {head}"
    watch = ActivityWatch(clock(), stall_seconds)
    stalled, finished = threading.Event(), threading.Event()

    def watcher() -> None:
        last_scan = float("-inf")
        while not finished.wait(sample_seconds):
            try:
                now = clock()
                newest = None
                if now - last_scan >= scan_seconds:
                    last_scan = now
                    activity = newest_run_activity(runs_root, before)
                    newest = activity["mtime"] if activity else None
                watch.observe(now, newest, gpu_fn())
                if watch.stalled(now) and kill_kernel(client):
                    stalled.set()
                    return
            except Exception:  # noqa: BLE001 - a failing probe must not take the job down; try again next time
                continue

    thread = threading.Thread(target=watcher, name="job-stall-watch", daemon=True)
    if stall_seconds > 0:
        thread.start()
    try:
        client.execute()
    except CellTimeoutError as error:
        status, reason = "timeout", f"a cell ran longer than {hours} h: {str(error)[-500:]}"
    except (CellExecutionError, DeadKernelError) as error:
        status, reason = "failed", str(error)[-3000:]
    except Exception:  # noqa: BLE001 - after a stall kill nbclient may raise something else; anything else is a real failure
        if not stalled.is_set():
            raise
    finally:
        finished.set()
        if thread.is_alive():
            thread.join(timeout=10)
    if stalled.is_set():
        status = "stalled"
        reason = (f"no new file in the run directories and no GPU activity for more than {stall_seconds / 60:.1f} min; the kernel was killed "
                  f"(a partial run directory may remain and can be resumed with resume_run_id where the notebook supports it)")
    executed = results_dir / "executed_notebook.ipynb"
    nbformat.write(result_nb, executed)
    if executed.stat().st_size > EXECUTED_MAX_BYTES:
        executed.unlink()
        reason += f" (executed notebook larger than {EXECUTED_MAX_BYTES} bytes, not kept)"
        executed_name = None
    else:
        executed_name = executed.name
    return RunOutcome(status, reason, collect_runs(runs_root, before, results_dir), executed_name)


# ---- progress while a job runs ------------------------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes}m{secs:02d}s"


def newest_run_activity(runs_root: Path, before: set[str], *, now: Callable[[], float] = time.time, limit: int = 20000) -> dict | None:
    """The most recently written file under the run directories created since ``before``: ``{run_id, file, age_seconds, files}``, or ``None`` if there is none yet."""
    best: tuple[float, str, str] | None = None
    files = 0
    for run_id in sorted(list_runs(runs_root) - before):
        for path in Path(runs_root, run_id).rglob("*"):
            if files >= limit:
                break
            try:
                if not path.is_file():
                    continue
                mtime = path.stat().st_mtime
            except OSError:          # a file that vanishes or is mid-sync is not progress evidence either way
                continue
            files += 1
            if best is None or mtime > best[0]:
                best = (mtime, run_id, path.relative_to(Path(runs_root, run_id)).as_posix())
    if best is None:
        return None
    return {"run_id": best[1], "file": best[2], "age_seconds": max(0.0, now() - best[0]), "files": files, "mtime": best[0]}


class ActivityWatch:
    """Decides when a running job is idle: it has shown no sign of life for ``stall_seconds``.

    A sign of life is a file in the job's new run directories newer than any seen before, or a GPU utilisation of at least ``GPU_ACTIVE_PERCENT``.
    The clock starts at construction, so the start-up of a job counts against the limit. ``stall_seconds <= 0`` never reports a stall."""

    def __init__(self, now: float, stall_seconds: float):
        self.stall_seconds = float(stall_seconds)
        self.last_active = now
        self.newest_mtime: float | None = None

    def observe(self, now: float, newest_mtime: float | None = None, gpu_percent: float | None = None) -> None:
        if newest_mtime is not None and (self.newest_mtime is None or newest_mtime > self.newest_mtime):
            self.newest_mtime = newest_mtime
            self.last_active = now
        if gpu_percent is not None and gpu_percent >= GPU_ACTIVE_PERCENT:
            self.last_active = now

    def idle_seconds(self, now: float) -> float:
        return max(0.0, now - self.last_active)

    def stalled(self, now: float) -> bool:
        return self.stall_seconds > 0 and self.idle_seconds(now) > self.stall_seconds


def kill_kernel(client) -> bool:
    """Kill the kernel process of a running ``nbclient`` client (POSIX); nbclient then raises ``DeadKernelError`` in ``execute``. ``False`` if there is no kernel yet."""
    pid = getattr(getattr(getattr(client, "km", None), "provisioner", None), "pid", None)
    if not pid:
        return False
    try:
        os.kill(int(pid), signal.SIGKILL)
    except (OSError, ValueError):
        return False
    return True


class ProgressTicker:
    """Calls ``tick()`` every ``interval`` seconds on a daemon thread while the ``with`` block runs (never at entry; ``interval <= 0`` disables it).

    A tick that raises is logged through ``on_error`` and does not stop the ticker or the job."""

    def __init__(self, interval: float, tick: Callable[[], None], on_error: Callable[[str], None] = lambda message: None):
        self.interval, self.tick, self.on_error = float(interval), tick, on_error
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.tick()
            except Exception as error:  # noqa: BLE001 - progress reporting must never take a job down
                self.on_error(f"progress report failed: {type(error).__name__}: {error}")

    def __enter__(self) -> "ProgressTicker":
        if self.interval > 0:
            self._thread = threading.Thread(target=self._run, name="job-progress", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)


# ---- the worker loop ------------------------------------------------------------------------------------------------------


def run_worker(queue: JobQueue, repo_dir: Path, repo_url: str, runs_root: Path, *, poll_seconds: float = 30.0, max_idle_seconds: float = DEFAULT_IDLE_MINUTES * 60,
               max_session_seconds: float = 10 * 3600, once: bool = False, dry_run: bool = False, progress_seconds: float = DEFAULT_PROGRESS_SECONDS,
               max_stall_seconds: float = DEFAULT_STALL_MINUTES * 60,
               runner: Callable[[Mapping, Mapping, Path], RunOutcome] | None = None, allowlist_loader: Callable[[], Mapping] | None = None,
               sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time, log: Callable[[str], None] = print,
               gpu_name_fn: Callable[[], str] = gpu_name, worker_commit: str | None = None) -> list[dict]:
    """Poll the inbox and process jobs one at a time until idle too long, the session budget would be exceeded, a ``STOP`` file appears, or (``once``)
    the inbox is empty. The allow-list is re-read from ``origin/main`` for every job. ``dry_run`` validates and reports without running.
    While a job runs, every ``progress_seconds`` the worker logs how long it has run and which file the job wrote last, and records the same in
    ``worker_status.json``, so a multi-hour job that prints nothing is distinguishable from a hung one. A job that shows no sign of life for
    ``max_stall_seconds`` is killed (``stalled``; see ``run_job_notebook``) and the worker then exits without starting the next job, so that the
    runtime can be released; 0 turns the watch off."""
    queue.ensure()
    info = worker_info(worker_commit)
    info["worker_id"] = f"{platform.node()}-{os.getpid()}"
    started_utc = utcnow()
    started = clock()
    last_work = started
    load = allowlist_loader or (lambda: (ensure_checkout(repo_dir, repo_url), trusted_allowlist(repo_dir))[1])
    run = runner or (lambda spec, entry, results: run_job_notebook(spec, entry, results, repo_dir=repo_dir, repo_url=repo_url, runs_root=runs_root,
                                                                   stall_seconds=max_stall_seconds))
    results: list[dict] = []

    beat_lock = threading.Lock()            # the progress thread and the loop both write worker_status.json
    settings = {"max_idle_seconds": max_idle_seconds, "max_stall_seconds": max_stall_seconds, "progress_seconds": progress_seconds, "poll_seconds": poll_seconds, "once": once}

    def beat(state: str, **extra) -> None:
        with beat_lock:
            queue.heartbeat({**info, "state": state, "started_utc": started_utc, "last_poll_utc": utcnow(), "jobs_done": len(results), "dry_run": dry_run,
                             "session_budget_hours": max_session_seconds / 3600, "settings": settings, **extra})

    while True:
        if queue.stop_requested():
            log("STOP file found; leaving.")
            break
        pending = queue.pending()
        if pending:
            path = pending[0]
            allow = load()
            spec, errors = read_job(path)
            default = allow["notebooks"].get(spec.get("notebook"), {}).get("default_hours", 2) if isinstance(spec, dict) else 0
            hours = spec.get("timeout_hours", default) if isinstance(spec, dict) else 0
            if not errors and isinstance(hours, (int, float)) and (clock() - started) + float(hours) * 3600 > max_session_seconds and not dry_run:
                log(f"{path.name}: {hours} h would exceed the session budget of {max_session_seconds / 3600:.1f} h; leaving it in the inbox.")
                break
            beat(f"running {path.name}")
            log(f"processing {path.name}")
            if dry_run:
                errs = errors or validate_job(spec, allow)
                outcome = {"job_file": path.name, "status": "rejected" if errs else "would_run", "reason": "; ".join(errs)}
                log(json.dumps(outcome))
                results.append(outcome)
                queue.move(path, "done")
            else:
                before, job_started = list_runs(runs_root), clock()

                def report(job_file: str = path.name, before: set[str] = before, job_started: float = job_started) -> None:
                    activity = newest_run_activity(runs_root, before)
                    elapsed = clock() - job_started
                    where = (f"newest file {activity['run_id']}/{activity['file']} written {format_duration(activity['age_seconds'])} ago ({activity['files']} files)"
                             if activity else "no run directory yet")
                    beat(f"running {job_file}", job_elapsed_seconds=round(elapsed), newest_run_activity=activity)   # the status first: a visible log line implies it is on disk
                    log(f"{job_file}: still running, {format_duration(elapsed)} elapsed; {where}")

                with ProgressTicker(progress_seconds, report, on_error=log):
                    result = process_job(queue, path, allow, run, gpu_name_fn=gpu_name_fn, info=info)
                log(f"{path.name}: {result['status']} {result.get('reason', '')[:200]}")
                results.append(result)
                if result["status"] == "stalled":
                    log(f"{path.name}: stalled; stopping the worker. Jobs still in the inbox stay there.")
                    break
            last_work = clock()
            continue
        beat("idle")
        if once or clock() - last_work > max_idle_seconds or clock() - started > max_session_seconds:
            break
        sleep(poll_seconds)
    beat("stopped")
    return results
