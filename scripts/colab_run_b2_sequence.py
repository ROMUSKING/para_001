#!/usr/bin/env python3
"""Execute B2 jobs sequentially on Colab L4 runtime:
1. Job 1: B2 Seed 0 Probe (seeds=[0]) -> tests revised dynamics gate and end-to-end allocators/traces.
2. Job 2: B2 5-Seed Confirmatory (seeds=[0, 1, 2, 3, 4]) -> full confirmatory evaluation.

Monitors cell progress and logs all outputs to Drive and local console.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production")
REPO_DIR = Path("/content/para_001")
SOURCE_NB = REPO_DIR / "notebooks/01-production/AdjointRWM_Production_Pilot_v2.ipynb"
SHARED_CACHE = DRIVE_ROOT / "runs/droid100_adjoint_v2_20260930T165409Z/cache/cache_manifest.json"
LOG_DIR = DRIVE_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MASTER_LOG = LOG_DIR / "b2_sequential_master.log"
STATUS_FILE = DRIVE_ROOT / "jobs" / "sequential_status.json"


def log(msg: str) -> None:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(MASTER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def update_status(job_name: str, state: str, details: dict | None = None) -> None:
    payload = {
        "job": job_name,
        "state": state,
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **(details or {}),
    }
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(STATUS_FILE)


def prepare_notebook_for_job(job_name: str, seeds: list[int], run_id: str) -> nbformat.NotebookNode:
    nb = nbformat.read(SOURCE_NB, as_version=4)
    
    # Patch Cell 0: ensure it doesn't checkout or overwrite /content/para_001
    cell0 = nb.cells[0]
    if cell0.cell_type == "code":
        cell0_lines = []
        for line in cell0.source.splitlines():
            if "subprocess.run(['git', '-C', REPO_DIR, 'checkout'" in line:
                cell0_lines.append("    # git checkout skipped to preserve local gate revisions")
            elif "subprocess.run(['git', '-C', REPO_DIR, 'fetch'" in line:
                cell0_lines.append("    # git fetch skipped")
            elif "subprocess.run(['git', 'clone'" in line:
                cell0_lines.append("    pass  # repo already exists")
            else:
                cell0_lines.append(line)
        cell0.source = "\n".join(cell0_lines)

    # Patch Cell 1: seeds and run_id
    for cell in nb.cells:
        if cell.cell_type == "code" and "PilotV2Config" in cell.source:
            lines = cell.source.splitlines()
            new_lines = []
            for l in lines:
                if l.strip().startswith("seeds: tuple ="):
                    indent = l[:l.index("seeds:")]
                    new_lines.append(f"{indent}seeds: tuple = {tuple(seeds)}")
                elif l.strip().startswith("RUN_ID = 'droid100_adjoint_v2_' + SESSION_ID"):
                    indent = l[:l.index("RUN_ID =")]
                    new_lines.append(f"{indent}RUN_ID = '{run_id}'")
                else:
                    new_lines.append(l)
            cell.source = "\n".join(new_lines)
            break

    # Patch Cell 2: pre-seed cache if needed
    for cell in nb.cells:
        if cell.cell_type == "code" and "CACHE_MANIFEST = PATHS['cache'] / 'cache_manifest.json'" in cell.source:
            lines = cell.source.splitlines()
            new_lines = []
            for l in lines:
                new_lines.append(l)
                if l.strip().startswith("LOCAL_CACHE = LOCAL / 'cache'"):
                    new_lines.append("shared_cache = Path('/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production/runs/droid100_adjoint_v2_20260930T165409Z/cache/cache_manifest.json')")
                    new_lines.append("if not CACHE_MANIFEST.exists() and shared_cache.exists():")
                    new_lines.append("    atomic_copy(shared_cache, CACHE_MANIFEST)")
                    new_lines.append("    print('Pre-seeded cache manifest from', shared_cache)")
            cell.source = "\n".join(new_lines)
            break

    return nb


def execute_job(job_name: str, seeds: list[int], timeout_seconds: int = 14400) -> tuple[bool, str, dict]:
    session_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"droid100_adjoint_v2_{job_name}_{session_id}"
    run_dir = DRIVE_ROOT / "runs" / run_id
    # Note: Do NOT mkdir run_dir beforehand! Cell 1 of the notebook will check if RUN_DIR.exists()
    # to enforce immutability, and create it together with PATHS.

    log(f"================================================================")
    log(f"STARTING {job_name} | Run ID: {run_id} | Seeds: {seeds}")
    log(f"================================================================")
    update_status(job_name, "RUNNING", {"run_id": run_id, "seeds": seeds, "run_dir": str(run_dir)})

    nb = prepare_notebook_for_job(job_name, seeds, run_id)
    total_cells = len(nb.cells)

    def on_cell_start(cell, cell_index, **kw):
        header = cell.source.splitlines()[0][:60] if cell.source else ""
        log(f"[{job_name}] Starting Cell {cell_index+1}/{total_cells}: {header}")

    def on_cell_executed(cell, cell_index, execute_reply=None, **kw):
        outputs = cell.get("outputs", [])
        for out in outputs:
            txt = out.get("text", "") or out.get("data", {}).get("text/plain", "")
            if txt:
                for line in txt.strip().splitlines()[:6]:
                    log(f"  [{job_name}] {line}")

    client = NotebookClient(
        nb,
        timeout=timeout_seconds,
        kernel_name="python3",
        resources={"metadata": {"path": str(SOURCE_NB.parent)}},
        on_cell_start=on_cell_start,
        on_cell_executed=on_cell_executed,
    )

    start_time = time.time()
    execution_error = None
    duration = 0.0
    try:
        client.execute()
        duration = time.time() - start_time
        log(f"[{job_name}] NotebookClient completed successfully in {duration/60:.1f} min")
    except Exception as e:
        duration = time.time() - start_time
        execution_error = str(e)
        log(f"[{job_name}] Cell execution raised an exception after {duration/60:.1f} min: {e}")
        log(traceback.format_exc()[-1500:])

    # Save executed notebook (with all cell outputs and errors)
    if run_dir.exists():
        executed_nb_path = run_dir / "executed_notebook.ipynb"
        nbformat.write(nb, executed_nb_path)
        log(f"[{job_name}] Saved executed notebook to {executed_nb_path}")

    # Inspect results
    results_summary = {"run_id": run_id, "duration_seconds": duration, "error": execution_error}
    gate_file = run_dir / "artifacts" / "dynamics_gate.json"
    gate_passed = False
    if gate_file.exists():
        try:
            gate_data = json.loads(gate_file.read_text())
            results_summary["dynamics_gate"] = gate_data
            log(f"[{job_name}] Dynamics gate output:")
            for s, g in gate_data.items():
                p = g.get("passed", False)
                rel_imp = g.get("relative_improvement", 0.0)
                rel_h4 = g.get("relative_improvement_h4", 0.0)
                log(f"  Seed {s}: passed={p}, rel_improvement={rel_imp:+.4f}, rel_improvement_h4={rel_h4:+.4f}")
                if p:
                    gate_passed = True
        except Exception as ex:
            log(f"[{job_name}] Failed to parse dynamics_gate.json: {ex}")
    else:
        log(f"[{job_name}] No dynamics_gate.json found")

    report_file = run_dir / "reports" / "acceptance_report.json"
    if report_file.exists():
        try:
            report_data = json.loads(report_file.read_text())
            results_summary["acceptance_report"] = report_data
            log(f"[{job_name}] Acceptance report status: {report_data.get('status')}")
            for s, r in report_data.get("seeds", {}).items():
                p = r.get("adjoint_minus_critic_test", {})
                log(f"  Seed {s}: verdict={r.get('verdict')}, adjoint-critic={p.get('estimate', 'n/a')}, CI={p.get('ci_95', 'n/a')}")
            if "pooled" in report_data:
                pooled_diff = report_data["pooled"].get("adjoint_minus_critic_test", {})
                log(f"  POOLED: verdict={report_data['pooled'].get('verdict')}, adjoint-critic={pooled_diff.get('estimate', 'n/a')}, CI={pooled_diff.get('ci_95', 'n/a')}")
        except Exception as ex:
            log(f"[{job_name}] Failed to parse acceptance_report.json: ex={ex}")
    else:
        log(f"[{job_name}] No acceptance_report.json found")

    allocators_file = run_dir / "artifacts" / "allocator_summaries.json"
    if allocators_file.exists():
        try:
            alloc_data = json.loads(allocators_file.read_text())
            log(f"[{job_name}] Allocator summaries present for {len(alloc_data)} seed/split entries.")
        except Exception:
            pass

    success = (execution_error is None) and (report_file.exists())
    if success:
        (run_dir / "COMPLETE").touch()
        log(f"[{job_name}] Job marked COMPLETE.")
        update_status(job_name, "COMPLETED", results_summary)
    else:
        update_status(job_name, "FAILED", results_summary)

    return success, run_id, results_summary


def main() -> int:
    log("Starting B2 sequential job orchestrator on Colab L4...")
    log(f"Python: {sys.version}")
    log(f"Repo Dir: {REPO_DIR}")
    log(f"Drive Root: {DRIVE_ROOT}")

    # =========================================================================
    # Step 1: Execute Job 1 (B2 Seed 0 Probe)
    # =========================================================================
    job1_success, job1_run_id, job1_summary = execute_job(
        job_name="probe_seed0",
        seeds=[0],
        timeout_seconds=7200,
    )

    if not job1_success:
        log(f"ABORTING: Job 1 (probe_seed0, run_id={job1_run_id}) did not complete successfully.")
        log(f"Job 1 summary: {json.dumps(job1_summary, indent=2)}")
        update_status("orchestrator", "ABORTED_AFTER_JOB1", {"job1_run_id": job1_run_id})
        return 1

    log(f"Job 1 (probe_seed0) SUCCEEDED! Proceeding immediately to Job 2 (5-seed confirmatory run)...")

    # =========================================================================
    # Step 2: Execute Job 2 (B2 5-Seed Confirmatory)
    # =========================================================================
    job2_success, job2_run_id, job2_summary = execute_job(
        job_name="5seeds",
        seeds=[0, 1, 2, 3, 4],
        timeout_seconds=18000,
    )

    final_payload = {
        "job1": {"run_id": job1_run_id, "success": job1_success, "summary": job1_summary},
        "job2": {"run_id": job2_run_id, "success": job2_success, "summary": job2_summary},
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    final_summary_file = DRIVE_ROOT / "jobs" / "sequential_b2_final_summary.json"
    final_summary_file.write_text(json.dumps(final_payload, indent=2))
    log(f"Sequential execution finished. Final summary written to {final_summary_file}")
    update_status("orchestrator", "ALL_JOBS_COMPLETED" if job2_success else "JOB2_FAILED", final_payload)

    return 0 if job2_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
