#!/usr/bin/env python3
"""Execute Track B1 (Rival World Models Benchmark) on Colab L4 runtime:
Trains and evaluates AdjointRWM vs 4 rival families:
1. adjoint_rwm (reference)
2. dreamerv3_rssm
3. tdmpc2
4. dino_wm
5. vjepa2_ac
plus classical baselines (persistence, ridge) across 5 seeds on DROID-100.

Pre-seeds the 100-episode feature cache from Drive for instant startup.
Supports seamless resumption of existing runs.
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
SOURCE_NB = REPO_DIR / "notebooks/03-benchmarks/rival_world_models_droid100.ipynb"
SHARED_CACHE = DRIVE_ROOT / "runs/droid100_adjoint_v2_20260930T165409Z/cache/cache_manifest.json"
LOG_DIR = DRIVE_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MASTER_LOG = LOG_DIR / "b1_master.log"
STATUS_FILE = DRIVE_ROOT / "jobs" / "b1_status.json"


def log(msg: str) -> None:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(MASTER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def update_status(state: str, details: dict | None = None) -> None:
    payload = {
        "job": "rival_world_models_b1",
        "state": state,
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **(details or {}),
    }
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(STATUS_FILE)


def prepare_notebook(run_id: str, resume_run_id: str | None = None) -> nbformat.NotebookNode:
    nb = nbformat.read(SOURCE_NB, as_version=4)

    # Patch Cell 0: ensure it doesn't checkout or overwrite /content/para_001
    cell0 = nb.cells[0]
    if cell0.cell_type == "code":
        cell0_lines = []
        for line in cell0.source.splitlines():
            if "subprocess.run(['git', '-C', REPO_DIR, 'checkout'" in line:
                cell0_lines.append("    # git checkout skipped to preserve local repo")
            elif "subprocess.run(['git', '-C', REPO_DIR, 'fetch'" in line:
                cell0_lines.append("    # git fetch skipped")
            elif "subprocess.run(['git', 'clone'" in line:
                cell0_lines.append("    pass  # repo already exists")
            else:
                cell0_lines.append(line)
        cell0.source = "\n".join(cell0_lines)

    # Patch Cell 1: set RUN_ID or RESUME_RUN_ID explicitly, and skip drive.mount
    for cell in nb.cells:
        if cell.cell_type == "code" and "BenchmarkConfig" in cell.source:
            lines = cell.source.splitlines()
            new_lines = []
            for l in lines:
                if "drive.mount(" in l:
                    new_lines.append("    # drive.mount skipped (already mounted)")
                elif resume_run_id and l.strip().startswith("RESUME_RUN_ID = None"):
                    indent = l[:l.index("RESUME_RUN_ID =")]
                    new_lines.append(f"{indent}RESUME_RUN_ID = '{resume_run_id}'")
                elif l.strip().startswith("RUN_ID = 'droid100_rivals_' + SESSION_ID"):
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


def run_benchmark(resume_run_id: str | None = None) -> None:
    if resume_run_id:
        run_id = resume_run_id
        is_resumed = True
    else:
        session_ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"droid100_rivals_{session_ts}"
        is_resumed = False

    run_dir = DRIVE_ROOT / "runs" / run_id

    log(f"============================================================")
    log(f"Starting Milestone B1: Rival World Models Benchmark ({'RESUMING' if is_resumed else 'NEW'})")
    log(f"Run ID: {run_id}")
    log(f"Target Run Dir: {run_dir}")
    log(f"============================================================")

    update_status("PREPARING", {"run_id": run_id, "run_dir": str(run_dir), "resumed": is_resumed})

    try:
        nb = prepare_notebook(run_id, resume_run_id=resume_run_id)
        # timeout=None ensures nbclient never times out on long-running multi-seed training loops
        client = NotebookClient(
            nb,
            timeout=None,
            kernel_name="python3",
            resources={"metadata": {"path": "/content"}},
        )

        total_cells = len(nb.cells)
        log(f"Notebook prepared with {total_cells} cells. Starting execution client...")

        with client.setup_kernel():
            for idx, cell in enumerate(nb.cells):
                cell_id = cell.get("id", f"cell_{idx}")
                log(f"[Cell {idx+1}/{total_cells}] (id={cell_id}) Type: {cell.cell_type} ...")
                cell_start = time.time()
                update_status("RUNNING", {
                    "run_id": run_id,
                    "current_cell": idx + 1,
                    "total_cells": total_cells,
                    "cell_id": cell_id,
                })

                client.execute_cell(cell, idx)
                cell_elapsed = time.time() - cell_start
                log(f"[Cell {idx+1}/{total_cells}] Completed in {cell_elapsed:.1f}s")

                # Log cell outputs if any
                for out in cell.get("outputs", []):
                    if out.get("output_type") == "stream":
                        text = out.get("text", "").strip()
                        if text:
                            for tline in text.splitlines():
                                log(f"   | {tline}")
                    elif out.get("output_type") == "error":
                        ename = out.get("ename", "Error")
                        evalue = out.get("evalue", "")
                        log(f"   [ERROR] {ename}: {evalue}")

        # Save executed notebook to Drive
        exec_nb_path = run_dir / "executed_rival_benchmark.ipynb"
        nbformat.write(nb, str(exec_nb_path))
        log(f"Saved executed notebook to {exec_nb_path}")

        # Check acceptance report
        report_file = run_dir / "reports" / "acceptance_report.json"
        complete_file = run_dir / "COMPLETE"
        if report_file.exists() and complete_file.exists():
            log(f"Milestone B1 Benchmark COMPLETED successfully!")
            log(f"COMPLETE file written at: {complete_file}")
            update_status("COMPLETED", {
                "run_id": run_id,
                "status": "PASS",
                "run_dir": str(run_dir),
            })
        else:
            log(f"Milestone B1 finished cell execution but COMPLETE marker missing. Report exists: {report_file.exists()}")
            update_status("FINISHED_NO_MARKER", {
                "run_id": run_id,
                "run_dir": str(run_dir),
            })

    except Exception as e:
        tb = traceback.format_exc()
        log(f"FATAL EXCEPTION in Milestone B1 Benchmark: {e}\n{tb}")
        update_status("FAILED", {
            "run_id": run_id,
            "error": str(e),
            "traceback": tb,
        })
        raise


if __name__ == "__main__":
    resume_id = sys.argv[1] if len(sys.argv) > 1 else None
    run_benchmark(resume_id)
