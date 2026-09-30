"""Tests for scripts/dynamics_kfold_compare.py on synthetic run folders built with the study's own helper functions (random fixtures, not data)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from adjointrwm.eval import (
    classify_against_margin, cluster_bootstrap_relative_improvement, episode_error_table, pooled_relative_improvement, random_subset_gate_rate,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/dynamics_kfold_compare.py"
HORIZON = 4


def episodes(noise, seed):
    rng = np.random.default_rng(seed)
    ids, ctx, tgt = [], [], []
    for e in range(100):
        c = rng.normal(size=(6, 8, 5)); d = np.cumsum(rng.normal(scale=0.4, size=(6, HORIZON, 5)), axis=1)
        ctx.append(c); tgt.append(c[:, -1:, :] + d); ids += [f"ep{e:03d}"] * 6
    context, target = np.concatenate(ctx), np.concatenate(tgt)
    pred = target + np.random.default_rng(seed + 1).normal(scale=noise, size=target.shape)
    return episode_error_table(pred, target, context, ids, HORIZON)


def write_run(root: Path, regime: str, own_mode: str, noise: float, anchor_ok: bool = True) -> Path:
    table = episodes(noise, seed=3)                 # the same windows for both regimes; only the model's error differs
    run = root / f"dynamics_kfold_{regime}_x"
    (run / "reports").mkdir(parents=True); (run / "artifacts").mkdir()
    boot = cluster_bootstrap_relative_improvement(table, 300, 0)
    est = {"pooled": pooled_relative_improvement(table), "bootstrap": boot, "random_ten_episode_split_gate_rate": random_subset_gate_rate(table, 10, 300, 0, 0.02)}
    entry = {"status": "OK" if anchor_ok else "ANCHOR_FAILED", "own_mode": own_mode, "primary_class": classify_against_margin(boot["ci_low"], boot["ci_high"]) if anchor_ok else None,
             "estimates": {f"best|{own_mode}": est}}
    (run / "reports" / "acceptance_report.json").write_text(json.dumps({"regimes": {regime: entry}}))
    long = table.assign(regime=regime, kind="best", mode=own_mode, fold=[i % 5 for i in range(len(table))])
    long.to_csv(run / "artifacts" / "episode_errors.csv", index=False)
    return run


def run_script(*paths):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, paths), "--resamples", "500"], capture_output=True, text=True, timeout=120)


def test_the_script_prints_each_regimes_class_and_the_paired_difference_with_its_label(tmp_path):
    v2, pilot = write_run(tmp_path, "v2", "base", noise=0.2), write_run(tmp_path, "pilot", "full", noise=1.0)
    done = run_script(v2, pilot)
    assert done.returncode == 0, done.stderr
    assert "v2: status OK | primary class" in done.stdout and "pilot: status OK | primary class" in done.stdout and "-> V2_BETTER" in done.stdout and "WARNING" not in done.stdout
    swapped = run_script(write_run(tmp_path / "s", "v2", "base", noise=1.0), write_run(tmp_path / "s", "pilot", "full", noise=0.2))
    assert swapped.returncode == 0 and "-> PILOT_BETTER" in swapped.stdout
    equal = run_script(write_run(tmp_path / "e", "v2", "base", noise=0.5), write_run(tmp_path / "e", "pilot", "full", noise=0.5))
    assert equal.returncode == 0 and "-> NOT_DISTINGUISHED" in equal.stdout


def test_the_script_warns_when_an_anchor_failed_and_refuses_the_wrong_pair(tmp_path):
    bad = run_script(write_run(tmp_path / "a", "v2", "base", 0.3, anchor_ok=False), write_run(tmp_path / "a", "pilot", "full", 0.3))
    assert bad.returncode == 0 and "WARNING: at least one anchor failed" in bad.stdout and "primary class None" in bad.stdout
    wrong = run_script(write_run(tmp_path / "b", "pilot", "full", 0.3), write_run(tmp_path / "b2", "v2", "base", 0.3))            # the wrong order
    assert wrong.returncode != 0 and "expected a v2 run then a pilot run" in wrong.stderr


def test_the_script_refuses_a_table_that_does_not_reproduce_the_reports_estimate(tmp_path):
    v2, pilot = write_run(tmp_path, "v2", "base", 0.3), write_run(tmp_path, "pilot", "full", 0.3)
    report = json.loads((v2 / "reports/acceptance_report.json").read_text())
    report["regimes"]["v2"]["estimates"]["best|base"]["pooled"]["relative_improvement"] += 0.05
    (v2 / "reports/acceptance_report.json").write_text(json.dumps(report))
    done = run_script(v2, pilot)
    assert done.returncode != 0 and "does not reproduce the report's estimate" in done.stderr
