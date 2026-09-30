#!/usr/bin/env python3
"""Print the dynamics-parity result tables (Markdown) from an imported run directory.

    python scripts/dynamics_parity_tables.py results/runs/dynamics_parity_<id> [--pilot-run results/runs/droid100_adjoint_20260929T070629Z]

Everything comes from the run's files. The script stops if the CSV and the JSON disagree with the acceptance report, if a row's relative improvement
does not recompute from its two RMSEs, or if the config hash does not verify. The pilot's own test figures (for the cross-split row) come from the
pilot's committed ``artifacts/dynamics_evaluation.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

HORIZON_ROWS = (("v1", "full", "validation"), ("v2", "base", "validation"), ("v2", "full", "validation"), ("v1", "full", "train"), ("v2", "base", "train"))
LABELS = {"v1": "v1", "v2": "v2"}


def key(row: dict) -> tuple:
    return (row["checkpoint"], row["prediction_mode"], row["split"], bool(row["amp"]))


def load(run: Path) -> tuple[dict, list[dict]]:
    report = json.loads((run / "reports/acceptance_report.json").read_text())
    config = json.loads((run / "config/run_config.json").read_text())
    recomputed = hashlib.sha256(json.dumps(config["config"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert config["config_sha256"] == recomputed == report["config_hash"], "config hash does not verify"
    rows = report["rows"]
    assert json.loads((run / "artifacts/dynamics_parity.json").read_text()) == rows, "artifacts/dynamics_parity.json differs from the report's rows"
    with open(run / "artifacts/dynamics_parity.csv", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == len(rows)
    for csv_row, row in zip(table, rows):
        assert (csv_row["checkpoint"], csv_row["prediction_mode"], csv_row["split"], csv_row["amp"] == "True", int(csv_row["windows"])) == (*key(row), row["windows"])
        for field in ("model_rmse", "persistence_rmse", "relative_improvement"):
            assert float(csv_row[field]) == row[field], (field, csv_row, row)
    for row in rows:
        assert abs((row["persistence_rmse"] - row["model_rmse"]) / row["persistence_rmse"] - row["relative_improvement"]) < 1e-12, row
        assert row["passed"] == (row["relative_improvement"] >= row["required_margin"])
    return report, rows


def rmse_table(rows: list[dict], amp: bool) -> str:
    out = ["| Checkpoint | Mode | Split | Windows | Model RMSE | Persistence RMSE | Relative improvement | Passes 2 % |", "|---|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        if bool(r["amp"]) is amp:
            out.append(f"| {r['checkpoint']} | {r['prediction_mode']} | {r['split']} | {r['windows']} | {r['model_rmse']:.4f} | {r['persistence_rmse']:.4f} | "
                       f"{r['relative_improvement']:+.3f} | {'yes' if r['passed'] else 'no'} |")
    return "\n".join(out)


def horizon_table(rows: list[dict], pilot: dict | None) -> str:
    by = {key(r): r for r in rows}
    out = ["| Checkpoint, mode, split | Step 1 | Step 2 | Step 3 | Step 4 |", "|---|---:|---:|---:|---:|"]
    for checkpoint, mode, split in HORIZON_ROWS:
        r = by[(checkpoint, mode, split, True)]
        out.append(f"| {checkpoint}, `{mode}`, {split} | " + " | ".join(f"{x:+.3f}" for x in r["improvement_by_horizon"]) + " |")
    if pilot is not None:
        model, persistence = pilot["test"]["full_rmse_by_horizon"], pilot["test"]["persistence_rmse_by_horizon"]
        out.append("| pilot checkpoint, `full`, **test** (the pilot's committed `dynamics_evaluation.json`) | " + " | ".join(f"{(p - m) / p:+.3f}" for m, p in zip(model, persistence)) + " |")
    return "\n".join(out)


def anchor_table(report: dict) -> str:
    out = ["| Anchor | Recomputed | Stored by the original run | Absolute difference | Tolerance |", "|---|---:|---:|---:|---:|"]
    for name, a in report["anchors"].items():
        out.append(f"| {name} | {a['recomputed']!r} | {a['stored']!r} | {a['abs_diff']!r} | {a['atol']} |")
    return "\n".join(out)


def cross_split(rows: list[dict], pilot: dict | None) -> str:
    by = {key(r): r for r in rows}
    train, val = by[("v1", "full", "train", True)], by[("v1", "full", "validation", True)]
    line = (f"pilot checkpoint, full mode: model RMSE {train['model_rmse']:.4f} (train), {val['model_rmse']:.4f} (validation)"
            f"; persistence RMSE {train['persistence_rmse']:.4f}, {val['persistence_rmse']:.4f}; relative improvement {train['relative_improvement']:+.3f}, {val['relative_improvement']:+.3f}")
    if pilot is not None:
        gate = pilot["gate"]
        line += (f"\n  test (pilot's committed file): model RMSE {gate['full_rmse']:.4f}, persistence RMSE {gate['persistence_rmse']:.4f}, relative improvement {gate['relative_improvement']:+.3f}")
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pilot-run", type=Path, default=Path("results/runs/droid100_adjoint_20260929T070629Z"))
    args = parser.parse_args()
    report, rows = load(args.run_dir)
    pilot_path = args.pilot_run / "artifacts/dynamics_evaluation.json"
    pilot = json.loads(pilot_path.read_text()) if pilot_path.exists() else None
    bf16 = {key(r): r for r in rows if r["amp"]}
    fp32 = {key(r)[:3]: r for r in rows if not r["amp"]}
    worst = max(abs(bf16[k]["model_rmse"] - fp32[k[:3]]["model_rmse"]) for k in bf16)
    print(f"run {report['run_id']} | status {report['status']} | anchors_ok {report['anchors_ok']} | test_split_read {report['test_split_read']} | commit {report['repo_commit'][:12]}\n")
    print("## Anchors\n"); print(anchor_table(report))
    print("\n## Horizon-mean RMSE against persistence, BF16\n"); print(rmse_table(rows, True))
    print(f"\nmax |BF16 - FP32| model RMSE over the pairs: {worst:.3e}")
    print("\n## Relative improvement by horizon step (BF16)\n"); print(horizon_table(rows, pilot))
    print("\n## Across splits\n"); print(cross_split(rows, pilot))
    print("\n## Appendix: FP32\n"); print(rmse_table(rows, False))


if __name__ == "__main__":
    main()
