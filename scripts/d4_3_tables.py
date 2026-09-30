#!/usr/bin/env python3
"""Print the D4-3 result tables (Markdown) from an imported run directory.

    python scripts/d4_3_tables.py results/runs/d4_3_varying_goal_<id> [--d4-2 results/runs/d4_2_flop_scoring_<id>]

Everything comes from ``artifacts/validation_summary.json``, ``artifacts/validation_work_precision.parquet``,
``artifacts/theta_tuning.csv`` and ``reports/acceptance_report.json``; nothing is re-simulated. The frozen rules
(R3v, R4v) are re-applied to the stored summary for every interpolation method with the code the notebook used, and the
primary method is checked against the run's own report. The D4-2 run directory supplies the fixed-goal ratio
``cheap_adjoint / cheap`` for comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adjointrwm.domains import highdim as hd  # noqa: E402

DEFAULT_D4_2 = Path("results/runs/d4_2_flop_scoring_20260929T233556Z")
MEDIAN_ARMS = ("uniform_pass", "cheap", "cheap_adjoint", "cheap_adjoint_setup", "adjoint", "adjoint_free")
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def load(run_dir: Path) -> dict:
    with (run_dir / "artifacts/theta_tuning.csv").open(newline="") as f:
        tuning = list(csv.DictReader(f))
    return {"summary": json.loads((run_dir / "artifacts/validation_summary.json").read_text()),
            "report": json.loads((run_dir / "reports/acceptance_report.json").read_text()), "tuning": tuning,
            "frame": pd.read_parquet(run_dir / "artifacts/validation_work_precision.parquet")}


def cell_order(report: dict) -> list[str]:
    return [r["cell"] for r in report["cells"]]


def ratio(entry: dict) -> str:
    return f"{entry['ratio_of_geometric_means']:.2f} [{math.exp(entry['ci_low']):.2f}, {math.exp(entry['ci_high']):.2f}]"


def price_table(report: dict) -> str:
    lines = ["| Cell | m | CN step (FLOPs) | Cheap ÷ step-doubling | Cheap-adjoint ÷ step-doubling | Co-state table setup, per instance (CN steps) |", "|---|---:|---:|---:|---:|---:|"]
    for cell in cell_order(report):
        p = report["price_reports"][cell]
        m = next(r["m"] for r in report["cells"] if r["cell"] == cell)
        r = p["ratio_to_step_doubling"]
        lines.append(f"| {cell} | {m} | {p['flops_per_step']:,} | {r['cheap']:.3f} | {r['cheap_adjoint']:.3f} | {p['table_setup_steps']:.0f} |")
    return "\n".join(lines)


def compute_table(summary: dict, method: str, order: list[str]) -> str:
    lines = ["| Cell | Target | Median compute: uniform / cheap / cheap-adjoint / cheap-adjoint + setup / adjoint / adjoint-free | Cheap ÷ uniform | Cheap-adjoint ÷ uniform | Cheap-adjoint ÷ cheap [95 % CI] | Setup-charged ÷ cheap | Goal-local ÷ residual | Adjoint-free ÷ cheap-adjoint |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cell in order:
        for target in hd.TARGETS:
            e = summary["summary"][cell][method][target]
            d = e["differences"]
            medians = " / ".join(f"{e['median_compute'][a]:.0f}" for a in MEDIAN_ARMS)
            lines.append(f"| {cell} | {target[3:]} | {medians} | {ratio(d['cheap / uniform_pass'])} | {ratio(d['cheap_adjoint / uniform_pass'])} | "
                         f"{ratio(d['cheap_adjoint / cheap'])} | {ratio(d['cheap_adjoint_setup / cheap'])} | {ratio(d['goal_local / residual'])} | {ratio(d['adjoint_free / cheap_adjoint'])} |")
    return "\n".join(lines)


def rules_table(data: dict) -> str:
    summary, report = data["summary"], data["report"]
    methods = summary["methods"]
    per = {m: {cell: hd.cell_rules_varying_goal(blocks[m]) for cell, blocks in summary["summary"].items()} for m in methods}
    primary = summary["primary_method"]
    for row in report["cells"]:
        rules = per[primary][row["cell"]]
        assert row["R3v"] == rules["R3v_costate_weight_helps_cheap_estimator"] and row["R4v"] == rules["R4v_goal_aware_cheap_arm_beats_uniform"], \
            "the rule code does not reproduce the run's own report"
    lines = ["| Cell | R3v weight helps cheap | R4v cheap-adjoint beats uniform | D4-1 candidate | Setup-charged beats cheap | R3v semilog | R3v staircase | R4v semilog | R4v staircase |", "|---|---|---|---|---|---|---|---|---|"]
    for cell in cell_order(report):
        p = per[primary][cell]
        s = {m: per[m][cell] for m in methods}
        lines.append(f"| {cell} | {p['R3v_costate_weight_helps_cheap_estimator']} | {p['R4v_goal_aware_cheap_arm_beats_uniform']} | {p['candidate_regime_for_d4_1']} | "
                     f"{p['setup_charged_cheap_adjoint_beats_cheap']} | {s['semilog']['R3v_costate_weight_helps_cheap_estimator']} | {s['staircase']['R3v_costate_weight_helps_cheap_estimator']} | "
                     f"{s['semilog']['R4v_goal_aware_cheap_arm_beats_uniform']} | {s['staircase']['R4v_goal_aware_cheap_arm_beats_uniform']} |")
    return "\n".join(lines)


def fixed_goal_table(summary: dict, d4_2_dir: Path, order: list[str]) -> str:
    """``cheap_adjoint / cheap`` with one goal per cell (D4-2) next to a goal per instance (D4-3), log-log."""
    old = json.loads((d4_2_dir / "artifacts/validation_summary.json").read_text())["summary"]
    lines = ["| Cell | Target | One goal per cell (D4-2) [95 % CI] | A goal per instance (D4-3) [95 % CI] |", "|---|---|---|---|"]
    for cell in order:
        for target in hd.TARGETS:
            a = old[cell]["loglog"][target]["differences"]["cheap_adjoint / cheap"]
            b = summary["summary"][cell]["loglog"][target]["differences"]["cheap_adjoint / cheap"]
            lines.append(f"| {cell} | {target[3:]} | {ratio(a)} | {ratio(b)} |")
    return "\n".join(lines)


def instance_table(frame: pd.DataFrame, order: list[str]) -> str:
    """Per-instance distribution of ``cheap_adjoint / cheap`` (log-log, real price, unreached targets clipped at the cap)."""
    lines = ["| Cell | Target | Instances | Win rate of cheap-adjoint | Ratio quantiles 10 / 25 / 50 / 75 / 90 % | Largest ratio |", "|---|---|---:|---:|---|---:|"]
    sub = frame[(frame["method"] == "loglog") & (frame["decision_scale"] == 1.0) & (frame["policy"].isin(["cheap", "cheap_adjoint"]))]
    for cell in order:
        for target in hd.TARGETS:
            block = sub[(sub["cell"] == cell) & (sub["target"] == target)]
            wide = block.pivot_table(index="instance", columns="policy", values="compute", aggfunc="mean")
            ratios = (wide["cheap_adjoint"] / wide["cheap"]).to_numpy()
            q = np.quantile(ratios, QUANTILES)
            lines.append(f"| {cell} | {target[3:]} | {len(ratios)} | {np.mean(ratios < 1.0):.2f} | {' / '.join(f'{v:.2f}' for v in q)} | {ratios.max():.2f} |")
    return "\n".join(lines)


def censoring_table(summary: dict, order: list[str]) -> str:
    arms = ("uniform_pass", "cheap", "cheap_adjoint", "residual", "goal_local", "adjoint")
    lines = ["| Cell | Target | " + " | ".join(arms) + " |", "|---|---|" + "---:|" * len(arms)]
    worst = 1.0
    for cell in order:
        for target in hd.TARGETS:
            fr = summary["summary"][cell]["loglog"][target]["fraction_reached"]
            worst = min(worst, *(fr[a] for a in arms))
            lines.append(f"| {cell} | {target[3:]} | " + " | ".join(f"{fr[a]:.2f}" for a in arms) + " |")
    return "\n".join(lines) + f"\n\nLowest fraction of instances reaching a target: {worst:.2f}."


def tuning_table(tuning: list[dict]) -> str:
    thetas = [k for k in tuning[0] if k.startswith("theta_")]
    lines = ["| Cell | Arm | " + " | ".join(f"θ = {t[6:]}" for t in thetas) + " | Selected | At grid edge |", "|---|---|" + "---:|" * len(thetas) + "---:|---|"]
    for row in tuning:
        lines.append(f"| {row['cell']} | {row['arm']} | " + " | ".join(f"{float(row[t]):.0f}" for t in thetas) + f" | {row['selected']} | {row['selected_at_grid_edge']} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--d4-2", type=Path, default=DEFAULT_D4_2)
    args = parser.parse_args()
    data = load(args.run_dir)
    order, primary = cell_order(data["report"]), data["summary"]["primary_method"]
    print("## Scoring price (FLOPs, in CN steps)\n")
    print(price_table(data["report"]))
    print(f"\n## Validation: compute to reach each target ({primary} interpolation)\n")
    print(compute_table(data["summary"], primary, order))
    print("\n## Frozen rules, and the same rules under the sensitivity pricings\n")
    print(rules_table(data))
    print("\n## The co-state weight with one goal per cell (D4-2) and a goal per instance (D4-3), cheap-adjoint ÷ cheap\n")
    print(fixed_goal_table(data["summary"], args.d4_2, order))
    print("\n## Per-instance distribution of cheap-adjoint ÷ cheap\n")
    print(instance_table(data["frame"], order))
    print("\n## Fraction of validation instances reaching each target\n")
    print(censoring_table(data["summary"], order))
    print("\n## Marking fraction (tuning family)\n")
    print(tuning_table(data["tuning"]))


if __name__ == "__main__":
    main()
