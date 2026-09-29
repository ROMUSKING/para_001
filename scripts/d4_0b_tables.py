#!/usr/bin/env python3
"""Print the D4-0b result tables (Markdown) from an imported run directory.

    python scripts/d4_0b_tables.py results/runs/d4_0b_adaptivity_<id>

Everything comes from ``artifacts/validation_summary.json``, ``artifacts/theta_tuning.csv`` and
``reports/acceptance_report.json``; nothing is recomputed, so the tables in a research note can be
regenerated and diffed against it. Confidence intervals are the exponentials of the log-ratio
bootstrap intervals stored in the summary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TARGETS = ("gap10%", "gap3%", "gap1%")
POLICIES = ("uniform_pass", "residual", "goal_local", "adjoint")


def load(run_dir: Path) -> dict:
    return {
        "summary": json.loads((run_dir / "artifacts/validation_summary.json").read_text()),
        "report": json.loads((run_dir / "reports/acceptance_report.json").read_text()),
        "tuning": (run_dir / "artifacts/theta_tuning.csv").read_text().splitlines(),
    }


def tuning_table(lines: list[str]) -> str:
    header = lines[0].split(",")
    thetas = [h for h in header if h.startswith("theta_")]
    out = ["| Family | Policy | " + " | ".join(f"θ = {t[6:]}" for t in thetas) + " | Selected | At grid edge |",
           "|---|---|" + "---:|" * len(thetas) + "---:|---|"]
    for line in lines[1:]:
        row = dict(zip(header, line.split(",")))
        out.append(f"| {row['family']} | {row['kind']} | " + " | ".join(f"{float(row[t]):.0f}" for t in thetas) +
                   f" | {row['selected']} | {row['selected_at_grid_edge']} |")
    return "\n".join(out)


def ratio(entry: dict) -> str:
    return f"{entry['ratio_of_geometric_means']:.2f} [{math.exp(entry['ci_low']):.2f}, {math.exp(entry['ci_high']):.2f}]"


def real_ledger_table(summary: dict, families=("smooth", "sharp", "sharper")) -> str:
    lines = ["| Family | Target (gap left) | Median compute: uniform / residual / goal-local / co-state | Co-state ÷ uniform [95 % CI] | Residual ÷ uniform | Co-state ÷ residual [95 % CI] |",
             "|---|---|---|---|---|---|"]
    for fam in families:
        for target in TARGETS:
            e = summary["summary"][fam]["1"][target]
            m, d = e["median_compute"], e["differences"]
            lines.append(f"| {fam} | {target[3:]} | " + " / ".join(f"{m[p]:.0f}" for p in POLICIES) +
                         f" | {ratio(d['adjoint / uniform_pass'])} | {d['residual / uniform_pass']['ratio_of_geometric_means']:.2f} | {ratio(d['adjoint / residual'])} |")
    return "\n".join(lines)


def what_if_table(summary: dict, families=("smooth", "sharp", "sharper")) -> str:
    lines = ["| Family | Scoring price | Target | Co-state ÷ uniform [95 % CI] | Residual ÷ uniform | Co-state ÷ residual [95 % CI] | Co-state ÷ goal-local [95 % CI] |",
             "|---|---|---|---|---|---|---|"]
    for fam in families:
        for scale in ("0.25", "0"):
            for target in TARGETS:
                d = summary["summary"][fam][scale][target]["differences"]
                lines.append(f"| {fam} | ×{scale} | {target[3:]} | {ratio(d['adjoint / uniform_pass'])} | {d['residual / uniform_pass']['ratio_of_geometric_means']:.2f} | "
                             f"{ratio(d['adjoint / residual'])} | {ratio(d['adjoint / goal_local'])} |")
    return "\n".join(lines)


def cells_table(report: dict) -> str:
    lines = ["| Family | Scoring price | Co-state beats uniform | Co-state beats residual and goal-local | Candidate regime for D4-1 |", "|---|---|---|---|---|"]
    for c in report["cells"]:
        lines.append(f"| {c['family']} | ×{c['scoring_price_scale']:g} | {c['adjoint_beats_uniform_two_adjacent_targets']} | "
                     f"{c['adjoint_beats_residual_and_goal_local_two_adjacent_targets']} | {c['candidate_regime_for_d4_1']} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    data = load(args.run_dir)
    print("## Tuning (geometric-mean compute on the tuning family)\n")
    print(tuning_table(data["tuning"]))
    print("\n## Real ledger (scoring price x1), validation family\n")
    print(real_ledger_table(data["summary"]))
    print("\n## Hypothetical scoring prices, validation family\n")
    print(what_if_table(data["summary"]))
    print("\n## Frozen decision rule\n")
    print(cells_table(data["report"]))


if __name__ == "__main__":
    main()
