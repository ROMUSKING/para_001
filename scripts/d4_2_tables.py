#!/usr/bin/env python3
"""Print the D4-2 result tables (Markdown) from an imported run directory.

    python scripts/d4_2_tables.py results/runs/d4_2_flop_scoring_<id>

Everything comes from ``artifacts/validation_summary.json``, ``artifacts/theta_tuning.csv``,
``artifacts/hidden_size_selection.csv`` and ``reports/acceptance_report.json``; no simulation is re-run, so the
tables in a research note can be regenerated and diffed against it. Confidence intervals are the exponentials of the
log-ratio bootstrap intervals stored in the summary. The frozen rules (R1, R3) are re-applied to the stored summary
for every interpolation method with the code the notebook used, and the primary method is checked against the run's
own report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adjointrwm.domains import highdim as hd  # noqa: E402

ARMS = ("uniform_pass", "amortised", "cheap", "cheap_adjoint", "adjoint", "adjoint_free")


def load(run_dir: Path) -> dict:
    with (run_dir / "artifacts/theta_tuning.csv").open(newline="") as f:
        tuning = list(csv.DictReader(f))
    with (run_dir / "artifacts/hidden_size_selection.csv").open(newline="") as f:
        hidden = list(csv.DictReader(f))
    return {"summary": json.loads((run_dir / "artifacts/validation_summary.json").read_text()),
            "report": json.loads((run_dir / "reports/acceptance_report.json").read_text()), "tuning": tuning, "hidden": hidden}


def ratio(entry: dict) -> str:
    return f"{entry['ratio_of_geometric_means']:.2f} [{math.exp(entry['ci_low']):.2f}, {math.exp(entry['ci_high']):.2f}]"


def cell_order(report: dict) -> list[str]:
    """Cells in the order the run reported them (the JSON files store keys alphabetically)."""
    return [r["cell"] for r in report["cells"]]


def price_table(report: dict) -> str:
    lines = ["| Cell | m | CN step (FLOPs) | Amortised (steps per interval) | Amortised ÷ step-doubling | Cheap ÷ step-doubling | Cheap-adjoint ÷ step-doubling | Indicator ÷ step-doubling | Co-state table setup (steps, one-off) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    m_of = {r["cell"]: r["m"] for r in report["cells"]}
    for cell in cell_order(report):
        p = report["price_reports"][cell]
        r = p["ratio_to_step_doubling"]
        lines.append(f"| {cell} | {m_of[cell]} | {p['flops_per_step']:,} | {p['steps_per_interval']['amortised']:.2f} | {r['amortised']:.3f} | {r['cheap']:.3f} | "
                     f"{r['cheap_adjoint']:.3f} | {r['indicator']:.3f} | {p['table_setup_steps']:.0f} |")
    return "\n".join(lines)


def scorer_table(hidden: list[dict], report: dict) -> str:
    lines = ["| Cell | Hidden sizes tried: tuning compute (geometric mean) | Chosen | Held-out RMSE on training-family instances (log10) | Epochs | Training cost (CN steps) |", "|---|---|---:|---:|---:|---:|"]
    for cell in cell_order(report):
        chosen = report["selected_hidden"][cell]
        rows = [r for r in hidden if r["cell"] == cell]
        tried = ", ".join(f"{r['hidden']}: {float(r['tuning_geometric_mean_compute']):.0f}" for r in rows)
        pick = next(r for r in rows if int(r["hidden"]) == int(chosen))
        steps = report["one_off_costs_and_break_even"][cell]["amortised_training_steps"]
        lines.append(f"| {cell} | {tried} | {chosen} | {float(pick['val_rmse_log10']):.2f} | {pick['epochs_run']} | {steps:,.0f} |")
    return "\n".join(lines)


def compute_table(summary: dict, method: str, order: list[str] | None = None) -> str:
    lines = ["| Cell | Target | Median compute: uniform / amortised / cheap / cheap-adjoint / adjoint / adjoint-free | Amortised ÷ uniform [95 % CI] | Cheap ÷ uniform | Cheap-adjoint ÷ uniform | Cheap-adjoint ÷ cheap | Adjoint ÷ uniform | Adjoint-free ÷ uniform |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cell in order or list(summary["summary"]):
        methods = summary["summary"][cell]
        for target in hd.TARGETS:
            e = methods[method][target]
            d = e["differences"]
            medians = " / ".join(f"{e['median_compute'][a]:.0f}" for a in ARMS)
            lines.append(f"| {cell} | {target[3:]} | {medians} | {ratio(d['amortised / uniform_pass'])} | {ratio(d['cheap / uniform_pass'])} | "
                         f"{ratio(d['cheap_adjoint / uniform_pass'])} | {ratio(d['cheap_adjoint / cheap'])} | {ratio(d['adjoint / uniform_pass'])} | {ratio(d['adjoint_free / uniform_pass'])} |")
    return "\n".join(lines)


def pairs_table(summary: dict, method: str, order: list[str] | None = None) -> str:
    lines = ["| Cell | Target | Amortised ÷ cheap [95 % CI] | Amortised ÷ cheap-adjoint | Adjoint-free ÷ amortised | Adjoint ÷ amortised | Residual ÷ uniform | Goal-local ÷ uniform | Indicator ÷ uniform (censored at the cap) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cell in order or list(summary["summary"]):
        for target in hd.TARGETS:
            d = summary["summary"][cell][method][target]["differences"]
            lines.append(f"| {cell} | {target[3:]} | {ratio(d['amortised / cheap'])} | {ratio(d['amortised / cheap_adjoint'])} | {ratio(d['adjoint_free / amortised'])} | "
                         f"{ratio(d['adjoint / amortised'])} | {ratio(d['residual / uniform_pass'])} | {ratio(d['goal_local / uniform_pass'])} | {ratio(d['indicator / uniform_pass'])} |")
    return "\n".join(lines)


def rules_rows(summary: dict, method: str) -> dict:
    return {cell: hd.cell_rules(methods[method]) for cell, methods in summary["summary"].items()}


def rules_table(data: dict) -> str:
    summary, report = data["summary"], data["report"]
    methods = summary["methods"]
    per = {m: rules_rows(summary, m) for m in methods}
    primary = summary["primary_method"]
    for cell, row in ((r["cell"], r) for r in report["cells"]):        # the notebook's own report must equal the recomputation
        assert row["R1"] == per[primary][cell]["R1_amortised_beats_uniform"] and row["R3"] == per[primary][cell]["R3_costate_weight_helps_cheap_estimator"], \
            "the rule code does not reproduce the run's own report"
    lines = ["| Cell | R1 amortised beats uniform | R3 co-state weight helps the cheap estimator | D4-1 candidate | R1 semilog | R1 staircase | R3 semilog | R3 staircase |", "|---|---|---|---|---|---|---|---|"]
    for cell in cell_order(report):
        p = per[primary][cell]
        s = {m: per[m][cell] for m in methods}
        lines.append(f"| {cell} | {p['R1_amortised_beats_uniform']} | {p['R3_costate_weight_helps_cheap_estimator']} | {p['candidate_regime_for_d4_1']} | "
                     f"{s['semilog']['R1_amortised_beats_uniform']} | {s['staircase']['R1_amortised_beats_uniform']} | "
                     f"{s['semilog']['R3_costate_weight_helps_cheap_estimator']} | {s['staircase']['R3_costate_weight_helps_cheap_estimator']} |")
    return "\n".join(lines)


def headroom_table(report: dict) -> str:
    lines = ["| Cell | Retained headroom: 10 % | 3 % | 1 % | Amortised break-even instances | Cheap-adjoint break-even instances |", "|---|---:|---:|---:|---:|---:|"]

    def fmt(value, spec="{:.2f}"):
        return "n/a" if value is None else spec.format(value)

    for cell in cell_order(report):
        h = report["retained_headroom"][cell]
        b = report["one_off_costs_and_break_even"][cell]
        lines.append(f"| {cell} | {fmt(h['gap10%'])} | {fmt(h['gap3%'])} | {fmt(h['gap1%'])} | {fmt(b['amortised_break_even_instances'], '{:,.0f}')} | {fmt(b['cheap_adjoint_break_even_instances'], '{:.1f}')} |")
    return "\n".join(lines)


def tuning_table(tuning: list[dict]) -> str:
    thetas = [k for k in tuning[0] if k.startswith("theta_")]
    lines = ["| Cell | Arm | " + " | ".join(f"θ = {t[6:]}" for t in thetas) + " | Selected | At grid edge |", "|---|---|" + "---:|" * len(thetas) + "---:|---|"]
    for row in tuning:
        lines.append(f"| {row['cell']} | {row['arm']} | " + " | ".join(f"{float(row[t]):.0f}" for t in thetas) + f" | {row['selected']} | {row['selected_at_grid_edge']} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    data = load(args.run_dir)
    primary = data["summary"]["primary_method"]
    print("## Scoring price (FLOP-priced, in CN steps)\n")
    print(price_table(data["report"]))
    print("\n## Scorer selection and training cost\n")
    print(scorer_table(data["hidden"], data["report"]))
    print(f"\n## Validation: compute to reach each target at the real price ({primary} interpolation)\n")
    print(compute_table(data["summary"], primary, cell_order(data["report"])))
    print(f"\n## Validation: the learned scorer against the cheap arms and the free ceiling ({primary} interpolation)\n")
    print(pairs_table(data["summary"], primary, cell_order(data["report"])))
    print("\n## Frozen rules, and the same rules under the sensitivity pricings\n")
    print(rules_table(data))
    print("\n## Retained headroom and break-even instances\n")
    print(headroom_table(data["report"]))
    print("\n## Marking fraction and network size (tuning family)\n")
    print(tuning_table(data["tuning"]))


if __name__ == "__main__":
    main()
