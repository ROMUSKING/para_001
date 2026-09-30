#!/usr/bin/env python3
"""Print the D1-0 result tables (Markdown) from an imported run directory.

    python scripts/d1_0_tables.py results/runs/d1_0_sensor_opportunity_<id>

Everything comes from ``artifacts/validation_summary.json``, ``artifacts/detector_selection.json``,
``artifacts/best_fixed_tuning.json``, ``artifacts/validation_window_losses_primary.parquet`` and
``reports/acceptance_report.json``; nothing is re-simulated. The primary run's areas and headroom are recomputed
from the per-window parquet with the package's own area code and asserted equal to the stored summary, so the
tables cannot drift from the raw losses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adjointrwm.domains import sensor as sn  # noqa: E402


def load(run_dir: Path) -> dict:
    read = lambda p: json.loads((run_dir / p).read_text())  # noqa: E731
    return {"summary": read("artifacts/validation_summary.json"), "selection": read("artifacts/detector_selection.json"),
            "best_fixed": read("artifacts/best_fixed_tuning.json"), "report": read("reports/acceptance_report.json"),
            "losses": pd.read_parquet(run_dir / "artifacts/validation_window_losses_primary.parquet")}


def ci(interval: list, digits: int = 3) -> str:
    return f"[{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def recompute_primary(losses: pd.DataFrame, budgets: list[int]) -> dict:
    """Mean over machines of each policy's normalised area, from the per-window losses."""
    per_machine = {}
    for machine, frame in losses.groupby("machine"):
        mean = frame.groupby(["policy", "budget"])["loss"].mean().unstack("budget")[budgets]
        hold = float(mean.loc[sn.REFERENCE, 0])
        per_machine[machine] = mean / hold
    areas = {name: float(np.mean([sn.normalised_area(per_machine[m].loc[name].to_numpy(), budgets) for m in per_machine]))
             for name in next(iter(per_machine.values())).index}
    curves = {name: np.mean([per_machine[m].loc[name].to_numpy() for m in per_machine], axis=0) for name in areas}
    return {"areas": areas, "curves": curves, "per_machine": per_machine}


def detector_table(selection: dict) -> str:
    lines = ["| Floor | Ridge | Mean F1 (tuning machines) | Median alarm rate on unlabelled minutes | Selected |", "|---:|---:|---:|---:|---|"]
    chosen = selection["selected"]
    for row in selection["grid"]:
        rate = float(np.median(list(row["per_machine_alarm_rate_normal"].values())))
        lines.append(f"| {row['floor']} | {row['ridge']:g} | {row['mean_f1']:.3f} | {rate:.3f} | {'yes' if (row['floor'], row['ridge']) == (chosen['floor'], chosen['ridge']) else ''} |")
    return "\n".join(lines)


def fixed_table(best_fixed: dict) -> str:
    lengths = list(best_fixed["areas"])
    lines = ["| Fixed policy | " + " | ".join(f"L = {n}" for n in lengths) + " |", "|---|" + "---:|" * len(lengths)]
    for name in sn.FIXED_POLICIES:
        cells = " | ".join(f"{best_fixed['areas'][n][name]:.4f}{' (selected)' if best_fixed['selected'][n] == name else ''}" for n in lengths)
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines)


def gate_table(summary: dict) -> str:
    lines = ["| Run | L | Objective | Best fixed | Area: fixed | Area: reference | Area: greedy oracle | Relative headroom [95 % CI] | G1 (≥ 0.15) | Robust |", "|---|---:|---|---|---:|---:|---:|---|---|---|"]
    for run, s in summary["runs"].items():
        a = s["areas"]
        lines.append(f"| {run} | {s['length']} | {s['objective']} | {s['fixed']} | {a[s['fixed']]:.4f} | {a[sn.REFERENCE]:.4f} | {a[sn.ORACLE]:.4f} | "
                     f"{s['relative_headroom']:.3f} {ci(s['bootstrap']['headroom_ci'])} | {'pass' if s['passes'] else 'fail'} | {'yes' if s['robust'] else 'no'} |")
    return "\n".join(lines)


def policy_table(run: dict) -> str:
    fixed = run["fixed"]
    lines = ["| Policy | Kind | Normalised area [95 % CI] | Headroom kept (fraction of fixed-to-reference) [95 % CI] |", "|---|---|---|---|"]
    kinds = {**{n: "fixed" for n in sn.FIXED_POLICIES}, **{n: "dynamic" for n in sn.DYNAMIC_POLICIES}, sn.ORACLE: "privileged greedy", sn.REFERENCE: "best-known reference"}
    for name in list(sn.DEPLOYABLE_POLICIES) + [sn.ORACLE, sn.REFERENCE]:
        kept = ""
        if name in run["retained"]:
            kept = f"{run['retained'][name]:.2f} {ci(run['bootstrap']['retained_ci'][name], 2)}"
        best = " (best fixed)" if name == fixed else ""
        lines.append(f"| {name}{best} | {kinds[name]} | {run['areas'][name]:.4f} {ci(run['bootstrap']['areas_ci'][name], 4)} | {kept} |")
    return "\n".join(lines)


def paired_table(summary: dict) -> str:
    runs = list(summary["runs"])
    per_run = {}
    for run in runs:
        s = summary["runs"][run]
        labelled = {}
        for pair, interval in s["bootstrap"]["paired_area_difference_ci"].items():
            first, second = pair.split(" - ")
            label = f"{first} - {'best fixed' if second == s['fixed'] else second}"
            labelled[label] = f"{s['areas'][first] - s['areas'][second]:+.4f} {ci(interval, 4)}"
        per_run[run] = labelled
    labels = list(dict.fromkeys(label for run in runs for label in per_run[run]))
    lines = ["| Paired difference of normalised areas (negative: first is better) | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
    for label in labels:
        lines.append(f"| {label} | " + " | ".join(per_run[run].get(label, "") for run in runs) + " |")
    return "\n".join(lines)


def machine_table(summary: dict) -> str:
    primary = summary["runs"]["primary"]
    lines = ["| Machine | Windows | Windows with a labelled anomaly | Headroom (best fixed to reference) | Fully observed detector: F1 | Alarm rate on unlabelled minutes |", "|---|---:|---:|---:|---:|---:|"]
    for machine, info in primary["machines"].items():
        f = summary["full_observation_f1"][machine]
        lines.append(f"| {machine} | {info['windows']} | {info['anomalous_windows']} | {primary['per_machine_headroom'][machine]:.3f} | {f['f1']:.3f} | {f['alarm_rate_normal']:.3f} |")
    return "\n".join(lines)


def curve_table(recomputed: dict, budgets: list[int]) -> str:
    lines = ["| Policy | " + " | ".join(f"k = {k}" for k in budgets) + " |", "|---|" + "---:|" * len(budgets)]
    for name in list(sn.DEPLOYABLE_POLICIES) + [sn.ORACLE, sn.REFERENCE]:
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in recomputed["curves"][name]) + " |")
    return "\n".join(lines)


def f1_table(run: dict, name_run: str) -> str:
    budgets = run["budgets"]
    lines = [f"| Policy ({name_run}) | " + " | ".join(f"k = {k}" for k in budgets) + " |", "|---|" + "---:|" * len(budgets)]
    for name in list(sn.DEPLOYABLE_POLICIES) + [sn.ORACLE]:
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in run["f1_by_budget"][name]) + " |")
    return "\n".join(lines)


def facts(summary: dict, recomputed: dict) -> str:
    """Derived quantities quoted in the research note."""
    primary = summary["runs"]["primary"]
    machines = primary["machines"]
    windows = sum(m["windows"] for m in machines.values())
    anomalous = sum(m["anomalous_windows"] for m in machines.values())
    full = summary["full_observation_f1"]
    rates = [f["alarm_rate_normal"] for f in full.values()]
    areas = recomputed["areas"]
    best_validation_fixed = min(sn.FIXED_POLICIES, key=lambda n: areas[n])
    f1 = primary["f1_by_budget"]
    lines = [f"- Validation windows: {windows}, of which {anomalous} contain a labelled anomalous minute ({anomalous / windows:.3f}).",
             f"- Fully observed detector, mean per-machine F1: {np.mean([f['f1'] for f in full.values()]):.3f} (range {min(f['f1'] for f in full.values()):.3f} to {max(f['f1'] for f in full.values()):.3f}).",
             f"- Fully observed detector, alarm rate on unlabelled minutes: median {np.median(rates):.3f}, range {min(rates):.3f} to {max(rates):.3f}; machines above 0.10: {sum(r > 0.10 for r in rates)} of {len(rates)}.",
             f"- Pooled detection F1 at k = 0 (hold) and k = 38 (full observation): {f1['round_robin'][0]:.3f} and {f1['round_robin'][-1]:.3f}.",
             f"- Best fixed policy on validation by area: {best_validation_fixed} ({areas[best_validation_fixed]:.4f}); the tuning-chosen best fixed ({primary['fixed']}) has {areas[primary['fixed']]:.4f}. "
             f"Headroom against the validation-best fixed policy: {sn.headroom(areas[best_validation_fixed], areas[sn.REFERENCE]):.3f}.",
             f"- Greedy oracle share of the reference area: {areas[sn.ORACLE]:.4f} against {areas[sn.REFERENCE]:.4f}.",
             f"- Policies with a headroom-kept fraction whose interval excludes 0 in the primary run: "
             f"{', '.join(n for n in sn.DYNAMIC_POLICIES if primary['bootstrap']['retained_ci'][n][0] > 0)}."]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    data = load(args.run_dir)
    summary = data["summary"]
    primary = summary["runs"]["primary"]
    budgets = primary["budgets"]
    recomputed = recompute_primary(data["losses"], budgets)
    for name, area in recomputed["areas"].items():
        assert abs(area - primary["areas"][name]) < 1e-9, f"{name}: parquet area {area} differs from the stored summary {primary['areas'][name]}"
    head = sn.headroom(recomputed["areas"][primary["fixed"]], recomputed["areas"][sn.REFERENCE])
    assert abs(head - primary["relative_headroom"]) < 1e-9
    print("## Detector choice on the tuning machines\n")
    print(detector_table(data["selection"]))
    print("\n## Best fixed allocation on the tuning machines (normalised area, lower is better)\n")
    print(fixed_table(data["best_fixed"]))
    print("\n## Gate G1 and the pre-registered sensitivities (validation machines)\n")
    print(gate_table(summary))
    print("\n## Policies in the primary run\n")
    print(policy_table(primary))
    print("\n## Paired differences of normalised areas\n")
    print(paired_table(summary))
    print("\n## Machines (primary run)\n")
    print(machine_table(summary))
    print("\n## Mean normalised loss by budget (primary run, recomputed from the per-window parquet)\n")
    print(curve_table(recomputed, budgets))
    print("\n## Detection F1 against the dataset labels by budget, pooled over validation windows\n")
    print(f1_table(primary, "primary"))
    print("\n## Derived quantities quoted in the note\n")
    print(facts(summary, recomputed))


if __name__ == "__main__":
    main()
