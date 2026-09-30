#!/usr/bin/env python3
"""Print the D1-0b result tables (Markdown) from an imported run directory.

    python scripts/d1_0b_tables.py results/runs/d1_0b_forecast_sensing_<id>

Everything comes from ``artifacts/tuning_summary.json``, ``artifacts/validation_summary.json`` (absent if the endpoint-moves
gate failed on the tuning machines and stage B was not run), ``artifacts/validation_window_losses_primary.parquet`` and
``reports/acceptance_report.json``; nothing is re-simulated. The primary setting's areas and headroom are recomputed from the
per-window parquet with the package's own area code and asserted equal to the stored summary.
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
from adjointrwm.domains import sensor_forecast as sf  # noqa: E402

PRIMARY = "h5_L60"
ALL_POLICIES = list(sf.DEPLOYABLE_POLICIES) + [sf.ORACLE]


def ci(interval: list, digits: int = 3) -> str:
    return f"[{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def load(run_dir: Path) -> dict:
    read = lambda p: json.loads((run_dir / p).read_text())  # noqa: E731
    out = {"tuning": read("artifacts/tuning_summary.json"), "report": read("reports/acceptance_report.json")}
    if (run_dir / "artifacts/validation_summary.json").exists():
        out["summary"] = read("artifacts/validation_summary.json")
        out["losses"] = pd.read_parquet(run_dir / "artifacts/validation_window_losses_primary.parquet")
    return out


def stage_a_table(tuning: dict) -> str:
    lines = ["| Setting | Endpoint moves `r` (mean over tuning machines) | Best fixed | " + " | ".join(f"{n} area" for n in sf.FIXED_POLICIES) + " |",
             "|---|---:|---|" + "---:|" * len(sf.FIXED_POLICIES)]
    for key, v in tuning["settings"].items():
        lines.append(f"| {key} | {v['endpoint_moves_mean']:.3f} | {v['best_fixed']} | " + " | ".join(f"{v['fixed_areas'][n]:.4f}" for n in sf.FIXED_POLICIES) + " |")
    return "\n".join(lines)


def tuning_machine_table(tuning: dict) -> str:
    v = tuning["settings"][PRIMARY]
    lines = ["| Tuning machine | Windows | Endpoint moves `r` |", "|---|---:|---:|"]
    for m, r in v["endpoint_moves_per_machine"].items():
        lines.append(f"| {m} | {v['windows'][m]} | {r:.3f} |")
    return "\n".join(lines)


def gate_table(summary: dict) -> str:
    lines = ["| Setting | Best fixed | Endpoint moves `r` [95 % CI] | G0 (≥ 0.05) | Area: fixed | Area: oracle | Relative headroom [95 % CI] | G1 (≥ 0.15) | Note |", "|---|---|---|---|---:|---:|---|---|---|"]
    for key, s in summary["runs"].items():
        note = "vacuous (G0 failed)" if s["G1_vacuous"] else ("robust" if s["G1_robust"] else "not robust")
        lines.append(f"| {key} | {s['fixed']} | {s['endpoint_moves']['mean']:.3f} {ci(s['bootstrap']['endpoint_moves_ci'])} | {'pass' if s['G0_passes'] else 'fail'}"
                     f"{' (robust)' if s['G0_robust'] else ''} | {s['areas'][s['fixed']]:.4f} | {s['areas'][sf.REFERENCE]:.4f} | "
                     f"{s['relative_headroom']:.3f} {ci(s['bootstrap']['headroom_ci'])} | {'pass' if s['G1_passes'] else 'fail'} | {note} |")
    return "\n".join(lines)


def policy_table(run: dict) -> str:
    kinds = {"round_robin": "fixed", "top_volatility": "fixed, uncertainty-only", "top_sensitivity": "fixed, sensitivity-only",
             "top_weighted_volatility": "fixed, sensitivity × uncertainty", "recent_change": "dynamic, uncertainty-only",
             "recent_change_weighted": "dynamic, sensitivity × uncertainty", sf.ORACLE: "privileged"}
    lines = ["| Policy | Kind | Normalised excess area [95 % CI] | Headroom kept (fixed to oracle) [95 % CI] | Fidelity area (the oracle's objective) |", "|---|---|---|---|---:|"]
    for name in ALL_POLICIES:
        kept = ""
        if name in run["retained"]:
            kept = f"{run['retained'][name]:.2f} {ci(run['bootstrap']['retained_ci'][name], 2)}"
        best = " (best fixed)" if name == run["fixed"] else ""
        lines.append(f"| {name}{best} | {kinds[name]} | {run['areas'][name]:.4f} {ci(run['bootstrap']['areas_ci'][name], 4)} | {kept} | {run['fidelity_areas'][name]:.4f} |")
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
    primary = summary["runs"][PRIMARY]
    lines = ["| Machine | Windows | Endpoint moves `r` | Headroom (best fixed to oracle) |", "|---|---:|---:|---:|"]
    for m, info in primary["machines"].items():
        lines.append(f"| {m} | {info['windows']} | {primary['endpoint_moves']['per_machine'][m]:.3f} | {primary['per_machine_headroom'][m]:.3f} |")
    return "\n".join(lines)


def recompute_primary(losses: pd.DataFrame, budgets: list[int]) -> dict:
    per_machine = {}
    for machine, frame in losses.groupby("machine"):
        full = float(frame.groupby("window")["full_observation_loss"].first().mean())
        mean = frame.groupby(["policy", "budget"])["native_loss"].mean().unstack("budget")[budgets] - full
        per_machine[machine] = mean / full
    areas = {name: float(np.mean([sn.normalised_area(per_machine[m].loc[name].to_numpy(), budgets) for m in per_machine])) for name in per_machine[next(iter(per_machine))].index}
    curves = {name: np.mean([per_machine[m].loc[name].to_numpy() for m in per_machine], axis=0) for name in areas}
    return {"areas": areas, "curves": curves}


def curve_table(recomputed: dict, budgets: list[int]) -> str:
    lines = ["| Policy | " + " | ".join(f"k = {k}" for k in budgets) + " |", "|---|" + "---:|" * len(budgets)]
    for name in ALL_POLICIES:
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in recomputed["curves"][name]) + " |")
    return "\n".join(lines)


def facts(summary: dict, losses: pd.DataFrame, recomputed: dict) -> str:
    primary = summary["runs"][PRIMARY]
    per_machine_full = losses.groupby(["machine", "window"])["full_observation_loss"].first().groupby("machine").mean()
    hold = losses[(losses["policy"] == "round_robin") & (losses["budget"] == 0)].groupby("machine")["native_loss"].mean()
    windows = sum(m["windows"] for m in primary["machines"].values())
    better = [n for n in sf.DEPLOYABLE_POLICIES if recomputed["areas"][n] < recomputed["areas"][sf.ORACLE]]
    return "\n".join([
        f"- Validation windows (primary setting): {windows} over {len(primary['machines'])} machines.",
        f"- Mean native forecast error (standardised units squared, per channel-minute) with every channel observed: {per_machine_full.mean():.4f} (machine range {per_machine_full.min():.4f} to {per_machine_full.max():.4f}); hold only: {hold.mean():.4f} (range {hold.min():.4f} to {hold.max():.4f}).",
        f"- Endpoint moves `r` per machine: min {min(primary['endpoint_moves']['per_machine'].values()):.3f}, max {max(primary['endpoint_moves']['per_machine'].values()):.3f}; machines below 0.05: "
        f"{sum(v < 0.05 for v in primary['endpoint_moves']['per_machine'].values())} of {len(primary['machines'])}.",
        f"- Deployable policies with a lower normalised area than the privileged oracle (primary): {', '.join(better) if better else 'none'}.",
        f"- Headroom against the validation-best fixed policy: "
        f"{sn.headroom(min(recomputed['areas'][n] for n in sf.FIXED_POLICIES), recomputed['areas'][sf.REFERENCE]):.3f} "
        f"({min(sf.FIXED_POLICIES, key=lambda n: recomputed['areas'][n])}, area {min(recomputed['areas'][n] for n in sf.FIXED_POLICIES):.4f}); the tuning-chosen best fixed ({primary['fixed']}) has {recomputed['areas'][primary['fixed']]:.4f}.",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    data = load(args.run_dir)
    print("## Stage A: the endpoint-moves statistic and the fixed policies on the tuning machines\n")
    print(stage_a_table(data["tuning"]))
    print("\n## Tuning machines (primary setting)\n")
    print(tuning_machine_table(data["tuning"]))
    if "summary" not in data:
        print("\nStage B was not run (the endpoint-moves gate failed on the tuning machines).")
        return
    summary = data["summary"]
    primary = summary["runs"][PRIMARY]
    budgets = primary["budgets"]
    recomputed = recompute_primary(data["losses"], budgets)
    for name, area in recomputed["areas"].items():
        assert abs(area - primary["areas"][name]) < 1e-9, f"{name}: parquet area {area} differs from the stored summary {primary['areas'][name]}"
    assert abs(sn.headroom(recomputed["areas"][primary["fixed"]], recomputed["areas"][sf.REFERENCE]) - primary["relative_headroom"]) < 1e-9
    print("\n## Gates G0 and G1 and the pre-registered sensitivities (validation machines)\n")
    print(gate_table(summary))
    print("\n## Policies in the primary run\n")
    print(policy_table(primary))
    print("\n## Paired differences of normalised areas (G-VOI and G2 inputs)\n")
    print(paired_table(summary))
    print("\n## Machines (primary run)\n")
    print(machine_table(summary))
    print("\n## Mean excess forecast error over full observation, relative to the full-observation error, by budget (primary run, recomputed from the per-window parquet)\n")
    print(curve_table(recomputed, budgets))
    print("\n## Derived quantities quoted in the note\n")
    print(facts(summary, data["losses"], recomputed))


if __name__ == "__main__":
    main()
