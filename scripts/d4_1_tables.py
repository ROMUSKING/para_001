#!/usr/bin/env python3
"""Print the D4-1 result tables (Markdown) from an imported run directory.

    python scripts/d4_1_tables.py results/runs/d4_1_learned_critics_<id>

Everything comes from the run's files. The primary rules are re-derived from ``artifacts/validation_work_precision.parquet`` with the package's own
bootstrap and compared with ``reports/acceptance_report.json`` and ``figures/comparisons.csv``; the script stops if they differ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adjointrwm.domains import critics as cr  # noqa: E402

TARGETS = ("gap10%", "gap3%", "gap1%")
ARMS = tuple(cr.ARMS)


def ci(row) -> str:
    return f"{row['ratio_of_geometric_means']:.2f} [{np.exp(row['ci_low']):.2f}, {np.exp(row['ci_high']):.2f}]"


def gate_table(report: dict) -> str:
    lines = ["| Cell | Head width | Direct saturated | Estimator floor met (seeds of 5) | R0 | RC1 | RC1-L | RC2-L | RC3 | RC3-L | RT-L | Exit class (primary; semilog; staircase) |",
             "|---|---:|---|---:|---|---|---|---|---|---|---|---|"]
    for cell, r in report["rules"].items():
        lines.append(f"| {cell} | {report['widths'][cell]['width']} | {r['saturated']} | {r['estimator_floor_met_seeds']} | {r['R0']} | {r['RC1']} | {r['RC1L']} | {r['RC2L']} | "
                     f"{r['RC3']} | {r['RC3L']} | {r['RTL']} | {r['exit_class']}; {r['exit_class_semilog']}; {r['exit_class_staircase']} |")
    return "\n".join(lines)


def width_table(widths: pd.DataFrame) -> str:
    lines = ["| Cell | Head width | Direct width | Direct compute (lookup price, θ = 0.95) | Co-state compute (lookup price) | Direct val RMSE | Co-state val RMSE |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in widths.iterrows():
        lines.append(f"| {r['cell']} | {r['width']} | {r['direct_width']} | {r['direct_lookup_compute']:.0f} | {r['costate_lookup_compute']:.0f} | "
                     f"{r['direct_val_rmse_log10']:.3f} | {r['costate_val_rmse_log10']:.3f} |")
    return "\n".join(lines)


def training_table(training: pd.DataFrame) -> str:
    lines = ["| Cell | Arm | Head parameters | Estimator parameters | Head val RMSE (log10) | Within-state Spearman | Estimator error vs exact co-state (mean [min, max]) | Steps per interval | Lookup price as a fraction of it |",
             "|---|---|---:|---:|---:|---:|---|---:|---:|"]
    for (cell, arm), g in training.groupby(["cell", "arm"], sort=False):
        err = g["estimator_val_error_true_costate"].dropna()
        est = f"{err.mean():.3f} [{err.min():.3f}, {err.max():.3f}]" if len(err) else ""
        lines.append(f"| {cell} | {arm} | {int(g['head_parameters'].iloc[0])} | {int(g['estimator_parameters'].iloc[0])} | {g['head_val_rmse_log10'].mean():.3f} | "
                     f"{g['head_val_spearman_within_state'].mean():.3f} | {est} | {g['steps_per_interval'].iloc[0]:.3f} | {g['lookup_price_scale'].iloc[0]:.2f} |")
    return "\n".join(lines)


def comparison_table(comparisons: pd.DataFrame, cell: str, price: str, pairs: list[tuple[str, str]]) -> str:
    lines = [f"| Comparison (ratio of geometric-mean compute, 95 % interval) | " + " | ".join(t[3:] + " of the gap left" for t in TARGETS) + " | Seeds favouring the first (gap3%) |", "|---|---|---|---|---|"]
    sub = comparisons[(comparisons["cell"] == cell) & (comparisons["price"] == price) & (comparisons["method"] == "loglog")]
    for a, b in pairs:
        rows = {t: sub[(sub["a"] == a) & (sub["b"] == b) & (sub["target"] == t)] for t in TARGETS}
        if any(r.empty for r in rows.values()):
            continue
        lines.append(f"| {a} / {b} | " + " | ".join(ci(rows[t].iloc[0]) for t in TARGETS) + f" | {int(rows['gap3%'].iloc[0]['seeds_favouring_a'])} of 5 |")
    return "\n".join(lines)


def compute_table(extra: pd.DataFrame, price: str) -> str:
    lines = [f"| Cell | Arm | " + " | ".join(f"geometric-mean compute, {t[3:]} left" for t in TARGETS) + " | Ratio to uniform (gap3%) | Retained fraction of the cheap → cheap_adjoint headroom (gap3%, mean [min, max] over seeds) | Break-even instances (gap3%) |",
             "|---|---|---:|---:|---:|---:|---|---:|"]
    for (cell, arm), g in extra[extra["price"] == price].groupby(["cell", "arm"], sort=False):
        by = g.set_index("target")
        kept = by.loc["gap3%"]
        retained = "" if pd.isna(kept["retained_cheap_to_cheap_adjoint_mean"]) else f"{kept['retained_cheap_to_cheap_adjoint_mean']:.2f} [{kept['retained_min']:.2f}, {kept['retained_max']:.2f}]"
        be = "" if pd.isna(kept["break_even_instances_mean_over_seeds"]) else f"{kept['break_even_instances_mean_over_seeds']:.0f}"
        lines.append(f"| {cell} | {arm} | " + " | ".join(f"{by.loc[t, 'geometric_mean_compute']:.0f}" for t in TARGETS) + f" | {kept['ratio_to_uniform']:.2f} | {retained} | {be} |")
    return "\n".join(lines)


def reference_table(frame: pd.DataFrame, cells: list[str]) -> str:
    lines = ["| Cell | Reference | " + " | ".join(f"median compute, {t[3:]} left" for t in TARGETS) + " |", "|---|---|---:|---:|---:|"]
    sub = frame[(frame["method"] == "loglog") & frame["policy"].isin(["uniform_pass", "cheap", "cheap_adjoint", "adjoint", "adjoint_free"])]
    for cell in cells:
        for policy in ("uniform_pass", "cheap", "cheap_adjoint", "adjoint", "adjoint_free"):
            g = sub[(sub["cell"] == cell) & (sub["policy"] == policy)]
            lines.append(f"| {cell} | {policy} | " + " | ".join(f"{g[g['target'] == t]['compute'].median():.0f}" for t in TARGETS) + " |")
    return "\n".join(lines)


def recompute(frame: pd.DataFrame, report: dict, comparisons: pd.DataFrame, cfg: dict, instances: pd.DataFrame) -> None:
    """Re-derive the primary comparisons of C against D and R from the per-instance parquet and assert they equal the stored intervals and flags."""
    seeds = list(cfg["seeds"])
    for cell in report["rules"]:
        sub = frame[frame["cell"] == cell]
        names = sorted(instances[instances["cell"] == cell]["name"])        # zero-padded names: sorted order is the notebook's instance order
        cap = float(cfg["cap_in_finest_grids"] * int(instances[instances["cell"] == cell]["n_fine"].iloc[0]))
        flags = {}
        for price in ("real", "lookup"):
            for other in ("direct", "costate_randomised"):
                rows = []
                for t in TARGETS:
                    a = cr.log_compute_matrix(sub, [f"costate_critic|s{s}|{price}" for s in seeds], names, t, "loglog", cap)
                    b = cr.log_compute_matrix(sub, [f"{other}|s{s}|{price}" for s in seeds], names, t, "loglog", cap)
                    got = cr.two_stage_bootstrap(a - b, cfg["bootstrap_resamples"], seed=0)
                    stored = comparisons[(comparisons["cell"] == cell) & (comparisons["price"] == price) & (comparisons["method"] == "loglog")
                                         & (comparisons["a"] == "costate_critic") & (comparisons["b"] == other) & (comparisons["target"] == t)].iloc[0]
                    assert abs(got["ci_low"] - stored["ci_low"]) < 1e-9 and abs(got["ci_high"] - stored["ci_high"]) < 1e-9, (cell, price, other, t)
                    rows.append(got)
                flags[(price, other)] = (cr.two_adjacent([r["ci_high"] < 0 for r in rows]), cr.within_margin(rows, cfg["equivalence_margin"]))
        r = report["rules"][cell]
        assert flags[("real", "direct")][0] == r["RC1"] and flags[("lookup", "direct")][0] == r["RC1L"] and flags[("lookup", "direct")][1] == r["RC2L"], cell
        assert flags[("real", "costate_randomised")][0] == r["RC3"] and flags[("lookup", "costate_randomised")][0] == r["RC3L"], cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir
    report = json.loads((run / "reports/acceptance_report.json").read_text())
    cfg = json.loads((run / "config/run_config.json").read_text())["config"]
    frame = pd.read_parquet(run / "artifacts/validation_work_precision.parquet")
    comparisons = pd.read_csv(run / "figures/comparisons.csv")
    extra = pd.read_csv(run / "figures/retained_and_break_even.csv")
    training = pd.read_csv(run / "artifacts/training_summary.csv")
    widths = pd.read_csv(run / "artifacts/width_selection.csv")
    recompute(frame, report, comparisons, cfg, pd.read_parquet(run / "config/instances_validation.parquet"))
    cells = list(report["rules"])
    print("## Gate summary and exit class per cell\n")
    print(gate_table(report))
    print("\n## Width selection (seed 0, tuning instances, lookup price)\n")
    print(width_table(widths))
    print("\n## Training: head fit, estimator quality and price per interval (seeds pooled)\n")
    print(training_table(training))
    core = [("costate_critic", "direct"), ("costate_critic", "costate_randomised"), ("teacher_feature", "direct"), ("costate_critic", "uniform_pass"), ("direct", "uniform_pass"),
            ("costate_critic", "cheap"), ("direct", "cheap"), ("costate_critic", "cheap_adjoint"), ("direct", "cheap_adjoint"), ("teacher_feature", "cheap_adjoint"),
            ("teacher_feature", "uniform_pass"), ("costate_critic", "adjoint_free"), ("direct", "adjoint_free")]
    for cell in cells:
        for price in ("real", "lookup"):
            print(f"\n## Comparisons, {cell}, {'real price' if price == 'real' else 'HYPOTHETICAL lookup price'} (primary method log-log)\n")
            print(comparison_table(comparisons, cell, price, core))
    for price in ("real", "lookup"):
        print(f"\n## Compute, retained headroom and break-even, {'real price' if price == 'real' else 'HYPOTHETICAL lookup price'}\n")
        print(compute_table(extra, price))
    print("\n## References (median compute over validation instances, real price)\n")
    print(reference_table(frame, cells))


if __name__ == "__main__":
    main()
