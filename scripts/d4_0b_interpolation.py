#!/usr/bin/env python3
"""Post-hoc robustness check for a D4-0b run: remove the pass-quantisation of compute-to-target.

    python scripts/d4_0b_interpolation.py results/runs/d4_0b_adaptivity_<id> [--out <dir>]

``uniform_pass`` can only stop after a whole doubling of the grid, so its compute-to-target is
quantised by up to a factor 2, which flatters the adaptive policies. This script re-runs the run's
validation family with the run's own selected ``theta`` values, prices every policy's compute-to-target
by log-linear interpolation between the bracketing passes (symmetrically for all policies, see
``adjointrwm.domains.interpolated_compute_to_target``), and writes ``interpolated_summary.json`` and
``interpolated_work_precision.parquet`` to ``--out`` (default: ``<run>/diagnostics``).

It is exploratory and was designed after the main results were read. It needs no test data. It asserts that its
targets equal those stored in the run, so it cannot silently evaluate something else.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adjointrwm.domains import (
    AdaptiveTimeSteppingDomain,
    evaluate_work_precision,
    marking_policy,
    sample_family,
    uniform_pass_policy,
    work_precision_summary,
)

PAIRS = [("adjoint", "uniform_pass"), ("residual", "uniform_pass"), ("goal_local", "uniform_pass"),
         ("adjoint", "residual"), ("adjoint", "goal_local")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run = args.run_dir
    out = args.out or run / "diagnostics"
    config = json.loads((run / "config/run_config.json").read_text())["config"]
    report = json.loads((run / "reports/acceptance_report.json").read_text())
    theta = report["selected_theta"]
    stored = pd.read_parquet(run / "artifacts/validation_work_precision.parquet")
    domain = AdaptiveTimeSteppingDomain(objective_kind="bound")
    scales = tuple(config["decision_scales"])

    def targets(d, inst):
        floor = d.finest_objective(inst)
        start = d.objective(d.initial_state(inst), inst)
        return {f"gap{int(round(100 * phi))}%": floor + phi * (start - floor) for phi in config["gap_fractions"]}

    frames, summary = [], {}
    for family in config["families"]:
        instances = sample_family(family, config["validation_seed"], config["validation_instances"])
        policies = [uniform_pass_policy()] + [dataclasses.replace(marking_policy(kind, theta[f"{family}/{kind}"]), name=kind)
                                              for kind in config["kinds"]]
        frame = pd.concat([
            evaluate_work_precision(domain, [inst], policies, targets, compute_cap=config["cap_in_finest_grids"] * inst.n_fine,
                                    max_steps=config["max_passes"], random_draws=1, decision_scales=scales, interpolate=True)
            for inst in instances], ignore_index=True)
        # The targets must be exactly the ones the run used.
        mine = frame.drop_duplicates(["instance", "target"]).set_index(["instance", "target"])["target_value"]
        # Instance names repeat across families (d4_s<seed>_<i>), so compare within the family only.
        theirs = stored[stored["family"] == family].drop_duplicates(["instance", "target"]).set_index(["instance", "target"])["target_value"]
        common = mine.index.intersection(theirs.index)
        assert len(common) == len(mine) and np.allclose(mine.loc[common], theirs.loc[common], rtol=1e-12), "targets differ from the run's"
        frame.insert(0, "family", family)
        frames.append(frame)
        cap = float(config["cap_in_finest_grids"] * instances[0].n_fine)
        summary[family] = work_precision_summary(frame.drop(columns="family"), PAIRS, compute_cap=cap,
                                                 num_resamples=config["bootstrap_resamples"])
        print(f"{family} done")
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(out / "interpolated_work_precision.parquet", index=False)
    (out / "interpolated_summary.json").write_text(json.dumps({"pairs": PAIRS, "summary": summary,
                                                               "note": "post-hoc robustness check: compute-to-target interpolated between passes, for every policy"}, indent=2, sort_keys=True))
    print("wrote", out)


if __name__ == "__main__":
    main()
