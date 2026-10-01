#!/usr/bin/env python3
"""Compare the two regimes of the episode-level k-fold dynamics study (plan: docs/plans/dynamics-kfold-plan.md, section 4).

    python scripts/dynamics_kfold_compare.py results/runs/dynamics_kfold_v2_<id> results/runs/dynamics_kfold_pilot_<id>

Reads each run's ``reports/acceptance_report.json`` and ``artifacts/episode_errors.csv``, prints each regime's frozen class, and the paired difference
``R(v2) - R(pilot)`` over the same episodes with a paired episode-cluster bootstrap (10,000 resamples, seed 0), labelled ``V2_BETTER``, ``PILOT_BETTER`` or
``NOT_DISTINGUISHED`` by whether the 95 % interval excludes 0. It stops if the two runs are not one ``v2`` and one ``pilot`` run over the same 100 episodes,
and says so plainly if either run's anchor failed (the comparison is then not interpretable).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adjointrwm.eval import paired_bootstrap_difference, pooled_relative_improvement  # noqa: E402

LABELS = {"A_BETTER": "V2_BETTER", "B_BETTER": "PILOT_BETTER", "NOT_DISTINGUISHED": "NOT_DISTINGUISHED"}


def load_run(run: Path) -> tuple[str, dict, pd.DataFrame]:
    report = json.loads((run / "reports/acceptance_report.json").read_text())
    (regime,) = report["regimes"]                                   # one regime per job
    entry = report["regimes"][regime]
    errors = pd.read_csv(run / "artifacts/episode_errors.csv")
    primary = errors[(errors["regime"] == regime) & (errors["kind"] == "best") & (errors["mode"] == entry["own_mode"])].drop(columns=["regime", "kind", "mode", "fold"])
    assert len(primary) in (99, 100) and primary["episode_id"].is_unique, f"{run}: the primary table has {len(primary)} episodes (expected 99 or 100)"
    stored = entry["estimates"][f"best|{entry['own_mode']}"]["pooled"]["relative_improvement"]
    assert abs(pooled_relative_improvement(primary)["relative_improvement"] - stored) < 1e-9, f"{run}: the episode table does not reproduce the report's estimate"
    return regime, entry, primary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("v2_run", type=Path)
    parser.add_argument("pilot_run", type=Path)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    (name_a, entry_a, table_a), (name_b, entry_b, table_b) = load_run(args.v2_run), load_run(args.pilot_run)
    assert (name_a, name_b) == ("v2", "pilot"), f"expected a v2 run then a pilot run, got {name_a} and {name_b}"
    assert sorted(table_a["episode_id"]) == sorted(table_b["episode_id"]), "the two runs do not cover the same episodes"
    for name, entry in ((name_a, entry_a), (name_b, entry_b)):
        est = entry["estimates"][f"best|{entry['own_mode']}"]
        b = est["bootstrap"]
        print(f"{name}: status {entry['status']} | primary class {entry['primary_class']} | R {b['estimate']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}] | "
              f"random ten-episode splits passing 2 %: {est['random_ten_episode_split_gate_rate']['pass_rate']:.3f}")
    paired = paired_bootstrap_difference(table_a, table_b, args.resamples, 0)
    print(f"\nR(v2) - R(pilot) = {paired['estimate']:+.3f} [{paired['ci_low']:+.3f}, {paired['ci_high']:+.3f}] over {paired['episodes']} episodes -> {LABELS[paired['label']]}")
    if entry_a["status"] != "OK" or entry_b["status"] != "OK":
        print("\nWARNING: at least one anchor failed, so these numbers are not interpreted (plan section 5).")


if __name__ == "__main__":
    main()
