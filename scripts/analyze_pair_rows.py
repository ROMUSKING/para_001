#!/usr/bin/env python3
"""scripts/analyze_pair_rows.py

Reanalysis of committed pair-interaction rows (review: existing records first, no new GPU).

Reads ``pair_interactions_v2/pair_interactions_summary.json`` (32 windows with per-window
singleton/pair J values) and recomputes, through the tested ``analyse_pair_rows`` path:

1. per-window demeaned epsilon and ranking fidelity of summed singletons vs measured
   pair gains (a common within-window shift cannot change ordering; pair-specific
   deviations can);
2. residual interaction fraction after removing window means;
3. the same-budget decision consequence with full contracts (measured 2-patch joints on
   both sides where available, estimated-fallback flagged, episode-level paired stats).

Usage::

    python scripts/analyze_pair_rows.py \\
        --input results/benchmarks/pair_interactions_v2/pair_interactions_summary.json \\
        --output results/benchmarks/pair_interactions_v2/pair_reanalysis.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent
for candidate in (REPO_DIR / "src", Path("/content/para_001/src")):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO_DIR / "scripts"))

import json

from adjointrwm.io import atomic_write_json  # noqa: E402
from diagnose_pair_interactions import analyse_pair_rows  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Reanalyse committed pair-interaction rows")
    parser.add_argument("--input", type=str,
                        default=str(REPO_DIR / "results/benchmarks/pair_interactions_v2"
                                             "/pair_interactions_summary.json"))
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = json.loads(Path(args.input).read_text())
    out = analyse_pair_rows(summary["window_rows"], summary["pair_manifest"],
                            _sites(summary), summary.get("seed", 0))
    report = {
        "benchmark": "session_6c_pair_reanalysis",
        "source": str(args.input),
        "n_windows": len(summary["window_rows"]),
        "ranking_fidelity_additive_vs_measured": float(np.nanmean(out["raw_rhos"])),
        "residual_interaction_fraction": float(np.nanmean(out["resid_frac"])),
        "same_budget": {
            "mean": float(np.mean(out["differences"])),
            "wilcoxon": out["wilcoxon"],
            "site_clustered_ci_95": out["cluster_ci"],
        },
        "contracts": {
            "quantity": "per-window gain difference: best measured pair gain minus joint "
                        "gain of the global top-2 singleton set",
            "best_scope": "minimum over the evaluated pair manifest (restricted hindsight)",
            "singleton_side": "measured joint where the top-2 set is in the manifest, "
                              "additive fallback flagged per window otherwise",
            "budgets": "unconstrained 2-patch sets on both sides",
        },
    }
    output = Path(args.output) if args.output else Path(args.input).parent / "pair_reanalysis.json"
    atomic_write_json(output, report)
    print(f"rank fidelity: {report['ranking_fidelity_additive_vs_measured']:.4f}, "
          f"residual fraction: {report['residual_interaction_fraction']:.4f}, "
          f"same-budget mean: {report['same_budget']['mean']:.6f}")
    print(f"Saved reanalysis to {output}")
    return 0


def _sites(summary: dict) -> dict:
    """Episode -> site map: window rows first, else the committed E3.1 manifest.

    The v2 rows carry no site field, so sites resolve through the frozen shard manifest
    (same split assignment the runs used). Anything still unmapped stays "unknown" and
    the bootstrap degrades explicitly rather than silently.
    """
    mapping = {}
    for row in summary["window_rows"]:
        mapping.setdefault(row["episode_id"], row.get("site", "unknown"))
    if any(v == "unknown" for v in mapping.values()):
        manifest_path = (REPO_DIR / "results/data/droid_e3_1/e3_1_droid_500_manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for episode in manifest.get("episodes", []):
                mapping.setdefault(episode["episode_id"], episode.get("site", "unknown"))
    return mapping


if __name__ == "__main__":
    raise SystemExit(main())
