#!/usr/bin/env python3
"""Summarise an allocation_traces.parquet file from a pilot run.

Example:
    python scripts/analyze_allocation_traces.py \
        results/runs/droid100_adjoint_20260929T070629Z/artifacts/allocation_traces.parquet \
        --num-candidates 4 --out results/runs/droid100_adjoint_20260929T070629Z/artifacts/trace_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adjointrwm.analysis import load_traces, per_episode_means, summarize_traces  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traces", type=Path)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="write the JSON summary here")
    args = parser.parse_args()

    frame = load_traces(str(args.traces))
    summary = summarize_traces(frame, args.num_candidates, seed=args.seed)
    text = json.dumps(summary, indent=2)
    print(text)
    print("\nPer-episode mean regret:\n", per_episode_means(frame).round(4).to_string())
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
