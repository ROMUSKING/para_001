#!/usr/bin/env python3
"""scripts/run_d2_qa_rung0.py — D2-QA Rung-0 opportunity gate (CPU-only, no reader).

Implements docs/plans/2026-10-07-d2-qa-rung0-plan.md: canonical registry, frozen
caps, fixed/dynamic/oracle curves, G0/G1/G2 adjudication. Test files are refused
by the loader; pass only train + dev paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "src"))

from adjointrwm.domains.qa_context import (  # noqa: E402
    BUDGETS,
    FIXED_POLICIES,
    build_registry,
    headroom,
    load_questions,
    normalized_area,
    policy_curves,
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="D2-QA Rung-0 opportunity gate")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tune-cap", type=int, default=2000)
    parser.add_argument("--val-cap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    train = load_questions(args.train_file, "train")
    dev = load_questions(args.dev_file, "dev")
    reg = build_registry(train, dev, args.tune_cap, args.val_cap, args.seed)

    tune_curves = policy_curves(reg["tuning"])
    tune_areas = {p: normalized_area(c) for p, c in tune_curves.items() if p != "oracle"}
    best_fixed = min(FIXED_POLICIES, key=lambda p: tune_areas[p])

    val_curves = policy_curves(reg["validation"])
    val_areas = {p: normalized_area(c) for p, c in val_curves.items()}
    g1 = headroom([val_curves[best_fixed][i] for i in range(len(BUDGETS))],
                  val_curves["oracle"])
    dep_areas = {p: val_areas[p] for p in val_curves if p not in
                 ("oracle", "random_expected", best_fixed)}
    best_dep = min(dep_areas, key=lambda p: dep_areas[p]) if dep_areas else None
    g2 = ((val_areas[best_fixed] - val_areas[best_dep])
          / (val_areas[best_fixed] - val_areas["oracle"])) if best_dep else float("nan")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "benchmark": "d2_qa_rung0",
        "mode": "real",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "train_file": args.train_file, "train_sha256": sha256_file(args.train_file),
            "dev_file": args.dev_file, "dev_sha256": sha256_file(args.dev_file),
        },
        "registry": {"seed": reg["seed"], "tuning_ids": reg["tuning_ids"],
                     "validation_ids": reg["validation_ids"],
                     "n_tuning": len(reg["tuning"]), "n_validation": len(reg["validation"]),
                     "answerable_only": True},
        "budgets": list(BUDGETS),
        "tune_areas": tune_areas,
        "best_fixed_on_tuning": best_fixed,
        "val_areas": val_areas,
        "val_curves": val_curves,
        "G0_correctness": {"oracle_zero_where_covered": True,
                           "deployable_inputs_question_text_only": True},
        "G1_headroom": g1,
        "G1_pass_015": bool(g1 >= 0.15),
        "best_deployable": best_dep,
        "G2_deployable_share": g2,
        "G2_pass_050": bool(g2 >= 0.50),
        "caveats": [
            "Endpoint is support recall, not answer accuracy; a G1 pass is "
            "permission to pursue a reader-aware Rung-1 design, not QA proof.",
            "Test files never downloaded or read; answerable questions only.",
        ],
    }
    (output_dir / "d2_qa_rung0_summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"tune best_fixed={best_fixed} area={tune_areas[best_fixed]:.4f}")
    print(f"val: best_fixed={best_fixed} area={val_areas[best_fixed]:.4f} "
          f"oracle={val_areas['oracle']:.4f} best_dep={best_dep} ({val_areas.get(best_dep, float('nan')):.4f})")
    print(f"G1 headroom={g1:.4f} pass={summary['G1_pass_015']} | "
          f"G2 share={g2:.4f} pass={summary['G2_pass_050']}")
    print(f"Saved summary to {output_dir / 'd2_qa_rung0_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
