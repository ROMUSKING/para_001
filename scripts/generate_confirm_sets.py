#!/usr/bin/env python3
"""scripts/generate_confirm_sets.py — frozen learned sets for the D2 confirmatory run.

Applies the FROZEN Rung-1 ranker (coefs from the committed ranker summary) to the
1,417 manifest ids using committed feature code. No fitting, no tuning, no
outcomes. Output hash recorded in the run log before the reader stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "src"))

import numpy as np

from adjointrwm.domains.qa_context import build_registry, load_questions, rung1_features


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate frozen confirmatory sets")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--ranker-summary", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    ranker = json.load(open(args.ranker_summary))
    coef = np.array(ranker["coef"]).ravel()
    intercept = float(np.array(ranker["intercept"]).ravel()[0])
    man = json.load(open(args.manifest))
    train = load_questions(args.train_file, "train")
    dev = load_questions(args.dev_file, "dev")
    reg = build_registry(train, dev, tune_cap=10 ** 9, val_cap=10 ** 9, seed=0)
    by_id = {q["qid"]: q for q in reg["validation"]}
    missing = [q for q in man["remaining_ids"] if q not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} manifest ids not in registry")
    sets = {}
    for qid in man["remaining_ids"]:
        q = by_id[qid]
        logits = (rung1_features(q) @ coef + intercept).ravel()
        probs = 1.0 / (1.0 + np.exp(-logits))
        sets[qid] = sorted(range(len(probs)), key=lambda i: (-float(probs[i]), i))
    Path(args.output).write_text(json.dumps(sets))
    digest = hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
    print(f"sets: {len(sets)} sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
