#!/usr/bin/env python3
"""scripts/run_d2_r2_redesign.py — frozen R2 candidate study (CPU-only).

Implements docs/plans/2026-10-08-d2-r2-redesign-plan.md: the three frozen R2
constructions against BM25 at k=4,8 (one-sided Wilcoxon + Holm over six tests),
plus precision, complete-support recovery, hop buckets, and Phi-tokenizer token
accounting. Same frozen registry/ids as Rung-0; test files refused by the loader.
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

import numpy as np  # noqa: E402

from adjointrwm.domains.qa_context import (  # noqa: E402
    POLICIES,
    R2_POLICIES,
    build_registry,
    complete_support_recovery,
    hop_bucket,
    load_questions,
    miss_rate,
    support_precision,
)
from adjointrwm.spatial_selection import wilcoxon_signed_rank  # noqa: E402

BUDGETS = (4, 8)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def count_tokens(tokenizer, titles, texts, selected) -> int:
    """Phi-tokenizer count over the deployed rendering: '[title] text' + newlines."""
    rendered = "\n".join("[%s] %s" % (titles[i], texts[i]) for i in selected)
    return len(tokenizer.encode(rendered))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="D2 R2 redesign study")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--validation-ids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    train = load_questions(args.train_file, "train")
    dev = load_questions(args.dev_file, "dev")
    reg = build_registry(train, dev, tune_cap=10 ** 9, val_cap=10 ** 9, seed=args.seed)
    wanted = [l.strip() for l in open(args.validation_ids) if l.strip()]
    by_id = {r["qid"]: r for r in reg["validation"]}
    missing = [q for q in wanted if q not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} frozen ids not in registry")
    questions = [by_id[q] for q in wanted]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct",
                                              trust_remote_code=False)

    per_q = []
    for rec in questions:
        entry = {"qid": rec["qid"], "hop": hop_bucket(rec["qid"]),
                 "n_supports": sum(rec["supports"])}
        for k in BUDGETS:
            for name, fn in {**POLICIES, **R2_POLICIES}.items():
                if name in ("first_k", "doc_round_robin"):
                    continue  # reported in Rung-0; not re-compared here
                sel = fn(rec, k)
                entry[f"{name}_k{k}_miss"] = miss_rate(sel, rec["supports"])
                entry[f"{name}_k{k}_prec"] = support_precision(sel, rec["supports"])
                entry[f"{name}_k{k}_complete"] = complete_support_recovery(
                    sel, rec["supports"])
                entry[f"{name}_k{k}_tokens"] = count_tokens(
                    tokenizer, rec["titles"], rec["texts"], sel)
                entry[f"{name}_k{k}_n"] = len(sel)
        per_q.append(entry)

    # Gate: each R2 candidate vs BM25 recall at k=4,8 (one-sided improvement).
    rec = lambda name, k: np.array([1.0 - q[f"{name}_k{k}_miss"] for q in per_q])
    bm = {k: rec("bm25", k) for k in BUDGETS}
    tests = []
    for name in R2_POLICIES:
        for k in BUDGETS:
            diff = rec(name, k) - bm[k]
            w = wilcoxon_signed_rank(diff)
            tests.append({"candidate": name, "budget": k,
                          "mean_gain_pp": float(diff.mean() * 100),
                          "p_one_sided": float(w["p_value"]) / 2
                          if float(diff.mean()) > 0 else 1.0 - float(w["p_value"]) / 2})
    ordered = sorted(tests, key=lambda t: t["p_one_sided"])
    m = len(ordered)
    running_max = 0.0
    for rank, t in enumerate(ordered):
        running_max = max(running_max, min(1.0, t["p_one_sided"] * (m - rank)))
        t["holm_p"] = running_max
        t["passes"] = bool(t["holm_p"] < 0.05 and t["mean_gain_pp"] >= 5.0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "benchmark": "d2_r2_redesign",
        "mode": "real",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "train_file": args.train_file, "train_sha256": sha256_file(args.train_file),
            "dev_file": args.dev_file, "dev_sha256": sha256_file(args.dev_file),
        },
        "n_questions": len(per_q),
        "tokenizer": "microsoft/Phi-4-mini-instruct",
        "tests": tests,
        "study_passes": any(t["passes"] for t in tests),
        "per_question": per_q,
        "caveats": [
            "Paragraph budgets primary; tokens reported, not gated (except the "
            "documented token-capped secondary in a follow-up).",
            "Passing candidates are reported, not deployed; no learned rescue.",
        ],
    }
    (output_dir / "d2_r2_summary.json").write_text(json.dumps(summary, indent=1))
    for t in tests:
        print("  %s k=%d gain=%+.2fpp holm_p=%.4g passes=%s"
              % (t["candidate"], t["budget"], t["mean_gain_pp"], t["holm_p"],
                 t["passes"]), flush=True)
    print("study_passes =", summary["study_passes"])
    print(f"Saved summary to {output_dir / 'd2_r2_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
