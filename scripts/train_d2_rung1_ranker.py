#!/usr/bin/env python3
"""scripts/train_d2_rung1_ranker.py — frozen Rung-1 learned ranker (CPU).

Implements docs/plans/2026-10-08-d2-rung1-plan.md §§1-2 (recall legs): L2
logistic regression on six deployment-valid features, C from {0.1, 1.0, 10.0}
by question-grouped 5-fold CV on tuning (C-selection only), final fit on all
tuning, recall evaluated on the frozen 300 validation ids vs BM25 + longest
(Holm family shared with the EM legs, computed in analysis). Exports learned
paragraph sets for the GPU reader stage. No reader, no GPU here.
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
    build_registry,
    load_questions,
    miss_rate,
    rung1_features,
    select_topk,
)
from adjointrwm.spatial_selection import wilcoxon_signed_rank  # noqa: E402

BUDGETS = (4, 8)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def design_matrix(records):
    """Stacked (X, y, question_index) over paragraphs of tuning questions."""
    parts, labels, qid = [], [], []
    for qi, rec in enumerate(records):
        feats = rung1_features(rec)
        parts.append(feats)
        labels.extend([1 if f else 0 for f in rec["supports"]])
        qid.extend([qi] * len(rec["texts"]))
    return np.vstack(parts), np.asarray(labels, dtype=int), np.asarray(qid)


def grouped_cv_score(X, y, qid, seed, C: float) -> float:
    """Mean recall@4 over question-grouped 5-fold CV (C-selection rule only)."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    questions = np.unique(qid)
    rng.shuffle(questions)
    folds = np.array_split(questions, 5)
    recalls = []
    for fold in folds:
        test_mask = np.isin(qid, fold)
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
        clf.fit(X[~test_mask], y[~test_mask])
        proba = clf.predict_proba(X[test_mask])[:, 1]
        # Per-question recall@4 within the held-out fold (positions are local
        # to the test-masked arrays, not global row ids).
        test_qid = qid[test_mask]
        for q in fold:
            m = test_qid == q
            idx = np.where(m)[0]
            top = idx[np.argsort(-proba[m], kind="stable")[:4]]
            hit = sum(y[test_mask][top])
            total = y[test_mask][m].sum()
            if total > 0:
                recalls.append(hit / total)
    return float(np.mean(recalls))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="D2 Rung-1 ranker training")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--validation-ids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    train = load_questions(args.train_file, "train")
    dev = load_questions(args.dev_file, "dev")
    reg = build_registry(train, dev, tune_cap=2000, val_cap=10 ** 9, seed=args.seed)
    X, y, qid = design_matrix(reg["tuning"])
    print(f"tuning paragraphs: {len(y)}, positives: {int(y.sum())}", flush=True)

    cv = {C: grouped_cv_score(X, y, qid, args.seed, C) for C in (0.1, 1.0, 10.0)}
    best_C = max(cv, key=lambda c: cv[c])
    print(f"CV recall@4: {cv} -> C={best_C}", flush=True)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=best_C, class_weight="balanced", max_iter=2000)
    clf.fit(X, y)

    wanted = [l.strip() for l in open(args.validation_ids) if l.strip()]
    by_id = {r["qid"]: r for r in reg["validation"]}
    missing = [q for q in wanted if q not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} frozen ids not in registry")
    questions = [by_id[q] for q in wanted]

    learned_sets, rec = {}, {}
    for qu in questions:
        probs = clf.predict_proba(rung1_features(qu))[:, 1]
        order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
        learned_sets[qu["qid"]] = order
        for k in BUDGETS:
            rec.setdefault(k, []).append(
                miss_rate(order[:k], qu["supports"]))
    for k in BUDGETS:
        print(f"k={k}: learned miss={np.mean(rec[k]):.4f}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "benchmark": "d2_rung1_ranker",
        "mode": "real",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "train_file": args.train_file, "train_sha256": sha256_file(args.train_file),
            "dev_file": args.dev_file, "dev_sha256": sha256_file(args.dev_file),
        },
        "cv_recall_at_4": cv,
        "selected_C": best_C,
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "n_questions": len(questions),
        "learned_sets": learned_sets,
        "recall_miss": {str(k): rec[k] for k in BUDGETS},
    }
    (output_dir / "rung1_ranker_summary.json").write_text(json.dumps(summary))
    print(f"Saved summary to {output_dir / 'rung1_ranker_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
