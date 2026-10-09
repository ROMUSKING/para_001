#!/usr/bin/env python3
"""scripts/analyze_d2_confirm.py — frozen confirmatory adjudication.

Primary: all 1,417 remaining ids with component-clustered inference (cluster
bootstrap over the 439 union groups). Sensitivity: clean-282 subset clustered
over its 200 groups. Six comparisons, one Holm family (recall superiority
learned-vs-BM25 +5pp at k=4,8; recall vs longest lower bounds -2pp; EM vs BM25
lower bounds -2pp), one-sided Wilcoxon for ordering, Holm-adjusted lower edges.
Exploratory labels removed ONLY for this preregistered analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "src"))

import numpy as np

from adjointrwm.domains.qa_context import (
    POLICIES,
    build_registry,
    load_questions,
    miss_rate,
)
from adjointrwm.spatial_selection import wilcoxon_signed_rank

CACHE = Path.home() / ".cache/musique/data/data"


def cluster_bootstrap_means(group_values: dict, n_resamples: int,
                            seed: int) -> np.ndarray:
    """Bootstrap distribution of the mean resampling whole clusters."""
    groups = list(group_values.keys())
    sums = np.array([float(np.sum(group_values[g])) for g in groups])
    counts = np.array([len(group_values[g]) for g in groups], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(n_resamples, len(groups)))
    return sums[draws].sum(axis=1) / counts[draws].sum(axis=1)


def adjudicate(rows: list, clusters: dict, seed: int = 0,
               num_resamples: int = 5000) -> dict:
    """Six-test Holm family with cluster-bootstrapped lower edges."""
    by_qid = {r["qid"]: r for r in rows}
    qids = [r["qid"] for r in rows]

    specs = [
        ("recall learned-bm25 k=4", "recall", "learned", "bm25", 4, 0.05, True),
        ("recall learned-bm25 k=8", "recall", "learned", "bm25", 8, 0.05, True),
        ("recall learned-longest k=4", "recall", "learned", "longest", 4, -0.02, False),
        ("recall learned-longest k=8", "recall", "learned", "longest", 8, -0.02, False),
        ("EM learned-bm25 k=4", "em", "learned", "bm25", 4, -0.02, False),
        ("EM learned-bm25 k=8", "em", "learned", "bm25", 8, -0.02, False),
    ]
    tests = []
    for name, kind, arm, ref, k, margin, superiority in specs:
        if kind == "recall":
            a = np.array([1.0 - miss_rate(r[arm][k], r["supports"]) for r in rows])
            b = np.array([1.0 - miss_rate(r[ref][k], r["supports"]) for r in rows])
        else:
            a = np.array([r[f"{arm}_k{k}_em"] for r in rows], dtype=float)
            b = np.array([r[f"{ref}_k{k}_em"] for r in rows], dtype=float)
        d = a - b
        w = wilcoxon_signed_rank(d)
        # One-sided p in the predeclared improvement direction (positive
        # differences): p/2 when the mean is positive, 1 - p/2 otherwise.
        # Review note: this is the superiority-test tail, not a degradation
        # test — negative means yield large p here by construction; the bound
        # legs decide on Holm-adjusted lower edges, never on this p-value.
        one_sided = float(w["p_value"]) / 2 if float(d.mean()) > 0 else \
            1.0 - float(w["p_value"]) / 2
        groups: dict = {}
        for q, v in zip(qids, d):
            groups.setdefault(clusters[q], []).append(float(v))
        boot = cluster_bootstrap_means(groups, num_resamples, seed)
        tests.append({"name": name, "mean": float(d.mean()), "n": len(d),
                      "n_clusters": len(groups), "one_sided_p": one_sided,
                      "margin": margin, "superiority": superiority,
                      "boot_means": boot})
    ordered = sorted(tests, key=lambda t: t["one_sided_p"])
    running_max = 0.0
    for rank, t in enumerate(ordered):
        running_max = max(running_max,
                          min(1.0, t["one_sided_p"] * (len(ordered) - rank)))
        t["holm_p"] = running_max
        alpha_adj = 0.05 / (len(ordered) - rank)
        t["holm_alpha"] = alpha_adj
        t["holm_lower"] = float(np.quantile(t["boot_means"], alpha_adj))
        del t["boot_means"]
        if t["superiority"]:
            t["passes"] = bool(t["holm_p"] < 0.05 and t["mean"] >= t["margin"])
        else:
            t["passes"] = bool(t["holm_lower"] > t["margin"])
    return {"family": "holm-6-clustered", "tests": tests,
            "gate_passes": all(t["passes"] for t in tests)}


def main() -> int:
    import argparse
    import hashlib

    parser = argparse.ArgumentParser(description="D2 confirmatory adjudication")
    parser.add_argument("--em-summary", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    em = json.load(open(args.em_summary))
    rows = em["rows"]
    man = json.load(open(args.manifest))
    assert len(rows) == len(man["remaining_ids"]) == 1417, \
        f"row count {len(rows)} != 1417"
    assert [r["qid"] for r in rows] == man["remaining_ids"], \
        "row order differs from manifest order"
    train = load_questions(args.train_file, "train")
    dev = load_questions(args.dev_file, "dev")
    reg = build_registry(train, dev, tune_cap=10 ** 9, val_cap=10 ** 9, seed=0)
    by_id = {q["qid"]: q for q in reg["validation"]}

    # Learned sets are regenerated deterministically from frozen coefs (audit).
    ranker = json.load(open(REPO_DIR / "results/benchmarks/d2_rung1_ranker/rung1_ranker_summary.json"))
    coef = np.array(ranker["coef"]).ravel()
    intercept = float(np.array(ranker["intercept"]).ravel()[0])
    sys.path.insert(0, str(REPO_DIR / "src"))
    from adjointrwm.domains.qa_context import rung1_features
    for r in rows:
        q = by_id[r["qid"]]
        logits = (rung1_features(q) @ coef + intercept).ravel()
        probs = 1.0 / (1.0 + np.exp(-logits))
        order = sorted(range(len(probs)), key=lambda i: (-float(probs[i]), i))
        r["learned"] = {4: order[:4], 8: order[:8]}
        r["bm25"] = {4: POLICIES["bm25"](q, 4), 8: POLICIES["bm25"](q, 8)}
        r["longest"] = {4: POLICIES["longest_first"](q, 4),
                        8: POLICIES["longest_first"](q, 8)}
        r["supports"] = q["supports"]

    clusters = man["cluster_of"]
    primary = adjudicate(rows, clusters, args.seed)
    clean_ids = set(man["clean_ids"])
    sens_rows = [r for r in rows if r["qid"] in clean_ids]
    sens_clust = {q: c for q, c in clusters.items() if q in clean_ids}
    sensitivity = adjudicate(sens_rows, sens_clust, args.seed + 1)
    out = {"primary_1417_clustered": primary,
           "sensitivity_clean282_clustered": sensitivity,
           "exploratory_labels_removed_for_this_analysis_only": True}
    Path(args.output).write_text(json.dumps(out, indent=1))
    for label, res in (("PRIMARY", primary), ("SENSITIVITY", sensitivity)):
        print(f"== {label} ==")
        for t in res["tests"]:
            print("  %-26s mean=%+.4f holm_low=%+.4f holm_p=%.4g passes=%s ncl=%d" % (
                t["name"], t["mean"], t["holm_lower"], t["holm_p"],
                t["passes"], t["n_clusters"]))
        print("  GATE:", "PASS" if res["gate_passes"] else "FAIL")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
