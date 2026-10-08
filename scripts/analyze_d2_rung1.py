#!/usr/bin/env python3
"""scripts/analyze_d2_rung1.py — frozen Rung-1 gate adjudication (Holm-6).

Six comparisons, one family: recall learned-vs-BM25 (k=4,8; superiority, +5pp),
recall learned-vs-longest lower bounds (k=4,8; margin -2pp), EM learned-vs-BM25
lower bounds (k=4,8; margin -2pp). Superiority legs use one-sided paired
Wilcoxon; bound legs use Holm-adjusted Wald lower edges (adjusted alpha per
Holm rank — a pointwise CI is NOT familywise-adjusted). Exploratory labels
throughout (frozen 300 ids reused across studies).
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


def main() -> int:
    ranker = json.load(open(
        REPO_DIR / "results/benchmarks/d2_rung1_ranker/rung1_ranker_summary.json"))
    em = json.load(open(
        REPO_DIR / "results/benchmarks/d2_rung1_em/reader_bridge_summary.json"))
    rows = em["rows"]
    train = load_questions(str(CACHE / "musique_full_v1.0_train.jsonl"), "train")
    dev = load_questions(str(CACHE / "musique_full_v1.0_dev.jsonl"), "dev")
    reg = build_registry(train, dev, tune_cap=10 ** 9, val_cap=10 ** 9, seed=0)
    vids = [l.strip() for l in open(
        REPO_DIR / "results/benchmarks/d2_qa_rung0/validation_ids_300.txt")
        if l.strip()]
    by_id = {q["qid"]: q for q in reg["validation"]}
    qs = [by_id[i] for i in vids]
    learned = ranker["learned_sets"]

    def recall(name, k):
        if name == "learned":
            return np.array([1.0 - miss_rate(learned[q["qid"]][:k], q["supports"])
                             for q in qs])
        return np.array([1.0 - miss_rate(POLICIES[name](q, k), q["supports"])
                         for q in qs])

    tests = []
    for k in (4, 8):
        rl = recall("learned", k)
        rb = recall("bm25", k)
        ll = recall("longest_first", k)
        eo = np.array([r[f"learned_k{k}_em"] for r in rows], dtype=float)
        eb = np.array([r[f"bm25_k{k}_em"] for r in rows], dtype=float)
        tests.append({"name": f"recall learned-bm25 k={k}", "diff": rl - rb,
                      "margin": 0.05, "superiority": True})
        tests.append({"name": f"recall learned-longest k={k}", "diff": rl - ll,
                      "margin": -0.02, "superiority": False})
        tests.append({"name": f"EM learned-bm25 k={k}", "diff": eo - eb,
                      "margin": -0.02, "superiority": False})
    for t in tests:
        d = t["diff"]
        t["mean"] = float(d.mean())
        t["n"] = int(len(d))
        w = wilcoxon_signed_rank(d)
        t["wilcoxon_two_sided_p"] = float(w["p_value"])
        # One-sided superiority p (improvement direction only).
        t["one_sided_p"] = float(w["p_value"]) / 2 if t["mean"] > 0 else \
            1.0 - float(w["p_value"]) / 2
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        t["wald_se"] = se
        t["wald_ci95"] = [t["mean"] - 1.96 * se, t["mean"] + 1.96 * se]
    # Holm step-down over the one-sided family p-values; bound legs use the
    # Holm-adjusted alpha for their lower edges (review: pointwise CIs are not
    # familywise-adjusted, so the old code's unadjusted edges were mislabelled).
    import statistics as _stats

    ordered = sorted(tests, key=lambda t: t["one_sided_p"])
    running_max = 0.0
    for rank, t in enumerate(ordered):
        running_max = max(running_max,
                          min(1.0, t["one_sided_p"] * (len(ordered) - rank)))
        t["holm_p"] = running_max
        # Holm-adjusted one-sided lower edge at coverage 1 - alpha_adj,
        # alpha_adj = 0.05/(m - rank) (step-down).
        alpha_adj = 0.05 / (len(ordered) - rank)
        z = _stats.NormalDist().inv_cdf(1.0 - alpha_adj)
        t["holm_alpha"] = alpha_adj
        t["holm_lower"] = t["mean"] - z * t["wald_se"]
        if t["superiority"]:
            t["passes"] = bool(t["holm_p"] < 0.05 and t["mean"] >= t["margin"])
        else:
            t["passes"] = bool(t["holm_lower"] > t["margin"])
    out = {"family": "holm-6", "exploratory": True, "tests": [
        {k: (v if not isinstance(v, np.ndarray) else None)
         for k, v in t.items() if k != "diff"} for t in tests]}
    out["gate_passes"] = all(t["passes"] for t in tests)
    for t in tests:
        print("  %-28s mean=%+.4f holm_low=%+.4f holm_p=%.4g passes=%s" % (
            t["name"], t["mean"], t["holm_lower"], t["holm_p"], t["passes"]))
    print("GATE:", "PASS" if out["gate_passes"] else "FAIL")
    Path(REPO_DIR / "results/benchmarks/d2_rung1_gate.json").write_text(
        json.dumps(out, indent=1))
    print("wrote results/benchmarks/d2_rung1_gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
