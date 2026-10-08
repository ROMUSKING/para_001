"""D2-QA Rung-0: paragraph-selection opportunity gate on MuSiQue (CPU-only).

Non-learned policies rank 20 paragraphs per question under a paragraph budget;
the endpoint is support miss rate (no reader model). See
docs/plans/2026-10-07-d2-qa-rung0-plan.md (peer-reviewed). Test files are refused
at load time, never read.
"""

from __future__ import annotations

import json
import math
import re
from typing import Callable, Mapping, Sequence

import numpy as np

BUDGETS = (1, 2, 4, 6, 8, 10, 15, 20)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    """Lowercase alphanumeric tokens (frozen tokenization for all rankers)."""
    return _TOKEN.findall(text.lower())


def load_questions(path: str, split: str) -> list:
    """Load one MuSiQue full jsonl file; refuses anything test-like.

    Returns records with ``qid``, ``question``, ``texts``, ``supports`` (bool
    mask), ``answerable``. Only answerable questions are kept: with an empty
    support set the miss rate has no denominator (plan amendment 2026-10-07,
    mirrors the theta near-zero-denominator rule).
    """
    import os as _os
    lowered = _os.path.basename(path).lower()
    if "test" in lowered:
        raise ValueError(
            f"refusing test file {path!r}: the test split is never downloaded or read")
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("answerable", True):
                continue
            supports = [bool(p["is_supporting"]) for p in rec["paragraphs"]]
            if not any(supports):
                continue
            records.append({
                "qid": rec["id"],
                "question": rec["question"],
                "texts": [p["paragraph_text"] for p in rec["paragraphs"]],
                "titles": [p.get("title", "") for p in rec["paragraphs"]],
                "supports": supports,
                "split": split,
                # Scoring-only fields (EM): never passed to selection functions.
                # The input-boundary tests pin that rankers read question+texts only.
                "answer": rec.get("answer", ""),
                "answer_aliases": rec.get("answer_aliases", []) or [],
            })
    return records


def build_registry(train_recs: Sequence[Mapping], dev_recs: Sequence[Mapping],
                   tune_cap: int = 2000, val_cap: int = 1000,
                   seed: int = 0) -> dict:
    """Canonical id registry: dedupe across files, frozen shuffle, in-split caps.

    Returns ``{"tuning": [...], "validation": [...]}`` with frozen question-id
    lists (commit them in the run manifest). Raises on any id present in both
    files — cross-file contamination is a hard error, not a warning.
    """
    train_ids = [r["qid"] for r in train_recs]
    dev_ids = [r["qid"] for r in dev_recs]
    overlap = set(train_ids) & set(dev_ids)
    if overlap:
        raise ValueError(
            f"{len(overlap)} question ids in both files (e.g. {sorted(overlap)[0]}): "
            "deduplicate before splitting")
    rng = np.random.default_rng(seed)
    order = lambda ids: [ids[i] for i in rng.permutation(len(ids)).tolist()]
    by_id = {r["qid"]: r for r in list(train_recs) + list(dev_recs)}
    tune_ids = order(train_ids)[:tune_cap]
    val_ids = order(dev_ids)[:val_cap]
    return {"tuning": [by_id[i] for i in tune_ids],
            "validation": [by_id[i] for i in val_ids],
            "tuning_ids": tune_ids, "validation_ids": val_ids, "seed": seed}


def _doc_term_matrix(texts: Sequence[str]):
    """Term counts per doc plus document frequencies over the question's 20 docs."""
    tokenized = [tokenize(t) for t in texts]
    vocab: dict = {}
    rows = []
    for toks in tokenized:
        counts: dict = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
            if t not in vocab:
                vocab[t] = len(vocab)
        rows.append(counts)
    df = np.zeros(len(vocab))
    for counts in rows:
        for t in counts:
            df[vocab[t]] += 1
    return rows, vocab, df


def bm25_scores(question: str, texts: Sequence[str], k1: float = 1.2,
                b: float = 0.75, extra_query_terms: Sequence[str] = (),
                boost: Mapping[str, float] | None = None) -> np.ndarray:
    """BM25 of each paragraph against the question (per-question IDF, frozen).

    ``extra_query_terms`` appends deployment-visible terms (count 1 each);
    ``boost`` multiplies individual query-term weights (each distinct boosted
    term capped at its boost; unboosted terms weight 1).
    """
    rows, vocab, df = _doc_term_matrix(texts)
    n = len(texts)
    idf = {t: math.log((n - df[i] + 0.5) / (df[i] + 0.5) + 1.0) for t, i in vocab.items()}
    lens = np.array([sum(r.values()) for r in rows], dtype=float)
    avg = lens.mean() if n else 1.0
    qtf: dict = {}
    for t in tokenize(question):
        qtf[t] = qtf.get(t, 0) + 1
    for t in extra_query_terms:
        qtf[t] = qtf.get(t, 0) + 1
    boost = boost or {}
    scores = np.zeros(n)
    for t, q in qtf.items():
        if t not in vocab:
            continue
        w = min(q * boost.get(t, 1.0), 2.0) if t in boost else q
        for d, counts in enumerate(rows):
            f = counts.get(t, 0)
            if f:
                denom = f + k1 * (1.0 - b + b * lens[d] / avg)
                scores[d] += idf[t] * f * (k1 + 1.0) / denom * w
    return scores


def tfidf_cosine_scores(question: str, texts: Sequence[str]) -> np.ndarray:
    """TF-IDF cosine similarity of each paragraph to the question."""
    rows, vocab, df = _doc_term_matrix([question, *texts])
    n = len(texts) + 1
    idf = np.log(n / (df + 1.0)) + 1.0
    dim = len(vocab)
    q = np.zeros(dim)
    for t in tokenize(question):
        q[vocab[t]] += 1
    q = q * idf
    qn = float(np.linalg.norm(q)) or 1.0
    out = np.zeros(len(texts))
    for d, counts in enumerate(rows[1:]):
        v = np.zeros(dim)
        for t, f in counts.items():
            v[vocab[t]] = f
        v = v * idf
        vn = float(np.linalg.norm(v)) or 1.0
        out[d] = float(v @ q / (vn * qn))
    return out


def select_topk(scores: np.ndarray, k: int) -> list:
    """Top-k indices, ties toward the lower paragraph index (frozen)."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[:k]


def round_robin_groups(groups: Sequence[str], k: int) -> list:
    """Round-robin over title groups (frozen order of first appearance)."""
    seen: dict = {}
    for i, g in enumerate(groups):
        seen.setdefault(g, []).append(i)
    pools = list(seen.values())
    out = []
    turn = 0
    while len(out) < k and any(len(p) > turn for p in pools):
        for p in pools:
            if len(out) >= k:
                break
            if len(p) > turn:
                out.append(p[turn])
        turn += 1
    return out


def miss_rate(selected: Sequence[int], supports: Sequence[bool]) -> float:
    """Support miss rate of a selected set (lower is better)."""
    s = set(selected)
    hit = sum(1 for i, flag in enumerate(supports) if flag and i in s)
    total = sum(1 for flag in supports if flag)
    return 1.0 - hit / total


def support_precision(selected: Sequence[int], supports: Sequence[bool]) -> float:
    """Fraction of selected paragraphs that support (exposes bundle bloat)."""
    if not selected:
        return float("nan")
    s = set(selected)
    return sum(1 for i in s if supports[i]) / len(s)


def complete_support_recovery(selected: Sequence[int], supports: Sequence[bool]) -> int:
    """1 iff every supporting paragraph is selected, else 0."""
    s = set(selected)
    return int(all(i in s for i, flag in enumerate(supports) if flag))


def hop_bucket(qid: str) -> str:
    """2/3/4-hop bucket from the MuSiQue id prefix (unknown → 'other')."""
    m = re.match(r"(\d+)hop", qid)
    return (m.group(1) + "hop") if m else "other"


def rung1_features(rec: Mapping) -> np.ndarray:
    """Six deployment-valid per-paragraph features (Rung-1, frozen definitions).

    BM25, TF-IDF cosine, token length, position index, title-match count
    (shared tokens between question and title), entity-match count (shared
    tokens between the paragraph and ``entity_spans(question)``). Only
    question + visible texts/titles enter; supports, answers, aliases and
    decompositions never do (boundary-tested).
    """
    question, texts, titles = rec["question"], rec["texts"], rec["titles"]
    bm = bm25_scores(question, texts)
    tf = tfidf_cosine_scores(question, texts)
    qtok = set(tokenize(question))
    espans = entity_spans(question)
    espan_toks = set()
    for span in espans:
        espan_toks.update(tokenize(span))
    feats = np.zeros((len(texts), 6))
    for i, (text, title) in enumerate(zip(texts, titles)):
        ptok = tokenize(text)
        feats[i, 0] = bm[i]
        feats[i, 1] = tf[i]
        feats[i, 2] = len(ptok)
        feats[i, 3] = i
        feats[i, 4] = len(qtok & set(tokenize(title)))
        feats[i, 5] = len(espan_toks & set(ptok))
    return feats


def entity_spans(question: str) -> list:
    """Visible entity spans: `[A-Z][a-z]+` runs (length ≥ 1) plus quoted spans.

    Overlapping spans merged, each distinct span counted once. Question text
    only — never labels, answers, or decompositions.
    """
    spans = set()
    words = re.findall(r"[A-Za-z0-9']+", question)
    run = []
    for w in words + [""]:
        if re.fullmatch(r"[A-Z][a-z]+", w or " "):
            run.append(w)
        else:
            if run:
                spans.add(" ".join(run))
            run = []
    for quoted in re.findall(r"'([^']+)'|\"([^\"]+)\"", question):
        span = quoted[0] or quoted[1]
        if span.strip():
            spans.add(span.strip())
    return sorted(spans)


def top_tfidf_terms(text: str, corpus_texts: Sequence[str], question: str,
                    n: int = 5) -> list:
    """Top-n TF-IDF terms of one paragraph absent from the question.

    Corpus = the question's own paragraphs, repo tokenization, no stopwords,
    ties to first occurrence (frozen conventions).
    """
    rows, vocab, df = _doc_term_matrix(corpus_texts)
    total_docs = len(corpus_texts)
    excluded = set(tokenize(question))
    doc_idx = list(corpus_texts).index(text)
    counts = rows[doc_idx]
    scored = []
    seen_terms = set()
    for position, t in enumerate(tokenize(text)):
        if t in excluded or t in seen_terms:
            continue
        seen_terms.add(t)
        tfidf = counts[t] * math.log(total_docs / (df[vocab[t]] + 1.0))
        scored.append((t, tfidf, position))
    scored.sort(key=lambda s: (-s[1], s[2]))
    return [t for t, _, _ in scored[:n]]


def incremental_bm25(question: str, texts: Sequence[str], k: int) -> list:
    """Two frozen rounds: BM25 top-1, then BM25 with 5 new TF-IDF terms.

    Duplicate suppression: the round-1 pick leads; remaining slots fill from the
    round-2 ranking with the round-1 pick skipped.
    """
    first = select_topk(bm25_scores(question, texts), 1)[0]
    new_terms = top_tfidf_terms(texts[first], texts, question, 5)
    order = select_topk(bm25_scores(question, texts, extra_query_terms=new_terms),
                        len(texts))
    out = [first]
    for i in order:
        if i != first:
            out.append(i)
        if len(out) >= k:
            break
    return out[:k]


def title_bundle(question: str, texts: Sequence[str], titles: Sequence[str],
                 k: int) -> list:
    """Whole title groups in BM25 group-rank order, skipping non-fitting groups.

    Budget strictly enforced: a group is taken only if it fits the remaining
    slots, otherwise skipped (undershoot recorded by the caller via length).
    """
    seen: dict = {}
    for i, g in enumerate(titles):
        seen.setdefault(g, []).append(i)
    group_text = {g: " ".join(texts[i] for i in idx) for g, idx in seen.items()}
    names = list(seen)
    scores = bm25_scores(question, [group_text[g] for g in names])
    order = sorted(range(len(names)), key=lambda i: (-scores[i], i))
    out = []
    for i in order:
        members = seen[names[i]]
        if len(out) + len(members) <= k:
            out.extend(members)
        if len(out) >= k:
            break
    return out[:k]


R2_POLICIES: dict = {
    "entity_expanded": lambda rec, k: select_topk(bm25_scores(
        rec["question"], rec["texts"],
        boost={t: 2.0 for span in entity_spans(rec["question"])
               for t in tokenize(span)}), k),
    "incremental_bm25": lambda rec, k: incremental_bm25(rec["question"], rec["texts"], k),
    "title_bundle": lambda rec, k: title_bundle(rec["question"], rec["texts"],
                                                rec["titles"], k),
}


def random_expected_miss(supports: Sequence[bool], k: int) -> float:
    """Exact expected miss rate of uniform random selection (no simulation)."""
    total = sum(1 for flag in supports if flag)
    n = len(supports)
    return 1.0 - min(k, n) * total / (n * total) if total else float("nan")


POLICIES: dict = {
    "first_k": lambda rec, k: list(range(min(k, len(rec["texts"])))),
    "longest_first": lambda rec, k: select_topk(
        np.array([len(t) for t in rec["texts"]], dtype=float), k),
    "doc_round_robin": lambda rec, k: round_robin_groups(rec["titles"], k),
    "bm25": lambda rec, k: select_topk(bm25_scores(rec["question"], rec["texts"]), k),
    "tfidf": lambda rec, k: select_topk(tfidf_cosine_scores(rec["question"], rec["texts"]), k),
}

#: Policies that use no question information (plan §2 row 6: the fixed set from
#: which best-fixed is chosen; dynamic rankers are never the fixed reference).
FIXED_POLICIES = ("first_k", "longest_first", "doc_round_robin")


def policy_curves(records: Sequence[Mapping], budgets: Sequence[int] = BUDGETS,
                  policies: Mapping[str, Callable] | None = None) -> dict:
    """Mean miss rate per policy per budget + exact oracle curve."""
    policies = POLICIES if policies is None else policies
    curves = {}
    for name, fn in policies.items():
        if name == "random_expected":
            continue
        means = []
        for k in budgets:
            vals = [miss_rate(fn(rec, k), rec["supports"]) for rec in records]
            means.append(float(np.mean(vals)))
        curves[name] = means
    # Oracle: supports first (exact up to ties).
    oracle = []
    for k in budgets:
        vals = []
        for rec in records:
            sup_idx = [i for i, f in enumerate(rec["supports"]) if f]
            rest = [i for i in range(len(rec["texts"])) if i not in set(sup_idx)]
            vals.append(miss_rate((sup_idx + rest)[:k], rec["supports"]))
        oracle.append(float(np.mean(vals)))
    curves["oracle"] = oracle
    # Random-expected: exact per-question expectation, then mean.
    curves["random_expected"] = [
        float(np.mean([random_expected_miss(rec["supports"], k) for rec in records]))
        for k in budgets]
    return curves


def normalized_area(curve: Sequence[float], budgets: Sequence[int] = BUDGETS) -> float:
    """Trapezoidal area over k/kmax (all curves share the grid)."""
    xs = np.asarray(budgets, dtype=float) / max(budgets)
    return float(np.trapezoid(np.asarray(curve, dtype=float), xs))


def headroom(fixed_curve: Sequence[float], ref_curve: Sequence[float],
             budgets: Sequence[int] = BUDGETS) -> float:
    """Relative headroom area(fixed − ref)/area(fixed) (D1-0 opportunity rule)."""
    denom = normalized_area(fixed_curve, budgets)
    if denom <= 0:
        return float("nan")
    return (denom - normalized_area(ref_curve, budgets)) / denom
