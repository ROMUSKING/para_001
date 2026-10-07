"""D2-QA Rung-0 tests: loader discipline, ranker sanity, gate maths, tripwires."""

import json

import numpy as np
import pytest

from adjointrwm.analysis.allocation import assert_predictions_ignore_futures
from adjointrwm.domains import qa_context as qa


def _rec(n_docs=6, supports=(0, 2), question="who developed the series"):
    texts = [f"paragraph {i} about " + ("the series developed by walt disney" if i in supports
                                        else "unrelated filler words here")
             for i in range(n_docs)]
    return {"qid": "q", "question": question, "texts": texts,
            "titles": [f"doc{i // 2}" for i in range(n_docs)],
            "supports": [i in supports for i in range(n_docs)]}


def test_loader_refuses_test_files(tmp_path):
    p = tmp_path / "musique_full_v1.0_test.jsonl"
    p.write_text("{}\n")
    with pytest.raises(ValueError, match="never downloaded or read"):
        qa.load_questions(str(p), "test")


def test_loader_keeps_answerable_with_supports_only(tmp_path):
    rows = [
        {"id": "a", "question": "q?", "answerable": True,
         "paragraphs": [{"idx": 0, "is_supporting": True, "paragraph_text": "t", "title": "d"}]},
        {"id": "b", "question": "q?", "answerable": False,
         "paragraphs": [{"idx": 0, "is_supporting": False, "paragraph_text": "t", "title": "d"}]},
        {"id": "c", "question": "q?", "answerable": True,
         "paragraphs": [{"idx": 0, "is_supporting": False, "paragraph_text": "t", "title": "d"}]},
    ]
    p = tmp_path / "dev.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    recs = qa.load_questions(str(p), "validation")
    assert [r["qid"] for r in recs] == ["a"]


def test_registry_rejects_cross_file_overlap():
    a = [{"qid": "x", "question": "q", "texts": ["t"], "titles": ["d"], "supports": [True]}]
    b = [{"qid": "x", "question": "q", "texts": ["t"], "titles": ["d"], "supports": [True]}]
    with pytest.raises(ValueError, match="deduplicate"):
        qa.build_registry(a, b)


def test_registry_caps_and_freezes_ids():
    train = [{**_rec(), "qid": f"t{i}"} for i in range(10)]
    dev = [{**_rec(), "qid": f"d{i}"} for i in range(10)]
    first = qa.build_registry(train, dev, tune_cap=4, val_cap=3, seed=0)
    second = qa.build_registry(train, dev, tune_cap=4, val_cap=3, seed=0)
    assert first["tuning_ids"] == second["tuning_ids"]
    assert len(first["tuning"]) == 4 and len(first["validation"]) == 3
    assert not (set(first["tuning_ids"]) & set(first["validation_ids"]))


def test_bm25_prefers_supporting_text():
    rec = _rec()
    scores = qa.bm25_scores(rec["question"], rec["texts"])
    assert scores.shape == (6,)
    assert set(qa.select_topk(scores, 2)) == {0, 2}


def test_tfidf_prefers_supporting_text():
    rec = _rec()
    scores = qa.tfidf_cosine_scores(rec["question"], rec["texts"])
    assert set(qa.select_topk(scores, 2)) == {0, 2}


def test_oracle_zero_when_budget_covers_supports():
    rec = _rec()
    curves = qa.policy_curves([rec], budgets=(1, 2, 6))
    assert curves["oracle"][1] == 0.0 and curves["oracle"][2] == 0.0
    assert curves["oracle"][0] == pytest.approx(0.5)


def test_random_expected_is_exact():
    assert qa.random_expected_miss([True, True, False, False], 2) == pytest.approx(0.5)
    assert qa.random_expected_miss([True, False], 2) == pytest.approx(0.0)


def test_deployable_rankers_ignore_support_labels():
    """Boundary tripwire: shuffling privileged support ids must not move selections."""
    rec = _rec()
    base = {"question": rec["question"], "texts": rec["texts"]}
    for name in ("bm25", "tfidf"):
        fn = lambda trial, _n=name: qa.select_topk(  # noqa: E731
            (qa.bm25_scores if _n == "bm25" else qa.tfidf_cosine_scores)(
                trial["question"], trial["texts"]), 3)
        assert_predictions_ignore_futures(
            fn, base, [{"supports": [True] * 6}, {"supports": [False] * 6}])


def test_fixed_policies_exclude_dynamic_rankers():
    """best-fixed is chosen from question-blind policies only (plan §2 row 6)."""
    assert set(qa.FIXED_POLICIES) == {"first_k", "longest_first", "doc_round_robin"}
    assert "bm25" not in qa.FIXED_POLICIES and "tfidf" not in qa.FIXED_POLICIES
