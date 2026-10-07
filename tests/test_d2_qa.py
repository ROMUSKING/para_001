"""D2-QA Rung-0 tests: loader discipline, ranker sanity, gate maths, tripwires."""

import json
from pathlib import Path

import numpy as np
import pytest

from adjointrwm.analysis.allocation import assert_predictions_ignore_futures
from adjointrwm.domains import qa_context as qa

ROOT = Path(__file__).resolve().parent.parent


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



def _load_bridge_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_d2_qa_reader_bridge", ROOT / "scripts/run_d2_qa_reader_bridge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_match_accepts_aliases_and_ignores_case_and_articles():
    module = _load_bridge_script()
    assert module.exact_match("Walt Disney", ["Walt Disney", "Disney"])
    assert module.exact_match("  the walt  DISNEY! ", ["Walt Disney"])
    assert module.exact_match("Disney", ["Walt Disney", "Disney"])
    assert not module.exact_match("Mickey Mouse", ["Walt Disney", "Disney"])


def test_token_f1_partial_credit():
    module = _load_bridge_script()
    assert module.token_f1("Walt Disney", ["Walt Disney"]) == pytest.approx(1.0)
    assert module.token_f1("nothing relevant", ["Walt Disney"]) == pytest.approx(0.0)
    partial = module.token_f1("Disney company", ["Walt Disney"])
    assert 0.0 < partial < 1.0


def test_prompt_contains_ranked_titled_paragraphs():
    module = _load_bridge_script()
    prompt = module.build_prompt("Q?", ["aaa", "bbb", "ccc"], [2, 0], ["T2", "T0", "T1"])
    assert prompt.index("[T1]") < prompt.index("[T2]")
    assert "ccc" in prompt and "aaa" in prompt and "Q?" in prompt


def test_smoke_malformed_definition():
    module = _load_bridge_script()
    assert module.is_malformed("", "Q?")
    assert module.is_malformed("Q?", "Q?")
    assert not module.is_malformed("Walt Disney", "Who developed it?")
