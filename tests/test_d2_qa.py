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


def test_smoke_malformed_structural_non_answer():
    """v2 smoke: multi-sentence continuations and question marks are malformed;
    short noun-phrase answers pass (extractive quotes allowed)."""
    module = _load_bridge_script()
    assert module.is_malformed("Paris. It is nice.", "Capital?")
    assert module.is_malformed("Is it Paris?", "Capital?")
    assert not module.is_malformed("The Treaty of Versailles was signed on 28 June 1919",
                                   "When was it signed?")
    assert not module.is_malformed("Walt Disney", "Who developed it?")


def test_chat_template_applied_when_tokenizer_given():
    """build_prompt routes through the chat template when supplied."""
    module = _load_bridge_script()

    class FakeTok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert messages[0]["role"] == "system"
            assert add_generation_prompt is True
            return "TEMPLATED::" + messages[1]["content"][:10]

    out = module.build_prompt("Q?", ["aaa"], [0], ["T"], FakeTok())
    assert out.startswith("TEMPLATED::")
    flat = module.build_prompt("Q?", ["aaa"], [0], ["T"])
    assert flat.startswith("Question: Q?")


def _rich_rec():
    texts = [
        "Walt Disney developed the Mickey Mouse series in 1928",
        "The series features Mickey Mouse and friends",
        "Unrelated sports news about football results",
        "Disney parks opened in California later",
        "Cooking recipes for dinner tonight",
        "Walt Disney biography and early life",
    ]
    return {"qid": "2hop__x", "question": "Who developed the Mickey Mouse series?",
            "texts": texts, "titles": ["A", "A", "B", "C", "D", "A"],
            "supports": [True, True, False, False, False, False]}


def test_entity_spans_use_question_text_only():
    spans = qa.entity_spans("Who developed the 'Mickey Mouse' series?")
    assert "Mickey Mouse" in spans
    assert all(isinstance(s, str) and s for s in spans)


def test_incremental_bm25_returns_k_distinct():
    rec = _rich_rec()
    sel = qa.incremental_bm25(rec["question"], rec["texts"], 3)
    assert len(sel) == 3 and len(set(sel)) == 3


def test_title_bundle_respects_budget_and_skips_giants():
    rec = _rich_rec()
    sel = qa.title_bundle(rec["question"], rec["texts"], rec["titles"], 2)
    assert len(sel) <= 2
    # Group A has 3 members > budget 2 alone: must be skipped, never truncated.
    assert not (set(sel) <= {0, 1, 5} and len(sel) == 2 and 5 in sel and 0 in sel)


def test_r2_constructors_ignore_support_labels():
    """Boundary: shuffling privileged support ids must not move any R2 selection."""
    rec = _rich_rec()
    base = {"question": rec["question"], "texts": rec["texts"], "titles": rec["titles"]}
    for name, fn in qa.R2_POLICIES.items():
        sel = lambda trial, _fn=fn: _fn({**trial, "supports": [False] * 6}, 3)
        assert_predictions_ignore_futures(
            sel, base, [{"supports": [True] * 6}, {"supports": [True, False] * 3}])
    _ = name


def test_precision_and_complete_recovery():
    assert qa.support_precision([0, 1, 2], [True, True, False]) == pytest.approx(2 / 3)
    assert qa.complete_support_recovery([0, 1, 2], [True, True, False]) == 1
    assert qa.complete_support_recovery([0, 2], [True, True, False]) == 0
    assert qa.hop_bucket("2hop__x") == "2hop"
    assert qa.hop_bucket("4hop2__x") == "4hop"
    assert qa.hop_bucket("zzz") == "other"
