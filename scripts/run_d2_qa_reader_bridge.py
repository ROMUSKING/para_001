#!/usr/bin/env python3
"""scripts/run_d2_qa_reader_bridge.py — frozen reader over frozen paragraph sets.

Implements docs/plans/2026-10-07-d2-qa-reader-bridge-plan.md: smoke probe first
(stops the run on failure), then oracle/BM25/longest sets at k=4,8 over the first
300 frozen validation ids, greedy decoding, EM (with answer_aliases) + token-F1,
per-question rows. Inference only; no training, no Drive dependency.
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
    POLICIES,
    build_registry,
    load_questions,
    select_topk,
)

PROMPT_SYSTEM = ("Answer the question using only the context. "
                 "Reply with the answer alone, no explanation.")
PROMPT_TEMPLATE = ("Question: {question}\nContext:\n{context}\nAnswer in a few words:")

READER_SWITCH_RATIONALE = (
    "v1 used SmolLM3-3B (base, non-instruction-tuned) with an ad-hoc prompt and "
    "the reader continued context instead of answering (EM 0 everywhere; "
    "non-diagnostic). v2 uses Phi-4-mini-instruct for the instruction-following "
    "property plus MIT licence clearance — selected for that property, not for "
    "any observed score. No further reader swap on v2 outcomes.")


def normalize_answer(text: str) -> str:
    """Lowercase, strip articles/punctuation/extra whitespace (SQuAD-style EM)."""
    import re
    import string

    text = text.lower()
    text = "".join(c for c in text if c not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: list) -> bool:
    """True if normalized prediction matches the answer or any alias."""
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(a) for a in answers)


def token_f1(prediction: str, answers: list) -> float:
    """Max token-F1 over the answer and its aliases."""
    pred = normalize_answer(prediction).split()
    best = 0.0
    for ans in answers:
        gold = normalize_answer(ans).split()
        common = {}
        for t in pred:
            common[t] = min(pred.count(t), gold.count(t))
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred) if pred else 0.0
        recall = num_same / len(gold) if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        best = max(best, f1)
    return best


def build_prompt(question: str, texts: list, selected: list, titles: list,
                 tokenizer=None) -> str:
    """Frozen prompt: chat template (when a tokenizer is supplied) or the legacy
    flat template otherwise. Selected paragraphs in rank order, titled."""
    body = PROMPT_TEMPLATE.format(
        question=question,
        context="\n".join("[%s] %s" % (titles[i], texts[i]) for i in selected))
    if tokenizer is None:
        return body
    messages = [{"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": body}]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def is_malformed(output: str, question: str) -> bool:
    """Frozen v2 smoke definition: empty, verbatim question echo, uncapped runaway,
    or structural non-answer (more than one sentence, or any question mark).

    Legitimate answers are short noun phrases; the v1 failure emitted
    multi-sentence context continuations.
    """
    import re

    if not output.strip():
        return True
    if normalize_answer(output) == normalize_answer(question):
        return True
    stripped = output.strip()
    if "?" in stripped:
        return True
    sentences = [s for s in re.split(r"[.!?]+", stripped) if s.strip()]
    return len(sentences) > 1


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_reader(model_id: str, device: str):
    """Frozen greedy reader (transformers, fp16 on CUDA, fp32 on CPU)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, trust_remote_code=False).to(device).eval()
    return tokenizer, model


def generate(tokenizer, model, prompt: str, device: str, max_new_tokens: int = 32) -> str:
    """Greedy decode, single sequence (frozen)."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="D2-QA reader bridge")
    parser.add_argument("--model-id", default="microsoft/Phi-4-mini-instruct")
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-ids", required=True,
                        help="frozen id list (one per line) from the Rung-0 artefact")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-questions", type=int, default=300)
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--learned-sets", default=None,
                        help="Rung-1 learned paragraph orders (qid -> ranked index list "
                             "JSON, from train_d2_rung1_ranker); adds learned_k4/learned_k8 "
                             "arms to the frozen reader eval. No other arm changes.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args(argv)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    train = load_questions(args.train_file, "train")
    dev = load_questions(args.dev_file, "dev")
    reg = build_registry(train, dev, tune_cap=10 ** 9, val_cap=10 ** 9, seed=0)
    wanted = [l.strip() for l in open(args.validation_ids) if l.strip()][:args.n_questions]
    by_id = {r["qid"]: r for r in reg["validation"]}
    missing = [q for q in wanted if q not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} frozen ids not in registry (e.g. {missing[0]})")
    questions = [by_id[q] for q in wanted]

    tokenizer, model = load_reader(args.model_id, device)

    def answer_fn(rec, selected):
        prompt = build_prompt(rec["question"], rec["texts"], selected, rec["titles"],
                              tokenizer)
        return generate(tokenizer, model, prompt, device), prompt

    # Smoke probe (frozen v2): fresh questions 20-39, oracle sets at k=8.
    smoke = questions[20:40]
    malformed = 0
    for rec in smoke:
        sup_idx = [i for i, f in enumerate(rec["supports"]) if f]
        rest = [i for i in range(len(rec["texts"])) if i not in set(sup_idx)]
        out, _ = answer_fn(rec, (sup_idx + rest)[:8])
        if is_malformed(out, rec["question"]) or len(out.split()) > 32:
            malformed += 1
    print(f"smoke malformed: {malformed}/20", flush=True)
    if malformed >= 6:
        raise SystemExit("smoke probe failed (>=6/20 malformed): stopping, design returns "
                         "for review instead of swapping readers")
    if args.smoke_only:
        return 0

    sets = {
        "oracle": lambda rec, k: ([i for i, f in enumerate(rec["supports"]) if f]
                                  + [i for i in range(len(rec["texts"]))
                                     if not rec["supports"][i]])[:k],
        "bm25": lambda rec, k: POLICIES["bm25"](rec, k),
        "longest": lambda rec, k: POLICIES["longest_first"](rec, k),
    }
    if args.learned_sets:
        learned = {q: list(map(int, v)) for q, v in
                   json.load(open(args.learned_sets)).items()}
        missing_l = [q["qid"] for q in questions if q["qid"] not in learned]
        if missing_l:
            raise SystemExit(f"{len(missing_l)} questions lack learned sets")
        sets["learned"] = lambda rec, k: learned[rec["qid"]][:k]
    rows = []
    for rec in questions:
        answers = [rec.get("answer", "")] + rec.get("answer_aliases", [])
        row = {"qid": rec["qid"]}
        for k in args.budgets:
            for name, fn in sets.items():
                out, _ = answer_fn(rec, fn(rec, k))
                row[f"{name}_k{k}_em"] = exact_match(out, answers)
                row[f"{name}_k{k}_f1"] = token_f1(out, answers)
                row[f"{name}_k{k}_text"] = out
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f"  {len(rows)}/{len(questions)}", flush=True)

    # Records need answer fields for EM: reload lightweight (already in recs? no).
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "benchmark": "d2_qa_reader_bridge",
        "mode": "real",
        "device": device,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": args.model_id,
        "reader_switch_rationale": READER_SWITCH_RATIONALE,
        "prompt_template_sha256": sha256_text(PROMPT_TEMPLATE),
        "prompt_system_sha256": sha256_text(PROMPT_SYSTEM),
        "budgets": args.budgets,
        "n_questions": len(rows),
        "smoke_malformed": malformed,
        "rows": rows,
    }
    (output_dir / "reader_bridge_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"Saved summary to {output_dir / 'reader_bridge_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
