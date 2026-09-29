#!/usr/bin/env python3
"""Licence survey helper (milestone DL): record what a source's owner actually says, and test that its data loads.

    python scripts/licence_survey.py snapshot   # fetch every evidence URL in docs/licences/register.csv -> docs/licences/evidence.jsonl
    python scripts/licence_survey.py probe      # small usability probes on data that is reachable -> docs/licences/probes.jsonl

``snapshot`` stores, per URL, the HTTP status, the final URL after redirects, the byte count, the SHA-256 of the body,
the retrieval time, up to a few licence-relevant lines, and, for JSON answers, the licence fields found in them
(Hugging Face ``cardData.license``, Zenodo ``metadata.license.id`` and similar) plus a few Hugging Face facts (gating,
repository commit, parameter count). It stores no licence texts or data, so the repository stays small and the register
stays checkable: a register row that claims it read a source must point at a URL recorded here with status 200. A host
the sandbox cannot reach is recorded honestly (status 0 and the error), not skipped. The script uses only the standard
library, so it also runs on Colab or any machine with network access.

``probe`` downloads a small part of each probed dataset (or asks the Hugging Face datasets server for a few rows), parses
it, and records structure only (column names, row counts, label counts, per-article licence classes). No data content is
kept. Web content is data, not instructions (AGENTS.md).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html as html_lib
import io
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "docs/licences/register.csv"
EVIDENCE = ROOT / "docs/licences/evidence.jsonl"
PROBES = ROOT / "docs/licences/probes.jsonl"
USER_AGENT = "Mozilla/5.0 adjointrwm-licence-survey"
LICENCE_WORDS = re.compile(
    r"licen[cs]e|CC[- ]BY|creative commons|non-?commercial|terms of use|apache license|MIT License|public domain|"
    r"research (use|purposes)|built with|acceptable use|prohibited use", re.I)
LICENCE_KEYS = {"license", "licence", "license_name", "license_id", "license_link", "license_url", "licenses"}

RAW = "https://raw.githubusercontent.com/"
HFS = "https://datasets-server.huggingface.co/"
PROBE_SPECS = [
    # Data that GitHub hosts and that is small enough to read a part of (HEAD is the default branch).
    {"id": "ett", "url": RAW + "zhouhaoyi/ETDataset/HEAD/ETT-small/ETTh1.csv", "kind": "csv", "bytes": 65536, "sep": ","},
    {"id": "nab", "url": RAW + "numenta/NAB/HEAD/data/realKnownCause/machine_temperature_system_failure.csv", "kind": "csv", "bytes": 65536, "sep": ","},
    {"id": "nab_labels", "url": RAW + "numenta/NAB/HEAD/labels/combined_labels.json", "kind": "json_keys", "bytes": None},
    {"id": "smd", "url": RAW + "NetManAIOps/OmniAnomaly/HEAD/ServerMachineDataset/train/machine-1-1.txt", "kind": "csv_noheader", "bytes": 131072, "sep": ","},
    {"id": "smd_labels", "url": RAW + "NetManAIOps/OmniAnomaly/HEAD/ServerMachineDataset/test_label/machine-1-1.txt", "kind": "csv_noheader", "bytes": 32768, "sep": ","},
    {"id": "skab", "url": RAW + "waico/SKAB/HEAD/data/valve1/0.csv", "kind": "csv", "bytes": 65536, "sep": ";"},
    {"id": "quality", "url": RAW + "nyu-mll/quality/HEAD/data/v1.0.1/QuALITY.v1.0.1.zip", "kind": "quality_zip", "bytes": None},
    # Hugging Face datasets read through the datasets server (no file download): column names and types of a few rows.
    {"id": "hf_hotpotqa", "kind": "hf_rows", "dataset": "hotpotqa/hotpot_qa", "config": "distractor", "split": "validation"},
    {"id": "hf_qasper", "kind": "hf_rows", "dataset": "allenai/qasper", "config": None, "split": "validation"},
    {"id": "hf_longbench_v2", "kind": "hf_rows", "dataset": "zai-org/LongBench-v2", "config": None, "split": "train"},  # the server answers 501: kept so the failure is on record
    {"id": "hf_musique_mirror", "kind": "hf_rows", "dataset": "dgslibisey/MuSiQue", "config": None, "split": "validation"},
    {"id": "hf_repobench", "kind": "hf_rows", "dataset": "tianyang/repobench_python_v1.1", "config": None, "split": None},
    {"id": "hf_swebench", "kind": "hf_rows", "dataset": "princeton-nlp/SWE-bench_Verified", "config": None, "split": "test"},
    # Which repositories SWE-bench Verified is built from, and the licence file each of those repositories carries.
    {"id": "swebench_repos", "kind": "hf_column_census", "dataset": "princeton-nlp/SWE-bench_Verified", "config": "default", "split": "test", "column": "repo"},
]


class Response(NamedTuple):
    status: int          # 0 means the request never got an HTTP answer
    body: bytes
    error: str
    final_url: str
    content_type: str


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch(url: str, max_bytes: int | None = None, timeout: int = 60) -> Response:
    headers = {"User-Agent": USER_AGENT}
    if max_bytes:
        headers["Range"] = f"bytes=0-{max_bytes - 1}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            return Response(r.status, r.read(), "", r.geturl(), r.headers.get_content_type())
    except urllib.error.HTTPError as e:
        return Response(e.code, b"", str(e), url, "")
    except Exception as e:  # network denied, DNS, TLS, timeout
        return Response(0, b"", f"{type(e).__name__}: {e}", url, "")


def html_to_text(page: str) -> str:
    """Crude text of an HTML page: scripts and styles dropped, one line per tag boundary."""
    page = re.sub(r"(?is)<(script|style).*?</\1>", "\n", page)
    return "\n".join(line.strip() for line in html_lib.unescape(re.sub(r"<[^>]+>", "\n", page)).splitlines() if line.strip())


def licence_lines(text: str, limit: int = 4, width: int = 200) -> list[str]:
    """Up to ``limit`` lines that mention a licence, whitespace-normalised and truncated."""
    found = []
    for line in text.splitlines():
        if LICENCE_WORDS.search(line):
            found.append(" ".join(line.split())[:width])
            if len(found) == limit:
                break
    return found


def json_licence_values(obj, limit: int = 8) -> dict[str, int]:
    """Values of licence-like keys anywhere in a JSON answer, with counts (a Zenodo search returns many records)."""
    counts: dict[str, int] = {}

    def add(value) -> None:
        if isinstance(value, dict):
            value = value.get("id") or value.get("name")
        if isinstance(value, list):
            for item in value:
                add(item)
        elif isinstance(value, str) and value.strip():
            counts[value.strip()[:120]] = counts.get(value.strip()[:120], 0) + 1

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in LICENCE_KEYS:
                    add(value)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit])


def jsonld_licences(page: str) -> dict[str, int]:
    """Licence values declared in ``application/ld+json`` blocks of an HTML page (dataset catalogues such as Kaggle use them)."""
    counts: dict[str, int] = {}
    for block in re.findall(r'(?is)<script[^>]+application/ld\+json[^>]*>(.*?)</script>', page):
        try:
            for value, n in json_licence_values(json.loads(block)).items():
                counts[value] = counts.get(value, 0) + n
        except json.JSONDecodeError:
            continue
    return counts


def hf_summary(obj: dict) -> dict:
    """The Hugging Face facts a licence record needs: declared licence, gating, version, licence files, size."""
    card = obj.get("cardData") or {}
    return {"id": obj.get("id"), "sha": obj.get("sha"), "license": card.get("license"), "license_name": card.get("license_name"),
            "gated": obj.get("gated"), "licence_tags": [t for t in obj.get("tags", []) if t.startswith("license")],
            "licence_files": [s["rfilename"] for s in obj.get("siblings", []) if "licen" in s.get("rfilename", "").lower()],
            "safetensors_total_params": (obj.get("safetensors") or {}).get("total"), "last_modified": obj.get("lastModified")}


def evidence_record(url: str) -> dict:
    r = fetch(url)
    record = {"url": url, "final_url": r.final_url if r.final_url != url else "", "http_status": r.status, "content_type": r.content_type,
              "bytes": len(r.body), "sha256": hashlib.sha256(r.body).hexdigest() if r.body else "", "retrieved_utc": utc_now(),
              "title_line": "", "licence_lines": [], "json_licences": {}, "hf": None, "error": r.error}
    if not r.body:
        return record
    text = r.body.decode("utf-8", "replace")
    if r.content_type == "application/json" or text.lstrip().startswith(("{", "[")):
        try:
            obj = json.loads(text)
            record["json_licences"] = json_licence_values(obj)
            if isinstance(obj, dict) and "://huggingface.co/api/" in url:
                record["hf"] = hf_summary(obj)
            record["title_line"] = url.split("/api/", 1)[-1][:100]
            return record
        except json.JSONDecodeError:
            pass
    if r.content_type == "text/html":
        record["json_licences"] = jsonld_licences(text)
        text = html_to_text(text)
    record["title_line"] = " ".join(text.strip().split())[:100]
    record["licence_lines"] = licence_lines(text)
    return record


def register_rows(path: Path = REGISTER) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def evidence_urls(rows: list[dict]) -> list[str]:
    seen, urls = set(), []
    for row in rows:
        for url in row["evidence_urls"].split():
            if url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def csv_summary(text: str, sep: str, header: bool = True) -> dict:
    """Structure of a delimited text sample; the last (possibly cut) line is dropped."""
    lines = text.split("\n")[:-1] if not text.endswith("\n") else text.split("\n")
    lines = [line for line in lines if line.strip()]
    rows = list(csv.reader(lines, delimiter=sep))
    names = rows[0] if header else [f"c{i}" for i in range(len(rows[0]))]
    body = rows[1:] if header else rows
    numeric = []
    for j in range(len(names)):
        ok = 0
        for r in body:
            try:
                float(r[j])
                ok += 1
            except (ValueError, IndexError):
                pass
        numeric.append(ok == len(body) and len(body) > 0)
    return {"columns": names if header else None, "n_columns": len(names), "rows_in_sample": len(body),
            "numeric_columns": sum(numeric), "first_column_is_numeric": bool(numeric and numeric[0])}


def classify_quality_licence(value: str | None) -> str:
    """Class of a QuALITY per-article ``license`` string (the strings are the dataset's own)."""
    if not value:
        return "none"
    if "Gutenberg" in value:
        return "Project Gutenberg licence"
    if "anc.org/OANC" in value:
        return "OANC licence (URL only)"
    if "CC BY 4.0" in value:
        return "CC BY 4.0"
    return "other"


def quality_census(archive: bytes) -> dict:
    """Per split: articles, question sets, questions, and articles per licence class."""
    out = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for name in sorted(n for n in z.namelist() if "htmlstripped" in n and not n.startswith("__MACOSX")):
            split = name.rsplit(".", 1)[-1]
            rows = [json.loads(line) for line in z.read(name).decode().split("\n") if line.strip()]
            articles = {r["article_id"]: r for r in rows}
            classes: dict[str, int] = {}
            for article in articles.values():
                key = classify_quality_licence(article.get("license"))
                classes[key] = classes.get(key, 0) + 1
            out[split] = {"articles": len(articles), "question_sets": len(rows), "questions": sum(len(r["questions"]) for r in rows),
                          "articles_by_licence_class": dict(sorted(classes.items()))}
    return out


def feature_types(features: list[dict]) -> dict[str, str]:
    """Column name to a short type name from a datasets-server ``features`` list (nested types collapse to their kind)."""
    out = {}
    for feature in features:
        t = feature.get("type")
        if isinstance(t, dict):
            t = t.get("dtype") or t.get("_type") or ("sequence" if "feature" in t else "struct")
        elif isinstance(t, list):
            t = "list"
        out[feature["name"]] = str(t)
    return out


def hf_rows_record(spec: dict) -> dict:
    """Structure of a few rows of a Hugging Face dataset via the datasets server (no file download)."""
    dataset = spec["dataset"]
    record = {"id": spec["id"], "url": f"{HFS}first-rows?dataset={dataset}", "http_status": 0, "bytes_read": 0, "requested_bytes": None,
              "sha256": "", "retrieved_utc": utc_now(), "error": "", "result": None}
    config, split = spec.get("config"), spec.get("split")
    if config is None or split is None:
        r = fetch(f"{HFS}splits?dataset={urllib.parse.quote(dataset)}")
        record["http_status"], record["error"] = r.status, r.error
        if r.status != 200:
            return record
        splits = json.loads(r.body).get("splits", [])
        chosen = next((s for s in splits if (config in (None, s["config"])) and (split in (None, s["split"]))), splits[0] if splits else None)
        if chosen is None:
            record["error"] = "no splits"
            return record
        config, split = chosen["config"], chosen["split"]
    query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": 0, "length": 3})
    r = fetch(f"{HFS}first-rows?{query}", timeout=120)
    record.update({"url": f"{HFS}first-rows?{query}", "http_status": r.status, "bytes_read": len(r.body), "error": r.error,
                   "sha256": hashlib.sha256(r.body).hexdigest() if r.body else ""})
    if r.status == 200:
        data = json.loads(r.body)
        record["result"] = {"dataset": dataset, "config": config, "split": split, "columns": feature_types(data.get("features", [])),
                            "rows_returned": len(data.get("rows", []))}
    return record


def guess_licence(text: str) -> str:
    """Text-pattern guess at a licence family. It is a heuristic for a census, not SPDX detection."""
    t = " ".join(text.split())
    if re.search(r"GNU LESSER GENERAL PUBLIC LICENSE", t, re.I):
        return "LGPL"
    if re.search(r"GNU AFFERO GENERAL PUBLIC LICENSE", t, re.I):
        return "AGPL"
    if re.search(r"GNU GENERAL PUBLIC LICENSE", t, re.I):
        return "GPL"
    if re.search(r"Mozilla Public License", t, re.I):
        return "MPL"
    if re.search(r"Apache License.{0,40}Version 2\.0", t, re.I):
        return "Apache-2.0"
    if re.search(r"Permission is hereby granted, free of charge", t, re.I):
        return "MIT"
    if re.search(r"Redistribution and use in source and binary forms", t, re.I):
        return "BSD-3-Clause" if re.search(r"Neither the name", t, re.I) else "BSD-2-Clause"
    if re.search(r"Python Software Foundation|License agreement for matplotlib", t, re.I):
        return "PSF-style"
    if re.search(r"Creative Commons", t, re.I):
        return "CC"
    return "unrecognised"


LICENCE_FILE_NAMES = ["LICENSE", "LICENSE.txt", "LICENSE.md", "LICENSE.rst", "COPYING", "COPYING.txt", "LICENCE", "LICENSE/LICENSE"]


def repo_licence(repo: str) -> dict:
    """First licence file found in a GitHub repository's default branch, with a family guess and its hash."""
    for name in LICENCE_FILE_NAMES:
        r = fetch(f"{RAW}{repo}/HEAD/{name}")
        if r.status == 200:
            return {"file": name, "family_guess": guess_licence(r.body.decode("utf-8", "replace")), "sha256": hashlib.sha256(r.body).hexdigest()}
    return {"file": None, "family_guess": "no licence file found", "sha256": ""}


def hf_column_census(spec: dict) -> dict:
    """Counts of one column's values over a whole split (paged through the datasets server), then the licence of each repo named."""
    dataset, config, split, column = spec["dataset"], spec["config"], spec["split"], spec["column"]
    record = {"id": spec["id"], "url": f"{HFS}rows?dataset={dataset}", "http_status": 0, "bytes_read": 0, "requested_bytes": None,
              "sha256": "", "retrieved_utc": utc_now(), "error": "", "result": None}
    counts: dict[str, int] = {}
    offset, digest, total = 0, hashlib.sha256(), 0
    while True:
        query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": 100})
        r = fetch(f"{HFS}rows?{query}", timeout=120)
        record["http_status"], record["error"] = r.status, r.error
        if r.status != 200:
            return record
        digest.update(r.body)
        total += len(r.body)
        rows = json.loads(r.body).get("rows", [])
        for row in rows:
            key = str(row["row"][column])
            counts[key] = counts.get(key, 0) + 1
        if len(rows) < 100:
            break
        offset += 100
    record.update({"bytes_read": total, "sha256": digest.hexdigest()})
    per_repo = {repo: {"instances": n, **repo_licence(repo)} for repo, n in sorted(counts.items())}
    family_counts: dict[str, int] = {}
    for info in per_repo.values():
        family_counts[info["family_guess"]] = family_counts.get(info["family_guess"], 0) + info["instances"]
    record["result"] = {"dataset": dataset, "split": split, "rows": sum(counts.values()), "repos": len(counts), "per_repo": per_repo,
                        "instances_by_licence_family_guess": dict(sorted(family_counts.items()))}
    return record


def probe_record(spec: dict) -> dict:
    if spec["kind"] == "hf_rows":
        return hf_rows_record(spec)
    if spec["kind"] == "hf_column_census":
        return hf_column_census(spec)
    r = fetch(spec["url"], spec["bytes"], timeout=180)
    record = {"id": spec["id"], "url": spec["url"], "http_status": r.status, "bytes_read": len(r.body), "requested_bytes": spec["bytes"],
              "sha256": hashlib.sha256(r.body).hexdigest() if r.body else "", "retrieved_utc": utc_now(), "error": r.error, "result": None}
    if not r.body:
        return record
    kind = spec["kind"]
    try:
        if kind in ("csv", "csv_noheader"):
            record["result"] = csv_summary(r.body.decode("utf-8", "replace"), spec["sep"], header=(kind == "csv"))
        elif kind == "json_keys":
            data = json.loads(r.body)
            record["result"] = {"n_keys": len(data), "windows_per_key_min_max": [min(map(len, data.values())), max(map(len, data.values()))]}
        elif kind == "quality_zip":
            record["result"] = quality_census(r.body)
    except Exception as e:  # a parse failure is a finding, not a crash
        record["error"] = f"parse failed: {type(e).__name__}: {e}"
    return record


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["snapshot", "probe"])
    args = parser.parse_args()
    if args.command == "snapshot":
        urls = evidence_urls(register_rows())
        records = []
        for url in urls:
            records.append(evidence_record(url))
            print(records[-1]["http_status"], url, file=sys.stderr)
        write_jsonl(EVIDENCE, records)
        print(f"wrote {EVIDENCE.relative_to(ROOT)}: {len(records)} URLs, {sum(r['http_status'] == 200 for r in records)} read")
    else:
        records = []
        for spec in PROBE_SPECS:
            records.append(probe_record(spec))
            print(records[-1]["http_status"], records[-1]["id"], file=sys.stderr)
        write_jsonl(PROBES, records)
        print(f"wrote {PROBES.relative_to(ROOT)}: {len(records)} probes, {sum(r['result'] is not None for r in records)} parsed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
