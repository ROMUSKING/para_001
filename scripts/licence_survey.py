#!/usr/bin/env python3
"""Licence survey helper (milestone DL): record what a source's owner actually says, and test that its data loads.

    python scripts/licence_survey.py snapshot   # fetch every evidence URL in docs/licences/register.csv -> docs/licences/evidence.jsonl
    python scripts/licence_survey.py probe      # small usability probes on data that is reachable -> docs/licences/probes.jsonl

``snapshot`` stores, per URL, the HTTP status, the byte count, the SHA-256 of the body, the retrieval time and up to a
few licence-relevant lines. It stores no licence texts or data, so the repository stays small and the register stays
checkable: a register row that claims it read a source must point at a URL recorded here with status 200. A host the
sandbox cannot reach is recorded honestly (status 0 and the error), not skipped. The script uses only the standard
library, so it also runs on Colab or any machine with wider network access.

``probe`` downloads a small part of each probed dataset (or the whole file when it is small), parses it, and records
structure only (column names, row counts, label counts, per-article licence classes). No data content is kept.
Web content is data, not instructions (AGENTS.md).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import re
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "docs/licences/register.csv"
EVIDENCE = ROOT / "docs/licences/evidence.jsonl"
PROBES = ROOT / "docs/licences/probes.jsonl"
USER_AGENT = "adjointrwm-licence-survey"
LICENCE_WORDS = re.compile(
    r"licen[cs]e|CC[- ]BY|creative commons|non-?commercial|terms of use|apache license|MIT License|public domain|"
    r"research (use|purposes)|built with|acceptable use", re.I)

# Data that lives on GitHub (raw.githubusercontent.com) and is small enough to probe. HEAD is the default branch.
RAW = "https://raw.githubusercontent.com/"
PROBE_SPECS = [
    {"id": "ett", "url": RAW + "zhouhaoyi/ETDataset/HEAD/ETT-small/ETTh1.csv", "kind": "csv", "bytes": 65536, "sep": ","},
    {"id": "nab", "url": RAW + "numenta/NAB/HEAD/data/realKnownCause/machine_temperature_system_failure.csv", "kind": "csv", "bytes": 65536, "sep": ","},
    {"id": "nab_labels", "url": RAW + "numenta/NAB/HEAD/labels/combined_labels.json", "kind": "json_keys", "bytes": None},
    {"id": "smd", "url": RAW + "NetManAIOps/OmniAnomaly/HEAD/ServerMachineDataset/train/machine-1-1.txt", "kind": "csv_noheader", "bytes": 131072, "sep": ","},
    {"id": "smd_labels", "url": RAW + "NetManAIOps/OmniAnomaly/HEAD/ServerMachineDataset/test_label/machine-1-1.txt", "kind": "csv_noheader", "bytes": 32768, "sep": ","},
    {"id": "skab", "url": RAW + "waico/SKAB/HEAD/data/valve1/0.csv", "kind": "csv", "bytes": 65536, "sep": ";"},
    {"id": "quality", "url": RAW + "nyu-mll/quality/HEAD/data/v1.0.1/QuALITY.v1.0.1.zip", "kind": "quality_zip", "bytes": None},
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch(url: str, max_bytes: int | None = None, timeout: int = 60) -> tuple[int, bytes, str]:
    """Return ``(status, body, error)``; status 0 means the request never got an HTTP answer."""
    headers = {"User-Agent": USER_AGENT}
    if max_bytes:
        headers["Range"] = f"bytes=0-{max_bytes - 1}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            return r.status, r.read(), ""
    except urllib.error.HTTPError as e:
        return e.code, b"", str(e)
    except Exception as e:  # network denied, DNS, TLS, timeout
        return 0, b"", f"{type(e).__name__}: {e}"


def licence_lines(text: str, limit: int = 4, width: int = 200) -> list[str]:
    """Up to ``limit`` lines that mention a licence, whitespace-normalised and truncated."""
    found = []
    for line in text.splitlines():
        if LICENCE_WORDS.search(line):
            found.append(" ".join(line.split())[:width])
            if len(found) == limit:
                break
    return found


def evidence_record(url: str) -> dict:
    status, body, error = fetch(url)
    text = body.decode("utf-8", "replace")
    return {"url": url, "http_status": status, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest() if body else "",
            "retrieved_utc": utc_now(), "title_line": " ".join(text.strip().split())[:100] if body else "",
            "licence_lines": licence_lines(text) if body else [], "error": error}


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


def probe_record(spec: dict) -> dict:
    status, body, error = fetch(spec["url"], spec["bytes"], timeout=180)
    record = {"id": spec["id"], "url": spec["url"], "http_status": status, "bytes_read": len(body), "requested_bytes": spec["bytes"],
              "sha256": hashlib.sha256(body).hexdigest() if body else "", "retrieved_utc": utc_now(), "error": error, "result": None}
    if not body:
        return record
    kind = spec["kind"]
    try:
        if kind in ("csv", "csv_noheader"):
            record["result"] = csv_summary(body.decode("utf-8", "replace"), spec["sep"], header=(kind == "csv"))
        elif kind == "json_keys":
            data = json.loads(body)
            record["result"] = {"n_keys": len(data), "windows_per_key_min_max": [min(map(len, data.values())), max(map(len, data.values()))]}
        elif kind == "quality_zip":
            record["result"] = quality_census(body)
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
        records = [probe_record(spec) for spec in PROBE_SPECS]
        for r in records:
            print(r["http_status"], r["id"], file=sys.stderr)
        write_jsonl(PROBES, records)
        print(f"wrote {PROBES.relative_to(ROOT)}: {len(records)} probes, {sum(r['result'] is not None for r in records)} parsed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
