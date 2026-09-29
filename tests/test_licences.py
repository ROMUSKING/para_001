"""The licence register (docs/licences/register.csv) must not claim more than the recorded evidence supports."""

from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("licence_survey", ROOT / "scripts" / "licence_survey.py")
survey = importlib.util.module_from_spec(spec)
spec.loader.exec_module(survey)

KINDS = {"dataset", "model", "benchmark", "code"}
PERMISSIVE = {"yes", "conditions", "no", "unknown"}
VERIFICATION = {"primary_text_read", "owner_statement_read", "code_only_read", "no_licence_found", "unread"}
VERDICTS = {"adopt", "adopt_with_conditions", "avoid", "unverified"}
READ_LEVELS = {"primary_text_read", "owner_statement_read"}


@pytest.fixture(scope="module")
def rows():
    return survey.register_rows()


@pytest.fixture(scope="module")
def evidence():
    return {r["url"]: r for r in map(json.loads, survey.EVIDENCE.read_text().splitlines())}


@pytest.fixture(scope="module")
def probes():
    return {r["id"]: r for r in map(json.loads, survey.PROBES.read_text().splitlines())}


def test_register_is_well_formed(rows):
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)) and all(ids)
    for r in rows:
        assert r["kind"] in KINDS and r["permissive"] in PERMISSIVE and r["verification"] in VERIFICATION and r["verdict"] in VERDICTS, r["id"]
        assert r["licence_as_read"].strip() and r["domains"].strip(), r["id"]


def test_verdicts_follow_the_rubric(rows):
    for r in rows:
        if r["verdict"] == "adopt":
            assert r["permissive"] == "yes" and r["verification"] in READ_LEVELS, f"{r['id']}: adopt needs a permissive licence that was read"
        if r["verdict"] == "adopt_with_conditions":
            assert r["permissive"] in {"yes", "conditions"} and r["verification"] in READ_LEVELS, r["id"]
            assert r["conditions"].strip(), f"{r['id']}: adopt_with_conditions must state its conditions"
        if r["verdict"] == "avoid":
            assert r["permissive"] == "no", r["id"]
        if r["permissive"] == "unknown":
            assert r["verdict"] == "unverified", f"{r['id']}: an unknown licence cannot be adopted or avoided"
        if r["verification"] in {"code_only_read", "unread"}:
            assert r["verdict"] == "unverified", f"{r['id']}: only code or nothing was read, so the asset is unverified"


def test_every_read_claim_points_at_fetched_evidence(rows, evidence):
    for r in rows:
        urls = r["evidence_urls"].split()
        for url in urls:
            assert url in evidence, f"{r['id']}: {url} is not in evidence.jsonl (run `python scripts/licence_survey.py snapshot`)"
        if r["verification"] in READ_LEVELS | {"code_only_read"}:
            assert urls and all(evidence[u]["http_status"] == 200 and evidence[u]["sha256"] for u in urls), f"{r['id']}: claims a read but a URL was not fetched"
        if r["verification"] == "no_licence_found":
            statuses = [evidence[u]["http_status"] for u in urls]
            assert 404 in statuses and 200 in statuses and 0 not in statuses, f"{r['id']}: an absent licence needs a 404 licence path and a readable README"
        if r["verification"] == "unread":
            assert not any(evidence[u]["http_status"] == 200 for u in urls), f"{r['id']}: evidence was fetched, so update the verification level"


def test_probe_ids_exist_and_parsed(rows, probes):
    for r in rows:
        for pid in r["probe"].split():
            assert pid in probes, f"{r['id']}: unknown probe {pid}"
            assert probes[pid]["result"] is not None and probes[pid]["http_status"] in (200, 206), pid


def test_probes_are_run_for_every_reachable_data_asset(rows, probes):
    assert {"ett", "nab", "smd", "skab", "quality"} <= set(probes)
    assert probes["smd"]["result"]["n_columns"] == 38  # from the run, pinned so a changed file is noticed
    assert probes["ett"]["result"]["columns"][0] == "date"


def test_licence_lines_picks_out_licence_statements():
    text = "Intro\nAll code is under the MIT License.\nunrelated\nData: CC BY-SA 4.0\n"
    assert survey.licence_lines(text) == ["All code is under the MIT License.", "Data: CC BY-SA 4.0"]
    assert survey.licence_lines("nothing here") == []
    assert len(survey.licence_lines("license\n" * 10, limit=3)) == 3


def test_csv_summary_handles_headers_partial_lines_and_no_header():
    sample = "date,a,b\n2020-01-01,1.5,2\n2020-01-02,3,4\n2020-01-03,5,"  # last line is cut
    s = survey.csv_summary(sample, ",")
    assert s["columns"] == ["date", "a", "b"] and s["rows_in_sample"] == 2 and s["numeric_columns"] == 2 and not s["first_column_is_numeric"]
    s = survey.csv_summary("1,2,3\n4,5,6\n7,8", ",", header=False)
    assert s["columns"] is None and s["n_columns"] == 3 and s["rows_in_sample"] == 2 and s["numeric_columns"] == 3


def test_quality_census_counts_licence_classes():
    def line(article_id, licence, n_questions):
        return json.dumps({"article_id": article_id, "license": licence, "questions": [{}] * n_questions})

    lines = [line("1", "This eBook ... Project Gutenberg License ...", 2), line("1", "This eBook ... Project Gutenberg License ...", 3),
             line("2", "https://www.anc.org/OANC/license.txt", 1), line("3", "Creative Commons Attribution 4.0 International License (CC BY 4.0); x", 1),
             line("4", None, 1)]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("QuALITY.v1.0.1.htmlstripped.dev", "\n".join(lines) + "\n")
        z.writestr("QuALITY.v1.0.1.dev", "ignored")
        z.writestr("__MACOSX/._QuALITY.v1.0.1.htmlstripped.dev", "ignored")
    census = survey.quality_census(buffer.getvalue())
    assert list(census) == ["dev"]
    dev = census["dev"]
    assert dev["articles"] == 4 and dev["question_sets"] == 5 and dev["questions"] == 8
    assert dev["articles_by_licence_class"] == {"CC BY 4.0": 1, "OANC licence (URL only)": 1, "Project Gutenberg licence": 1, "none": 1}
