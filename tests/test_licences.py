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


def test_json_licence_values_finds_nested_and_list_values():
    hf = {"cardData": {"license": ["cc-by-sa-4.0"], "license_name": "Custom"}, "tags": ["license:x"]}
    assert survey.json_licence_values(hf) == {"Custom": 1, "cc-by-sa-4.0": 1}
    zenodo = {"hits": {"hits": [{"metadata": {"license": {"id": "cc-by-4.0"}}}, {"metadata": {"license": {"id": "cc-by-4.0"}}},
                                {"metadata": {"license": {"id": "cc-by-3.0"}}}, {"metadata": {}}]}}
    assert survey.json_licence_values(zenodo) == {"cc-by-4.0": 2, "cc-by-3.0": 1}
    assert survey.json_licence_values({"name": "no licence key"}) == {}


def test_jsonld_licences_reads_catalogue_markup():
    page = '<html><script type="application/ld+json">{"@type":"Dataset","license":{"@type":"CreativeWork","name":"CC0: Public Domain"}}</script><script>var x=1</script></html>'
    assert survey.jsonld_licences(page) == {"CC0: Public Domain": 1}
    assert survey.jsonld_licences("<html>none</html>") == {}
    assert survey.jsonld_licences('<script type="application/ld+json">{broken</script>') == {}


def test_hf_summary_extracts_gating_licence_files_and_size():
    obj = {"id": "org/model", "sha": "abc", "cardData": {"license": "apache-2.0"}, "gated": "manual", "tags": ["x", "license:apache-2.0"],
           "siblings": [{"rfilename": "LICENSE"}, {"rfilename": "model.safetensors"}], "safetensors": {"total": 123}, "lastModified": "2026-01-01"}
    s = survey.hf_summary(obj)
    assert s["license"] == "apache-2.0" and s["gated"] == "manual" and s["licence_files"] == ["LICENSE"] and s["safetensors_total_params"] == 123
    assert s["licence_tags"] == ["license:apache-2.0"]
    assert survey.hf_summary({"id": "org/x"})["license"] is None


def test_html_to_text_drops_scripts_and_splits_on_tags():
    page = "<html><style>p{}</style><body><h1>Terms</h1><p>Licensed under <b>CC BY 4.0</b></p><script>var license='x'</script></body></html>"
    text = survey.html_to_text(page)
    assert "var license" not in text and "p{}" not in text and "Terms" in text.splitlines()
    assert survey.licence_lines(text) == ["Licensed under", "CC BY 4.0"]


def test_guess_licence_recognises_common_families():
    assert survey.guess_licence("MIT License\n\nPermission is hereby granted, free of charge, to any person") == "MIT"
    assert survey.guess_licence("Apache License\n Version 2.0, January 2004") == "Apache-2.0"
    assert survey.guess_licence("Redistribution and use in source and binary forms ... Neither the name of") == "BSD-3-Clause"
    assert survey.guess_licence("Redistribution and use in source and binary forms, with or without") == "BSD-2-Clause"
    assert survey.guess_licence("GNU LESSER GENERAL PUBLIC LICENSE Version 3") == "LGPL"
    assert survey.guess_licence("GNU GENERAL PUBLIC LICENSE Version 2") == "GPL"
    assert survey.guess_licence("License agreement for matplotlib versions 1.3.0 and later") == "PSF-style"
    assert survey.guess_licence("something else entirely") == "unrecognised"


def test_feature_types_collapses_nested_types():
    features = [{"name": "q", "type": {"dtype": "string", "_type": "Value"}}, {"name": "ctx", "type": {"feature": {"dtype": "string"}, "_type": "Sequence"}},
                {"name": "rows", "type": [{"a": 1}]}, {"name": "d", "type": {"a": {"dtype": "int64"}}}]
    assert survey.feature_types(features) == {"q": "string", "ctx": "Sequence", "rows": "list", "d": "struct"}


def test_swebench_census_probe_is_consistent(probes):
    census = probes["swebench_repos"]["result"]
    assert census["rows"] == sum(v["instances"] for v in census["per_repo"].values()) == sum(census["instances_by_licence_family_guess"].values())
    assert census["repos"] == len(census["per_repo"]) and census["repos"] >= 10


def test_no_probe_named_in_the_register_failed(rows, probes):
    named = {pid for r in rows for pid in r["probe"].split()}
    assert named and all(probes[pid]["http_status"] in (200, 206) and probes[pid]["result"] is not None for pid in named)
