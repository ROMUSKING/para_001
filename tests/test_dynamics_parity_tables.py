"""The dynamics-parity note must say what the imported run says.

Reads the committed run ``results/runs/dynamics_parity_20260930T214043Z`` through ``scripts/dynamics_parity_tables.py`` (which itself stops if the CSV, the JSON
and the report disagree, if a relative improvement does not recompute, or if the config hash does not verify) and checks that every table row and number the
research note quotes appears in the note.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/runs/dynamics_parity_20260930T214043Z"
PILOT = ROOT / "results/runs/droid100_adjoint_20260929T070629Z"
NOTE = ROOT / "docs/research-notes/2026-09-30-dynamics-parity.md"

pytestmark = pytest.mark.skipif(not RUN.is_dir(), reason="the imported run is not in this checkout")


def script():
    spec = importlib.util.spec_from_file_location("dynamics_parity_tables", ROOT / "scripts/dynamics_parity_tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalised(text: str) -> str:
    return text.replace("−", "-")


def rows_of(table: str) -> list[str]:
    return [line for line in table.splitlines() if line.startswith("|") and not line.startswith("|---") and "Checkpoint" not in line.split("|")[1]]


def test_every_table_row_in_the_note_is_printed_from_the_committed_run():
    import json

    module = script()
    report, rows = module.load(RUN)
    pilot = json.loads((PILOT / "artifacts/dynamics_evaluation.json").read_text())
    note = normalised(NOTE.read_text())
    for name, table in (("BF16", module.rmse_table(rows, True)), ("FP32", module.rmse_table(rows, False)), ("by horizon", module.horizon_table(rows, pilot))):
        body = rows_of(table)
        assert len(body) >= 5, name
        for line in body:
            assert normalised(line) in note, f"{name} table row missing from the note: {line}"
    for anchor in report["anchors"].values():
        for field in ("recomputed", "stored", "abs_diff"):
            assert repr(anchor[field]) in note or f"{anchor[field]:.4f}" in note, (field, anchor)


def test_the_cross_split_numbers_and_the_hashes_the_note_and_readme_quote_match_the_files():
    import json

    module = script()
    report, rows = module.load(RUN)
    pilot = json.loads((PILOT / "artifacts/dynamics_evaluation.json").read_text())
    by = {module.key(r): r for r in rows}
    note, readme = normalised(NOTE.read_text()), (RUN / "README.md").read_text()
    train, val = by[("v1", "full", "train", True)], by[("v1", "full", "validation", True)]
    for number in (f"{train['model_rmse']:.4f}", f"{val['model_rmse']:.4f}", f"{pilot['gate']['full_rmse']:.4f}", f"{val['persistence_rmse']:.4f}", f"{pilot['gate']['persistence_rmse']:.4f}"):
        assert number in note, number
    assert f"{pilot['gate']['relative_improvement']:+.3f}" in note and f"{val['relative_improvement']:+.3f}" in note and f"{train['relative_improvement']:+.3f}" in note
    assert report["config_hash"] in note and report["config_hash"] in readme
    for rel in ("COMPLETE", "config/run_config.json", "artifacts/dynamics_parity.json", "artifacts/dynamics_parity.csv", "reports/acceptance_report.json", "reports/run_summary.md"):
        digest = hashlib.sha256((RUN / rel).read_bytes()).hexdigest()
        assert digest in readme, f"README does not list the SHA-256 of {rel}"
    for label, checkpoint in report["checkpoints"].items():
        assert checkpoint["sha256"] in readme and checkpoint["sha256"][:8] in note, label


def test_the_run_used_the_frozen_rules_the_note_relies_on():
    import json

    report = json.loads((RUN / "reports/acceptance_report.json").read_text())
    config = json.loads((RUN / "config/run_config.json").read_text())
    assert report["status"] == "OK" and report["anchors_ok"] is True and report["test_split_read"] is False and config["test_split_read"] is False
    assert {r["split"] for r in report["rows"]} == {"train", "validation"} and config["config"]["splits"] == ["train", "validation"]
    assert report["split_parity_with_pilot"]["matches"] is True and report["load_errors"] == {} and (RUN / "COMPLETE").exists()
    validation = [r for r in report["rows"] if r["split"] == "validation"]
    assert validation and all(not r["passed"] for r in validation)                       # the claim the README and the roadmap now make
