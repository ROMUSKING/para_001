"""Smoke test of notebooks/02-diagnostics/dynamics_parity.ipynb on CPU.

The notebook needs an L4, Drive and two real checkpoints, so this runs its code cells against a fake Drive tree of TINY RANDOM FIXTURES (random episodes, randomly
initialised small models). The fixtures exercise the plumbing only (paths, payload keys, hash checks, anchors, the no-test-split rule, the report); nothing here is
data or evidence. Patches to the notebook source assert that their target text exists, so an edit that moves those lines fails this test loudly.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pandas")

from adjointrwm.data import WindowDataset, WindowSpec, episode_split, fit_normaliser  # noqa: E402
from adjointrwm.models import AdjointRecursiveWorldModel, AdjointRWMConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/02-diagnostics/dynamics_parity.ipynb"
V1_RUN, V2_RUN = "droid100_adjoint_20260929T070629Z", "droid100_adjoint_v2_20260930T165409Z"
SMALL = dict(d_model=32, transformer_layers=1, transformer_heads=2, transformer_ff=64)
STATE_DIM, ACTION_DIM, EMBED, LENGTH, EPISODES = 6, 3, 4, 30, 20


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_model(mask_mode="single_choice", prediction_mode="base", seed=0):
    config = AdjointRWMConfig(context_len=8, horizon=4, num_refinement_candidates=4, candidate_costs=(1.0, 1.0, 1.5, 2.0), rate_beta=0.002, dropout=0.1,
                              mask_mode=mask_mode, prediction_mode=prediction_mode, **SMALL)
    torch.manual_seed(seed)
    return AdjointRecursiveWorldModel(STATE_DIM, ACTION_DIM, 2 * EMBED, config).eval()


def manual_gate(model, dataset, prediction_mode):
    """The gate computed with a plain loop (independent of predict_dataset and dynamics_gate_row): horizon-mean RMSE of the model and of persistence."""
    import dataclasses

    model.config = dataclasses.replace(model.config, prediction_mode=prediction_mode)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    err_model, err_persist, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            pred = model.predict(batch)["state_mean"].double()
            target = batch["target_state"].double()
            persistence = batch["context_state"][:, -1:].double().expand(-1, 4, -1)
            err_model = err_model + ((pred - target) ** 2).mean(dim=2).sum(dim=0)
            err_persist = err_persist + ((persistence - target) ** 2).mean(dim=2).sum(dim=0)
            n += len(target)
    return float(torch.sqrt(err_model / n).mean()), float(torch.sqrt(err_persist / n).mean())


@pytest.fixture()
def tree(tmp_path):
    drive = tmp_path / "drive" / "runs"
    rng = np.random.default_rng(0)
    ids = [f"ep{i:03d}" for i in range(EPISODES)]
    assignment = episode_split(ids)
    v1, v2 = drive / V1_RUN, drive / V2_RUN
    (v2 / "cache" / "episodes").mkdir(parents=True)
    (v1 / "config").mkdir(parents=True)
    (v1 / "checkpoints").mkdir()
    (v2 / "artifacts").mkdir()
    episodes = []
    for episode_id in ids:
        arrays = {"states": rng.normal(size=(LENGTH, STATE_DIM)) * 3 + 1, "actions": rng.normal(size=(LENGTH, ACTION_DIM)),
                  "exterior_embeddings": rng.normal(size=(LENGTH, EMBED)).astype(np.float16), "wrist_embeddings": rng.normal(size=(LENGTH, EMBED)).astype(np.float16)}
        path = v2 / "cache" / "episodes" / f"{episode_id}.npz"
        np.savez(path, **arrays)
        episodes.append({"episode_id": episode_id, "length": LENGTH, "persisted_path": str(path), "cached_sha256": sha(path)})
    (v2 / "cache" / "cache_manifest.json").write_text(json.dumps({"episodes": episodes, "skipped": []}))
    (v1 / "config" / "data_manifest.json").write_text(json.dumps({"episodes": [{"episode_id": e, "split": assignment[e]} for e in ids]}))

    records = [{**e, "cached_path": e["persisted_path"], "split": assignment[e["episode_id"]]} for e in episodes]
    spec = WindowSpec(8, 4, 2)
    train_arrays = [dict(np.load(r["cached_path"])) for r in records if r["split"] == "train"]
    norm_v2 = fit_normaliser([a["states"] for a in train_arrays], [a["actions"] for a in train_arrays])
    norm_v1 = {**norm_v2, "state_mean": norm_v2["state_mean"] + 0.5}          # deliberately different, so that scoring a checkpoint in the other one's units is visible
    val = {label: WindowDataset([r for r in records if r["split"] == "validation"], spec, norm, visual_layout="flat") for label, norm in (("v1", norm_v1), ("v2", norm_v2))}

    model_v1, model_v2 = build_model("subset", "full", seed=1), build_model("single_choice", "base", seed=2)
    v1_model_rmse, v1_persistence = manual_gate(model_v1, val["v1"], "full")
    v1_base_rmse, _ = manual_gate(model_v1, val["v1"], "base")
    v2_model_rmse, v2_persistence = manual_gate(model_v2, val["v2"], "base")
    torch.save({"model_state_dict": model_v1.state_dict(), "normalisation": norm_v1, "global_step": 1000,
                "best_metrics": {"full_rmse_by_horizon": [v1_model_rmse] * 4, "base_rmse_by_horizon": [v1_base_rmse] * 4,
                                 "persistence_rmse_by_horizon": [v1_persistence] * 4, "visual_cosine": 0.5}}, v1 / "checkpoints" / "best_dynamics.pt")
    (v2 / "jobs" / "seed_0" / "dynamics").mkdir(parents=True)
    torch.save({"model_state_dict": model_v2.state_dict(), "global_step": 1000}, v2 / "jobs" / "seed_0" / "dynamics" / "best.pt")
    (v2 / "artifacts" / "dynamics_gate.json").write_text(json.dumps({"0": {"model_rmse": v2_model_rmse, "persistence_rmse": v2_persistence, "split": "validation"}}))
    return {"drive": tmp_path / "drive", "v1": v1 / "checkpoints" / "best_dynamics.pt", "v2": v2 / "jobs" / "seed_0" / "dynamics" / "best.pt", "gate": v2 / "artifacts" / "dynamics_gate.json",
            "v2_expected": (v2_model_rmse, v2_persistence)}


def patched_cells(tree):
    nb = json.loads(NOTEBOOK.read_text())
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"][1:]        # skip the install and checkout cell
    assert len(sources) == 5

    def swap(text, old, new):
        assert old in text, f"the notebook no longer contains: {old[:70]!r}"
        return text.replace(old, new)

    c = sources[0]
    c = swap(c, "from google.colab import drive\n\ndrive.mount('/content/drive', force_remount=False)\n", "")
    c = swap(c, "assert torch.cuda.is_available(), 'Select a GPU runtime before continuing.'\nDEVICE = torch.device('cuda')\nGPU_NAME = torch.cuda.get_device_name(0)\n"
                "if 'L4' not in GPU_NAME:\n    raise RuntimeError(f'This diagnostic is scheduled for an L4; this runtime has {GPU_NAME}.')\n",
             "DEVICE = torch.device('cpu')\nGPU_NAME = 'cpu (test)'\n")
    c = swap(c, "v1_checkpoint_sha256: str = '4b0f07177221ae63467bd0ce7f8ae97545e9fc92a27b3a0731551648709fcc3d'", f"v1_checkpoint_sha256: str = '{sha(tree['v1'])}'")
    c = swap(c, "v2_checkpoint_sha256: str = '66c80f464a442da0ba9e3ee424aef692a1be4b76eff860fe1d8640c08555882b'", f"v2_checkpoint_sha256: str = '{sha(tree['v2'])}'")
    c = swap(c, "DRIVE_ROOT = Path('/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production')", f"DRIVE_ROOT = Path({str(tree['drive'])!r})")
    c = swap(c, "LOCAL = Path('/content/adjoint_rwm_work') / RUN_ID", f"LOCAL = Path({str(tree['drive'].parent / 'local')!r}) / RUN_ID")
    sources[0] = c
    sources[2] = swap(sources[2], "MODEL_FIELDS = dict(context_len=CFG.context_len, horizon=CFG.horizon, d_model=512, transformer_layers=6, transformer_heads=8, transformer_ff=2048, dropout=0.10,",
                      "MODEL_FIELDS = dict(context_len=CFG.context_len, horizon=CFG.horizon, d_model=32, transformer_layers=1, transformer_heads=2, transformer_ff=64, dropout=0.10,")
    return sources


def run_notebook(tree, monkeypatch, mutate=None):
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    sources = patched_cells(tree)
    module = types.ModuleType("__notebook__")                              # @dataclass looks its module up in sys.modules
    monkeypatch.setitem(sys.modules, "__notebook__", module)
    namespace = module.__dict__
    namespace.update({"sys": sys, "REPO_COMMIT": "f" * 40, "REPO_DIRTY": False, "REPO_REF": "main"})
    if mutate:
        mutate(sources)
    for index, source in enumerate(sources):
        exec(compile(source, f"<cell {index + 1}>", "exec"), namespace)   # noqa: S102 - our own notebook, against temporary files
    return namespace


def test_the_notebook_runs_end_to_end_reproduces_its_anchors_and_never_builds_the_test_split(tree, monkeypatch, capsys):
    ns = run_notebook(tree, monkeypatch)
    report = ns["REPORT"]
    assert report["status"] == "OK" and report["anchors_ok"] is True and report["test_split_read"] is False and report["load_errors"] == {}
    rows = report["rows"]
    assert len(rows) == 16 and {r["split"] for r in rows} == {"train", "validation"}                         # 2 checkpoints x 2 modes x 2 splits x 2 amp settings
    assert {(r["checkpoint"], r["prediction_mode"], r["amp"]) for r in rows} == {(c, m, a) for c in ("v1", "v2") for m in ("base", "full") for a in (True, False)}
    assert set(ns["BY_SPLIT"]) == {"train", "validation"} and all("test" not in d for d in ns["DATASETS"].values())
    assert len(report["anchors"]) == 5 and all(a["within"] for a in report["anchors"].values())              # 2 for v2, 3 for the pilot
    stored_model, stored_persistence = tree["v2_expected"]
    cell = next(r for r in rows if (r["checkpoint"], r["prediction_mode"], r["split"], r["amp"]) == ("v2", "base", "validation", True))
    assert cell["model_rmse"] == pytest.approx(stored_model, abs=1e-6) and cell["persistence_rmse"] == pytest.approx(stored_persistence, abs=1e-9)
    assert report["split_parity_with_pilot"]["matches"] is True and report["normaliser_comparison"]["max_abs_diff_overall"] == pytest.approx(0.5)
    assert report["model_fields_differing_from_pilot_config"] == {}
    run_dir = ns["RUN_DIR"]
    assert (run_dir / "COMPLETE").exists() and (run_dir / "reports" / "acceptance_report.json").exists() and (run_dir / "artifacts" / "dynamics_parity.csv").exists()
    assert json.loads((run_dir / "config" / "run_config.json").read_text())["test_split_read"] is False
    summary = (run_dir / "reports" / "run_summary.md").read_text()
    assert "Status: **OK**" in summary and "| v1 | full | validation | True |" in summary and report["claim_boundary"] in summary
    assert report["readings"] and isinstance(report["readings"][0], str)


def test_a_stored_number_that_the_pipeline_cannot_reproduce_gives_anchor_failed_and_no_readings(tree, monkeypatch):
    gate = json.loads(tree["gate"].read_text())
    gate["0"]["model_rmse"] += 0.05                                                                              # far outside the 1e-3 tolerance
    tree["gate"].write_text(json.dumps(gate))
    ns = run_notebook(tree, monkeypatch)
    report = ns["REPORT"]
    assert report["status"] == "ANCHOR_FAILED" and report["anchors_ok"] is False and report["readings"] == []
    failed = [k for k, a in report["anchors"].items() if not a["within"]]
    assert failed == ["v2_validation_base_rmse_vs_dynamics_gate_json"]
    assert len(report["rows"]) == 16 and not (ns["RUN_DIR"] / "COMPLETE").exists()                              # the table is still written, the run is not marked complete


def test_scoring_the_pilot_checkpoint_in_the_wrong_input_units_is_caught_by_its_anchors(tree, monkeypatch):
    def mix_up_units(sources):
        old = "NORMS = {'v1': NORM_V1 if NORM_V1 is not None else NORM_V2, 'v2': NORM_V2}"
        assert old in sources[2]
        sources[2] = sources[2].replace(old, "NORMS = {'v1': NORM_V2, 'v2': NORM_V2}")

    report = run_notebook(tree, monkeypatch, mutate=mix_up_units)["REPORT"]
    assert report["status"] == "ANCHOR_FAILED" and report["readings"] == []
    failed = sorted(k for k, a in report["anchors"].items() if not a["within"])
    assert failed and all(k.startswith("v1_validation_") for k in failed)                                          # only the pilot's anchors fail; the v2 ones still hold


def test_a_checkpoint_with_the_wrong_hash_is_refused_before_it_is_loaded(tree, monkeypatch):
    def wrong_hash(sources):
        sources[0] = sources[0].replace("v1_checkpoint_sha256: str = '" + sha(tree["v1"]) + "'", "v1_checkpoint_sha256: str = '" + "0" * 64 + "'")

    with pytest.raises(RuntimeError, match="Not the checkpoint this diagnostic was written for"):
        run_notebook(tree, monkeypatch, mutate=wrong_hash)


def test_the_notebook_source_keeps_the_test_split_out_and_needs_an_l4():
    nb = json.loads(NOTEBOOK.read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "splits: tuple = ('train', 'validation')" in code and "assert 'test' not in CFG.splits" in code and "'test'" not in code.split("BY_SPLIT = ")[1].split("\n")[0]
    assert "if 'L4' not in GPU_NAME" in code and nb["metadata"]["colab"]["gpuType"] == "L4"
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
