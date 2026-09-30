"""Smoke test of notebooks/02-diagnostics/dynamics_kfold.ipynb on CPU.

The notebook needs an L4, Drive, the probe run's feature cache and a pilot checkpoint, so this runs its code cells against a fake Drive tree of TINY RANDOM FIXTURES
(100 random episodes, tiny randomly initialised models trained for a handful of steps). The fixtures exercise the plumbing only (folds, leakage rules, anchors,
scoring, bootstrap, classification, report, resume); nothing here is data or evidence. Patches to the notebook source assert that their target text exists.
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

from adjointrwm.data import episode_split  # noqa: E402
from adjointrwm.eval.dynamics_parity import CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/02-diagnostics/dynamics_kfold.ipynb"
V1_RUN, V2_RUN = "droid100_adjoint_20260929T070629Z", "droid100_adjoint_v2_20260930T165409Z"
STATE_DIM, ACTION_DIM, EMBED, LENGTH, EPISODES = 6, 3, 4, 30, 100


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_tree(base: Path, stored_v2: float, stored_pilot: float):
    drive = base / "drive" / "runs"
    rng = np.random.default_rng(0)
    ids = [f"ep{i:03d}" for i in range(EPISODES)]
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
    split = episode_split(ids)
    (v1 / "config" / "data_manifest.json").write_text(json.dumps({"episodes": [{"episode_id": e, "split": split[e]} for e in ids]}))
    torch.save({"model_state_dict": {}, "best_metrics": {"full_rmse_by_horizon": [stored_pilot] * 4}, "global_step": 1400}, v1 / "checkpoints" / "best_dynamics.pt")
    (v2 / "artifacts" / "dynamics_gate.json").write_text(json.dumps({"0": {"model_rmse": stored_v2, "persistence_rmse": 0.2, "split": "validation"}}))
    return {"drive": base / "drive", "v1": v1 / "checkpoints" / "best_dynamics.pt", "split": split}


TINY = {"    d_model: int = 512": "    d_model: int = 32", "    transformer_layers: int = 6": "    transformer_layers: int = 1", "    transformer_heads: int = 8": "    transformer_heads: int = 2",
        "    transformer_ff: int = 2048": "    transformer_ff: int = 64", "    steps: int = 1500": "    steps: int = 6", "    batch_size: int = 64": "    batch_size: int = 8",
        "    warmup_steps: int = 100": "    warmup_steps: int = 2", "    eval_interval: int = 100": "    eval_interval: int = 3", "    checkpoint_interval: int = 250": "    checkpoint_interval: int = 3",
        "    bootstrap_resamples: int = 10000": "    bootstrap_resamples: int = 300", "    subset_draws: int = 10000": "    subset_draws: int = 300",
        "    regimes: tuple = ('v2',)": "    regimes: tuple = ('pilot', 'v2')"}


def patched_cells(tree):
    nb = json.loads(NOTEBOOK.read_text())
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"][1:]        # skip the install and checkout cell
    assert len(sources) == 4

    def swap(text, old, new):
        assert old in text, f"the notebook no longer contains: {old[:70]!r}"
        return text.replace(old, new)

    c = sources[0]
    c = swap(c, "from google.colab import drive\n\ndrive.mount('/content/drive', force_remount=False)\n", "")
    c = swap(c, "assert torch.cuda.is_available(), 'Select a GPU runtime before continuing.'\nDEVICE = torch.device('cuda')\nGPU_NAME = torch.cuda.get_device_name(0)\n"
                "if 'L4' not in GPU_NAME:\n    raise RuntimeError(f'This study is scheduled for an L4; this runtime has {GPU_NAME}.')\n", "DEVICE = torch.device('cpu')\nGPU_NAME = 'cpu (test)'\n")
    c = swap(c, "v1_checkpoint_sha256: str = '4b0f07177221ae63467bd0ce7f8ae97545e9fc92a27b3a0731551648709fcc3d'", f"v1_checkpoint_sha256: str = '{sha(tree['v1'])}'")
    c = swap(c, "DRIVE_ROOT = Path('/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production')", f"DRIVE_ROOT = Path({str(tree['drive'])!r})")
    c = swap(c, "LOCAL = Path('/content/adjoint_rwm_work') / RUN_ID", f"LOCAL = Path({str(tree['drive'].parent / 'local')!r}) / RUN_ID")
    for old, new in TINY.items():
        c = swap(c, old, new)
    sources[0] = c
    return sources


def execute(tree, monkeypatch, upto=4, namespace=None):
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    if namespace is None:
        module = types.ModuleType("__kfold__")
        monkeypatch.setitem(sys.modules, "__kfold__", module)
        namespace = module.__dict__
        namespace.update({"sys": sys, "REPO_COMMIT": "f" * 40, "REPO_DIRTY": False, "REPO_REF": "main"})
    for index, source in enumerate(patched_cells(tree)[:upto]):
        exec(compile(source, f"<cell {index + 1}>", "exec"), namespace)   # noqa: S102 - our own notebook, against temporary files
    return namespace


@pytest.fixture(scope="module")
def first_pass(tmp_path_factory):
    """One full pass with stored anchors that cannot be reproduced; returns the namespace and the anchors the units actually reached."""
    mp = pytest.MonkeyPatch()
    tree = make_tree(tmp_path_factory.mktemp("failed_anchor"), stored_v2=9.0, stored_pilot=9.0)
    try:
        ns = execute(tree, mp)
        yield ns
    finally:
        mp.undo()


def test_a_stored_number_the_anchor_unit_cannot_reproduce_gives_anchor_failed_and_no_classification(first_pass):
    report = first_pass["REPORT"]
    assert report["status"] == "ANCHOR_FAILED" and report["kfold_uses_all_100_episodes"] is True and report["anchor_reads_pilot_test_episodes"] is False
    for regime in ("pilot", "v2"):
        entry = report["regimes"][regime]
        assert entry["status"] == "ANCHOR_FAILED" and entry["primary_class"] is None and entry["anchor"]["within"] is False and entry["anchor"]["stored"] == 9.0
        assert {e["kind"] + "|" + e["mode"] for e in entry["estimates"].values()} == {"best|" + m for m in ("base", "full")} | {"final|" + m for m in ("base", "full")}
        primary = [e for e in entry["estimates"].values() if e["primary"]]
        assert len(primary) == 1 and primary[0]["mode"] == entry["own_mode"] == {"pilot": "full", "v2": "base"}[regime] and primary[0]["kind"] == "best"
        assert len(primary[0]["per_fold_relative_improvement"]) == 5 and primary[0]["pooled"]["episodes"] == 100 and primary[0]["class"] in CLASSES
        assert 0.0 <= primary[0]["random_ten_episode_split_gate_rate"]["pass_rate"] <= 1.0 and len(primary[0]["bootstrap"]["by_horizon"]) == 4
    run_dir = first_pass["RUN_DIR"]
    assert not (run_dir / "COMPLETE").exists()                                                  # the tables are written, the run is not marked complete
    assert (run_dir / "artifacts" / "episode_errors.csv").exists() and (run_dir / "reports" / "acceptance_report.json").exists()
    assert sorted(p.name for p in (run_dir / "artifacts" / "units").glob("*.json")) == sorted(f"{r}_{u}.json" for r in ("pilot", "v2") for u in ("anchor", 0, 1, 2, 3, 4))


def test_the_units_respect_the_leakage_rules_of_the_plan(first_pass):
    ns, split = first_pass, episode_split(first_pass["EPISODE_IDS"])
    folds = ns["FOLDS"]
    all_held = []
    for fold in range(5):
        train, validation, held, seed = ns["unit_plan"](fold)
        assert (len(train), len(validation), len(held)) == (70, 10, 20) and seed == ns["CFG"].seed_base + 1 + fold
        assert not set(held) & (set(train) | set(validation)) and not set(train) & set(validation)        # the held-out episodes are never trained on or used for selection
        assert all(folds[e] == fold for e in held) and all(folds[e] != fold for e in train + validation)
        all_held += held
    assert sorted(all_held) == sorted(ns["EPISODE_IDS"])                                        # every episode is held out exactly once
    train, validation, held, seed = ns["unit_plan"]("anchor")
    assert held == [] and seed == ns["CFG"].seed_base and all(split[e] == "train" for e in train) and all(split[e] == "validation" for e in validation)
    assert not (set(train) | set(validation)) & {e for e, s in split.items() if s == "test"}           # the anchor unit never reads the pilot's test episodes


def test_each_fold_scores_only_its_held_out_episodes(first_pass):
    import pandas as pd

    ns = first_pass
    for regime in ("pilot", "v2"):
        units = ns["load_units"](regime)
        for fold in range(5):
            expected = sorted(e for e in ns["EPISODE_IDS"] if ns["FOLDS"][e] == fold)
            for key, rows in units[fold]["tables"].items():
                assert sorted(r["episode_id"] for r in rows) == expected, (regime, fold, key)
        assert units["anchor"]["tables"] == {}                                                    # the anchor unit scores nothing held out
        table = ns["pooled_tables"](units, regime)[("best", ns["REGIMES"][regime]["prediction_mode"])]
        assert isinstance(table, pd.DataFrame) and len(table) == 100 and table["windows"].sum() == 100 * 10     # 10 windows per fixture episode


def test_reproduced_anchors_give_a_classification_and_a_complete_run_and_a_finished_unit_is_not_retrained(first_pass, tmp_path, monkeypatch):
    reached = {r: first_pass["REPORT"]["regimes"][r]["anchor"]["recomputed"] for r in ("pilot", "v2")}
    tree = make_tree(tmp_path, stored_v2=reached["v2"], stored_pilot=reached["pilot"])          # deterministic on CPU: the same seeds reach the same validation scores
    ns = execute(tree, monkeypatch)
    report = ns["REPORT"]
    assert report["status"] == "OK" and all(e["status"] == "OK" and e["primary_class"] in CLASSES and e["anchor"]["within"] for e in report["regimes"].values())
    assert {r: report["regimes"][r]["anchor"]["abs_diff"] for r in reached} == {"pilot": 0.0, "v2": 0.0}
    assert (ns["RUN_DIR"] / "COMPLETE").exists() and (ns["RUN_DIR"] / "reports" / "run_summary.md").exists()
    summary = (ns["RUN_DIR"] / "reports" / "run_summary.md").read_text()
    assert "Status: **OK**" in summary and "Regime `pilot`" in summary and "Regime `v2`" in summary and report["claim_boundary"] in summary
    config = json.loads((ns["RUN_DIR"] / "config" / "run_config.json").read_text())
    assert config["kfold_uses_all_100_episodes"] is True and config["anchor_reads_pilot_test_episodes"] is False and config["config_sha256"] == ns["CONFIG_HASH"]
    # resume: every unit is finished, so the training cell must not train anything
    def refuse(*args, **kwargs):
        raise AssertionError("train_job was called for a unit that had already finished")

    ns["train_job"] = refuse
    source = patched_cells(tree)[3]
    source = source.replace("from adjointrwm.training import TrainConfig, load_best_weights, predict_dataset, seed_everything, train_job", "from adjointrwm.training import TrainConfig, load_best_weights, predict_dataset, seed_everything")
    exec(compile(source, "<cell 4 resumed>", "exec"), ns)                                          # noqa: S102


def test_a_pilot_checkpoint_with_the_wrong_hash_is_refused_before_any_training(tmp_path, monkeypatch):
    tree = make_tree(tmp_path, stored_v2=0.2, stored_pilot=0.2)
    sources = patched_cells(tree)
    sources[0] = sources[0].replace("v1_checkpoint_sha256: str = '" + sha(tree["v1"]) + "'", "v1_checkpoint_sha256: str = '" + "0" * 64 + "'")
    monkeypatch.setattr(sys.modules[__name__], "patched_cells", lambda t: sources)
    with pytest.raises(RuntimeError, match="expected " + "0" * 64):
        execute(tree, monkeypatch, upto=3)                                                          # fails in the data cell, three cells before any unit trains


def test_the_notebook_source_keeps_its_frozen_settings_and_needs_an_l4():
    nb = json.loads(NOTEBOOK.read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for text in ("k: int = 5", "inner_validation_episodes: int = 10", "steps: int = 1500", "batch_size: int = 64", "learning_rate: float = 3.0e-4", "margin: float = 0.02",
                 "bootstrap_resamples: int = 10000", "subset_size: int = 10", "anchor_atol_v2: float = 0.01", "anchor_atol_pilot: float = 0.02", "regimes: tuple = ('v2',)",
                 "RESUME_RUN_ID = None", "'pilot': {'mask_mode': 'subset', 'prediction_mode': 'full'}", "'v2': {'mask_mode': 'single_choice', 'prediction_mode': 'base'}"):
        assert text in code, text
    assert "if 'L4' not in GPU_NAME" in code and nb["metadata"]["colab"]["gpuType"] == "L4" and all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
    assert code.count("regimes: tuple = ") == 1 and code.count("RESUME_RUN_ID = ") == 1        # exactly one line per override prefix, as the job worker requires
