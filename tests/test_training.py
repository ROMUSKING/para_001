"""Training runner: resume equivalence (roadmap N0.3), checkpoint integrity, RNG isolation,
and the benchmark fairness contract. Episodes are random fixtures, not data."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adjointrwm.benchmark import check_fairness, split_hash, windows_hash  # noqa: E402
from adjointrwm.data import WindowDataset, WindowSpec, fit_normaliser  # noqa: E402
from adjointrwm.models import ArmDims, build_arm  # noqa: E402
from adjointrwm.training import (  # noqa: E402
    PermutedActions,
    StatefulBatchSampler,
    TrainConfig,
    load_best_weights,
    load_checkpoint,
    predict_dataset,
    seed_everything,
    train_world_model,
)

SPEC = WindowSpec(context_len=4, horizon=3, stride=1)
DIMS = ArmDims(state_dim=3, action_dim=2, visual_tokens=2, visual_token_dim=4, target_visual_dim=8, context_len=4, horizon=3)
CFG = TrainConfig(steps=6, batch_size=4, lr=1e-3, warmup_steps=2, eval_interval=3, checkpoint_interval=2,
                  log_interval=1, seed=5, amp=False, eval_batch_size=8)


def _dataset(n_episodes, seed):
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_episodes):
        length = 12
        records.append({
            "episode_id": f"{seed}-{i}",
            "length": length,
            "arrays": {
                "states": rng.normal(size=(length, 3)).astype(np.float32),
                "actions": rng.normal(size=(length, 2)).astype(np.float32),
                "exterior_embeddings": rng.normal(size=(length, 4)).astype(np.float16),
                "wrist_embeddings": rng.normal(size=(length, 4)).astype(np.float16),
            },
        })
    norm = fit_normaliser([r["arrays"]["states"] for r in records], [r["arrays"]["actions"] for r in records])
    return WindowDataset(records, SPEC, norm, visual_layout="tokens")


TRAIN, VAL = _dataset(3, 0), _dataset(2, 1)


def _identity(**overrides):
    ident = {"run_id": "test_run", "arm": "adjoint_rwm", "seed": 5, "config_hash": "c", "data_manifest_hash": "d", "source_hash": "s"}
    ident.update(overrides)
    return ident


def _model():
    seed_everything(5)
    return build_arm("adjoint_rwm", DIMS, width=16, transformer_heads=2, transformer_layers=1)


def test_resume_reproduces_uninterrupted_training(tmp_path):
    straight = _model()
    done = train_world_model(straight, TRAIN, VAL, CFG, identity=_identity(), local_dir=tmp_path / "a_local", persist_dir=tmp_path / "a")
    assert done["status"] == "DONE"

    first = _model()
    paused = train_world_model(first, TRAIN, VAL, CFG, identity=_identity(), local_dir=tmp_path / "b_local",
                               persist_dir=tmp_path / "b", max_steps_this_session=3)
    assert paused["status"] == "PAUSED" and paused["global_step"] == 3
    seed_everything(999)  # a new session starts from an unrelated RNG state
    resumed_model = build_arm("adjoint_rwm", DIMS, width=16, transformer_heads=2, transformer_layers=1)
    resumed = train_world_model(resumed_model, TRAIN, VAL, CFG, identity=_identity(), local_dir=tmp_path / "b_local2", persist_dir=tmp_path / "b")
    assert resumed["status"] == "DONE" and resumed["global_step"] == 6

    for (name, a), (_, b) in zip(straight.state_dict().items(), resumed_model.state_dict().items()):
        assert torch.equal(a, b), name
    assert done["best"]["validation_score"] == pytest.approx(resumed["best"]["validation_score"])


def test_finished_jobs_keep_only_what_is_needed(tmp_path):
    summary = train_world_model(_model(), TRAIN, VAL, CFG, identity=_identity(), local_dir=tmp_path / "l", persist_dir=tmp_path / "p")
    assert summary["best_checkpoint_kept"]
    assert (tmp_path / "p" / "best.pt").exists() and not (tmp_path / "p" / "latest.pt").exists()
    assert not list((tmp_path / "l").glob("*.pt"))  # local copies are dropped after upload
    trial = train_world_model(_model(), TRAIN, VAL, CFG, identity=_identity(), local_dir=tmp_path / "l2", persist_dir=tmp_path / "q",
                              keep_best=False)
    assert not list((tmp_path / "q").glob("*.pt")) and trial["best"]["checkpoint"]["sha256"]


def test_done_job_is_skipped_and_best_weights_reload(tmp_path):
    model = _model()
    summary = train_world_model(model, TRAIN, VAL, CFG, identity=_identity(), local_dir=tmp_path / "l", persist_dir=tmp_path / "p")
    again = train_world_model(_model(), TRAIN, VAL, CFG, identity=_identity(), local_dir=tmp_path / "l", persist_dir=tmp_path / "p")
    assert again == json.loads((tmp_path / "p" / "DONE.json").read_text())
    fresh = _model()
    meta = load_best_weights(fresh, tmp_path / "p", identity=_identity())
    assert meta["global_step"] == summary["best"]["step"]
    saved = load_checkpoint(tmp_path / "p" / "best.pt")["model_state_dict"]  # weights round-trip exactly
    assert saved.keys() == fresh.state_dict().keys()
    assert all(torch.equal(saved[k], v) for k, v in fresh.state_dict().items())
    with pytest.raises(RuntimeError):
        load_best_weights(fresh, tmp_path / "p", identity=_identity(seed=6))


def test_resume_refuses_a_different_configuration(tmp_path):
    train_world_model(_model(), TRAIN, VAL, CFG, identity=_identity(), local_dir=tmp_path / "l", persist_dir=tmp_path / "p",
                      max_steps_this_session=2)
    with pytest.raises(RuntimeError, match="identity differs"):
        train_world_model(_model(), TRAIN, VAL, CFG, identity=_identity(config_hash="other"), local_dir=tmp_path / "l",
                          persist_dir=tmp_path / "p")


def test_corrupted_checkpoint_is_rejected(tmp_path):
    train_world_model(_model(), TRAIN, VAL, CFG, identity=_identity(), local_dir=tmp_path / "l", persist_dir=tmp_path / "p",
                      max_steps_this_session=2)
    latest = tmp_path / "p" / "latest.pt"
    with open(latest, "ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        load_checkpoint(latest)


def test_checkpoint_carries_the_operator_brief_contract(tmp_path):
    train_world_model(_model(), TRAIN, VAL, CFG, identity=_identity(), local_dir=tmp_path / "l", persist_dir=tmp_path / "p",
                      max_steps_this_session=2)
    payload = load_checkpoint(tmp_path / "p" / "latest.pt")
    required = {"schema_version", "run_id", "global_step", "epoch", "model_state_dict", "critic_state_dict",
                "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "sampler_state", "rng_state",
                "data_manifest_hash", "config_hash", "source_hash", "best_metrics"}
    assert required <= set(payload)
    assert set(payload["rng_state"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}


def test_prediction_is_reproducible_and_leaves_training_rng_alone():
    model = build_arm("dreamerv3_rssm", DIMS, width=16, eval_samples=2)
    torch.manual_seed(0)
    before = torch.get_rng_state()
    a = predict_dataset(model, VAL, 4, "cpu", eval_seed=3, amp=False)
    assert torch.equal(before, torch.get_rng_state())
    b = predict_dataset(model, VAL, 7, "cpu", eval_seed=3, amp=False)
    np.testing.assert_array_equal(a["episode_id"], b["episode_id"])
    assert a["state_mean"].shape == (len(VAL), 3, 3)


def test_permuted_actions_view():
    perm = list(range(1, len(VAL))) + [0]
    view = PermutedActions(VAL, perm)
    np.testing.assert_array_equal(view[0]["future_actions"], VAL[1]["future_actions"])
    np.testing.assert_array_equal(view[0]["target_state"], VAL[0]["target_state"])
    with pytest.raises(ValueError):
        PermutedActions(VAL, [0] * len(VAL))


def test_sampler_state_roundtrip():
    s = StatefulBatchSampler(10, 3, seed=1)
    it = iter(s)
    next(it)
    state = s.state_dict()
    s2 = StatefulBatchSampler(10, 3, seed=1)
    s2.load_state_dict(state)
    assert list(iter(s2)) == list(it)


# --- fairness contract -------------------------------------------------------------

def _record(**overrides):
    rec = {
        "data_manifest_hash": "d", "split_hash": split_hash({"a": "train", "b": "test"}), "window_spec": {"context_len": 8},
        "test_windows_hash": windows_hash(["a", "b"], [0, 2]), "input_encoder": "resnet18", "target_encoder": "resnet18",
        "steps": 1500, "batch_size": 64, "seeds": [0, 1], "lr_grid": [1e-4, 3e-4], "tuning_steps": 450, "gpu": "L4",
        "prediction_parameters": 1000, "seeds_finished": [0, 1], "seeds_failed": [],
    }
    rec.update(overrides)
    return rec


def test_fairness_contract_passes_and_fails_precisely():
    ok = check_fairness({"ref": _record(), "riv": _record(prediction_parameters=1080)}, "ref")
    assert ok["passed"]
    too_big = check_fairness({"ref": _record(), "riv": _record(prediction_parameters=1200)}, "ref")
    assert not too_big["passed"]
    assert [c["check"] for c in too_big["checks"] if not c["passed"]] == ["parameters:riv"]
    other_steps = check_fairness({"ref": _record(), "riv": _record(steps=3000)}, "ref")
    assert [c["check"] for c in other_steps["checks"] if not c["passed"]] == ["equal:steps"]
    failed_seed = check_fairness({"ref": _record(), "riv": _record(seeds_finished=[0], seeds_failed=[1])}, "ref")
    assert failed_seed["passed"]  # a failed seed is reported, not a contract violation
    dropped_seed = check_fairness({"ref": _record(), "riv": _record(seeds_finished=[0])}, "ref")
    assert not dropped_seed["passed"]


def test_windows_hash_is_order_independent():
    assert windows_hash(["a", "b"], [1, 2]) == windows_hash(["b", "a"], [2, 1])
    assert windows_hash(["a", "b"], [1, 2]) != windows_hash(["a", "b"], [1, 3])
