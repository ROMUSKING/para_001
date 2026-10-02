"""Data contract tests: split parity with the pilot, window alignment and causality.

``_pilot_*`` functions are copied from notebooks/01-production/AdjointRWM_Production_Pilot.ipynb
(sections 2 and 3) and serve as the reference the lifted package code must reproduce.
"""

import hashlib

import numpy as np
import pytest

from adjointrwm.data import (
    WindowDataset,
    WindowSpec,
    as_tokens,
    assignment_from_manifest,
    choose_episode_id,
    compare_splits,
    episode_split,
    extract_episode,
    fit_normaliser,
    slice_window,
    split_sets,
    state_groups,
    window_starts,
)


def _pilot_split(episode_ids):
    records = [{"episode_id": e} for e in episode_ids]
    records_sorted = sorted(records, key=lambda row: hashlib.sha256(row["episode_id"].encode()).hexdigest())
    n_total = len(records_sorted)
    n_train = max(1, int(0.80 * n_total))
    n_val = max(1, int(0.10 * n_total))
    for index, record in enumerate(records_sorted):
        if index < n_train:
            record["split"] = "train"
        elif index < n_train + n_val:
            record["split"] = "validation"
        else:
            record["split"] = "test"
    return {r["episode_id"]: r["split"] for r in records_sorted}


def _pilot_window(states_norm, actions_norm, visual, start, context_len, horizon):
    context_end = start + context_len
    context_action = np.zeros((context_len, actions_norm.shape[-1]), dtype=np.float32)
    if context_len > 1:
        context_action[1:] = actions_norm[start : context_end - 1]
    return {
        "context_state": states_norm[start:context_end],
        "context_action": context_action,
        "context_visual": visual[start:context_end],
        "future_actions": actions_norm[context_end - 1 : context_end - 1 + horizon],
        "target_state": states_norm[context_end : context_end + horizon],
        "target_visual": visual[context_end : context_end + horizon],
    }


def _ids(n, salt="ep"):
    return [hashlib.sha256(f"{salt}{i}".encode()).hexdigest()[:24] for i in range(n)]


@pytest.mark.parametrize("n", [1, 2, 7, 20, 100, 173])
def test_episode_split_matches_pilot_rule(n):
    ids = _ids(n)
    assert episode_split(ids) == _pilot_split(ids)


def test_episode_split_is_order_invariant_and_disjoint():
    ids = _ids(100)
    a = episode_split(ids)
    b = episode_split(list(reversed(ids)))
    assert a == b
    sets = split_sets(a)
    assert [len(sets[s]) for s in ("train", "validation", "test")] == [80, 10, 10]
    assert sets["train"].isdisjoint(sets["validation"]) and sets["train"].isdisjoint(sets["test"])


def test_episode_split_rejects_duplicates():
    with pytest.raises(ValueError):
        episode_split(["a", "a"])


def test_compare_splits_reports_every_difference():
    ids = _ids(20)
    ref = episode_split(ids)
    manifest = {"episodes": [{"episode_id": e, "split": s} for e, s in ref.items()]}
    assert compare_splits(ref, assignment_from_manifest(manifest))["matches"]
    changed = dict(ref)
    first = next(iter(changed))
    changed[first] = "test" if changed[first] != "test" else "train"
    changed["new_episode"] = "train"
    diff = compare_splits(changed, ref)
    assert not diff["matches"]
    assert diff["different_split"] == [first]
    assert diff["only_in_new"] == ["new_episode"]


def test_assignment_from_manifest_handles_val_alias():
    manifest = {
        "episodes": [
            {"episode_id": "ep1", "split": "train"},
            {"episode_id": "ep2", "split": "val"},
            {"episode_id": "ep3", "split": "test"},
        ]
    }
    asgn = assignment_from_manifest(manifest)
    assert asgn["ep2"] == "validation"
    s = split_sets(asgn)
    assert s["validation"] == {"ep2"}
    assert s["train"] == {"ep1"}
    assert s["test"] == {"ep3"}


@pytest.mark.parametrize("length", [5, 12, 13, 50, 151])
def test_window_starts_match_pilot_formula(length):
    spec = WindowSpec(context_len=8, horizon=4, stride=2)
    maximum_start = length - 8 - 4
    assert window_starts(length, spec) == list(range(0, maximum_start + 1, 2))
    for start in window_starts(length, spec):
        assert start + 8 + 4 <= length  # windows never cross the episode end


def test_slice_window_matches_pilot_and_is_causal():
    length, s_dim, a_dim = 30, 3, 2
    t = np.arange(length, dtype=np.float32)
    states = np.repeat(t[:, None], s_dim, axis=1)          # state[t] == t
    actions = np.repeat(t[:, None] + 0.5, a_dim, axis=1)   # action[t] == t + 0.5
    visual = np.repeat(t[:, None] * 10, 4, axis=1)
    spec = WindowSpec(context_len=8, horizon=4, stride=2)
    for start in window_starts(length, spec):
        ours = slice_window(states, actions, start, spec, input_visual=visual, target_visual=visual)
        ref = _pilot_window(states, actions, visual, start, 8, 4)
        for key, value in ref.items():
            np.testing.assert_array_equal(ours[key], value, err_msg=key)
        # Causality: every target time is strictly after every context time.
        assert ours["target_state"][:, 0].min() > ours["context_state"][:, 0].max()
        assert ours["context_state"][-1, 0] == start + 7
        # action[t] drives state[t] -> state[t+1]: first future action is taken at the last context step.
        assert ours["future_actions"][0, 0] == start + 7 + 0.5
        assert ours["context_action"][0, 0] == 0.0 and ours["context_action"][1, 0] == start + 0.5
        np.testing.assert_array_equal(ours["future_visual"], ours["target_visual"])


def test_slice_window_rejects_out_of_range():
    with pytest.raises(IndexError):
        slice_window(np.zeros((10, 2)), np.zeros((10, 1)), 0, WindowSpec(8, 4, 2))


def test_fit_normaliser_matches_pilot_formula():
    rng = np.random.default_rng(0)
    states = [rng.normal(size=(20, 3)).astype(np.float32) for _ in range(3)]
    states[0][:, 2] = 1.0  # a constant column gets the 1e-6 floor only if all episodes are constant
    actions = [rng.normal(size=(20, 2)).astype(np.float32) for _ in range(3)]
    norm = fit_normaliser(states, actions)
    s = np.concatenate([x.astype(np.float64) for x in states])
    np.testing.assert_allclose(norm["state_mean"], s.mean(0).astype(np.float32))
    np.testing.assert_allclose(norm["state_std"], np.maximum(s.std(0), 1e-6).astype(np.float32))
    assert norm["state_std"].dtype == np.float32


def _episode(length, seed, s_dim=3, a_dim=2, d=4):
    rng = np.random.default_rng(seed)
    return {
        "states": rng.normal(size=(length, s_dim)).astype(np.float32),
        "actions": rng.normal(size=(length, a_dim)).astype(np.float32),
        "exterior_embeddings": rng.normal(size=(length, d)).astype(np.float16),
        "wrist_embeddings": rng.normal(size=(length, d)).astype(np.float16),
    }


def test_window_dataset_flat_layout_equals_pilot_concat(tmp_path):
    episodes = [_episode(20, 0), _episode(25, 1)]
    path = tmp_path / "ep1.npz"
    np.savez_compressed(path, **episodes[1])
    records = [
        {"episode_id": "e0", "length": 20, "arrays": episodes[0]},
        {"episode_id": "e1", "length": 25, "cached_path": str(path)},
    ]
    spec = WindowSpec(8, 4, 2)
    norm = fit_normaliser([e["states"] for e in episodes], [e["actions"] for e in episodes])
    ds = WindowDataset(records, spec, norm)
    assert len(ds) == len(window_starts(20, spec)) + len(window_starts(25, spec))
    item = ds[len(ds) - 1]
    ep = episodes[1]
    states_norm = (ep["states"] - norm["state_mean"]) / norm["state_std"]
    actions_norm = (ep["actions"] - norm["action_mean"]) / norm["action_std"]
    visual = np.concatenate([ep["exterior_embeddings"].astype(np.float32), ep["wrist_embeddings"].astype(np.float32)], axis=-1)
    ref = _pilot_window(states_norm, actions_norm, visual, item["window_start"], 8, 4)
    for key, value in ref.items():
        np.testing.assert_allclose(item[key], value, err_msg=key, rtol=1e-6)
    assert item["episode_id"] == "e1"


def test_window_dataset_tokens_layout():
    episodes = [_episode(20, 0)]
    norm = fit_normaliser([episodes[0]["states"]], [episodes[0]["actions"]])
    records = [{"episode_id": "e0", "length": 20, "arrays": episodes[0]}]
    ds = WindowDataset(records, WindowSpec(8, 4, 2), norm, visual_layout="tokens")
    item = ds[0]
    assert item["context_visual"].shape == (8, 2, 4)
    assert item["future_visual"].shape == (4, 2, 4)
    assert item["target_visual"].shape == (4, 8)
    np.testing.assert_allclose(item["context_visual"].reshape(8, -1), WindowDataset(records, WindowSpec(8, 4, 2), norm)[0]["context_visual"])


def test_as_tokens_mixes_global_and_token_features():
    tokens = as_tokens([np.zeros((5, 3)), np.ones((5, 2, 3))])
    assert tokens.shape == (5, 3, 3)
    with pytest.raises(ValueError):
        as_tokens([np.zeros((5, 3)), np.zeros((5, 4))])


# --- DROID parsing -----------------------------------------------------------------

def _step(i, with_images=True, constant=False):
    rng = np.random.default_rng(i)
    image = np.full((4, 4, 3), 7, np.uint8) if constant else rng.integers(0, 255, (4, 4, 3), dtype=np.uint8)
    obs = {
        "cartesian_position": np.arange(6, dtype=np.float32) + i,
        "gripper_position": np.array([0.5], np.float32),
        "joint_position": np.arange(7, dtype=np.float32) - i,
    }
    if with_images:
        obs["exterior_image_1_left"] = image
        obs["wrist_image_left"] = image
    return {"observation": obs, "action": np.arange(7, dtype=np.float32) * i, "language_instruction": b"pick the cup" if i == 2 else b""}


def test_extract_episode_strides_and_records_groups():
    ep = extract_episode([_step(i) for i in range(10)], frame_stride=2)
    assert ep["states"].shape == (5, 14) and ep["actions"].shape == (5, 7)
    assert ep["states"][1, 0] == 2.0  # step 2 kept, step 1 skipped
    assert ep["instruction"] == "pick the cup"
    groups = state_groups(ep["state_keys"], ep["state_dims"])
    assert groups == {
        "cartesian_position": list(range(0, 6)),
        "gripper_position": [6],
        "joint_position": list(range(7, 14)),
    }


def test_extract_episode_refuses_missing_or_constant_images():
    with pytest.raises(KeyError):
        extract_episode([_step(0, with_images=False)])
    with pytest.raises(ValueError):
        extract_episode([_step(0, constant=True)])


def test_choose_episode_id_matches_pilot_rule():
    episode = {"episode_metadata": {"recording_folderpath": b"/a/b", "file_path": b"/a/b/traj.h5"}}
    expected = hashlib.sha256("/a/b|/a/b/traj.h5".encode()).hexdigest()[:24]
    assert choose_episode_id(episode, 3) == expected
    assert choose_episode_id({}, 3) == hashlib.sha256(b"droid100_episode_0003").hexdigest()[:24]


# --- feature cache -----------------------------------------------------------------

def test_feature_cache_build_persist_restore(tmp_path):
    from adjointrwm.data import build_feature_cache, feature_key, persist_cache, restore_cache, write_cache_manifest

    episodes = [
        (0, {"episode_metadata": {"file_path": b"/a"}}, [_step(i) for i in range(20)]),
        (1, {"episode_metadata": {"file_path": b"/b"}}, [_step(i) for i in range(5)]),  # too short
    ]
    embed = lambda images: np.full((len(images), 3), len(images), dtype=np.float16)  # noqa: E731
    records, skipped = build_feature_cache(episodes, {"resnet18": embed}, tmp_path / "local", log=lambda _: None)
    assert [s["reason"] for s in skipped] == ["EPISODE_TOO_SHORT"]
    assert records[0]["length"] == 10 and records[0]["state_dim"] == 14
    assert feature_key("resnet18", "exterior_image_1_left") == "exterior_embeddings"
    assert feature_key("dinov2_vits14", "wrist_image_left") == "wrist_dinov2_vits14"
    with np.load(records[0]["cached_path"]) as cached:
        assert cached["exterior_embeddings"].shape == (10, 3)
    persisted = persist_cache(records, tmp_path / "drive")
    manifest = write_cache_manifest(persisted, skipped, tmp_path / "drive" / "manifest.json")
    restored = restore_cache(manifest, tmp_path / "local2")
    assert restored[0]["cached_path"].startswith(str(tmp_path / "local2"))
    with open(persisted[0]["persisted_path"], "ab") as handle:
        handle.write(b"x")
    with pytest.raises(ValueError):
        restore_cache(manifest, tmp_path / "local3")
