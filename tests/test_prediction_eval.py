"""Prediction metrics: method blindness, paired bootstrap, classification and baselines.

Arrays here are small random test fixtures that exercise the code; they are not data.
"""

import numpy as np
import pandas as pd
import pytest

from adjointrwm.eval import (
    RidgeForecaster,
    classify_relative_difference,
    derangement,
    effective_rank,
    horizon_mean_rmse,
    interval_coverage,
    paired_relative_difference,
    per_window_mse,
    persistence_forecast,
    prediction_frame,
    rmse_by_horizon,
    summarize_frame,
)
from adjointrwm.io import atomic_copy, atomic_write_json, file_record, sha256_file, sha256_json


def _frame(arm, seed, pred, target, episodes, std=None):
    n = len(pred)
    return prediction_frame(
        arm=arm, seed=seed, episode_ids=episodes, window_starts=np.arange(n),
        pred_state=pred, target_state=target, state_std=np.ones(pred.shape[-1]) if std is None else std,
        groups={"a": [0], "b": [1, 2]},
    )


def _fixture(n=60, h=4, s=3, episodes=6, seed=0):
    rng = np.random.default_rng(seed)
    target = rng.normal(size=(n, h, s))
    eps = np.array([f"e{i % episodes}" for i in range(n)])
    return rng, target, eps


def test_metrics_are_method_blind():
    rng, target, eps = _fixture()
    pred = target + rng.normal(scale=0.3, size=target.shape)
    a = _frame("adjoint_rwm", 0, pred, target, eps)
    b = _frame("some_rival", 0, pred, target, eps)
    metric_cols = [c for c in a.columns if c not in ("arm",)]
    pd.testing.assert_frame_equal(a[metric_cols], b[metric_cols])
    sa = summarize_frame(a).drop(columns="arm")
    sb = summarize_frame(b).drop(columns="arm")
    pd.testing.assert_frame_equal(sa, sb)


def test_summary_primary_endpoint_is_mean_of_horizon_rmse():
    rng, target, eps = _fixture()
    pred = target + rng.normal(scale=0.5, size=target.shape)
    summary = summarize_frame(_frame("x", 0, pred, target, eps)).iloc[0]
    mse = per_window_mse(pred, target)
    assert summary["rmse_norm"] == pytest.approx(horizon_mean_rmse(mse))
    assert summary["rmse_norm_h2"] == pytest.approx(rmse_by_horizon(mse)[1])
    assert summary["num_episodes"] == 6


def test_native_rmse_scales_with_state_std():
    rng, target, eps = _fixture()
    pred = target + 0.1
    frame = _frame("x", 0, pred, target, eps, std=np.array([2.0, 1.0, 1.0]))
    summary = summarize_frame(frame).iloc[0]
    assert summary["rmse_native_a"] == pytest.approx(0.2)
    assert summary["rmse_native_b"] == pytest.approx(0.1)


def test_paired_difference_is_zero_for_identical_predictions():
    rng, target, eps = _fixture()
    pred = target + rng.normal(scale=0.3, size=target.shape)
    frame = pd.concat([_frame("ref", s, pred, target, eps) for s in (0, 1)] + [_frame("riv", s, pred, target, eps) for s in (0, 1)])
    out = paired_relative_difference(frame, "ref", "riv", num_resamples=200)
    assert out["relative_difference"] == pytest.approx(0.0)
    assert out["ci_low"] == pytest.approx(0.0) and out["ci_high"] == pytest.approx(0.0)
    assert classify_relative_difference(out["ci_low"], out["ci_high"], 0.02) == "equivalent_within_margin"


def test_paired_difference_detects_better_reference_and_unpaired_seeds():
    rng, target, eps = _fixture(n=120, episodes=10)
    frames = []
    for s in (0, 1, 2):
        frames.append(_frame("ref", s, target + rng.normal(scale=0.1, size=target.shape), target, eps))
        frames.append(_frame("riv", s, target + rng.normal(scale=0.5, size=target.shape), target, eps))
    frames.append(_frame("riv", 9, target, target, eps))  # seed without a partner
    out = paired_relative_difference(pd.concat(frames), "ref", "riv", num_resamples=500)
    assert out["relative_difference"] < -0.5
    assert out["ci_high"] < 0
    assert out["unpaired_seeds"] == [9] and out["seeds"] == [0, 1, 2]
    assert set(out["per_seed"]) == {0, 1, 2}
    assert classify_relative_difference(out["ci_low"], out["ci_high"], 0.02) == "reference_better"


def test_paired_difference_requires_identical_windows():
    rng, target, eps = _fixture()
    a = _frame("ref", 0, target, target, eps)
    b = _frame("riv", 0, target[:-1], target[:-1], eps[:-1])
    with pytest.raises(ValueError, match="same windows"):
        paired_relative_difference(pd.concat([a, b]), "ref", "riv")


@pytest.mark.parametrize(
    "low,high,expected",
    [(-0.2, -0.01, "reference_better"), (0.01, 0.3, "rival_better"), (-0.01, 0.015, "equivalent_within_margin"), (-0.1, 0.05, "inconclusive")],
)
def test_classification_rule(low, high, expected):
    assert classify_relative_difference(low, high, 0.02) == expected


def test_persistence_repeats_last_context_state():
    context = np.arange(24, dtype=float).reshape(2, 4, 3)
    pred = persistence_forecast(context, 5)
    assert pred.shape == (2, 5, 3)
    np.testing.assert_array_equal(pred[:, 3], context[:, -1])


def _linear_split(n, seed):
    rng = np.random.default_rng(seed)
    cs = rng.normal(size=(n, 8, 3))
    ca = rng.normal(size=(n, 8, 2))
    fa = rng.normal(size=(n, 4, 2))
    b = np.array([[1.0, -0.5, 0.2], [0.3, 0.1, -1.0]])
    delta = np.cumsum(fa @ b, axis=1)  # state increments driven by actions only
    return {"context_state": cs, "context_action": ca, "future_actions": fa, "target_state": cs[:, -1:, :] + delta}


def test_ridge_recovers_linear_action_dynamics():
    train, val, test = _linear_split(400, 0), _linear_split(100, 1), _linear_split(50, 2)
    model = RidgeForecaster(lambdas=(1e-6, 1e-3, 10.0)).fit(train, val)
    pred = model.predict(test["context_state"], test["context_action"], test["future_actions"])
    assert np.abs(pred - test["target_state"]).max() < 1e-3
    assert model.lambda_ in (1e-6, 1e-3)


def test_derangement_has_no_fixed_points():
    for n in (2, 3, 10, 101):
        for seed in range(5):
            perm = derangement(n, seed)
            assert sorted(perm) == list(range(n))
            assert not (perm == np.arange(n)).any()


def test_effective_rank_bounds():
    rng = np.random.default_rng(0)
    iso = rng.normal(size=(2000, 8))
    assert effective_rank(iso) > 7.5
    rank_one = np.outer(rng.normal(size=500), rng.normal(size=8))
    assert effective_rank(rank_one) < 1.01


def test_interval_coverage():
    mean = np.zeros((1, 1, 4))
    target = np.array([[[0.5, 1.5, -2.5, 0.0]]])
    assert interval_coverage(mean, np.zeros_like(mean), target, 1.0)[0, 0] == pytest.approx(0.5)
    assert interval_coverage(mean, np.zeros_like(mean), target, 2.0)[0, 0] == pytest.approx(0.75)


def test_atomic_io_roundtrip(tmp_path):
    path = atomic_write_json(tmp_path / "a" / "x.json", {"b": 1, "a": [1, 2]})
    assert path.read_text().startswith("{")
    copy = tmp_path / "drive" / "x.json"
    digest = atomic_copy(path, copy)
    assert digest == sha256_file(copy) == (tmp_path / "drive" / "x.json.sha256").read_text().strip()
    assert file_record(copy)["bytes"] == path.stat().st_size
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})
    with pytest.raises(FileNotFoundError):
        file_record(tmp_path / "missing.pt")
