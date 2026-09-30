"""Tests for the same-split dynamics-versus-persistence helpers (adjointrwm.eval.dynamics_parity). NumPy only; no torch needed."""

from __future__ import annotations

import numpy as np
import pytest

from adjointrwm.eval import (
    anchor_check, compare_normalisers, dynamics_gate_row, horizon_mean_rmse, per_window_mse, persistence_forecast,
)


def windows(n=40, context=8, horizon=4, dim=5, seed=0):
    rng = np.random.default_rng(seed)
    context_state = rng.normal(size=(n, context, dim))
    drift = np.cumsum(rng.normal(scale=0.3, size=(n, horizon, dim)), axis=1)
    target = context_state[:, -1:, :] + drift
    return context_state, target


def test_a_model_that_is_persistence_has_zero_improvement_and_fails_the_margin():
    context, target = windows()
    row = dynamics_gate_row(persistence_forecast(context, 4), target, context, 4, margin=0.02)
    assert row["relative_improvement"] == pytest.approx(0.0, abs=1e-12) and row["passed"] is False
    assert row["model_rmse"] == pytest.approx(row["persistence_rmse"]) and row["windows"] == 40
    assert row["improvement_by_horizon"] == pytest.approx([0.0] * 4, abs=1e-12)


def test_a_perfect_model_improves_by_one_and_a_worse_one_is_negative():
    context, target = windows()
    perfect = dynamics_gate_row(target, target, context, 4)
    assert perfect["model_rmse"] == 0.0 and perfect["relative_improvement"] == pytest.approx(1.0) and perfect["passed"] is True
    worse = dynamics_gate_row(target + 5.0, target, context, 4)
    assert worse["relative_improvement"] < 0 and worse["passed"] is False


def test_the_row_matches_the_formula_the_pilot_v2_notebook_uses():
    context, target = windows(seed=3)
    pred = target + np.random.default_rng(9).normal(scale=0.2, size=target.shape)
    row = dynamics_gate_row(pred, target, context, 4, margin=0.02)
    model = horizon_mean_rmse(per_window_mse(pred, target))                       # pilot v2, section 4: model_rmse
    persistence = horizon_mean_rmse(per_window_mse(persistence_forecast(context, 4), target))
    assert row["model_rmse"] == pytest.approx(model, abs=1e-15) and row["persistence_rmse"] == pytest.approx(persistence, abs=1e-15)
    assert row["relative_improvement"] == pytest.approx((persistence - model) / persistence, abs=1e-15)
    assert row["model_rmse"] == pytest.approx(np.mean(row["model_rmse_by_horizon"])) and len(row["improvement_by_horizon"]) == 4


def test_the_margin_decides_the_flag_at_the_boundary():
    context, target = windows(seed=5)
    pred = persistence_forecast(context, 4) * 0.0 + target + 0.5 * (persistence_forecast(context, 4) - target)
    row = dynamics_gate_row(pred, target, context, 4, margin=0.02)
    assert row["relative_improvement"] == pytest.approx(0.5, abs=1e-9) and row["passed"] is True
    assert dynamics_gate_row(pred, target, context, 4, margin=0.6)["passed"] is False


def test_shape_mismatches_are_refused():
    context, target = windows()
    with pytest.raises(ValueError, match="shape mismatch"):
        dynamics_gate_row(target[:, :3], target, context, 4)
    with pytest.raises(ValueError, match="shape mismatch"):
        dynamics_gate_row(target, target, context[:-1], 4)
    with pytest.raises(ValueError, match="shape mismatch"):
        dynamics_gate_row(target, target, context, 3)


def test_compare_normalisers_reports_differences_missing_keys_and_shapes():
    a = {"state_mean": np.zeros(3), "state_std": np.ones(3), "action_mean": np.zeros(2), "extra": np.zeros(1)}
    b = {"state_mean": np.array([0.0, 0.001, 0.0]), "state_std": np.ones(3), "action_mean": np.zeros(4), "other": np.zeros(1)}
    report = compare_normalisers(a, b)
    assert report["only_in_a"] == ["extra"] and report["only_in_b"] == ["other"] and report["shape_mismatch"] == ["action_mean"]
    assert report["max_abs_diff"] == {"state_mean": pytest.approx(0.001), "state_std": 0.0}
    assert report["max_abs_diff_overall"] == pytest.approx(0.001)
    same = compare_normalisers({"x": [1.0, 2.0]}, {"x": np.array([1.0, 2.0])})
    assert same["max_abs_diff_overall"] == 0.0 and same["only_in_a"] == [] and compare_normalisers({}, {})["max_abs_diff_overall"] is None


def test_anchor_check_accepts_a_reproduction_and_rejects_a_drift():
    ok = anchor_check(0.21250, 0.2124711301360882, atol=1e-3)
    assert ok["within"] is True and ok["abs_diff"] == pytest.approx(0.2125 - 0.2124711301360882)
    bad = anchor_check(0.2300, 0.2124711301360882, atol=1e-3)
    assert bad["within"] is False and bad["recomputed"] == 0.23 and bad["stored"] == 0.2124711301360882 and bad["atol"] == 1e-3


# ---- the two model assumptions the diagnostic notebook relies on (tiny CPU models; random fixtures, not data) ----------------------------------

torch = pytest.importorskip("torch")

from adjointrwm.models import AdjointRecursiveWorldModel, AdjointRWMConfig  # noqa: E402  (after the importorskip)


def tiny_model(**changes):
    config = AdjointRWMConfig(context_len=4, horizon=3, d_model=32, transformer_layers=1, transformer_heads=2, transformer_ff=64, **changes)
    torch.manual_seed(0)
    return AdjointRecursiveWorldModel(3, 2, 6, config).eval()


def tiny_batch(b=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    return {"context_state": r(b, 4, 3), "context_action": r(b, 4, 2), "context_visual": r(b, 4, 6), "future_actions": r(b, 3, 2),
            "target_state": r(b, 3, 3), "target_visual": r(b, 3, 6)}


def test_replacing_the_config_switches_the_prediction_mode_and_matches_a_model_built_in_that_mode():
    import dataclasses

    batch = tiny_batch()
    model = tiny_model(prediction_mode="base")
    with torch.no_grad():
        base = model.predict(batch)["state_mean"]
        model.config = dataclasses.replace(model.config, prediction_mode="full")
        full = model.predict(batch)["state_mean"]
        built_full = tiny_model(prediction_mode="full")
        built_full.load_state_dict(model.state_dict(), strict=True)
        expected = built_full.predict(batch)["state_mean"]
    assert not torch.allclose(base, full)                                   # the two modes really differ
    assert torch.equal(full, expected)                                      # switching the config equals building in that mode


def test_the_mask_mode_does_not_change_the_parameters_so_a_pilot_v1_checkpoint_loads_into_the_v2_model():
    subset, single = tiny_model(mask_mode="subset"), tiny_model(mask_mode="single_choice")
    assert list(subset.state_dict()) == list(single.state_dict())
    assert all(a.shape == b.shape for a, b in zip(subset.state_dict().values(), single.state_dict().values()))
    single.load_state_dict(subset.state_dict(), strict=True)
    batch = tiny_batch()
    with torch.no_grad():
        assert torch.equal(single.predict(batch)["state_mean"], subset.predict(batch)["state_mean"])      # same parameters, same prediction mode, same output
