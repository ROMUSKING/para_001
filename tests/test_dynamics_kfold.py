"""Tests for the k-fold dynamics study's statistics (adjointrwm.eval.dynamics_parity; plan: docs/plans/dynamics-kfold-plan.md). NumPy and pandas only.

Arrays are random test fixtures that exercise code paths; nothing here is data or evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adjointrwm.data.windows import split_order_key
from adjointrwm.eval import (
    classify_against_margin, cluster_bootstrap_relative_improvement, dynamics_gate_row, episode_error_table, inner_validation_split, kfold_assignment,
    paired_bootstrap_difference, pooled_relative_improvement, random_subset_gate_rate,
)

HORIZON = 4


def windows(episodes=12, per_episode=7, dim=5, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    ids, context, target = [], [], []
    for e in range(episodes):
        scale = 0.2 + 0.4 * rng.random()                                          # episodes differ in how much they move
        ctx = rng.normal(size=(per_episode, 8, dim))
        drift = np.cumsum(rng.normal(scale=scale, size=(per_episode, HORIZON, dim)), axis=1)
        context.append(ctx); target.append(ctx[:, -1:, :] + drift); ids += [f"ep{e:03d}"] * per_episode
    context, target = np.concatenate(context), np.concatenate(target)
    pred = target + rng.normal(scale=noise, size=target.shape)
    return pred, target, context, ids


# ---- folds --------------------------------------------------------------------------------------------------------------------------------------

def test_kfold_assignment_is_balanced_deterministic_and_uses_the_pilots_hash_order():
    ids = [f"episode-{i}" for i in range(100)]
    folds = kfold_assignment(ids, 5)
    assert sorted(folds) == sorted(ids) and {v for v in folds.values()} == {0, 1, 2, 3, 4}
    assert all(list(folds.values()).count(f) == 20 for f in range(5))                     # 20 episodes per fold, each held out exactly once
    assert folds == kfold_assignment(reversed(ids), 5)                                      # independent of input order
    ordered = sorted(ids, key=split_order_key)
    assert [folds[e] for e in ordered] == [rank % 5 for rank in range(100)]
    with pytest.raises(ValueError, match="unique"):
        kfold_assignment(["a", "a", "b"], 2)
    with pytest.raises(ValueError, match="k must be"):
        kfold_assignment(["a", "b", "c"], 5)


def test_inner_validation_split_takes_the_first_ids_in_hash_order_and_never_overlaps():
    ids = [f"episode-{i}" for i in range(80)]
    train, validation = inner_validation_split(ids, 10)
    assert len(train) == 70 and len(validation) == 10 and not set(train) & set(validation) and set(train) | set(validation) == set(ids)
    assert validation == sorted(ids, key=split_order_key)[:10]
    with pytest.raises(ValueError, match="n_validation"):
        inner_validation_split(ids, 0)
    with pytest.raises(ValueError, match="n_validation"):
        inner_validation_split(ids, 80)


def test_a_held_out_fold_shares_no_episode_with_its_training_or_validation_episodes():
    ids = [f"episode-{i}" for i in range(100)]
    folds = kfold_assignment(ids, 5)
    for fold in range(5):
        held_out = {e for e, f in folds.items() if f == fold}
        train, validation = inner_validation_split([e for e in ids if e not in held_out], 10)
        assert len(held_out) == 20 and not held_out & set(train) and not held_out & set(validation)


# ---- the pooled estimate and its bootstrap ------------------------------------------------------------------------------------------------------

def test_the_episode_table_pools_to_exactly_the_gate_row_computed_on_all_windows():
    pred, target, context, ids = windows()
    table = episode_error_table(pred, target, context, ids, HORIZON)
    pooled = pooled_relative_improvement(table)
    gate = dynamics_gate_row(pred, target, context, HORIZON)
    assert pooled["windows"] == len(pred) and pooled["episodes"] == 12
    for key in ("model_rmse", "persistence_rmse", "relative_improvement"):
        assert pooled[key] == pytest.approx(gate[key], abs=1e-12), key
    assert pooled["improvement_by_horizon"] == pytest.approx(gate["improvement_by_horizon"], abs=1e-12)
    assert list(table["episode_id"]) == sorted(set(ids)) and table["windows"].sum() == len(pred)


def test_episode_error_table_rejects_shape_mismatches():
    pred, target, context, ids = windows(episodes=3)
    with pytest.raises(ValueError, match="shape mismatch"):
        episode_error_table(pred, target, context, ids[:-1], HORIZON)
    with pytest.raises(ValueError, match="shape mismatch"):
        episode_error_table(pred[:, :3], target, context, ids, HORIZON)


def test_bootstrap_brackets_the_estimate_is_deterministic_and_collapses_when_every_episode_agrees():
    pred, target, context, ids = windows(seed=2)
    table = episode_error_table(pred, target, context, ids, HORIZON)
    a = cluster_bootstrap_relative_improvement(table, resamples=2000, seed=3)
    b = cluster_bootstrap_relative_improvement(table, resamples=2000, seed=3)
    assert a == b and a["ci_low"] < a["estimate"] < a["ci_high"] and a["episodes"] == 12 and len(a["by_horizon"]) == HORIZON
    assert a["ci_high"] - a["ci_low"] > 0.0
    assert cluster_bootstrap_relative_improvement(table, resamples=2000, seed=4)["ci_low"] != a["ci_low"]      # the seed matters
    uniform = pd.DataFrame({"episode_id": list("abcde"), "windows": 10, **{f"model_sse_h{h}": 4.0 for h in range(1, 5)}, **{f"persistence_sse_h{h}": 9.0 for h in range(1, 5)}})
    flat = cluster_bootstrap_relative_improvement(uniform, resamples=500, seed=0)
    assert flat["ci_low"] == pytest.approx(flat["estimate"]) == pytest.approx(flat["ci_high"]) == pytest.approx(1 - np.sqrt(4 / 9))


def test_a_model_equal_to_persistence_has_zero_improvement_and_a_perfect_model_has_one():
    pred, target, context, ids = windows(seed=5)
    same = episode_error_table(np.repeat(context[:, -1:, :], HORIZON, axis=1), target, context, ids, HORIZON)
    assert pooled_relative_improvement(same)["relative_improvement"] == pytest.approx(0.0, abs=1e-12)
    perfect = episode_error_table(target, target, context, ids, HORIZON)
    boot = cluster_bootstrap_relative_improvement(perfect, resamples=300, seed=0)
    assert boot["estimate"] == pytest.approx(1.0) and boot["ci_low"] == pytest.approx(1.0) and boot["ci_high"] == pytest.approx(1.0)


def test_the_interval_reflects_how_much_the_episodes_differ():
    rng = np.random.default_rng(0)
    def table(spread):
        rows = {"episode_id": [f"e{i}" for i in range(40)], "windows": 10}
        gain = 0.6 + spread * rng.normal(size=40)                                  # per-episode model/persistence error ratio
        for h in range(1, 5):
            rows[f"persistence_sse_h{h}"] = np.full(40, 10.0); rows[f"model_sse_h{h}"] = 10.0 * np.clip(gain, 0.05, 3.0) ** 2
        return pd.DataFrame(rows)
    narrow = cluster_bootstrap_relative_improvement(table(0.02), resamples=3000, seed=1)
    wide = cluster_bootstrap_relative_improvement(table(0.5), resamples=3000, seed=1)
    assert (wide["ci_high"] - wide["ci_low"]) > 5 * (narrow["ci_high"] - narrow["ci_low"])


# ---- the gate pass rate of a random ten-episode split -------------------------------------------------------------------------------------------

def test_the_subset_gate_rate_is_one_or_zero_when_every_episode_agrees_and_between_when_they_differ():
    def uniform(model):
        return pd.DataFrame({"episode_id": [f"e{i}" for i in range(30)], "windows": 10, **{f"model_sse_h{h}": model for h in range(1, 5)}, **{f"persistence_sse_h{h}": 10.0 for h in range(1, 5)}})
    assert random_subset_gate_rate(uniform(5.0), 10, 500, 0, 0.02)["pass_rate"] == 1.0           # R = 1 - sqrt(0.5) = 0.29 >= 0.02
    assert random_subset_gate_rate(uniform(12.0), 10, 500, 0, 0.02)["pass_rate"] == 0.0          # the model is worse than persistence
    pred, target, context, ids = windows(episodes=40, seed=7, noise=0.7)
    mixed = random_subset_gate_rate(episode_error_table(pred, target, context, ids, HORIZON), 10, 2000, 0, 0.02)
    assert 0.0 < mixed["pass_rate"] < 1.0 and mixed["quantiles"]["q05"] < mixed["quantiles"]["q50"] < mixed["quantiles"]["q95"]
    with pytest.raises(ValueError, match="subset_size"):
        random_subset_gate_rate(uniform(5.0), 31)


def test_subset_draws_are_without_replacement():
    table = pd.DataFrame({"episode_id": ["a", "b", "c"], "windows": [1, 1, 1], **{f"model_sse_h{h}": [1.0, 2.0, 3.0] for h in range(1, 5)}, **{f"persistence_sse_h{h}": [4.0, 4.0, 4.0] for h in range(1, 5)}})
    full = random_subset_gate_rate(table, 3, 50, 0)                                # drawing all three every time: no variation across draws
    assert full["quantiles"]["q05"] == pytest.approx(full["quantiles"]["q95"])


# ---- the frozen decision rule -------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("low, high, expected", [
    (0.03, 0.30, "BEATS"), (0.0201, 0.05, "BEATS"), (0.02, 0.30, "INCONCLUSIVE"),            # strict: a bound equal to the margin is not above it
    (-0.30, -0.01, "WORSE_THAN_PERSISTENCE"), (-0.30, 0.0, "FAILS"),                       # an upper bound of exactly 0 is not below 0
    (-0.10, 0.01, "FAILS"), (-0.10, 0.02, "INCONCLUSIVE"), (-0.05, 0.10, "INCONCLUSIVE"),
])
def test_the_decision_rule_classes_match_the_plan_including_its_boundaries(low, high, expected):
    assert classify_against_margin(low, high, 0.02) == expected


# ---- the paired comparison between regimes ------------------------------------------------------------------------------------------------------

def test_the_paired_difference_is_zero_for_identical_tables_and_detects_a_clear_gap():
    pred, target, context, ids = windows(episodes=30, seed=9, noise=0.3)
    good = episode_error_table(pred, target, context, ids, HORIZON)
    worse_pred = target + np.random.default_rng(1).normal(scale=1.5, size=target.shape)
    bad = episode_error_table(worse_pred, target, context, ids, HORIZON)
    same = paired_bootstrap_difference(good, good, resamples=500, seed=0)
    assert same["estimate"] == 0.0 and same["ci_low"] == 0.0 and same["ci_high"] == 0.0 and same["label"] == "NOT_DISTINGUISHED"
    gap = paired_bootstrap_difference(good, bad, resamples=2000, seed=0)
    assert gap["label"] == "A_BETTER" and gap["ci_low"] > 0 and gap["estimate"] == pytest.approx(
        pooled_relative_improvement(good)["relative_improvement"] - pooled_relative_improvement(bad)["relative_improvement"])
    assert paired_bootstrap_difference(bad, good, resamples=2000, seed=0)["label"] == "B_BETTER"
    with pytest.raises(ValueError, match="same episodes"):
        paired_bootstrap_difference(good, good.iloc[:-1], resamples=10)
