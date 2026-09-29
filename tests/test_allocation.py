import numpy as np
import pandas as pd
import pytest

from adjointrwm.analysis import (
    choice_distribution,
    episode_bootstrap_ci,
    first_order_scores,
    opportunity_audit,
    policy_regret,
    summarize_traces,
)


def _traces(n_per_episode=20, episodes=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(episodes):
        for w in range(n_per_episode):
            oracle = int(rng.integers(0, 4))
            rows.append(
                {
                    "episode_id": f"ep{e}",
                    "window_start": w,
                    "oracle_choice": oracle,
                    "adjoint_choice": 0,
                    "critic_choice": 2,
                    "hybrid_choice": 2,
                    "gate_probability": 0.1,
                    "adjoint_regret": 0.0 if oracle == 0 else 1.0,
                    "critic_regret": 0.0 if oracle == 2 else 1.0,
                    "hybrid_regret": 0.0 if oracle == 2 else 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_choice_distribution_rows_sum_to_one():
    dist = choice_distribution(_traces(), num_candidates=4)
    assert np.allclose(dist.sum(axis=1), 1.0)
    assert dist.loc["adjoint", "c0"] == 1.0


def test_summary_flags_collapsed_policy():
    summary = summarize_traces(_traces(), num_candidates=4)
    assert summary["unused_candidates"]["adjoint"] == [1, 2, 3]
    assert summary["adjoint_critic_agreement"] == 0.0
    assert summary["gate"]["invocation_rate"] == 0.0


def test_bootstrap_ci_contains_estimate_and_counts_episodes():
    ci = episode_bootstrap_ci(_traces(), "adjoint_regret", num_resamples=2000)
    assert ci["ci_low"] <= ci["estimate"] <= ci["ci_high"]
    assert ci["num_episodes"] == 5


def test_bootstrap_paired_difference_of_identical_columns_is_zero():
    frame = _traces()
    ci = episode_bootstrap_ci(frame, "critic_regret", baseline="hybrid_regret", num_resamples=500)
    assert ci["estimate"] == ci["ci_low"] == ci["ci_high"] == 0.0


def test_opportunity_audit_fails_when_one_candidate_always_wins():
    gain = np.tile([0.0, 0.1, 0.05, 0.02], (100, 1))
    report = opportunity_audit(gain)
    assert report.best_fixed_candidate == 1
    assert report.headroom_over_best_fixed == pytest.approx(0.0)
    assert not report.passes


def test_opportunity_audit_passes_when_best_candidate_varies():
    gain = np.zeros((100, 4))
    gain[np.arange(100), np.arange(100) % 4] = 1.0
    report = opportunity_audit(gain, candidate_costs=[0.002, 0.002, 0.003, 0.004])
    assert report.passes
    assert report.fraction_oracle_differs_from_best_fixed == pytest.approx(0.75)
    assert report.cost_gap == pytest.approx(0.002)


def test_policy_regret_zero_for_oracle():
    rng = np.random.default_rng(1)
    gain = rng.normal(size=(50, 4))
    assert np.allclose(policy_regret(gain, gain.argmax(axis=1)), 0.0)


def test_first_order_scores_match_manual_dot_product():
    costate = np.array([[1.0, -2.0]])
    effects = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    scores = first_order_scores(costate, effects, np.array([0.1, 0.2]))
    assert np.allclose(scores, [[-1.1, 1.8]])
