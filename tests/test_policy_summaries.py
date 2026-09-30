"""Pilot v2 policy summaries: regrets from gains only, fixed baselines, critic floor (pure pandas).

Traces are random test fixtures, not data.
"""

import numpy as np
import pandas as pd

from adjointrwm.analysis import critic_realization_floor, policy_regret_frame, summarize_policies


def _traces(n=200, k=5, seed=0):
    rng = np.random.default_rng(seed)
    gains = rng.normal(size=(n, k)) * 0.01
    gains[:, 0] = 0.0
    frame = pd.DataFrame({f"gain_{j}": gains[:, j] for j in range(k)})
    frame["episode_id"] = [f"e{i % 10}" for i in range(n)]
    frame["oracle_choice"] = gains.argmax(1)
    frame["adjoint_choice"] = gains.argmax(1)            # perfect
    frame["critic_choice"] = gains.argmin(1)             # always the worst option
    frame["uncertainty_choice"] = rng.integers(0, k, n)
    return frame, gains


def test_policy_regrets_are_computed_from_gains_only():
    frame, gains = _traces()
    regrets = policy_regret_frame(frame)
    best = gains.max(1)
    np.testing.assert_allclose(regrets["adjoint"], 0.0)
    np.testing.assert_allclose(regrets["critic"], best - gains.min(1))
    np.testing.assert_allclose(regrets["always_hold"], best)
    np.testing.assert_allclose(regrets["random_expected"], (best[:, None] - gains).mean(1))
    # Renaming a policy does not change its numbers.
    renamed = frame.rename(columns={"critic_choice": "other_choice"})
    np.testing.assert_allclose(policy_regret_frame(renamed)["other"], regrets["critic"])


def test_summarize_policies_and_critic_floor():
    frame, _ = _traces()
    summary = summarize_policies(frame, candidate_costs=[0, 0.002, 0.002, 0.003, 0.004], num_resamples=500)
    assert summary["num_options"] == 5
    assert summary["policies"]["adjoint"]["top1"] == 1.0
    assert summary["paired_differences"]["adjoint-critic"]["ci_high"] < 0
    floor = critic_realization_floor(summary)
    assert not floor["passed"] and not floor["beats_random_expected"]
    frame["critic_choice"] = frame["oracle_choice"]
    good = critic_realization_floor(summarize_policies(frame, num_resamples=200))
    assert good["passed"]
