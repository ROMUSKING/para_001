"""Correctness tests for adjointrwm.domains.critics (D4-1: learned direct critics against co-state critics)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from adjointrwm.domains import critics as cr
from adjointrwm.domains import highdim as hd
from adjointrwm.domains.linear_ode import AdaptiveTimeSteppingDomain, marking_policy, uniform_pass_policy


@pytest.fixture(scope="module")
def cell():
    return dataclasses.replace(hd.CELLS["m4"], max_depth=5)


@pytest.fixture(scope="module")
def domain():
    return AdaptiveTimeSteppingDomain()


@pytest.fixture(scope="module")
def instances(cell):
    return hd.varying_goal_instances(cell, "train", 6)


@pytest.fixture(scope="module")
def samples(domain, instances):
    return cr.collect_states(domain, instances, [uniform_pass_policy(), marking_policy("residual", 0.85)], max_passes=6, per_state=16, seed=0)


@pytest.fixture(scope="module")
def rows(domain, samples, instances):
    return cr.build_rows(domain, samples, instances, seed=0, goals_per_state=4, max_rows=600)


# ---------------------------------------------------------------- co-state operators and labels

def test_the_propagator_reproduces_the_tabulated_costate_of_any_goal(instances):
    inst = instances[0]
    phi = cr.propagator(inst)
    assert phi.shape == (inst.n_fine + 1, inst.m, inst.m) and np.allclose(phi[-1], np.eye(inst.m))
    for other in instances[:3]:
        table = hd.costate_table(dataclasses.replace(inst, goal=other.goal))
        assert np.allclose(phi @ np.asarray(other.goal), table, atol=1e-10)


def test_continuous_costate_picks_the_requested_node_and_goal(instances):
    inst = instances[0]
    nodes = np.array([0, 5, 77, inst.n_fine])
    goals = np.array([instances[i % 3].goal for i in range(4)])
    out = cr.continuous_costate(inst, nodes, goals, chunk=3)
    for k in range(4):
        assert np.allclose(out[k], hd.costate_table(dataclasses.replace(inst, goal=tuple(goals[k])))[nodes[k]], atol=1e-10)


def test_discrete_costates_equal_the_domain_costate_for_each_goal(domain, instances):
    inst = instances[1]
    start = domain.initial_state(inst)
    state = domain.apply_batch(start, domain.legal_candidates(start)[:5])
    goals = np.array([g.goal for g in instances[:4]])
    many = cr.discrete_costates(domain, inst, state.nodes, goals)
    for g, goal in enumerate(goals):
        carrying = dataclasses.replace(inst, goal=tuple(goal))
        assert np.allclose(many[g], domain.costate(type(state)(carrying, state.nodes, state.y)), atol=1e-12)


def test_gain_labels_equal_the_objectives_own_summand_for_the_instances_goal(domain, instances):
    inst = instances[0]
    state = domain.initial_state(inst)
    tau = domain.exact_local_errors(state, inst)
    disc = cr.discrete_costates(domain, inst, state.nodes, np.array([inst.goal]))[0]
    weighted = np.abs(domain.weighted_local_errors(state, inst))
    assert np.allclose(10.0 ** cr.gain_labels(disc, tau) - cr.LABEL_FLOOR, weighted, rtol=1e-9, atol=1e-14)


def test_costate_features_with_the_exact_costate_are_the_cheap_adjoint_score_terms(domain, instances):
    inst = instances[0]
    state = domain.initial_state(inst)
    d, e = hd.forcing_discrepancy(inst, state), hd.difference_estimate(inst, state)
    w = hd.costate_table(inst)[np.asarray(state.nodes[1:])]
    feats = cr.costate_features(w, d, e)
    teacher = np.abs(np.einsum("jm,jm->j", w, d)) + np.abs(np.einsum("jm,jm->j", w, e))
    assert np.allclose(10.0 ** feats[:, 0] + 10.0 ** feats[:, 1] - 2 * cr.LABEL_FLOOR, teacher, rtol=1e-9, atol=1e-18)
    assert np.allclose(10.0 ** feats[:, 2] - cr.LABEL_FLOOR, np.linalg.norm(w, axis=1))


# ---------------------------------------------------------------- states and rows

def test_collected_states_carry_p0_inputs_and_privileged_errors_for_sampled_intervals(samples):
    assert samples and all(len(s.intervals) <= 16 for s in samples)
    s = samples[0]
    k, m = len(s.intervals), s.d.shape[1]
    assert s.scalars.shape == (k, hd.FEATURE_DIM) and s.e.shape == (k, m) and s.tau.shape == (k, m)
    assert len({x.state_id for x in samples}) == len(samples)


def test_rows_are_capped_seeded_and_split_by_instance(domain, samples, instances, rows):
    assert len(rows) == 600
    again = cr.build_rows(domain, samples, instances, seed=0, goals_per_state=4, max_rows=600)
    other = cr.build_rows(domain, samples, instances, seed=1, goals_per_state=4, max_rows=600)
    assert np.array_equal(rows.y, again.y) and not np.array_equal(rows.y, other.y)
    assert np.allclose(np.linalg.norm(rows.goal, axis=1), 1.0) and ((rows.t_right > 0) & (rows.t_right <= 1)).all()
    mask = cr.split_by_group(rows.group, seed=0)
    assert mask.any() and not mask.all()
    assert set(rows.group[mask]).isdisjoint(rows.group[~mask])


def test_the_split_equals_the_one_train_scorer_makes(rows):
    x = np.random.default_rng(0).normal(size=(len(rows), hd.FEATURE_DIM))
    model = hd.train_scorer(x, rows.y, rows.group, seed=3, hidden=(4, 4), max_epochs=1, max_samples=len(rows))
    mask = cr.split_by_group(rows.group, seed=3)
    assert model.report["n_val"] == int(mask.sum()) and model.report["n_train"] == int((~mask).sum())


def test_row_costate_label_is_the_propagator_at_the_right_node_times_the_goal(samples, instances, rows):
    first = rows.take(slice(0, 4))
    for i in range(4):
        s = next(x for x in samples if x.state_id == first.state_id[i])
        inst = instances[s.instance_index]
        match = [j for j in range(len(s.intervals)) if np.allclose(s.d[j], first.d[i]) and np.allclose(s.e[j], first.e[i])]
        assert match
        right = s.nodes[s.intervals[match[0]] + 1]
        assert first.t_right[i] == pytest.approx(right / inst.n_fine)
        assert np.allclose(first.lam[i], cr.propagator(inst)[right] @ first.goal[i], atol=1e-10)


# ---------------------------------------------------------------- estimator

def test_estimator_backpropagation_matches_finite_differences():
    rng = np.random.default_rng(0)
    mlp = hd.TinyMLP((5, 6, 6, 3), rng)
    x, target = rng.normal(size=(9, 5)), rng.normal(size=(9, 3))
    _, gw, gb = cr.estimator_loss_and_grads(mlp, x, target)
    for params, grads in ((mlp.weights, gw), (mlp.biases, gb)):
        for p, g in zip(params, grads):
            flat, gflat = p.reshape(-1), g.reshape(-1)
            for idx in np.random.default_rng(3).choice(len(flat), size=min(6, len(flat)), replace=False):
                old = flat[idx]
                flat[idx] = old + 1e-6
                up = cr.estimator_loss_and_grads(mlp, x, target)[0]
                flat[idx] = old - 1e-6
                down = cr.estimator_loss_and_grads(mlp, x, target)[0]
                flat[idx] = old
                assert gflat[idx] == pytest.approx((up - down) / 2e-6, rel=1e-5, abs=1e-8)


@pytest.fixture(scope="module")
def estimators(rows):
    mask = cr.split_by_group(rows.group, seed=0)
    train, val = rows.take(~mask), rows.take(mask)
    return {"true": cr.train_estimator(train, val, hidden=8, seed=0, max_epochs=25, patience=8, batch_size=64),
            "permuted": cr.train_estimator(train, val, hidden=8, seed=0, permute=True, max_epochs=25, patience=8, batch_size=64)}, train, val


def test_the_estimator_learns_the_costate_and_the_permuted_control_does_not(instances):
    inst = instances[0]
    rng = np.random.default_rng(0)
    n = 12_000
    nodes = rng.integers(1, inst.n_fine + 1, size=n)
    goal = cr.random_unit_goals(rng, n, inst.m)
    synthetic = cr.Rows(np.zeros((n, hd.FEATURE_DIM)), goal, np.zeros((n, inst.m)), np.zeros((n, inst.m)), nodes / inst.n_fine,
                        cr.continuous_costate(inst, nodes, goal), np.zeros(n), rng.integers(0, 30, size=n), np.arange(n) // 20)
    mask = cr.split_by_group(synthetic.group, seed=0)
    kw = dict(hidden=16, seed=0, max_epochs=150, patience=150, batch_size=128)
    true = cr.train_estimator(synthetic.take(~mask), synthetic.take(mask), **kw)
    permuted = cr.train_estimator(synthetic.take(~mask), synthetic.take(mask), permute=True, **kw)
    assert true.report["val_relative_error_against_true_costate"] < 0.8            # 0.63 when written
    assert permuted.report["val_relative_error_against_true_costate"] > 0.9        # predicting zero scores 1
    assert true.report["permuted_labels"] is False and permuted.report["permuted_labels"] is True
    assert true.predict(synthetic.t_right[:5], synthetic.goal[:5]).shape == (5, inst.m)


def test_estimator_training_is_deterministic_and_round_trips(estimators):
    models, train, val = estimators
    again = cr.train_estimator(train, val, hidden=8, seed=0, max_epochs=25, patience=8, batch_size=64)
    assert all(np.array_equal(a, b) for a, b in zip(models["true"].mlp.weights, again.mlp.weights))
    clone = cr.Estimator.from_json(models["true"].to_json())
    assert np.allclose(clone.predict(val.t_right, val.goal), models["true"].predict(val.t_right, val.goal))


def test_relative_error_is_one_for_a_zero_prediction():
    target = np.random.default_rng(0).normal(size=(10, 4))
    assert cr.relative_costate_error(np.zeros_like(target), target) == pytest.approx(1.0)
    assert cr.relative_costate_error(target, target) == 0.0


# ---------------------------------------------------------------- heads and critics

@pytest.fixture(scope="module")
def critics(rows, estimators):
    models, _, _ = estimators
    est = {"estimator": models["true"], "costate_critic": models["true"], "costate_randomised": models["permuted"]}
    return {arm: cr.train_critic(arm, rows, hidden=6, seed=0, estimator=est.get(arm), max_epochs=6, patience=3) for arm in cr.ARMS}


def test_head_inputs_add_exactly_three_costate_features_to_the_co_state_arms(rows, estimators):
    models, _, _ = estimators
    base = cr.head_inputs("direct", rows, None)
    assert base.shape == (len(rows), hd.FEATURE_DIM + 3 * rows.goal.shape[1])
    for arm in ("costate_critic", "costate_randomised"):
        x = cr.head_inputs(arm, rows, models["true"])
        assert x.shape[1] == base.shape[1] + cr.COSTATE_FEATURES and np.array_equal(x[:, :base.shape[1]], base)
    teacher = cr.head_inputs("teacher_feature", rows, None)
    assert np.allclose(teacher[:, base.shape[1]:], cr.costate_features(rows.lam, rows.d, rows.e))
    with pytest.raises(ValueError):
        cr.train_critic("costate_critic", rows, hidden=4, seed=0, estimator=None)


def test_critics_train_predict_and_report(critics, rows):
    mask = cr.split_by_group(rows.group, seed=0)
    for arm, critic in critics.items():
        assert critic.report["val_rmse_log10"] < critic.report["target_std_log10"] * 1.5
        assert critic.report["epochs_run"] <= 6 and critic.report["head_parameters"] > 0
        teacher_w = rows.lam if arm == "teacher_feature" else None
        pred = critic.predict_log10(rows.scalars, rows.goal, rows.d, rows.e, rows.t_right, teacher_w)
        assert pred.shape == rows.y.shape and np.isfinite(pred).all()
        assert np.isfinite(critic.report["val_spearman_within_state"])
    assert critics["direct"].estimator is None and critics["teacher_feature"].estimator is None
    assert critics["costate_critic"].estimator is not None
    assert critics["costate_critic"].report["estimator_parameters"] > 0 and critics["direct"].report["estimator_parameters"] == 0


def test_the_teacher_needs_the_exact_costate_and_the_direct_arm_ignores_it(critics, rows):
    with pytest.raises(ValueError):
        critics["teacher_feature"].predict_log10(rows.scalars, rows.goal, rows.d, rows.e, rows.t_right)
    a = critics["direct"].predict_log10(rows.scalars, rows.goal, rows.d, rows.e, rows.t_right)
    b = critics["direct"].predict_log10(rows.scalars, rows.goal, rows.d, rows.e, rows.t_right, rows.lam)
    assert np.array_equal(a, b)


def test_critic_json_round_trip(critics, rows):
    for arm, critic in critics.items():
        clone = cr.Critic.from_json(critic.to_json())
        teacher_w = rows.lam if arm == "teacher_feature" else None
        assert np.allclose(clone.predict_log10(rows.scalars, rows.goal, rows.d, rows.e, rows.t_right, teacher_w),
                           critic.predict_log10(rows.scalars, rows.goal, rows.d, rows.e, rows.t_right, teacher_w))


def test_direct_width_matching_brings_the_parameter_counts_together():
    n_direct, n_costate, est = hd.FEATURE_DIM + 3 * 64, hd.FEATURE_DIM + 3 * 64 + 3, (65, 16, 16, 64)
    for hidden in (8, 16, 32):
        h = cr.match_direct_hidden(n_direct, n_costate, hidden, est)
        target = cr.param_count(n_costate, hidden) + sum(i * o + o for i, o in zip(est[:-1], est[1:]))
        assert abs(cr.param_count(n_direct, h) - target) <= min(abs(cr.param_count(n_direct, h + 1) - target), abs(cr.param_count(n_direct, h - 1) - target))
        assert abs(cr.param_count(n_direct, h) - target) / target < 0.05


def test_within_state_spearman_is_one_for_a_monotone_prediction():
    y = np.array([1.0, 3.0, 2.0, 9.0, 8.0, 7.0])
    state = np.array([0, 0, 0, 1, 1, 1])
    assert cr.within_state_spearman(y * 2 + 1, y, state) == pytest.approx(1.0)
    assert cr.within_state_spearman(-y, y, state) == pytest.approx(-1.0)


# ---------------------------------------------------------------- prices and policies

def test_flops_match_a_count_from_the_weights(critics, instances):
    inst = instances[0]
    m = inst.m
    for critic in critics.values():
        expected = hd.flops_amortised_per_interval(inst, ()) + 2 * m + 2 * (critic.head.dims[0] - hd.FEATURE_DIM)
        expected += sum(2 * w.shape[0] * w.shape[1] + w.shape[1] for w in critic.head.mlp.weights) + hd.TANH_FLOPS * sum(w.shape[1] for w in critic.head.mlp.weights[:-1])
        if critic.features != "none":
            expected += 4 * m + 2 * m + 1 + 3 * hd.LOG_FLOPS + 6
        if critic.estimator is not None:
            est = critic.estimator.mlp
            expected += 2 * (1 + m) + sum(2 * w.shape[0] * w.shape[1] + w.shape[1] for w in est.weights) + hd.TANH_FLOPS * sum(w.shape[1] for w in est.weights[:-1])
        assert cr.flops_critic_per_interval(inst, critic) == expected
    assert cr.flops_critic_per_interval(inst, critics["costate_critic"]) > cr.flops_critic_per_interval(inst, critics["teacher_feature"]) > 0


def test_the_lookup_price_is_below_a_real_network_and_policies_score_every_interval(critics, domain, instances):
    inst = instances[0]
    state = domain.initial_state(inst)
    candidates = domain.legal_candidates(state)
    for arm, critic in critics.items():
        assert 0 < cr.lookup_price_scale(inst, critic) < 1
        scores = cr.critic_per_interval(critic, inst, state)
        assert scores.shape == (len(state.nodes) - 1,) and (scores > 0).all()
        policy = cr.critic_policy(critic, 0.9)
        picked = policy.select(domain, state, candidates, None)
        assert 0 < len(picked) <= len(candidates)
        cost = policy.decision_cost(domain, state, candidates)
        assert cost.compute == pytest.approx((len(state.nodes) - 1) * cr.flops_critic_per_interval(inst, critic) / hd.flops_step(inst))


# ---------------------------------------------------------------- statistics

def test_two_stage_bootstrap_brackets_the_mean_and_respects_the_structure():
    rng = np.random.default_rng(0)
    diff = -0.2 + 0.05 * rng.normal(size=(5, 20))
    out = cr.two_stage_bootstrap(diff, num_resamples=2000, seed=1)
    assert out["ci_low"] < out["estimate"] < out["ci_high"] < 0 and out["seeds_favouring_a"] == 5
    assert out["t_ci_low"] < out["estimate"] < out["t_ci_high"]
    assert cr.two_stage_bootstrap(diff, 2000, 1) == out
    assert cr.two_stage_bootstrap(np.zeros((5, 20)), 500)["ci_high"] == 0.0
    by_seed = np.repeat(np.array([[-1.0], [-0.5], [0.5], [1.0], [0.0]]), 20, axis=1)           # all variation between seeds
    wide = cr.two_stage_bootstrap(by_seed, 4000, 2)
    assert wide["ci_low"] < 0 < wide["ci_high"]


def test_rule_helpers():
    assert cr.two_adjacent([True, True, False]) and cr.two_adjacent([False, True, True]) and not cr.two_adjacent([True, False, True])
    rows_in = [{"ci_low": -0.03, "ci_high": 0.03}] * 3
    assert cr.within_margin(rows_in, np.log(1.05)) and not cr.within_margin(rows_in + [{"ci_low": -0.03, "ci_high": 0.06}], np.log(1.05))


# ---------------------------------------------------------------- running at a price, matrices and exit classes

def test_price_scales_and_the_critic_policy_run_at_both_prices(critics, domain, instances):
    inst = instances[0]
    critic = critics["costate_critic"]
    assert cr.price_scale(inst, critic, "real") == 1.0 and 0 < cr.price_scale(inst, critic, "lookup") < 1
    with pytest.raises(ValueError):
        cr.price_scale(inst, critic, "cheap")
    targets = lambda d, i: hd.gap_targets(d, i, (0.1,))  # noqa: E731
    cap_of = lambda i: 2.0 * i.n_fine  # noqa: E731
    real = cr.evaluate_critic(domain, instances[:2], critic, 0.9, "real", targets, cap_of, 12, "c|s0|real", methods=("loglog", "staircase"))
    lookup = cr.evaluate_critic(domain, instances[:2], critic, 0.9, "lookup", targets, cap_of, 12, "c|s0|lookup", methods=("loglog",))
    assert set(real["policy"]) == {"c|s0|real"} and set(real["method"]) == {"loglog", "staircase"} and len(real) == 4
    both = real[real["method"] == "loglog"].sort_values("instance")["compute"].to_numpy()
    cheap = lookup.sort_values("instance")["compute"].to_numpy()
    assert (cheap <= both + 1e-9).all()                                        # a cheaper price never needs more compute on the same trace
    tuned = cr.tune_critic_theta(domain, instances[:2], critic, (0.7, 0.9), "real", targets, cap_of, 12)
    assert set(tuned) == {0.7, 0.9} and all(v > 0 for v in tuned.values())


def test_log_compute_matrix_reads_the_requested_policies_in_order_and_caps_unreached_targets():
    import pandas as pd

    frame = pd.DataFrame([{"instance": i, "policy": p, "target": "gap10%", "method": "loglog", "compute": c}
                          for p, vals in (("a", (10.0, 20.0)), ("b", (30.0, float("inf")))) for i, c in zip(("x", "y"), vals)])
    out = cr.log_compute_matrix(frame, ["b", "a"], ["y", "x"], "gap10%", "loglog", cap=100.0)
    assert np.allclose(out, np.log([[100.0, 30.0], [20.0, 10.0]]))


def test_exit_class_follows_the_plans_table():
    base = {"R0": False, "RC1": False, "RC1L": False, "RC2L": False, "RC3": False, "RC3L": False, "RTL": False, "estimator_ok": True, "saturated": True}
    assert cr.exit_class({**base, "R0": True, "RC1": True, "RC3": True}).startswith("deployable")
    assert cr.exit_class({**base, "R0": True, "RC1": True}).startswith("effect not attributable")
    assert cr.exit_class({**base, "RC1L": True, "RC3L": True}).startswith("value conditional")
    assert cr.exit_class({**base, "RC1L": True}).startswith("effect not attributable")
    assert cr.exit_class({**base, "RTL": True}) == "teacher-only value"
    assert cr.exit_class({**base, "RTL": True, "estimator_ok": False, "RC1L": True, "RC3L": True}) == "teacher-only value"
    assert cr.exit_class({**base, "RC2L": True}) == "direct utility sufficient"
    assert cr.exit_class(base).startswith("inconclusive: precision")
    assert cr.exit_class({**base, "saturated": False, "RC1": True}).startswith("inconclusive: the direct baseline")
