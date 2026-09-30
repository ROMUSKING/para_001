"""D4-2: systems, cells, the FLOP ledger, the cheap indicators, the tabulated co-state and the amortised scorer."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from adjointrwm.domains import AdaptiveTimeSteppingDomain, Trace, Cost, evaluate_work_precision, run_batch_policy, uniform_pass_policy
from adjointrwm.domains import highdim as hd
from adjointrwm.domains.linear_ode import LinearODEInstance, ODEState, Pulse
from adjointrwm.domains.runner import INTERPOLATION_METHODS, compute_by_method, compute_to_target, interpolated_compute_to_target

DIMS = (hd.FEATURE_DIM, 32, 32, 1)


@pytest.fixture(scope="module")
def domain():
    return AdaptiveTimeSteppingDomain()


@pytest.fixture(scope="module")
def small():
    """Two tuning instances of the cheapest cell shape: m = 4, finest grid 2,048 intervals."""
    cell = dataclasses.replace(hd.CELLS["m4"], max_depth=7)
    return hd.cell_instances(cell, "tuning", 2)


# ---- systems, cells, families ---------------------------------------------------------------

def test_make_system_is_dense_stable_deterministic_and_validated():
    s = hd.make_system(8, 5)
    a = np.array(s.A)
    assert a.shape == (8, 8) and (np.abs(a) > 1e-9).mean() > 0.9                 # dense after the rotation
    assert np.linalg.eigvals(a).real.max() < 0.0                                # damped oscillators only
    assert abs(np.linalg.norm(s.goal) - 1.0) < 1e-12
    assert hd.make_system(8, 5) == s and hd.make_system(8, 6) != s
    for bad in (3, 0, 1):
        with pytest.raises(ValueError):
            hd.make_system(bad, 0)


def test_cells_vary_one_factor_at_a_time_from_base():
    base = hd.CELLS["base"]
    fields = ("m", "pulse_width", "amplitude_scale", "max_depth")
    for name, cell in hd.CELLS.items():
        if name == "base":
            continue
        changed = [f for f in fields if getattr(cell, f) != getattr(base, f)]
        assert len(changed) == 1, (name, changed)


def test_cell_instances_share_the_system_within_a_cell_family_and_hide_the_test_family():
    tuning = hd.cell_instances("base", "tuning", 3)
    train = hd.cell_instances("base", "train", 2)
    assert len(tuning) == 3 and tuning[0].A == train[0].A and tuning[0].goal == train[0].goal        # fixed A and goal per system
    assert tuning[0].pulses != tuning[1].pulses and tuning[0].y0 != tuning[1].y0                   # forcing and y0 vary
    assert tuning[0].name == "d4_2_base_tuning_000" and train[0].name == "d4_2_base_train_000"
    assert hd.cell_instances("m64", "tuning", 1)[0].A != tuning[0].A                               # another m, another system
    assert hd.cell_instances("base", "tuning", 3) == tuning                                        # frozen seeds
    for family in ("test", "anything"):
        with pytest.raises(ValueError, match="never generated"):
            hd.cell_instances("base", family, 1)
    assert "test" not in hd.FAMILY_SEEDS and set(hd.FAMILY_SEEDS.values()).isdisjoint({2002})


def test_cell_depth_sets_the_finest_grid_and_widths_follow_the_cell():
    assert hd.cell_instances("depth11", "validation", 1)[0].n_fine == 16 * 2 ** 11
    assert hd.cell_instances("depth7", "validation", 1)[0].n_fine == 16 * 2 ** 7
    wide = [p.width for i in hd.cell_instances("wide", "tuning", 5) for p in i.pulses]
    assert min(wide) >= 0.01 and max(wide) <= 0.04
    sharp = [p.width for i in hd.cell_instances("base", "tuning", 5) for p in i.pulses]
    assert max(sharp) <= 0.004


def test_gap_targets_are_fractions_of_the_gap(domain, small):
    inst = small[0]
    t = hd.gap_targets(domain, inst)
    floor, start = domain.finest_objective(inst), domain.objective(domain.initial_state(inst), inst)
    assert set(t) == {"gap10%", "gap3%", "gap1%"} and t["gap10%"] > t["gap3%"] > t["gap1%"] > floor
    assert t["gap10%"] == pytest.approx(floor + 0.1 * (start - floor))


# ---- FLOP ledger ----------------------------------------------------------------------------

def _tiny_instance(m=4, pulses=2):
    return LinearODEInstance("x", tuple(tuple(1.0 if i == j else 0.0 for j in range(m)) for i in range(m)), (0.0,) * m, (1.0,) + (0.0,) * (m - 1),
                             pulses=tuple(Pulse(0.5, 0.01, 1.0, (1.0,) + (0.0,) * (m - 1)) for _ in range(pulses)))


def test_flop_formulas_match_hand_counts():
    inst = _tiny_instance(m=4, pulses=2)
    assert hd.flops_forcing(4, 2) == 2 * (2 * 4 + 8) == 32
    assert hd.flops_step(inst) == 4 * 16 + 3 * 4 + 2 * 32 == 140
    assert hd.flops_indicator_per_interval(inst) == 2 * 32 + 6 * 4 + 2 * 4 + 1 == 97
    # (8 -> 32 -> 32 -> 1): products + biases + one tanh per hidden unit
    assert hd.flops_mlp_forward((8, 32, 32, 1)) == (2 * 8 * 32 + 32 + 10 * 32) + (2 * 32 * 32 + 32 + 10 * 32) + (2 * 32 * 1 + 1) == 3329
    assert hd.flops_costate_lookup_per_interval(inst) == 8
    assert hd.flops_difference_per_interval(inst) == 16 * 4 + 1
    assert hd.flops_cheap_per_interval(inst) == 97 + 65 + 1
    assert hd.flops_cheap_adjoint_per_interval(inst) == 97 + 65 + 1 + 16
    assert hd.flops_table_setup(inst) == inst.n_fine * 2 * 16
    assert hd.steps_from_flops(inst, 280) == 2.0


def test_scoring_price_falls_with_dimension_while_step_doubling_does_not():
    ratios = {}
    for name in ("m4", "base", "m64"):
        inst = hd.cell_instances(name, "tuning", 1)[0]
        report = hd.price_report(inst, DIMS)
        assert report["steps_per_interval"]["residual"] == 2.0 and report["steps_per_interval"]["adjoint"] == 3.0     # solver-step pricing does not depend on m
        ratios[name] = report["ratio_to_step_doubling"]
        r = report["ratio_to_step_doubling"]
        assert r["indicator"] < r["cheap"] < r["cheap_adjoint"] and r["indicator"] < r["amortised"]
    assert ratios["m4"]["amortised"] > ratios["base"]["amortised"] > ratios["m64"]["amortised"]
    assert ratios["m4"]["amortised"] > 1.0 > ratios["m64"]["amortised"]      # at m=4 solving is cheaper than scoring; at m=64 it is not
    assert ratios["m4"]["indicator"] > ratios["m64"]["indicator"]


# ---- indicator and tabulated co-state -------------------------------------------------------

class _LinearForcing(LinearODEInstance):
    direction: tuple = ()

    def forcing(self, t):
        t = np.asarray(t, dtype=float)
        return t[..., None] * np.ones(self.m)


class _QuadraticForcing(LinearODEInstance):
    def forcing(self, t):
        t = np.asarray(t, dtype=float)
        return (t ** 2)[..., None] * np.ones(self.m)


def _state_for(cls, domain, m=2):
    a = tuple(tuple(-1.0 if i == j else 0.0 for j in range(m)) for i in range(m))
    inst = cls("f", a, (0.0,) * m, (1.0,) + (0.0,) * (m - 1), pulses=(), initial_intervals=8, max_depth=3)
    return inst, domain.initial_state(inst)


def test_forcing_discrepancy_vanishes_for_linear_and_zero_forcing_and_is_minus_h3_over_6_for_quadratic(domain):
    zero, zero_state = _state_for(LinearODEInstance, domain)
    assert np.abs(hd.forcing_discrepancy(zero, zero_state)).max() == 0.0
    lin, lin_state = _state_for(_LinearForcing, domain)
    assert np.abs(hd.forcing_discrepancy(lin, lin_state)).max() < 1e-15
    quad, quad_state = _state_for(_QuadraticForcing, domain)
    h = np.diff(quad.time_of(quad_state.nodes))
    np.testing.assert_allclose(hd.forcing_discrepancy(quad, quad_state), -(h ** 3 / 6.0)[:, None] * np.ones(2), atol=1e-15)


def test_costate_table_ends_at_the_goal_matches_the_matrix_exponential_and_approaches_the_discrete_costate(domain, small):
    inst = small[0]
    table = hd.costate_table(inst)
    assert table.shape == (inst.n_fine + 1, inst.m) and hd.costate_table(inst) is table                       # cached
    np.testing.assert_allclose(table[-1], inst.goal)
    from adjointrwm.domains.linear_ode import expm

    np.testing.assert_allclose(table[0], expm(inst.matrix().T * inst.T) @ np.asarray(inst.goal), rtol=1e-6, atol=1e-9)
    gaps = []
    for level in (0, 3, 5):                                   # uniform meshes with 16 * 2^level intervals
        step = 2 ** (inst.max_depth - level)
        nodes = tuple(range(0, inst.n_fine + 1, step))
        state = ODEState(inst, nodes, domain.stepper(inst).solve(nodes))
        discrete = domain.costate(state)
        gaps.append(np.linalg.norm(discrete - table[np.asarray(nodes[1:])]) / np.linalg.norm(discrete))
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < 0.05     # the continuous weight is a coarse proxy only on coarse meshes


def test_indicator_policies_select_near_the_forcing_and_charge_flops(domain):
    a = ((-1.0, 0.0), (0.0, -2.0))
    inst = LinearODEInstance("p", a, (0.0, 0.0), (1.0, 1.0), pulses=(Pulse(0.5, 0.01, 30.0, (1.0, 1.0)),), initial_intervals=16, max_depth=8)
    state = domain.initial_state(inst)
    candidates = domain.legal_candidates(state)
    for make in (hd.indicator_policy, hd.indicator_adjoint_policy):
        policy = make(0.5)
        picked = policy.select(domain, state, candidates, None)
        centres = [np.mean(inst.time_of(np.array(candidates[i].target))) for i in picked]
        assert all(abs(c - 0.5) < 0.07 for c in centres), (policy.name, centres)                      # only the two intervals touching the pulse
        n = len(state.nodes) - 1
        expected = n * hd.flops_indicator_per_interval(inst) / hd.flops_step(inst)
        if make is hd.indicator_adjoint_policy:
            expected = n * (hd.flops_indicator_per_interval(inst) + hd.flops_costate_lookup_per_interval(inst)) / hd.flops_step(inst)
        assert policy.decision_cost(domain, state, candidates).compute == pytest.approx(expected)
    with pytest.raises(ValueError):
        hd.indicator_policy(0.0)


def test_indicator_policies_reduce_the_objective_over_passes(domain, small):
    inst = small[0]
    for policy in (hd.indicator_policy(0.9), hd.indicator_adjoint_policy(0.9)):
        trace = run_batch_policy(domain, inst, policy, 12)
        assert trace.objective[-1] < trace.objective[0] and trace.decision_cost[-1].compute > 0


# ---- the MLP and the scorer -----------------------------------------------------------------

def test_mlp_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    net = hd.TinyMLP((3, 5, 4, 1), rng)
    x, y = rng.normal(size=(7, 3)), rng.normal(size=7)
    loss, gw, gb = net.loss_and_grads(x, y)
    eps = 1e-6
    for arrays, grads in ((net.weights, gw), (net.biases, gb)):
        for a, g in zip(arrays, grads):
            flat = a.reshape(-1)
            for k in rng.choice(flat.size, size=min(4, flat.size), replace=False):
                old = flat[k]
                flat[k] = old + eps
                up = net.loss_and_grads(x, y)[0]
                flat[k] = old - eps
                down = net.loss_and_grads(x, y)[0]
                flat[k] = old
                assert (up - down) / (2 * eps) == pytest.approx(g.reshape(-1)[k], rel=1e-5, abs=1e-8)
    assert loss > 0


def test_train_scorer_learns_a_smooth_function_splits_by_instance_and_round_trips():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, size=(4000, hd.FEATURE_DIM))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2
    groups = np.repeat(np.arange(20), 200)
    model = hd.train_scorer(x, y, groups, seed=3, max_epochs=60, patience=8, lr=1e-2, batch_size=256)
    pred = model.predict_log10(x)
    assert np.sqrt(np.mean((pred - y) ** 2)) < 0.25 * y.std()
    assert model.report["n_val"] == 800 and model.report["n_train"] == 3200                        # 4 of 20 instances held out
    assert model.report["training_flops"] == 3 * hd.flops_mlp_forward(model.dims) * 3200 * model.report["epochs_run"] + hd.flops_mlp_forward(model.dims) * 800 * model.report["epochs_run"]
    again = hd.ScorerModel.from_json(model.to_json())
    np.testing.assert_allclose(again.predict_log10(x[:50]), pred[:50])
    same = hd.train_scorer(x, y, groups, seed=3, max_epochs=60, patience=8, lr=1e-2, batch_size=256)
    np.testing.assert_allclose(same.predict_log10(x[:50]), pred[:50])                              # deterministic


def test_collect_scorer_data_shapes_labels_and_per_state_cap(domain, small):
    policies = [uniform_pass_policy()]
    x, y, g = hd.collect_scorer_data(domain, small, policies, max_passes=6, per_state=20, seed=1)
    assert x.shape[1] == hd.FEATURE_DIM and len(x) == len(y) == len(g) and np.isfinite(x).all() and np.isfinite(y).all()
    assert set(g) == {0, 1} and len(x) <= 2 * 6 * 20
    inst = small[0]
    state = domain.initial_state(inst)
    labels = hd.scorer_labels(domain, inst, state)
    np.testing.assert_allclose(labels, np.log10(np.abs(domain.weighted_local_errors(state, inst)) + hd.LABEL_FLOOR))


def test_amortised_policy_runs_end_to_end_and_charges_features_plus_network(domain, small):
    inst = small[0]
    x, y, g = hd.collect_scorer_data(domain, small, [uniform_pass_policy()], max_passes=7, per_state=64, seed=0)
    model = hd.train_scorer(x, y, g, seed=0, max_epochs=5, patience=3)
    policy = hd.amortised_policy(model, 0.9)
    state = domain.initial_state(inst)
    candidates = domain.legal_candidates(state)
    picked = policy.select(domain, state, candidates, None)
    assert len(picked) >= 1 and len(set(int(i) for i in picked)) == len(picked)
    n = len(state.nodes) - 1
    assert policy.decision_cost(domain, state, candidates).compute == pytest.approx(n * hd.flops_amortised_per_interval(inst, model.dims) / hd.flops_step(inst))
    trace = run_batch_policy(domain, inst, policy, 10)
    assert trace.objective[-1] < trace.objective[0]


def test_tune_theta_returns_a_positive_cost_for_every_theta(domain, small):
    scores = hd.tune_theta(domain, small[:1], hd.indicator_policy, (0.7, 0.95), max_passes=15)
    assert set(scores) == {0.7, 0.95} and all(v > 0 and math.isfinite(v) for v in scores.values())


# ---- runner additions: log-log interpolation and the multi-method frame ---------------------

def _trace(objective, spent):
    costs = [Cost(compute=float(c)) for c in spent]
    return Trace("p", True, list(objective), costs, [Cost() for _ in spent], [], None)


def test_loglog_interpolation_is_exact_for_a_power_law_and_semilog_overstates_it():
    trace = _trace([4.0, 1.0], [100.0, 200.0])                       # error falls 4x while compute doubles
    target = 2.0                                                     # the geometric midpoint in objective
    assert compute_to_target(trace, target) == 200.0                 # staircase: the whole second pass
    assert interpolated_compute_to_target(trace, target, method="semilog") == pytest.approx(150.0)
    assert interpolated_compute_to_target(trace, target, method="loglog") == pytest.approx(100.0 * math.sqrt(2.0))
    assert interpolated_compute_to_target(trace, target, method="loglog") < interpolated_compute_to_target(trace, target, method="semilog")
    # a target already met at the first entry, an unreachable target, and an exact hit
    assert interpolated_compute_to_target(trace, 5.0, method="loglog") == 100.0
    assert interpolated_compute_to_target(trace, 0.1, method="loglog") == math.inf
    assert interpolated_compute_to_target(trace, 1.0, method="loglog") == pytest.approx(200.0)
    with pytest.raises(ValueError):
        interpolated_compute_to_target(trace, target, method="cubic")


def test_loglog_falls_back_to_semilog_when_the_earlier_step_cost_nothing():
    trace = _trace([4.0, 1.0], [0.0, 200.0])
    assert interpolated_compute_to_target(trace, 2.0, method="loglog") == pytest.approx(100.0)


def test_evaluate_work_precision_can_price_every_method_from_the_same_traces(domain, small):
    inst = small[0]
    frame = evaluate_work_precision(domain, [inst], [uniform_pass_policy(), hd.indicator_policy(0.9)], lambda d, i: hd.gap_targets(d, i),
                                    compute_cap=2.0 * inst.n_fine, max_steps=15, random_draws=1, methods=INTERPOLATION_METHODS)
    assert set(frame["method"]) == set(INTERPOLATION_METHODS) and len(frame) == 2 * 3 * 3
    wide = frame.pivot_table(index=["policy", "target"], columns="method", values="compute")
    assert (wide["staircase"] >= wide["semilog"] - 1e-9).all() and (wide["semilog"] >= wide["loglog"] - 1e-9).all()
    legacy = evaluate_work_precision(domain, [inst], [uniform_pass_policy()], lambda d, i: hd.gap_targets(d, i), compute_cap=2.0 * inst.n_fine, max_steps=15)
    assert "method" not in legacy.columns
    with pytest.raises(ValueError):
        evaluate_work_precision(domain, [inst], [uniform_pass_policy()], lambda d, i: hd.gap_targets(d, i), compute_cap=1.0, max_steps=2, methods=("nope",))
    assert compute_by_method(_trace([4.0, 1.0], [100.0, 200.0]), 2.0, 1.0, "staircase") == 200.0


def test_difference_estimate_is_zero_for_quadratics_and_h3_over_2_for_cubics(domain):
    inst = LinearODEInstance("d", ((-1.0, 0.0), (0.0, -1.0)), (0.0, 0.0), (1.0, 0.0), initial_intervals=8, max_depth=3)
    nodes = (0, 1, 2, 4, 5, 6, 8, 9, 12, 16, 24)                                   # a non-uniform mesh on the finest grid
    t = inst.time_of(nodes)
    quadratic = np.column_stack([1.0 + 2.0 * t + 3.0 * t ** 2, t ** 2])
    zero = hd.difference_estimate(inst, ODEState(inst, nodes, quadratic))
    assert np.abs(zero).max() < 1e-9
    cubic = np.column_stack([t ** 3, 2.0 * t ** 3])                                # third divided difference is the leading coefficient
    est = hd.difference_estimate(inst, ODEState(inst, nodes, cubic))
    h = np.diff(t)
    np.testing.assert_allclose(est, (h ** 3 / 2.0)[:, None] * np.array([1.0, 2.0]), rtol=1e-8)
    with pytest.raises(ValueError):
        hd.difference_estimate(inst, ODEState(inst, (0, 8, 16), np.zeros((3, 2))))


def test_cheap_arms_see_the_homogeneous_error_that_the_forcing_indicator_misses_and_charge_flops(domain, small):
    inst = small[0]
    state = domain.initial_state(inst)
    d, e = hd.forcing_discrepancy(inst, state), hd.difference_estimate(inst, state)
    assert np.linalg.norm(e, axis=1).max() > 0
    candidates = domain.legal_candidates(state)
    for make, flops in ((hd.cheap_policy, hd.flops_cheap_per_interval), (hd.cheap_adjoint_policy, hd.flops_cheap_adjoint_per_interval)):
        policy = make(0.9)
        picked = policy.select(domain, state, candidates, None)
        assert len(picked) >= 1 and len(set(int(i) for i in picked)) == len(picked)
        n = len(state.nodes) - 1
        assert policy.decision_cost(domain, state, candidates).compute == pytest.approx(n * flops(inst) / hd.flops_step(inst))
        trace = run_batch_policy(domain, inst, policy, 12)
        assert trace.objective[-1] < trace.objective[0]


# ---- frozen rules and bookkeeping -------------------------------------------------------------

def _block(pairs_high):
    """A fake work_precision_summary block: ``pairs_high[pair] = (ci_high per target)``."""
    out = {}
    for k, t in enumerate(hd.TARGETS):
        out[t] = {"differences": {pair: {"ci_high": highs[k]} for pair, highs in pairs_high.items()}}
    return out


def test_two_adjacent_needs_neighbouring_targets():
    assert hd.two_adjacent([True, True, False]) and hd.two_adjacent([False, True, True]) and hd.two_adjacent([True, True, True])
    assert not hd.two_adjacent([True, False, True]) and not hd.two_adjacent([False, False, True])


def test_cell_rules_combine_r1_and_r3_and_report_other_arms():
    good = {"amortised / uniform_pass": (0.1, -0.1, -0.2), "cheap_adjoint / cheap": (-0.1, -0.2, 0.3), "adjoint / uniform_pass": (0.2, 0.2, 0.2)}
    rules = hd.cell_rules(_block(good))
    assert rules["R1_amortised_beats_uniform"] and rules["R3_costate_weight_helps_cheap_estimator"] and rules["candidate_regime_for_d4_1"]
    assert rules["arm_beats_uniform_two_adjacent_targets"] == {"adjoint": False}
    only_r1 = hd.cell_rules(_block({**good, "cheap_adjoint / cheap": (0.1, -0.2, 0.3)}))
    assert only_r1["R1_amortised_beats_uniform"] and not only_r1["R3_costate_weight_helps_cheap_estimator"] and not only_r1["candidate_regime_for_d4_1"]
    neither = hd.cell_rules(_block({**good, "amortised / uniform_pass": (0.1, -0.1, 0.2)}))
    assert not neither["R1_amortised_beats_uniform"] and not neither["candidate_regime_for_d4_1"]


def _frame(rows):
    return pd.DataFrame(rows, columns=["instance", "policy", "target", "decision_scale", "compute"])


def test_with_free_reference_relabels_scale_zero_rows_of_one_policy():
    rows = []
    for scale in (1.0, 0.0):
        for policy, c in (("uniform_pass", 100.0), ("adjoint", 300.0 if scale == 1.0 else 50.0), ("amortised", 80.0)):
            rows.append(("i0", policy, "gap3%", scale, c))
    out = hd.with_free_reference(_frame(rows))
    assert set(out["decision_scale"]) == {1.0}
    assert dict(zip(out["policy"], out["compute"])) == {"uniform_pass": 100.0, "adjoint": 300.0, "amortised": 80.0, "adjoint_free": 50.0}


def test_retained_headroom_is_the_share_of_the_free_advantage_that_the_scorer_keeps():
    rows = []
    for i in range(4):
        for policy, c in (("uniform_pass", 1000.0), ("amortised", 100.0), ("adjoint_free", 10.0)):
            rows.append((f"i{i}", policy, "gap3%", 1.0, c))
    out = hd.retained_headroom(_frame(rows), cap=1e9, targets=("gap3%",))
    assert out["gap3%"] == pytest.approx(0.5)                                   # log(10) of log(100)
    worse = _frame([(f"i{i}", p, "gap3%", 1.0, c) for i in range(2) for p, c in (("uniform_pass", 100.0), ("amortised", 400.0), ("adjoint_free", 10.0))])
    assert hd.retained_headroom(worse, cap=1e9, targets=("gap3%",))["gap3%"] < 0
    none = _frame([(f"i{i}", p, "gap3%", 1.0, c) for i in range(2) for p, c in (("uniform_pass", 10.0), ("amortised", 20.0), ("adjoint_free", 50.0))])
    assert hd.retained_headroom(none, cap=1e9, targets=("gap3%",))["gap3%"] is None
    capped = _frame([(f"i{i}", p, "gap3%", 1.0, c) for i in range(2) for p, c in (("uniform_pass", np.inf), ("amortised", 100.0), ("adjoint_free", 10.0))])
    assert hd.retained_headroom(capped, cap=1000.0, targets=("gap3%",))["gap3%"] == pytest.approx((np.log(1000.0) - np.log(100.0)) / (np.log(1000.0) - np.log(10.0)))   # uniform capped at 1000


def test_break_even_instances():
    assert hd.break_even_instances(1000.0, [500.0, 700.0], [100.0, 300.0]) == pytest.approx(1000.0 / 400.0)
    assert hd.break_even_instances(1000.0, [100.0], [100.0]) is None and hd.break_even_instances(1000.0, [100.0], [200.0]) is None


# ---- the table script ---------------------------------------------------------------------------

def test_d4_2_table_script_reads_a_run_directory_in_report_order(tmp_path):
    import importlib.util
    import json as _json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("d4_2_tables", Path(__file__).resolve().parents[1] / "scripts" / "d4_2_tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def diff(high):
        return {"ci_low": high - 0.4, "ci_high": high, "estimate": high - 0.2, "ratio_of_geometric_means": math.exp(high - 0.2)}

    pairs = ["amortised / uniform_pass", "cheap / uniform_pass", "cheap_adjoint / uniform_pass", "adjoint / uniform_pass", "residual / uniform_pass",
             "goal_local / uniform_pass", "indicator / uniform_pass", "cheap_adjoint / cheap", "amortised / cheap", "amortised / cheap_adjoint",
             "adjoint_free / amortised", "adjoint_free / uniform_pass", "adjoint / amortised"]
    arms = ("uniform_pass", "amortised", "cheap", "cheap_adjoint", "adjoint", "adjoint_free", "residual", "goal_local", "indicator")

    def block(r1_high, r3_high):
        out = {}
        for t in hd.TARGETS:
            highs = {p: 0.5 for p in pairs}
            highs["amortised / uniform_pass"], highs["cheap_adjoint / cheap"] = r1_high, r3_high
            out[t] = {"median_compute": {a: 100.0 for a in arms}, "fraction_reached": {a: 1.0 for a in arms}, "differences": {p: diff(highs[p]) for p in pairs}}
        return out

    summary = {"summary": {c: {m: block(-0.1, -0.1 if c == "b" else 0.1) for m in ("loglog", "semilog", "staircase")} for c in ("b", "a")},
               "methods": ["loglog", "semilog", "staircase"], "primary_method": "loglog", "pairs": [], "selected_theta": {}, "selected_hidden": {}}
    price = {"flops_per_step": 4768, "ratio_to_step_doubling": {"amortised": 0.49, "cheap": 0.14, "cheap_adjoint": 0.15, "indicator": 0.09, "residual": 1.0, "goal_local": 1.0, "adjoint": 1.5},
             "steps_per_interval": {"amortised": 0.98, "cheap": 0.28, "cheap_adjoint": 0.31, "indicator": 0.17, "residual": 2.0, "goal_local": 2.0, "adjoint": 3.0}, "table_setup_steps": 3518.7}
    cells = [{"cell": "b", "m": 32, "hidden": 16, "R1": True, "R3": True, "candidate_regime_for_d4_1": True},
             {"cell": "a", "m": 4, "hidden": 8, "R1": True, "R3": False, "candidate_regime_for_d4_1": False}]
    report = {"cells": cells, "price_reports": {"a": price, "b": price}, "selected_hidden": {"a": 8, "b": 16},
              "one_off_costs_and_break_even": {c: {"amortised_training_steps": 1e6, "amortised_break_even_instances": None, "costate_table_setup_steps": 3518.7,
                                                   "cheap_adjoint_break_even_instances": 6.5} for c in ("a", "b")},
              "retained_headroom": {c: {t: 0.5 if t != "gap1%" else None for t in hd.TARGETS} for c in ("a", "b")}}
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "artifacts/validation_summary.json").write_text(_json.dumps(summary))
    (tmp_path / "reports/acceptance_report.json").write_text(_json.dumps(report))
    (tmp_path / "artifacts/theta_tuning.csv").write_text("cell,arm,theta_0.85,theta_0.95,selected,selected_at_grid_edge\nb,cheap,10,20,0.85,True\n")
    (tmp_path / "artifacts/hidden_size_selection.csv").write_text(
        "cell,hidden,tuning_geometric_mean_compute,val_rmse_log10,epochs_run,training_flops,n_train\nb,16,300,0.45,40,1,1\na,8,200,0.5,30,1,1\n")
    data = module.load(tmp_path)
    prices = module.price_table(data["report"]).splitlines()
    assert prices[2].startswith("| b | 32 |") and prices[3].startswith("| a | 4 |")                    # report order, not alphabetical
    assert "0.490" in prices[2] and "3519" in prices[2]
    assert "8: 200" not in module.scorer_table(data["hidden"], data["report"]).splitlines()[2]           # cell b lists its own sizes only
    table = module.compute_table(data["summary"], "loglog", module.cell_order(data["report"]))
    assert len(table.splitlines()) == 2 + 2 * 3 and "| b | 10% |" in table
    pairs = module.pairs_table(data["summary"], "loglog", module.cell_order(data["report"]))
    assert len(pairs.splitlines()) == 2 + 2 * 3 and "| a | 1% |" in pairs
    rules = module.rules_table(data).splitlines()
    assert rules[2].startswith("| b | True | True | True |") and rules[3].startswith("| a | True | False | False |")
    assert "n/a" in module.headroom_table(data["report"]) and "6.5" in module.headroom_table(data["report"])
    report["cells"][1]["R1"] = False                                                                       # a report that disagrees with the rules is caught
    with pytest.raises(AssertionError, match="does not reproduce"):
        module.rules_table({**data, "report": report})
