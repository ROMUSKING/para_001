"""Correctness tests for the D4-3 additions to adjointrwm.domains.highdim (a goal per instance)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from adjointrwm.domains import highdim as hd


@pytest.fixture(scope="module")
def small_cell():
    return dataclasses.replace(hd.CELLS["m4"], max_depth=7)


def test_varying_goal_instances_keep_everything_but_the_goal_and_the_name(small_cell):
    fixed = hd.cell_instances(small_cell, "tuning", 4)
    varying = hd.varying_goal_instances(small_cell, "tuning", 4)
    assert [v.name for v in varying] == [f"d4_3_m4_tuning_{i:03d}" for i in range(4)]
    for f, v in zip(fixed, varying):
        assert (f.A, f.y0, f.pulses, f.max_depth, f.initial_intervals) == (v.A, v.y0, v.pulses, v.max_depth, v.initial_intervals)
        assert f.goal != v.goal
        assert np.linalg.norm(v.goal) == pytest.approx(1.0)
    goals = np.array([v.goal for v in varying])
    assert len({v.goal for v in varying}) == 4 and np.abs(goals @ goals.T - np.eye(4)).max() < 0.99     # distinct, not collinear
    assert hd.varying_goal_instances(small_cell, "tuning", 4) == varying                                   # frozen seeds


def test_goals_differ_across_families_and_cells_and_the_test_family_is_refused(small_cell):
    tuning = hd.varying_goal_instances(small_cell, "tuning", 2)
    validation = hd.varying_goal_instances(small_cell, "validation", 2)
    assert tuning[0].goal != validation[0].goal
    base = hd.varying_goal_instances("base", "tuning", 1)[0]
    assert len(base.goal) == 32 and base.goal != hd.cell_instances("base", "tuning", 1)[0].goal
    for family in ("test", "anything"):
        with pytest.raises(ValueError, match="never generated"):
            hd.varying_goal_instances("base", family, 1)
    assert hd.GOAL_STREAM_OFFSET not in hd.FAMILY_SEEDS.values() and "test" not in hd.FAMILY_SEEDS


def test_the_tabulated_costate_is_linear_in_the_goal_and_not_shared_between_goals(small_cell):
    a, b = hd.varying_goal_instances(small_cell, "tuning", 2)
    mixed = dataclasses.replace(a, goal=tuple(0.3 * np.array(a.goal) - 1.7 * np.array(b.goal)))
    table_a, table_b, table_mixed = hd.costate_table(a), hd.costate_table(b), hd.costate_table(mixed)
    assert np.allclose(table_a[-1], a.goal) and not np.allclose(table_a, table_b)
    assert np.allclose(table_mixed, 0.3 * table_a - 1.7 * table_b, atol=1e-10)


def test_setup_steps_equal_the_flop_formula_over_the_step_price(small_cell):
    inst = hd.varying_goal_instances(small_cell, "tuning", 1)[0]
    assert hd.setup_steps(inst) == pytest.approx(inst.n_fine * 2 * inst.m ** 2 / hd.flops_step(inst))
    assert hd.setup_steps(inst) == pytest.approx(hd.price_report(inst, (hd.FEATURE_DIM, 8, 8, 1))["table_setup_steps"])


def _frame(rows):
    return pd.DataFrame(rows, columns=["instance", "policy", "target", "decision_scale", "compute"])


def test_with_setup_charged_adds_each_instances_setup_to_one_policy_only(small_cell):
    insts = hd.varying_goal_instances(small_cell, "tuning", 2)
    rows = []
    for i, inst in enumerate(insts):
        for policy, c in (("uniform_pass", 100.0), ("cheap", 60.0 + i), ("cheap_adjoint", 50.0 + i)):
            rows.append((inst.name, policy, "gap3%", 1.0, c))
        rows.append((inst.name, "cheap_adjoint", "gap1%", 1.0, float("inf")))          # an unreached target stays unreached
        rows.append((inst.name, "cheap_adjoint", "gap3%", 0.0, 1.0))                   # scale-0 rows are dropped
    out = hd.with_setup_charged(_frame(rows), insts)
    assert set(out["decision_scale"]) == {1.0}
    charged = out[out["policy"] == "cheap_adjoint_setup"].set_index(["instance", "target"])["compute"]
    for i, inst in enumerate(insts):
        assert charged[(inst.name, "gap3%")] == pytest.approx(50.0 + i + hd.setup_steps(inst))
        assert np.isinf(charged[(inst.name, "gap1%")])
    same = out[out["policy"] == "cheap_adjoint"].set_index(["instance", "target"])["compute"]
    assert same[(insts[0].name, "gap3%")] == 50.0 and set(out["policy"]) == {"uniform_pass", "cheap", "cheap_adjoint", "cheap_adjoint_setup"}


def _block(pairs_high):
    return {t: {"differences": {pair: {"ci_high": highs[k]} for pair, highs in pairs_high.items()}} for k, t in enumerate(hd.TARGETS)}


def test_varying_goal_rules_combine_r3v_and_r4v_and_report_the_priced_variant():
    good = {"cheap_adjoint / cheap": (-0.1, -0.2, 0.3), "cheap_adjoint / uniform_pass": (0.1, -0.1, -0.2), "cheap / uniform_pass": (0.2, 0.2, 0.2),
            "cheap_adjoint_setup / uniform_pass": (0.5, 0.5, 0.5), "cheap_adjoint_setup / cheap": (0.4, 0.4, 0.4)}
    rules = hd.cell_rules_varying_goal(_block(good))
    assert rules["R3v_costate_weight_helps_cheap_estimator"] and rules["R4v_goal_aware_cheap_arm_beats_uniform"] and rules["candidate_regime_for_d4_1"]
    assert rules["setup_charged_cheap_adjoint_beats_cheap"] is False
    assert rules["arm_beats_uniform_two_adjacent_targets"] == {"cheap": False, "cheap_adjoint": True, "cheap_adjoint_setup": False}
    weight_only = hd.cell_rules_varying_goal(_block({**good, "cheap_adjoint / uniform_pass": (0.1, -0.1, 0.2)}))
    assert weight_only["R3v_costate_weight_helps_cheap_estimator"] and not weight_only["candidate_regime_for_d4_1"]
    neither = hd.cell_rules_varying_goal(_block({"cheap_adjoint / cheap": (0.1, 0.1, 0.1), "cheap_adjoint / uniform_pass": (0.1, 0.1, 0.1)}))
    assert not neither["candidate_regime_for_d4_1"] and neither["setup_charged_cheap_adjoint_beats_cheap"] is None
