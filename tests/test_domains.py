"""Domain-neutral allocation layer and the D4 reference domain (adaptive time stepping).

D4 instances are the in-repo analytic benchmark (docs/plans/cross-domain-plan.md §5); the
closed-form and finite-difference checks below are correctness tests, not results.
"""

import math

import numpy as np
import pytest

from adjointrwm.domains import (
    AdaptiveTimeSteppingDomain,
    Candidate,
    Cost,
    CostWeights,
    DomainSpec,
    LinearODEInstance,
    Policy,
    PolicyContext,
    Pulse,
    aurc,
    aurc_table,
    compute_level_summary,
    d4_policies,
    evaluate_at_compute,
    evaluate_policies,
    exhaustive_oracle_curve,
    fraction_of_oracle_advantage,
    objective_at_compute,
    one_step_oracle,
    opportunity_over_budgets,
    paired_aurc_difference,
    regret_curve,
    run_policy,
    sample_instances,
    transfer_summary,
)
from adjointrwm.domains.linear_ode import expm, reference_solution


def scalar_instance(a=-1.3, y0=0.7, amp=12.0, center=0.4, width=0.05, **kw):
    return LinearODEInstance("scalar", A=((a,),), y0=(y0,), goal=(1.0,),
                             pulses=(Pulse(center, width, amp, (1.0,)),), **kw)


# --- numerics ----------------------------------------------------------------------

def test_expm_matches_rotation_and_eigendecomposition():
    theta = 2.3
    rot = expm(np.array([[0.0, -theta], [theta, 0.0]]))
    np.testing.assert_allclose(rot, [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], atol=1e-12)
    rng = np.random.default_rng(0)
    s = rng.normal(size=(3, 3))
    s = (s + s.T) * 3.0
    w, v = np.linalg.eigh(s)
    np.testing.assert_allclose(expm(s), (v * np.exp(w)) @ v.T, rtol=1e-10, atol=1e-10)


def test_reference_solution_matches_closed_form():
    a, y0, amp, c, w = -1.3, 0.7, 12.0, 0.4, 0.05
    inst = scalar_instance(a, y0, amp, c, w, initial_intervals=8, max_depth=6)
    mu = c - a * w ** 2
    integral = w * math.sqrt(math.pi / 2) * (math.erf((1 - mu) / (math.sqrt(2) * w)) - math.erf(-mu / (math.sqrt(2) * w)))
    exact = math.exp(a) * y0 + amp * math.exp(a - a * c + a * a * w * w / 2) * integral
    assert reference_solution(inst)[-1, 0] == pytest.approx(exact, rel=1e-11)


# --- D4 identities -----------------------------------------------------------------

@pytest.fixture(scope="module")
def domain():
    return AdaptiveTimeSteppingDomain()


@pytest.fixture(scope="module")
def instances():
    return sample_instances(seed=11, count=3, initial_intervals=8, max_depth=6)


def test_error_representation_is_exact(domain, instances):
    for inst in instances:
        state = domain.initial_state(inst)
        for _ in range(3):  # also on non-uniform grids
            weighted = domain.weighted_local_errors(state, inst)
            assert weighted.sum() == pytest.approx(domain.signed_error(state, inst), rel=1e-10, abs=1e-13)
            state = domain.apply(state, domain.legal_candidates(state)[len(state.nodes) // 3])


def test_costate_matches_finite_differences(domain, instances):
    inst = instances[0]
    state = domain.apply(domain.initial_state(inst), domain.legal_candidates(domain.initial_state(inst))[2])
    lam = domain.costate(state)  # lam[k-1] = d(c^T y_N) / d y_k
    stepper, t = domain.stepper(inst), inst.time_of(state.nodes)
    c = np.asarray(inst.goal)

    def qoi_from(k, yk):
        y = yk.copy()
        for j in range(k, len(t) - 1):
            y = stepper.step(y, t[j], t[j + 1] - t[j])
        return c @ y

    for k in (1, len(t) // 2, len(t) - 1):
        for i in range(inst.m):
            eps = 1e-6
            e = np.zeros(inst.m)
            e[i] = eps
            fd = (qoi_from(k, state.y[k] + e) - qoi_from(k, state.y[k] - e)) / (2 * eps)
            assert fd == pytest.approx(lam[k - 1, i], rel=1e-6, abs=1e-9)


def test_crank_nicolson_is_second_order_and_estimates_local_error():
    smooth = scalar_instance(width=0.15, initial_intervals=64, max_depth=4)
    d = AdaptiveTimeSteppingDomain(objective_kind="abs_error")
    errors = []
    for depth in range(3):
        n0 = 64 * 2 ** depth
        inst = LinearODEInstance(**{**smooth.__dict__, "initial_intervals": n0, "max_depth": 4 - depth})
        errors.append(d.objective(d.initial_state(inst), inst))
    ratios = [errors[i] / errors[i + 1] for i in range(2)]
    assert all(3.5 < r < 4.5 for r in ratios), ratios
    state = d.initial_state(smooth)
    np.testing.assert_allclose(d.local_error_estimates(state), d.exact_local_errors(state, smooth), rtol=0.1, atol=1e-9)


def test_bound_objective_upper_bounds_the_qoi_error(domain, instances):
    for inst in instances:
        metrics = domain.native_metrics(domain.initial_state(inst), inst)
        assert metrics["goal_error_bound"] >= metrics["abs_qoi_error"] - 1e-15
        assert metrics["objective"] == metrics["goal_error_bound"]


def test_candidates_respect_max_depth(domain):
    inst = scalar_instance(initial_intervals=2, max_depth=1)
    state = domain.initial_state(inst)
    assert len(domain.legal_candidates(state)) == 2
    for candidate in list(domain.legal_candidates(state)):
        state = domain.apply(state, next(c for c in domain.legal_candidates(state) if c.target == candidate.target))
    assert domain.legal_candidates(state) == []
    assert state.nodes == (0, 1, 2, 3, 4)


# --- runner and privilege ----------------------------------------------------------

def test_deployable_policies_never_touch_the_exact_solution(instances):
    d = AdaptiveTimeSteppingDomain()
    inst = instances[1]
    state = d.initial_state(inst)
    candidates = d.legal_candidates(state)

    def forbidden(*args, **kwargs):
        raise AssertionError("deployable policy read the exact solution")

    d.reference = forbidden
    for policy in d4_policies():
        if policy.deployable:
            context = PolicyContext(observation=d.observation(inst), instance=None, rng=np.random.default_rng(0))
            assert policy.score(d, state, candidates, context).shape == (len(candidates),)


def test_runner_gives_instance_only_to_privileged_policies(domain, instances):
    seen = {}

    def spy(name):
        def score(d, state, candidates, context):
            seen[name] = context.instance is not None
            return np.zeros(len(candidates))
        return score

    run_policy(domain, instances[0], Policy("dep", spy("dep")), 1)
    run_policy(domain, instances[0], Policy("priv", spy("priv"), deployable=False), 1)
    assert seen == {"dep": False, "priv": True}


def test_stop_and_padding(domain, instances):
    never = Policy("never", lambda d, s, c, ctx: -np.ones(len(c)))
    trace = run_policy(domain, instances[0], never, 5, allow_stop=True)
    assert trace.stopped_at == 0 and trace.decisions == []
    assert len(trace.objective) == 6 and len(set(trace.objective)) == 1


def test_exhaustive_oracle_bounds_the_one_step_oracle(domain):
    inst = sample_instances(seed=3, count=1, initial_intervals=3, max_depth=2)[0]
    exhaustive = exhaustive_oracle_curve(domain, inst, 4, state_key=lambda s: s.nodes)
    greedy = run_policy(domain, inst, one_step_oracle(), 4).objective
    assert all(e <= g + 1e-15 for e, g in zip(exhaustive, greedy))
    assert exhaustive[0] == pytest.approx(greedy[0])


def test_refinement_is_charged_for_the_resolve_it_causes(domain, instances):
    state = domain.initial_state(instances[0])
    n = len(state.nodes) - 1
    assert [c.cost.compute for c in domain.legal_candidates(state)] == [float(n + 1 - j) for j in range(n)]
    assert all(c.cost.rate == 1.0 for c in domain.legal_candidates(state))
    # Refining interval j really changes every later step, but no earlier one.
    j = 3
    child = domain.apply(state, domain.legal_candidates(state)[j])
    assert np.array_equal(child.y[: j + 1], state.y[: j + 1])
    assert not np.allclose(child.y[-1], state.y[-1])


def test_decision_costs_are_charged(domain, instances):
    policies = {p.name: p for p in d4_policies()}
    trace = run_policy(domain, instances[0], policies["adjoint"], 2)
    n0 = instances[0].initial_intervals
    assert trace.decision_cost[1].compute == 3 * n0  # 2 estimate steps + 1 backward step per interval
    assert trace.decision_cost[2].compute == 3 * n0 + 3 * (n0 + 1)
    assert trace.action_cost[2].rate == 2.0 and trace.action_cost[2].compute > 2.0
    assert run_policy(domain, instances[0], policies["uniform"], 2).decision_cost[2].compute == 0


def test_compute_budget_is_a_hard_cap_and_cheap_policies_get_more_actions(domain, instances):
    policies = {p.name: p for p in d4_policies()}
    budget = 400.0
    traces = {name: run_policy(domain, instances[0], policies[name], 500, compute_budget=budget) for name in ("uniform", "adjoint")}
    for trace in traces.values():
        spent = [a.compute + d.compute for a, d in zip(trace.action_cost, trace.decision_cost)]
        assert max(spent) <= budget and trace.stopped_at is not None
    assert len(traces["uniform"].decisions) > 2 * len(traces["adjoint"].decisions)


def test_objective_at_compute_reads_the_last_affordable_step(domain, instances):
    trace = run_policy(domain, instances[0], {p.name: p for p in d4_policies()}["uniform"], 6)
    spent = [a.compute + d.compute for a, d in zip(trace.action_cost, trace.decision_cost)]
    objective, actions, used = objective_at_compute(trace, spent[3] + 0.5)
    assert (objective, actions, used) == (trace.objective[3], 3, spent[3])
    assert objective_at_compute(trace, 0.0) == (trace.objective[0], 0, 0.0)


def test_equal_compute_evaluation_skips_privileged_policies_and_summarises(domain, instances):
    policies = [p for p in d4_policies() if p.name in ("uniform", "random", "residual", "adjoint", "one_step_oracle")]
    frame = evaluate_at_compute(domain, instances, policies, levels=[150.0, 600.0], max_steps=200, random_draws=2)
    assert set(frame["policy"]) == {"uniform", "random", "residual", "adjoint"}
    assert (frame["compute_spent"] <= frame["compute_level"]).all()
    summary = compute_level_summary(frame, [("adjoint", "uniform")], num_resamples=200)
    assert set(summary) == {150.0, 600.0}
    assert summary[600.0]["mean_actions"]["uniform"] > summary[600.0]["mean_actions"]["adjoint"]
    assert summary[600.0]["differences"]["adjoint - uniform"]["num_episodes"] == len(instances)


# --- metrics -----------------------------------------------------------------------

def test_aurc_regret_fraction_and_opportunity():
    assert aurc([0, 1, 1]) == pytest.approx(1.5)
    assert aurc([2, 2], budgets=[0, 3]) == pytest.approx(6.0)
    np.testing.assert_allclose(regret_curve([3, 2], [1, 1]), [2, 1])
    assert fraction_of_oracle_advantage(1.0, 0.5, 0.0, delta_min=0.1) == pytest.approx(0.5)
    assert math.isnan(fraction_of_oracle_advantage(1.0, 0.99, 0.95, delta_min=0.1))
    assert opportunity_over_budgets([1, 1, 1], [1, 0.5, 0.5])["passes"]
    assert not opportunity_over_budgets([1, 1, 1], [1, 0.99, 0.99])["passes"]


def test_transfer_summary_guards_the_worst_domain():
    out = transfer_summary({"d1": 0.2, "d2": 0.1, "d3": -0.05}, m_domain=0.02)
    assert out["transfer_macro"] == pytest.approx(0.25 / 3)
    assert out["transfer_worst"] == pytest.approx(-0.05) and not out["worst_domain_guard_passes"]


def test_aurc_table_and_paired_difference(domain, instances):
    policies = [p for p in d4_policies() if p.name in ("uniform", "random", "adjoint", "one_step_oracle")]
    frame = evaluate_policies(domain, instances, policies, max_steps=6, random_draws=3)
    assert set(frame.loc[frame.policy == "random", "draw"]) == {0, 1, 2}
    table = aurc_table(frame, reference="one_step_oracle", fixed="uniform")
    oracle = table[table.policy == "one_step_oracle"]
    assert (oracle["aurc"] == 0).all()
    best = aurc_table(frame, fixed="uniform")  # default: regret to the best-known curve
    assert (best["aurc"] >= -1e-15).all() and "best_known" not in set(best["policy"])
    diff = paired_aurc_difference(table, "adjoint", "uniform", num_resamples=200)
    assert diff["num_episodes"] == len(instances)


def test_spec_validation_and_checksum():
    spec = AdaptiveTimeSteppingDomain.spec
    assert spec.checksum() == DomainSpec(**spec.to_dict()).checksum()
    with pytest.raises(ValueError):
        DomainSpec(**{**spec.to_dict(), "family": "astrology"})
    with pytest.raises(ValueError):
        Candidate("x", "teleport", 0)
    assert Cost(1, 2, 3, 4).weighted(CostWeights(1, 1, 1, 1)) == 10


# --- pass-based (batch) semantics and work-precision evaluation (D4-0b) --------------------

from adjointrwm.domains import (  # noqa: E402
    BatchPolicy,
    compute_to_target,
    evaluate_work_precision,
    marking_policy,
    run_batch_policy,
    sample_family,
    uniform_pass_policy,
    work_precision_summary,
)
from adjointrwm.domains.linear_ode import _select_dorfler  # noqa: E402


def test_batch_application_equals_sequential_application(domain, instances):
    inst = instances[0]
    state = domain.initial_state(inst)
    picked = [domain.legal_candidates(state)[j] for j in (1, 4, 5)]
    batch = domain.apply_batch(state, picked)
    sequential = state
    for candidate in picked:
        sequential = domain.apply(sequential, candidate)
    assert batch.nodes == sequential.nodes
    np.testing.assert_allclose(batch.y, sequential.y, rtol=1e-12, atol=1e-14)


def test_batch_cost_is_one_resolve_and_matches_single_refinement(domain, instances):
    state = domain.initial_state(instances[0])
    n = len(state.nodes) - 1
    candidates = domain.legal_candidates(state)
    assert domain.batch_cost(state, [candidates[3]]) == candidates[3].cost
    batch = domain.batch_cost(state, [candidates[2], candidates[6], candidates[5]])
    assert batch.rate == 3.0 and batch.compute == float(n + 3 - 2)
    assert batch.compute < sum(c.cost.compute for c in (candidates[2], candidates[5], candidates[6]))


def test_dorfler_selection_is_minimal_and_reaches_theta():
    scores = np.array([5.0, 1.0, 3.0, 0.5, 0.5])
    chosen = _select_dorfler(scores, 0.6)
    assert list(chosen) == [0, 2]  # 8 of 10 >= 6, and 5 alone is < 6
    assert scores[chosen].sum() >= 0.6 * scores.sum()
    assert scores[chosen[:-1]].sum() < 0.6 * scores.sum()
    assert list(_select_dorfler(np.zeros(4), 0.5)) == [0]
    assert len(_select_dorfler(scores, 1.0)) == len(scores)


def test_uniform_pass_refines_every_longest_interval(domain, instances):
    inst = instances[0]
    trace = run_batch_policy(domain, inst, uniform_pass_policy(), 3)
    assert [len(d) for d in trace.decisions] == [inst.initial_intervals, 2 * inst.initial_intervals, 4 * inst.initial_intervals]
    assert trace.decision_cost[-1].compute == 0  # no scoring


def test_marking_policies_use_the_same_scores_as_their_single_step_versions(domain, instances):
    inst = instances[1]
    state = domain.initial_state(inst)
    candidates = domain.legal_candidates(state)
    from adjointrwm.domains import PolicyContext

    context = PolicyContext(observation=inst, instance=None, rng=np.random.default_rng(0))
    single = {p.name: p for p in d4_policies()}
    for kind in ("residual", "goal_local", "adjoint"):
        top_single = int(np.argmax(single[kind].score(domain, state, candidates, context)))
        picked = marking_policy(kind, 0.3).select(domain, state, candidates, context)
        assert int(picked[0]) == top_single  # the first marked interval is the single-step choice (for adjoint: same |.| ordering)


def test_marking_charges_scoring_once_per_pass(domain, instances):
    inst = instances[0]
    n0 = inst.initial_intervals
    trace = run_batch_policy(domain, inst, marking_policy("adjoint", 0.5), 2)
    assert trace.decision_cost[1].compute == 3 * n0
    residual = run_batch_policy(domain, inst, marking_policy("residual", 0.5), 1)
    assert residual.decision_cost[1].compute == 2 * n0


def test_batch_runner_honours_budget_and_scoring_price(domain, instances):
    inst = instances[0]
    pol = marking_policy("adjoint", 0.5)
    capped = run_batch_policy(domain, inst, pol, 30, compute_budget=600.0)
    spent = [a.compute + d.compute for a, d in zip(capped.action_cost, capped.decision_cost)]
    assert max(spent) <= 600.0 and capped.stopped_at is not None
    cheap = run_batch_policy(domain, inst, pol, 30, compute_budget=600.0, decision_scale=0.0)
    assert len(cheap.decisions) > len(capped.decisions)
    with pytest.raises(ValueError):
        run_batch_policy(domain, inst, BatchPolicy("bad", lambda d, s, c, x: []), 1)


def test_compute_to_target_and_scoring_what_if(domain, instances):
    inst = instances[0]
    trace = run_batch_policy(domain, inst, marking_policy("adjoint", 0.5), 6)
    j = trace.objective
    target = 0.5 * (j[0] + j[3])  # reached at pass >= 1
    first = next(i for i, v in enumerate(j) if v <= target)
    full = compute_to_target(trace, target)
    assert full == trace.action_cost[first].compute + trace.decision_cost[first].compute
    assert compute_to_target(trace, target, decision_scale=0.0) == trace.action_cost[first].compute
    assert compute_to_target(trace, -1.0) == float("inf")


def test_work_precision_pipeline_and_censoring(domain, instances):
    policies = [uniform_pass_policy(), marking_policy("adjoint", 0.5), marking_policy("residual", 0.5)]

    def targets(d, inst):
        j0 = d.objective(d.initial_state(inst), inst)
        return {"gap50": 0.5 * j0, "gap5": 0.05 * j0}

    frame = evaluate_work_precision(domain, instances[:2], policies, targets, compute_cap=2000.0, max_steps=12, decision_scales=(1.0, 0.0))
    assert set(frame["decision_scale"]) == {0.0, 1.0} and set(frame["target"]) == {"gap50", "gap5"}
    uni = frame[(frame.policy == "uniform_pass")]
    assert (uni.groupby(["instance", "target"])["compute"].nunique() == 1).all()  # uniform has no scoring: price-independent
    adj = frame[(frame.policy == "mark_adjoint_0.5")]
    for (_, _), block in adj.groupby(["instance", "target"]):
        assert block.set_index("decision_scale").loc[0.0, "compute"] <= block.set_index("decision_scale").loc[1.0, "compute"]
    summary = work_precision_summary(frame, [("mark_adjoint_0.5", "uniform_pass")], compute_cap=2000.0, num_resamples=100)
    assert set(summary) == {"0", "1"} and set(summary["1"]) == {"gap50", "gap5"}
    entry = summary["1"]["gap50"]["differences"]["mark_adjoint_0.5 / uniform_pass"]
    assert entry["num_episodes"] == 2 and entry["ratio_of_geometric_means"] > 0


def test_families_are_named_and_the_smooth_one_is_the_d4_0_family():
    assert sample_instances(1001, 5) == sample_family("smooth", 1001, 5)
    sharp = sample_family("sharp", 7, 3)
    assert sharp[0].n_fine == 16 * 2 ** 11
    assert max(p.width for i in sharp for p in i.pulses) <= 0.004
    assert min(p.width for i in sample_family("smooth", 7, 3) for p in i.pulses) >= 0.01
