"""Correctness tests for the D1-0b forecasting objective (adjointrwm.domains.sensor_forecast).

The arrays below are tiny generated fixtures for testing code paths. They are not evidence about any dataset.
"""

import numpy as np
import pytest

from adjointrwm.domains import sensor as sn
from adjointrwm.domains import sensor_forecast as sf

C, L, H = 6, 12, 3


MIX = np.random.default_rng(123).normal(size=(2, C))


def stream(n_steps: int, seed: int) -> np.ndarray:
    """Two stationary AR(1) latents mixed into C channels (the same mixing for every seed, so a forecaster fitted on one stream applies to another)."""
    rng = np.random.default_rng(seed)
    latent = np.zeros((n_steps, 2))
    for t in range(1, n_steps):
        latent[t] = 0.95 * latent[t - 1] + rng.normal(size=2) * 0.2
    return latent @ MIX + rng.normal(size=(n_steps, C)) * 0.05


def setup(seed=0):
    train, test = stream(800, seed), stream(480, seed + 1)
    fc = sf.fit_forecaster(train, H, L)
    wins = sn.make_windows(fc.standardise(test), np.zeros(len(test)), L)
    return fc, wins, test


def test_forecaster_reproduces_the_ridge_solution_and_its_statistics():
    train = stream(800, 0)
    fc = sf.fit_forecaster(train, H, L)
    z = (train - train.mean(axis=0)) / np.maximum(train.std(axis=0), sf.FLOOR)
    x, y = z[:-H], z[H:]
    assert np.allclose(fc.weights, np.linalg.solve(x.T @ x + sf.RIDGE * np.eye(C), x.T @ y))
    assert np.allclose(fc.sensitivity, (fc.weights ** 2).sum(axis=1)) and np.allclose(fc.volatility, (z[L:] - z[:-L]).std(axis=0))
    assert (fc.horizon, fc.lag, fc.floor, fc.ridge) == (H, L, sf.FLOOR, sf.RIDGE)
    flat = train.copy()
    flat[:, 1] = 0.4
    assert np.isfinite(sf.fit_forecaster(flat, H, L).weights).all()                 # a constant channel stays finite


def test_native_loss_at_full_observation_equals_a_direct_computation_and_is_positive():
    fc, wins, _ = setup()
    full = sf.native_loss(fc, wins, wins.z)
    direct = []
    for w in range(len(wins.index)):
        errs = [np.mean((wins.z[w, t] @ fc.weights - wins.z[w, t + H]) ** 2) for t in range(1, L - H)]
        direct.append(np.mean(errs))
    assert np.allclose(full, direct) and (full > 0).all()
    assert np.allclose(sf.fidelity_loss(fc, wins, wins.z), 0.0)
    with pytest.raises(ValueError):
        sf.native_loss(fc, sn.make_windows(fc.standardise(stream(100, 2)), np.zeros(100), 4), sn.make_windows(fc.standardise(stream(100, 2)), np.zeros(100), 4).z)


def test_hold_loses_against_full_observation_on_average_and_fidelity_is_zero_at_full():
    fc, wins, _ = setup()
    hold = sf.native_loss(fc, wins, sn.observed(wins.z, np.zeros((len(wins.index), C), bool)))
    assert hold.mean() > sf.native_loss(fc, wins, wins.z).mean()
    assert (sf.fidelity_loss(fc, wins, sn.observed(wins.z, np.zeros((len(wins.index), C), bool))) > 0).all()


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_incremental_fidelity_greedy_equals_direct_recomputation(direction):
    fc, wins, _ = setup()
    j_by_size, order = sf.oracle_greedy(fc, wins, direction)
    assert j_by_size.shape == (len(wins.index), C + 1) and all(sorted(row) == list(range(C)) for row in order)
    for size in range(C + 1):
        mask = sn.oracle_mask(order, size, direction)
        assert (mask.sum(axis=1) == size).all()
        direct = sf.fidelity_loss(fc, wins, sn.observed(wins.z, mask))
        assert np.allclose(j_by_size[:, size], direct, atol=1e-10), (direction, size)
    assert np.allclose(j_by_size[:, C], 0.0, atol=1e-12) and (j_by_size[:, 0] > 0).all()


def test_oracle_best_masks_attain_the_running_minimum_and_never_exceed_either_direction():
    fc, wins, _ = setup()
    running, masks = sf.oracle_best(fc, wins)
    j_f, _ = sf.oracle_greedy(fc, wins, "forward")
    j_b, _ = sf.oracle_greedy(fc, wins, "backward")
    assert np.array_equal(running, np.minimum.accumulate(np.minimum(j_f, j_b), axis=1))
    for size in range(C + 1):
        assert (masks[size].sum(axis=1) <= size).all()
        direct = sf.fidelity_loss(fc, wins, sn.observed(wins.z, masks[size]))
        assert np.allclose(direct, running[:, size], atol=1e-10)
    with pytest.raises(ValueError):
        sf.oracle_greedy(fc, wins, "sideways")


def test_deployable_policies_ignore_everything_after_the_snapshot_and_match_their_definitions():
    fc, wins, _ = setup()
    perturbed_z = wins.z.copy()
    perturbed_z[:, 1:, :] = np.random.default_rng(9).normal(size=perturbed_z[:, 1:, :].shape) * 40
    perturbed = sn.Windows(perturbed_z, wins.prev1, wins.prev2, wins.labels, wins.index, wins.length)
    for name in sf.DEPLOYABLE_POLICIES:
        for k in (1, 3):
            assert np.array_equal(sf.policy_mask(name, k, fc, wins), sf.policy_mask(name, k, fc, perturbed)), name
    snap = wins.z[:, 0, :]
    assert np.allclose(sf.policy_scores("recent_change", fc, wins), np.abs(snap - wins.prev1))
    assert np.allclose(sf.policy_scores("recent_change_weighted", fc, wins), fc.sensitivity * (snap - wins.prev1) ** 2)
    assert np.allclose(sf.policy_scores("top_weighted_volatility", fc, wins)[0], fc.sensitivity * fc.volatility ** 2)
    assert np.allclose(sf.policy_scores("top_sensitivity", fc, wins)[0], fc.sensitivity)
    with pytest.raises(ValueError):
        sf.policy_scores("round_robin", fc, wins)
    assert (sf.policy_mask("round_robin", 2, fc, wins).sum(axis=1) == 2).all()


BUDGETS = (0, 1, 2, 4, C)


def result(seed=0, name="machine-1-1", with_oracle=True):
    train, test = stream(800, seed), stream(480, seed + 1)
    return sf.evaluate_machine(sf.fit_forecaster(train, H, L), test, name, L, BUDGETS, with_oracle)


def test_evaluate_machine_tables_and_the_reference_alias():
    r = result()
    assert r.n_windows == 38 and r.full.shape == (38,)
    for name in (*sf.DEPLOYABLE_POLICIES, sf.ORACLE, sf.REFERENCE):
        assert r.native[name].shape == (38, len(BUDGETS)) and r.fidelity[name].shape == (38, len(BUDGETS))
        assert np.allclose(r.native[name][:, -1], r.full)                               # budget C is full observation for everyone
        assert np.allclose(r.fidelity[name][:, -1], 0.0)
    assert np.array_equal(r.native[sf.REFERENCE], r.native[sf.ORACLE])
    hold = r.native["round_robin"][:, 0]
    for name in sf.DEPLOYABLE_POLICIES:
        assert np.allclose(r.native[name][:, 0], hold)
    assert r.fidelity[sf.ORACLE][:, 1:].mean() <= r.fidelity["round_robin"][:, 1:].mean()
    with pytest.raises(ValueError):
        sf.evaluate_machine(sf.fit_forecaster(stream(800, 0), H, L), stream(480, 1), "m", L, (0, C + 1))


def test_curves_start_at_r_end_at_zero_and_tolerate_a_machine_where_sensing_does_not_help():
    r = result()
    curves = sf.machine_curves(r)
    move = sf.endpoint_moves({"a": r})
    assert all(abs(c[0] - move["mean"]) < 1e-12 and abs(c[-1]) < 1e-12 for c in curves.values())
    assert move["mean"] > 0 and move["per_machine"]["a"] == pytest.approx((r.native["round_robin"][:, 0].mean() - r.full.mean()) / r.full.mean())
    negative = sf.MachineResult("m", L, H, (0, 1), native={"a": np.array([[0.5, 1.0], [0.6, 1.0]])}, full=np.array([1.0, 1.0]))     # hold below full
    assert sf.endpoint_moves({"m": negative})["mean"] < 0 and sf.machine_curves(negative)["a"][0] == pytest.approx(-0.45)
    with pytest.raises(ValueError):
        sf.machine_curves(sf.MachineResult("m", L, H, (1, 2), native={"a": np.ones((3, 2))}, full=np.ones(3)))
    with pytest.raises(ValueError):
        sf.machine_curves(sf.MachineResult("m", L, H, (0, 1), native={"a": np.ones((3, 2))}, full=np.zeros(3)))


def test_gate_summary_and_bootstrap_are_consistent_and_reproducible():
    results = {"machine-1-1": result(0, "machine-1-1"), "machine-1-5": result(1, "machine-1-5"), "machine-2-1": result(2, "machine-2-1")}
    areas = sf.overall_areas(results, C)
    assert all(np.isfinite(v) for v in areas.values()) and 0 < areas[sf.REFERENCE] <= 1.5
    chosen = sf.choose_best_fixed({m: r for m, r in results.items()}, C)
    assert chosen["selected"] in sf.FIXED_POLICIES and set(chosen["areas"]) == set(sf.FIXED_POLICIES)
    kwargs = dict(num_resamples=400, seed=1, bank=400, n_channels=C)
    a = sf.bootstrap_gate(results, chosen["selected"], **kwargs)
    b = sf.bootstrap_gate(results, chosen["selected"], **kwargs)
    assert a == b
    lo, hi = a["headroom_ci"]
    assert lo <= hi and np.isfinite([lo, hi]).all() and a["endpoint_moves_ci"][0] <= a["endpoint_moves_ci"][1]
    summary = sf.gate_summary(results, chosen["selected"], **kwargs)
    assert summary["G0_passes"] == bool(summary["endpoint_moves"]["mean"] >= sf.MIN_ENDPOINT_MOVE)
    assert summary["G1_vacuous"] == (not summary["G0_passes"])
    point = summary["relative_headroom"]
    assert lo - 0.3 <= point <= hi + 0.3
    assert "recent_change_weighted - recent_change" in a["paired_area_difference_ci"] and set(summary["retained"]) == set(sf.DYNAMIC_POLICIES)
    assert set(summary["per_machine_headroom"]) == set(results)
