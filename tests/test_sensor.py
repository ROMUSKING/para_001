"""Correctness tests for the D1 sensor-stream domain (adjointrwm.domains.sensor).

The arrays below are tiny generated fixtures for testing code paths. They are not evidence about any dataset.
"""

import numpy as np
import pytest

from adjointrwm.domains import sensor as sn

C, L = 6, 10


def stream(n_steps: int, seed: int, bursts: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = np.cumsum(rng.normal(size=(n_steps, 2)) * 0.05, axis=0)
    mix = rng.normal(size=(2, C))
    x = latent @ mix + rng.normal(size=(n_steps, C)) * 0.05
    if bursts:
        for t0 in range(37, n_steps - 15, 47):
            x[t0: t0 + 6, rng.integers(0, C, size=2)] += rng.normal(size=2) * 2.0
    return x


def machine(seed: int = 0, name: str = "machine-1-1") -> sn.MachineData:
    test = stream(400, seed + 1, bursts=True)
    labels = np.zeros(len(test))
    labels[100:140] = 1
    return sn.MachineData(name, stream(600, seed), test, labels)


def detector(data=None, floor=0.05, ridge=1e-2, lag=L):
    data = data or machine()
    return sn.fit_detector(data.train, lag, floor, ridge)


# ---- splits and files ----------------------------------------------------------------------------------------------

def test_split_is_a_partition_by_machine():
    ids = sn.machine_ids()
    assert len(ids) == 28 and len(set(ids)) == 28
    parts = {s: set(sn.split_machines(s)) for s in sn.SPLITS}
    assert [len(parts[s]) for s in sn.SPLITS] == [7, 7, 14]
    assert set.union(*parts.values()) == set(ids)
    assert sum(len(p) for p in parts.values()) == 28
    assert sn.split_of("machine-1-1") == "tuning" and sn.split_of("machine-1-2") == "validation"
    assert sn.split_of("machine-1-3") == "test" and sn.split_of("machine-1-4") == "test"
    with pytest.raises(ValueError):
        sn.split_of("machine-9-9")


def test_test_machines_are_never_downloaded_or_loaded(tmp_path):
    calls = []
    test_machine = sn.split_machines("test")[0]
    with pytest.raises(PermissionError):
        sn.download_machine(tmp_path, test_machine, fetch=lambda url: calls.append(url) or b"")
    assert calls == [] and not any(tmp_path.iterdir())
    with pytest.raises(PermissionError):
        sn.load_machine(tmp_path, test_machine)
    with pytest.raises(PermissionError):
        sn.download_machine(tmp_path, "machine-1-2", allowed=("tuning",), fetch=lambda url: b"")


def test_download_writes_files_and_a_manifest_and_caches(tmp_path):
    calls = []

    def fake(url):
        calls.append(url)
        return b"1,2\n3,4\n"

    rows = sn.download_machine(tmp_path, "machine-1-1", fetch=fake)
    assert [r["kind"] for r in rows] == list(sn.FILE_KINDS) and len(calls) == 3
    assert all(r["sha256"] == __import__("hashlib").sha256(b"1,2\n3,4\n").hexdigest() and r["bytes"] == 8 for r in rows)
    assert all(r["split"] == "tuning" and r["url"].startswith(sn.SMD_BASE_URL) for r in rows)
    assert (tmp_path / "train" / "machine-1-1.txt").read_bytes() == b"1,2\n3,4\n"
    again = sn.download_machine(tmp_path, "machine-1-1", fetch=fake)
    assert len(calls) == 3 and all(r["retrieved_utc"] == "cached" for r in again)
    assert [r["sha256"] for r in again] == [r["sha256"] for r in rows]


def test_load_machine_checks_shapes(tmp_path):
    for kind, text in (("train", "\n".join(",".join("0.5" for _ in range(sn.NUM_CHANNELS)) for _ in range(5))),
                       ("test", "\n".join(",".join("0.1" for _ in range(sn.NUM_CHANNELS)) for _ in range(4))),
                       ("test_label", "0\n1\n0\n0")):
        path = sn.machine_path(tmp_path, kind, "machine-1-1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    data = sn.load_machine(tmp_path, "machine-1-1")
    assert data.train.shape == (5, 38) and data.test.shape == (4, 38) and data.labels.tolist() == [0, 1, 0, 0]
    sn.machine_path(tmp_path, "test_label", "machine-1-1").write_text("0\n1\n")
    with pytest.raises(ValueError):
        sn.load_machine(tmp_path, "machine-1-1")


# ---- the detector -------------------------------------------------------------------------------------------------

def test_detector_precision_is_symmetric_positive_definite_and_threshold_is_the_quantile():
    data = machine()
    det = detector(data)
    assert np.allclose(det.precision, det.precision.T)
    assert np.linalg.eigvalsh(det.precision).min() > 0
    scores = det.score(det.standardise(data.train))
    assert det.threshold == pytest.approx(np.quantile(scores, 0.995))
    assert det.volatility.shape == (C,) and (det.volatility >= 0).all()
    assert det.score(np.zeros((3, C))).tolist() == [0.0, 0.0, 0.0]


def test_floor_keeps_a_constant_channel_finite():
    data = machine()
    train = data.train.copy()
    train[:, 2] = 0.3
    det = sn.fit_detector(train, L, floor=0.05, ridge=1e-2)
    assert np.isfinite(det.precision).all() and det.scale[2] == pytest.approx(0.05)
    moved = np.zeros((1, C))
    moved[0, 2] = 0.05 / 0.05      # a move of one floor unit
    assert np.isfinite(det.score(moved)).all()


def test_select_detector_takes_the_best_f1_and_breaks_ties_toward_smaller_ridge():
    machines = [machine(0, "machine-1-1"), machine(1, "machine-1-5")]
    out = sn.select_detector(machines, L, floors=(0.01, 0.05), ridges=(1e-6, 1e-2, 1e-1))
    assert len(out["grid"]) == 6
    best = max(out["grid"], key=lambda r: r["mean_f1"])
    assert out["selected"]["floor"] in (0.01, 0.05)
    assert {"floor": best["floor"], "ridge": best["ridge"]} == out["selected"] or \
        sum(r["mean_f1"] == best["mean_f1"] for r in out["grid"]) > 1
    assert sn.f1_from_counts(0, 0, 0) == 0.0 and sn.f1_from_counts(2, 1, 1) == pytest.approx(2 * 2 / (4 + 1 + 1))


# ---- windows and the stream under an allocation -----------------------------------------------------------------------

def test_windows_are_tiled_and_snapshots_line_up():
    t = np.repeat(np.arange(105.0)[:, None], C, axis=1)           # channel value = minute
    wins = sn.make_windows(t, np.zeros(105), L, skip=2)
    assert wins.z.shape == (8, L, C) and wins.index.tolist() == list(range(2, 10))      # 10 full windows, tail dropped
    assert (wins.z[:, 0, 0] == wins.index * L).all()
    assert (wins.prev1[:, 0] == (wins.index - 1) * L).all() and (wins.prev2[:, 0] == (wins.index - 2) * L).all()
    with pytest.raises(ValueError):
        sn.make_windows(t, np.zeros(105), L, skip=1)
    with pytest.raises(ValueError):
        sn.make_windows(t[:25], np.zeros(25), L)


def test_observed_holds_unopened_channels_at_the_snapshot():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(4, L, C))
    mask = np.zeros((4, C), bool)
    mask[:, [1, 4]] = True
    obs = sn.observed(z, mask)
    assert np.array_equal(obs[:, :, [1, 4]], z[:, :, [1, 4]])
    for c in (0, 2, 3, 5):
        assert (obs[:, :, c] == z[:, :1, c]).all()


def test_full_observation_has_zero_loss_and_hold_has_positive_loss():
    data = machine()
    det = detector(data)
    wins = sn.make_windows(det.standardise(data.test), data.labels, L)
    s_full = det.score(wins.z)
    for kind in sn.OBJECTIVES:
        full = sn.objective_value(det.score(sn.observed(wins.z, np.ones((len(wins.index), C), bool))), s_full, det.threshold, kind)
        hold = sn.objective_value(det.score(sn.observed(wins.z, np.zeros((len(wins.index), C), bool))), s_full, det.threshold, kind)
        assert np.all(full == 0) and hold.mean() > 0


# ---- policies ------------------------------------------------------------------------------------------------------

def test_top_k_mask_breaks_ties_toward_the_lower_index():
    scores = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 3.0, 3.0, 1.0]])
    assert sn.top_k_mask(scores, 2).tolist() == [[True, True, False, False], [False, True, True, False]]
    assert sn.top_k_mask(scores, 0).sum() == 0 and sn.top_k_mask(scores, 4).all()


def test_round_robin_opens_k_distinct_channels_and_covers_all_over_time():
    det = detector()
    wins = sn.make_windows(det.standardise(machine().test), machine().labels, L)
    for k in (0, 1, 2, 4, C):
        mask = sn.policy_mask("round_robin", k, det, wins)
        assert (mask.sum(axis=1) == k).all()
    two = sn.policy_mask("round_robin", 3, det, wins)[:2]
    assert two.any(axis=0).all()                                 # two consecutive windows of 3 cover all 6 channels


def test_deployable_policies_ignore_everything_after_the_snapshot():
    det = detector()
    wins = sn.make_windows(det.standardise(machine().test), machine().labels, L)
    rng = np.random.default_rng(9)
    z2 = wins.z.copy()
    z2[:, 1:, :] = rng.normal(size=z2[:, 1:, :].shape) * 50
    perturbed = sn.Windows(z2, wins.prev1, wins.prev2, wins.labels, wins.index, wins.length)
    for name in sn.DEPLOYABLE_POLICIES:
        for k in (1, 3):
            assert np.array_equal(sn.policy_mask(name, k, det, wins), sn.policy_mask(name, k, det, perturbed)), name


def test_policy_scores_match_their_definitions():
    det = detector()
    wins = sn.make_windows(det.standardise(machine().test), machine().labels, L)
    snap = wins.z[:, 0, :]
    assert np.allclose(sn.policy_scores("recent_change", det, wins), np.abs(snap - wins.prev1))
    assert np.allclose(sn.policy_scores("sensitivity", det, wins), np.abs(snap @ det.precision))
    assert np.allclose(sn.policy_scores("sensitivity_x_volatility", det, wins), np.abs(snap @ det.precision) * det.volatility)
    assert np.allclose(sn.policy_scores("top_weighted_volatility", det, wins)[0], det.volatility ** 2 * np.diag(det.precision))
    with pytest.raises(ValueError):
        sn.policy_scores("round_robin", det, wins)


# ---- the greedy oracle ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("kind", sn.OBJECTIVES)
def test_incremental_greedy_update_equals_direct_recomputation(kind, direction):
    det = detector()
    wins = sn.make_windows(det.standardise(machine().test), machine().labels, L)
    j_by_size, order = sn.oracle_greedy(det, wins, kind, direction)
    s_full = det.score(wins.z)
    assert j_by_size.shape == (len(wins.index), C + 1) and order.shape == (len(wins.index), C)
    assert all(sorted(row) == list(range(C)) for row in order)
    for size in range(C + 1):
        mask = sn.oracle_mask(order, size, direction)
        assert (mask.sum(axis=1) == size).all()
        direct = sn.objective_value(det.score(sn.observed(wins.z, mask)), s_full, det.threshold, kind)
        assert np.allclose(j_by_size[:, size], direct, atol=1e-8), (kind, direction, size)
    assert np.allclose(j_by_size[:, C], 0.0, atol=1e-12)         # the incremental update drifts by rounding only
    assert np.all(j_by_size[:, 0] >= 0) and j_by_size[:, 0].max() > 0      # hold can already agree with the full stream for some windows


def test_oracle_best_is_the_elementwise_minimum_and_its_masks_attain_it():
    det = detector()
    wins = sn.make_windows(det.standardise(machine().test), machine().labels, L)
    s_full = det.score(wins.z)
    j_f, _ = sn.oracle_greedy(det, wins, "score", "forward")
    j_b, _ = sn.oracle_greedy(det, wins, "score", "backward")
    best, mask_for = sn.oracle_best(det, wins, "score")
    assert np.array_equal(best, np.minimum(j_f, j_b))
    for size in range(C + 1):
        mask = mask_for(size)
        assert (mask.sum(axis=1) == size).all()
        direct = sn.objective_value(det.score(sn.observed(wins.z, mask)), s_full, det.threshold, "score")
        assert np.allclose(best[:, size], direct, atol=1e-8)
    with pytest.raises(ValueError):
        sn.oracle_greedy(det, wins, "score", "sideways")


def test_first_greedy_pick_is_the_best_single_channel():
    det = detector()
    wins = sn.make_windows(det.standardise(machine().test), machine().labels, L)
    _, order = sn.oracle_greedy(det, wins, "score")
    s_full = det.score(wins.z)
    for w in range(min(5, len(wins.index))):
        losses = []
        for c in range(C):
            mask = np.zeros((len(wins.index), C), bool)
            mask[:, c] = True
            losses.append(sn.objective_value(det.score(sn.observed(wins.z, mask)), s_full, det.threshold, "score")[w])
        assert order[w, 0] == int(np.argmin(losses))


# ---- one machine, the reference and the gate ----------------------------------------------------------------------------

BUDGETS = (0, 1, 2, 4, C)


def result(seed=0, name="machine-1-1", objective="score", with_oracle=True):
    data = machine(seed, name)
    return sn.evaluate_machine(detector(data), data, L, BUDGETS, objective, with_oracle)


def test_reference_is_never_above_any_policy_and_the_curve_ends_at_zero():
    r = result()
    for name, table in r.tables.items():
        if name != sn.REFERENCE:
            assert np.all(r.tables[sn.REFERENCE] <= table + 1e-12), name
    assert np.all(r.tables[sn.REFERENCE][:, -1] == 0) and np.all(r.tables[sn.REFERENCE][:, 0] > 0)
    for name in sn.DEPLOYABLE_POLICIES:
        assert np.allclose(r.tables[name][:, 0], r.tables[sn.REFERENCE][:, 0])            # budget 0 is hold for everyone
    assert r.n_windows == 38 and r.n_anomalous_windows > 0
    assert all(c.shape == (len(BUDGETS), 3) for c in r.counts.values())


def test_evaluate_rejects_a_budget_above_the_channel_count():
    data = machine()
    with pytest.raises(ValueError):
        sn.evaluate_machine(detector(data), data, L, (0, C + 1))


def test_areas_headroom_and_retained_fraction():
    assert sn.normalised_area([1.0, 0.0], [0, 38]) == pytest.approx(0.5)
    assert sn.normalised_area([1.0, 1.0, 0.0], [0, 19, 38]) == pytest.approx(0.75)
    assert sn.headroom(0.5, 0.4) == pytest.approx(0.2) and np.isnan(sn.headroom(0.0, 0.0))
    assert sn.retained_fraction(0.5, 0.45, 0.4) == pytest.approx(0.5) and np.isnan(sn.retained_fraction(0.4, 0.4, 0.4))
    with pytest.raises(ValueError):
        sn.machine_curves(sn.MachineResult("m", L, (1, 2), "score", tables={"a": np.ones((3, 2))}))


def test_gate_summary_and_bootstrap_are_consistent_and_reproducible():
    results = {"machine-1-1": result(0, "machine-1-1"), "machine-1-5": result(1, "machine-1-5"), "machine-2-1": result(2, "machine-2-1")}
    areas = sn.overall_areas(results, n_channels=C)
    assert 0 < areas[sn.REFERENCE] <= areas[sn.ORACLE] + 1e-12 and all(0 <= areas[n] <= 1.0 + 1e-9 for n in areas)
    chosen = sn.choose_best_fixed(results)
    assert chosen["selected"] in sn.FIXED_POLICIES and set(chosen["areas"]) == set(sn.FIXED_POLICIES)
    a = sn.bootstrap_gate(results, chosen["selected"], num_resamples=500, seed=1, bank=500, n_channels=C)
    b = sn.bootstrap_gate(results, chosen["selected"], num_resamples=500, seed=1, bank=500, n_channels=C)
    assert a == b
    lo, hi = a["headroom_ci"]
    assert lo <= hi and np.isfinite([lo, hi]).all()
    point = sn.headroom(areas[chosen["selected"]], areas[sn.REFERENCE])
    assert lo - 0.2 <= point <= hi + 0.2
    assert "sensitivity_x_volatility - sensitivity" in a["paired_area_difference_ci"]


def test_pooled_f1_has_one_value_per_budget_and_full_observation_matches_the_detector():
    data = machine()
    det = detector(data)
    r = sn.evaluate_machine(det, data, L, BUDGETS)
    f1 = sn.pooled_f1({"machine-1-1": r})
    assert all(len(v) == len(BUDGETS) for v in f1.values())
    full = f1["round_robin"][-1]
    assert f1["top_volatility"][-1] == pytest.approx(full) and f1[sn.ORACLE][-1] == pytest.approx(full)
    wins = sn.make_windows(det.standardise(data.test), data.labels, L)
    tp, fp, fn = sn.alarm_counts(det.score(wins.z), wins.labels, det.threshold)
    assert full == pytest.approx(sn.f1_from_counts(tp, fp, fn))


def test_spec_is_a_valid_domain_spec():
    assert sn.SPEC.oracle_support == "approximate" and sn.SPEC.family == "temporal"
    assert len(sn.SPEC.checksum()) == 64
