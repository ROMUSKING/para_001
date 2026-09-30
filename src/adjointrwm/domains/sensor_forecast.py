"""D1-0b: sensing allocation scored by a frozen forecaster's error (docs/plans/d1-0b-plan.md). NumPy only.

Reuses the D1-0 loader, windows and oracle-mask code from :mod:`adjointrwm.domains.sensor`. A window starts with a
snapshot of every channel; the allocator opens ``k`` channels (observed at every minute) and the rest are held at the
snapshot. A frozen linear forecaster maps the stream as seen to a forecast of every channel ``h`` minutes ahead. The loss
is the squared forecast error against the true future values, the native endpoint; the privileged oracle chooses the
subset that makes the forecast closest to the full-information forecast, knowing the window's states but not its targets.

Nothing here is called a co-state: a co-state would need the unknown target (AGENTS.md rule 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .sensor import (DEFAULT_BUDGETS, NUM_CHANNELS, Windows, headroom, make_windows, normalised_area, observed, oracle_mask, retained_fraction,
                     top_k_mask)

FIXED_POLICIES = ("round_robin", "top_volatility", "top_sensitivity", "top_weighted_volatility")
DYNAMIC_POLICIES = ("recent_change", "recent_change_weighted")
DEPLOYABLE_POLICIES = FIXED_POLICIES + DYNAMIC_POLICIES
ORACLE = "oracle_fidelity"
REFERENCE = "reference"
FLOOR = 0.05
RIDGE = 1.0
HORIZON = 5
MIN_ENDPOINT_MOVE = 0.05
MIN_HEADROOM = 0.15


# ---- the frozen forecaster -----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Forecaster:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray        # B, (C, C): the forecast is z B
    volatility: np.ndarray     # d_c: std of the lag-L difference of z_c on the training stream
    sensitivity: np.ndarray    # s_c = ||B[c, :]||^2
    horizon: int
    lag: int
    floor: float
    ridge: float

    def standardise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale

    def predict(self, z: np.ndarray) -> np.ndarray:
        return z @ self.weights


def fit_forecaster(train: np.ndarray, horizon: int = HORIZON, lag: int = 60, floor: float = FLOOR, ridge: float = RIDGE) -> Forecaster:
    """Ridge regression of ``z_{t+h}`` on ``z_t`` over the machine's training stream."""
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), floor)
    z = (train - mean) / scale
    x, y = z[:-horizon], z[horizon:]
    weights = np.linalg.solve(x.T @ x + ridge * np.eye(z.shape[1]), x.T @ y)
    return Forecaster(mean, scale, weights, (z[lag:] - z[:-lag]).std(axis=0), (weights ** 2).sum(axis=1), horizon, lag, floor, ridge)


# ---- losses --------------------------------------------------------------------------------------------------------

def _span(wins: Windows, horizon: int) -> tuple[int, int]:
    length = wins.z.shape[1]
    if length - 1 - horizon < 1:
        raise ValueError("the window is too short for this horizon")
    return 1, length - horizon


def native_loss(fc: Forecaster, wins: Windows, z_obs: np.ndarray) -> np.ndarray:
    """Per window: mean over minutes ``t = 1 … L - 1 - h`` and channels of ``(ŷ_{t+h} - z_{t+h})^2``, ``ŷ`` from the stream as seen."""
    a, b = _span(wins, fc.horizon)
    pred = z_obs[:, a:b, :] @ fc.weights
    target = wins.z[:, a + fc.horizon:, :]
    return ((pred - target) ** 2).mean(axis=(1, 2))


def fidelity_loss(fc: Forecaster, wins: Windows, z_obs: np.ndarray) -> np.ndarray:
    """Per window: mean over minutes and channels of ``((z~_t - z_t) B)^2``, the distance of the forecast from the full-information forecast."""
    a, b = _span(wins, fc.horizon)
    return (((z_obs[:, a:b, :] - wins.z[:, a:b, :]) @ fc.weights) ** 2).mean(axis=(1, 2))


# ---- policies ------------------------------------------------------------------------------------------------------

def policy_scores(name: str, fc: Forecaster, wins: Windows) -> np.ndarray:
    """Deployable channel scores, shape (W, C). Reads only the snapshots and the training statistics."""
    snap = wins.z[:, 0, :]
    shape = snap.shape
    if name == "top_volatility":
        return np.broadcast_to(fc.volatility[None, :], shape).copy()
    if name == "top_sensitivity":
        return np.broadcast_to(fc.sensitivity[None, :], shape).copy()
    if name == "top_weighted_volatility":
        return np.broadcast_to((fc.sensitivity * fc.volatility ** 2)[None, :], shape).copy()
    if name == "recent_change":
        return np.abs(snap - wins.prev1)
    if name == "recent_change_weighted":
        return fc.sensitivity[None, :] * (snap - wins.prev1) ** 2
    raise ValueError(f"no channel score for policy {name!r}")


def policy_mask(name: str, k: int, fc: Forecaster, wins: Windows) -> np.ndarray:
    c = wins.z.shape[2]
    if not 0 <= k <= c:
        raise ValueError("budget k must lie in [0, C]")
    if name == "round_robin":
        cols = (wins.index[:, None] * k + np.arange(k)[None, :]) % c
        mask = np.zeros((len(wins.index), c), dtype=bool)
        np.put_along_axis(mask, cols, True, axis=1)
        return mask
    return top_k_mask(policy_scores(name, fc, wins), k)


# ---- the privileged oracle (fidelity objective) ---------------------------------------------------------------------------

def oracle_greedy(fc: Forecaster, wins: Windows, direction: str = "forward") -> tuple[np.ndarray, np.ndarray]:
    """Greedy selection minimising the fidelity loss with the true window values. Returns the fidelity loss by number of opened channels,
    shape (W, C + 1), and the selection order, shape (W, C); see :func:`adjointrwm.domains.sensor.oracle_greedy` for the conventions.

    With ``e_t = (z~_t - z_t) B``, toggling channel ``c`` changes ``e_t`` by ``-/+ delta_{t,c} B[c, :]``, so each candidate's loss is
    ``|e|^2 - 2 delta (e . B_c) + delta^2 |B_c|^2`` (forward) and the same with opposite signs (backward): O(W L C^2) per step."""
    if direction not in ("forward", "backward"):
        raise ValueError("direction must be 'forward' or 'backward'")
    forward = direction == "forward"
    a, b = _span(wins, fc.horizon)
    z = wins.z[:, a:b, :]
    hold = np.repeat(wins.z[:, :1, :], b - a, axis=1)
    n_win, _, n_ch = z.shape
    w = fc.weights
    wrow_sq = fc.sensitivity
    delta = (hold - z) if forward else np.zeros_like(z)            # forward: z~ - z for every channel still held; backward: 0 while open
    away = (hold - z)                                               # the change of z~ - z if the channel is held
    e = delta @ w
    rows = np.arange(n_win)
    done = np.zeros((n_win, n_ch), dtype=bool)                      # forward: opened; backward: closed
    order = np.empty((n_win, n_ch), dtype=int)
    j_by_size = np.empty((n_win, n_ch + 1))
    j_by_size[:, 0 if forward else n_ch] = (e ** 2).mean(axis=(1, 2))
    for step in range(n_ch):
        proj = e @ w.T                                              # (W, T, C): e . B_c for every channel
        if forward:
            step_delta = delta                                      # current z~ - z of each channel (opening zeroes it)
            cost = ((e ** 2).sum(axis=2)[:, :, None] - 2.0 * step_delta * proj + step_delta ** 2 * wrow_sq[None, None, :]).mean(axis=1)
        else:
            cost = ((e ** 2).sum(axis=2)[:, :, None] + 2.0 * away * proj + away ** 2 * wrow_sq[None, None, :]).mean(axis=1)
        cost = np.where(done, np.inf, cost)
        pick = cost.argmin(axis=1)
        order[:, step] = pick
        done[rows, pick] = True
        if forward:
            d = delta[rows, :, pick]
            e = e - d[:, :, None] * w[pick][:, None, :]
            delta[rows, :, pick] = 0.0
        else:
            d = away[rows, :, pick]
            e = e + d[:, :, None] * w[pick][:, None, :]
        j_by_size[:, step + 1 if forward else n_ch - step - 1] = (e ** 2).mean(axis=(1, 2))
    return j_by_size, order


def oracle_best(fc: Forecaster, wins: Windows) -> tuple[np.ndarray, np.ndarray]:
    """The better, per window and size, of forward selection and backward elimination: the fidelity loss by size (W, C + 1) and, with unused
    budget legal, the masks at every size: ``masks[size]`` has shape (W, C) and is the best size ``j <= size`` by the oracle's own objective."""
    j_f, order_f = oracle_greedy(fc, wins, "forward")
    j_b, order_b = oracle_greedy(fc, wins, "backward")
    best = np.minimum(j_f, j_b)
    n_win, n_ch = order_f.shape
    per_size = np.stack([np.where((j_f[:, s] <= j_b[:, s])[:, None], oracle_mask(order_f, s, "forward"), oracle_mask(order_b, s, "backward"))
                         for s in range(n_ch + 1)])                # (C + 1, W, C)
    running = np.minimum.accumulate(best, axis=1)
    star = np.empty((n_win, n_ch + 1), dtype=int)
    for s in range(n_ch + 1):
        star[:, s] = np.argmin(best[:, : s + 1], axis=1)
    masks = np.stack([per_size[star[:, s], np.arange(n_win)] for s in range(n_ch + 1)])
    return running, masks


# ---- evaluating one machine --------------------------------------------------------------------------------------

@dataclass
class MachineResult:
    machine: str
    length: int
    horizon: int
    budgets: tuple
    native: dict = field(default_factory=dict)       # policy -> (W, K) native loss
    fidelity: dict = field(default_factory=dict)     # policy -> (W, K) fidelity loss
    full: np.ndarray | None = None                    # (W,) native loss with every channel observed
    n_windows: int = 0

    def excess(self) -> dict[str, np.ndarray]:
        """Native loss above full observation, per policy, shape (W, K)."""
        return {name: table - self.full[:, None] for name, table in self.native.items()}


def evaluate_machine(fc: Forecaster, test: np.ndarray, machine: str, length: int, budgets: Sequence[int] = DEFAULT_BUDGETS, with_oracle: bool = True,
                     policies: Sequence[str] = DEPLOYABLE_POLICIES) -> MachineResult:
    z = fc.standardise(test)
    wins = make_windows(z, np.zeros(len(z)), length)
    budgets = tuple(int(k) for k in budgets)
    if max(budgets) > z.shape[1]:
        raise ValueError("a budget exceeds the number of channels")
    full = native_loss(fc, wins, wins.z)
    result = MachineResult(machine, length, fc.horizon, budgets, full=full, n_windows=len(wins.index))

    def record(name: str, masks_for_budget) -> None:
        nat, fid = [], []
        for k in budgets:
            z_obs = observed(wins.z, masks_for_budget(k))
            nat.append(native_loss(fc, wins, z_obs))
            fid.append(fidelity_loss(fc, wins, z_obs))
        result.native[name], result.fidelity[name] = np.stack(nat, axis=1), np.stack(fid, axis=1)

    for name in policies:
        record(name, lambda k, name=name: policy_mask(name, k, fc, wins))
    if with_oracle:
        _, masks = oracle_best(fc, wins)
        record(ORACLE, lambda k: masks[k])
        result.native[REFERENCE], result.fidelity[REFERENCE] = result.native[ORACLE], result.fidelity[ORACLE]
    return result


# ---- areas, the gates and the bootstrap --------------------------------------------------------------------------

def machine_curves(result: MachineResult) -> dict[str, np.ndarray]:
    """Per policy, the machine's mean excess curve divided by its mean full-observation loss: it starts at the machine's ``r`` (which may be
    zero or negative) and is 0 at the full budget."""
    if result.budgets[0] != 0:
        raise ValueError("the budget grid must start at 0")
    full = float(result.full.mean())
    if full <= 0:
        raise ValueError(f"{result.machine}: the full-observation loss is not positive, so the excess cannot be normalised")
    return {name: table.mean(axis=0) / full for name, table in result.excess().items()}


def overall_areas(results: Mapping[str, MachineResult], n_channels: int = NUM_CHANNELS) -> dict[str, float]:
    per = {m: machine_curves(r) for m, r in results.items()}
    first = next(iter(results.values()))
    return {name: float(np.mean([normalised_area(per[m][name], first.budgets, n_channels) for m in results])) for name in first.native}


def choose_best_fixed(tuning: Mapping[str, MachineResult], n_channels: int = NUM_CHANNELS) -> dict:
    areas = overall_areas(tuning, n_channels)
    fixed = {n: areas[n] for n in FIXED_POLICIES}
    return {"areas": fixed, "selected": min(fixed, key=lambda n: (fixed[n], FIXED_POLICIES.index(n)))}


def endpoint_moves(results: Mapping[str, MachineResult]) -> dict:
    """``r = (mean J(0) - mean J(C)) / mean J(C)`` per machine and its mean over machines (point estimates; the hold column of any policy is hold)."""
    per = {}
    for m, r in results.items():
        table = next(t for n, t in r.native.items() if n != REFERENCE)
        per[m] = float((table[:, 0].mean() - r.full.mean()) / r.full.mean())
    return {"per_machine": per, "mean": float(np.mean(list(per.values())))}


def _interval(values: np.ndarray) -> list:
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


def bootstrap_gate(results: Mapping[str, MachineResult], fixed: str, num_resamples: int = 10_000, seed: int = 0, bank: int = 20_000,
                   n_channels: int = NUM_CHANNELS) -> dict:
    """Two-stage bootstrap (machines with replacement, then windows within each resampled machine; curves normalised by the resampled
    full-observation loss) of the headroom, each policy's normalised
    area, the fraction of headroom each dynamic policy keeps, the paired differences of the plan's G-VOI check, and the endpoint-moves statistic."""
    machines = list(results)
    first = results[machines[0]]
    names = list(first.native)
    rng = np.random.default_rng(seed)
    budgets = first.budgets
    banks, end_banks = [], []
    for m in machines:
        r = results[m]
        excess = r.excess()
        stack = np.stack([excess[n] for n in names], axis=1)                              # (W, P, K)
        raw_hold = np.stack([next(t for n, t in r.native.items() if n != REFERENCE)[:, 0], r.full], axis=1)    # (W, 2)
        n_win = stack.shape[0]
        counts = rng.multinomial(n_win, np.full(n_win, 1.0 / n_win), size=bank).astype(float)
        curves = (counts @ stack.reshape(n_win, -1) / n_win).reshape(bank, len(names), len(budgets))
        means = counts @ raw_hold / n_win                                                   # (bank, 2): mean hold and mean full loss
        banks.append(curves / means[:, 1][:, None, None])
        end_banks.append((means[:, 0] - means[:, 1]) / means[:, 1])
    banks, end_banks = np.stack(banks, axis=0), np.stack(end_banks, axis=0)                 # (M, bank, P, K), (M, bank)
    slot_machine = rng.integers(0, len(machines), size=(num_resamples, len(machines)))
    slot_bank = rng.integers(0, bank, size=(num_resamples, len(machines)))
    mean_curves = banks[slot_machine, slot_bank].mean(axis=1)                               # (R, P, K)
    endpoint = end_banks[slot_machine, slot_bank].mean(axis=1)                              # (R,)
    x = np.asarray(budgets, dtype=float) / n_channels
    areas = ((mean_curves[..., 1:] + mean_curves[..., :-1]) * np.diff(x)).sum(axis=-1) / 2.0
    col = {n: i for i, n in enumerate(names)}
    head = (areas[:, col[fixed]] - areas[:, col[REFERENCE]]) / areas[:, col[fixed]]
    denom = areas[:, col[fixed]] - areas[:, col[REFERENCE]]
    out = {"num_resamples": num_resamples, "seed": seed, "fixed": fixed, "headroom_ci": _interval(head), "endpoint_moves_ci": _interval(endpoint),
           "areas_ci": {n: _interval(areas[:, col[n]]) for n in names}}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["retained_ci"] = {n: _interval(np.where(denom > 0, (areas[:, col[fixed]] - areas[:, col[n]]) / denom, np.nan)) for n in DYNAMIC_POLICIES if n in col}
    pairs = (("top_weighted_volatility", "top_sensitivity"), ("top_weighted_volatility", "top_volatility"), ("recent_change_weighted", "recent_change"),
             ("recent_change_weighted", "top_sensitivity"), ("recent_change_weighted", fixed), ("recent_change", fixed), (ORACLE, fixed))
    out["paired_area_difference_ci"] = {f"{a} - {b}": _interval(areas[:, col[a]] - areas[:, col[b]]) for a, b in pairs if a in col and b in col}
    return out


def gate_summary(results: Mapping[str, MachineResult], fixed: str, min_endpoint: float = MIN_ENDPOINT_MOVE, min_headroom: float = MIN_HEADROOM,
                 **bootstrap_kwargs) -> dict:
    """G0 (the endpoint moves with sensing), G1 (opportunity), G2 (retained fractions) and G-VOI inputs, with bootstrap intervals."""
    n_channels = bootstrap_kwargs.get("n_channels", NUM_CHANNELS)
    areas = overall_areas(results, n_channels)
    head = headroom(areas[fixed], areas[REFERENCE])
    boot = bootstrap_gate(results, fixed, **bootstrap_kwargs)
    move = endpoint_moves(results)
    per_machine = {}
    for m, r in results.items():
        curves = machine_curves(r)
        per_machine[m] = headroom(normalised_area(curves[fixed], r.budgets, n_channels), normalised_area(curves[REFERENCE], r.budgets, n_channels))
    g0 = bool(move["mean"] >= min_endpoint)
    return {"fixed": fixed, "areas": areas, "endpoint_moves": move, "min_endpoint": min_endpoint, "G0_passes": g0,
            "G0_robust": bool(boot["endpoint_moves_ci"][0] >= min_endpoint), "relative_headroom": head, "min_headroom": min_headroom,
            "G1_passes": bool(head >= min_headroom), "G1_robust": bool(boot["headroom_ci"][0] >= min_headroom), "G1_vacuous": not g0,
            "retained": {n: retained_fraction(areas[fixed], areas[n], areas[REFERENCE]) for n in DYNAMIC_POLICIES}, "bootstrap": boot,
            "per_machine_headroom": per_machine}
