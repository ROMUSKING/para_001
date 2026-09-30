"""D1: adaptive sensing on a multivariate sensor stream (Server Machine Dataset). NumPy only.

See docs/plans/d1-0-plan.md for the question, the frozen choices and the gate. In short: a window of ``L``
minutes starts with a snapshot of every channel; the allocator opens ``k`` channels (observed at every
minute of the window) and the rest are held at their snapshot value. The loss is how far a frozen
Mahalanobis detector's log-score, computed on that partial stream, is from its score on the full stream.
Policies are non-learned; the oracle is a privileged greedy selection (the better of forward selection and backward
elimination) and the gate reference is the best-known curve over the oracle and every deployable policy.

Nothing here is called a co-state: no ``dJ/d(state)`` is supervised or derived (AGENTS.md rule 4).
"""

from __future__ import annotations

import datetime
import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .base import DomainSpec

SMD_BASE_URL = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
GROUPS = ((1, 8), (2, 9), (3, 11))
FILE_KINDS = ("train", "test", "test_label")
SPLITS = ("tuning", "validation", "test")
NUM_CHANNELS = 38
DEFAULT_BUDGETS = (0, 1, 2, 4, 8, 12, 16, 24, 38)
FIXED_POLICIES = ("round_robin", "top_volatility", "top_weighted_volatility")
DYNAMIC_POLICIES = ("recent_change", "sensitivity", "sensitivity_x_volatility")
DEPLOYABLE_POLICIES = FIXED_POLICIES + DYNAMIC_POLICIES
ORACLE = "oracle_greedy"
REFERENCE = "reference"
OBJECTIVES = ("score", "decision")
DECISION_TIEBREAK = 1e-5     # weight of the smooth loss inside the greedy decision objective

SPEC = DomainSpec(
    domain_id="d1_sensor_stream_smd", family="temporal",
    native_endpoint="score-fidelity loss of a frozen Mahalanobis detector on the partially observed window "
                    "(secondary: alarm disagreement with the fully observed detector, detection F1 against labels)",
    candidate_kinds=("sample",),
    cost_units={"rate": "channel-minutes observed in a window (k * L), plus a fixed all-channel snapshot",
                "compute": "not charged in D1-0", "latency": "not charged in D1-0", "query": "not used"},
    oracle_support="approximate",
    privilege={"P0": "window-start snapshot, the two previous window-start snapshots, statistics of the machine's training stream",
               "P3": "every value inside the window after the snapshot, and the labels"},
    data_source="Server Machine Dataset (NetManAIOps/OmniAnomaly), tuning and validation machines only",
    licence="MIT (repository LICENSE; docs/licences/register.csv id smd)",
    notes="Rung 0: non-learned policies only. No statement about H2.")


# ---- machines, splits and files --------------------------------------------------------------------------------

def machine_ids() -> tuple[str, ...]:
    return tuple(f"machine-{g}-{i}" for g, n in GROUPS for i in range(1, n + 1))


def split_of(machine: str) -> str:
    """Index mod 4: 0 tuning, 1 validation, 2 and 3 test. Split by machine, before windowing."""
    ids = machine_ids()
    if machine not in ids:
        raise ValueError(f"unknown machine {machine!r}")
    return ("tuning", "validation", "test", "test")[ids.index(machine) % 4]


def split_machines(split: str) -> tuple[str, ...]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    return tuple(m for m in machine_ids() if split_of(m) == split)


def smd_url(kind: str, machine: str) -> str:
    if kind not in FILE_KINDS:
        raise ValueError(f"kind must be one of {FILE_KINDS}")
    return f"{SMD_BASE_URL}/{kind}/{machine}.txt"


def machine_path(root: Path | str, kind: str, machine: str) -> Path:
    return Path(root) / kind / f"{machine}.txt"


def _http_get(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "adjointrwm-d1-0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_machine(root: Path | str, machine: str, allowed: Sequence[str] = ("tuning", "validation"),
                     fetch: Callable[[str], bytes] | None = None) -> list[dict]:
    """Download one machine's files to ``root`` and return manifest rows. Refuses any machine outside ``allowed``
    (the default excludes the test machines, which are never downloaded)."""
    split = split_of(machine)
    if split not in allowed:
        raise PermissionError(f"{machine} belongs to the {split!r} split, which is not allowed here")
    getter = fetch or _http_get
    rows = []
    for kind in FILE_KINDS:
        url = smd_url(kind, machine)
        path = machine_path(root, kind, machine)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            data, retrieved = path.read_bytes(), "cached"
        else:
            try:
                data = getter(url)
            except (urllib.error.URLError, TimeoutError) as e:      # surfaced, never replaced by other data
                raise RuntimeError(f"could not download {url}: {e}") from e
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
            retrieved = datetime.datetime.now(datetime.UTC).isoformat()
        rows.append({"machine": machine, "split": split, "kind": kind, "url": url, "path": f"{kind}/{machine}.txt",
                     "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "retrieved_utc": retrieved})
    return rows


@dataclass(frozen=True)
class MachineData:
    machine: str
    train: np.ndarray       # (T_train, C) normal behaviour
    test: np.ndarray        # (T_test, C)
    labels: np.ndarray      # (T_test,) 0/1


def load_machine(root: Path | str, machine: str, allowed: Sequence[str] = ("tuning", "validation")) -> MachineData:
    import pandas as pd

    split = split_of(machine)
    if split not in allowed:
        raise PermissionError(f"{machine} belongs to the {split!r} split, which is not allowed here")
    train = pd.read_csv(machine_path(root, "train", machine), header=None).to_numpy(dtype=float)
    test = pd.read_csv(machine_path(root, "test", machine), header=None).to_numpy(dtype=float)
    labels = pd.read_csv(machine_path(root, "test_label", machine), header=None).to_numpy(dtype=float).ravel()
    if train.shape[1] != NUM_CHANNELS or test.shape[1] != NUM_CHANNELS or len(labels) != len(test):
        raise ValueError(f"{machine}: unexpected shapes {train.shape}, {test.shape}, {labels.shape}")
    return MachineData(machine, train, test, (labels > 0.5).astype(float))


# ---- the frozen detector -----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Detector:
    mean: np.ndarray
    scale: np.ndarray
    precision: np.ndarray       # P, symmetric positive definite
    threshold: float            # tau
    volatility: np.ndarray      # d_c: std of the lag-L difference of z_c on the training stream
    lag: int
    floor: float
    ridge: float
    shrinkage: float
    quantile: float

    def standardise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale

    def score(self, z: np.ndarray) -> np.ndarray:
        """``z^T P z`` along the last axis."""
        return ((z @ self.precision) * z).sum(axis=-1)


def fit_detector(train: np.ndarray, lag: int, floor: float, ridge: float, shrinkage: float = 0.1,
                 quantile: float = 0.995) -> Detector:
    """Fit on one machine's training stream only."""
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), floor)
    z = (train - mean) / scale
    cov = np.cov(z, rowvar=False)
    sigma = (1.0 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov)) + ridge * np.eye(cov.shape[0])
    precision = np.linalg.inv(sigma)
    precision = (precision + precision.T) / 2.0
    det = Detector(mean, scale, precision, 0.0, (z[lag:] - z[:-lag]).std(axis=0), lag, floor, ridge, shrinkage, quantile)
    threshold = float(np.quantile(det.score(z), quantile))
    return Detector(mean, scale, precision, threshold, det.volatility, lag, floor, ridge, shrinkage, quantile)


def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def full_stream_f1(det: Detector, test: np.ndarray, labels: np.ndarray) -> dict:
    """The fully observed detector against the labels over the whole test stream, at ``tau``."""
    alarm = det.score(det.standardise(test)) > det.threshold
    y = labels > 0.5
    tp, fp, fn = int((alarm & y).sum()), int((alarm & ~y).sum()), int((~alarm & y).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "f1": f1_from_counts(tp, fp, fn),
            "alarm_rate": float(alarm.mean()), "alarm_rate_normal": float(alarm[~y].mean()) if (~y).any() else float("nan")}


def select_detector(machines: Sequence[MachineData], lag: int, floors: Sequence[float], ridges: Sequence[float],
                    shrinkage: float = 0.1, quantile: float = 0.995) -> dict:
    """Pick ``(floor, ridge)`` by the highest mean per-machine F1 of the fully observed detector (tuning machines only)."""
    rows = []
    for floor in floors:
        for ridge in ridges:
            fits = [full_stream_f1(fit_detector(m.train, lag, floor, ridge, shrinkage, quantile), m.test, m.labels) for m in machines]
            rows.append({"floor": float(floor), "ridge": float(ridge), "mean_f1": float(np.mean([f["f1"] for f in fits])),
                         "per_machine_f1": {m.machine: f["f1"] for m, f in zip(machines, fits)},
                         "per_machine_alarm_rate_normal": {m.machine: f["alarm_rate_normal"] for m, f in zip(machines, fits)}})
    best = max(rows, key=lambda r: (r["mean_f1"], -r["ridge"], -r["floor"]))     # ties: smaller ridge, then smaller floor
    return {"grid": rows, "selected": {"floor": best["floor"], "ridge": best["ridge"]}}


# ---- windows -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Windows:
    z: np.ndarray        # (W, L, C) standardised; index 0 of axis 1 is the all-channel snapshot
    prev1: np.ndarray    # (W, C) snapshot of the previous window
    prev2: np.ndarray    # (W, C) snapshot two windows earlier
    labels: np.ndarray   # (W, L)
    index: np.ndarray    # (W,) window number in the stream
    length: int


def make_windows(z: np.ndarray, labels: np.ndarray, length: int, skip: int = 2) -> Windows:
    """Non-overlapping windows tiled from the start; the partial tail and the first ``skip`` windows are dropped."""
    if skip < 2:
        raise ValueError("skip must be at least 2 so that two previous snapshots exist")
    n = z.shape[0] // length
    if n <= skip:
        raise ValueError("stream too short for the window length")
    blocks = z[: n * length].reshape(n, length, z.shape[1])
    snap = blocks[:, 0, :]
    return Windows(z=blocks[skip:], prev1=snap[skip - 1: n - 1], prev2=snap[skip - 2: n - 2],
                   labels=labels[: n * length].reshape(n, length)[skip:], index=np.arange(skip, n), length=length)


def observed(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The stream under an allocation: opened channels are true, the others are held at the snapshot."""
    return np.where(mask[:, None, :], z, z[:, :1, :])


# ---- objectives --------------------------------------------------------------------------------------------------

def _phi(score: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(score, 0.0))


def objective_value(s_obs: np.ndarray, s_full: np.ndarray, threshold: float, kind: str = "score") -> np.ndarray:
    """Per-window loss over the minutes after the snapshot. ``s_*`` have shape (W, L)."""
    if kind == "score":
        return ((_phi(s_obs[:, 1:]) - _phi(s_full[:, 1:])) ** 2).mean(axis=1)
    if kind == "decision":
        return ((s_obs[:, 1:] > threshold) != (s_full[:, 1:] > threshold)).mean(axis=1).astype(float)
    raise ValueError(f"objective must be one of {OBJECTIVES}")


def alarm_counts(s_obs: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[int, int, int]:
    pred = s_obs[:, 1:] > threshold
    y = labels[:, 1:] > 0.5
    return int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())


# ---- policies ----------------------------------------------------------------------------------------------------

def policy_scores(name: str, det: Detector, wins: Windows) -> np.ndarray:
    """Deployable channel scores, shape (W, C). Reads only the snapshots and the training statistics."""
    snap = wins.z[:, 0, :]
    volatility, diag = det.volatility[None, :], np.diag(det.precision)[None, :]
    if name == "top_volatility":
        return np.broadcast_to(volatility, snap.shape).copy()
    if name == "top_weighted_volatility":
        return np.broadcast_to(volatility ** 2 * diag, snap.shape).copy()
    if name == "recent_change":
        return np.abs(snap - wins.prev1)
    if name == "sensitivity":
        return np.abs(snap @ det.precision)
    if name == "sensitivity_x_volatility":
        return np.abs(snap @ det.precision) * volatility
    raise ValueError(f"no channel score for policy {name!r}")


def top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Open the ``k`` highest-scoring channels of each window; ties go to the lower channel index."""
    order = np.argsort(-scores, axis=1, kind="stable")
    mask = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(mask, order[:, :k], True, axis=1)
    return mask


def policy_mask(name: str, k: int, det: Detector, wins: Windows) -> np.ndarray:
    c = wins.z.shape[2]
    if not 0 <= k <= c:
        raise ValueError("budget k must lie in [0, C]")
    if name == "round_robin":
        cols = (wins.index[:, None] * k + np.arange(k)[None, :]) % c
        mask = np.zeros((len(wins.index), c), dtype=bool)
        np.put_along_axis(mask, cols, True, axis=1)
        return mask
    return top_k_mask(policy_scores(name, det, wins), k)


# ---- the privileged greedy oracle --------------------------------------------------------------------------------

def oracle_greedy(det: Detector, wins: Windows, objective: str = "score", direction: str = "forward") -> tuple[np.ndarray, np.ndarray]:
    """Greedy selection using the true window values. Returns ``J`` by number of opened channels, shape (W, C + 1),
    and the selection order, shape (W, C). Not optimal, and ``J`` need not fall at every step.

    ``direction="forward"`` starts from hold and opens the channel that most lowers the loss at each step;
    ``order`` is the opening order. ``direction="backward"`` starts from full observation and holds the channel whose
    closing raises the loss least; ``order`` is the closing order, and the channels open at size ``k`` are all but
    the first ``C - k`` of it (see :func:`oracle_mask`). Forward selection is better at small budgets and backward
    elimination at large ones.

    The score is updated in place: ``s' = s + 2 d (P z~)_c + d^2 P_cc`` with ``d`` the change of channel ``c``'s value
    in ``z~``, so each step costs O(W L C) instead of O(W L C^2).
    """
    if direction not in ("forward", "backward"):
        raise ValueError("direction must be 'forward' or 'backward'")
    forward = direction == "forward"
    z = wins.z
    n_win, length, n_ch = z.shape
    p, diag_p = det.precision, np.diag(det.precision)
    hold = np.repeat(z[:, :1, :], length, axis=1)
    z_cur = hold.copy() if forward else z.copy()
    a = z_cur @ p
    s = (a * z_cur).sum(axis=-1)
    s_full = det.score(z)
    target = _phi(s_full)[:, 1:]
    alarm_full = (s_full > det.threshold)[:, 1:]
    rows = np.arange(n_win)
    done = np.zeros((n_win, n_ch), dtype=bool)          # forward: opened; backward: closed
    order = np.empty((n_win, n_ch), dtype=int)
    j_by_size = np.empty((n_win, n_ch + 1))
    j_by_size[:, 0 if forward else n_ch] = objective_value(s, s_full, det.threshold, objective)
    for step in range(n_ch):
        delta = (z - z_cur) if forward else (hold - z_cur)     # the change in z~ if the channel is toggled now
        s_new = np.maximum(s[:, :, None] + 2.0 * delta * a + delta ** 2 * diag_p[None, None, :], 0.0)
        smooth = ((_phi(s_new)[:, 1:, :] - target[:, :, None]) ** 2).mean(axis=1)
        if objective == "score":
            cost = smooth
        else:
            cost = (((s_new[:, 1:, :] > det.threshold) != alarm_full[:, :, None]).mean(axis=1)
                    + DECISION_TIEBREAK * smooth)
        cost = np.where(done, np.inf, cost)
        pick = cost.argmin(axis=1)
        order[:, step] = pick
        done[rows, pick] = True
        d = delta[rows, :, pick]
        a = a + d[:, :, None] * p[pick][:, None, :]
        z_cur[rows, :, pick] = (z if forward else hold)[rows, :, pick]
        s = s_new[rows, :, pick]
        j_by_size[:, step + 1 if forward else n_ch - step - 1] = objective_value(s, s_full, det.threshold, objective)
    return j_by_size, order


def oracle_mask(order: np.ndarray, size: int, direction: str) -> np.ndarray:
    """Channels open at ``size`` for one greedy path returned by :func:`oracle_greedy`."""
    n_win, n_ch = order.shape
    mask = np.zeros((n_win, n_ch), dtype=bool)
    if direction == "forward":
        np.put_along_axis(mask, order[:, :size], True, axis=1)
        return mask
    np.put_along_axis(mask, order[:, : n_ch - size], True, axis=1)
    return ~mask


def oracle_best(det: Detector, wins: Windows, objective: str = "score") -> tuple[np.ndarray, Callable[[int], np.ndarray]]:
    """The better, per window and size, of forward selection and backward elimination: ``J`` by size (W, C + 1) and a
    function giving the corresponding open-channel mask for a size."""
    j_f, order_f = oracle_greedy(det, wins, objective, "forward")
    j_b, order_b = oracle_greedy(det, wins, objective, "backward")
    best = np.minimum(j_f, j_b)

    def mask_for(size: int) -> np.ndarray:
        use_forward = (j_f[:, size] <= j_b[:, size])[:, None]
        return np.where(use_forward, oracle_mask(order_f, size, "forward"), oracle_mask(order_b, size, "backward"))

    return best, mask_for


# ---- evaluating one machine --------------------------------------------------------------------------------------

@dataclass
class MachineResult:
    machine: str
    length: int
    budgets: tuple
    objective: str
    tables: dict = field(default_factory=dict)      # policy -> (W, K) per-window loss
    counts: dict = field(default_factory=dict)      # policy -> (K, 3) true positives, false positives, false negatives
    n_windows: int = 0
    n_anomalous_windows: int = 0


def evaluate_machine(det: Detector, data: MachineData, length: int, budgets: Sequence[int] = DEFAULT_BUDGETS,
                     objective: str = "score", with_oracle: bool = True, policies: Sequence[str] = DEPLOYABLE_POLICIES) -> MachineResult:
    z = det.standardise(data.test)
    wins = make_windows(z, data.labels, length)
    budgets = tuple(int(k) for k in budgets)
    if max(budgets) > z.shape[1]:
        raise ValueError("a budget exceeds the number of channels")
    s_full = det.score(wins.z)
    result = MachineResult(data.machine, length, budgets, objective, n_windows=len(wins.index),
                           n_anomalous_windows=int((wins.labels[:, 1:].sum(axis=1) > 0).sum()))
    for name in policies:
        cols, cnts = [], []
        for k in budgets:
            s_obs = det.score(observed(wins.z, policy_mask(name, k, det, wins)))
            cols.append(objective_value(s_obs, s_full, det.threshold, objective))
            cnts.append(alarm_counts(s_obs, wins.labels, det.threshold))
        result.tables[name] = np.stack(cols, axis=1)
        result.counts[name] = np.array(cnts, dtype=float)
    if with_oracle:
        j_by_size, mask_for = oracle_best(det, wins, objective)
        result.tables[ORACLE] = j_by_size[:, list(budgets)]
        result.counts[ORACLE] = np.array([alarm_counts(det.score(observed(wins.z, mask_for(k))), wins.labels, det.threshold)
                                          for k in budgets], dtype=float)
        running = np.minimum.accumulate(j_by_size, axis=1)[:, list(budgets)]      # unused budget is legal (stop)
        result.tables[REFERENCE] = np.minimum.reduce([running] + [result.tables[n] for n in policies])
    return result


# ---- areas, the gate and the bootstrap ---------------------------------------------------------------------------

def normalised_area(curve: np.ndarray, budgets: Sequence[int], n_channels: int = NUM_CHANNELS) -> float:
    """Trapezoidal area of a curve that is already divided by its hold-only value, over ``k / C``."""
    x = np.asarray(budgets, dtype=float) / n_channels
    y = np.asarray(curve, dtype=float)
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0)


def machine_curves(result: MachineResult) -> dict[str, np.ndarray]:
    """Per policy, the machine's mean loss curve divided by its hold-only value (budget 0 must be in the grid)."""
    if result.budgets[0] != 0:
        raise ValueError("the budget grid must start at 0")
    hold = float(result.tables[REFERENCE][:, 0].mean()) if REFERENCE in result.tables else float(next(iter(result.tables.values()))[:, 0].mean())
    if hold <= 0:
        raise ValueError(f"{result.machine}: hold-only loss is zero, so the curves cannot be normalised")
    return {name: table.mean(axis=0) / hold for name, table in result.tables.items()}


def overall_areas(results: Mapping[str, MachineResult], n_channels: int = NUM_CHANNELS) -> dict[str, float]:
    """Mean over machines of each policy's normalised area."""
    per = {m: machine_curves(r) for m, r in results.items()}
    first = next(iter(results.values()))
    return {name: float(np.mean([normalised_area(per[m][name], first.budgets, n_channels) for m in results]))
            for name in first.tables}


def choose_best_fixed(tuning: Mapping[str, MachineResult]) -> dict:
    areas = overall_areas(tuning)
    fixed = {n: areas[n] for n in FIXED_POLICIES}
    return {"areas": fixed, "selected": min(fixed, key=lambda n: (fixed[n], FIXED_POLICIES.index(n)))}


def headroom(area_fixed: float, area_reference: float) -> float:
    """``(A_fixed - A_reference) / A_fixed``, the rule of ``opportunity_over_budgets`` on normalised areas."""
    return float((area_fixed - area_reference) / area_fixed) if area_fixed > 0 else float("nan")


def retained_fraction(area_fixed: float, area_policy: float, area_reference: float) -> float:
    denom = area_fixed - area_reference
    return float((area_fixed - area_policy) / denom) if denom > 0 else float("nan")


def bootstrap_gate(results: Mapping[str, MachineResult], fixed: str, num_resamples: int = 10_000, seed: int = 0,
                   bank: int = 20_000, n_channels: int = NUM_CHANNELS) -> dict:
    """Two-stage bootstrap: resample machines with replacement, then windows within each resampled machine.

    Window resamples are drawn as multinomial counts, so ``bank`` resampled curves per machine are one matrix
    product each. Returns percentile 95 % intervals for the headroom, for each policy's normalised area, for the
    fraction of headroom each dynamic policy keeps, and for paired area differences.
    """
    machines = list(results)
    first = results[machines[0]]
    names = list(first.tables)
    rng = np.random.default_rng(seed)
    budgets = first.budgets
    banks = []
    for m in machines:
        stack = np.stack([results[m].tables[n] for n in names], axis=1)                 # (W, P, K)
        n_win = stack.shape[0]
        counts = rng.multinomial(n_win, np.full(n_win, 1.0 / n_win), size=bank).astype(float)
        curves = (counts @ stack.reshape(n_win, -1) / n_win).reshape(bank, len(names), len(budgets))
        hold = curves[:, names.index(REFERENCE) if REFERENCE in names else 0, 0]      # the hold-only value, equal for every policy
        banks.append(curves / hold[:, None, None])
    banks = np.stack(banks, axis=0)                                                     # (M, bank, P, K)
    slot_machine = rng.integers(0, len(machines), size=(num_resamples, len(machines)))
    slot_bank = rng.integers(0, bank, size=(num_resamples, len(machines)))
    mean_curves = banks[slot_machine, slot_bank].mean(axis=1)                            # (R, P, K)
    x = np.asarray(budgets, dtype=float) / n_channels
    areas = ((mean_curves[..., 1:] + mean_curves[..., :-1]) * np.diff(x)).sum(axis=-1) / 2.0     # (R, P)
    col = {n: i for i, n in enumerate(names)}

    def interval(values: np.ndarray) -> list:
        return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]

    head = (areas[:, col[fixed]] - areas[:, col[REFERENCE]]) / areas[:, col[fixed]]
    out = {"num_resamples": num_resamples, "seed": seed, "fixed": fixed,
           "headroom_ci": interval(head), "areas_ci": {n: interval(areas[:, col[n]]) for n in names}}
    denom = areas[:, col[fixed]] - areas[:, col[REFERENCE]]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["retained_ci"] = {n: interval(np.where(denom > 0, (areas[:, col[fixed]] - areas[:, col[n]]) / denom, np.nan))
                              for n in DYNAMIC_POLICIES if n in col}
    pairs = (("sensitivity_x_volatility", "sensitivity"), ("sensitivity_x_volatility", "recent_change"),
             ("sensitivity_x_volatility", fixed), ("recent_change", fixed), (ORACLE, fixed))
    out["paired_area_difference_ci"] = {f"{a} - {b}": interval(areas[:, col[a]] - areas[:, col[b]])
                                        for a, b in pairs if a in col and b in col}
    return out


def gate_summary(results: Mapping[str, MachineResult], fixed: str, min_relative: float = 0.15, **bootstrap_kwargs) -> dict:
    """Point estimates, the frozen gate G1 and the descriptive G2 quantities, with bootstrap intervals."""
    areas = overall_areas(results)
    head = headroom(areas[fixed], areas[REFERENCE])
    boot = bootstrap_gate(results, fixed, **bootstrap_kwargs)
    per_machine = {}
    for m, r in results.items():
        curves = machine_curves(r)
        per_machine[m] = headroom(normalised_area(curves[fixed], r.budgets), normalised_area(curves[REFERENCE], r.budgets))
    return {"fixed": fixed, "areas": areas, "relative_headroom": head, "min_relative": min_relative,
            "passes": bool(head >= min_relative), "robust": bool(boot["headroom_ci"][0] >= min_relative),
            "retained": {n: retained_fraction(areas[fixed], areas[n], areas[REFERENCE]) for n in DYNAMIC_POLICIES},
            "oracle_greedy_area": areas.get(ORACLE), "bootstrap": boot, "per_machine_headroom": per_machine}


def pooled_f1(results: Mapping[str, MachineResult]) -> dict[str, list]:
    """Detection F1 against the labels per policy and budget, pooled over the machines' windows."""
    first = next(iter(results.values()))
    out = {}
    for name in first.counts:
        total = sum(r.counts[name] for r in results.values())
        out[name] = [f1_from_counts(*row) for row in total]
    return out
