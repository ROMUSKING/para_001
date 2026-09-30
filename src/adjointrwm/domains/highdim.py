"""D4-2: higher-dimensional systems, a FLOP ledger for scoring, cheap indicators and an amortised scorer.

See ``docs/plans/d4-2-plan.md``. D4-0b priced scoring in solver steps (step-doubling: ``2n`` steps per pass,
``3n`` with a co-state sweep) and could only *assume* what a cheaper scorer would cost. Here every scoring
scheme is priced in floating-point operations and converted to the CN-step unit of D4-0 and D4-0b (one step
of the same system), so the price is a ratio that can be read off, and it changes with the dimension ``m``:
a CN step costs about ``4 m^2`` operations, a scorer on a fixed feature vector costs about the same at any ``m``.

Contents
    * ``make_system`` / ``CELLS`` / ``cell_instances``: dense stable systems with a fixed ``A`` and goal ``c``,
      and the cells that vary one factor at a time. The test family (seed 2002) is deliberately not available.
    * FLOP functions: ``flops_step``, ``flops_indicator_per_interval``, ``flops_mlp_forward`` and friends.
    * ``forcing_discrepancy``, ``difference_estimate`` and ``costate_table``: cheap, solve-free error indicators (the
      forcing's quadrature discrepancy; a third divided difference of the numerical solution), optionally weighted by
      a tabulated continuous co-state (a one-off per-system cost).
    * ``ScorerModel`` / ``train_scorer``: an amortised scorer, a small MLP trained (NumPy, Adam) to predict the
      exact co-state-weighted local error from features that are cheap to compute. It is supervised by the
      adjoint-weighted error, so it is an *amortised adjoint scorer*, not a direct critic (AGENTS.md rule 4).
    * Pass-based policies ``indicator_policy``, ``cheap_policy``, ``cheap_adjoint_policy`` and ``amortised_policy``.

Everything is generated from frozen seeds and is an analytic benchmark, never real data.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from .base import Cost
from .linear_ode import LinearODEInstance, Pulse, _interval_scores, _select_dorfler, expm
from .runner import BatchPolicy

# ---------------------------------------------------------------------------
# Systems, cells and families
# ---------------------------------------------------------------------------

# The test family (seed 2002) is deliberately absent: D4-2 never generates it.
FAMILY_SEEDS = {"train": 4004, "tuning": 3003, "validation": 1001}


@dataclass(frozen=True)
class System:
    """A dense stable linear system with a fixed goal (the same for every instance of a cell)."""

    name: str
    m: int
    A: tuple
    goal: tuple


def make_system(m: int, seed: int) -> System:
    """``m/2`` damped oscillators (frequency ``2 pi U(1, 4)``, damping ratio ``U(0.02, 0.3)``) rotated by a random
    orthogonal matrix so ``A`` is dense; the goal is a random unit vector."""
    if m < 2 or m % 2:
        raise ValueError("m must be an even number >= 2")
    rng = np.random.default_rng(seed)
    p = m // 2
    omega = 2 * math.pi * rng.uniform(1.0, 4.0, size=p)
    zeta = rng.uniform(0.02, 0.3, size=p)
    block = np.zeros((m, m))
    for i in range(p):
        block[2 * i, 2 * i + 1] = 1.0
        block[2 * i + 1, 2 * i] = -omega[i] ** 2
        block[2 * i + 1, 2 * i + 1] = -2.0 * zeta[i] * omega[i]
    q, _ = np.linalg.qr(rng.normal(size=(m, m)))
    a = q @ block @ q.T
    goal = rng.normal(size=m)
    return System(f"sys_m{m}_s{seed}", m, tuple(tuple(float(x) for x in row) for row in a), tuple(float(x) for x in goal / np.linalg.norm(goal)))


@dataclass(frozen=True)
class Cell:
    name: str
    m: int
    pulse_width: tuple
    amplitude_scale: float
    max_depth: int

    @property
    def system_seed(self) -> int:
        return 7000 + self.m


_BASE = dict(m=32, pulse_width=(0.001, 0.004), amplitude_scale=10.0, max_depth=9)
CELLS = {
    "base": Cell("base", **_BASE),
    "m4": Cell("m4", **{**_BASE, "m": 4}),
    "m64": Cell("m64", **{**_BASE, "m": 64}),
    "wide": Cell("wide", **{**_BASE, "pulse_width": (0.01, 0.04)}),
    "amp1": Cell("amp1", **{**_BASE, "amplitude_scale": 1.0}),
    "depth7": Cell("depth7", **{**_BASE, "max_depth": 7}),
    "depth11": Cell("depth11", **{**_BASE, "max_depth": 11}),
}


def sample_forced_instances(system: System, seed: int, count: int, pulse_width: tuple, amplitude_scale: float, max_depth: int,
                            initial_intervals: int = 16) -> list[LinearODEInstance]:
    """Instances of one system: 2-4 Gaussian forcing pulses and a random initial state."""
    rng = np.random.default_rng(seed)
    m = system.m

    def unit():
        v = rng.normal(size=m)
        return tuple(float(x) for x in v / np.linalg.norm(v))

    out = []
    for i in range(count):
        pulses = tuple(Pulse(float(rng.uniform(0.05, 0.95)), float(rng.uniform(*pulse_width)),
                             float(rng.choice([-1, 1]) * rng.uniform(5.0, 40.0) * amplitude_scale), unit())
                       for _ in range(int(rng.integers(2, 5))))
        out.append(LinearODEInstance(name=f"{system.name}_s{seed}_{i:03d}", A=system.A, y0=tuple(float(x) for x in rng.normal(size=m)),
                                     goal=system.goal, pulses=pulses, initial_intervals=initial_intervals, max_depth=max_depth))
    return out


def cell_instances(cell: Cell | str, family: str, count: int) -> list[LinearODEInstance]:
    """Instances of a named cell and family (``train``, ``tuning`` or ``validation``); names carry the cell and family."""
    cell = CELLS[cell] if isinstance(cell, str) else cell
    if family not in FAMILY_SEEDS:
        raise ValueError(f"family {family!r} is not available in D4-2 (the test family is never generated); use one of {sorted(FAMILY_SEEDS)}")
    system = make_system(cell.m, cell.system_seed)
    instances = sample_forced_instances(system, FAMILY_SEEDS[family], count, cell.pulse_width, cell.amplitude_scale, cell.max_depth)
    return [dataclasses.replace(inst, name=f"d4_2_{cell.name}_{family}_{i:03d}") for i, inst in enumerate(instances)]


def gap_targets(domain, instance, fractions=(0.1, 0.03, 0.01)) -> dict:
    """Objective targets: a fraction of the initial-to-finest-grid gap may remain. Evaluation only (privileged)."""
    floor = domain.finest_objective(instance)
    start = domain.objective(domain.initial_state(instance), instance)
    return {f"gap{int(round(100 * phi))}%": floor + phi * (start - floor) for phi in fractions}


# ---------------------------------------------------------------------------
# FLOP ledger (operations are counted per interval and converted to CN-step units)
# ---------------------------------------------------------------------------

TANH_FLOPS = 10          # cost charged for one tanh
LOG_FLOPS = 20           # cost charged for one logarithm
FEATURE_DIM = 9
FEATURE_NAMES = ("log_h", "t_mid", "log_b_a", "log_b_mid", "log_b_b", "log_indicator", "log_dy_over_h", "log_y_a", "log_diff_estimate")
FEATURE_LOGS = 8      # logarithms taken when the features are formed
FEATURE_FLOOR = 1e-12


def flops_forcing(m: int, pulses: int) -> int:
    """One evaluation of the forcing ``b(t)``: per pulse a Gaussian (~8 operations) and an ``m``-vector update."""
    return pulses * (2 * m + 8)


def flops_step(instance: LinearODEInstance) -> int:
    """One CN step as implemented (:meth:`CrankNicolson.steps`): two matrix-vector products with cached operators
    (``4 m^2``), the forcing average and the sum (``3 m``), and two forcing evaluations."""
    m, p = instance.m, len(instance.pulses)
    return 4 * m * m + 3 * m + 2 * flops_forcing(m, p)


def flops_indicator_per_interval(instance: LinearODEInstance) -> int:
    """The forcing quadrature discrepancy of one interval: forcing at the mid-point and at one node (nodes are shared
    by neighbours), the combination ``h/6 (b_a + 4 b_m + b_b) - h/2 (b_a + b_b)`` (``6 m``) and its norm (``2 m + 1``)."""
    m, p = instance.m, len(instance.pulses)
    return 2 * flops_forcing(m, p) + 6 * m + 2 * m + 1


def flops_difference_per_interval(instance: LinearODEInstance) -> int:
    """Third divided difference of the numerical solution over four nodes (``12 m``), the ``h^3/2`` scaling (``2 m``) and the norm (``2 m + 1``)."""
    return 16 * instance.m + 1


def flops_cheap_per_interval(instance: LinearODEInstance) -> int:
    """The forcing indicator plus the solution-difference estimate plus their sum."""
    return flops_indicator_per_interval(instance) + flops_difference_per_interval(instance) + 1


def flops_mlp_forward(dims: tuple, tanh_flops: int = TANH_FLOPS) -> int:
    """Forward pass of an MLP with tanh hidden layers: ``2 in out`` for each product, ``out`` for the bias, and one tanh per hidden unit."""
    total = 0
    for i, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:])):
        total += 2 * n_in * n_out + n_out
        if i < len(dims) - 2:
            total += tanh_flops * n_out
    return total


def flops_amortised_per_interval(instance: LinearODEInstance, dims: tuple) -> int:
    """Features (the indicator, three forcing norms ``6 m``, two solution norms ``5 m``, the difference estimate, logarithms and
    standardisation) plus the network."""
    m = instance.m
    features = (flops_indicator_per_interval(instance) + 6 * m + 5 * m + flops_difference_per_interval(instance)
                + FEATURE_LOGS * LOG_FLOPS + 2 * FEATURE_DIM)
    return features + flops_mlp_forward(dims)


def flops_costate_lookup_per_interval(instance: LinearODEInstance) -> int:
    """Dot product of the tabulated co-state with one error vector."""
    return 2 * instance.m


def flops_cheap_adjoint_per_interval(instance: LinearODEInstance) -> int:
    """The cheap estimator with both vectors weighted: two dot products are charged on top of the norms (conservative)."""
    return flops_cheap_per_interval(instance) + 2 * flops_costate_lookup_per_interval(instance)


def flops_table_setup(instance: LinearODEInstance) -> int:
    """One-off per system: the tabulated co-state on the finest grid is ``n_fine`` backward matrix-vector products."""
    return instance.n_fine * 2 * instance.m * instance.m


def steps_from_flops(instance: LinearODEInstance, flops: float) -> float:
    return float(flops) / flops_step(instance)


def price_report(instance: LinearODEInstance, dims: tuple) -> dict:
    """Scoring price per interval in CN steps for every arm, and the ratio to step-doubling scoring (2 steps per interval)."""
    prices = {"residual": 2.0, "goal_local": 2.0, "adjoint": 3.0,
              "indicator": steps_from_flops(instance, flops_indicator_per_interval(instance)),
              "cheap": steps_from_flops(instance, flops_cheap_per_interval(instance)),
              "cheap_adjoint": steps_from_flops(instance, flops_cheap_adjoint_per_interval(instance)),
              "amortised": steps_from_flops(instance, flops_amortised_per_interval(instance, dims))}
    return {"flops_per_step": flops_step(instance), "steps_per_interval": prices, "ratio_to_step_doubling": {k: v / 2.0 for k, v in prices.items()},
            "table_setup_steps": steps_from_flops(instance, flops_table_setup(instance))}


def _flop_decision_cost(per_interval_flops):
    """A ``decision_cost`` hook charging ``per_interval_flops(instance)`` for every interval of the mesh, in CN steps."""
    def cost(domain, state, candidates):
        n = len(state.nodes) - 1
        return Cost(compute=n * per_interval_flops(state.instance) / flops_step(state.instance))
    return cost


# ---------------------------------------------------------------------------
# Cheap indicator and tabulated co-state
# ---------------------------------------------------------------------------

def forcing_samples(instance: LinearODEInstance, state):
    """Forcing at the start, mid-point and end of every interval of the mesh, and the interval sizes."""
    t = instance.time_of(state.nodes)
    a, b = t[:-1], t[1:]
    return np.diff(t), instance.forcing(a), instance.forcing((a + b) / 2.0), instance.forcing(b)


def forcing_discrepancy(instance: LinearODEInstance, state) -> np.ndarray:
    """Per interval, Simpson minus trapezoid quadrature of the forcing: ``h/6 (b_a + 4 b_m + b_b) - h/2 (b_a + b_b)``, ``[n, m]``.

    It vanishes where the forcing is linear in time (or zero) and grows where the forcing is not resolved by the mesh.
    It says nothing about the homogeneous part of the truncation error."""
    h, b_a, b_m, b_b = forcing_samples(instance, state)
    h = h[:, None]
    return h / 6.0 * (b_a + 4.0 * b_m + b_b) - h / 2.0 * (b_a + b_b)


def difference_estimate(instance: LinearODEInstance, state) -> np.ndarray:
    """Truncation-error estimate from the numerical solution alone: ``(h^3 / 2) f[x_s, .., x_{s+3}]``, ``[n, m]``.

    CN's local error is ``-(h^3 / 12) y'''`` and the third divided difference over four nodes estimates ``y''' / 6``. The
    four nodes are the interval's own two and its neighbours (shifted inward at the ends of the mesh). It costs no solves
    and sees the roughness of whatever the mesh already resolves, including the homogeneous part."""
    t, y = instance.time_of(state.nodes), state.y
    n = len(t) - 1
    if n < 3:
        raise ValueError("the difference estimate needs at least 3 intervals")
    s = np.clip(np.arange(n) - 1, 0, n - 3)
    x0, x1, x2, x3 = t[s], t[s + 1], t[s + 2], t[s + 3]
    y0, y1, y2, y3 = y[s], y[s + 1], y[s + 2], y[s + 3]
    f01, f12, f23 = (y1 - y0) / (x1 - x0)[:, None], (y2 - y1) / (x2 - x1)[:, None], (y3 - y2) / (x3 - x2)[:, None]
    f012, f123 = (f12 - f01) / (x2 - x0)[:, None], (f23 - f12) / (x3 - x1)[:, None]
    f0123 = (f123 - f012) / (x3 - x0)[:, None]
    return (np.diff(t) ** 3 / 2.0)[:, None] * f0123


_TABLES: dict = {}


def costate_table(instance: LinearODEInstance) -> np.ndarray:
    """Continuous co-state ``Lambda(t_k) = exp(A^T (T - t_k)) c`` at every node of the finest grid, ``[n_fine + 1, m]``.

    For a fixed system and goal it is computed once and shared by every instance (see :func:`flops_table_setup`)."""
    key = (instance.A, instance.goal, instance.n_fine, instance.T)
    if key not in _TABLES:
        delta = instance.T / instance.n_fine
        phi = expm(instance.matrix().T * delta)
        table = np.empty((instance.n_fine + 1, instance.m))
        table[instance.n_fine] = np.asarray(instance.goal, dtype=float)
        for k in range(instance.n_fine - 1, -1, -1):
            table[k] = phi @ table[k + 1]
        _TABLES[key] = table
    return _TABLES[key]


def _dorfler_policy(name: str, per_interval, theta: float, per_interval_flops) -> BatchPolicy:
    if not 0.0 < theta <= 1.0:
        raise ValueError("theta must be in (0, 1]")

    def select(domain, state, candidates, context):
        return _select_dorfler(_interval_scores(candidates, per_interval(state)), theta)

    return BatchPolicy(f"{name}_{theta:g}", select, decision_cost=_flop_decision_cost(per_interval_flops))


def indicator_policy(theta: float) -> BatchPolicy:
    """Dorfler marking on the norm of the forcing discrepancy (goal-agnostic, no solves)."""
    return _dorfler_policy("mark_indicator", lambda s: np.linalg.norm(forcing_discrepancy(s.instance, s), axis=1), theta, flops_indicator_per_interval)


def indicator_adjoint_policy(theta: float) -> BatchPolicy:
    """The same indicator weighted by the tabulated co-state at the interval's right node: ``|Lambda(t_b)^T d_j|``."""
    def per_interval(s):
        weights = costate_table(s.instance)[np.asarray(s.nodes[1:])]
        return np.abs(np.einsum("jm,jm->j", weights, forcing_discrepancy(s.instance, s)))

    return _dorfler_policy("mark_indicator_adjoint", per_interval, theta,
                           lambda inst: flops_indicator_per_interval(inst) + flops_costate_lookup_per_interval(inst))


def cheap_policy(theta: float) -> BatchPolicy:
    """Dorfler marking on ``||d_j|| + ||e_j||``: the forcing discrepancy plus the solution-difference estimate (goal-agnostic, no solves)."""
    def per_interval(s):
        return np.linalg.norm(forcing_discrepancy(s.instance, s), axis=1) + np.linalg.norm(difference_estimate(s.instance, s), axis=1)

    return _dorfler_policy("mark_cheap", per_interval, theta, flops_cheap_per_interval)


def cheap_adjoint_policy(theta: float) -> BatchPolicy:
    """The same two estimates, each weighted by the tabulated co-state at the right node: ``|Lambda^T d_j| + |Lambda^T e_j|``."""
    def per_interval(s):
        weights = costate_table(s.instance)[np.asarray(s.nodes[1:])]
        return (np.abs(np.einsum("jm,jm->j", weights, forcing_discrepancy(s.instance, s)))
                + np.abs(np.einsum("jm,jm->j", weights, difference_estimate(s.instance, s))))

    return _dorfler_policy("mark_cheap_adjoint", per_interval, theta, flops_cheap_adjoint_per_interval)


# ---------------------------------------------------------------------------
# Amortised scorer: features, a small MLP, training
# ---------------------------------------------------------------------------

def scorer_features(instance: LinearODEInstance, state) -> np.ndarray:
    """Features of every interval of the mesh, ``[n, FEATURE_DIM]``. All are P0 or already computed by the solver:
    step size, position, forcing magnitude at three points, the indicator, the solution's local change and size, and the
    solution-difference estimate."""
    h, b_a, b_m, b_b = forcing_samples(instance, state)
    t = instance.time_of(state.nodes)
    mid = (t[:-1] + t[1:]) / 2.0
    d = forcing_discrepancy(instance, state)
    y = state.y
    dy = np.linalg.norm(y[1:] - y[:-1], axis=1) / h
    f = FEATURE_FLOOR
    return np.column_stack([
        np.log(h / instance.T), mid / instance.T,
        np.log(np.linalg.norm(b_a, axis=1) + f), np.log(np.linalg.norm(b_m, axis=1) + f), np.log(np.linalg.norm(b_b, axis=1) + f),
        np.log(np.linalg.norm(d, axis=1) + f), np.log(dy + f), np.log(np.linalg.norm(y[:-1], axis=1) + f),
        np.log(np.linalg.norm(difference_estimate(instance, state), axis=1) + f)])


class TinyMLP:
    """Fully connected network with tanh hidden layers and a linear output (float64 NumPy)."""

    def __init__(self, sizes: tuple, rng: np.random.Generator | None = None, params: list | None = None):
        self.sizes = tuple(sizes)
        if params is not None:
            self.weights = [np.asarray(w, dtype=float) for w, _ in params]
            self.biases = [np.asarray(b, dtype=float) for _, b in params]
        else:
            rng = rng or np.random.default_rng(0)
            self.weights = [rng.normal(size=(i, o)) * math.sqrt(1.0 / i) for i, o in zip(sizes[:-1], sizes[1:])]
            self.biases = [np.zeros(o) for o in sizes[1:]]

    def forward(self, x: np.ndarray, keep: bool = False):
        activations = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.tanh(x)
            activations.append(x)
        return (x, activations) if keep else x

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray):
        """Mean squared error against ``y`` (shape ``[n]``) and its gradients (backpropagation)."""
        out, acts = self.forward(x, keep=True)
        err = out[:, 0] - y
        n = len(y)
        delta = (2.0 / n) * err[:, None]
        gw, gb = [None] * len(self.weights), [None] * len(self.weights)
        for i in range(len(self.weights) - 1, -1, -1):
            gw[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * (1.0 - acts[i] ** 2)
        return float(np.mean(err ** 2)), gw, gb

    def to_json(self) -> list:
        return [[w.tolist(), b.tolist()] for w, b in zip(self.weights, self.biases)]


@dataclass
class ScorerModel:
    """A trained amortised scorer: predicts ``log10 |Lambda_{j+1}^T tau_j|`` from standardised features."""

    mlp: TinyMLP
    mean: np.ndarray
    std: np.ndarray
    report: dict = field(default_factory=dict)

    @property
    def dims(self) -> tuple:
        return self.mlp.sizes

    def predict_log10(self, features: np.ndarray) -> np.ndarray:
        return self.mlp.forward((features - self.mean) / self.std)[:, 0]

    def to_json(self) -> dict:
        return {"dims": list(self.dims), "params": self.mlp.to_json(), "mean": self.mean.tolist(), "std": self.std.tolist(),
                "feature_names": list(FEATURE_NAMES), "report": self.report}

    @classmethod
    def from_json(cls, blob: dict) -> "ScorerModel":
        mlp = TinyMLP(tuple(blob["dims"]), params=[(w, b) for w, b in blob["params"]])
        return cls(mlp, np.asarray(blob["mean"]), np.asarray(blob["std"]), blob.get("report", {}))


LABEL_FLOOR = 1e-15


def scorer_labels(domain, instance, state) -> np.ndarray:
    """Supervision (privileged, training only): ``log10 |Lambda_{j+1}^T tau_j|`` with the exact co-state and local error."""
    return np.log10(np.abs(domain.weighted_local_errors(state, instance)) + LABEL_FLOOR)


def collect_scorer_data(domain, instances, policies, max_passes: int = 40, per_state: int = 256, seed: int = 0):
    """Features, labels and instance indices from the meshes visited by ``policies`` (pass-based) on ``instances``.

    Each visited state contributes at most ``per_state`` randomly chosen refinable intervals."""
    rng = np.random.default_rng(seed)
    xs, ys, groups = [], [], []
    for g, instance in enumerate(instances):
        for policy in policies:
            state = domain.initial_state(instance)
            context = None
            for _ in range(max_passes):
                candidates = domain.legal_candidates(state)
                if not candidates:
                    break
                idx = np.array([c.payload["interval"] for c in candidates])
                if len(idx) > per_state:
                    idx = rng.choice(idx, size=per_state, replace=False)
                xs.append(scorer_features(instance, state)[idx])
                ys.append(scorer_labels(domain, instance, state)[idx])
                groups.append(np.full(len(idx), g))
                picked = [int(i) for i in policy.select(domain, state, candidates, context)]
                state = domain.apply_batch(state, [candidates[i] for i in picked])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)


def train_scorer(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0, hidden: tuple = (16, 16), max_epochs: int = 200,
                 patience: int = 20, lr: float = 3e-3, batch_size: int = 1024, val_fraction: float = 0.2, max_samples: int = 200_000) -> ScorerModel:
    """Fit the scorer with Adam on the mean squared error of the log10 target. The validation part is a random subset of
    whole *instances* (never of intervals of a training instance); the best-validation weights are kept."""
    rng = np.random.default_rng(seed)
    if len(x) > max_samples:
        keep = rng.choice(len(x), size=max_samples, replace=False)
        x, y, groups = x[keep], y[keep], groups[keep]
    unique = np.unique(groups)
    val_groups = set(rng.permutation(unique)[: max(1, int(round(val_fraction * len(unique))))].tolist()) if len(unique) > 1 else set()
    is_val = np.isin(groups, list(val_groups))
    xt, yt, xv, yv = x[~is_val], y[~is_val], x[is_val], y[is_val]
    mean, std = xt.mean(axis=0), xt.std(axis=0) + 1e-9
    xt, xv = (xt - mean) / std, (xv - mean) / std
    y_mean, y_std = float(yt.mean()), float(yt.std() + 1e-9)
    ytn, yvn = (yt - y_mean) / y_std, (yv - y_mean) / y_std
    mlp = TinyMLP((x.shape[1], *hidden, 1), rng)
    m1 = [np.zeros_like(p) for p in mlp.weights + mlp.biases]
    m2 = [np.zeros_like(p) for p in mlp.weights + mlp.biases]
    best, best_state, wait, step, epochs_run = math.inf, None, 0, 0, 0
    history = []
    for epoch in range(max_epochs):
        order = rng.permutation(len(xt))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            _, gw, gb = mlp.loss_and_grads(xt[idx], ytn[idx])
            step += 1
            for k, (p, g) in enumerate(zip(mlp.weights + mlp.biases, gw + gb)):
                m1[k] = 0.9 * m1[k] + 0.1 * g
                m2[k] = 0.999 * m2[k] + 0.001 * g * g
                p -= lr * (m1[k] / (1 - 0.9 ** step)) / (np.sqrt(m2[k] / (1 - 0.999 ** step)) + 1e-8)
        epochs_run += 1
        val_loss = float(np.mean((mlp.forward(xv)[:, 0] - yvn) ** 2)) if len(xv) else float(np.mean((mlp.forward(xt)[:, 0] - ytn) ** 2))
        history.append(val_loss)
        if val_loss < best - 1e-6:
            best, wait = val_loss, 0
            best_state = ([w.copy() for w in mlp.weights], [b.copy() for b in mlp.biases])
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        mlp.weights, mlp.biases = best_state
    # Fold the target normalisation into the last layer so that predictions are in log10 units.
    mlp.weights[-1] = mlp.weights[-1] * y_std
    mlp.biases[-1] = mlp.biases[-1] * y_std + y_mean
    fwd = flops_mlp_forward(mlp.sizes)
    report = {"n_train": int(len(xt)), "n_val": int(len(xv)), "epochs_run": epochs_run, "best_val_loss_normalised": best,
              "val_rmse_log10": float(math.sqrt(best) * y_std) if math.isfinite(best) else None, "target_std_log10": y_std,
              "dims": list(mlp.sizes), "training_flops": int(3 * fwd * len(xt) * epochs_run + fwd * len(xv) * epochs_run), "seed": seed}
    return ScorerModel(mlp, mean, std, report)


def amortised_policy(model: ScorerModel, theta: float) -> BatchPolicy:
    """Dorfler marking on the scorer's prediction, priced by its FLOPs (features plus network)."""
    def per_interval(s):
        return 10.0 ** model.predict_log10(scorer_features(s.instance, s))

    return _dorfler_policy("mark_amortised", per_interval, theta, lambda inst: flops_amortised_per_interval(inst, model.dims))


# ---------------------------------------------------------------------------
# Tuning helper
# ---------------------------------------------------------------------------

def tune_theta(domain, instances, make_policy, thetas, fractions=(0.1, 0.03, 0.01), cap_in_finest_grids: float = 2.0,
               max_passes: int = 40, method: str = "loglog") -> dict:
    """Geometric-mean compute (over instances and targets, real price) of ``make_policy(theta)`` for each ``theta``.

    Uses the tuning family only. The caller picks the minimum and must flag a minimum at the grid's edge."""
    from .runner import evaluate_work_precision  # noqa: PLC0415

    policies = [make_policy(t) for t in thetas]
    logs = {p.name: [] for p in policies}
    for instance in instances:
        cap = cap_in_finest_grids * instance.n_fine
        frame = evaluate_work_precision(domain, [instance], policies, lambda d, i: gap_targets(d, i, fractions), compute_cap=cap,
                                        max_steps=max_passes, random_draws=1, decision_scales=(1.0,), methods=(method,))
        for name, block in frame.groupby("policy"):
            logs[name].extend(np.log(np.minimum(block["compute"].to_numpy(), cap)))
    return {t: float(np.exp(np.mean(logs[p.name]))) for t, p in zip(thetas, policies)}


# ---------------------------------------------------------------------------
# Frozen decision rules and bookkeeping for the D4-2 notebook
# ---------------------------------------------------------------------------

TARGETS = ("gap10%", "gap3%", "gap1%")     # loosest to tightest


def two_adjacent(flags) -> bool:
    return any(flags[i] and flags[i + 1] for i in range(len(flags) - 1))


def beats(block: dict, pair: str, targets=TARGETS) -> list[bool]:
    """Per target: the CI of ``mean(log(compute_a / compute_b))`` for ``pair`` ("a / b") lies below 0, so ``a`` needs less compute."""
    return [bool(block[t]["differences"][pair]["ci_high"] < 0) for t in targets]


def cell_rules(block: dict, targets=TARGETS) -> dict:
    """The frozen D4-2 rules (docs/plans/d4-2-plan.md §8) from ``work_precision_summary(...)[scale '1'][target]`` blocks.

    R1: ``amortised`` beats ``uniform_pass`` at two adjacent targets. R3: ``cheap_adjoint`` beats ``cheap`` at two adjacent targets.
    A D4-1 candidate cell needs both."""
    r1 = two_adjacent(beats(block, "amortised / uniform_pass", targets))
    r3 = two_adjacent(beats(block, "cheap_adjoint / cheap", targets))
    others = {arm: two_adjacent(beats(block, f"{arm} / uniform_pass", targets)) for arm in ("residual", "goal_local", "adjoint", "cheap", "cheap_adjoint", "indicator")
              if f"{arm} / uniform_pass" in block[targets[0]]["differences"]}
    return {"R1_amortised_beats_uniform": r1, "R3_costate_weight_helps_cheap_estimator": r3, "candidate_regime_for_d4_1": bool(r1 and r3),
            "arm_beats_uniform_two_adjacent_targets": others}


def with_free_reference(frame, policy: str = "adjoint", name: str = "adjoint_free"):
    """Scale-1 frame plus the scale-0 rows of ``policy`` relabelled ``name``: the hypothetical free-scoring ceiling, comparable with real-price arms."""
    import pandas as pd  # noqa: PLC0415

    real = frame[frame["decision_scale"] == 1.0]
    free = frame[(frame["decision_scale"] == 0.0) & (frame["policy"] == policy)].assign(policy=name, decision_scale=1.0)
    return pd.concat([real, free], ignore_index=True)


def _geometric_mean_compute(frame, policy: str, target: str, cap: float) -> float:
    block = frame[(frame["policy"] == policy) & (frame["target"] == target)]
    return float(np.exp(np.log(np.minimum(block["compute"].to_numpy(dtype=float), cap)).mean()))


def retained_headroom(frame, cap: float, uniform: str = "uniform_pass", learned: str = "amortised", free: str = "adjoint_free",
                      targets=TARGETS) -> dict:
    """Per target, ``(log u - log a) / (log u - log f)`` from geometric-mean compute: 1 means the learned scorer keeps all of the free
    exact co-state's advantage over uniform refinement, 0 none, negative worse than uniform. ``None`` where the free reference does not beat uniform."""
    out = {}
    for t in targets:
        u, a, f = (_geometric_mean_compute(frame, p, t, cap) for p in (uniform, learned, free))
        out[t] = float((np.log(u) - np.log(a)) / (np.log(u) - np.log(f))) if u > f else None
    return out


def break_even_instances(one_off_steps: float, uniform_compute, arm_compute) -> float | None:
    """One-off cost divided by the mean compute the arm saves per instance against uniform refinement; ``None`` if it saves nothing."""
    saving = float(np.mean(np.asarray(uniform_compute, dtype=float) - np.asarray(arm_compute, dtype=float)))
    return float(one_off_steps / saving) if saving > 0 else None


# ---------------------------------------------------------------------------
# D4-3: the goal varies per instance (docs/plans/d4-3-plan.md)
# ---------------------------------------------------------------------------

GOAL_STREAM_OFFSET = 10_000     # the goals come from their own random stream, so forcing and initial states match D4-2's instances


def varying_goal_instances(cell: Cell | str, family: str, count: int) -> list[LinearODEInstance]:
    """The instances of :func:`cell_instances` (same system, pulses and initial states) with a fresh random unit goal per instance.

    Names carry ``d4_3``. The goals are drawn from ``default_rng(FAMILY_SEEDS[family] + GOAL_STREAM_OFFSET + m)``; the test family is
    refused exactly as in :func:`cell_instances`."""
    cell = CELLS[cell] if isinstance(cell, str) else cell
    base = cell_instances(cell, family, count)
    rng = np.random.default_rng(FAMILY_SEEDS[family] + GOAL_STREAM_OFFSET + cell.m)
    out = []
    for i, inst in enumerate(base):
        v = rng.normal(size=inst.m)
        out.append(dataclasses.replace(inst, goal=tuple(float(x) for x in v / np.linalg.norm(v)), name=f"d4_3_{cell.name}_{family}_{i:03d}"))
    return out


def setup_steps(instance: LinearODEInstance) -> float:
    """The per-instance cost, in CN steps, of tabulating the co-state of this instance's goal on the finest grid."""
    return steps_from_flops(instance, flops_table_setup(instance))


def with_setup_charged(frame, instances, policy: str = "cheap_adjoint", name: str = "cheap_adjoint_setup"):
    """Scale-1 frame plus the rows of ``policy`` relabelled ``name`` with that instance's table setup added to its compute.

    With a goal per instance the tabulated co-state cannot be shared, so its one-off cost is paid by every instance. Targets the
    policy did not reach stay unreached (infinite compute)."""
    import pandas as pd  # noqa: PLC0415

    setup = {inst.name: setup_steps(inst) for inst in instances}
    real = frame[frame["decision_scale"] == 1.0]
    extra = real[real["policy"] == policy].copy()
    extra["compute"] = extra["compute"] + extra["instance"].map(setup)
    extra["policy"] = name
    return pd.concat([real, extra], ignore_index=True)


def cell_rules_varying_goal(block: dict, targets=TARGETS) -> dict:
    """The frozen D4-3 rules (docs/plans/d4-3-plan.md §6) from ``work_precision_summary(...)[scale '1'][target]`` blocks.

    R3v: ``cheap_adjoint`` (co-state given) beats ``cheap`` at two adjacent targets. R4v: ``cheap_adjoint`` beats ``uniform_pass`` at two
    adjacent targets. A D4-1 candidate cell needs both. ``cheap_adjoint_setup`` (table setup charged per instance) is reported, not gated."""
    def has(pair):
        return pair in block[targets[0]]["differences"]

    r3 = two_adjacent(beats(block, "cheap_adjoint / cheap", targets))
    r4 = two_adjacent(beats(block, "cheap_adjoint / uniform_pass", targets))
    others = {arm: two_adjacent(beats(block, f"{arm} / uniform_pass", targets))
              for arm in ("residual", "goal_local", "adjoint", "cheap", "cheap_adjoint", "cheap_adjoint_setup") if has(f"{arm} / uniform_pass")}
    setup_vs_cheap = two_adjacent(beats(block, "cheap_adjoint_setup / cheap", targets)) if has("cheap_adjoint_setup / cheap") else None
    return {"R3v_costate_weight_helps_cheap_estimator": r3, "R4v_goal_aware_cheap_arm_beats_uniform": r4, "candidate_regime_for_d4_1": bool(r3 and r4),
            "setup_charged_cheap_adjoint_beats_cheap": setup_vs_cheap, "arm_beats_uniform_two_adjacent_targets": others}
