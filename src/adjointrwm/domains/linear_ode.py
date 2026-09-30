"""D4: goal-oriented adaptive time stepping for a linear dynamical system.

``y' = A y + b(t)`` on ``[0, T]``, ``y(0) = y0``, forced by Gaussian pulses. The allocator
chooses which time interval to bisect to reduce the error in the quantity of interest
``J = c^T y(T)``.

Why this domain (docs/plans/cross-domain-plan.md §5): a local error made early is carried
forward and damped, amplified or rotated by the dynamics, so *where* to refine depends on the
future through the discrete co-state. For Crank-Nicolson (CN) with step map
``y_{j+1} = M_j y_j + g_j`` the error obeys ``e_{j+1} = M_j e_j + tau_{j+1}`` exactly, hence

    c^T (y(T) - y_N) = sum_j  Lambda_{j+1}^T tau_{j+1},   Lambda_N = c,  Lambda_j = M_j^T Lambda_{j+1}

which is the co-state recursion of comprehensive plan §4.2. ``tests/test_domains.py`` checks
this identity, the co-state against finite differences, and the reference solution against a
closed form.

**Declared objective** (lower is better): the cancellation-free goal-oriented error

    objective = sum_j | Lambda_{j+1}^T tau_{j+1} |   >=   | c^T (y(T) - y_N) |

The signed QoI error lets local errors of opposite sign cancel, so a greedy oracle on
``|c^T e_N|`` wins by finding lucky cancellations instead of by allocating well (seen while
designing D4-0). The bound is the standard adaptivity target of dual-weighted-residual methods;
``|c^T e_N|`` is still reported by :meth:`AdaptiveTimeSteppingDomain.native_metrics`.
``objective_kind='abs_error'`` restores the signed version for sensitivity analyses.

Privilege is *computational* here: the model (A, forcing, y0, c) is P0. What deployed policies
cannot have is the exact solution, which the objective and the privileged teachers use. The
exact solution comes from the matrix exponential on the finest dyadic grid with 10-point
Gauss-Legendre quadrature of the forcing.

Instances come from :func:`sample_instances`, an analytic benchmark generated in the repository
with frozen seeds (like ``HJoinBench-0``). They are never presented as real data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .base import AllocationDomain, Candidate, Cost, DomainSpec
from .runner import BatchPolicy, Policy

LOCAL_ERROR_REDUCTION = 0.75  # CN is second order: bisecting leaves ~1/4 of an interval's local error


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def expm(matrix: np.ndarray, terms: int = 24) -> np.ndarray:
    """Matrix exponential by scaling and squaring with a Taylor series (small, well-scaled matrices)."""
    m = np.asarray(matrix, dtype=float)
    norm = float(np.abs(m).sum(axis=1).max()) if m.size else 0.0
    squarings = max(0, int(math.ceil(math.log2(norm))) + 1) if norm > 0 else 0
    scaled = m / (2.0 ** squarings)
    result = np.eye(m.shape[0])
    term = np.eye(m.shape[0])
    for k in range(1, terms + 1):
        term = term @ scaled / k
        result = result + term
    for _ in range(squarings):
        result = result @ result
    return result


@dataclass(frozen=True)
class Pulse:
    time: float
    width: float
    amplitude: float
    direction: tuple


@dataclass(frozen=True)
class LinearODEInstance:
    name: str
    A: tuple                   # rows
    y0: tuple
    goal: tuple                # c
    pulses: tuple = ()
    T: float = 1.0
    initial_intervals: int = 16
    max_depth: int = 7         # finest grid = initial_intervals * 2**max_depth intervals

    @property
    def m(self) -> int:
        return len(self.y0)

    @property
    def n_fine(self) -> int:
        return self.initial_intervals * 2 ** self.max_depth

    def matrix(self) -> np.ndarray:
        return np.array(self.A, dtype=float)

    def forcing(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        out = np.zeros(t.shape + (self.m,))
        for p in self.pulses:
            out += (p.amplitude * np.exp(-((t - p.time) ** 2) / (2.0 * p.width ** 2)))[..., None] * np.asarray(p.direction)
        return out

    def time_of(self, index) -> np.ndarray:
        return np.asarray(index, dtype=float) * (self.T / self.n_fine)


def reference_solution(instance: LinearODEInstance) -> np.ndarray:
    """Exact solution at every node of the finest grid, ``[n_fine + 1, m]``."""
    delta = instance.T / instance.n_fine
    a = instance.matrix()
    phi = expm(a * delta)
    x, w = np.polynomial.legendre.leggauss(10)
    s = (x + 1.0) / 2.0 * delta
    weights = w / 2.0 * delta
    kernels = np.stack([expm(a * (delta - sq)) for sq in s])                  # [10, m, m]
    t0 = np.arange(instance.n_fine) * delta
    forcing = instance.forcing(t0[:, None] + s[None, :])                      # [n, 10, m]
    increments = np.einsum("q,qij,kqj->ki", weights, kernels, forcing)       # [n, m]
    y = np.empty((instance.n_fine + 1, instance.m))
    y[0] = np.asarray(instance.y0, dtype=float)
    for k in range(instance.n_fine):
        y[k + 1] = phi @ y[k] + increments[k]
    return y


class CrankNicolson:
    """CN steps ``y_{j+1} = M y_j + L^{-1} (h/2)(b(t) + b(t+h))`` with per-step-size caching."""

    def __init__(self, instance: LinearODEInstance):
        self.instance = instance
        self.a = instance.matrix()
        self._cache = {}

    def operators(self, h: float):
        if h not in self._cache:
            eye = np.eye(self.instance.m)
            l_inv = np.linalg.inv(eye - 0.5 * h * self.a)
            self._cache[h] = (l_inv, l_inv @ (eye + 0.5 * h * self.a))
        return self._cache[h]

    def step(self, y, t: float, h: float) -> np.ndarray:
        l_inv, m = self.operators(h)
        b = self.instance.forcing(np.array([t, t + h]))
        return m @ y + l_inv @ (0.5 * h * (b[0] + b[1]))

    def steps(self, y: np.ndarray, t: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Many independent CN steps at once (``y [n, m]`` from times ``t [n]`` with sizes ``h [n]``)."""
        b0, b1 = self.instance.forcing(t), self.instance.forcing(t + h)
        out = np.empty_like(y)
        for size in np.unique(h):
            idx = np.flatnonzero(h == size)
            l_inv, m = self.operators(float(size))
            out[idx] = y[idx] @ m.T + (0.5 * size * (b0[idx] + b1[idx])) @ l_inv.T
        return out

    def solve(self, nodes: tuple) -> np.ndarray:
        t = self.instance.time_of(nodes)
        b = self.instance.forcing(t)
        y = np.empty((len(nodes), self.instance.m))
        y[0] = np.asarray(self.instance.y0, dtype=float)
        for j in range(len(nodes) - 1):
            h = t[j + 1] - t[j]
            l_inv, m = self.operators(h)
            y[j + 1] = m @ y[j] + l_inv @ (0.5 * h * (b[j] + b[j + 1]))
        return y


@dataclass(frozen=True, eq=False)
class ODEState:
    instance: LinearODEInstance    # the P0 model
    nodes: tuple                   # indices on the finest grid, sorted, first 0, last n_fine
    y: np.ndarray                  # CN solution at the nodes


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

class AdaptiveTimeSteppingDomain(AllocationDomain):
    spec = DomainSpec(
        domain_id="d4_adaptive_time_stepping",
        family="temporal",
        native_endpoint="|c^T y(T) - c^T y_N| (error in the quantity of interest)",
        candidate_kinds=("refine",),
        cost_units={"rate": "stored states (one per new node)",
                    "compute": "Crank-Nicolson steps: re-solve from the refined interval to the end, plus scoring cost",
                    "latency": "not measured", "query": "none"},
        oracle_support="exact",
        privilege={"model (A, forcing, y0, c), CN solution": "P0",
                   "exact solution y(t)": "teacher/oracle only (computational privilege)"},
        data_source="generated analytic benchmark: adjointrwm.domains.linear_ode.sample_instances with frozen seeds",
        licence="n/a (generated in this repository)",
    )

    OBJECTIVES = ("bound", "abs_error")

    def __init__(self, objective_kind: str = "bound"):
        if objective_kind not in self.OBJECTIVES:
            raise ValueError(f"objective_kind must be one of {self.OBJECTIVES}")
        self.objective_kind = objective_kind
        self._reference, self._steppers = {}, {}

    # P0 helpers -----------------------------------------------------------
    def stepper(self, instance) -> CrankNicolson:
        if instance not in self._steppers:
            self._steppers[instance] = CrankNicolson(instance)
        return self._steppers[instance]

    def observation(self, instance):
        return instance

    def initial_state(self, instance):
        nodes = tuple(range(0, instance.n_fine + 1, 2 ** instance.max_depth))
        return ODEState(instance, nodes, self.stepper(instance).solve(nodes))

    def legal_candidates(self, state):
        """``refine(j)`` bisects interval ``j``. Its compute cost is the number of CN steps that must be
        re-solved: the two new half steps and every later step (an earlier error correction changes
        the whole downstream solution), i.e. ``n + 1 - j`` for ``n`` intervals before the refinement."""
        n = len(state.nodes) - 1
        out = []
        for j, (a, b) in enumerate(zip(state.nodes[:-1], state.nodes[1:])):
            if b - a >= 2:
                out.append(Candidate(f"refine:{a}:{b}", "refine", (a, b), Cost(rate=1.0, compute=float(n + 1 - j)), {"interval": j}))
        return out

    def apply(self, state, candidate):
        a, b = candidate.target
        nodes = tuple(sorted(set(state.nodes) | {(a + b) // 2}))
        return ODEState(state.instance, nodes, self.stepper(state.instance).solve(nodes))

    def apply_batch(self, state, candidates):
        """Bisect every chosen interval, then re-solve once."""
        mids = {(a + b) // 2 for a, b in (c.target for c in candidates)}
        nodes = tuple(sorted(set(state.nodes) | mids))
        return ODEState(state.instance, nodes, self.stepper(state.instance).solve(nodes))

    def batch_cost(self, state, candidates) -> Cost:
        """One re-solve from the first refined interval to the end: ``n + k - j_min`` CN steps for ``k`` refinements
        of ``n`` intervals (equal to ``n + 1 - j`` for a single refinement)."""
        n = len(state.nodes) - 1
        j_min = min(c.payload["interval"] for c in candidates)
        return Cost(rate=float(len(candidates)), compute=float(n + len(candidates) - j_min))

    def finest_objective(self, instance) -> float:
        """Objective on the finest grid (uniform, every interval refined to the maximum depth). Evaluation only."""
        nodes = tuple(range(instance.n_fine + 1))
        return self.objective(ODEState(instance, nodes, self.stepper(instance).solve(nodes)), instance)

    def local_error_estimates(self, state) -> np.ndarray:
        """Step-doubling estimate ``tau_hat_j = 4/3 (two half steps - one full step)``; 2 extra steps each."""
        stepper, t = self.stepper(state.instance), state.instance.time_of(state.nodes)
        start, h = t[:-1], np.diff(t)
        full = stepper.steps(state.y[:-1], start, h)
        half = stepper.steps(stepper.steps(state.y[:-1], start, h / 2), start + h / 2, h / 2)
        return (4.0 / 3.0) * (half - full)

    def costate(self, state) -> np.ndarray:
        """Weight of each interval's local error: ``out[j] = Lambda_{j+1}`` (one backward sweep)."""
        stepper, t = self.stepper(state.instance), state.instance.time_of(state.nodes)
        n = len(state.nodes) - 1
        out = np.empty((n, state.instance.m))
        out[n - 1] = np.asarray(state.instance.goal, dtype=float)
        for j in range(n - 1, 0, -1):
            _, m = stepper.operators(t[j + 1] - t[j])
            out[j - 1] = m.T @ out[j]
        return out

    # Privileged -----------------------------------------------------------
    def reference(self, instance) -> np.ndarray:
        if instance not in self._reference:
            self._reference[instance] = reference_solution(instance)
        return self._reference[instance]

    def exact_local_errors(self, state, instance) -> np.ndarray:
        exact = self.reference(instance)[list(state.nodes)]
        t = instance.time_of(state.nodes)
        return exact[1:] - self.stepper(instance).steps(exact[:-1], t[:-1], np.diff(t))

    def signed_error(self, state, instance) -> float:
        c = np.asarray(instance.goal, dtype=float)
        return float(c @ (self.reference(instance)[-1] - state.y[-1]))

    def weighted_local_errors(self, state, instance) -> np.ndarray:
        """Exact ``w_j = Lambda_{j+1}^T tau_{j+1}``; they sum to the signed QoI error."""
        return np.einsum("jm,jm->j", self.costate(state), self.exact_local_errors(state, instance))

    def objective(self, state, instance) -> float:
        if self.objective_kind == "abs_error":
            return abs(self.signed_error(state, instance))
        return float(np.abs(self.weighted_local_errors(state, instance)).sum())

    def native_metrics(self, state, instance) -> dict:
        return {"objective": self.objective(state, instance), "objective_kind": self.objective_kind,
                "goal_error_bound": float(np.abs(self.weighted_local_errors(state, instance)).sum()),
                "abs_qoi_error": abs(self.signed_error(state, instance)),
                "signed_qoi_error": self.signed_error(state, instance), "intervals": len(state.nodes) - 1}


# ---------------------------------------------------------------------------
# Policies for D4-0 (no learning). All deployable ones share the same tau_hat.
# ---------------------------------------------------------------------------

def _interval_scores(candidates, per_interval):
    return np.array([per_interval[c.payload["interval"]] for c in candidates])


def _estimate_cost(domain, state, candidates):
    return Cost(compute=2.0 * (len(state.nodes) - 1))


def _estimate_and_sweep_cost(domain, state, candidates):
    return Cost(compute=3.0 * (len(state.nodes) - 1))


def uniform_policy() -> Policy:
    """Fixed rule: bisect the longest interval, earliest first."""
    def score(domain, state, candidates, context):
        n = state.instance.n_fine
        return np.array([(b - a) - 1e-6 * a / n for a, b in (c.target for c in candidates)])
    return Policy("uniform", score)


def residual_policy() -> Policy:
    """Goal-agnostic: the largest estimated local error norm."""
    def score(domain, state, candidates, context):
        return _interval_scores(candidates, np.linalg.norm(domain.local_error_estimates(state), axis=1))
    return Policy("residual", score, decision_cost=_estimate_cost)


def goal_local_policy() -> Policy:
    """Goal projection without propagation: ``|c^T tau_hat_j|``."""
    def score(domain, state, candidates, context):
        c = np.asarray(state.instance.goal, dtype=float)
        return _interval_scores(candidates, np.abs(domain.local_error_estimates(state) @ c))
    return Policy("goal_local", score, decision_cost=_estimate_cost)


def adjoint_policy() -> Policy:
    """Dual-weighted residual with the discrete co-state (deployable): predicted bound reduction
    ``0.75 |Lambda_{j+1}^T tau_hat_j|`` (the constant does not change the ranking)."""
    def score(domain, state, candidates, context):
        weighted = np.einsum("jm,jm->j", domain.costate(state), domain.local_error_estimates(state))
        return _interval_scores(candidates, LOCAL_ERROR_REDUCTION * np.abs(weighted))
    return Policy("adjoint", score, decision_cost=_estimate_and_sweep_cost)


def exact_tau_adjoint_policy() -> Policy:
    """Diagnostic ceiling (privileged): the co-state with the *true* local errors."""
    def score(domain, state, candidates, context):
        weighted = domain.weighted_local_errors(state, context.instance)
        return _interval_scores(candidates, LOCAL_ERROR_REDUCTION * np.abs(weighted))
    return Policy("exact_tau_adjoint", score, deployable=False)


# ---- pass-based (batch) policies: one scoring pass, many refinements, one re-solve ----------

def _select_dorfler(scores: np.ndarray, theta: float) -> np.ndarray:
    """Smallest set of largest scores whose sum reaches ``theta`` of the total (at least one)."""
    order = np.argsort(-scores, kind="stable")
    total = float(scores.sum())
    if total <= 0.0:
        return order[:1]
    reached = np.cumsum(scores[order]) >= theta * total
    return order[: int(np.argmax(reached)) + 1]


def uniform_pass_policy() -> BatchPolicy:
    """Fixed rule, no scoring: refine every interval of the current maximum length (one full level)."""
    def select(domain, state, candidates, context):
        lengths = np.array([b - a for a, b in (c.target for c in candidates)])
        return np.flatnonzero(lengths == lengths.max())
    return BatchPolicy("uniform_pass", select)


def marking_policy(kind: str, theta: float) -> BatchPolicy:
    """Dorfler marking on ``kind`` scores: ``residual`` (``||tau_hat||``), ``goal_local`` (``|c^T tau_hat|``)
    or ``adjoint`` (``|Lambda^T tau_hat|``). Scoring is charged once per pass (2n steps, 3n with the sweep)."""
    if kind not in ("residual", "goal_local", "adjoint"):
        raise ValueError(f"unknown marking kind {kind!r}")
    if not 0.0 < theta <= 1.0:
        raise ValueError("theta must be in (0, 1]")

    def select(domain, state, candidates, context):
        tau = domain.local_error_estimates(state)
        if kind == "residual":
            per_interval = np.linalg.norm(tau, axis=1)
        elif kind == "goal_local":
            per_interval = np.abs(tau @ np.asarray(state.instance.goal, dtype=float))
        else:
            per_interval = np.abs(np.einsum("jm,jm->j", domain.costate(state), tau))
        scores = _interval_scores(candidates, per_interval)
        return _select_dorfler(scores, theta)

    cost = _estimate_and_sweep_cost if kind == "adjoint" else _estimate_cost
    return BatchPolicy(f"mark_{kind}_{theta:g}", select, decision_cost=cost)


def d4_policies() -> list[Policy]:
    from .runner import one_step_oracle, random_policy  # noqa: PLC0415

    return [uniform_policy(), random_policy(), residual_policy(), goal_local_policy(), adjoint_policy(),
            exact_tau_adjoint_policy(), one_step_oracle()]


# ---------------------------------------------------------------------------
# Instance family (analytic benchmark, frozen seeds)
# ---------------------------------------------------------------------------

def _unit(rng, m):
    v = rng.normal(size=m)
    return tuple(float(x) for x in v / np.linalg.norm(v))


FAMILIES = {
    # name: (pulse width range, amplitude scale, default max_depth)
    "smooth": ((0.01, 0.04), 1.0, 7),
    "sharp": ((0.001, 0.004), 10.0, 11),   # localised features: uniform resolution is expensive
    "sharper": ((0.0001, 0.0004), 100.0, 14),
}


def sample_family(name: str, seed: int, count: int, initial_intervals: int = 16) -> list["LinearODEInstance"]:
    """Instances of a named family (``smooth`` is the D4-0 family, ``sharp`` has 10x narrower pulses)."""
    widths, scale, depth = FAMILIES[name]
    return sample_instances(seed, count, initial_intervals, depth, pulse_width=widths, amplitude_scale=scale)


def sample_instances(seed: int, count: int, initial_intervals: int = 16, max_depth: int = 7,
                     pulse_width: tuple = (0.01, 0.04), amplitude_scale: float = 1.0) -> list[LinearODEInstance]:
    """A damped oscillator coupled to one scalar mode (decaying or growing), 2-4 forcing pulses.

    The defaults reproduce the D4-0 family exactly (same random stream, same values)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        omega = 2 * math.pi * rng.uniform(1.0, 4.0)
        zeta = rng.uniform(0.02, 0.3)
        growth = rng.uniform(-4.0, 1.5)
        coupling = rng.uniform(-0.5, 0.5)
        a = ((0.0, 1.0, 0.0), (-omega ** 2, -2 * zeta * omega, coupling), (coupling, 0.0, growth))
        pulses = tuple(
            Pulse(float(rng.uniform(0.05, 0.95)), float(rng.uniform(*pulse_width)),
                  float(rng.choice([-1, 1]) * rng.uniform(5.0, 40.0) * amplitude_scale), _unit(rng, 3))
            for _ in range(int(rng.integers(2, 5)))
        )
        out.append(LinearODEInstance(
            name=f"d4_s{seed}_{i:03d}", A=tuple(tuple(float(x) for x in row) for row in a),
            y0=tuple(float(x) for x in rng.normal(size=3)), goal=_unit(rng, 3), pulses=pulses,
            initial_intervals=initial_intervals, max_depth=max_depth,
        ))
    return out
