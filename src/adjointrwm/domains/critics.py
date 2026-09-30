"""D4-1: learned direct critics against learned co-state critics on the D4 benchmark (``docs/plans/d4-1-plan.md``).

Two components, as in the governing protocol §5.5:

    *co-state estimator*  ``w_hat(t_b, c)``: an MLP trained on the co-state vector ``Lambda(t_b; c)`` (supervision by
                          ``dJ/dstate``), evaluated independently of any value head;
    *value head*          an MLP predicting ``log10`` of the interval's gain from the P0 inputs (nine scalar features, the goal,
                          the unit directions of the two cheap error vectors) and, for the co-state arms, three co-state features
                          ``[log10|w.d|, log10|w.e|, log10||w||]``.

Arms (all share rows, the validation split and the head's batch order):

    ``direct``              head on the P0 inputs only (the information-equivalent direct baseline);
    ``costate_critic``      head on the P0 inputs plus features from the trained estimator (the deployable adjoint critic);
    ``costate_randomised``  as ``costate_critic`` with the estimator trained on co-state labels permuted across rows (control);
    ``teacher_feature``     head on the P0 inputs plus features from the exact co-state (privileged diagnostic, never a deployment).

Everything is NumPy, generated from frozen seeds, never real data. Labels (exact local errors, the discrete and continuous
co-state) are privileged and used for training only; policies read only P0 quantities (the teacher arm also reads the exact table).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from .highdim import (FEATURE_DIM, FEATURE_FLOOR, LABEL_FLOOR, LOG_FLOPS, ScorerModel, TinyMLP, _dorfler_policy, costate_table, difference_estimate,
                      flops_amortised_per_interval, flops_cheap_adjoint_per_interval, flops_mlp_forward, forcing_discrepancy, scorer_features,
                      train_scorer)
from .linear_ode import expm

ARMS = {
    "direct": {"features": "none", "estimator": False, "permute": False},
    "costate_critic": {"features": "estimator", "estimator": True, "permute": False},
    "costate_randomised": {"features": "estimator", "estimator": True, "permute": True},
    "teacher_feature": {"features": "teacher", "estimator": False, "permute": False},
}
COSTATE_FEATURES = 3
VECTOR_FLOOR = 1e-12        # floor on ``|Lambda|^2`` in the relative co-state loss


# ---------------------------------------------------------------------------
# Co-state operators and labels
# ---------------------------------------------------------------------------

_PROPAGATORS: dict = {}


def propagator(instance) -> np.ndarray:
    """``Phi_k = exp(A^T (T - t_k))`` at every node of the finest grid, ``[n_fine + 1, m, m]``; the continuous co-state of goal ``c`` is ``Phi_k c``.

    The recursion is the one of :func:`adjointrwm.domains.highdim.costate_table`, applied to the identity instead of one goal."""
    key = (instance.A, instance.n_fine, instance.T)
    if key not in _PROPAGATORS:
        m, n = instance.m, instance.n_fine
        phi = expm(instance.matrix().T * (instance.T / n))
        out = np.empty((n + 1, m, m))
        out[n] = np.eye(m)
        for k in range(n - 1, -1, -1):
            out[k] = phi @ out[k + 1]
        _PROPAGATORS[key] = out
    return _PROPAGATORS[key]


def continuous_costate(instance, node_index, goals, chunk: int = 2048) -> np.ndarray:
    """``Lambda(t_k; c)`` for row ``i`` at node ``node_index[i]`` with goal ``goals[i]``, ``[rows, m]``."""
    phi = propagator(instance)
    node_index, goals = np.asarray(node_index), np.asarray(goals, dtype=float)
    out = np.empty((len(node_index), instance.m))
    for start in range(0, len(node_index), chunk):
        block = slice(start, start + chunk)
        out[block] = np.einsum("rij,rj->ri", phi[node_index[block]], goals[block])
    return out


def discrete_costates(domain, instance, nodes, goals) -> np.ndarray:
    """``out[g, j] = Lambda_{j+1}`` of the mesh ``nodes`` for goal ``goals[g]``, ``[G, n, m]``: :meth:`domain.costate` for many goals at once."""
    stepper = domain.stepper(instance)
    t = instance.time_of(nodes)
    goals = np.asarray(goals, dtype=float)
    n = len(nodes) - 1
    out = np.empty((len(goals), n, goals.shape[1]))
    out[:, n - 1] = goals
    current = goals.T.copy()
    for j in range(n - 1, 0, -1):
        _, step_matrix = stepper.operators(t[j + 1] - t[j])
        current = step_matrix.T @ current
        out[:, j - 1] = current.T
    return out


def gain_labels(costates: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """``log10(|Lambda_{j+1}^T tau_j| + floor)`` per row (the same floor as D4-2's scorer labels)."""
    return np.log10(np.abs(np.einsum("rm,rm->r", costates, tau)) + LABEL_FLOOR)


def costate_features(w: np.ndarray, d: np.ndarray, e: np.ndarray) -> np.ndarray:
    """``[log10(|w.d| + floor), log10(|w.e| + floor), log10(||w|| + floor)]`` per row: the directional products and the size of a co-state."""
    return np.column_stack([np.log10(np.abs(np.einsum("nm,nm->n", w, d)) + LABEL_FLOOR), np.log10(np.abs(np.einsum("nm,nm->n", w, e)) + LABEL_FLOOR),
                            np.log10(np.linalg.norm(w, axis=1) + LABEL_FLOOR)])


# ---------------------------------------------------------------------------
# Training states and rows
# ---------------------------------------------------------------------------

@dataclass
class StateSample:
    """The privileged and P0 quantities of up to ``per_state`` intervals of one visited mesh (goal-free)."""

    instance_index: int
    state_id: int
    nodes: tuple
    intervals: np.ndarray       # [k] interval indices
    scalars: np.ndarray         # [k, FEATURE_DIM]
    d: np.ndarray               # [k, m] forcing discrepancy
    e: np.ndarray               # [k, m] solution-difference estimate
    tau: np.ndarray             # [k, m] exact local error (privileged)


def collect_states(domain, instances, policies, max_passes: int = 40, per_state: int = 64, seed: int = 0) -> list[StateSample]:
    """Meshes visited by ``policies`` (pass-based) on ``instances``, with up to ``per_state`` randomly chosen refinable intervals each."""
    rng = np.random.default_rng(seed)
    samples: list[StateSample] = []
    for g, instance in enumerate(instances):
        for policy in policies:
            state = domain.initial_state(instance)
            for _ in range(max_passes):
                candidates = domain.legal_candidates(state)
                if not candidates:
                    break
                idx = np.array([c.payload["interval"] for c in candidates])
                if len(idx) > per_state:
                    idx = np.sort(rng.choice(idx, size=per_state, replace=False))
                samples.append(StateSample(
                    g, len(samples), state.nodes, idx, scorer_features(instance, state)[idx], forcing_discrepancy(instance, state)[idx],
                    difference_estimate(instance, state)[idx], domain.exact_local_errors(state, instance)[idx]))
                picked = [int(i) for i in policy.select(domain, state, candidates, None)]
                state = domain.apply_batch(state, [candidates[i] for i in picked])
    return samples


ROW_FIELDS = ("scalars", "goal", "d", "e", "t_right", "lam", "y", "group", "state_id")


@dataclass
class Rows:
    """Training rows: P0 inputs, the goal, and the two labels."""

    scalars: np.ndarray         # [N, FEATURE_DIM]
    goal: np.ndarray            # [N, m]
    d: np.ndarray               # [N, m]
    e: np.ndarray               # [N, m]
    t_right: np.ndarray         # [N] time of the interval's right node as a fraction of T
    lam: np.ndarray             # [N, m] continuous co-state at the right node (estimator label)
    y: np.ndarray               # [N] log10 gain label
    group: np.ndarray           # [N] train-instance index
    state_id: np.ndarray        # [N]

    def __len__(self) -> int:
        return len(self.y)

    def take(self, index) -> "Rows":
        return Rows(*(getattr(self, f)[index] for f in ROW_FIELDS))


def random_unit_goals(rng: np.random.Generator, count: int, m: int) -> np.ndarray:
    v = rng.normal(size=(count, m))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def build_rows(domain, samples: list[StateSample], instances, seed: int, goals_per_state: int = 4, max_rows: int = 60_000) -> Rows:
    """Rows for one seed: each sampled interval takes one of ``goals_per_state`` fresh random unit goals of its state.

    At most ``max_rows`` rows are kept (a seeded random subset)."""
    rng = np.random.default_rng(seed)
    total = sum(len(s.intervals) for s in samples)
    keep = np.zeros(total, dtype=bool)
    keep[rng.permutation(total)[:max_rows]] = True
    offset = 0
    parts = {f: [] for f in ROW_FIELDS}
    for s in samples:
        k = len(s.intervals)
        mask = keep[offset:offset + k]
        offset += k
        instance = instances[s.instance_index]
        goals = random_unit_goals(rng, goals_per_state, instance.m)     # drawn for every state so the stream does not depend on the subset
        if not mask.any():
            continue
        which = np.arange(k) % goals_per_state
        sel = np.flatnonzero(mask)
        row_goals = goals[which[sel]]
        costate = discrete_costates(domain, instance, s.nodes, goals)[which[sel], s.intervals[sel]]
        right = np.asarray(s.nodes)[s.intervals[sel] + 1]
        parts["scalars"].append(s.scalars[sel])
        parts["goal"].append(row_goals)
        parts["d"].append(s.d[sel])
        parts["e"].append(s.e[sel])
        parts["t_right"].append(right / instance.n_fine)
        parts["lam"].append(continuous_costate(instance, right, row_goals))
        parts["y"].append(gain_labels(costate, s.tau[sel]))
        parts["group"].append(np.full(len(sel), s.instance_index))
        parts["state_id"].append(np.full(len(sel), s.state_id))
    return Rows(**{f: np.concatenate(v) for f, v in parts.items()})


def split_by_group(group: np.ndarray, seed: int, val_fraction: float = 0.2) -> np.ndarray:
    """Boolean validation mask made of whole train instances. It is the split :func:`~adjointrwm.domains.highdim.train_scorer` makes for the
    same groups and seed, so the estimator and the value heads of one seed validate on the same instances."""
    unique = np.unique(group)
    chosen = np.random.default_rng(seed).permutation(unique)[: max(1, int(round(val_fraction * len(unique))))]
    return np.isin(group, chosen)


# ---------------------------------------------------------------------------
# Co-state estimator
# ---------------------------------------------------------------------------

def mlp_backward(mlp: TinyMLP, acts: list, dout: np.ndarray):
    """Gradients of weights and biases given the gradient ``dout`` of the loss with respect to the network output."""
    gw, gb = [None] * len(mlp.weights), [None] * len(mlp.weights)
    delta = dout
    for i in range(len(mlp.weights) - 1, -1, -1):
        gw[i] = acts[i].T @ delta
        gb[i] = delta.sum(axis=0)
        if i > 0:
            delta = (delta @ mlp.weights[i].T) * (1.0 - acts[i] ** 2)
    return gw, gb


def relative_costate_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean over rows of ``||pred - target||^2 / ||target||^2`` (1 is what predicting zero scores)."""
    return float(np.mean(((pred - target) ** 2).sum(axis=1) / ((target ** 2).sum(axis=1) + VECTOR_FLOOR)))


@dataclass
class Estimator:
    """``w_hat(t_b, c)``: an MLP on ``[t_b / T, c]`` (standardised) whose output, times ``lam_scale``, estimates ``Lambda(t_b; c)``."""

    mlp: TinyMLP
    x_mean: np.ndarray
    x_std: np.ndarray
    lam_scale: float
    m: int
    report: dict = field(default_factory=dict)

    @property
    def dims(self) -> tuple:
        return self.mlp.sizes

    def predict(self, t_right: np.ndarray, goal: np.ndarray) -> np.ndarray:
        x = (np.column_stack([t_right, goal]) - self.x_mean) / self.x_std
        return self.lam_scale * self.mlp.forward(x)

    def to_json(self) -> dict:
        return {"dims": list(self.dims), "params": self.mlp.to_json(), "x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(),
                "lam_scale": self.lam_scale, "m": self.m, "report": self.report}

    @classmethod
    def from_json(cls, blob: dict) -> "Estimator":
        return cls(TinyMLP(tuple(blob["dims"]), params=[(w, b) for w, b in blob["params"]]), np.asarray(blob["x_mean"]), np.asarray(blob["x_std"]),
                   blob["lam_scale"], blob["m"], blob.get("report", {}))


def estimator_loss_and_grads(mlp: TinyMLP, x: np.ndarray, target: np.ndarray):
    """Relative squared error ``mean ||out - target||^2 / ||target||^2`` and its gradients."""
    out, acts = mlp.forward(x, keep=True)
    n = len(x)
    norm2 = (target ** 2).sum(axis=1) + VECTOR_FLOOR
    diff = out - target
    loss = float(np.mean((diff ** 2).sum(axis=1) / norm2))
    gw, gb = mlp_backward(mlp, acts, 2.0 * diff / (n * norm2[:, None]))
    return loss, gw, gb


def train_estimator(train: Rows, val: Rows, hidden: int, seed: int, permute: bool = False, max_epochs: int = 200, patience: int = 20, lr: float = 3e-3,
                    batch_size: int = 256) -> Estimator:
    """Adam on the relative co-state loss; the best epoch by the same loss on the validation rows. With ``permute`` the labels of the training
    rows (and, for early stopping, of the validation rows) are permuted across rows, which keeps their marginals and breaks their alignment."""
    m = train.goal.shape[1]
    x_raw = np.column_stack([train.t_right, train.goal])
    x_mean, x_std = x_raw.mean(axis=0), x_raw.std(axis=0) + 1e-9
    xt, xv = (x_raw - x_mean) / x_std, (np.column_stack([val.t_right, val.goal]) - x_mean) / x_std
    lam_scale = float(np.linalg.norm(train.lam, axis=1).mean())
    lt, lv = train.lam / lam_scale, val.lam / lam_scale
    if permute:
        lt = lt[np.random.default_rng(10_000 * seed + 9).permutation(len(lt))]
        lv = lv[np.random.default_rng(10_000 * seed + 10).permutation(len(lv))]
    mlp = TinyMLP((xt.shape[1], hidden, hidden, m), np.random.default_rng(10_000 * seed + 11))
    params = mlp.weights + mlp.biases
    m1, m2 = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
    order_rng = np.random.default_rng(10_000 * seed + 12)
    best, best_state, wait, step, epochs = math.inf, None, 0, 0, 0
    for _ in range(max_epochs):
        order = order_rng.permutation(len(xt))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            _, gw, gb = estimator_loss_and_grads(mlp, xt[idx], lt[idx])
            step += 1
            for k, (p, g) in enumerate(zip(params, gw + gb)):
                m1[k] = 0.9 * m1[k] + 0.1 * g
                m2[k] = 0.999 * m2[k] + 0.001 * g * g
                p -= lr * (m1[k] / (1 - 0.9 ** step)) / (np.sqrt(m2[k] / (1 - 0.999 ** step)) + 1e-8)
        epochs += 1
        loss = relative_costate_error(mlp.forward(xv), lv)
        if loss < best - 1e-6:
            best, wait = loss, 0
            best_state = ([w.copy() for w in mlp.weights], [b.copy() for b in mlp.biases])
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        mlp.weights, mlp.biases = best_state
    est = Estimator(mlp, x_mean, x_std, lam_scale, m)
    forward = flops_mlp_forward(mlp.sizes)
    est.report = {"seed": seed, "permuted_labels": bool(permute), "dims": list(mlp.sizes), "n_train": int(len(xt)), "n_val": int(len(xv)), "epochs_run": epochs,
                  "hit_max_epochs": bool(epochs >= max_epochs), "val_relative_error_on_training_labels": best,
                  "val_relative_error_against_true_costate": relative_costate_error(est.predict(val.t_right, val.goal), val.lam),
                  "train_relative_error_against_true_costate": relative_costate_error(est.predict(train.t_right, train.goal), train.lam),
                  "training_flops": int(3 * forward * len(xt) * epochs + forward * len(xv) * epochs)}
    return est


# ---------------------------------------------------------------------------
# Value head and the composite critic
# ---------------------------------------------------------------------------

def assemble_inputs(scalars: np.ndarray, goal: np.ndarray, d: np.ndarray, e: np.ndarray) -> np.ndarray:
    """``[N, FEATURE_DIM + 3m]``: scalar features, goal, unit direction of ``d``, unit direction of ``e``."""
    ud = d / (np.linalg.norm(d, axis=1, keepdims=True) + FEATURE_FLOOR)
    ue = e / (np.linalg.norm(e, axis=1, keepdims=True) + FEATURE_FLOOR)
    return np.concatenate([scalars, goal, ud, ue], axis=1)


def param_count(n_in: int, hidden: int, n_out: int = 1) -> int:
    return n_in * hidden + hidden + hidden * hidden + hidden + hidden * n_out + n_out


def match_direct_hidden(n_in_direct: int, n_in_costate: int, costate_hidden: int, estimator_dims: tuple) -> int:
    """Width of the direct head whose parameter count is closest to the co-state head's plus the estimator's (width adjustment, never withheld inputs)."""
    target = param_count(n_in_costate, costate_hidden) + sum(i * o + o for i, o in zip(estimator_dims[:-1], estimator_dims[1:]))
    return min(range(1, 513), key=lambda h: (abs(param_count(n_in_direct, h) - target), h))


@dataclass
class Critic:
    arm: str
    head: ScorerModel
    m: int
    estimator: Estimator | None = None
    report: dict = field(default_factory=dict)

    @property
    def features(self) -> str:
        return ARMS[self.arm]["features"]

    def weights(self, goal, t_right, teacher_w=None) -> np.ndarray | None:
        """The co-state the features are built from: the estimator's output, the exact co-state (teacher) or ``None``."""
        if self.features == "estimator":
            return self.estimator.predict(t_right, goal)
        if self.features == "teacher":
            if teacher_w is None:
                raise ValueError("the teacher arm needs the exact co-state")
            return teacher_w
        return None

    def predict_log10(self, scalars, goal, d, e, t_right, teacher_w=None) -> np.ndarray:
        x = assemble_inputs(scalars, goal, d, e)
        w = self.weights(goal, t_right, teacher_w)
        if w is not None:
            x = np.concatenate([x, costate_features(w, d, e)], axis=1)
        return self.head.predict_log10(x)

    def to_json(self) -> dict:
        return {"arm": self.arm, "m": self.m, "head": {"dims": list(self.head.dims), "params": self.head.mlp.to_json(), "mean": self.head.mean.tolist(),
                                                     "std": self.head.std.tolist(), "report": self.head.report},
                "estimator": self.estimator.to_json() if self.estimator is not None else None, "report": self.report}

    @classmethod
    def from_json(cls, blob: dict) -> "Critic":
        h = blob["head"]
        head = ScorerModel(TinyMLP(tuple(h["dims"]), params=[(w, b) for w, b in h["params"]]), np.asarray(h["mean"]), np.asarray(h["std"]), h.get("report", {}))
        return cls(blob["arm"], head, blob["m"], Estimator.from_json(blob["estimator"]) if blob["estimator"] else None, blob.get("report", {}))


def within_state_spearman(pred: np.ndarray, y: np.ndarray, state_id: np.ndarray) -> float:
    """Mean over states (with at least 3 rows) of the Spearman rank correlation between ``pred`` and ``y``."""
    def rank(v):
        return np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)

    values = []
    for sid in np.unique(state_id):
        idx = np.flatnonzero(state_id == sid)
        if len(idx) < 3:
            continue
        a, b = rank(pred[idx]), rank(y[idx])
        if a.std() > 0 and b.std() > 0:
            values.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def head_inputs(arm: str, rows: Rows, estimator: Estimator | None) -> np.ndarray:
    x = assemble_inputs(rows.scalars, rows.goal, rows.d, rows.e)
    features = ARMS[arm]["features"]
    if features == "none":
        return x
    w = estimator.predict(rows.t_right, rows.goal) if features == "estimator" else rows.lam
    return np.concatenate([x, costate_features(w, rows.d, rows.e)], axis=1)


def train_critic(arm: str, rows: Rows, hidden: int, seed: int, estimator: Estimator | None = None, max_epochs: int = 40, patience: int = 6) -> Critic:
    """The value head of ``arm`` on all ``rows`` (the train/validation split is by instance, made inside :func:`train_scorer` with ``seed``, and equals
    :func:`split_by_group`). The estimator arms need the trained ``estimator`` (:func:`train_estimator` on the non-validation rows)."""
    spec = ARMS[arm]
    if spec["estimator"] and estimator is None:
        raise ValueError(f"{arm} needs a trained estimator")
    x = head_inputs(arm, rows, estimator)
    head = train_scorer(x, rows.y, rows.group, seed=seed, hidden=(hidden, hidden), max_epochs=max_epochs, patience=patience, max_samples=len(rows))
    mask = split_by_group(rows.group, seed)
    pred = head.predict_log10(x[mask])
    critic = Critic(arm, head, rows.goal.shape[1], estimator if spec["estimator"] else None)
    critic.report = {"arm": arm, "seed": seed, "head_dims": list(head.dims), "head_parameters": param_count(x.shape[1], hidden),
                     "estimator_parameters": sum(w.size + b.size for w, b in zip(estimator.mlp.weights, estimator.mlp.biases)) if spec["estimator"] else 0,
                     "val_spearman_within_state": within_state_spearman(pred, rows.y[mask], rows.state_id[mask]), **head.report}
    return critic


# ---------------------------------------------------------------------------
# Policies and prices
# ---------------------------------------------------------------------------

def flops_critic_per_interval(instance, critic: Critic) -> int:
    """Features (as the amortised scorer's), the unit directions and standardisation of the remaining inputs, the head, and for the co-state arms the
    estimator (when there is one) and the three co-state features (two dot products, a norm, three logarithms)."""
    m = instance.m
    n_in = critic.head.dims[0]
    total = flops_amortised_per_interval(instance, ()) + 2 * m + 2 * (n_in - FEATURE_DIM) + flops_mlp_forward(critic.head.dims)
    if critic.features != "none":
        total += 4 * m + (2 * m + 1) + 3 * LOG_FLOPS + 6
    if critic.estimator is not None:
        total += 2 * (1 + m) + flops_mlp_forward(critic.estimator.dims)
    return total


def lookup_price_scale(instance, critic: Critic) -> float:
    """Ratio of the ``cheap_adjoint`` price per interval (features plus two dot products) to this critic's: the hypothetical ideal-amortisation price."""
    return flops_cheap_adjoint_per_interval(instance) / flops_critic_per_interval(instance, critic)


def critic_per_interval(critic: Critic, instance, state) -> np.ndarray:
    scalars, d, e = scorer_features(instance, state), forcing_discrepancy(instance, state), difference_estimate(instance, state)
    goal = np.tile(np.asarray(instance.goal, dtype=float), (len(d), 1))
    right = np.asarray(state.nodes[1:])
    teacher = costate_table(instance)[right] if critic.features == "teacher" else None
    return 10.0 ** critic.predict_log10(scalars, goal, d, e, right / instance.n_fine, teacher)


def critic_policy(critic: Critic, theta: float, name: str | None = None):
    """Dorfler marking on the critic's score at its real price."""
    return _dorfler_policy(name or f"mark_{critic.arm}", lambda s: critic_per_interval(critic, s.instance, s), theta,
                           lambda inst: flops_critic_per_interval(inst, critic))


# ---------------------------------------------------------------------------
# Running critics at the real and the lookup price
# ---------------------------------------------------------------------------

def price_scale(instance, critic: Critic, price: str) -> float:
    """Decision-cost scale of a price scenario: 1 for the real ledger, the ``cheap_adjoint`` ratio for the hypothetical lookup price."""
    if price not in ("real", "lookup"):
        raise ValueError("price must be 'real' or 'lookup'")
    return 1.0 if price == "real" else lookup_price_scale(instance, critic)


def evaluate_critic(domain, instances, critic: Critic, theta: float, price: str, targets, cap_of, max_passes: int, name: str, methods=("loglog",)):
    """Compute to reach each target for one critic at one price: the rows of :func:`~adjointrwm.domains.runner.evaluate_work_precision`, policy ``name``."""
    import pandas as pd  # noqa: PLC0415

    from .runner import evaluate_work_precision  # noqa: PLC0415

    policy = dataclasses.replace(critic_policy(critic, theta), name=name)
    parts = [evaluate_work_precision(domain, [inst], [policy], targets, compute_cap=cap_of(inst), max_steps=max_passes, random_draws=1,
                                     decision_scales=(price_scale(inst, critic, price),), methods=methods) for inst in instances]
    return pd.concat(parts, ignore_index=True)


def tune_critic_theta(domain, instances, critic: Critic, thetas, price: str, targets, cap_of, max_passes: int, method: str = "loglog") -> dict:
    """Geometric-mean compute (over instances and targets) of the critic's policy for each ``theta`` at ``price``; tuning instances only."""
    out = {}
    for theta in thetas:
        frame = evaluate_critic(domain, instances, critic, theta, price, targets, cap_of, max_passes, "tuned", methods=(method,))
        caps = frame["instance"].map({inst.name: cap_of(inst) for inst in instances}).to_numpy(dtype=float)
        out[theta] = float(np.exp(np.log(np.minimum(frame["compute"].to_numpy(dtype=float), caps)).mean()))
    return out


def log_compute_matrix(frame, names: list[str], instance_names: list[str], target: str, method: str, cap: float) -> np.ndarray:
    """``[len(names), len(instance_names)]`` of ``log(min(compute, cap))`` for the given policy names (unreached targets count as the cap)."""
    out = np.empty((len(names), len(instance_names)))
    for k, name in enumerate(names):
        block = frame[(frame["policy"] == name) & (frame["target"] == target) & (frame["method"] == method)].set_index("instance")["compute"]
        out[k] = np.log(np.minimum(block.loc[instance_names].to_numpy(dtype=float), cap))
    return out


def exit_class(flags: dict) -> str:
    """The plan's §7 exit class of one cell from booleans ``R0, RC1, RC1L, RC2L, RC3, RC3L, RTL, estimator_ok, saturated``."""
    if not flags["saturated"]:
        return "inconclusive: the direct baseline was not shown saturated"
    if flags["R0"] and flags["RC1"] and flags["estimator_ok"]:
        return "deployable adjoint contribution (analytic domain only)" if flags["RC3"] else "effect not attributable to co-state alignment"
    if flags["RC1L"] and flags["estimator_ok"]:
        return "value conditional on a cheap critic" if flags["RC3L"] else "effect not attributable to co-state alignment"
    if flags["RTL"]:
        return "teacher-only value"
    if flags["RC2L"]:
        return "direct utility sufficient"
    return "inconclusive: precision"


# ---------------------------------------------------------------------------
# Statistics for the frozen rules
# ---------------------------------------------------------------------------

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def two_stage_bootstrap(diff: np.ndarray, num_resamples: int = 10_000, seed: int = 0) -> dict:
    """95 % interval of the mean of ``diff [seeds, instances]`` (log compute ratios): resample seeds with replacement, then instances within
    each drawn seed. Also the t-interval over the seed means as a check."""
    diff = np.asarray(diff, dtype=float)
    n_seeds, n_inst = diff.shape
    rng = np.random.default_rng(seed)
    seed_idx = rng.integers(n_seeds, size=(num_resamples, n_seeds))
    inst_idx = rng.integers(n_inst, size=(num_resamples, n_seeds, n_inst))
    means = diff[seed_idx[:, :, None], inst_idx].mean(axis=(1, 2))
    lo, hi = np.percentile(means, [2.5, 97.5])
    seed_means = diff.mean(axis=1)
    t_half = _T95[n_seeds - 1] * seed_means.std(ddof=1) / math.sqrt(n_seeds) if n_seeds >= 2 else float("nan")
    return {"estimate": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi), "ratio_of_geometric_means": float(np.exp(diff.mean())),
            "t_ci_low": float(seed_means.mean() - t_half), "t_ci_high": float(seed_means.mean() + t_half),
            "per_seed": [float(v) for v in seed_means], "seeds_favouring_a": int((seed_means < 0).sum())}


def two_adjacent(flags) -> bool:
    return any(flags[i] and flags[i + 1] for i in range(len(flags) - 1))


def within_margin(ci_rows: list[dict], margin: float) -> bool:
    """Every target's 95 % interval lies inside ``(-margin, margin)``."""
    return all(r["ci_low"] > -margin and r["ci_high"] < margin for r in ci_rows)
