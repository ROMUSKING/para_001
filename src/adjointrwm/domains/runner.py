"""Run allocation policies on any :class:`~adjointrwm.domains.base.AllocationDomain`.

A policy scores the legal candidates of the current state (higher = better predicted net
benefit). The runner takes the best candidate, or stops when a stopping policy's best score is
not positive, and records the objective and the resource ledger after every step.

**Privilege is enforced by construction:** deployable policies get a :class:`PolicyContext`
whose ``instance`` is ``None``, so they can only use the P0 observation and the state.
Privileged references (the oracles, exact teachers) set ``deployable = False`` and receive the
instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .base import AllocationDomain, Candidate, Cost


@dataclass
class PolicyContext:
    observation: Any
    instance: Any = None          # only for privileged (non-deployable) policies
    rng: np.random.Generator | None = None


@dataclass
class Policy:
    """A named scorer. ``score(domain, state, candidates, context) -> array`` (higher is better).

    ``decision_cost(domain, state, candidates) -> Cost`` is what computing the scores costs
    (for example the extra solver steps of a local-error estimate or a backward co-state sweep);
    it goes into the compute ledger next to the cost of the chosen actions.
    """

    name: str
    score: Callable
    deployable: bool = True
    decision_cost: Callable = field(default=lambda domain, state, candidates: Cost())


@dataclass
class BatchPolicy:
    """A pass-based policy: one scoring pass, then *several* candidates applied together.

    ``select(domain, state, candidates, context) -> indices`` returns the positions in
    ``candidates`` to apply in this pass. ``decision_cost`` is charged once per pass. The
    domain's ``batch_cost`` charges the actions (one re-solve for the whole batch, not one each).
    """

    name: str
    select: Callable
    deployable: bool = True
    decision_cost: Callable = field(default=lambda domain, state, candidates: Cost())


@dataclass
class Trace:
    policy: str
    deployable: bool
    objective: list            # J after 0, 1, ..., max_steps steps (padded after a stop)
    action_cost: list          # cumulative Cost of executed actions, same length
    decision_cost: list        # cumulative Cost of computing scores, same length
    decisions: list            # candidate ids, one per executed step
    stopped_at: int | None


def run_policy(
    domain: AllocationDomain,
    instance,
    policy: Policy,
    max_steps: int,
    allow_stop: bool = False,
    seed: int = 0,
    compute_budget: float | None = None,
    decision_scale: float = 1.0,
) -> Trace:
    """Run ``policy`` for up to ``max_steps`` actions.

    With ``compute_budget`` set, a step is taken only if the cumulative *compute* ledger (actions
    plus the cost of scoring) stays within the budget after it; otherwise the run ends with
    ``stopped_at`` set. This is how equal-compute comparisons are made: cheap policies simply
    get more actions. ``decision_scale`` multiplies the scoring cost in that budget check only
    (a what-if for cheaper scoring; the recorded ledger is never scaled).
    """
    state = domain.initial_state(instance)
    context = PolicyContext(
        observation=domain.observation(instance),
        instance=None if policy.deployable else instance,
        rng=np.random.default_rng(seed),
    )
    objective = [float(domain.objective(state, instance))]
    action_cost, decision_cost = [Cost()], [Cost()]
    decisions, stopped_at = [], None
    for step in range(max_steps):
        candidates = domain.legal_candidates(state)
        if not candidates:
            stopped_at = step
            break
        scores = np.asarray(policy.score(domain, state, candidates, context), dtype=float)
        if scores.shape != (len(candidates),):
            raise ValueError(f"{policy.name} returned {scores.shape} scores for {len(candidates)} candidates")
        spent = policy.decision_cost(domain, state, candidates)
        best = int(np.argmax(scores))
        if allow_stop and scores[best] <= 0.0:
            decision_cost[-1] = decision_cost[-1] + spent
            stopped_at = step
            break
        chosen = candidates[best]
        if compute_budget is not None:
            total = action_cost[-1].compute + decision_scale * (decision_cost[-1].compute + spent.compute) + chosen.cost.compute
            if total > compute_budget:
                stopped_at = step
                break
        state = domain.apply(state, chosen)
        decisions.append(chosen.id)
        objective.append(float(domain.objective(state, instance)))
        action_cost.append(action_cost[-1] + chosen.cost)
        decision_cost.append(decision_cost[-1] + spent)
    while len(objective) < max_steps + 1:  # budget unused after stopping: J stays put
        objective.append(objective[-1])
        action_cost.append(action_cost[-1])
        decision_cost.append(decision_cost[-1])
    return Trace(policy.name, policy.deployable, objective, action_cost, decision_cost, decisions, stopped_at)


def run_batch_policy(
    domain: AllocationDomain,
    instance,
    policy: BatchPolicy,
    max_passes: int,
    seed: int = 0,
    compute_budget: float | None = None,
    decision_scale: float = 1.0,
) -> Trace:
    """Run a pass-based policy. Trace entry ``i`` is the state after ``i`` passes."""
    state = domain.initial_state(instance)
    context = PolicyContext(observation=domain.observation(instance), instance=None if policy.deployable else instance,
                            rng=np.random.default_rng(seed))
    objective = [float(domain.objective(state, instance))]
    action_cost, decision_cost = [Cost()], [Cost()]
    decisions, stopped_at = [], None
    for step in range(max_passes):
        candidates = domain.legal_candidates(state)
        if not candidates:
            stopped_at = step
            break
        picked = [int(i) for i in policy.select(domain, state, candidates, context)]
        if not picked or len(set(picked)) != len(picked) or not all(0 <= i < len(candidates) for i in picked):
            raise ValueError(f"{policy.name} returned an invalid selection {picked}")
        chosen = [candidates[i] for i in picked]
        spent = policy.decision_cost(domain, state, candidates)
        batch = domain.batch_cost(state, chosen)
        if compute_budget is not None and action_cost[-1].compute + batch.compute + decision_scale * (decision_cost[-1].compute + spent.compute) > compute_budget:
            stopped_at = step
            break
        state = domain.apply_batch(state, chosen)
        decisions.append([c.id for c in chosen])
        objective.append(float(domain.objective(state, instance)))
        action_cost.append(action_cost[-1] + batch)
        decision_cost.append(decision_cost[-1] + spent)
    while len(objective) < max_passes + 1:
        objective.append(objective[-1])
        action_cost.append(action_cost[-1])
        decision_cost.append(decision_cost[-1])
    return Trace(policy.name, policy.deployable, objective, action_cost, decision_cost, decisions, stopped_at)


def run_any(domain, instance, policy, max_steps: int, seed: int = 0, compute_budget: float | None = None,
            decision_scale: float = 1.0) -> Trace:
    if isinstance(policy, BatchPolicy):
        return run_batch_policy(domain, instance, policy, max_steps, seed=seed, compute_budget=compute_budget,
                                decision_scale=decision_scale)
    return run_policy(domain, instance, policy, max_steps, seed=seed, compute_budget=compute_budget, decision_scale=decision_scale)


def compute_to_target(trace: Trace, target: float, decision_scale: float = 1.0) -> float:
    """Total compute at the first step whose objective is at most ``target``; ``inf`` if never.

    ``decision_scale`` re-prices scoring (``1`` = as recorded, ``0`` = scoring is free); it is a
    what-if computed from the same trace, since decisions never depend on costs.
    """
    for j, a, d in zip(trace.objective, trace.action_cost, trace.decision_cost):
        if j <= target:
            return float(a.compute + decision_scale * d.compute)
    return float("inf")


INTERPOLATION_METHODS = ("staircase", "semilog", "loglog")


def interpolated_compute_to_target(trace: Trace, target: float, decision_scale: float = 1.0, method: str = "semilog") -> float:
    """Like :func:`compute_to_target`, but interpolated between the last step above the target and the first below.

    Pass-based policies can stop only at whole passes, so their compute-to-target is quantised
    (``uniform_pass`` doubles the grid per pass, i.e. up to a factor 2). Interpolation prices a partial
    pass as a proportional part of a full one, sharing the interval in ``log(objective)``. It is not
    realisable; it is applied symmetrically to every policy. Objectives must be positive.

    ``method='semilog'``: compute is linear in ``log(objective)`` between the two steps; for a power-law
    error this overstates a fractional pass's compute by up to 6.1 % for a doubling pass.
    ``method='loglog'``: ``log(compute)`` is linear in ``log(objective)``, exact for a power law. Where the
    earlier step cost nothing (``log 0``), it falls back to the semilog rule.
    """
    if method not in ("semilog", "loglog"):
        raise ValueError(f"method must be 'semilog' or 'loglog', not {method!r}")
    spent = [a.compute + decision_scale * d.compute for a, d in zip(trace.action_cost, trace.decision_cost)]
    for i, j in enumerate(trace.objective):
        if j <= target:
            if i == 0:
                return float(spent[0])
            j_prev = trace.objective[i - 1]
            if not (j_prev > target >= j > 0):
                return float(spent[i])
            share = (np.log(j_prev) - np.log(target)) / (np.log(j_prev) - np.log(j))
            if method == "loglog" and spent[i - 1] > 0 and spent[i] > 0:
                return float(np.exp(np.log(spent[i - 1]) + share * (np.log(spent[i]) - np.log(spent[i - 1]))))
            return float(spent[i - 1] + share * (spent[i] - spent[i - 1]))
    return float("inf")


def compute_by_method(trace: Trace, target: float, decision_scale: float, method: str) -> float:
    """Compute to reach ``target`` under one of :data:`INTERPOLATION_METHODS`."""
    if method == "staircase":
        return compute_to_target(trace, target, decision_scale=decision_scale)
    return interpolated_compute_to_target(trace, target, decision_scale=decision_scale, method=method)


def evaluate_work_precision(
    domain: AllocationDomain,
    instances: Sequence,
    policies: Sequence,
    targets: Callable,
    compute_cap: float,
    max_steps: int,
    random_draws: int = 8,
    instance_id: Callable = lambda inst: getattr(inst, "name", repr(inst)),
    decision_scales: Sequence[float] = (1.0,),
    interpolate: bool = False,
    methods: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute needed by each deployable policy to reach per-instance objective targets.

    ``targets(domain, instance) -> {label: value}`` is evaluation-only (it may read privileged
    quantities). A target not reached within ``compute_cap`` is recorded as ``inf`` (censored).
    Each policy runs once, under the most permissive scoring price in ``decision_scales``; the
    other prices are read off the same trace (decisions do not depend on costs). Scale ``1`` is
    the real ledger; smaller scales are what-ifs and must be labelled as such. ``interpolate=True``
    uses :func:`interpolated_compute_to_target` (a robustness check, not a realisable cost).

    ``methods`` (a subset of :data:`INTERPOLATION_METHODS`) prices every target under each method from the
    *same* traces and adds a ``method`` column, so the primary analysis and its sensitivities cost one run;
    it overrides ``interpolate``. Filter on ``method`` before summarising.
    """
    if methods is not None and not set(methods) <= set(INTERPOLATION_METHODS):
        raise ValueError(f"methods must be a subset of {INTERPOLATION_METHODS}")
    rows = []
    cheapest = min(decision_scales)
    for instance in instances:
        wanted = targets(domain, instance)
        for policy in policies:
            if not policy.deployable:
                continue
            for draw in range(random_draws if policy.name == "random" else 1):
                trace = run_any(domain, instance, policy, max_steps, seed=draw, compute_budget=compute_cap, decision_scale=cheapest)
                for scale in decision_scales:
                    for label, value in wanted.items():
                        for method in (methods if methods is not None else (None,)):
                            if method is None:
                                c = (interpolated_compute_to_target if interpolate else compute_to_target)(trace, value, decision_scale=scale)
                            else:
                                c = compute_by_method(trace, value, scale, method)
                            row = {"instance": instance_id(instance), "policy": policy.name, "draw": draw, "target": label,
                                   "target_value": float(value), "decision_scale": float(scale),
                                   "compute": c if c <= compute_cap else float("inf")}
                            if method is not None:
                                row["method"] = method
                            rows.append(row)
    return pd.DataFrame(rows)


def objective_at_compute(trace: Trace, level: float) -> tuple[float, int, float]:
    """``(objective, actions taken, compute spent)`` after the last step within ``level`` compute."""
    spent = [a.compute + d.compute for a, d in zip(trace.action_cost, trace.decision_cost)]
    step = max(i for i, c in enumerate(spent) if c <= level)  # spent is non-decreasing; index 0 costs 0
    actions = len(trace.decisions) if step >= len(trace.decisions) else step
    return trace.objective[step], actions, spent[step]


def evaluate_at_compute(
    domain: AllocationDomain,
    instances: Sequence,
    policies: Sequence[Policy],
    levels: Sequence[float],
    max_steps: int,
    random_draws: int = 8,
    instance_id: Callable = lambda inst: getattr(inst, "name", repr(inst)),
) -> pd.DataFrame:
    """Objective of each *deployable* policy at fixed total-compute levels (equal-compute comparison).

    Privileged policies (oracles, exact teachers) are skipped: their scoring cost is not
    charged here, so a compute-matched number for them would not be honest.
    ``max_steps`` only caps the number of actions (a safety bound, not a budget).
    """
    rows = []
    for instance in instances:
        for policy in policies:
            if not policy.deployable:
                continue
            for draw in range(random_draws if policy.name == "random" else 1):
                trace = run_any(domain, instance, policy, max_steps, seed=draw, compute_budget=max(levels))
                for level in levels:
                    objective, actions, spent = objective_at_compute(trace, level)
                    rows.append({"instance": instance_id(instance), "policy": policy.name, "draw": draw, "compute_level": float(level),
                                 "objective": objective, "actions": actions, "compute_spent": spent})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Generic references
# ---------------------------------------------------------------------------

def _one_step_scores(domain, state, candidates, context):
    current = domain.objective(state, context.instance)
    return np.array([current - domain.objective(domain.apply(state, c), context.instance) for c in candidates])


def one_step_oracle() -> Policy:
    """Greedy with exact one-step gains (privileged). Its curve is labelled one-step-oracle."""
    return Policy("one_step_oracle", _one_step_scores, deployable=False)


def random_policy() -> Policy:
    return Policy("random", lambda domain, state, candidates, context: context.rng.random(len(candidates)))


def exhaustive_oracle_curve(domain: AllocationDomain, instance, max_steps: int, state_key: Callable) -> list:
    """Best objective reachable with exactly ``b`` actions, for ``b = 0..max_steps`` (tiny cases).

    Breadth-first over distinct states (``state_key`` dedupes orderings that reach the same
    state). Cost grows combinatorially; use only on bounded instances.
    """
    frontier = {state_key(s): s for s in [domain.initial_state(instance)]}
    best = [min(domain.objective(s, instance) for s in frontier.values())]
    for _ in range(max_steps):
        nxt = {}
        for state in frontier.values():
            for candidate in domain.legal_candidates(state):
                child = domain.apply(state, candidate)
                nxt.setdefault(state_key(child), child)
        if not nxt:
            best.append(best[-1])
            continue
        frontier = nxt
        best.append(min(domain.objective(s, instance) for s in frontier.values()))
    return [float(b) for b in best]


def evaluate_policies(
    domain: AllocationDomain,
    instances: Sequence,
    policies: Sequence[Policy],
    max_steps: int,
    allow_stop: bool = False,
    random_draws: int = 8,
    instance_id: Callable = lambda inst: getattr(inst, "name", repr(inst)),
) -> pd.DataFrame:
    """Long table: one row per (instance, policy, draw, budget step)."""
    rows = []
    for instance in instances:
        for policy in policies:
            draws = random_draws if policy.name == "random" else 1
            for draw in range(draws):
                trace = run_policy(domain, instance, policy, max_steps, allow_stop, seed=draw)
                for b, j in enumerate(trace.objective):
                    rows.append({
                        "instance": instance_id(instance), "policy": policy.name, "deployable": policy.deployable,
                        "draw": draw, "budget": b, "objective": j,
                        "action_rate": trace.action_cost[b].rate, "action_compute": trace.action_cost[b].compute,
                        "decision_compute": trace.decision_cost[b].compute, "stopped_at": trace.stopped_at,
                    })
    return pd.DataFrame(rows)
