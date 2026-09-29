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
) -> Trace:
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
