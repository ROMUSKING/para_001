"""Domain-neutral allocation interface (docs/plans/cross-domain-plan.md §4).

Every domain supplies a state, legal candidates with a cost vector, an executor and a
lower-is-better objective. Everything else (policies, oracles, ledgers, metrics) is written
once in :mod:`adjointrwm.domains.runner` and :mod:`adjointrwm.domains.metrics`.

Privilege (comprehensive plan §3.9): :meth:`AllocationDomain.objective` may read targets and is
used only to *evaluate* decisions. Deployable scorers receive :meth:`observation` (P0) and the
current state, never the objective or the instance's targets.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Hashable, Mapping

# Action kinds from the brief and comprehensive plan §4.1. ``hold`` and ``stop`` are generic.
ACTION_KINDS = (
    "refine", "sample", "query", "retain", "compress", "summarize", "drop", "simulate",
    "expand", "retrieve", "route", "join", "request",
)
ORACLE_SUPPORT = ("exact", "bounded", "approximate", "none")
FAMILIES = ("spatial", "temporal", "relational", "event", "control", "language")


@dataclass(frozen=True)
class Cost:
    """Resource use of one candidate, split into the plan's ledgers (§3.6, §3.7, §4.1)."""

    rate: float = 0.0      # R: bits, tokens, stored states, samples
    compute: float = 0.0   # C: solver steps, model calls, FLOPs (declared unit)
    latency: float = 0.0   # L: milliseconds or declared unit
    query: float = 0.0     # Q: acquisition risk, human time, message maintenance

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(self.rate + other.rate, self.compute + other.compute,
                    self.latency + other.latency, self.query + other.query)

    def weighted(self, weights: "CostWeights") -> float:
        return (weights.rate * self.rate + weights.compute * self.compute
                + weights.latency * self.latency + weights.query * self.query)


@dataclass(frozen=True)
class CostWeights:
    """``beta_R, beta_C, beta_L, beta_Q`` of the net score ``S(u)`` (plan §4.4)."""

    rate: float = 0.0
    compute: float = 0.0
    latency: float = 0.0
    query: float = 0.0


@dataclass(frozen=True)
class Candidate:
    id: str
    kind: str
    target: Hashable
    cost: Cost = Cost()
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"unknown action kind {self.kind!r}")


@dataclass(frozen=True)
class DomainSpec:
    """The subset of the plan's ``DomainSpec`` (§3.10.2) needed to run and report a domain."""

    domain_id: str
    family: str
    native_endpoint: str
    candidate_kinds: tuple
    cost_units: Mapping[str, str]
    oracle_support: str
    privilege: Mapping[str, str]
    data_source: str
    licence: str
    notes: str = ""

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if self.oracle_support not in ORACLE_SUPPORT:
            raise ValueError(f"oracle_support must be one of {ORACLE_SUPPORT}")
        unknown = set(self.candidate_kinds) - set(ACTION_KINDS)
        if unknown:
            raise ValueError(f"unknown candidate kinds {sorted(unknown)}")
        missing = {"rate", "compute", "latency", "query"} - set(self.cost_units)
        if missing:
            raise ValueError(f"cost_units must name every ledger; missing {sorted(missing)}")

    def to_dict(self) -> dict:
        return asdict(self)

    def checksum(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()


class AllocationDomain(ABC):
    """One allocation domain. States are treated as immutable: ``apply`` returns a new state."""

    spec: DomainSpec

    @abstractmethod
    def initial_state(self, instance) -> Any:
        """Encode the observation and build the hierarchy the allocator acts on."""

    @abstractmethod
    def legal_candidates(self, state) -> list[Candidate]:
        """Legal non-trivial actions (hold/stop are handled by the runner)."""

    @abstractmethod
    def apply(self, state, candidate: Candidate) -> Any:
        """Execute a candidate and return the new state."""

    @abstractmethod
    def objective(self, state, instance) -> float:
        """Lower-is-better declared objective. May read targets: evaluation only."""

    @abstractmethod
    def observation(self, instance) -> Any:
        """The deployable (P0) view of an instance given to scorers."""

    def native_metrics(self, state, instance) -> dict:
        return {"objective": float(self.objective(state, instance))}
