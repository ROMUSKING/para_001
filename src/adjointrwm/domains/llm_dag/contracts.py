"""Typed boundary contracts, DAG nodes, and discrete costate packets.

Codified in docs/plans/d2_d3_hierarchical_adjoint_plan.md §2.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Optional

Tier = Literal["macro", "meso", "micro"]
NodeStatus = Literal["pending", "ready", "generating", "verified", "failed", "invalidated"]

VALID_TIERS = ("macro", "meso", "micro")
VALID_STATUSES = ("pending", "ready", "generating", "verified", "failed", "invalidated")


@dataclass
class BoundaryContract:
    """Explicit typed boundary contract defining interfaces and constraints for a node.

    Interfaces map symbol names (functions, classes, variables) to type signatures or specs.
    Pre-conditions specify expectations about input states.
    Post-conditions guarantee properties of the generated output.
    Invariants specify global properties that must be preserved.
    """

    interfaces: dict[str, str] = field(default_factory=dict)
    pre_conditions: list[str] = field(default_factory=list)
    post_conditions: list[str] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validate internal consistency of the contract."""
        if not isinstance(self.interfaces, dict):
            raise TypeError("interfaces must be a dict")
        if not isinstance(self.pre_conditions, list):
            raise TypeError("pre_conditions must be a list")
        if not isinstance(self.post_conditions, list):
            raise TypeError("post_conditions must be a list")
        if not isinstance(self.invariants, list):
            raise TypeError("invariants must be a list")

    def relax_interface(self, symbol: str, relaxed_signature: str) -> BoundaryContract:
        """Return a new contract with the specified symbol signature relaxed."""
        new_interfaces = dict(self.interfaces)
        new_interfaces[symbol] = relaxed_signature
        return BoundaryContract(
            interfaces=new_interfaces,
            pre_conditions=list(self.pre_conditions),
            post_conditions=list(self.post_conditions),
            invariants=list(self.invariants),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BoundaryContract:
        return cls(
            interfaces=dict(data.get("interfaces", {})),
            pre_conditions=list(data.get("pre_conditions", [])),
            post_conditions=list(data.get("post_conditions", [])),
            invariants=list(data.get("invariants", [])),
        )


@dataclass
class DAGNode:
    """A node in the Rooted Dependency DAG G = (V, E).

    Each node corresponds to an architectural level:
    - macro (Level 0): high-level system decomposition / crate layout
    - meso (Level 1): module orchestration / class interface definitions
    - micro (Level 2): concrete leaf implementations (functions, algorithms)
    """

    node_id: str
    tier: Tier
    dependencies: list[str] = field(default_factory=list)
    boundary_contract: BoundaryContract = field(default_factory=BoundaryContract)
    content_payload: Optional[str] = None
    status: NodeStatus = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.tier not in VALID_TIERS:
            raise ValueError(f"Invalid tier {self.tier!r}; must be one of {VALID_TIERS}")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {self.status!r}; must be one of {VALID_STATUSES}")
        if not isinstance(self.dependencies, list):
            raise TypeError("dependencies must be a list of node_ids")
        self.boundary_contract.validate()

    def is_leaf(self) -> bool:
        return self.tier == "micro"

    def is_verified(self) -> bool:
        return self.status == "verified"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DAGNode:
        contract_data = data.get("boundary_contract", {})
        contract = (
            contract_data
            if isinstance(contract_data, BoundaryContract)
            else BoundaryContract.from_dict(contract_data)
        )
        return cls(
            node_id=str(data["node_id"]),
            tier=data["tier"],
            dependencies=list(data.get("dependencies", [])),
            boundary_contract=contract,
            content_payload=data.get("content_payload"),
            status=data.get("status", "pending"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class CostateSensitivityPacket:
    """Discrete costate sensitivity update packet: lambda_v = nabla_C FailureSeverity.

    Attributing downstream leaf execution failures directly back to upstream
    contractual clauses, enabling surgical upstream contract relaxation (Delta u_macro propto -lambda).
    """

    source_leaf_id: str
    target_ancestor_id: str
    failing_symbol: Optional[str]
    diagnostics: list[str]
    constraint_relaxation_delta: str  # Delta u_macro directing contract edit
    severity_weight: float = 1.0       # Shadow price factor lambda

    def __post_init__(self):
        if self.severity_weight < 0:
            raise ValueError(f"severity_weight must be non-negative, got {self.severity_weight}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CostateSensitivityPacket:
        return cls(
            source_leaf_id=str(data["source_leaf_id"]),
            target_ancestor_id=str(data["target_ancestor_id"]),
            failing_symbol=data.get("failing_symbol"),
            diagnostics=list(data.get("diagnostics", [])),
            constraint_relaxation_delta=str(data["constraint_relaxation_delta"]),
            severity_weight=float(data.get("severity_weight", 1.0)),
        )
