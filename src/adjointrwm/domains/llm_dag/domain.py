"""Domain-neutral allocation layer integration for Hierarchical LLM DAG generation.

Codified in docs/plans/d2_d3_hierarchical_adjoint_plan.md §4.
Subclasses AllocationDomain from adjointrwm.domains.base.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from adjointrwm.domains.base import (
    AllocationDomain,
    Candidate,
    Cost,
    CostWeights,
    DomainSpec,
)
from .adjoint_engine import DiscreteCostateEngine
from .contracts import BoundaryContract, CostateSensitivityPacket, DAGNode
from .dag import DependencyDAG
from .verifier import SandboxedVerifier, VerificationResult


@dataclass(frozen=True)
class LLMDAGState:
    """Immutable state representation of the LLM DAG generation domain."""

    dag: DependencyDAG
    tokens_consumed: int = 0
    model_calls: int = 0
    verifier_calls: int = 0
    repair_events: int = 0
    preserved_rate: float = 1.0

    def copy(self, **kwargs) -> LLMDAGState:
        return LLMDAGState(
            dag=kwargs.get("dag", self.dag.copy()),
            tokens_consumed=kwargs.get("tokens_consumed", self.tokens_consumed),
            model_calls=kwargs.get("model_calls", self.model_calls),
            verifier_calls=kwargs.get("verifier_calls", self.verifier_calls),
            repair_events=kwargs.get("repair_events", self.repair_events),
            preserved_rate=kwargs.get("preserved_rate", self.preserved_rate),
        )


@dataclass
class LLMDAGInstance:
    """An instance task defining an overall target artifact to generate."""

    instance_id: str
    target_spec: str
    initial_dag: DependencyDAG
    gold_code: Optional[str] = None
    target_tokens: int = 1000


class LLMDAGDomain(AllocationDomain):
    """AllocationDomain implementation for hierarchical LLM DAG generation."""

    def __init__(self):
        self.spec = DomainSpec(
            domain_id="llm_dag",
            family="language",
            native_endpoint="validation_pass_rate_and_token_efficiency",
            candidate_kinds=("refine", "sample", "expand", "drop"),
            cost_units={
                "rate": "tokens",
                "compute": "model_calls",
                "latency": "ms",
                "query": "verifier_calls",
            },
            oracle_support="approximate",
            privilege={"target": "evaluation_only"},
            data_source="hierarchical_code_dag_instances",
            licence="MIT",
            notes="Track D2/D3 hierarchical code generation with discrete costate sensitivity.",
        )
        self.verifier = SandboxedVerifier()
        self.costate_engine = DiscreteCostateEngine()

    def initial_state(self, instance: LLMDAGInstance) -> LLMDAGState:
        return LLMDAGState(
            dag=instance.initial_dag.copy(),
            tokens_consumed=0,
            model_calls=0,
            verifier_calls=0,
            repair_events=0,
            preserved_rate=1.0,
        )

    def legal_candidates(self, state: LLMDAGState) -> list[Candidate]:
        candidates: list[Candidate] = []
        dag = state.dag

        # 1. Ready nodes that can be expanded or sampled
        ready = dag.ready_nodes()
        for node in ready:
            if node.tier in ("macro", "meso"):
                # Expand action: decomposes architectural plan into downstream modules
                candidates.append(
                    Candidate(
                        id=f"expand_{node.node_id}",
                        kind="expand",
                        target=node.node_id,
                        cost=Cost(rate=150.0, compute=1.0, latency=250.0, query=0.0),
                        payload={"tier": node.tier},
                    )
                )
            elif node.tier == "micro":
                # Sample action: generates leaf code payload
                candidates.append(
                    Candidate(
                        id=f"sample_{node.node_id}",
                        kind="sample",
                        target=node.node_id,
                        cost=Cost(rate=200.0, compute=1.0, latency=300.0, query=1.0),
                        payload={"tier": node.tier},
                    )
                )

        # 2. Failed nodes that need repair or recovery
        failed_nodes = [n for n in dag.nodes.values() if n.status == "failed"]
        for node in failed_nodes:
            # Refine action: Adjoint-guided surgical upstream relaxation
            candidates.append(
                Candidate(
                    id=f"refine_costate_{node.node_id}",
                    kind="refine",
                    target=node.node_id,
                    cost=Cost(rate=80.0, compute=1.0, latency=150.0, query=1.0),
                    payload={"repair_mode": "costate_adjoint"},
                )
            )
            # Drop action: Monolithic teardown of entire tree / full re-generation
            candidates.append(
                Candidate(
                    id=f"drop_teardown_{node.node_id}",
                    kind="drop",
                    target=node.node_id,
                    cost=Cost(
                        rate=float(len(dag) * 150),
                        compute=float(len(dag)),
                        latency=float(len(dag) * 200),
                        query=float(len(dag)),
                    ),
                    payload={"repair_mode": "full_teardown"},
                )
            )

        return candidates

    def apply(self, state: LLMDAGState, candidate: Candidate) -> LLMDAGState:
        new_dag = state.dag.copy()
        node_id = str(candidate.target)
        node = new_dag.get_node(node_id)

        tokens_added = int(candidate.cost.rate)
        calls_added = int(candidate.cost.compute)
        verifs_added = int(candidate.cost.query)
        repairs_added = 0
        preservation = state.preserved_rate

        if candidate.kind == "expand":
            # Expansion marks node verified and unlocks downstream submodules
            node.status = "verified"
            if node.content_payload is None:
                node.content_payload = f"# Architecture spec for {node_id}"

        elif candidate.kind == "sample":
            # Generate dummy compliant leaf payload if not present
            if node.content_payload is None:
                # Default implementation providing contract interface
                funcs = []
                for sym in node.boundary_contract.interfaces:
                    clean_sym = sym.split("(")[0].strip()
                    funcs.append(f"def {clean_sym}():\n    return True\n")
                node.content_payload = "\n".join(funcs) if funcs else "pass\n"

            # Verify with sandboxed verifier
            v_res = self.verifier.verify_node(node)
            if v_res.is_valid:
                node.status = "verified"
            else:
                node.status = "failed"
                node.metadata["last_verification"] = v_res

        elif candidate.kind == "refine":
            # Adjoint-guided surgical repair
            repairs_added = 1
            last_v = node.metadata.get("last_verification")
            if last_v is None:
                last_v = self.verifier.verify_node(node)

            packet = self.costate_engine.attribute_failure(new_dag, node_id, last_v)
            new_dag, invalidated = self.costate_engine.apply_surgical_repair(new_dag, packet)
            preservation = new_dag.tree_preservation_rate(invalidated)

            # Re-verify relaxed node
            for inv_id in invalidated:
                inv_node = new_dag.get_node(inv_id)
                inv_node.status = "pending"

        elif candidate.kind == "drop":
            # Full teardown of tree
            repairs_added = 1
            for n in new_dag.nodes.values():
                n.status = "pending"
            preservation = 0.0

        return state.copy(
            dag=new_dag,
            tokens_consumed=state.tokens_consumed + tokens_added,
            model_calls=state.model_calls + calls_added,
            verifier_calls=state.verifier_calls + verifs_added,
            repair_events=state.repair_events + repairs_added,
            preserved_rate=preservation,
        )

    def objective(self, state: LLMDAGState, instance: LLMDAGInstance) -> float:
        """Lower-is-better declared objective.

        Penalty for unverified/failed nodes + token cost overhead.
        """
        failed_count = sum(1 for n in state.dag.nodes.values() if n.status != "verified")
        token_overhead = state.tokens_consumed / max(1, instance.target_tokens)
        repair_penalty = 0.5 * state.repair_events
        preservation_reward = 1.0 - state.preserved_rate

        return failed_count * 10.0 + token_overhead + repair_penalty + preservation_reward

    def observation(self, instance: LLMDAGInstance) -> dict[str, Any]:
        return {
            "instance_id": instance.instance_id,
            "target_spec": instance.target_spec,
            "node_count": len(instance.initial_dag),
        }
