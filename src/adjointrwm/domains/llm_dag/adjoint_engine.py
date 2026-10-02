"""Discrete costate adjoint engine for hierarchical dependency DAGs.

Codified in docs/plans/d2_d3_hierarchical_adjoint_plan.md §2.3.
Maps downstream leaf execution failures back to upstream contractual clauses,
computing sensitivity updates lambda_v and surgical contract relaxations Delta u_macro.
"""

from __future__ import annotations

from typing import Optional

from .contracts import BoundaryContract, CostateSensitivityPacket, DAGNode
from .dag import DependencyDAG
from .verifier import VerificationResult


class DiscreteCostateEngine:
    """Computes discrete adjoint costate sensitivity packets and applies surgical repairs."""

    def attribute_failure(
        self,
        dag: DependencyDAG,
        failing_leaf_id: str,
        verification: VerificationResult,
    ) -> CostateSensitivityPacket:
        """Trace a verification failure back to the responsible ancestor node in the DAG.

        Calculates discrete costate sensitivity lambda_v:
        lambda_v proportional to failure severity and downstream dependency blast radius.
        """
        failing_node = dag.get_node(failing_leaf_id)
        failing_sym = verification.failing_symbol

        # Identify upstream target ancestor
        target_ancestor_id: Optional[str] = None

        if failing_sym:
            # Check which ancestor declared or referenced this failing symbol in its contract
            for anc_id in dag.ancestors(failing_leaf_id):
                anc_node = dag.get_node(anc_id)
                if failing_sym in anc_node.boundary_contract.interfaces:
                    target_ancestor_id = anc_id
                    break

        if target_ancestor_id is None:
            # Fall back to immediate predecessor or root
            preds = dag.predecessors(failing_leaf_id)
            if preds:
                target_ancestor_id = preds[0]
            else:
                target_ancestor_id = failing_leaf_id

        # Calculate severity weight lambda (shadow price):
        # Base failure (1.0) + downstream dependency impact
        descendant_count = len(dag.descendants(target_ancestor_id))
        severity = 1.0 + 0.5 * len(verification.diagnostics) + 0.2 * descendant_count

        # Formulate constraint relaxation delta Delta u_macro
        if failing_sym:
            delta = f"RELAX_INTERFACE_SPEC: symbol={failing_sym!r} in ancestor {target_ancestor_id!r}"
        else:
            delta = f"RELAX_PRECONDITIONS: in ancestor {target_ancestor_id!r} to accommodate syntax/type repair"

        return CostateSensitivityPacket(
            source_leaf_id=failing_leaf_id,
            target_ancestor_id=target_ancestor_id,
            failing_symbol=failing_sym,
            diagnostics=list(verification.diagnostics),
            constraint_relaxation_delta=delta,
            severity_weight=severity,
        )

    def apply_surgical_repair(
        self,
        dag: DependencyDAG,
        packet: CostateSensitivityPacket,
        relaxed_spec: Optional[str] = None,
    ) -> tuple[DependencyDAG, set[str]]:
        """Apply surgical contract relaxation to the target ancestor and invalidate only its sub-DAG.

        Preserves independent sibling subtrees untouched!
        Returns (new_dag, invalidated_node_ids).
        """
        new_dag = dag.copy()
        target_node = new_dag.get_node(packet.target_ancestor_id)

        # Apply surgical edit to boundary contract
        old_contract = target_node.boundary_contract
        if packet.failing_symbol and packet.failing_symbol in old_contract.interfaces:
            new_spec = relaxed_spec or f"Any (relaxed from {old_contract.interfaces[packet.failing_symbol]})"
            target_node.boundary_contract = old_contract.relax_interface(
                packet.failing_symbol, new_spec
            )
        else:
            # Relax pre-conditions if general failure
            target_node.boundary_contract = BoundaryContract(
                interfaces=dict(old_contract.interfaces),
                pre_conditions=[],  # relaxed
                post_conditions=list(old_contract.post_conditions),
                invariants=list(old_contract.invariants),
            )

        # Invalidate target ancestor and its descendants, preserving sibling subtrees
        invalidated = new_dag.invalidate_subgraph(packet.target_ancestor_id)
        return new_dag, invalidated
