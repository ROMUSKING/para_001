"""Rooted Dependency Directed Acyclic Graph (DAG) and topological scheduler.

Codified in docs/plans/d2_d3_hierarchical_adjoint_plan.md §2.1.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Optional, Sequence

from .contracts import BoundaryContract, DAGNode, NodeStatus


class DependencyDAG:
    """A rooted dependency Directed Acyclic Graph G = (V, E) of generative nodes.

    Nodes represent components (macro, meso, micro).
    Edges represent dependencies: u -> v means v depends on u (u in v.dependencies).
    """

    def __init__(self, nodes: Optional[Mapping[str, DAGNode]] = None):
        self._nodes: dict[str, DAGNode] = {}
        if nodes:
            for node_id, node in nodes.items():
                self.add_node(node)

    @property
    def nodes(self) -> dict[str, DAGNode]:
        return self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_node(self, node_id: str) -> DAGNode:
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id!r} not found in DAG")
        return self._nodes[node_id]

    def add_node(self, node: DAGNode) -> None:
        """Add a node to the DAG and verify that adding it does not create a cycle."""
        if node.node_id in self._nodes:
            raise ValueError(f"Node {node.node_id!r} already exists in DAG")
        self._nodes[node.node_id] = node
        if self.has_cycle():
            del self._nodes[node.node_id]
            raise ValueError(f"Adding node {node.node_id!r} introduces a cycle into the DAG")

    def update_node(self, node: DAGNode) -> None:
        """Update an existing node in the DAG."""
        if node.node_id not in self._nodes:
            raise KeyError(f"Node {node.node_id!r} does not exist in DAG")
        self._nodes[node.node_id] = node

    def predecessors(self, node_id: str) -> list[str]:
        """Return direct dependencies of node_id (incoming edges u -> node_id)."""
        node = self.get_node(node_id)
        return list(node.dependencies)

    def successors(self, node_id: str) -> list[str]:
        """Return nodes that depend on node_id (outgoing edges node_id -> v)."""
        succs = []
        for nid, node in self._nodes.items():
            if node_id in node.dependencies:
                succs.append(nid)
        return succs

    def ancestors(self, node_id: str) -> set[str]:
        """Return all transitive predecessors of node_id."""
        visited: set[str] = set()
        queue = list(self.predecessors(node_id))
        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                if curr in self._nodes:
                    queue.extend(self.predecessors(curr))
        return visited

    def descendants(self, node_id: str) -> set[str]:
        """Return all transitive successors of node_id."""
        visited: set[str] = set()
        queue = list(self.successors(node_id))
        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                queue.extend(self.successors(curr))
        return visited

    def has_cycle(self) -> bool:
        """Check for cycles using Kahn's algorithm."""
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        for node in self._nodes.values():
            for dep in node.dependencies:
                if dep in in_degree:
                    in_degree[node.node_id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        visited_count = 0
        while queue:
            curr = queue.pop(0)
            visited_count += 1
            for succ in self.successors(curr):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        return visited_count < len(self._nodes)

    def topological_sort(self) -> list[str]:
        """Return nodes in topological order (predecessors before successors)."""
        if self.has_cycle():
            raise ValueError("Cannot topologically sort a graph with cycles")

        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        for node in self._nodes.values():
            for dep in node.dependencies:
                if dep in in_degree:
                    in_degree[node.node_id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[str] = []
        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for succ in self.successors(curr):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        return order

    def ready_nodes(self) -> list[DAGNode]:
        """Evaluate the Topological Readiness Barrier:

        ready(v) <=> forall u in Pred(v), C_out(u) is finalized and VALID (verified),
        and v.status in ('pending', 'ready').
        """
        ready: list[DAGNode] = []
        for node in self._nodes.values():
            if node.status not in ("pending", "ready"):
                continue
            preds_satisfied = True
            for pred_id in node.dependencies:
                if pred_id not in self._nodes:
                    preds_satisfied = False
                    break
                pred_node = self._nodes[pred_id]
                if pred_node.status != "verified":
                    preds_satisfied = False
                    break
            if preds_satisfied:
                ready.append(node)
        return ready

    def invalidate_subgraph(self, root_id: str) -> set[str]:
        """Selectively invalidate only the downstream transitive descendants of root_id.

        Preserves independent sibling subtrees untouched!
        Returns the set of invalidated node IDs.
        """
        invalidated = self.descendants(root_id)
        invalidated.add(root_id)
        for nid in invalidated:
            node = self._nodes[nid]
            node.status = "invalidated"
        return invalidated

    def tree_preservation_rate(self, invalidated_node_ids: set[str]) -> float:
        """Fraction of total nodes in the DAG that remained preserved and valid:

        (total_nodes - len(invalidated_node_ids)) / total_nodes
        """
        if not self._nodes:
            return 1.0
        preserved = len(self._nodes) - len(invalidated_node_ids)
        return max(0.0, preserved / len(self._nodes))

    def copy(self) -> DependencyDAG:
        """Return a deep copy of the DAG for immutable functional state transitions."""
        copied_nodes = {nid: copy.deepcopy(node) for nid, node in self._nodes.items()}
        return DependencyDAG(copied_nodes)
