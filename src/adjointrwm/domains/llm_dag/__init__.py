"""Track D2/D3: Hierarchical LLM Dependency DAG Generation with Discrete Adjoint Sensitivity.

Codified in docs/plans/d2_d3_hierarchical_adjoint_plan.md.
"""

from .adjoint_engine import DiscreteCostateEngine
from .contracts import BoundaryContract, CostateSensitivityPacket, DAGNode
from .dag import DependencyDAG
from .domain import LLMDAGDomain, LLMDAGInstance, LLMDAGState
from .verifier import SandboxedVerifier, VerificationResult

__all__ = [
    "BoundaryContract",
    "CostateSensitivityPacket",
    "DAGNode",
    "DependencyDAG",
    "DiscreteCostateEngine",
    "LLMDAGDomain",
    "LLMDAGInstance",
    "LLMDAGState",
    "SandboxedVerifier",
    "VerificationResult",
]
