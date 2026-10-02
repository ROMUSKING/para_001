"""Deterministic and symbolic verifier for DAG node contracts and payloads.

Codified in docs/plans/d2_d3_hierarchical_adjoint_plan.md §2.3 & §3.
Validation is deterministic and symbolic (AST parsing, interface conformance,
structural checks) rather than approximate latent-space matching.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Optional

from .contracts import BoundaryContract, DAGNode


@dataclass
class VerificationResult:
    """The outcome of symbolic verification on a DAG node."""

    is_valid: bool
    diagnostics: list[str] = field(default_factory=list)
    failing_symbol: Optional[str] = None
    syntax_valid: bool = True
    interface_valid: bool = True
    assertions_valid: bool = True

    def summary(self) -> str:
        if self.is_valid:
            return "VALID"
        return f"INVALID: {'; '.join(self.diagnostics)}"


class SandboxedVerifier:
    """Deterministic AST-based contract verifier for generative DAG nodes."""

    def verify_node(self, node: DAGNode) -> VerificationResult:
        """Verify node content payload against its boundary contract."""
        payload = node.content_payload
        if payload is None or not payload.strip():
            return VerificationResult(
                is_valid=False,
                diagnostics=["Empty or null content payload"],
                syntax_valid=False,
                interface_valid=False,
            )

        # 1. AST Syntax Parsing
        try:
            tree = ast.parse(payload)
        except SyntaxError as e:
            diag = f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}"
            return VerificationResult(
                is_valid=False,
                diagnostics=[diag],
                syntax_valid=False,
                interface_valid=False,
            )

        # 2. Interface Conformance Checking
        interface_diagnostics: list[str] = []
        failing_sym: Optional[str] = None

        defined_functions: dict[str, list[str]] = {}
        defined_classes: dict[str, list[str]] = {}
        defined_variables: set[str] = set()

        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [arg.arg for arg in stmt.args.args]
                defined_functions[stmt.name] = args
            elif isinstance(stmt, ast.ClassDef):
                methods = [
                    m.name
                    for m in stmt.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                defined_classes[stmt.name] = methods
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        defined_variables.add(target.id)

        # Check required symbols in boundary contract
        contract = node.boundary_contract
        for symbol, spec in contract.interfaces.items():
            # If spec specifies a function signature like "def foo(x, y)" or "fn(a, b) -> c"
            clean_sym = symbol.strip()
            if "(" in clean_sym:
                sym_name = clean_sym.split("(")[0].strip()
            else:
                sym_name = clean_sym

            is_defined = (
                sym_name in defined_functions
                or sym_name in defined_classes
                or sym_name in defined_variables
            )

            if not is_defined:
                interface_diagnostics.append(
                    f"Missing required interface symbol: {sym_name!r} (expected spec: {spec})"
                )
                if failing_sym is None:
                    failing_sym = sym_name

        if interface_diagnostics:
            return VerificationResult(
                is_valid=False,
                diagnostics=interface_diagnostics,
                failing_symbol=failing_sym,
                syntax_valid=True,
                interface_valid=False,
            )

        return VerificationResult(
            is_valid=True,
            diagnostics=[],
            syntax_valid=True,
            interface_valid=True,
            assertions_valid=True,
        )
