"""Unit tests for Track D2/D3 Hierarchical LLM DAG domain.

Codified in docs/plans/d2_d3_hierarchical_adjoint_plan.md §2, §3, §4.
"""

import pytest

from adjointrwm.domains.llm_dag import (
    BoundaryContract,
    CostateSensitivityPacket,
    DAGNode,
    DependencyDAG,
    DiscreteCostateEngine,
    LLMDAGDomain,
    LLMDAGInstance,
    LLMDAGState,
    SandboxedVerifier,
    VerificationResult,
)


def test_boundary_contract_and_dag_node_serialization():
    contract = BoundaryContract(
        interfaces={"execute(x)": "int -> int", "cleanup()": "None -> None"},
        pre_conditions=["x > 0"],
        post_conditions=["result >= 0"],
        invariants=["state_preserved"],
    )
    contract.validate()

    # Test relaxation
    relaxed = contract.relax_interface("execute(x)", "Any -> Any")
    assert relaxed.interfaces["execute(x)"] == "Any -> Any"
    assert contract.interfaces["execute(x)"] == "int -> int"  # original untouched

    # Test DAGNode
    node = DAGNode(
        node_id="module_auth",
        tier="meso",
        dependencies=["root_core"],
        boundary_contract=contract,
        content_payload="# auth module",
        status="ready",
    )
    d = node.to_dict()
    reconstructed = DAGNode.from_dict(d)
    assert reconstructed.node_id == "module_auth"
    assert reconstructed.tier == "meso"
    assert reconstructed.dependencies == ["root_core"]
    assert "execute(x)" in reconstructed.boundary_contract.interfaces


def test_dag_cycle_detection_and_topological_sort():
    dag = DependencyDAG()
    root = DAGNode("root", "macro")
    meso1 = DAGNode("meso1", "meso", dependencies=["root"])
    meso2 = DAGNode("meso2", "meso", dependencies=["root"])
    micro = DAGNode("micro", "micro", dependencies=["meso1", "meso2"])

    dag.add_node(root)
    dag.add_node(meso1)
    dag.add_node(meso2)
    dag.add_node(micro)

    assert not dag.has_cycle()
    topo = dag.topological_sort()
    assert topo[0] == "root"
    assert topo[-1] == "micro"

    # Test cycle rejection
    cycle_node = DAGNode("cycle_maker", "micro", dependencies=["micro"])
    dag.add_node(cycle_node)
    # Adding an edge from micro back to root would cause cycle; let's test directly:
    bad_dag = DependencyDAG()
    n1 = DAGNode("n1", "micro", dependencies=["n2"])
    n2 = DAGNode("n2", "micro", dependencies=["n1"])
    bad_dag.add_node(n1)
    with pytest.raises(ValueError, match="introduces a cycle"):
        bad_dag.add_node(n2)


def test_topological_readiness_barrier():
    dag = DependencyDAG()
    root = DAGNode("root", "macro", status="pending")
    child = DAGNode("child", "meso", dependencies=["root"], status="pending")

    dag.add_node(root)
    dag.add_node(child)

    # Initially, only root has satisfied dependencies (empty list)
    ready = dag.ready_nodes()
    assert len(ready) == 1
    assert ready[0].node_id == "root"

    # Verifying root unlocks child
    root.status = "verified"
    ready = dag.ready_nodes()
    assert len(ready) == 1
    assert ready[0].node_id == "child"


def test_sandboxed_verifier_ast_and_interfaces():
    verifier = SandboxedVerifier()

    # Valid Python code matching interface
    valid_node = DAGNode(
        node_id="worker_valid",
        tier="micro",
        boundary_contract=BoundaryContract(interfaces={"compute(a, b)": "float"}),
        content_payload="def compute(a, b):\n    return float(a + b)\n",
    )
    res = verifier.verify_node(valid_node)
    assert res.is_valid
    assert res.syntax_valid
    assert res.interface_valid

    # Syntax Error
    syntax_bad_node = DAGNode(
        node_id="worker_syntax_err",
        tier="micro",
        boundary_contract=BoundaryContract(interfaces={"foo()": "None"}),
        content_payload="def foo(:\n    return 42\n",
    )
    res_syn = verifier.verify_node(syntax_bad_node)
    assert not res_syn.is_valid
    assert not res_syn.syntax_valid
    assert any("SyntaxError" in d for d in res_syn.diagnostics)

    # Missing interface symbol
    missing_sym_node = DAGNode(
        node_id="worker_missing_sym",
        tier="micro",
        boundary_contract=BoundaryContract(interfaces={"required_bar(x)": "int"}),
        content_payload="def other_func():\n    return 1\n",
    )
    res_sym = verifier.verify_node(missing_sym_node)
    assert not res_sym.is_valid
    assert res_sym.syntax_valid
    assert not res_sym.interface_valid
    assert res_sym.failing_symbol == "required_bar"


def test_discrete_costate_attribution_and_surgical_repair():
    dag = DependencyDAG()
    root = DAGNode("root", "macro", status="verified")
    meso_a = DAGNode(
        "meso_a",
        "meso",
        dependencies=["root"],
        boundary_contract=BoundaryContract(interfaces={"parse_token(t)": "Token"}),
        status="verified",
    )
    meso_b = DAGNode(
        "meso_b",
        "meso",
        dependencies=["root"],
        boundary_contract=BoundaryContract(interfaces={"render_view(v)": "View"}),
        status="verified",
    )
    micro_a = DAGNode(
        "micro_a",
        "micro",
        dependencies=["meso_a"],
        boundary_contract=BoundaryContract(interfaces={"parse_token(t)": "Token"}),
        content_payload="def incorrect_func(): pass",
        status="failed",
    )
    micro_b = DAGNode(
        "micro_b",
        "micro",
        dependencies=["meso_b"],
        boundary_contract=BoundaryContract(interfaces={"render_view(v)": "View"}),
        content_payload="def render_view(v): return 'rendered'",
        status="verified",
    )

    for n in (root, meso_a, meso_b, micro_a, micro_b):
        dag.add_node(n)

    engine = DiscreteCostateEngine()
    verifier = SandboxedVerifier()
    v_res = verifier.verify_node(micro_a)
    assert not v_res.is_valid
    assert v_res.failing_symbol == "parse_token"

    # Attribute failure to ancestor meso_a (which introduced parse_token)
    packet = engine.attribute_failure(dag, "micro_a", v_res)
    assert packet.target_ancestor_id == "meso_a"
    assert packet.failing_symbol == "parse_token"
    assert packet.severity_weight >= 1.0

    # Apply surgical repair: meso_b and micro_b MUST remain preserved!
    repaired_dag, invalidated = engine.apply_surgical_repair(dag, packet)
    assert "meso_a" in invalidated
    assert "micro_a" in invalidated
    assert "meso_b" not in invalidated
    assert "micro_b" not in invalidated

    # Tree preservation rate
    preservation = repaired_dag.tree_preservation_rate(invalidated)
    # 3 preserved out of 5 nodes = 0.60
    assert abs(preservation - 0.60) < 1e-5
    assert repaired_dag.get_node("micro_b").status == "verified"


def test_llmdag_domain_execution():
    root = DAGNode("root", "macro", status="ready")
    leaf = DAGNode(
        "leaf",
        "micro",
        dependencies=["root"],
        boundary_contract=BoundaryContract(interfaces={"run_task()": "None"}),
        status="pending",
    )
    dag = DependencyDAG()
    dag.add_node(root)
    dag.add_node(leaf)

    instance = LLMDAGInstance("test_repo_1", "Generate task runner", dag)
    domain = LLMDAGDomain()
    state = domain.initial_state(instance)

    # Initially, root is ready to expand
    cands = domain.legal_candidates(state)
    assert any(c.kind == "expand" and c.target == "root" for c in cands)

    expand_cand = [c for c in cands if c.kind == "expand" and c.target == "root"][0]
    state2 = domain.apply(state, expand_cand)
    assert state2.dag.get_node("root").status == "verified"
    assert state2.tokens_consumed > 0

    # Leaf should now be ready to sample
    cands2 = domain.legal_candidates(state2)
    assert any(c.kind == "sample" and c.target == "leaf" for c in cands2)

    sample_cand = [c for c in cands2 if c.kind == "sample" and c.target == "leaf"][0]
    state3 = domain.apply(state2, sample_cand)
    assert state3.dag.get_node("leaf").status == "verified"
    assert state3.verifier_calls > 0

    # Check objective: all nodes verified -> minimal penalty
    obj = domain.objective(state3, instance)
    assert obj < 5.0
