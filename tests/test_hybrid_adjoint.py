"""Tests for HybridAdjointRecursiveWorldModel (HARP arm ``hybrid_adjoint_rwm``).

Verifies kinematic-residual integration, mathematical parity, gradient flow, and shape contracts.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from adjointrwm.models import (  # noqa: E402
    ArmDims,
    HybridAdjointConfig,
    HybridAdjointRecursiveWorldModel,
    KinematicTransition,
    build_arm,
    count_prediction_parameters,
    hybrid_stage1_loss,
    inputs_only,
)

DIMS = ArmDims(
    state_dim=14,
    action_dim=7,
    visual_tokens=2,
    visual_token_dim=16,
    target_visual_dim=32,
    context_len=6,
    horizon=4,
)


def make_test_batch(b=4, dims=DIMS, seed=42):
    g = torch.Generator().manual_seed(seed)
    T, H = dims.context_len, dims.horizon
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    return {
        "context_state": r(b, T, dims.state_dim),
        "context_action": torch.cat([torch.zeros(b, 1, dims.action_dim), r(b, T - 1, dims.action_dim)], dim=1),
        "context_visual": r(b, T, dims.visual_tokens, dims.visual_token_dim),
        "future_actions": r(b, H, dims.action_dim),
        "target_state": r(b, H, dims.state_dim),
        "target_visual": r(b, H, dims.target_visual_dim),
        "future_visual": r(b, H, dims.visual_tokens, dims.visual_token_dim),
    }


def test_kinematic_transition_step_by_step():
    trans = KinematicTransition(state_dim=3, action_dim=2, damping=0.90)
    s0 = torch.tensor([[1.0, 2.0, 3.0]])
    v0 = torch.tensor([[0.1, 0.2, 0.3]])
    u = torch.zeros(1, 3, 2)  # 3 horizon steps, zero actions

    states = trans(s0, v0, u)
    assert states.shape == (1, 3, 3)

    # Step 1: v1 = 0.9 * v0 = [0.09, 0.18, 0.27], s1 = s0 + v1 = [1.09, 2.18, 3.27]
    assert torch.allclose(states[0, 0], s0[0] + 0.90 * v0[0], atol=1e-5)
    # Step 2: v2 = 0.9 * v1 = 0.81 * v0, s2 = s1 + v2
    assert torch.allclose(states[0, 1], s0[0] + (0.90 + 0.81) * v0[0], atol=1e-5)


def test_hybrid_adjoint_builder_and_parameter_count():
    model = build_arm("hybrid_adjoint_rwm", DIMS, width=128, transformer_layers=2, transformer_heads=4)
    assert isinstance(model, HybridAdjointRecursiveWorldModel)
    params = count_prediction_parameters("hybrid_adjoint_rwm", DIMS, width=128, transformer_layers=2, transformer_heads=4)
    assert params > 0
    assert params == model.prediction_parameters()


def test_hybrid_adjoint_zero_residual_matches_kinematic():
    config = HybridAdjointConfig(
        context_len=DIMS.context_len,
        horizon=DIMS.horizon,
        d_model=64,
        transformer_layers=2,
        transformer_heads=2,
        transformer_ff=128,
    )
    model = HybridAdjointRecursiveWorldModel(DIMS.state_dim, DIMS.action_dim, DIMS.visual_flat_dim, config)
    batch = make_test_batch(b=2, dims=DIMS)

    # When residual head is zeroed, output matches kinematic baseline exactly
    with torch.no_grad():
        model.state_residual_head.weight.zero_()
        model.state_residual_head.bias.zero_()

    pred = model.predict(batch)
    s0 = batch["context_state"][:, -1]
    v0 = batch["context_state"][:, -1] - batch["context_state"][:, -2]
    expected_kinematic = model.kinematic_transition(s0, v0, batch["future_actions"])

    assert torch.allclose(pred["state_mean"], expected_kinematic, atol=1e-5)


def test_hybrid_adjoint_predict_shapes():
    model = build_arm("hybrid_adjoint_rwm", DIMS, width=64, transformer_layers=2, transformer_heads=2)
    batch = make_test_batch(b=3, dims=DIMS)

    pred = model.predict(batch)
    assert "state_mean" in pred
    assert "state_logvar" in pred
    assert "visual" in pred

    assert pred["state_mean"].shape == (3, DIMS.horizon, DIMS.state_dim)
    assert pred["state_logvar"].shape == (3, DIMS.horizon, DIMS.state_dim)
    assert pred["visual"].shape == (3, DIMS.horizon, DIMS.target_visual_dim)


def test_hybrid_adjoint_gradient_flow():
    model = build_arm("hybrid_adjoint_rwm", DIMS, width=64, transformer_layers=2, transformer_heads=2)
    batch = make_test_batch(b=4, dims=DIMS)

    loss, logs = model.training_loss(batch)
    assert loss.item() > 0
    assert "objective_allocated" in logs
    assert "objective_base" in logs

    loss.backward()

    # Check non-zero gradient in kinematic transition parameters
    assert model.kinematic_transition.A_v.weight.grad is not None
    assert model.kinematic_transition.A_v.weight.grad.abs().sum() > 0

    assert model.kinematic_transition.B_u.weight.grad is not None
    assert model.kinematic_transition.B_u.weight.grad.abs().sum() > 0

    # Check non-zero gradient in neural residual head
    assert model.state_residual_head.weight.grad is not None
    assert model.state_residual_head.weight.grad.abs().sum() > 0

    # Check non-zero gradient in temporal encoder
    enc_param = next(model.temporal_encoder.parameters())
    assert enc_param.grad is not None
    assert enc_param.grad.abs().sum() > 0


def test_hybrid_adjoint_costate_backward():
    """Verify Pontryagin adjoint co-state backpropagation dJ/dz."""
    model = build_arm("hybrid_adjoint_rwm", DIMS, width=64, transformer_layers=2, transformer_heads=2)
    batch = make_test_batch(b=2, dims=DIMS)

    latent = model.encode_inputs(inputs_only(batch))
    latent.requires_grad_(True)
    rollout = model.rollout(latent, batch["future_actions"], context_state=batch["context_state"])
    obj = (rollout["state_mean"] - batch["target_state"]).pow(2).mean()

    costate = torch.autograd.grad(obj, latent)[0]
    assert costate.shape == latent.shape
    assert costate.abs().sum() > 0


def test_hybrid_adjoint_causality():
    """Verify predictions are identical regardless of future visual or targets."""
    model = build_arm("hybrid_adjoint_rwm", DIMS, width=64, transformer_layers=2, transformer_heads=2)
    model.eval()
    batch1 = make_test_batch(b=2, dims=DIMS, seed=10)
    batch2 = dict(batch1)
    batch2["target_state"] = torch.randn_like(batch1["target_state"]) + 100.0
    batch2["future_visual"] = torch.randn_like(batch1["future_visual"]) + 100.0

    pred1 = model.predict(batch1)
    pred2 = model.predict(batch2)

    assert torch.allclose(pred1["state_mean"], pred2["state_mean"], atol=1e-6)
    assert torch.allclose(pred1["visual"], pred2["visual"], atol=1e-6)
