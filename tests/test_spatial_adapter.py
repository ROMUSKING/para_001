"""Unit tests for SpatialPatchAdapter."""

import pytest
import torch

from adjointrwm.models.spatial_adapter import SpatialPatchAdapter


def test_spatial_patch_adapter_pooled():
    """P=1 pooled input produces expected [B, T, d_model] output."""
    adapter = SpatialPatchAdapter(token_dim=384, d_model=512)
    x = torch.randn(4, 8, 1, 384)
    out = adapter(x)
    assert out.shape == (4, 8, 512)

    # Flat [B, T, D] input
    x_flat = torch.randn(4, 8, 384)
    out_flat = adapter(x_flat)
    assert out_flat.shape == (4, 8, 512)


def test_spatial_patch_adapter_multi_tokens():
    """Multi-token grids P=16 and P=64 produce correct shapes and gradient flow."""
    adapter = SpatialPatchAdapter(token_dim=384, d_model=512, max_patches=256)

    # Grid 4 (P=16)
    x16 = torch.randn(2, 4, 16, 384, requires_grad=True)
    out16 = adapter(x16)
    assert out16.shape == (2, 4, 512)

    loss = out16.sum()
    loss.backward()
    assert x16.grad is not None
    assert torch.all(torch.isfinite(x16.grad))

    # Grid 8 (P=64) flat with tokens argument
    x64_flat = torch.randn(2, 4, 64 * 384)
    out64 = adapter(x64_flat, tokens=64)
    assert out64.shape == (2, 4, 512)


def test_spatial_position_sensitivity():
    """Permuting spatial tokens alters output due to learned positional embeddings."""
    torch.manual_seed(42)
    adapter = SpatialPatchAdapter(token_dim=384, d_model=512, max_patches=16)

    x = torch.randn(1, 1, 16, 384)
    out_orig = adapter(x)

    # Reverse patch ordering
    x_perm = x[:, :, torch.arange(15, -1, -1), :]
    out_perm = adapter(x_perm)

    diff = (out_orig - out_perm).abs().max().item()
    assert diff > 1e-4, f"Spatial adapter failed to distinguish permuted patches: diff={diff}"


def test_max_patches_overflow():
    """Exceeding max_patches raises ValueError."""
    adapter = SpatialPatchAdapter(token_dim=384, d_model=512, max_patches=16)
    x = torch.randn(1, 1, 32, 384)
    with pytest.raises(ValueError, match="exceeds max_patches"):
        adapter(x)


def test_spatial_adjoint_world_model_forward():
    """SpatialAdjointRecursiveWorldModel forward and rollout operate with multi-token input."""
    from adjointrwm.models.common import ArmDims
    from adjointrwm.models.registry import build_arm

    dims = ArmDims(
        state_dim=14,
        action_dim=7,
        visual_tokens=16,
        visual_token_dim=384,
        target_visual_dim=512,
        context_len=4,
        horizon=2,
    )
    model = build_arm("spatial_adjoint_rwm", dims)
    assert model.arm == "spatial_adjoint_rwm"
    assert model.prediction_parameters() > 10_000_000

    b, t, p, d = 2, 4, 16, 384
    batch = {
        "context_visual": torch.randn(b, t, p, d),
        "context_state": torch.randn(b, t, 14),
        "context_action": torch.randn(b, t, 7),
        "future_actions": torch.randn(b, 2, 7),
    }
    out = model.predict(batch)
    assert "state_mean" in out
    assert out["state_mean"].shape == (b, 2, 14)
    assert "visual" in out
    assert out["visual"].shape == (b, 2, 512)

