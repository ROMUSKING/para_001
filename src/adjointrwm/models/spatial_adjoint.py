"""Spatial Adjoint Recursive World Model with SpatialPatchAdapter."""

from __future__ import annotations

import torch
import torch.nn as nn

from .adjoint_rwm import AdjointRWMConfig, AdjointRecursiveWorldModel
from .common import ArmDims
from .spatial_adapter import SpatialPatchAdapter


class SpatialAdjointRecursiveWorldModel(AdjointRecursiveWorldModel):
    arm = "spatial_adjoint_rwm"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        token_dim: int,
        config: AdjointRWMConfig,
        patch_tokens: int = 16,
        num_heads: int = 4,
    ):
        super().__init__(state_dim, action_dim, token_dim, config)
        d = config.d_model
        self.patch_tokens = patch_tokens
        self.visual_adapter = SpatialPatchAdapter(
            token_dim=token_dim,
            d_model=d,
            max_patches=max(512, patch_tokens),
            num_heads=num_heads,
            dropout=config.dropout,
        )

    def encode_context(self, context_visual, context_state, context_action):
        v = self.visual_adapter(context_visual, tokens=self.patch_tokens if context_visual.dim() == 3 else None)
        tokens = (
            v
            + self.state_adapter(context_state)
            + self.context_action_adapter(context_action)
        )
        tokens = tokens + self.position[:, : tokens.shape[1]]
        encoded = self.temporal_encoder(tokens)
        return self.temporal_norm(encoded[:, -1])


def build_spatial_adjoint_rwm(dims: ArmDims, width: int | None = None, **overrides) -> SpatialAdjointRecursiveWorldModel:
    fields = {"context_len": dims.context_len, "horizon": dims.horizon, **overrides}
    if width is not None:
        fields["d_model"] = int(width)
        fields.setdefault("transformer_ff", 4 * int(width))
    config = AdjointRWMConfig(**fields)
    model = SpatialAdjointRecursiveWorldModel(
        state_dim=dims.state_dim,
        action_dim=dims.action_dim,
        token_dim=dims.visual_token_dim,
        config=config,
        patch_tokens=dims.visual_tokens,
    )
    model.visual_head = nn.Linear(config.d_model, dims.target_visual_dim)
    return model
