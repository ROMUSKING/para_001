"""Spatial patch adapters for multi-token visual representations.

Maps spatio-temporal visual patch tokens [B, T, P, D] to latent frame vectors [B, T, d_model]
using learned spatial positional encodings and multi-head attention pooling.
Preserves spatial relationships while keeping recurrent dynamics parameter-matched.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPatchAdapter(nn.Module):
    """Adapts multi-token spatial patch grids [B, T, P, D] to latent frame vectors [B, T, d_model].

    When P=1 (global pooled vector), acts as a standard linear-norm adapter.
    When P > 1, projects each patch token to d_model, adds spatial position embeddings,
    and pools via multi-head attention over the patch dimension.
    """

    def __init__(
        self,
        token_dim: int,
        d_model: int,
        max_patches: int = 512,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.d_model = d_model
        self.max_patches = max_patches
        self.num_heads = num_heads

        self.patch_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, d_model),
            nn.GELU(),
        )

        # Learned spatial position embedding per patch position
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, max_patches, d_model))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

        # Learned query for attention pooling
        self.pool_query = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        nn.init.trunc_normal_(self.pool_query, std=0.02)

        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, visual: torch.Tensor, tokens: int | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            visual: [B, T, P, D] or [B, T, P*D] tensor of visual patch tokens.
            tokens: Number of tokens P if visual is flattened [B, T, P*D].

        Returns:
            Latent frame representation [B, T, d_model].
        """
        if visual.dim() == 3:
            if tokens is None:
                # Flat 1D representation with P=1
                p = 1
                d = visual.shape[-1]
                x = visual.unsqueeze(2)  # [B, T, 1, D]
            else:
                p = tokens
                d = visual.shape[-1] // p
                x = visual.view(*visual.shape[:2], p, d)
        elif visual.dim() == 4:
            x = visual
            p = visual.shape[2]
        else:
            raise ValueError(f"Expected 3D or 4D visual tensor, got shape {visual.shape}")

        b, t, p_cur, d_cur = x.shape
        if p_cur > self.max_patches:
            raise ValueError(f"Patch count {p_cur} exceeds max_patches {self.max_patches}")

        # Fast path for single pooled token (P=1)
        if p_cur == 1:
            projected = self.patch_proj(x.squeeze(2))  # [B, T, d_model]
            return self.out_norm(projected)

        # Multi-patch path (P > 1)
        h = self.patch_proj(x)  # [B, T, P, d_model]
        h = h + self.spatial_pos[:, :, :p_cur, :]

        # Flatten B and T for batch attention pooling: [B*T, P, d_model]
        h_flat = h.view(b * t, p_cur, self.d_model)
        query = self.pool_query.expand(b * t, -1, -1, -1).squeeze(2)  # [B*T, 1, d_model]

        pooled, _ = self.pool_attn(query, h_flat, h_flat)  # [B*T, 1, d_model]
        out = self.out_norm(pooled.view(b, t, self.d_model))
        return out
