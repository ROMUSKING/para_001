"""Action-conditioned predictors over frozen visual features (arms ``dino_wm``, ``vjepa2_ac``).

Both rivals predict the next frame's frozen-encoder features with a block-causal transformer
over frame tokens, conditioned on actions and proprioception:

* **DINO-WM** (Zhou et al., arXiv:2411.04983; github.com/gaoyuezhou/dino_wm, MIT). From the
  official configs (checked 2026-09-29): frozen ``dinov2_vits14`` patch tokens; ViT predictor
  depth 6, heads 16, ``mlp_dim`` 2048, dropout 0.1, vit-pytorch attention with 64-d heads and
  a block-causal frame mask; action and proprio embeddings (dim 10 each) concatenated to every
  visual token; teacher-forced MSE on next-frame features.
* **V-JEPA 2-AC** (Assran et al., arXiv:2506.09985; github.com/facebookresearch/vjepa2, mostly
  MIT). From ``configs/train/vitg16/droid-256px-8f.yaml``: predictor depth 24 with 16 heads
  (64-d), ``loss_exp: 1.0`` (L1), ``normalize_reps: true``, ``auto_steps: 2`` (an extra
  autoregressive rollout loss).

Deviations (plan §2): a matched-size predictor on our cached features (not the released
checkpoints), patch tokens pooled to a small grid, no pixel decoder, and state/visual read-out
heads trained with the shared prediction objective so all arms are scored identically.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ArmDims, WorldModel, as_tokens, prediction_objective


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float):
        super().__init__()
        inner = heads * dim_head
        self.heads, self.dim_head = heads, dim_head
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        self.dropout = dropout

    def forward(self, x, mask):
        b, n, _ = x.shape
        q, k, v = self.qkv(self.norm(x)).view(b, n, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.out(attn.transpose(1, 2).reshape(b, n, -1))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def block_causal_mask(frames: int, tokens_per_frame: int, device=None) -> torch.Tensor:
    """``True`` where attention is allowed: token of frame i sees every token of frames <= i."""
    frame_of = torch.arange(frames, device=device).repeat_interleave(tokens_per_frame)
    return frame_of[:, None] >= frame_of[None, :]


class FeaturePredictorWorldModel(WorldModel):
    def __init__(
        self,
        dims: ArmDims,
        arm: str = "dino_wm",
        dim: int = 384,
        depth: int = 6,
        heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int | None = None,
        action_emb_dim: int = 10,
        proprio_emb_dim: int = 10,
        loss_type: str = "mse",
        rollout_steps: int = 0,
        normalize_reps: bool = False,
        dropout: float = 0.1,
        feature_weight: float = 1.0,
        visual_weight: float = 0.25,
    ):
        super().__init__()
        if loss_type not in ("mse", "l1"):
            raise ValueError("loss_type must be 'mse' or 'l1'")
        self.arm = arm
        self.dims = dims
        self.P, self.D = dims.visual_tokens, dims.visual_token_dim
        self.loss_type, self.rollout_steps, self.normalize_reps = loss_type, rollout_steps, normalize_reps
        self.feature_weight, self.visual_weight = feature_weight, visual_weight
        max_frames = dims.context_len + dims.horizon
        self.action_encoder = nn.Linear(dims.action_dim, action_emb_dim)
        self.proprio_encoder = nn.Linear(dims.state_dim, proprio_emb_dim)
        self.in_proj = nn.Linear(self.D + action_emb_dim + proprio_emb_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, max_frames * self.P, dim) * 0.02)
        mlp_dim = mlp_dim or 4 * dim
        self.layers = nn.ModuleList(
            nn.ModuleList([Attention(dim, heads, dim_head, dropout), FeedForward(dim, mlp_dim, dropout)]) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.feature_head = nn.Linear(dim, self.D)
        self.state_head = nn.Linear(dim, 2 * dims.state_dim)
        self.visual_head = nn.Linear(dim, dims.target_visual_dim)

    # ---- core ----------------------------------------------------------------------

    def _norm_tokens(self, tokens):
        return F.layer_norm(tokens, (self.D,)) if self.normalize_reps else tokens

    def _hidden(self, tokens, states, actions):
        """Transformer states ``[B,F,P,dim]``; frame ``i`` only depends on frames ``<= i``."""
        b, f = tokens.shape[:2]
        a = self.action_encoder(actions).unsqueeze(2).expand(-1, -1, self.P, -1)
        s = self.proprio_encoder(states).unsqueeze(2).expand(-1, -1, self.P, -1)
        x = self.in_proj(torch.cat([tokens, s, a], dim=-1)).reshape(b, f * self.P, -1)
        x = x + self.pos[:, : f * self.P]
        mask = block_causal_mask(f, self.P, device=x.device)
        for attn, ff in self.layers:
            x = x + attn(x, mask)
            x = x + ff(x)
        return self.norm(x).view(b, f, self.P, -1)

    def _forward_frames(self, tokens, states, actions):
        """Frames ``[B,F,P,D]`` + states ``[B,F,S]`` + action taken at each frame ``[B,F,A]``
        -> predictions for frames ``1..F`` (feature tokens, state mean/logvar, shared visual)."""
        x = self._hidden(tokens, states, actions)
        pooled = x.mean(dim=2)
        mean, logvar = self.state_head(pooled).chunk(2, dim=-1)
        return {
            "features": self.feature_head(x),
            "state_mean": mean,
            "state_logvar": torch.clamp(logvar, -6.0, 3.0),
            "visual": self.visual_head(pooled),
        }

    @staticmethod
    def _next_actions(context_action, future_actions):
        """Action taken *at* each frame: ``a_next[i] = action[start + i]``."""
        return torch.cat([context_action[:, 1:], future_actions], dim=1)

    def _feature_loss(self, pred, target):
        diff = pred.float() - target.float()
        return diff.abs().mean() if self.loss_type == "l1" else diff.pow(2).mean()

    # ---- WorldModel interface ------------------------------------------------------

    def _predict(self, inputs):
        tokens = self._norm_tokens(as_tokens(inputs["context_visual"], self.P))
        states = inputs["context_state"]
        a_next = self._next_actions(inputs["context_action"], inputs["future_actions"])
        T, H = states.shape[1], inputs["future_actions"].shape[1]
        outputs = []
        for j in range(H):
            frames = tokens.shape[1]
            out = self._forward_frames(tokens, states, a_next[:, :frames])
            step = {k: v[:, -1] for k, v in out.items()}
            outputs.append(step)
            tokens = torch.cat([tokens, step["features"].unsqueeze(1)], dim=1)
            states = torch.cat([states, step["state_mean"].unsqueeze(1)], dim=1)
        assert tokens.shape[1] == T + H
        return {k: torch.stack([o[k] for o in outputs], dim=1) for k in ("state_mean", "state_logvar", "visual")}

    def training_loss(self, batch):
        tokens = self._norm_tokens(
            torch.cat([as_tokens(batch["context_visual"], self.P), as_tokens(batch["future_visual"], self.P)], dim=1)
        )
        states = torch.cat([batch["context_state"], batch["target_state"]], dim=1)
        visual_target = torch.cat([batch["context_target_visual"], batch["target_visual"]], dim=1)
        a_next = self._next_actions(batch["context_action"], batch["future_actions"])
        T = batch["context_state"].shape[1]

        # Teacher forcing: frames 0..L-2 predict frames 1..L-1.
        out = self._forward_frames(tokens[:, :-1], states[:, :-1], a_next)
        feature = self._feature_loss(out["features"], tokens[:, 1:])
        shared = prediction_objective(
            {k: out[k] for k in ("state_mean", "state_logvar", "visual")}, states[:, 1:], visual_target[:, 1:], self.visual_weight
        ).mean()
        loss = self.feature_weight * feature + shared
        parts = {"feature_tf": feature.detach(), "shared_tf": shared.detach()}

        if self.rollout_steps > 0:
            # V-JEPA 2-AC-style rollout loss: feed predictions back from the end of the context.
            roll_tokens, roll_states = tokens[:, :T], states[:, :T]
            roll_feat, roll_shared = [], []
            for j in range(min(self.rollout_steps, tokens.shape[1] - T)):
                step = self._forward_frames(roll_tokens, roll_states, a_next[:, : roll_tokens.shape[1]])
                step = {k: v[:, -1] for k, v in step.items()}
                roll_feat.append(self._feature_loss(step["features"], tokens[:, T + j]))
                roll_shared.append(
                    prediction_objective(
                        {k: step[k].unsqueeze(1) for k in ("state_mean", "state_logvar", "visual")},
                        states[:, T + j : T + j + 1],
                        visual_target[:, T + j : T + j + 1],
                        self.visual_weight,
                    ).mean()
                )
                roll_tokens = torch.cat([roll_tokens, step["features"].unsqueeze(1)], dim=1)
                roll_states = torch.cat([roll_states, step["state_mean"].unsqueeze(1)], dim=1)
            rollout = torch.stack(roll_feat).mean()
            rollout_shared = torch.stack(roll_shared).mean()
            loss = loss + self.feature_weight * rollout + rollout_shared
            parts.update({"feature_rollout": rollout.detach(), "shared_rollout": rollout_shared.detach()})
        return loss, parts

    def latent_for_diagnostics(self, batch):
        tokens = self._norm_tokens(as_tokens(batch["context_visual"], self.P))
        a_next = self._next_actions(batch["context_action"], batch["future_actions"])
        hidden = self._hidden(tokens, batch["context_state"], a_next[:, : tokens.shape[1]])
        return hidden[:, -1].mean(dim=1)


def build_dino_wm(dims: ArmDims, width: int | None = None, **overrides) -> FeaturePredictorWorldModel:
    config = {"arm": "dino_wm", "depth": 6, "heads": 16, "dim_head": 64, "loss_type": "mse", "rollout_steps": 0,
              "normalize_reps": False, "dropout": 0.1}
    if width is not None:
        config["dim"] = int(width)
    config.update(overrides)
    return FeaturePredictorWorldModel(dims, **config)


def build_vjepa2_ac(dims: ArmDims, width: int | None = None, **overrides) -> FeaturePredictorWorldModel:
    dim = int(width) if width is not None else 256
    config = {"arm": "vjepa2_ac", "dim": dim, "depth": 24, "heads": max(1, dim // 64), "dim_head": 64, "loss_type": "l1",
              "rollout_steps": 2, "normalize_reps": True, "dropout": 0.0}
    config.update(overrides)
    return FeaturePredictorWorldModel(dims, **config)
