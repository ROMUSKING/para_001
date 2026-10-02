"""Hybrid Kinematic + Neural Residual Adjoint Recursive World Model (HARP arm ``hybrid_adjoint_rwm``).

Combines an analytical kinematic state-action continuation base with a transformer-driven
non-linear neural residual and Pontryagin adjoint co-state allocation.

The kinematic base ensures that linear joint/Cartesian momentum is captured without wasting
neural capacity, while the neural residual model focuses on contact forces, visual grounding,
and multi-modal interactions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adjoint_rwm import (
    ALLOCATOR_HEADS,
    MLP,
    sample_candidate_mask,
    sample_single_choice_mask,
)
from .common import (
    ArmDims,
    WorldModel,
    flatten_visual,
    prediction_objective,
    variance_covariance_regularizer,
)


@dataclass(frozen=True)
class HybridAdjointConfig:
    """Configuration for the Hybrid Kinematic-Residual Adjoint World Model."""

    context_len: int = 8
    horizon: int = 4
    d_model: int = 512
    transformer_layers: int = 6
    transformer_heads: int = 8
    transformer_ff: int = 2048
    dropout: float = 0.10
    num_refinement_candidates: int = 4
    candidate_costs: tuple = (1.0, 1.0, 1.5, 2.0)
    rate_beta: float = 0.002
    mask_mode: str = "subset"
    prediction_mode: str = "full"
    kinematic_damping: float = 0.90

    def to_dict(self) -> dict:
        return asdict(self)


class KinematicTransition(nn.Module):
    """Linear state-space kinematic continuation: v_{t+1} = A_v v_t + B_u a_{t+1}, s_{t+1} = s_t + v_{t+1}."""

    def __init__(self, state_dim: int, action_dim: int, damping: float = 0.90):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.A_v = nn.Linear(state_dim, state_dim, bias=False)
        self.B_u = nn.Linear(action_dim, state_dim, bias=True)

        with torch.no_grad():
            self.A_v.weight.copy_(damping * torch.eye(state_dim))
            self.B_u.weight.zero_()
            self.B_u.bias.zero_()

    def forward(self, s0: torch.Tensor, v0: torch.Tensor, future_actions: torch.Tensor) -> torch.Tensor:
        """Rollout kinematic continuation across the horizon."""
        _, horizon, _ = future_actions.shape
        states = []
        curr_s = s0
        curr_v = v0
        for h in range(horizon):
            curr_v = self.A_v(curr_v) + self.B_u(future_actions[:, h])
            curr_s = curr_s + curr_v
            states.append(curr_s)
        return torch.stack(states, dim=1)


class HybridAdjointRecursiveWorldModel(WorldModel):
    arm = "hybrid_adjoint_rwm"

    def __init__(self, state_dim: int, action_dim: int, visual_dim: int, config: HybridAdjointConfig):
        super().__init__()
        d = config.d_model
        self.config = config
        self.d_model = d
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = config.horizon
        self.num_candidates = config.num_refinement_candidates

        # Adapters
        self.visual_adapter = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, d), nn.GELU())
        self.state_adapter = nn.Sequential(nn.Linear(state_dim, d), nn.LayerNorm(d), nn.GELU())
        self.context_action_adapter = nn.Sequential(nn.Linear(action_dim, d), nn.LayerNorm(d), nn.GELU())
        self.future_action_adapter = nn.Sequential(nn.Linear(action_dim, d), nn.LayerNorm(d), nn.GELU())

        # Temporal Context Encoder
        self.position = nn.Parameter(torch.zeros(1, config.context_len, d))
        nn.init.normal_(self.position, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
        self.temporal_norm = nn.LayerNorm(d)

        # Candidate Refinement
        self.candidate_embedding = nn.Embedding(self.num_candidates, d)
        self.shared_refiner = MLP(2 * d, 4 * d, d, dropout=config.dropout)

        # Latent Dynamics & Kinematics
        self.transition = nn.GRUCell(d, d)
        self.kinematic_transition = KinematicTransition(state_dim, action_dim, damping=config.kinematic_damping)

        # Read-out Heads
        self.state_residual_head = nn.Linear(d, state_dim)
        # Initialize residual head with small weights so network starts near kinematic baseline while ensuring gradient flow
        with torch.no_grad():
            nn.init.normal_(self.state_residual_head.weight, std=0.01)
            self.state_residual_head.bias.zero_()

        self.state_logvar_head = nn.Linear(d, state_dim)
        self.visual_head = nn.Linear(d, visual_dim)

        # Allocator Heads
        self.objective_condition = MLP(d + 2, 2 * d, d)
        self.costate_head = nn.Linear(d, d)
        self.critic_head = MLP(2 * d + 1, 2 * d, 1)
        self.gate_head = MLP(d, d, 1)

        self.register_buffer("candidate_costs", torch.tensor(config.candidate_costs, dtype=torch.float32))

    def encode_context(self, context_visual, context_state, context_action):
        tokens = (
            self.visual_adapter(flatten_visual(context_visual))
            + self.state_adapter(context_state)
            + self.context_action_adapter(context_action)
        )
        tokens = tokens + self.position[:, : tokens.shape[1]]
        encoded = self.temporal_encoder(tokens)
        return self.temporal_norm(encoded[:, -1])

    def candidate_effects(self, latent):
        batch = latent.shape[0]
        candidates = self.candidate_embedding.weight.unsqueeze(0).expand(batch, -1, -1)
        latent_expanded = latent.unsqueeze(1).expand(-1, self.num_candidates, -1)
        inputs = torch.cat([latent_expanded, candidates], dim=-1)
        return 0.10 * torch.tanh(self.shared_refiner(inputs))

    def apply_mask(self, latent, effects, mask):
        return latent + (effects * mask.unsqueeze(-1)).sum(dim=1)

    def rollout(self, latent, future_actions, context_state=None):
        residuals, logvars, visuals = [], [], []
        current = latent
        for horizon_index in range(future_actions.shape[1]):
            action = self.future_action_adapter(future_actions[:, horizon_index])
            current = self.transition(action, current)
            residuals.append(self.state_residual_head(current))
            logvars.append(torch.clamp(self.state_logvar_head(current), -6.0, 3.0))
            visuals.append(self.visual_head(current))

        stacked_residuals = torch.stack(residuals, dim=1)
        stacked_logvars = torch.stack(logvars, dim=1)
        stacked_visuals = torch.stack(visuals, dim=1)

        if context_state is not None:
            s0 = context_state[:, -1]
            if context_state.shape[1] >= 2:
                v0 = context_state[:, -1] - context_state[:, -2]
            else:
                v0 = torch.zeros_like(s0)
            kinematic_states = self.kinematic_transition(s0, v0, future_actions)
            pred_states = kinematic_states + stacked_residuals
        else:
            kinematic_states = torch.zeros_like(stacked_residuals)
            pred_states = stacked_residuals

        return {
            "state_mean": pred_states,
            "state_logvar": stacked_logvars,
            "visual": stacked_visuals,
            "kinematic_state": kinematic_states,
            "residual_state": stacked_residuals,
        }

    def predict_costate(self, latent, budget_fraction, horizon_fraction):
        condition = torch.cat([latent, budget_fraction.view(-1, 1), horizon_fraction.view(-1, 1)], dim=-1)
        return self.costate_head(self.objective_condition(condition))

    def predict_critic(self, latent, effects):
        batch, candidates, _ = effects.shape
        latent_expanded = latent.unsqueeze(1).expand(-1, candidates, -1)
        costs = self.candidate_costs.view(1, candidates, 1).expand(batch, -1, -1)
        critic_input = torch.cat([latent_expanded, effects, costs], dim=-1)
        return self.critic_head(critic_input).squeeze(-1)

    def predict_gate(self, latent):
        return self.gate_head(latent).squeeze(-1)

    # ---- WorldModel interface ------------------------------------------------------

    def encode_inputs(self, inputs):
        return self.encode_context(inputs["context_visual"], inputs["context_state"], inputs["context_action"])

    def _predict(self, inputs):
        latent = self.encode_inputs(inputs)
        if self.config.prediction_mode == "full":
            latent = latent + self.candidate_effects(latent).sum(dim=1)
        elif self.config.prediction_mode != "base":
            raise ValueError(f"unknown prediction_mode {self.config.prediction_mode!r}")
        return self.rollout(latent, inputs["future_actions"], context_state=inputs.get("context_state"))

    def training_loss(self, batch):
        return hybrid_stage1_loss(self, batch, self.config.rate_beta, self.config.mask_mode)

    def prediction_parameter_names(self):
        return [n for n, p in self.named_parameters() if p.requires_grad and not n.startswith(ALLOCATOR_HEADS)]

    def latent_for_diagnostics(self, batch):
        return self.encode_inputs(batch)


def hybrid_stage1_loss(model: HybridAdjointRecursiveWorldModel, batch: dict, rate_beta: float = 0.002, mask_mode: str = "subset"):
    """Training loss for the Hybrid Adjoint Recursive World Model with kinematic context."""
    device = batch["context_state"].device
    context_visual = batch["context_visual"]
    context_state = batch["context_state"]
    context_action = batch["context_action"]
    future_actions = batch["future_actions"]
    target_state = batch["target_state"]
    target_visual = batch["target_visual"]

    latent = model.encode_context(context_visual, context_state, context_action)
    effects = model.candidate_effects(latent)
    k = model.num_candidates
    zero_mask = torch.zeros(latent.shape[0], k, device=device)

    base_prediction = model.rollout(model.apply_mask(latent, effects, zero_mask), future_actions, context_state=context_state)
    objective_base = prediction_objective(base_prediction, target_state, target_visual)

    if mask_mode == "subset":
        mask, budgets = sample_candidate_mask(latent.shape[0], k, device)
        allocated = model.rollout(model.apply_mask(latent, effects, mask), future_actions, context_state=context_state)
        full = model.rollout(model.apply_mask(latent, effects, torch.ones_like(mask)), future_actions, context_state=context_state)
        objective_allocated = prediction_objective(allocated, target_state, target_visual)
        objective_full = prediction_objective(full, target_state, target_visual)
        allocation_term = objective_allocated.mean() + 0.50 * objective_full.mean()
        mean_budget = budgets.float().mean()
    elif mask_mode == "single_choice":
        mask, choice = sample_single_choice_mask(latent.shape[0], k, device)
        allocated = model.rollout(model.apply_mask(latent, effects, mask), future_actions, context_state=context_state)
        objective_allocated = prediction_objective(allocated, target_state, target_visual)
        objective_full = objective_allocated
        allocation_term = objective_allocated.mean()
        mean_budget = (choice > 0).float().mean()
    else:
        per_candidate = torch.stack(
            [
                prediction_objective(
                    model.rollout(latent + effects[:, j], future_actions, context_state=context_state),
                    target_state,
                    target_visual,
                )
                for j in range(k)
            ],
            dim=1,
        )
        objective_allocated, winner = per_candidate.min(dim=1)
        mask = F.one_hot(winner, k).float()
        objective_full = per_candidate.mean(dim=1)
        allocation_term = objective_allocated.mean() + 0.05 * objective_full.mean()
        mean_budget = torch.ones((), device=device)

    rate_cost = (mask * model.candidate_costs.view(1, -1)).sum(dim=1)
    normalized_effects = F.normalize(effects.float(), dim=-1)
    gram = normalized_effects @ normalized_effects.transpose(1, 2)
    eye = torch.eye(k, device=device).unsqueeze(0)
    diversity_loss = (gram - eye).pow(2).mean()
    collapse_loss = variance_covariance_regularizer(latent)

    loss = (
        allocation_term
        + 0.25 * objective_base.mean()
        + rate_beta * rate_cost.mean()
        + 0.01 * diversity_loss
        + 0.01 * collapse_loss
    )
    return loss, {
        "objective_allocated": objective_allocated.mean().detach(),
        "objective_base": objective_base.mean().detach(),
        "objective_full": objective_full.mean().detach(),
        "rate_cost": rate_cost.mean().detach(),
        "diversity": diversity_loss.detach(),
        "collapse": collapse_loss.detach(),
        "latent_std": latent.float().std().detach(),
        "mean_budget": mean_budget.detach(),
    }


def build_hybrid_adjoint_rwm(dims: ArmDims, width: int | None = None, **overrides) -> HybridAdjointRecursiveWorldModel:
    """Build a Hybrid Adjoint Recursive World Model matching ArmDims."""
    fields = {"context_len": dims.context_len, "horizon": dims.horizon, **overrides}
    if width is not None:
        fields["d_model"] = int(width)
        fields.setdefault("transformer_ff", 4 * int(width))
    config = HybridAdjointConfig(**fields)
    model = HybridAdjointRecursiveWorldModel(dims.state_dim, dims.action_dim, dims.visual_flat_dim, config)
    if dims.target_visual_dim != dims.visual_flat_dim:
        model.visual_head = nn.Linear(config.d_model, dims.target_visual_dim)
    return model
