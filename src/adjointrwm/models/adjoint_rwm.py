"""The pilot's recursive world model and stage-1 dynamics objective (reference arm).

Lifted from sections 3 and 4 of ``notebooks/01-production/AdjointRWM_Production_Pilot.ipynb``.
Module and parameter names are unchanged, so ``best_dynamics.pt`` / ``best_allocator.pt``
from run ``droid100_adjoint_20260929T070629Z`` load with ``strict=True``. The only additions
are methods (no parameters): the :class:`~adjointrwm.models.common.WorldModel` interface and
the v2 single-choice candidate regime used by pilot v2 (roadmap E2.1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import WorldModel, flatten_visual, prediction_objective, variance_covariance_regularizer

ALLOCATOR_HEADS = ("objective_condition", "costate_head", "critic_head", "gate_head")
MASK_MODES = ("subset", "single_choice", "single_choice_wta")


@dataclass(frozen=True)
class AdjointRWMConfig:
    """Model fields of the pilot ``Config`` (defaults are the pilot's values)."""

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
    mask_mode: str = "subset"          # pilot v1; "single_choice" is the v2 contract
    prediction_mode: str = "full"      # pilot dynamics gate used latent + sum of all effects

    def to_dict(self) -> dict:
        return asdict(self)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value):
        return self.net(value)


class AdjointRecursiveWorldModel(WorldModel):
    arm = "adjoint_rwm"

    def __init__(self, state_dim, action_dim, visual_dim, config):
        super().__init__()
        d = config.d_model
        self.config = config
        self.d_model = d
        self.horizon = config.horizon
        self.num_candidates = config.num_refinement_candidates

        self.visual_adapter = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, d), nn.GELU())
        self.state_adapter = nn.Sequential(nn.Linear(state_dim, d), nn.LayerNorm(d), nn.GELU())
        self.context_action_adapter = nn.Sequential(nn.Linear(action_dim, d), nn.LayerNorm(d), nn.GELU())
        self.future_action_adapter = nn.Sequential(nn.Linear(action_dim, d), nn.LayerNorm(d), nn.GELU())

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

        self.candidate_embedding = nn.Embedding(self.num_candidates, d)
        self.shared_refiner = MLP(2 * d, 4 * d, d, dropout=config.dropout)

        self.transition = nn.GRUCell(d, d)
        self.state_mean_head = nn.Linear(d, state_dim)
        self.state_logvar_head = nn.Linear(d, state_dim)
        self.visual_head = nn.Linear(d, visual_dim)

        self.objective_condition = MLP(d + 2, 2 * d, d)
        self.costate_head = nn.Linear(d, d)
        self.critic_head = MLP(2 * d + 1, 2 * d, 1)
        self.gate_head = MLP(d, d, 1)

        self.register_buffer("candidate_costs", torch.tensor(config.candidate_costs, dtype=torch.float32))

    # ---- pilot methods (unchanged) -------------------------------------------------

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

    def rollout(self, latent, future_actions):
        states, logvars, visuals = [], [], []
        current = latent
        for horizon_index in range(future_actions.shape[1]):
            action = self.future_action_adapter(future_actions[:, horizon_index])
            current = self.transition(action, current)
            states.append(self.state_mean_head(current))
            logvars.append(torch.clamp(self.state_logvar_head(current), -6.0, 3.0))
            visuals.append(self.visual_head(current))
        return {
            "state_mean": torch.stack(states, dim=1),
            "state_logvar": torch.stack(logvars, dim=1),
            "visual": torch.stack(visuals, dim=1),
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
        return self.rollout(latent, inputs["future_actions"])

    def training_loss(self, batch):
        return stage1_loss(self, batch, self.config.rate_beta, self.config.mask_mode)

    def prediction_parameter_names(self):
        return [n for n, p in self.named_parameters() if p.requires_grad and not n.startswith(ALLOCATOR_HEADS)]

    def latent_for_diagnostics(self, batch):
        return self.encode_inputs(batch)


def sample_candidate_mask(batch_size, num_candidates, device):
    """Pilot v1: a random subset of 0..K candidates per window."""
    budgets = torch.randint(low=0, high=num_candidates + 1, size=(batch_size,), device=device)
    random_scores = torch.rand(batch_size, num_candidates, device=device)
    ordering = torch.argsort(random_scores, dim=1, descending=True)
    mask = torch.zeros_like(random_scores)
    for row in range(batch_size):
        count = int(budgets[row].item())
        if count:
            mask[row, ordering[row, :count]] = 1.0
    return mask, budgets


def sample_single_choice_mask(batch_size, num_candidates, device):
    """v2: hold (all zeros) or exactly one candidate, uniformly over the K+1 options."""
    choice = torch.randint(low=0, high=num_candidates + 1, size=(batch_size,), device=device)
    mask = F.one_hot(choice, num_candidates + 1)[:, 1:].float()
    return mask, choice


def stage1_loss(model, batch, rate_beta=0.002, mask_mode="subset"):
    """Pilot stage-1 loss (``mask_mode='subset'``) and the v2 single-choice variants.

    * ``subset`` (pilot v1): allocated + 0.25 base + 0.5 full + rate + regularisers.
    * ``single_choice`` (v2): hold-or-one candidate, uniformly, + 0.25 base; no full-set term,
      because v2 never applies all candidates at once.
    * ``single_choice_wta`` (v2 variant): winner-takes-all over the K single-candidate rollouts
      (plus 0.05 x their mean, so every candidate keeps a gradient) + 0.25 base. This is the
      multiple-choice-learning regime, which rewards candidates that help on *different*
      windows. It changes what the candidates are, so a run using it must say so.
    """
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}")
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

    base_prediction = model.rollout(model.apply_mask(latent, effects, zero_mask), future_actions)
    objective_base = prediction_objective(base_prediction, target_state, target_visual)

    if mask_mode == "subset":
        mask, budgets = sample_candidate_mask(latent.shape[0], k, device)
        allocated = model.rollout(model.apply_mask(latent, effects, mask), future_actions)
        full = model.rollout(model.apply_mask(latent, effects, torch.ones_like(mask)), future_actions)
        objective_allocated = prediction_objective(allocated, target_state, target_visual)
        objective_full = prediction_objective(full, target_state, target_visual)
        allocation_term = objective_allocated.mean() + 0.50 * objective_full.mean()
        mean_budget = budgets.float().mean()
    elif mask_mode == "single_choice":
        mask, choice = sample_single_choice_mask(latent.shape[0], k, device)
        allocated = model.rollout(model.apply_mask(latent, effects, mask), future_actions)
        objective_allocated = prediction_objective(allocated, target_state, target_visual)
        objective_full = objective_allocated
        allocation_term = objective_allocated.mean()
        mean_budget = (choice > 0).float().mean()
    else:
        per_candidate = torch.stack(
            [
                prediction_objective(model.rollout(latent + effects[:, j], future_actions), target_state, target_visual)
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


def build_adjoint_rwm(dims, width=None, **overrides) -> AdjointRecursiveWorldModel:
    fields = {"context_len": dims.context_len, "horizon": dims.horizon, **overrides}
    if width is not None:
        fields["d_model"] = int(width)
        fields.setdefault("transformer_ff", 4 * int(width))
    config = AdjointRWMConfig(**fields)
    # The pilot decodes the visual target with the same width as its visual input; with a
    # different input encoder the head predicts the shared target instead.
    model = AdjointRecursiveWorldModel(dims.state_dim, dims.action_dim, dims.visual_flat_dim, config)
    if dims.target_visual_dim != dims.visual_flat_dim:
        model.visual_head = nn.Linear(config.d_model, dims.target_visual_dim)
    return model
