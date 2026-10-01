"""Pilot v2 allocation contract: exact gains with a hold option, and rival allocators.

Implements roadmap E2.1 (see docs/plans/rival-benchmark-plan.md §5). Lifted from section 5 of
the production pilot (``exact_adjoint_targets``, allocator losses, ``evaluate_allocator``) with
the v1 problems fixed:

* candidate 0 is ``hold``: zero effect, zero cost, exact gain 0 by definition;
* co-state estimator, direct critic and gate are trained as **separate jobs** on the same
  batches, steps and optimiser, and the critic's width is chosen to match the estimator's
  parameter count (the pilot's critic head had about 20 % fewer parameters);
* the gate label is the sign of ``critic_regret - adjoint_regret`` (ties get zero weight) with
  a positive-class weight estimated on validation;
* traces keep the full ``exact_gain[N, K+1]``.

Privilege (comprehensive plan §3.9): ``exact_costate`` and ``exact_gain`` are computed from the
realised future targets. They are training labels and diagnostic ceilings only; no deployable
policy reads them at decision time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .analysis.allocation import gain_matrix
from .models.adjoint_rwm import MLP
from .models.common import prediction_objective, upcast
from .training import _fork_rng, move_batch

HOLD = 0
HEAD_TYPES = ("costate", "critic", "gate")
DEPLOYABLE_POLICIES = ("adjoint", "critic", "hybrid", "uncertainty", "adjoint_randomized")
DIAGNOSTIC_POLICIES = ("exact_costate",)


def weighted_costs_with_hold(candidate_costs: torch.Tensor, cost_weight: float) -> torch.Tensor:
    return torch.cat([candidate_costs.new_zeros(1), candidate_costs * cost_weight])


def exact_targets(teacher, batch: dict, cost_weight: float) -> dict:
    """Exact co-state (autograd) and per-candidate gains, including hold, for one batch.

    ``exact_gain[:, k] = J(stop) - J(z + delta_k) - c_k`` with ``delta_0 = 0, c_0 = 0``, so
    ``exact_gain[:, 0] == 0``. Also returns ``self_objective[:, k]``: the model's *own* expected
    state NLL for each option, ``mean(0.5 * (1 + logvar))``, which needs no targets and is in
    the same units as the state term of ``J``; it feeds the uncertainty-only allocator.

    Labels are computed in FP32 with autocast disabled, even inside a BF16 training loop
    (operator brief: FP32 for sensitive quantities).
    """
    with torch.autocast(device_type=batch["context_state"].device.type, enabled=False):
        low = (torch.bfloat16, torch.float16)
        batch = {k: (v.float() if torch.is_tensor(v) and v.dtype in low else v) for k, v in batch.items()}
        return _exact_targets_fp32(teacher, batch, cost_weight)


def _self_objective(prediction) -> torch.Tensor:
    return (0.5 * (1.0 + upcast(prediction["state_logvar"]))).mean(dim=(1, 2))


def _exact_targets_fp32(teacher, batch, cost_weight):
    with torch.no_grad():
        latent = teacher.encode_context(batch["context_visual"], batch["context_state"], batch["context_action"])
        effects = teacher.candidate_effects(latent)
    latent_req = latent.detach().requires_grad_(True)
    with torch.enable_grad():
        stop_prediction = teacher.rollout(latent_req, batch["future_actions"])
        stop_objective = prediction_objective(stop_prediction, batch["target_state"], batch["target_visual"])
        exact_costate = torch.autograd.grad(stop_objective.sum(), latent_req)[0].detach()
    objectives, self_objective = [stop_objective.detach()], [_self_objective(stop_prediction).detach()]
    with torch.no_grad():
        for k in range(teacher.num_candidates):
            pred = teacher.rollout(latent + effects[:, k], batch["future_actions"])
            objectives.append(prediction_objective(pred, batch["target_state"], batch["target_visual"]))
            self_objective.append(_self_objective(pred))
    costs = weighted_costs_with_hold(teacher.candidate_costs, cost_weight)
    objectives = torch.stack(objectives, dim=1)
    exact_gain = objectives[:, :1] - objectives - costs.view(1, -1)
    effects_with_hold = torch.cat([torch.zeros_like(effects[:, :1]), effects], dim=1)
    return {
        "latent": latent.detach(),
        "effects": effects_with_hold.detach(),
        "costs": costs,
        "exact_costate": exact_costate,
        "exact_gain": exact_gain,
        "stop_objective": stop_objective.detach(),
        "self_objective": torch.stack(self_objective, dim=1),
    }


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class CostateEstimator(nn.Module):
    """The pilot's amortised co-state head: condition(latent, budget, horizon) -> lambda_hat."""

    def __init__(self, d: int):
        super().__init__()
        self.objective_condition = MLP(d + 2, 2 * d, d)
        self.costate_head = nn.Linear(d, d)

    def forward(self, latent, budget_fraction, horizon_fraction):
        condition = torch.cat([latent, budget_fraction.view(-1, 1), horizon_fraction.view(-1, 1)], dim=-1)
        return self.costate_head(self.objective_condition(condition))


class DirectCritic(nn.Module):
    """Direct marginal-gain critic: (latent, effect, cost, budget, horizon) -> gain.

    Information-equivalent to the co-state path (comprehensive plan §5.5): the same latent,
    candidate effect, cost, budget and horizon; only the co-state is absent.
    """

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.net = MLP(2 * d + 3, hidden, 1)

    def forward(self, latent, effects, costs, budget_fraction, horizon_fraction):
        b, k, _ = effects.shape
        extra = torch.stack([budget_fraction, horizon_fraction], dim=-1).unsqueeze(1).expand(b, k, 2)
        x = torch.cat([latent.unsqueeze(1).expand(b, k, -1), effects, costs.view(1, k, 1).expand(b, k, 1), extra], dim=-1)
        return self.net(x).squeeze(-1)


class Gate(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = MLP(d, d, 1)

    def forward(self, latent):
        return self.net(latent).squeeze(-1)


def matched_critic_hidden(d: int) -> int:
    """Critic hidden width whose parameter count is closest to the co-state estimator's."""
    target = sum(p.numel() for p in CostateEstimator(d).parameters())
    # DirectCritic params = (2d + 3) h + h + h + 1
    return max(1, round((target - 1) / (2 * d + 5)))


def first_order_scores(costate, effects, costs):
    """``-<lambda, delta_k> - c_k``; hold scores exactly 0."""
    return -(costate.unsqueeze(1) * effects).sum(-1) - costs.view(1, -1)


def normalized_first_order_scores(costate: torch.Tensor, effects: torch.Tensor, costs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale-invariant cosine coupling score from PARA architecture:

    score_k = - <costate, delta_k> / (||costate|| * ||delta_k|| + eps) - c_k
    For candidate 0 (hold, delta_0 = 0), returns exactly 0.0.
    """
    costate_norm = torch.norm(costate, dim=-1, keepdim=True).unsqueeze(1) + eps
    effects_norm = torch.norm(effects, dim=-1, keepdim=True) + eps
    dot = (costate.unsqueeze(1) * effects).sum(-1, keepdim=True)
    cosine = - dot / (costate_norm * effects_norm)
    scores = cosine.squeeze(-1) - costs.view(1, -1)
    # Ensure hold option (k=0) is identically 0.0
    return torch.cat([scores.new_zeros(scores.shape[0], 1), scores[:, 1:]], dim=1)


def lcb_decision_scores(scores: torch.Tensor, uncert: torch.Tensor, kappa: float = 0.5) -> torch.Tensor:
    """Cost-aware lower confidence bound decision scores from PARA architecture:

    adjusted_score_k = score_k - kappa * uncert_k  for k >= 1
    adjusted_score_0 = 0.0 (hold)
    """
    if kappa <= 0.0:
        return scores
    active_lcb = scores[:, 1:] - kappa * uncert[:, 1:]
    return torch.cat([scores.new_zeros(scores.shape[0], 1), active_lcb], dim=1)


def with_hold_zero(scores_refinements):
    """Prepend the exactly-known hold score (0) to learned refinement scores."""
    return torch.cat([torch.zeros_like(scores_refinements[:, :1]), scores_refinements], dim=1)


# ---------------------------------------------------------------------------
# Jobs (one module per trained head, so each runs through training.train_job)
# ---------------------------------------------------------------------------

class AllocatorJob(nn.Module):
    """A frozen dynamics teacher plus one trainable head (``costate``, ``critic`` or ``gate``).

    The teacher is held but **not registered** as a submodule: job checkpoints contain only the
    heads (the teacher is pinned by its own checkpoint hash in the job identity), and moving the
    job to a device does not move the teacher, so place the teacher on the device first.

    ``training_loss`` follows the pilot's allocator loss, split by head:

    * co-state: ``(1 - cos) + 0.1 * smoothL1(log |lambda|) + 0.5 * CE(first-order scores, oracle)``;
    * critic: ``smoothL1(gain) + 0.5 * CE(scores, oracle)``;
    * gate: weighted BCE on ``1[critic_regret > adjoint_regret]``, ties weighted 0.

    The gate job needs trained ``costate`` and ``critic`` heads (frozen) and ``gate_pos_weight``.
    """

    def __init__(self, teacher, head_type: str, cost_weight: float, costate=None, critic=None, gate_pos_weight: float = 1.0):
        super().__init__()
        if head_type not in HEAD_TYPES:
            raise ValueError(f"head_type must be one of {HEAD_TYPES}")
        d = teacher.d_model
        self.__dict__["teacher"] = teacher  # unregistered: excluded from state_dict and parameters()
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.head_type, self.cost_weight = head_type, cost_weight
        self.costate = costate if costate is not None else CostateEstimator(d)
        self.critic = critic if critic is not None else DirectCritic(d, matched_critic_hidden(d))
        self.gate = Gate(d)
        trainable = {"costate": self.costate, "critic": self.critic, "gate": self.gate}[head_type]
        for name, module in (("costate", self.costate), ("critic", self.critic), ("gate", self.gate)):
            for p in module.parameters():
                p.requires_grad_(module is trainable)
        self.register_buffer("gate_pos_weight", torch.tensor(float(gate_pos_weight)))

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()  # the teacher's labels must not see dropout noise
        return self

    def param_groups(self, lr):
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": lr}]

    def _conditions(self, latent):
        # The pilot's conditioning: budget fraction 1/K, full horizon.
        budget = torch.full((latent.shape[0],), 1.0 / self.teacher.num_candidates, device=latent.device)
        return budget, torch.ones_like(budget)

    def scores(self, targets, kappa: float = 0.5) -> dict:
        """Scores of every deployable policy (and the exact-co-state diagnostic) over K+1 options."""
        latent, effects, costs = targets["latent"], targets["effects"], targets["costs"]
        budget, horizon = self._conditions(latent)
        costate = self.costate(latent, budget, horizon)
        adjoint = first_order_scores(costate, effects, costs)
        adjoint_norm = normalized_first_order_scores(costate, effects, costs)
        critic = with_hold_zero(self.critic(latent, effects[:, 1:], costs[1:], budget, horizon))
        if latent.shape[0] > 1:
            perm = torch.roll(torch.arange(latent.shape[0], device=latent.device), 1)
            randomized = first_order_scores(costate[perm], effects, costs)
            randomized_norm = normalized_first_order_scores(costate[perm], effects, costs)
        else:
            randomized = first_order_scores(torch.zeros_like(costate), effects, costs)
            randomized_norm = normalized_first_order_scores(torch.zeros_like(costate), effects, costs)
        own = targets["self_objective"]
        uncertainty = own[:, :1] - own - costs.view(1, -1)  # predicted reduction of own NLL
        gate_logit = self.gate(latent)

        adjoint_lcb = lcb_decision_scores(adjoint, own, kappa=kappa)
        adjoint_norm_lcb = lcb_decision_scores(adjoint_norm, own, kappa=kappa)
        critic_lcb = lcb_decision_scores(critic, own, kappa=kappa)

        return {
            "adjoint": adjoint,
            "adjoint_norm": adjoint_norm,
            "adjoint_lcb": adjoint_lcb,
            "adjoint_norm_lcb": adjoint_norm_lcb,
            "critic": critic,
            "critic_lcb": critic_lcb,
            "adjoint_randomized": randomized,
            "adjoint_randomized_norm": randomized_norm,
            "uncertainty": uncertainty,
            "exact_costate": first_order_scores(targets["exact_costate"], effects, costs),
            "exact_costate_norm": normalized_first_order_scores(targets["exact_costate"], effects, costs),
            "predicted_costate": costate,
            "gate_logit": gate_logit,
        }

    def training_loss(self, batch):
        targets = exact_targets(self.teacher, batch, self.cost_weight)
        gain = targets["exact_gain"]
        oracle = gain.argmax(dim=1)
        s = self.scores(targets)
        if self.head_type == "costate":
            exact, pred = targets["exact_costate"], s["predicted_costate"]
            direction = (1.0 - F.cosine_similarity(pred, exact, dim=-1)).mean()
            magnitude = F.smooth_l1_loss(torch.log(pred.norm(dim=-1).clamp_min(1e-8)), torch.log(exact.norm(dim=-1).clamp_min(1e-8)))
            ranking = F.cross_entropy(s["adjoint"], oracle)
            loss = direction + 0.1 * magnitude + 0.5 * ranking
            parts = {"direction": direction, "magnitude": magnitude, "ranking": ranking}
        elif self.head_type == "critic":
            regression = F.smooth_l1_loss(s["critic"][:, 1:], gain[:, 1:])
            ranking = F.cross_entropy(s["critic"], oracle)
            loss = regression + 0.5 * ranking
            parts = {"regression": regression, "ranking": ranking}
        else:
            best = gain.max(dim=1).values
            adjoint_regret = best - gain.gather(1, s["adjoint"].argmax(1, keepdim=True)).squeeze(1)
            critic_regret = best - gain.gather(1, s["critic"].argmax(1, keepdim=True)).squeeze(1)
            diff = critic_regret - adjoint_regret
            label = (diff > 0).float()
            weight = (diff.abs() > 1e-9).float()
            loss_each = F.binary_cross_entropy_with_logits(s["gate_logit"], label, pos_weight=self.gate_pos_weight, reduction="none")
            loss = (loss_each * weight).sum() / weight.sum().clamp_min(1.0)
            parts = {"gate_bce": loss, "positive_rate": label[weight > 0].mean() if weight.sum() > 0 else torch.zeros(())}
        return loss, {k: v.detach() for k, v in parts.items()}

    def clip_gradients(self, max_norm):
        params = [p for p in self.parameters() if p.requires_grad and p.grad is not None]
        return torch.nn.utils.clip_grad_norm_(params, max_norm)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def allocation_traces(job: AllocatorJob, dataset, batch_size: int, device, gate_threshold: float = 0.5, eval_seed: int = 0) -> pd.DataFrame:
    """Per-window traces: every gain, every policy's choice, gate probability.

    Regrets are computed afterwards from the gains alone (method-blind; see
    :func:`adjointrwm.analysis.summarize_policies`).
    """
    was_training = job.training
    job.eval()
    rows = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with _fork_rng(device):
        torch.manual_seed(eval_seed)
        for batch in loader:
            batch = move_batch(batch, device)
            targets = exact_targets(job.teacher, batch, job.cost_weight)
            with torch.no_grad():
                s = job.scores(targets)
            gain = targets["exact_gain"].cpu().numpy()
            gate_p = torch.sigmoid(s["gate_logit"]).cpu().numpy()
            choices = {name: s[name].argmax(1).cpu().numpy() for name in ("adjoint", "critic", "adjoint_randomized", "uncertainty", "exact_costate")}
            choices["hybrid"] = np.where(gate_p >= gate_threshold, choices["adjoint"], choices["critic"])
            choices["oracle"] = gain.argmax(1)
            for i in range(gain.shape[0]):
                row = {"episode_id": batch["episode_id"][i], "window_start": int(batch["window_start"][i]),
                       "gate_probability": float(gate_p[i])}
                row.update({f"gain_{k}": float(gain[i, k]) for k in range(gain.shape[1])})
                row.update({f"{name}_choice": int(c[i]) for name, c in choices.items()})
                rows.append(row)
    job.train(was_training)
    return pd.DataFrame(rows)


def gate_positive_weight(traces: pd.DataFrame) -> dict:
    """Class balance of the gate label on (validation) traces: ``n_negative / n_positive``."""
    gains = gain_matrix(traces)
    best = gains.max(axis=1)
    idx = np.arange(len(gains))
    adjoint_regret = best - gains[idx, traces["adjoint_choice"].to_numpy()]
    critic_regret = best - gains[idx, traces["critic_choice"].to_numpy()]
    diff = critic_regret - adjoint_regret
    positives, negatives = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
    return {"positives": positives, "negatives": negatives, "ties": int(len(diff) - positives - negatives),
            "pos_weight": float(negatives / positives) if positives else 1.0}


def validation_regret(job: AllocatorJob, dataset, batch_size: int, device, policy: str) -> float:
    """Mean regret of one policy on ``dataset``: the checkpoint-selection score of each job."""
    traces = allocation_traces(job, dataset, batch_size, device)
    gains = gain_matrix(traces)
    choice = traces[f"{policy}_choice"].to_numpy()
    return float((gains.max(axis=1) - gains[np.arange(len(gains)), choice]).mean())
