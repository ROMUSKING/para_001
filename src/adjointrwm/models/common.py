"""Shared interface and objective for every world-model arm (needs PyTorch).

Batch keys follow :class:`adjointrwm.data.WindowDataset`:

* deployable inputs: ``context_state [B,T,S]``, ``context_action [B,T,A]``,
  ``context_visual [B,T,V]`` or ``[B,T,P,D]``, ``future_actions [B,H,A]``;
* training/evaluation targets: ``target_state [B,H,S]``, ``target_visual [B,H,Vt]``,
  ``future_visual`` (input-feature space over the horizon) and ``context_target_visual``.

:meth:`WorldModel.predict` passes **only the deployable inputs** to :meth:`_predict`, so no
arm can read a future target at prediction time, whatever its implementation does.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_KEYS = ("context_state", "context_action", "context_visual", "future_actions")
TARGET_KEYS = ("target_state", "target_visual", "future_visual", "context_target_visual")


@dataclass(frozen=True)
class ArmDims:
    state_dim: int
    action_dim: int
    visual_tokens: int        # P: tokens per frame (cameras x pooled patches)
    visual_token_dim: int     # D: feature width of one token
    target_visual_dim: int    # Vt: the shared evaluation target (frozen ResNet-18 global)
    context_len: int = 8
    horizon: int = 4

    @property
    def visual_flat_dim(self) -> int:
        return self.visual_tokens * self.visual_token_dim

    def to_dict(self) -> dict:
        return asdict(self)


def inputs_only(batch: dict) -> dict:
    return {key: batch[key] for key in INPUT_KEYS if key in batch}


def flatten_visual(visual: torch.Tensor) -> torch.Tensor:
    """``[B,T,P,D] -> [B,T,P*D]``; flat inputs pass through (pilot layout)."""
    return visual.flatten(-2) if visual.dim() == 4 else visual


def as_tokens(visual: torch.Tensor, tokens: int) -> torch.Tensor:
    """``[B,T,P*D] -> [B,T,P,D]``; token inputs pass through."""
    if visual.dim() == 4:
        return visual
    return visual.view(*visual.shape[:-1], tokens, visual.shape[-1] // tokens)


def upcast(x: torch.Tensor) -> torch.Tensor:
    """BF16/FP16 -> FP32; FP32 and FP64 (sampled audits) are left unchanged."""
    return x.float() if x.dtype in (torch.bfloat16, torch.float16) else x


def prediction_objective(pred: dict, target_state: torch.Tensor, target_visual: torch.Tensor, visual_weight: float = 0.25) -> torch.Tensor:
    """The pilot's ``objective_per_sample``: Gaussian NLL on state + ``w * (1 - cos)`` on visual.

    Works on ``[B, H, *]`` or ``[B, L, *]`` tensors and returns one value per sample ``[B]``.
    Identical to the pilot for BF16/FP32 inputs; FP64 inputs stay FP64.
    """
    mean = upcast(pred["state_mean"])
    logvar = upcast(pred["state_logvar"])
    state_nll = 0.5 * (torch.exp(-logvar) * (mean - upcast(target_state)).pow(2) + logvar)
    state_nll = state_nll.mean(dim=(1, 2))
    predicted_visual = F.normalize(upcast(pred["visual"]), dim=-1)
    target = F.normalize(upcast(target_visual), dim=-1)
    visual_loss = (1.0 - (predicted_visual * target).sum(dim=-1)).mean(dim=1)
    return state_nll + visual_weight * visual_loss


def variance_covariance_regularizer(latent: torch.Tensor) -> torch.Tensor:
    """VICReg-style anti-collapse term used by the pilot's stage-1 loss."""
    latent = latent.float()
    latent = latent - latent.mean(dim=0, keepdim=True)
    std = torch.sqrt(latent.var(dim=0, unbiased=False) + 1e-4)
    variance_loss = F.relu(1.0 - std).mean()
    if latent.shape[0] < 2:
        return variance_loss
    covariance = latent.T @ latent / max(1, latent.shape[0] - 1)
    diagonal = torch.diag(torch.diag(covariance))
    covariance_loss = (covariance - diagonal).pow(2).sum() / latent.shape[1]
    return variance_loss + 0.01 * covariance_loss


def gaussian_merge(means: torch.Tensor, logvars: torch.Tensor, sample_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Moment-match a mixture of equally weighted Gaussians (law of total variance)."""
    mean = means.mean(dim=sample_dim)
    var = torch.exp(logvars).mean(dim=sample_dim) + means.var(dim=sample_dim, unbiased=False)
    return mean, torch.log(var.clamp_min(1e-8))


class WorldModel(nn.Module):
    """Base class: subclasses implement ``_predict`` and ``training_loss``."""

    arm: str = "abstract"

    def predict(self, batch: dict) -> dict:
        """``{'state_mean','state_logvar' [B,H,S], 'visual' [B,H,Vt]}`` from deployable inputs only."""
        return self._predict(inputs_only(batch))

    def _predict(self, inputs: dict) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    def training_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:  # pragma: no cover - interface
        raise NotImplementedError

    def param_groups(self, lr: float) -> list[dict]:
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": lr}]

    def prediction_parameter_names(self) -> list[str]:
        """Parameters that can change :meth:`predict` (allocator-only heads are excluded)."""
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def prediction_parameters(self) -> int:
        wanted = set(self.prediction_parameter_names())
        return sum(p.numel() for name, p in self.named_parameters() if name in wanted)

    def latent_for_diagnostics(self, batch: dict) -> torch.Tensor | None:
        """A ``[B, d]`` latent summary used for the effective-rank collapse tripwire."""
        return None


def mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2, act=nn.GELU, dropout: float = 0.0) -> nn.Sequential:
    modules, width = [], in_dim
    for _ in range(max(0, layers - 1)):
        modules += [nn.Linear(width, hidden_dim), act(), nn.Dropout(dropout)]
        width = hidden_dim
    modules.append(nn.Linear(width, out_dim))
    return nn.Sequential(*modules)
