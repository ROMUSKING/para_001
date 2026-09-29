"""TD-MPC2-style implicit (decoder-free) latent dynamics (rival arm ``tdmpc2``).

A PyTorch re-implementation of the TD-MPC2 world model (Hansen et al., arXiv:2310.16828;
official code github.com/nicklashansen/tdmpc2, MIT). Taken from the official ``config.yaml``,
``common/layers.py`` and ``tdmpc2.py`` (checked 2026-09-29):

* ``SimNorm`` (softmax over groups of ``simnorm_dim = 8``) on encoder and dynamics outputs;
* ``NormedLinear`` = Linear -> LayerNorm -> Mish; the official ``mlp()`` layout
  (``num_enc_layers: 2`` -> one hidden encoder layer of ``enc_dim``, two ``mlp_dim`` hidden
  layers in the dynamics);
* consistency loss ``sum_t rho**t * mse(d(z_t, a_t), sg(h(o_{t+1})))`` with ``rho: 0.5`` and
  ``consistency_coef: 20``, targets from the online encoder under ``no_grad``, averaged over
  the horizon;
* ``enc_lr_scale: 0.3`` and ``grad_clip_norm: 20``.

Deviations (plan §2): reward, value and policy heads are replaced by state and visual read-out
heads trained with the shared prediction objective (offline data has no reward); the encoder
sees a frame stack of the whole context by default (``obs_stack = context_len``) so every arm
gets the same information; the consistency rollout uses the benchmark horizon (4), not 3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ArmDims, WorldModel, flatten_visual, prediction_objective


class SimNorm(nn.Module):
    def __init__(self, simnorm_dim: int = 8):
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x):
        shape = x.shape
        x = x.view(*shape[:-1], -1, self.dim)
        return F.softmax(x, dim=-1).view(*shape)


class NormedLinear(nn.Linear):
    def __init__(self, *args, dropout: float = 0.0, act: nn.Module | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act if act is not None else nn.Mish(inplace=False)
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.act(self.ln(x))


def tdmpc_mlp(in_dim: int, mlp_dims: list[int], out_dim: int, act: nn.Module | None = None, dropout: float = 0.0) -> nn.Sequential:
    """The official ``mlp()``: NormedLinear hidden layers (dropout on the first), then the output."""
    dims = [in_dim, *mlp_dims, out_dim]
    layers = [NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)) for i in range(len(dims) - 2)]
    layers.append(NormedLinear(dims[-2], dims[-1], act=act) if act is not None else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class TDMPC2WorldModel(WorldModel):
    arm = "tdmpc2"

    def __init__(
        self,
        dims: ArmDims,
        latent_dim: int = 512,
        mlp_dim: int = 512,
        enc_dim: int = 256,
        num_enc_layers: int = 2,
        simnorm_dim: int = 8,
        obs_stack: int | None = None,
        rho: float = 0.5,
        consistency_coef: float = 20.0,
        readout_weight: float = 1.0,
        enc_lr_scale: float = 0.3,
        visual_weight: float = 0.25,
    ):
        super().__init__()
        if latent_dim % simnorm_dim:
            raise ValueError("latent_dim must be divisible by simnorm_dim")
        self.dims = dims
        self.obs_stack = obs_stack or dims.context_len
        if not 1 <= self.obs_stack <= dims.context_len:
            raise ValueError("obs_stack must be between 1 and context_len")
        self.rho, self.consistency_coef = rho, consistency_coef
        self.readout_weight, self.enc_lr_scale, self.visual_weight = readout_weight, enc_lr_scale, visual_weight
        obs_dim = dims.state_dim + dims.visual_flat_dim
        self.encoder = tdmpc_mlp(
            self.obs_stack * obs_dim, max(num_enc_layers - 1, 1) * [enc_dim], latent_dim, act=SimNorm(simnorm_dim)
        )
        self.dynamics = tdmpc_mlp(latent_dim + dims.action_dim, 2 * [mlp_dim], latent_dim, act=SimNorm(simnorm_dim))
        self.state_head = tdmpc_mlp(latent_dim, 2 * [mlp_dim], 2 * dims.state_dim)
        self.visual_head = tdmpc_mlp(latent_dim, 2 * [mlp_dim], dims.target_visual_dim)

    def _observations(self, states, visual):
        return torch.cat([states, flatten_visual(visual)], dim=-1)

    def _stack(self, obs, end):
        """Flattened frames ``end - obs_stack + 1 .. end`` (``end`` inclusive, never negative)."""
        start = end - self.obs_stack + 1
        if start < 0:
            raise IndexError("frame stack reaches before the context start")
        return obs[:, start : end + 1].flatten(1)

    def _readout(self, latents):
        mean, logvar = self.state_head(latents).chunk(2, dim=-1)
        return {"state_mean": mean, "state_logvar": torch.clamp(logvar, -6.0, 3.0), "visual": self.visual_head(latents)}

    def _rollout(self, z, future_actions):
        latents = []
        for j in range(future_actions.shape[1]):
            z = self.dynamics(torch.cat([z, future_actions[:, j]], dim=-1))
            latents.append(z)
        return torch.stack(latents, dim=1)

    def _predict(self, inputs):
        obs = self._observations(inputs["context_state"], inputs["context_visual"])
        z = self.encoder(self._stack(obs, obs.shape[1] - 1))
        return self._readout(self._rollout(z, inputs["future_actions"]))

    def training_loss(self, batch):
        T = batch["context_state"].shape[1]
        obs = self._observations(
            torch.cat([batch["context_state"], batch["target_state"]], dim=1),
            torch.cat([flatten_visual(batch["context_visual"]), flatten_visual(batch["future_visual"])], dim=1),
        )
        horizon = batch["future_actions"].shape[1]
        z0 = self.encoder(self._stack(obs, T - 1))
        predicted = self._rollout(z0, batch["future_actions"])
        with torch.no_grad():
            targets = torch.stack([self.encoder(self._stack(obs, T + j)) for j in range(horizon)], dim=1)
        consistency = sum(
            F.mse_loss(predicted[:, j].float(), targets[:, j].float()) * self.rho**j for j in range(horizon)
        ) / horizon
        readout = prediction_objective(
            self._readout(predicted), batch["target_state"], batch["target_visual"], self.visual_weight
        ).mean()
        loss = self.consistency_coef * consistency + self.readout_weight * readout
        return loss, {"consistency": consistency.detach(), "readout": readout.detach()}

    def param_groups(self, lr):
        encoder = list(self.encoder.parameters())
        encoder_ids = {id(p) for p in encoder}
        rest = [p for p in self.parameters() if id(p) not in encoder_ids]
        return [{"params": encoder, "lr": lr * self.enc_lr_scale}, {"params": rest, "lr": lr}]

    def latent_for_diagnostics(self, batch):
        obs = self._observations(batch["context_state"], batch["context_visual"])
        return self.encoder(self._stack(obs, obs.shape[1] - 1))


def build_tdmpc2(dims: ArmDims, width: int | None = None, **overrides) -> TDMPC2WorldModel:
    if width is not None:
        w = int(width) // 8 * 8
        overrides.setdefault("latent_dim", w)
        overrides.setdefault("mlp_dim", w)
        overrides.setdefault("enc_dim", max(8, w // 2))
    return TDMPC2WorldModel(dims, **overrides)
