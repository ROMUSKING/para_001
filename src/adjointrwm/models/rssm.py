"""DreamerV3-style recurrent state-space model (rival arm ``dreamerv3_rssm``).

A PyTorch re-implementation of the *world-model* part of DreamerV3 (Hafner et al.,
arXiv:2301.04104; official JAX code github.com/danijar/dreamerv3, MIT). Values taken from the
official ``configs.yaml`` / ``rssm.py`` (checked 2026-09-29):

* ``stoch: 32`` categorical latents, ``unimix: 0.01``, ``free_nats: 1.0``;
* loss scales ``dyn: 1.0``, ``rep: 0.1``, ``rec: 1.0``;
* size presets scale ``deter = 8 * hidden`` and ``classes = hidden / 16``;
* block-GRU core (``BlockLinear`` over ``blocks`` groups, update-gate bias ``-1``), prior from the
  deterministic state (``imglayers: 2``), posterior from deterministic state + observation
  tokens (``obslayers: 1``), RMSNorm + SiLU layers, symlog on vector inputs;
* KL balancing: ``dyn = KL(sg(post) || prior)``, ``rep = KL(post || sg(prior))``, each floored at
  ``free_nats`` per (sample, step).

Deviations (listed in docs/plans/rival-benchmark-plan.md §2): no reward/continue/actor/critic
(offline prediction only); the decoder heads use the shared prediction objective instead of
symlog-MSE; open-loop predictions are the moment-matched mean of ``eval_samples`` prior samples.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ArmDims, WorldModel, flatten_visual, gaussian_merge, prediction_objective


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight).to(x.dtype)


class Dense(nn.Module):
    """Linear -> RMSNorm -> SiLU (DreamerV3's default layer)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = RMSNorm(out_dim)

    def forward(self, x):
        return F.silu(self.norm(self.linear(x)))


class BlockLinear(nn.Module):
    """Block-diagonal linear layer: ``blocks`` independent groups (as in DreamerV3's GRU)."""

    def __init__(self, in_features: int, out_features: int, blocks: int):
        super().__init__()
        if in_features % blocks or out_features % blocks:
            raise ValueError("features must be divisible by blocks")
        self.blocks, self.out_features = blocks, out_features
        self.weight = nn.Parameter(torch.empty(blocks, in_features // blocks, out_features // blocks))
        self.bias = nn.Parameter(torch.zeros(out_features))
        bound = 1.0 / math.sqrt(in_features // blocks)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x):
        x = x.reshape(*x.shape[:-1], self.blocks, -1)
        y = torch.einsum("...gi,gio->...go", x, self.weight)
        return y.reshape(*y.shape[:-2], self.out_features) + self.bias


class RSSMWorldModel(WorldModel):
    arm = "dreamerv3_rssm"

    def __init__(
        self,
        dims: ArmDims,
        hidden: int = 384,
        deter: int | None = None,
        stoch: int = 32,
        classes: int | None = None,
        blocks: int = 8,
        units: int | None = None,
        enc_layers: int = 3,
        dec_layers: int = 3,
        imglayers: int = 2,
        obslayers: int = 1,
        dynlayers: int = 1,
        unimix: float = 0.01,
        free_nats: float = 1.0,
        dyn_scale: float = 1.0,
        rep_scale: float = 0.1,
        rec_scale: float = 1.0,
        eval_samples: int = 8,
        visual_weight: float = 0.25,
    ):
        super().__init__()
        self.dims = dims
        self.hidden = hidden
        self.deter = deter or 8 * hidden
        self.stoch = stoch
        self.classes = classes or max(2, hidden // 16)
        self.blocks = blocks
        units = units or hidden
        self.unimix, self.free_nats = unimix, free_nats
        self.dyn_scale, self.rep_scale, self.rec_scale = dyn_scale, rep_scale, rec_scale
        self.eval_samples = eval_samples
        self.visual_weight = visual_weight
        if self.deter % blocks:
            raise ValueError("deter must be divisible by blocks")

        obs_dim = dims.state_dim + dims.visual_flat_dim
        self.encoder = nn.Sequential(*[Dense(obs_dim if i == 0 else units, units) for i in range(enc_layers)])
        flat_stoch = self.stoch * self.classes
        self.dynin0 = Dense(self.deter, hidden)
        self.dynin1 = Dense(flat_stoch, hidden)
        self.dynin2 = Dense(dims.action_dim, hidden)
        group_in = self.deter // blocks + 3 * hidden
        self.dynhid = nn.ModuleList(
            [BlockLinear(blocks * group_in if i == 0 else self.deter, self.deter, blocks) for i in range(dynlayers)]
        )
        self.dynhid_norm = nn.ModuleList([RMSNorm(self.deter) for _ in range(dynlayers)])
        self.dyngru = BlockLinear(self.deter, 3 * self.deter, blocks)
        self.prior_net = nn.Sequential(*[Dense(self.deter if i == 0 else hidden, hidden) for i in range(imglayers)])
        self.prior_logits = nn.Linear(hidden, flat_stoch)
        self.post_net = nn.Sequential(*[Dense(self.deter + units if i == 0 else hidden, hidden) for i in range(obslayers)])
        self.post_logits = nn.Linear(hidden, flat_stoch)
        feat = self.deter + flat_stoch
        self.decoder = nn.Sequential(*[Dense(feat if i == 0 else units, units) for i in range(dec_layers)])
        self.state_head = nn.Linear(units, 2 * dims.state_dim)
        self.visual_head = nn.Linear(units, dims.target_visual_dim)

    # ---- latent dynamics -----------------------------------------------------------

    def _core(self, deter, stoch_flat, action):
        action = action / torch.clamp(action.abs(), min=1.0).detach()
        x = torch.cat([self.dynin0(deter), self.dynin1(stoch_flat), self.dynin2(action)], dim=-1)
        g = self.blocks
        x = x.unsqueeze(-2).expand(*x.shape[:-1], g, x.shape[-1])
        x = torch.cat([deter.reshape(*deter.shape[:-1], g, -1), x], dim=-1).flatten(-2)
        for layer, norm in zip(self.dynhid, self.dynhid_norm):
            x = F.silu(norm(layer(x)))
        x = self.dyngru(x)
        x = x.reshape(*x.shape[:-1], g, 3, -1)
        reset, cand, update = (x[..., i, :].flatten(-2) for i in range(3))
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.0)
        return update * cand + (1.0 - update) * deter

    def _probs(self, logits):
        probs = F.softmax(logits.float().view(*logits.shape[:-1], self.stoch, self.classes), dim=-1)
        return (1.0 - self.unimix) * probs + self.unimix / self.classes

    def _sample(self, probs):
        index = torch.multinomial(probs.reshape(-1, self.classes), 1).view(probs.shape[:-1])
        onehot = F.one_hot(index, self.classes).to(probs.dtype)
        return (onehot + probs - probs.detach()).flatten(-2)  # straight-through

    @staticmethod
    def _kl(p, q):
        return (p * (torch.log(p) - torch.log(q))).sum(dim=(-2, -1))

    def _obs(self, inputs_state, inputs_visual):
        return torch.cat([symlog(inputs_state), symlog(flatten_visual(inputs_visual))], dim=-1)

    def _decode(self, deter, stoch_flat):
        h = self.decoder(torch.cat([deter, stoch_flat], dim=-1))
        mean, logvar = self.state_head(h).chunk(2, dim=-1)
        return mean, torch.clamp(logvar, -6.0, 3.0), self.visual_head(h)

    def _filter(self, obs, prev_actions):
        """Posterior filtering over observed steps. Returns final state and per-step terms."""
        batch = obs.shape[0]
        deter = obs.new_zeros(batch, self.deter)
        stoch = obs.new_zeros(batch, self.stoch * self.classes)
        tokens = self.encoder(obs)
        outs = []
        for t in range(obs.shape[1]):
            deter = self._core(deter, stoch, prev_actions[:, t])
            prior = self._probs(self.prior_logits(self.prior_net(deter)))
            post = self._probs(self.post_logits(self.post_net(torch.cat([deter, tokens[:, t]], dim=-1))))
            stoch = self._sample(post)
            outs.append((deter, stoch, prior, post))
        return deter, stoch, outs

    # ---- WorldModel interface ------------------------------------------------------

    def _predict(self, inputs):
        s = self.eval_samples
        rep = lambda x: x.repeat_interleave(s, dim=0)  # noqa: E731
        obs = rep(self._obs(inputs["context_state"], inputs["context_visual"]))
        deter, stoch, _ = self._filter(obs, rep(inputs["context_action"]))
        future = rep(inputs["future_actions"])
        means, logvars, visuals = [], [], []
        for j in range(future.shape[1]):
            deter = self._core(deter, stoch, future[:, j])
            stoch = self._sample(self._probs(self.prior_logits(self.prior_net(deter))))
            mean, logvar, visual = self._decode(deter, stoch)
            means.append(mean)
            logvars.append(logvar)
            visuals.append(visual)
        batch = inputs["context_state"].shape[0]
        shape = lambda x: torch.stack(x, dim=1).view(batch, s, len(x), -1)  # noqa: E731
        mean, logvar = gaussian_merge(shape(means), shape(logvars), sample_dim=1)
        return {"state_mean": mean, "state_logvar": logvar, "visual": shape(visuals).mean(dim=1)}

    def training_loss(self, batch):
        states = torch.cat([batch["context_state"], batch["target_state"]], dim=1)
        visual_in = torch.cat([flatten_visual(batch["context_visual"]), flatten_visual(batch["future_visual"])], dim=1)
        visual_target = torch.cat([batch["context_target_visual"], batch["target_visual"]], dim=1)
        prev_actions = torch.cat([batch["context_action"], batch["future_actions"]], dim=1)
        _, _, outs = self._filter(self._obs(states, visual_in), prev_actions)
        deter = torch.stack([o[0] for o in outs], dim=1)
        stoch = torch.stack([o[1] for o in outs], dim=1)
        prior = torch.stack([o[2] for o in outs], dim=1)
        post = torch.stack([o[3] for o in outs], dim=1)
        mean, logvar, visual = self._decode(deter, stoch)
        rec = prediction_objective(
            {"state_mean": mean, "state_logvar": logvar, "visual": visual}, states, visual_target, self.visual_weight
        ).mean()
        dyn = torch.clamp(self._kl(post.detach(), prior), min=self.free_nats).mean()
        rep = torch.clamp(self._kl(post, prior.detach()), min=self.free_nats).mean()
        loss = self.rec_scale * rec + self.dyn_scale * dyn + self.rep_scale * rep
        return loss, {"rec": rec.detach(), "kl_dyn": dyn.detach(), "kl_rep": rep.detach()}

    def latent_for_diagnostics(self, batch):
        deter, _, _ = self._filter(self._obs(batch["context_state"], batch["context_visual"]), batch["context_action"])
        return deter


def build_rssm(dims: ArmDims, width: int | None = None, **overrides) -> RSSMWorldModel:
    if width is not None:
        overrides["hidden"] = int(width)
    return RSSMWorldModel(dims, **overrides)
