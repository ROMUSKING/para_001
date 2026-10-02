"""World-model arms: pilot parity, causality, shapes, gradients and parameter matching.

Tensors are random test fixtures that exercise code paths; nothing here is data or evidence.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from adjointrwm.models import (  # noqa: E402
    AdjointRecursiveWorldModel,
    AdjointRWMConfig,
    ArmDims,
    SimNorm,
    block_causal_mask,
    build_arm,
    count_prediction_parameters,
    match_width,
    sample_single_choice_mask,
    stage1_loss,
)
from adjointrwm.models.rssm import BlockLinear, RSSMWorldModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PILOT_RUN = ROOT / "results/runs/droid100_adjoint_20260929T070629Z"

DIMS = ArmDims(state_dim=3, action_dim=2, visual_tokens=2, visual_token_dim=8, target_visual_dim=16, context_len=4, horizon=3)
SMALL = {
    "adjoint_rwm": {"width": 32, "transformer_heads": 4, "transformer_layers": 2},
    "hybrid_adjoint_rwm": {"width": 32, "transformer_heads": 4, "transformer_layers": 2},
    "dreamerv3_rssm": {"width": 32, "eval_samples": 3},
    "tdmpc2": {"width": 32},
    "dino_wm": {"width": 32, "heads": 2, "dim_head": 8, "depth": 2},
    "vjepa2_ac": {"width": 32, "heads": 2, "dim_head": 8, "depth": 2},
}


def make_batch(dims=DIMS, b=5, seed=0, tokens=True):
    g = torch.Generator().manual_seed(seed)
    T, H = dims.context_len, dims.horizon
    shape_v = (dims.visual_tokens, dims.visual_token_dim) if tokens else (dims.visual_flat_dim,)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    return {
        "context_state": r(b, T, dims.state_dim),
        "context_action": torch.cat([torch.zeros(b, 1, dims.action_dim), r(b, T - 1, dims.action_dim)], dim=1),
        "context_visual": r(b, T, *shape_v),
        "future_actions": r(b, H, dims.action_dim),
        "target_state": r(b, H, dims.state_dim),
        "target_visual": r(b, H, dims.target_visual_dim),
        "future_visual": r(b, H, *shape_v),
        "context_target_visual": r(b, T, dims.target_visual_dim),
    }


def build(name, **extra):
    kwargs = dict(SMALL[name])
    width = kwargs.pop("width")
    torch.manual_seed(0)
    return build_arm(name, DIMS, width=width, **kwargs, **extra)


# --- pilot parity -------------------------------------------------------------------

def test_pilot_model_parameter_count_matches_committed_manifest():
    manifest = json.loads((PILOT_RUN / "config/model_manifest.json").read_text())
    model = AdjointRecursiveWorldModel(manifest["state_dim"], manifest["action_dim"], manifest["visual_dim"], AdjointRWMConfig())
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total == manifest["trainable_parameters"]
    assert model.prediction_parameters() < total  # allocator heads excluded from the prediction path


def _pilot_objective_per_sample(predictions, target_state, target_visual):
    mean = predictions["state_mean"].float()
    logvar = predictions["state_logvar"].float()
    state_nll = 0.5 * (torch.exp(-logvar) * (mean - target_state.float()).pow(2) + logvar)
    state_nll = state_nll.mean(dim=(1, 2))
    predicted_visual = F.normalize(predictions["visual"].float(), dim=-1)
    target_visual = F.normalize(target_visual.float(), dim=-1)
    visual_loss = (1.0 - (predicted_visual * target_visual).sum(dim=-1)).mean(dim=1)
    return state_nll + 0.25 * visual_loss


def _pilot_vc(latent):
    latent = latent.float()
    latent = latent - latent.mean(dim=0, keepdim=True)
    std = torch.sqrt(latent.var(dim=0, unbiased=False) + 1e-4)
    variance_loss = F.relu(1.0 - std).mean()
    covariance = latent.T @ latent / max(1, latent.shape[0] - 1)
    diagonal = torch.diag(torch.diag(covariance))
    covariance_loss = (covariance - diagonal).pow(2).sum() / latent.shape[1]
    return variance_loss + 0.01 * covariance_loss


def _pilot_stage1_loss(model, batch, rate_beta):
    """Copied from the pilot notebook, section 4 (CFG/DEVICE replaced by arguments)."""
    latent = model.encode_context(batch["context_visual"], batch["context_state"], batch["context_action"])
    effects = model.candidate_effects(latent)
    num_candidates = model.num_candidates
    budgets = torch.randint(low=0, high=num_candidates + 1, size=(latent.shape[0],))
    random_scores = torch.rand(latent.shape[0], num_candidates)
    ordering = torch.argsort(random_scores, dim=1, descending=True)
    mask = torch.zeros_like(random_scores)
    for row in range(latent.shape[0]):
        count = int(budgets[row].item())
        if count:
            mask[row, ordering[row, :count]] = 1.0
    full_mask, zero_mask = torch.ones_like(mask), torch.zeros_like(mask)
    fa, ts, tv = batch["future_actions"], batch["target_state"], batch["target_visual"]
    o_alloc = _pilot_objective_per_sample(model.rollout(model.apply_mask(latent, effects, mask), fa), ts, tv)
    o_full = _pilot_objective_per_sample(model.rollout(model.apply_mask(latent, effects, full_mask), fa), ts, tv)
    o_base = _pilot_objective_per_sample(model.rollout(model.apply_mask(latent, effects, zero_mask), fa), ts, tv)
    rate_cost = (mask * model.candidate_costs.view(1, -1)).sum(dim=1)
    ne = F.normalize(effects.float(), dim=-1)
    gram = ne @ ne.transpose(1, 2)
    diversity = (gram - torch.eye(num_candidates).unsqueeze(0)).pow(2).mean()
    return o_alloc.mean() + 0.25 * o_base.mean() + 0.50 * o_full.mean() + rate_beta * rate_cost.mean() + 0.01 * diversity + 0.01 * _pilot_vc(latent)


def test_stage1_subset_loss_matches_pilot_code():
    model = build("adjoint_rwm")
    batch = make_batch(tokens=False)
    torch.manual_seed(123)
    ours, _ = stage1_loss(model, batch, rate_beta=0.002, mask_mode="subset")
    torch.manual_seed(123)
    ref = _pilot_stage1_loss(model, batch, rate_beta=0.002)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_single_choice_mask_has_at_most_one_candidate():
    mask, choice = sample_single_choice_mask(1000, 4, "cpu")
    assert mask.shape == (1000, 4)
    assert (mask.sum(1) <= 1).all()
    assert ((choice == 0) == (mask.sum(1) == 0)).all()
    assert set(choice.tolist()) == {0, 1, 2, 3, 4}


@pytest.mark.parametrize("mode", ["subset", "single_choice", "single_choice_wta"])
def test_stage1_modes_are_finite_and_differentiable(mode):
    model = build("adjoint_rwm")
    loss, parts = stage1_loss(model, make_batch(), mask_mode=mode)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.candidate_embedding.weight.grad is not None


# --- every arm ----------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(SMALL))
def test_arm_shapes_loss_and_gradients(name):
    model = build(name)
    batch = make_batch()
    model.eval()
    out = model.predict(batch)
    assert out["state_mean"].shape == (5, DIMS.horizon, DIMS.state_dim)
    assert out["state_logvar"].shape == (5, DIMS.horizon, DIMS.state_dim)
    assert out["visual"].shape == (5, DIMS.horizon, DIMS.target_visual_dim)
    model.train()
    loss, parts = model.training_loss(batch)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    names = set(model.prediction_parameter_names())
    with_grad = [n for n, p in model.named_parameters() if n in names and p.grad is not None and p.grad.abs().sum() > 0]
    assert len(with_grad) / len(names) > 0.9, sorted(names - set(with_grad))
    latent = model.latent_for_diagnostics(batch)
    assert latent is not None and latent.shape[0] == 5


@pytest.mark.parametrize("name", sorted(SMALL))
def test_predictions_never_read_future_targets(name):
    """Deployed predictions are invariant to replacing every realised future target."""
    model = build(name).eval()
    batch = make_batch()
    tampered = dict(batch)
    for key in ("target_state", "target_visual", "future_visual", "context_target_visual"):
        tampered[key] = torch.randn_like(batch[key]) * 100
    torch.manual_seed(7)
    a = model.predict(batch)
    torch.manual_seed(7)
    b = model.predict(tampered)
    for key in a:
        assert torch.equal(a[key], b[key]), key


@pytest.mark.parametrize("name", sorted(SMALL))
def test_predictions_depend_on_future_actions(name):
    model = build(name).eval()
    batch = make_batch()
    changed = dict(batch, future_actions=batch["future_actions"] + 1.0)
    torch.manual_seed(7)
    a = model.predict(batch)["state_mean"]
    torch.manual_seed(7)
    b = model.predict(changed)["state_mean"]
    assert not torch.allclose(a, b)


def test_block_causal_mask_and_predictor_causality():
    mask = block_causal_mask(3, 2)
    assert mask.tolist() == [
        [True, True, False, False, False, False],
        [True, True, False, False, False, False],
        [True, True, True, True, False, False],
        [True, True, True, True, False, False],
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
    ]
    model = build("dino_wm").eval()
    batch = make_batch()
    tokens = batch["context_visual"]
    states, actions = batch["context_state"], batch["context_action"]
    h1 = model._hidden(tokens, states, actions)
    tokens2 = tokens.clone()
    tokens2[:, -1] += 5.0  # perturb only the last frame
    h2 = model._hidden(tokens2, states, actions)
    assert torch.allclose(h1[:, :-1], h2[:, :-1], atol=1e-5)
    assert not torch.allclose(h1[:, -1], h2[:, -1])


def test_rssm_components():
    layer = BlockLinear(8, 12, blocks=2)
    x = torch.randn(3, 8)
    dense = torch.block_diag(layer.weight[0], layer.weight[1])
    assert torch.allclose(layer(x), x @ dense + layer.bias, atol=1e-6)
    model = build("dreamerv3_rssm")
    assert isinstance(model, RSSMWorldModel)
    assert model.deter == 8 * 32 and model.classes == 2
    probs = model._probs(torch.randn(4, model.stoch * model.classes))
    assert torch.allclose(probs.sum(-1), torch.ones(4, model.stoch))
    assert probs.min() >= model.unimix / model.classes - 1e-7
    assert torch.allclose(model._kl(probs, probs), torch.zeros(4), atol=1e-6)


def test_tdmpc2_simnorm_and_encoder_lr_scale():
    x = torch.randn(4, 16)
    y = SimNorm(8)(x).view(4, 2, 8)
    assert torch.allclose(y.sum(-1), torch.ones(4, 2))
    model = build("tdmpc2")
    groups = model.param_groups(1e-3)
    assert groups[0]["lr"] == pytest.approx(3e-4) and groups[1]["lr"] == pytest.approx(1e-3)
    n_params = sum(p.numel() for g in groups for p in g["params"])
    assert n_params == sum(p.numel() for p in model.parameters())


def test_match_width_hits_target_within_tolerance():
    target = count_prediction_parameters("adjoint_rwm", DIMS, width=64, transformer_heads=4)
    overrides = {"dreamerv3_rssm": {}, "tdmpc2": {}, "dino_wm": {"heads": 2, "dim_head": 8}}
    for name, extra in overrides.items():
        result = match_width(name, DIMS, target, tolerance=0.25, **extra)
        assert result["within_tolerance"], (name, result)
        assert result["parameters"] == count_prediction_parameters(name, DIMS, width=result["width"], **extra)


def test_pool_patch_grid_averages_quadrants():
    from adjointrwm.features import pool_patch_grid

    side, d = 4, 2
    grid = torch.arange(side * side, dtype=torch.float32).view(1, side * side, 1).repeat(1, 1, d)
    pooled = pool_patch_grid(grid, 2)
    assert pooled.shape == (1, 4, d)
    # top-left quadrant holds patches 0, 1, 4, 5
    assert pooled[0, 0, 0].item() == pytest.approx((0 + 1 + 4 + 5) / 4)
    with pytest.raises(ValueError):
        pool_patch_grid(torch.zeros(1, 5, 2), 2)
