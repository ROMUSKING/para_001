"""Pilot v2 allocation contract: exact co-state vs finite differences (roadmap N0.1), hold
semantics, separate heads, traces and method-blind policy summaries.

Tensors and traces are random test fixtures, not data.
"""

import numpy as np
import pytest

from adjointrwm.analysis import gain_matrix

torch = pytest.importorskip("torch")

from adjointrwm.allocators import (  # noqa: E402
    AllocatorJob,
    CostateEstimator,
    CurvatureCostateEstimator,
    DirectCritic,
    allocation_traces,
    belief_space_voi_scores,
    exact_targets,
    first_order_scores,
    gate_positive_weight,
    lcb_decision_scores,
    matched_critic_hidden,
    normalized_first_order_scores,
    pairwise_margin_ranking_loss,
    plackett_luce_loss,
    second_order_curvature_scores,
    validation_regret,
)
from adjointrwm.data import WindowDataset, WindowSpec, fit_normaliser  # noqa: E402
from adjointrwm.models import ArmDims, build_arm, prediction_objective  # noqa: E402
from adjointrwm.training import TrainConfig, seed_everything, train_job  # noqa: E402

DIMS = ArmDims(state_dim=3, action_dim=2, visual_tokens=2, visual_token_dim=4, target_visual_dim=8, context_len=4, horizon=3)


def teacher(dtype=torch.float32):
    seed_everything(0)
    model = build_arm("adjoint_rwm", DIMS, width=16, transformer_heads=2, transformer_layers=1)
    return model.to(dtype).eval()


def batch(b=6, dtype=torch.float32, seed=1):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=dtype)  # noqa: E731
    return {
        "context_state": r(b, 4, 3), "context_action": r(b, 4, 2), "context_visual": r(b, 4, 8),
        "future_actions": r(b, 3, 2), "target_state": r(b, 3, 3), "target_visual": r(b, 3, 8),
    }


def test_exact_costate_matches_finite_differences_in_float64():
    model, data = teacher(torch.float64), batch(dtype=torch.float64)
    out = exact_targets(model, data, cost_weight=0.002)
    latent, costate = out["latent"], out["exact_costate"]
    assert costate.dtype == torch.float64

    def objective(z):
        return prediction_objective(model.rollout(z, data["future_actions"]), data["target_state"], data["target_visual"])

    g = torch.Generator().manual_seed(3)
    for _ in range(3):
        v = torch.randn(latent.shape, generator=g, dtype=torch.float64)
        eps = 1e-6
        fd = (objective(latent + eps * v) - objective(latent - eps * v)) / (2 * eps)
        assert torch.allclose((costate * v).sum(-1), fd, rtol=1e-5, atol=1e-8)


def test_exact_gains_include_hold_and_match_brute_force():
    model, data = teacher(), batch()
    out = exact_targets(model, data, cost_weight=0.002)
    gain = out["exact_gain"]
    assert gain.shape == (6, model.num_candidates + 1)
    assert torch.equal(gain[:, 0], torch.zeros(6))
    with torch.no_grad():
        z = model.encode_context(data["context_visual"], data["context_state"], data["context_action"])
        effects = model.candidate_effects(z)
        stop = prediction_objective(model.rollout(z, data["future_actions"]), data["target_state"], data["target_visual"])
        k = 2
        j = prediction_objective(model.rollout(z + effects[:, k - 1], data["future_actions"]), data["target_state"], data["target_visual"])
    expected = stop - j - model.candidate_costs[k - 1] * 0.002
    assert torch.allclose(gain[:, k], expected, atol=1e-6)
    assert torch.equal(out["effects"][:, 0], torch.zeros_like(out["effects"][:, 0]))


def test_exact_targets_are_fp32_inside_bf16_autocast():
    model, data = teacher(), batch()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = exact_targets(model, data, cost_weight=0.002)
    assert out["exact_gain"].dtype == torch.float32 and out["exact_costate"].dtype == torch.float32


def test_critic_is_parameter_matched_to_costate_estimator():
    d = 512
    costate = sum(p.numel() for p in CostateEstimator(d).parameters())
    critic = sum(p.numel() for p in DirectCritic(d, matched_critic_hidden(d)).parameters())
    assert abs(critic - costate) / costate < 0.01


@pytest.mark.parametrize("head", ["costate", "critic", "gate"])
def test_jobs_train_only_their_head_and_keep_teacher_frozen(head):
    job = AllocatorJob(teacher(), head, cost_weight=0.002, gate_pos_weight=2.0)
    job.train()
    assert not job.teacher.training
    trainable = {n.split(".")[0] for n, p in job.named_parameters() if p.requires_grad}
    assert trainable == {head}
    loss, parts = job.training_loss(batch())
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in job.teacher.parameters())


def test_scores_hold_is_zero_and_randomized_costate_is_a_permutation():
    job = AllocatorJob(teacher(), "costate", cost_weight=0.002)
    targets = exact_targets(job.teacher, batch(), 0.002)
    s = job.scores(targets)
    for name in ("adjoint", "critic", "adjoint_randomized", "uncertainty", "exact_costate"):
        assert s[name].shape == (6, 5)
        assert torch.equal(s[name][:, 0], torch.zeros(6)), name
    budget, horizon = job._conditions(targets["latent"])
    costate = job.costate(targets["latent"], budget, horizon)
    perm = torch.roll(torch.arange(6), 1)
    expected = -(costate[perm].unsqueeze(1) * targets["effects"]).sum(-1) - targets["costs"].view(1, -1)
    assert torch.allclose(s["adjoint_randomized"], expected)


def _dataset(n, seed):
    rng = np.random.default_rng(seed)
    records = [{"episode_id": f"{seed}-{i}", "length": 10, "arrays": {
        "states": rng.normal(size=(10, 3)).astype(np.float32), "actions": rng.normal(size=(10, 2)).astype(np.float32),
        "exterior_embeddings": rng.normal(size=(10, 4)).astype(np.float16), "wrist_embeddings": rng.normal(size=(10, 4)).astype(np.float16),
    }} for i in range(n)]
    norm = fit_normaliser([r["arrays"]["states"] for r in records], [r["arrays"]["actions"] for r in records])
    return WindowDataset(records, WindowSpec(4, 3, 1), norm)


def test_traces_and_training_job_end_to_end(tmp_path):
    train, val = _dataset(3, 0), _dataset(2, 1)
    job = AllocatorJob(teacher(), "critic", cost_weight=0.002)
    traces = allocation_traces(job, val, batch_size=4, device="cpu")
    assert len(traces) == len(val)
    assert [c for c in traces.columns if c.startswith("gain_")] == [f"gain_{k}" for k in range(5)]
    np.testing.assert_array_equal(traces["oracle_choice"], gain_matrix(traces).argmax(1))
    cfg = TrainConfig(steps=4, batch_size=4, lr=1e-3, warmup_steps=1, eval_interval=2, checkpoint_interval=2, amp=False, eval_batch_size=8)
    identity = {"run_id": "r", "arm": "allocator_critic", "seed": 0, "config_hash": "c", "data_manifest_hash": "d", "source_hash": "s"}
    summary = train_job(job, train, val, cfg, identity=identity, local_dir=tmp_path / "l", persist_dir=tmp_path / "p",
                        score_fn=lambda m: validation_regret(m, val, 8, "cpu", "critic"), score_name="validation_regret", stage="allocator_critic")
    assert summary["status"] == "DONE" and summary["best"]["score_name"] == "validation_regret"
    weights = gate_positive_weight(traces)
    assert weights["positives"] + weights["negatives"] + weights["ties"] == len(traces)


def test_deployable_scores_never_read_future_targets():
    """Prohibited shortcut: future targets must not enter a deployable decision."""
    job = AllocatorJob(teacher(), "costate", cost_weight=0.002).eval()
    data = batch()
    tampered = dict(data, target_state=data["target_state"] * 5 + 3, target_visual=-data["target_visual"])
    with torch.no_grad():
        a = job.scores(exact_targets(job.teacher, data, 0.002))
        b = job.scores(exact_targets(job.teacher, tampered, 0.002))
    for name in ("adjoint", "adjoint_norm", "adjoint_lcb", "critic", "critic_lcb", "adjoint_randomized", "adjoint_randomized_norm", "uncertainty", "gate_logit"):
        assert torch.equal(a[name], b[name]), name
    assert not torch.equal(a["exact_costate"], b["exact_costate"])  # the privileged diagnostic does depend on them
    assert not torch.equal(a["exact_costate_norm"], b["exact_costate_norm"])


def test_normalized_first_order_scores_properties():
    """PARA enhancement: scale-invariance and hold option preservation."""
    b, k, d = 4, 3, 8
    costate = torch.randn(b, d)
    effects = torch.randn(b, k + 1, d)
    effects[:, 0] = 0.0  # hold option has zero effect
    costs = torch.tensor([0.0, 0.01, 0.02, 0.03])

    scores = normalized_first_order_scores(costate, effects, costs)
    assert scores.shape == (b, k + 1)
    # Hold option must be identically zero
    assert torch.equal(scores[:, 0], torch.zeros(b))

    # Scale-invariance: multiplying costate by positive alpha yields identical scores
    alpha = 42.5
    scores_scaled = normalized_first_order_scores(costate * alpha, effects, costs)
    assert torch.allclose(scores, scores_scaled, atol=1e-5)


def test_lcb_decision_scores_properties():
    """PARA enhancement: LCB trigger penalizes uncertain options and preserves hold."""
    b, k = 4, 3
    scores = torch.tensor([
        [0.0, 0.5, 0.2],
        [0.0, 0.1, 0.8],
        [0.0, 0.3, 0.4],
        [0.0, -0.1, -0.2],
    ])
    uncert = torch.tensor([
        [0.0, 1.0, 0.1],  # k=1 has high uncert
        [0.0, 0.1, 1.0],  # k=2 has high uncert
        [0.0, 0.5, 0.5],
        [0.0, 0.2, 0.2],
    ])
    # kappa = 0 returns identical scores
    assert torch.equal(lcb_decision_scores(scores, uncert, kappa=0.0), scores)

    # kappa = 1.0 penalizes active candidates
    lcb = lcb_decision_scores(scores, uncert, kappa=1.0)
    assert lcb.shape == (b, k)
    assert torch.equal(lcb[:, 0], torch.zeros(b))
    # Active candidate scores must be strictly lower
    assert (lcb[:, 1:] < scores[:, 1:]).all()


def test_pairwise_margin_ranking_loss_properties():
    """Milestone B2.3: pairwise hinge loss is zero when ordered, positive when flipped."""
    # Hand-checkable case: B=1, K=2, gains=[0, 1], scores=[0, 0].
    # Only positive pair is (i=1, j=0): margin=1, score_diff=0 -> loss=1.
    scores = torch.tensor([[0.0, 0.0]])
    gains = torch.tensor([[0.0, 1.0]])
    assert torch.allclose(pairwise_margin_ranking_loss(scores, gains), torch.tensor(1.0))

    # Perfectly ordered scores (scores == gains) satisfy every margin with equality -> zero.
    perfect = torch.tensor([[0.0, 0.5, 1.2], [2.0, 0.0, -1.0]])
    assert torch.allclose(
        pairwise_margin_ranking_loss(perfect, perfect), torch.tensor(0.0), atol=1e-6
    )

    # Fully reversed ordering incurs strictly positive loss, worse than perfect.
    gains_r = torch.tensor([[0.0, 1.0, 2.0]])
    reversed_scores = torch.tensor([[2.0, 1.0, 0.0]])
    loss_reversed = pairwise_margin_ranking_loss(reversed_scores, gains_r)
    loss_perfect = pairwise_margin_ranking_loss(gains_r, gains_r)
    assert torch.isfinite(loss_reversed) and loss_reversed > 0
    assert loss_perfect < loss_reversed

    # Larger margin_scale cannot reduce the loss on a misordered pair.
    small = pairwise_margin_ranking_loss(reversed_scores, gains_r, margin_scale=1.0)
    large = pairwise_margin_ranking_loss(reversed_scores, gains_r, margin_scale=2.0)
    assert large >= small

    # All-equal gains -> no positive pairs -> differentiable zero.
    tied = torch.tensor([[1.0, 1.0, 1.0]])
    zero = pairwise_margin_ranking_loss(torch.tensor([[0.5, -0.2, 0.1]]), tied)
    assert torch.allclose(zero, torch.tensor(0.0))
    grad_scores = torch.tensor([[0.5, -0.2, 0.1]], requires_grad=True)
    pairwise_margin_ranking_loss(grad_scores, tied).backward()
    assert grad_scores.grad is not None

    # Batching: mean over all positive pairs across the batch.
    b_scores = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    b_gains = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    # Row 0 contributes loss 1.0, row 1 contributes relu(1 - 0) = 1.0 -> mean 1.0.
    assert torch.allclose(pairwise_margin_ranking_loss(b_scores, b_gains), torch.tensor(1.0))

    # Gradient flows through the score argument.
    req = torch.tensor([[2.0, 1.0, 0.0]], requires_grad=True)
    pairwise_margin_ranking_loss(req, gains_r).backward()
    assert req.grad is not None and torch.isfinite(req.grad).all()


def test_plackett_luce_loss_properties():
    """Milestone B2.3: listwise KL is zero when distributions coincide, positive otherwise."""
    import torch.nn.functional as F

    # Identical scores and gains -> identical distributions -> KL == 0.
    x = torch.tensor([[0.0, 0.5, -0.3], [1.0, 2.0, 0.5]])
    assert torch.allclose(plackett_luce_loss(x, x), torch.tensor(0.0), atol=1e-6)

    # Misordered scores incur strictly larger loss than matched scores.
    gains = torch.tensor([[0.0, 1.0, 2.0]])
    matched = plackett_luce_loss(gains, gains)
    flipped = plackett_luce_loss(torch.tensor([[2.0, 1.0, 0.0]]), gains)
    assert torch.isfinite(flipped) and flipped > 0
    assert matched < flipped

    # Manual KL cross-check against log_softmax/softmax.
    scores = torch.tensor([[0.2, -0.1, 0.7]])
    tgt = torch.tensor([[1.0, 0.0, -1.0]])
    tau = 0.5
    p = F.softmax(tgt / tau, dim=-1)
    log_p = F.log_softmax(tgt / tau, dim=-1)
    log_q = F.log_softmax(scores / tau, dim=-1)
    expected = (p * (log_p - log_q)).sum(dim=-1).mean()
    assert torch.allclose(plackett_luce_loss(scores, tgt, temperature=tau), expected)

    # Non-negativity and finiteness on random inputs.
    g = torch.Generator().manual_seed(0)
    rs, rg = torch.randn(8, 5, generator=g), torch.randn(8, 5, generator=g)
    loss = plackett_luce_loss(rs, rg)
    assert torch.isfinite(loss) and loss >= 0

    # Gradient flows and temperature validation.
    req = torch.tensor([[0.2, -0.1, 0.7]], requires_grad=True)
    plackett_luce_loss(req, tgt).backward()
    assert req.grad is not None and torch.isfinite(req.grad).all()
    with pytest.raises(ValueError):
        plackett_luce_loss(scores, tgt, temperature=0.0)


def test_allocator_job_ranking_loss_variants():
    """Milestone B2.3: AllocatorJob supports ce/margin/listwise/hybrid ranking losses."""
    import torch.nn.functional as F

    data = batch()
    # Default preserves backward compatibility (CE on unnormalised adjoint scores).
    job_ce = AllocatorJob(teacher(), "costate", cost_weight=0.002)
    assert job_ce.ranking_loss_type == "ce"
    targets = exact_targets(job_ce.teacher, data, 0.002)
    s = job_ce.scores(targets)
    oracle = targets["exact_gain"].argmax(dim=1)
    expected_ce = F.cross_entropy(s["adjoint"], oracle)
    loss_ce, parts_ce = job_ce.training_loss(data)
    assert torch.isfinite(loss_ce) and torch.isfinite(parts_ce["ranking"])
    assert torch.allclose(parts_ce["ranking"], expected_ce.detach(), atol=1e-6)

    # Each variant runs end-to-end, is finite, differentiable, and uses adjoint_norm.
    for variant in ("margin", "listwise", "hybrid"):
        job = AllocatorJob(teacher(), "costate", cost_weight=0.002, ranking_loss_type=variant)
        loss, parts = job.training_loss(data)
        assert torch.isfinite(loss), variant
        assert torch.isfinite(parts["ranking"]), variant
        if variant == "margin":
            expected = pairwise_margin_ranking_loss(s["adjoint_norm"], targets["exact_gain"])
            assert torch.allclose(parts["ranking"], expected.detach(), atol=1e-5), variant
        elif variant == "listwise":
            expected = plackett_luce_loss(s["adjoint_norm"], targets["exact_gain"])
            assert torch.allclose(parts["ranking"], expected.detach(), atol=1e-5), variant
        else:
            m = pairwise_margin_ranking_loss(s["adjoint_norm"], targets["exact_gain"])
            l = plackett_luce_loss(s["adjoint_norm"], targets["exact_gain"])
            assert torch.allclose(parts["ranking"], (0.5 * m + 0.5 * l).detach(), atol=1e-5)
        job.zero_grad()
        loss.backward()
        grads = [p.grad for p in job.costate.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)

    # Margin/listwise generally differ from CE (continuous gains carry extra signal).
    job_margin = AllocatorJob(teacher(), "costate", cost_weight=0.002, ranking_loss_type="margin")
    _, parts_margin = job_margin.training_loss(data)
    assert not torch.equal(parts_margin["ranking"], parts_ce["ranking"])

    # Invalid option rejected; non-costate heads ignore the ranking switch but still run.
    with pytest.raises(ValueError):
        AllocatorJob(teacher(), "costate", cost_weight=0.002, ranking_loss_type="bogus")
    for head in ("critic", "gate"):
        job = AllocatorJob(teacher(), head, cost_weight=0.002, ranking_loss_type="hybrid")
        loss, _ = job.training_loss(data)
        assert torch.isfinite(loss)


def test_second_order_curvature_scores():
    """Direction 3: hold is 0, zero Hessian recovers first-order, curvature penalizes."""
    g = torch.Generator().manual_seed(0)
    b, k1, d = 4, 3, 5
    costate = torch.randn(b, d, generator=g)
    effects = torch.randn(b, k1 + 1, d, generator=g)
    diag_hessian = torch.rand(b, d, generator=g)
    costs = torch.tensor([0.0, 0.01, 0.02, 0.03])

    scores = second_order_curvature_scores(costate, diag_hessian, effects, costs)
    assert scores.shape == (b, k1 + 1)
    assert torch.equal(scores[:, 0], torch.zeros(b))

    # Zero Hessian matches first-order scores (except hold, both zero).
    zero_h = torch.zeros_like(diag_hessian)
    s_second = second_order_curvature_scores(costate, zero_h, effects, costs)
    s_first = first_order_scores(costate, effects, costs)
    # first_order already has hold == 0 since effects_0/costs_0 are 0 here only
    # if effects_0 arbitrary; force comparison on active candidates.
    assert torch.allclose(s_second[:, 1:], s_first[:, 1:], atol=1e-6)
    assert torch.equal(s_second[:, 0], torch.zeros(b))

    # Positive curvature penalizes large effects: scaling effects up lowers scores.
    big = effects * 4.0
    s_big = second_order_curvature_scores(costate, diag_hessian, big, costs)
    s_small = second_order_curvature_scores(costate, diag_hessian, effects, costs)
    # Curvature term is -0.5*h*||delta||^2, quadratic, so for large effects the
    # curvature penalty dominates: check explicit penalty form on one row.
    manual = -(costate[:1].unsqueeze(1) * big[:1]).sum(-1) - 0.5 * (
        diag_hessian[:1].unsqueeze(1) * big[:1].pow(2)
    ).sum(-1) - costs.view(1, -1)
    assert torch.allclose(s_big[:1, 1:], manual[:, 1:], atol=1e-6)
    # With positive Hessian, doubling effects more than doubles the penalty.
    s1 = second_order_curvature_scores(costate, torch.ones_like(diag_hessian), effects, torch.zeros(k1 + 1))
    s2 = second_order_curvature_scores(costate, torch.ones_like(diag_hessian), 2 * effects, torch.zeros(k1 + 1))
    penalty1 = (first_order_scores(costate, effects, torch.zeros(k1 + 1)) - s1)[:, 1:]
    penalty2 = (first_order_scores(costate, 2 * effects, torch.zeros(k1 + 1)) - s2)[:, 1:]
    assert torch.allclose(penalty2, 4 * penalty1, atol=1e-5)
    assert (penalty1 >= 0).all()


def test_belief_space_voi_scores():
    """Direction 3: hold is 0, positive delta_cov increases VOI score."""
    g = torch.Generator().manual_seed(1)
    b, k1, d = 4, 3, 5
    costate = torch.randn(b, d, generator=g)
    effects = torch.randn(b, k1 + 1, d, generator=g)
    diag_hessian = torch.rand(b, d, generator=g) + 0.5
    costs = torch.tensor([0.0, 0.01, 0.02, 0.03])
    delta_cov = torch.rand(b, k1 + 1, d, generator=g)

    scores = belief_space_voi_scores(costate, effects, delta_cov, diag_hessian, costs)
    assert scores.shape == (b, k1 + 1)
    assert torch.equal(scores[:, 0], torch.zeros(b))

    # Positive delta_cov increases the score vs zero uncertainty reduction.
    s_zero = belief_space_voi_scores(costate, effects, torch.zeros_like(delta_cov), diag_hessian, costs)
    assert (scores[:, 1:] > s_zero[:, 1:]).all()

    # Manual form check.
    manual = -(costate.unsqueeze(1) * effects).sum(-1) + 0.5 * 0.5 * (
        diag_hessian.unsqueeze(1) * delta_cov
    ).sum(-1) - costs.view(1, -1)
    manual = torch.cat([manual.new_zeros(b, 1), manual[:, 1:]], dim=1)
    assert torch.allclose(scores, manual, atol=1e-6)


def test_curvature_costate_estimator():
    """Direction 3: shapes, non-negative Hessian, grad flow."""
    d = 8
    est = CurvatureCostateEstimator(d)
    b = 5
    latent = torch.randn(b, d)
    budget = torch.full((b,), 0.25)
    horizon = torch.ones(b)
    costate, diag_hessian = est(latent, budget, horizon)
    assert costate.shape == (b, d)
    assert diag_hessian.shape == (b, d)
    assert (diag_hessian >= 0).all()

    # Grad flows to all parameters through both heads.
    loss = costate.sum() + diag_hessian.sum()
    loss.backward()
    grads = [p.grad for p in est.parameters() if p.grad is not None]
    assert len(grads) == len(list(est.parameters()))
    assert all(torch.isfinite(g).all() for g in grads)

