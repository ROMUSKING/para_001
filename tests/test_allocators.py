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
    DirectCritic,
    allocation_traces,
    exact_targets,
    gate_positive_weight,
    matched_critic_hidden,
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
    for name in ("adjoint", "critic", "adjoint_randomized", "uncertainty", "gate_logit"):
        assert torch.equal(a[name], b[name]), name
    assert not torch.equal(a["exact_costate"], b["exact_costate"])  # the privileged diagnostic does depend on them
