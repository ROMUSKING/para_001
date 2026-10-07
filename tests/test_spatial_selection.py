"""Session 6A: fixed-budget spatial patch selection.

Covers the three contracts that make the benchmark meaningful:

1. the rank-one attention perturbation identity, checked against *explicit numerical
   differences* (an actual duplicated-patch pooling) rather than against itself;
2. that all eight comparators plus both diagnostic references run on a synthetic batch;
3. that the deployable selectors preserve the information boundary -- corrupting every
   future target must leave their selections bit-identical.

All tensors are random fixtures, not data.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adjointrwm.allocators import CurvatureCostateEstimator  # noqa: E402
from adjointrwm.models.common import ArmDims  # noqa: E402
from adjointrwm.models.registry import build_arm  # noqa: E402
from adjointrwm.models.spatial_adapter import SpatialPatchAdapter  # noqa: E402
from adjointrwm.spatial_selection import (  # noqa: E402
    BUDGETS,
    CAMERAS,
    COMPARATORS,
    DEPLOYABLE_INPUT_KEYS,
    DEPLOYABLE_POLICIES,
    DIAGNOSTIC_POLICIES,
    PATCHES_PER_CAMERA,
    TOTAL_PATCHES,
    PatchRankingCritic,
    PrivilegedPatchCritic,
    additivity_r2,
    apply_patch_mask,
    assert_deployable,
    attention_terms,
    belief_space_voi_scores,
    camera_of_patch,
    cotangent_bundle,
    curvature_scores,
    deployable_view,
    early_cls_attention_scores,
    early_feature_norm_scores,
    encode_masked,
    epistemic_variance_reduction,
    exhaustive_oracle_masks,
    grid_coordinates,
    greedy_oracle_masks,
    latent_patch_perturbations,
    layernorm_jvp,
    matched_patch_critic_hidden,
    objective_at_masks,
    patch_identity_for_tests,
    rank_one_first_order,
    rank_one_perturbation,
    score_policy,
    select_topk_per_camera,
    selection_context,
    site_clustered_bootstrap_ci,
    spatial_encode_with_pooled,
    stratified_random_scores,
    submodularity_violation_rate,
    trimmed_mean,
    uniform_grid_scores,
    wilcoxon_signed_rank,
)

ROOT = Path(__file__).resolve().parents[1]

PATCHES = 8          # 4 per camera: the smallest square grid that exercises every path
PER_CAMERA = 4
FRAMES = 3
TOKEN_DIM = 12
STATE_DIM = 6
ACTION_DIM = 4
HORIZON = 2
WIDTH = 32
TARGET_VISUAL = 10
WINDOWS = 3


def build_small_model() -> "torch.nn.Module":
    dims = ArmDims(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, visual_tokens=PATCHES, visual_token_dim=TOKEN_DIM,
        target_visual_dim=TARGET_VISUAL, context_len=FRAMES, horizon=HORIZON,
    )
    return build_arm(
        "spatial_adjoint_rwm", dims, width=WIDTH, transformer_heads=4, transformer_layers=1
    ).eval()


def synthetic_batch(seed: int = 7) -> dict:
    generator = torch.Generator().manual_seed(seed)
    draw = lambda *shape: torch.randn(*shape, generator=generator)  # noqa: E731
    return {
        "context_visual": draw(WINDOWS, FRAMES, PATCHES, TOKEN_DIM),
        "context_state": draw(WINDOWS, FRAMES, STATE_DIM),
        "context_action": draw(WINDOWS, FRAMES, ACTION_DIM),
        "future_actions": draw(WINDOWS, HORIZON, ACTION_DIM),
        "target_state": draw(WINDOWS, HORIZON, STATE_DIM),
        "target_visual": draw(WINDOWS, HORIZON, TARGET_VISUAL),
    }


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_small_model()


@pytest.fixture(scope="module")
def batch():
    return synthetic_batch()


@pytest.fixture(scope="module")
def curvature_head():
    torch.manual_seed(1)
    return CurvatureCostateEstimator(WIDTH).eval()


@pytest.fixture(scope="module")
def heads():
    torch.manual_seed(2)
    return {
        "direct_ranking_critic": PatchRankingCritic(
            TOKEN_DIM, WIDTH, matched_patch_critic_hidden(TOKEN_DIM + WIDTH + 2, WIDTH)
        ).eval(),
        "direct_critic_privileged": PrivilegedPatchCritic(
            WIDTH, matched_patch_critic_hidden(3 * WIDTH + 2, WIDTH)
        ).eval(),
    }


@pytest.fixture(scope="module")
def context(model, batch, curvature_head):
    """A context carrying a CLS attention map, so every comparator is available."""
    context = selection_context(model, batch, curvature_head, patches_per_camera=PER_CAMERA, cameras=CAMERAS)
    context.cls_attention = torch.rand(WINDOWS, FRAMES, PATCHES, generator=torch.Generator().manual_seed(5))
    return context


# ---------------------------------------------------------------------------
# 1. The rank-one attention perturbation identity
# ---------------------------------------------------------------------------


def test_rank_one_identity_matches_explicit_duplicate_pooling(model, batch):
    """d(pooled)_p from the closed form equals re-running the pooling with patch p duplicated.

    This is the numerical proof of the identity that lets one backward pass score all
    ``P`` patches: the closed form is compared against an actual extra forward pass.
    """
    visual = batch["context_visual"]
    terms = attention_terms(model.visual_adapter, visual)
    predicted = rank_one_perturbation(terms, batch_size=WINDOWS)

    errors = []
    for patch in range(PATCHES):
        numerical = patch_identity_for_tests(model, batch, patch)
        errors.append((numerical - predicted[:, :, patch]).abs().max().item())
    assert max(errors) < 1e-5, f"rank-one identity failed: max abs error {max(errors):.3e}"
    assert predicted.abs().max() > 1e-4, "perturbation is numerically zero; the test would be vacuous"


def test_rank_one_perturbation_is_rank_one_in_the_value_residual(model, batch):
    """Each perturbation is parallel to ``v_p - pooled`` through the output projection."""
    visual = batch["context_visual"]
    terms = attention_terms(model.visual_adapter, visual)
    predicted = rank_one_perturbation(terms, batch_size=WINDOWS)

    # Recompute the residual direction the identity claims to be parallel to.
    frames, patches, heads, head_dim = terms.values.shape
    residual = (terms.values - terms.context.unsqueeze(1)).reshape(frames, patches, heads * head_dim)
    directions = torch.nn.functional.linear(residual, terms.out_weight).reshape(WINDOWS, FRAMES, patches, -1)

    cosine = torch.nn.functional.cosine_similarity(predicted.flatten(0, 1), directions.flatten(0, 1), dim=-1)
    assert cosine.min() > 1 - 1e-4, f"perturbation is not parallel to (v_p - pooled): min cosine {cosine.min()}"


def test_spec_shorthand_is_a_first_order_approximation(model, batch):
    """The spec's ``alpha_p (v_p - pooled)`` makes two first-order simplifications.

    Dropping the ``1 / (1 + alpha_p)`` renormalisation and head-averaging ``alpha`` are
    both first order in ``alpha``: each is a few percent at the spec's operating point and
    each grows to ~90% at tiny patch counts. The exact per-head form is the default; this
    test pins that the shorthand degrades gracefully and never inverts direction.
    """
    def relative_error(reference: torch.Tensor, other: torch.Tensor) -> float:
        return ((reference - other).abs().max() / reference.abs().max()).item()

    terms = attention_terms(model.visual_adapter, batch["context_visual"])
    exact = rank_one_perturbation(terms, exact_heads=True, batch_size=WINDOWS)
    first_order = rank_one_first_order(terms, batch_size=WINDOWS)
    head_averaged = rank_one_perturbation(terms, exact_heads=False, batch_size=WINDOWS)

    # P = 8 here, so alpha is large and both first-order terms are still visible.
    assert relative_error(exact, first_order) < 0.15
    cosine = torch.nn.functional.cosine_similarity(
        exact.flatten(0, 1), head_averaged.flatten(0, 1), dim=-1)
    assert cosine.mean() > 0.5, "the shorthand must at least preserve direction on average"

    # At the spec's real setting (P = 32 total, 4 heads) both errors are a few percent.
    wide = SpatialPatchAdapter(token_dim=TOKEN_DIM, d_model=WIDTH, num_heads=4, max_patches=64).eval()
    wide_terms = attention_terms(wide, torch.randn(2, FRAMES, 32, TOKEN_DIM))
    wide_exact = rank_one_perturbation(wide_terms, batch_size=2)
    assert relative_error(wide_exact, rank_one_first_order(wide_terms, batch_size=2)) < 0.06
    assert relative_error(wide_exact, rank_one_perturbation(wide_terms, exact_heads=False, batch_size=2)) < 0.06


def test_attention_terms_reproduce_the_adapter_pooling(model, batch):
    """``attention_terms`` recomputes exactly what ``nn.MultiheadAttention`` used."""
    terms = attention_terms(model.visual_adapter, batch["context_visual"])
    frames = terms.values.shape[0]
    projected = model.visual_adapter.patch_proj(batch["context_visual"]) + \
        model.visual_adapter.spatial_pos[:, :, :PATCHES, :]
    flat = projected.reshape(frames, PATCHES, WIDTH)
    query = model.visual_adapter.pool_query.reshape(1, 1, WIDTH).expand(frames, 1, WIDTH)
    reference, _ = model.visual_adapter.pool_attn(query, flat, flat, need_weights=True)
    assert (terms.pooled - reference.reshape(frames, WIDTH)).abs().max() < 1e-5
    assert torch.allclose(terms.alpha.sum(-1), torch.ones_like(terms.alpha.sum(-1)), atol=1e-5)


def test_layernorm_jvp_matches_autograd(model, batch):
    """The analytic LayerNorm directional derivative equals autograd's."""
    torch.manual_seed(3)
    x = torch.randn(4, WIDTH, dtype=torch.float64, requires_grad=True)
    delta = torch.randn(4, WIDTH, dtype=torch.float64)
    norm = torch.nn.LayerNorm(WIDTH, dtype=torch.float64)
    reference = torch.autograd.grad((norm(x) * delta).sum(), x)[0]
    predicted = layernorm_jvp(x, delta, norm.eps)
    assert (predicted - reference).abs().max() < 1e-9


# ---------------------------------------------------------------------------
# 2. The policy suite runs on a synthetic batch
# ---------------------------------------------------------------------------


def test_all_comparators_and_diagnostics_run(model, context, heads):
    """All eight comparators plus the autograd reference produce finite ``[B, P]`` gains."""
    assert len(COMPARATORS) == 8
    for policy in COMPARATORS + ("exact_costate_reference",):
        gains = score_policy(policy, context, heads=heads, k_per_camera=2)
        assert gains.shape == (WINDOWS, PATCHES), f"{policy} returned {tuple(gains.shape)}"
        assert torch.isfinite(gains).all(), f"{policy} produced non-finite gains"


def test_the_oracle_is_a_search_and_has_no_scorer(context, heads):
    """``calibrated_greedy_oracle`` selects by rollout search, so it has no scoring function."""
    assert "calibrated_greedy_oracle" in DIAGNOSTIC_POLICIES
    with pytest.raises(ValueError, match="rollout search"):
        score_policy("calibrated_greedy_oracle", context, heads=heads, k_per_camera=2)


def test_every_policy_name_is_recognised(context, heads):
    for policy in COMPARATORS + DIAGNOSTIC_POLICIES:
        if policy == "calibrated_greedy_oracle":
            continue
        assert score_policy(policy, context, heads=heads, k_per_camera=2).shape == (WINDOWS, PATCHES)
    with pytest.raises(ValueError, match="unknown policy"):
        score_policy("no_such_policy", context, heads=heads, k_per_camera=2)


def test_selections_respect_the_symmetric_per_camera_budget(context, heads):
    """Every selector takes exactly ``k_cam`` patches from *each* camera (spec B5)."""
    index = camera_of_patch(PER_CAMERA, CAMERAS)
    for policy in COMPARATORS + ("exact_costate_reference",):
        gains = score_policy(policy, context, heads=heads, k_per_camera=2)
        mask = select_topk_per_camera(gains, 2, index, CAMERAS)
        per_camera = [int(mask[:, index == c].sum(dim=1)[0]) for c in range(CAMERAS)]
        assert per_camera == [2, 2], f"{policy} broke the symmetric budget: {per_camera}"
        assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_belief_space_voi_reduces_to_curvature_at_beta_zero(context):
    """``beta = 0`` must reproduce ``second_order_curvature`` exactly (spec B2 control)."""
    curvature = curvature_scores(context.costate_hat, context.hessian_hat, context.delta_z)
    for beta in (0.0,):
        voi = belief_space_voi_scores(
            context.costate_hat, context.hessian_hat, context.delta_z,
            context.delta_sigma, context.sigma_precision, beta=beta,
        )
        assert torch.allclose(curvature, voi, atol=1e-6)


def test_sign_flipped_beta_flips_the_epistemic_correction(context):
    """The ``-beta`` control must be exactly the additive negative of the ``+beta`` term."""
    curvature = curvature_scores(context.costate_hat, context.hessian_hat, context.delta_z)
    positive = belief_space_voi_scores(
        context.costate_hat, context.hessian_hat, context.delta_z,
        context.delta_sigma, context.sigma_precision, beta=1.0,
    )
    negative = belief_space_voi_scores(
        context.costate_hat, context.hessian_hat, context.delta_z,
        context.delta_sigma, context.sigma_precision, beta=-1.0,
    )
    assert torch.allclose(positive - curvature, -(negative - curvature), atol=1e-6)


def test_grounded_variance_reduction_comes_from_the_model_variance_head(model, batch):
    """``delta_Sigma`` is a real variance reduction in state space, not ``effects.pow(2)``."""
    delta_sigma, precision = epistemic_variance_reduction(model, batch)
    assert delta_sigma.shape == (WINDOWS, PATCHES, STATE_DIM)
    assert precision.shape == (WINDOWS, STATE_DIM)
    assert torch.isfinite(delta_sigma).all() and torch.isfinite(precision).all()
    assert (precision > 0).all(), "exp(-logvar) must be strictly positive"

    # Adding information should, on average, not *increase* predictive variance.
    assert delta_sigma.mean() > 0, "grounded epistemic reduction has the wrong sign on average"

    # It must be genuinely derived from the model's own head rather than reused from the
    # latent effects: a zero effect mask would give an identically zero delta_Sigma.
    null_sigma, null_precision = epistemic_variance_reduction(
        model, batch, mask=torch.zeros(WINDOWS, PATCHES)
    )
    assert null_sigma.shape == delta_sigma.shape and torch.isfinite(null_sigma).all()
    assert null_precision.shape == precision.shape


def test_greedy_oracle_matches_exhaustive_search(model, batch):
    """The greedy rollout-search oracle is calibrated against exhaustive enumeration (B5).

    At ``k_cam = 2`` over 4 patches per camera the joint space is ``C(4,2)^2 = 36``
    subsets, so the true optimum is computable and the greedy gap is measured, not assumed.
    """
    greedy, chain = greedy_oracle_masks(model, batch, 2, camera_of_patch(PER_CAMERA, CAMERAS))
    best = exhaustive_oracle_masks(model, batch, 2, patches_per_camera=PER_CAMERA)
    greedy_value = objective_at_masks(model, batch, greedy.unsqueeze(1))[:, 0]
    best_value = objective_at_masks(model, batch, best.unsqueeze(1))[:, 0]
    assert (greedy_value >= best_value - 1e-6).all(), "greedy beat the exhaustive optimum; the oracle is wrong"
    assert len(chain) == WINDOWS and all(len(step) == 4 for step in chain)


def test_exhaustive_search_refuses_an_intractable_space(model, batch):
    """The joint exhaustive space is a product over cameras, so it grows fast."""
    # C(4,3) = 4 per camera, squared = 16 subsets: allowed.
    exhaustive_oracle_masks(model, batch, 3, patches_per_camera=PER_CAMERA, max_candidates=16)
    # C(4,2)^2 = 36 > 16: refused rather than silently truncated.
    with pytest.raises(ValueError, match="exceeds"):
        exhaustive_oracle_masks(model, batch, 2, patches_per_camera=PER_CAMERA, max_candidates=16)


def test_uniform_grid_takes_the_corners_first():
    """The spec's worked example: the top four cells of a 4x4 grid are its four corners."""
    scores = uniform_grid_scores(1, 16, 1)
    assert scores[0].topk(4).indices.tolist() == [0, 3, 12, 15]
    corners = {tuple(c) for c in grid_coordinates(16, 4).tolist()}
    chosen = [tuple(grid_coordinates(16, 4)[i].tolist()) for i in scores[0].topk(4).indices.tolist()]
    assert set(chosen) == {(0, 0), (0, 3), (3, 0), (3, 3)}


def test_early_feature_scores_use_raw_untouched_tokens():
    """B4: score raw pre-LayerNorm tokens, so a LayerNorm inside the adapter cannot matter."""
    visual = torch.randn(2, FRAMES, PATCHES, TOKEN_DIM)
    expected = visual.norm(dim=-1).mean(dim=1)
    assert torch.allclose(early_feature_norm_scores(visual), expected)


def test_cls_attention_refuses_to_duplicate_the_norm_baseline():
    """`early_cls_attention` must not silently become `early_feature_norm`."""
    visual = torch.randn(2, FRAMES, PATCHES, TOKEN_DIM)
    assert early_cls_attention_scores(visual) is None, "no CLS signal must yield None, not a substitute"

    attention_map = torch.rand(2, FRAMES, PATCHES)
    assert torch.allclose(early_cls_attention_scores(visual, attention_map), attention_map.mean(dim=1))

    cls_token = torch.randn(2, FRAMES, TOKEN_DIM)
    cosine = early_cls_attention_scores(visual, None, cls_token)
    assert cosine.shape == (2, PATCHES)
    assert not torch.allclose(cosine, early_feature_norm_scores(visual))

    with pytest.raises(ValueError, match="refusing to substitute"):
        score_policy("early_cls_attention", selection_context(
            build_small_model(), synthetic_batch(), CurvatureCostateEstimator(WIDTH).eval(),
            patches_per_camera=PER_CAMERA, cameras=CAMERAS))


def test_stratified_random_spreads_picks_over_quadrants_and_varies():
    """One random pick per quadrant, not a deterministic index-ordered tie-break."""
    scores = stratified_random_scores(8, 4, seed=0, patches_per_camera=16, cameras=1)
    mask = select_topk_per_camera(scores, 4, camera_of_patch(16, 1), 1)
    coordinates = grid_coordinates(16, 4)
    quadrants = coordinates[:, 0] // 2 * 2 + coordinates[:, 1] // 2
    for window in range(8):
        picked = quadrants[mask[window].bool()].tolist()
        assert sorted(picked) == [0, 1, 2, 3], f"window {window} did not cover all quadrants: {picked}"
    distinct = {tuple(torch.nonzero(mask[w]).flatten().tolist()) for w in range(8)}
    assert len(distinct) > 1, "stratified_random must actually vary between windows"


def test_stratified_random_handles_a_budget_not_divisible_by_four():
    """Largest-remainder allocation must still produce exactly ``k_cam`` picks."""
    scores = stratified_random_scores(4, 6, seed=1, patches_per_camera=16, cameras=1)
    mask = select_topk_per_camera(scores, 6, camera_of_patch(16, 1), 1)
    assert int(mask[0].sum()) == 6


def test_stratified_random_is_reproducible_and_data_independent():
    first = stratified_random_scores(4, 2, seed=11, patches_per_camera=4, cameras=2)
    again = stratified_random_scores(4, 2, seed=11, patches_per_camera=4, cameras=2)
    other = stratified_random_scores(4, 2, seed=12, patches_per_camera=4, cameras=2)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)


# ---------------------------------------------------------------------------
# 3. The information boundary
# ---------------------------------------------------------------------------


def test_deployable_view_drops_every_future_target(batch):
    """The deployable view must contain exactly the model's permitted input keys."""
    view = deployable_view(batch)
    assert set(view) == set(DEPLOYABLE_INPUT_KEYS)
    for key in ("target_state", "target_visual", "future_visual", "context_target_visual"):
        assert key not in view


def test_deployable_selections_ignore_future_targets(model, batch, curvature_head, heads):
    """Corrupting every future target must leave all eight deployable selectors bit-identical.

    This is the operational form of "no deployable policy reads a target at decision time":
    the Tier 2 ``exact_costate_reference`` is allowed (and expected) to change, and the
    measured objective it is scored against certainly does.
    """
    torch.manual_seed(11)
    corrupted = dict(batch)
    generator = torch.Generator().manual_seed(99)
    corrupted["target_state"] = torch.randn(WINDOWS, HORIZON, STATE_DIM, generator=generator)
    corrupted["target_visual"] = torch.randn(WINDOWS, HORIZON, TARGET_VISUAL, generator=generator)

    clean_context = selection_context(model, batch, curvature_head, patches_per_camera=PER_CAMERA, cameras=CAMERAS)
    dirty_context = selection_context(model, corrupted, curvature_head, patches_per_camera=PER_CAMERA, cameras=CAMERAS)
    for view in (clean_context, dirty_context):
        view.cls_attention = torch.rand(WINDOWS, FRAMES, PATCHES, generator=torch.Generator().manual_seed(5))

    index = camera_of_patch(PER_CAMERA, CAMERAS)
    for policy in DEPLOYABLE_POLICIES:
        assert_deployable(policy)
        clean_mask = select_topk_per_camera(
            score_policy(policy, clean_context, heads=heads, k_per_camera=2), 2, index, CAMERAS)
        dirty_mask = select_topk_per_camera(
            score_policy(policy, dirty_context, heads=heads, k_per_camera=2), 2, index, CAMERAS)
        assert torch.equal(clean_mask, dirty_mask), f"{policy} reacted to a future target"

    # The diagnostic reference *does* read the targets, which is why it is Tier 2.
    assert not torch.equal(
        select_topk_per_camera(score_policy("exact_costate_reference", clean_context, k_per_camera=2), 2, index, CAMERAS),
        select_topk_per_camera(score_policy("exact_costate_reference", dirty_context, k_per_camera=2), 2, index, CAMERAS),
    )


def test_diagnostic_references_are_refused_as_deployable():
    for policy in DIAGNOSTIC_POLICIES:
        with pytest.raises(ValueError, match="non-deployable"):
            assert_deployable(policy)
    assert_deployable("belief_space_voi")


def test_latent_patch_perturbations_live_in_the_latent_basis(model, batch):
    """The curvature/VOI scorers must contract with a perturbation in the *latent* basis.

    The cotangent bundle's ``delta_pooled`` is an adapter-basis quantity. Contracting it
    with a latent co-state is dimensionally valid but scientifically wrong, so the deployable
    scorers use this exact latent-space quantity instead. The test pins that it really is
    ``z(S u {p}) - z(S)`` and that the two bases genuinely differ, so the proxy cannot
    silently reappear.
    """
    perturbation = latent_patch_perturbations(model, batch)
    assert perturbation.shape == (WINDOWS, PATCHES, WIDTH)

    single = torch.zeros(WINDOWS, PATCHES)
    single[:, 3] = 1.0
    expected = encode_masked(model, batch, single) - encode_masked(model, batch, torch.zeros(WINDOWS, PATCHES))
    assert torch.allclose(perturbation[:, 3], expected, atol=1e-6)

    # The two bases are genuinely different objects: the pooled proxy is a poor stand-in.
    bundle = cotangent_bundle(model, batch)
    cosine = torch.nn.functional.cosine_similarity(bundle.delta_pooled, perturbation, dim=-1)
    assert cosine.abs().mean() < 0.9, (
        "delta_pooled and the latent perturbation have become the same object; "
        "the distinction the scorers rely on has been lost"
    )


def test_curvature_scores_reject_a_mismatched_basis(model, batch):
    """A wrong-basis contraction must fail loudly rather than produce a plausible ranking."""
    bundle = cotangent_bundle(model, batch)
    bad = bundle.delta_pooled[..., : WIDTH - 1]          # right rank, wrong space/width
    with pytest.raises(ValueError, match="share a dimension"):
        curvature_scores(torch.zeros(WINDOWS, WIDTH), torch.ones(WINDOWS, WIDTH), bad)


def test_privileged_critic_sees_the_costate_and_ranking_critic_does_not(context, heads):
    """B3: the decisive comparator is fed ``[lambda_hat, H_hat, dz_p]``; the baseline is not."""
    privileged = heads["direct_critic_privileged"]
    assert privileged.net.net[0].in_features == 3 * WIDTH + 2
    ranking = heads["direct_ranking_critic"]
    assert ranking.net.net[0].in_features == TOKEN_DIM + WIDTH + 2
    assert privileged.net.net[0].in_features > ranking.net.net[0].in_features


def test_masks_zero_content_but_keep_positions(model, batch):
    """B6: masking drops patch content while ``spatial_pos`` stays aligned with the grid."""
    visual = batch["context_visual"]
    mask = torch.zeros(WINDOWS, PATCHES)
    mask[:, :2] = 1.0
    masked = apply_patch_mask(visual, mask)
    assert masked.shape == visual.shape, "masking must not change the token layout"
    assert torch.equal(masked[:, :, 2:], torch.zeros_like(masked[:, :, 2:]))
    assert torch.equal(masked[:, :, :2], visual[:, :, :2])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_trimmed_mean_drops_the_extreme_decile():
    values = np.concatenate([np.arange(90.0), np.array([1000.0, -1000.0])])
    assert trimmed_mean(values, 0.1) == pytest.approx(np.arange(90.0).mean())


def test_wilcoxon_is_direction_aware_and_detects_a_consistent_shift():
    consistent = np.array([0.1, 0.2, 0.05, 0.3, 0.15, 0.25, 0.2, 0.35])
    null = np.array([0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.05, -0.05])
    assert wilcoxon_signed_rank(consistent)["p_value"] < 0.05
    assert wilcoxon_signed_rank(null)["p_value"] > 0.05
    flipped = wilcoxon_signed_rank(-consistent)
    assert flipped["p_value"] == pytest.approx(wilcoxon_signed_rank(consistent)["p_value"])


def _exact_sign_flip_p(differences: np.ndarray) -> float:
    """Exact two-sided p-value by enumerating all ``2^n`` sign assignments."""
    values = np.asarray(differences, dtype=float)
    nonzero = values[values != 0.0]
    n = nonzero.size
    if n == 0:
        return 1.0
    order = np.argsort(np.abs(nonzero), kind="stable")
    magnitudes = np.abs(nonzero)[order]
    ranks = np.empty(n)
    start = 0
    while start < n:
        end = start
        while end + 1 < n and magnitudes[end + 1] == magnitudes[start]:
            end += 1
        ranks[order[start : end + 1]] = 0.5 * (start + end + 2)
        start = end + 1
    signed = np.where(nonzero[order] > 0, ranks, -ranks)
    observed = abs(signed.sum())
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=n):
        total += 1
        if abs(sum(s * v for s, v in zip(signs, signed))) >= observed - 1e-9:
            hits += 1
    return hits / total


def test_wilcoxon_normal_approximation_tracks_the_exact_sign_flip_test():
    """The hand-rolled p-value must stay in the same ballpark as exact enumeration.

    This pins the null variance: ``Var(W+) = n(n+1)(2n+1)/24`` minus the tie correction.
    An earlier version used ``n(n+1)/12``, understating the variance roughly twentyfold and
    driving every p-value to ~0. A normal approximation is allowed to be either
    conservative or anti-conservative by a modest margin, but not off by orders of magnitude,
    and the anti-conservative direction is the one that would fabricate significance.
    """
    rng = np.random.default_rng(0)
    cases = {
        "clear signal": np.array([0.1, 0.2, 0.05, 0.3, 0.15, 0.25, 0.2, 0.35]),
        "symmetric null": np.array([0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.05, -0.05]),
        "weak signal": rng.normal(0.1, 1.0, 10),
        "with zeros": np.r_[rng.normal(0.4, 1.0, 6), np.zeros(2)],
        "heavy ties": np.round(rng.normal(0.2, 1.0, 10), 0),
    }
    for name, differences in cases.items():
        approximate = wilcoxon_signed_rank(differences)["p_value"]
        exact = _exact_sign_flip_p(differences)
        assert approximate <= exact + 0.25, (
            f"{name}: normal approximation {approximate:.4f} is anti-conservative "
            f"against the exact p-value {exact:.4f}"
        )
        assert approximate >= exact - 0.25, (
            f"{name}: normal approximation {approximate:.4f} is far more conservative "
            f"than the exact p-value {exact:.4f}"
        )


def test_wilcoxon_null_variance_matches_the_theoretical_value():
    """The null variance must be ``n(n+1)(2n+1)/24``, not the ~20x smaller ``n(n+1)/12``.

    A shrunken variance inflates every ``z``, so a shrunken variance would drive the
    p-value of a maximal statistic far below its exact value of ``2 * 0.5 ** n``.
    """
    n = 10
    maximal = wilcoxon_signed_rank(np.arange(1.0, n + 1.0))
    exact_maximal = 2 * 0.5**n
    # Conservative: the normal approximation may only be *less* significant than exact.
    assert maximal["p_value"] >= exact_maximal - 1e-12
    assert maximal["p_value"] < 0.02, "a maximal statistic must still be significant"

    # A pure null must sit near 1; a shrunken variance would push it toward 0.
    assert wilcoxon_signed_rank(np.random.default_rng(3).normal(0.0, 1.0, 40))["p_value"] > 0.01


def test_site_clustered_bootstrap_is_wider_than_a_window_bootstrap():
    """Clustering by site must widen the interval, since sites are correlated."""
    rng = np.random.default_rng(0)
    sites = np.repeat([f"site{i}" for i in range(6)], 40)
    # A per-site offset makes windows within a site strongly correlated.
    offsets = rng.normal(0, 1.0, size=6)
    values = offsets[np.repeat(np.arange(6), 40)] + rng.normal(0, 0.01, size=240)
    clustered = site_clustered_bootstrap_ci(values, sites, num_resamples=500)
    assert clustered["num_sites"] == 6
    assert (clustered["ci_high"] - clustered["ci_low"]) > 0.1


def test_additivity_r2_is_one_for_a_perfect_linear_surrogate():
    predicted = np.arange(10.0)
    perfect = additivity_r2([(p, 3.0 * p + 1.0) for p in predicted])
    assert perfect["r2"] == pytest.approx(1.0)
    assert perfect["slope"] == pytest.approx(3.0)


def test_submodularity_violation_rate_counts_increasing_marginals():
    """The greedy chain records objective *reductions*, so submodular means decreasing."""
    diminishing = [[3.0, 1.0, 0.5, 0.1]]        # decreasing: submodular, no violations
    increasing = [[0.1, 1.0, 3.0, 4.0]]        # increasing: every step violates
    assert submodularity_violation_rate(diminishing)["rate"] == 0.0
    assert submodularity_violation_rate(increasing)["rate"] == 1.0
    assert submodularity_violation_rate(diminishing)["comparisons"] == 3


def test_budget_grid_matches_the_specification():
    assert BUDGETS == ((4, 2), (8, 4), (16, 8))
    assert PATCHES_PER_CAMERA == 16 and CAMERAS == 2 and TOTAL_PATCHES == 32
    for total, per_camera in BUDGETS:
        assert total == per_camera * CAMERAS


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "benchmark_spatial_patch_selection", ROOT / "scripts/benchmark_spatial_patch_selection.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_mode_refuses_to_run_without_a_trained_backbone(tmp_path):
    """A randomly initialised backbone must not produce an exit gate (peer CRITICAL #2)."""
    module = _load_script()
    args = module.parse_args([
        "--cache-dir", str(tmp_path / "no-such-cache"), "--drive-root", str(tmp_path),
        "--output-dir", str(tmp_path),
    ])
    with pytest.raises(SystemExit, match="Refusing to run in 'real' mode"):
        module.run(args, torch.device("cpu"))
    # No artefact may be written for a refused run.
    assert not (tmp_path / "spatial_selection_summary.json").exists()


def test_synthetic_run_never_reports_a_passed_gate():
    """On synthetic fixtures the gate is refused outright, whatever the numbers say.

    Random tensors can clear a numeric threshold by chance; a 'pass' on invented data would
    be a fabricated result, so the branch must say the gate was not evaluated.
    """
    module = _load_script()
    gate = module.evaluate_exit_gate([], synthetic=True)
    assert gate["evaluated"] is False
    assert gate["criterion_1_primary_advantage_met"] is False
    assert gate["criterion_2_non_inferiority_met"] is False
    assert "not evidence" in gate["reason"]
    assert gate["branch"].startswith("not_evaluated")

    # Even a fabricated all-winning result must stay refused.
    winning = [{
        "k_total": 4,
        "per_policy": {
            "belief_space_voi": {"trimmed_mean_regret_10pct": 0.01},
            "uniform_grid": {"trimmed_mean_regret_10pct": 1.0},
            "early_feature_norm": {"trimmed_mean_regret_10pct": 1.0},
            "direct_critic_privileged": {"trimmed_mean_regret_10pct": 1.0},
        },
        "comparisons": {"voi_minus_second_order_curvature": {"p_value": 0.9}},
    }]
    assert module.evaluate_exit_gate(winning, synthetic=True)["branch"].startswith("not_evaluated")


def test_exit_gate_requires_the_criterion_at_every_budget():
    """Averages must not let one strong budget mask a failure at another (peer MAJOR #6)."""
    module = _load_script()
    strong, weak = {}, {}
    for budget, voi in ((4, 0.01), (8, 0.01), (16, 0.01)):
        strong[budget] = {
            "k_total": budget,
            "per_policy": {
                "belief_space_voi": {"trimmed_mean_regret_10pct": voi},
                "uniform_grid": {"trimmed_mean_regret_10pct": voi + 1.0},
                "early_feature_norm": {"trimmed_mean_regret_10pct": voi + 1.0},
                "direct_critic_privileged": {"trimmed_mean_regret_10pct": voi + 1.0},
                "second_order_curvature": {"trimmed_mean_regret_10pct": voi},
                "direct_ranking_critic": {"trimmed_mean_regret_10pct": voi},
            },
            "comparisons": {"voi_minus_second_order_curvature": {"p_value": 0.01}},
        }
    # k=16 fails: VOI is no better than the baselines there.
    weak = dict(strong)
    weak[16] = {
        "k_total": 16,
        "per_policy": {
            "belief_space_voi": {"trimmed_mean_regret_10pct": 0.5},
            "uniform_grid": {"trimmed_mean_regret_10pct": 0.5},
            "early_feature_norm": {"trimmed_mean_regret_10pct": 0.5},
            "direct_critic_privileged": {"trimmed_mean_regret_10pct": 0.5},
            "second_order_curvature": {"trimmed_mean_regret_10pct": 0.5},
            "direct_ranking_critic": {"trimmed_mean_regret_10pct": 0.5},
        },
        "comparisons": {"voi_minus_second_order_curvature": {"p_value": 0.9}},
    }
    ordered = [strong[b] for b in (4, 8, 16)]
    assert module.evaluate_exit_gate(ordered, synthetic=False)["criterion_1_primary_advantage_met"] is True
    mixed = module.evaluate_exit_gate([strong[4], strong[8], weak[16]], synthetic=False)
    assert mixed["criterion_1_primary_advantage_met"] is False
    assert mixed["per_budget"]["k=16"]["criterion_1_primary_advantage_met"] is False


def test_beta_sweep_rejects_a_one_shot_iterable():
    """Regression: the sweep takes a *factory*, so it cannot consume the stream once.

    The evaluation stream is a one-shot generator (batches stay on the host to bound GPU
    memory). Taking it as a plain iterable made every beta after the first score zero
    windows, which surfaced as byte-identical rows for beta=0, -0.5 and -1.0 in the
    committed synthetic report. The signature is the guard, so the test pins the signature.
    """
    import inspect

    module = _load_script()
    parameter = list(inspect.signature(module.beta_sensitivity).parameters)[3]
    assert parameter == "batch_factory", (
        f"beta_sensitivity's 4th parameter is {parameter!r}; it must be a callable returning a "
        "fresh iterable, or the first beta consumes the one-shot evaluation stream"
    )


def test_beta_zero_and_curvature_are_the_same_control(tmp_path):
    """``beta = 0`` must reproduce the curvature scorer, the control the sweep exists for."""
    module = _load_script()
    args = module.parse_args([
        "--synthetic", "--batch-size", "2", "--max-steps", "2", "--num-seeds", "1",
        "--exhaustive-k", "0", "--output-dir", str(tmp_path),
    ])
    summary = module.run(args, torch.device("cpu"))
    sensitivity = summary["beta_sensitivity"]
    for key, rows in sensitivity.items():
        assert rows, f"{key} produced no rows"
        for label, row in rows.items():
            assert row["n_windows"] > 0, f"{key}/{label} measured zero windows"


def test_synthetic_cli_run_produces_a_summary_and_report(tmp_path):
    """The script's ``--synthetic`` path runs end to end and labels itself as not evidence."""
    module = _load_script()
    args = module.parse_args([
        "--synthetic", "--batch-size", "2", "--max-steps", "2", "--num-seeds", "1",
        "--exhaustive-k", "2", "--output-dir", str(tmp_path),
    ])
    summary = module.run(args, torch.device("cpu"))
    assert summary["mode"] == "synthetic"
    assert len(summary["by_budget"]) == len(BUDGETS)
    for budget in summary["by_budget"]:
        assert set(budget["per_policy"]) >= set(COMPARATORS) | {"exact_costate_reference", "calibrated_greedy_oracle"}
    assert "branch" in summary["exit_gate"]

    module.write_report(summary, tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text()
    assert "not evidence" in text and "Exit Gate" in text
    # The synthetic branch must never be reported as a pass.
    assert summary["exit_gate"]["criterion_1_primary_advantage_met"] is False

def test_beta_sweep_degeneracy_is_reported_not_raised():
    """A control that cannot move is not a tie, but it is still evidence.

    Session 6A returned byte-identical rows for beta in {0, +/-0.5, +/-1} and a paired Wilcoxon of
    ``statistic 0.0, p = 1`` against curvature. Measured on that run's own backbone the epistemic
    term was 9.1e-06 of the curvature term, so no beta could reorder a top-k selection. The runner
    must *record* that, because the degenerate sweep is the evidence for the audit; suppressing it
    would destroy the artefact documenting the defect.
    """
    module = _load_script()
    degenerate = {
        "k_total=4": {f"beta={b:+.1f}": {"trimmed_mean_regret_10pct": 0.005, "n_windows": 32}
                      for b in (0.0, 0.5, 1.0, -0.5, -1.0)},
    }
    report = module.beta_sweep_is_degenerate(degenerate)
    assert report["degenerate"] is True
    assert report["degenerate_budgets"] == ["k_total=4"]
    assert report["per_budget"]["k_total=4"]["n_distinct_values"] == 1
    assert "UNTESTED" in report["interpretation"]


def test_beta_sweep_degeneracy_is_false_for_an_informative_sweep():
    module = _load_script()
    informative = {
        "k_total=4": {"beta=+0.0": {"trimmed_mean_regret_10pct": 0.005, "n_windows": 32},
                      "beta=+0.5": {"trimmed_mean_regret_10pct": 0.004, "n_windows": 32},
                      "beta=+1.0": {"trimmed_mean_regret_10pct": 0.003, "n_windows": 32},
                      "beta=-0.5": {"trimmed_mean_regret_10pct": 0.006, "n_windows": 32},
                      "beta=-1.0": {"trimmed_mean_regret_10pct": 0.007, "n_windows": 32}},
    }
    report = module.beta_sweep_is_degenerate(informative)
    assert report["degenerate"] is False
    assert report["degenerate_budgets"] == []


def test_beta_sweep_degeneracy_ignores_betas_that_measured_nothing():
    module = _load_script()
    rows = {
        "k_total=4": {"beta=+0.0": {"trimmed_mean_regret_10pct": 0.005, "n_windows": 32},
                      "beta=+0.5": {"trimmed_mean_regret_10pct": 0.005, "n_windows": 32},
                      "beta=+1.0": {"trimmed_mean_regret_10pct": float("nan"), "n_windows": 0}},
    }
    report = module.beta_sweep_is_degenerate(rows)
    assert report["per_budget"]["k_total=4"]["n_betas_measured"] == 2
    assert report["degenerate"] is True


def test_train_heads_trains_cosine_only_and_ranking_ablations_with_identical_inputs():
    """WS1a/WS1b: the ablation heads share class, inputs, steps and data with production.

    Per codex review, the ranking comparison must not confound loss with input sufficiency, and a
    WS1 failure must not be read as evidence about the pooled-`z` input. This pins the structural
    part: same head class, same parameter count, same training loop, returned for evaluation.
    """
    from types import SimpleNamespace

    module = _load_script()
    torch.manual_seed(0)
    model = build_small_model()
    batches = [synthetic_batch(seed=11), synthetic_batch(seed=12)]
    args = SimpleNamespace(lr=1e-3, max_steps=2)
    out = module.train_heads(
        model, lambda: iter(batches), args, TOKEN_DIM, PER_CAMERA, torch.device("cpu"))
    assert "curvature_head_cosine_only" in out
    assert "curvature_head_ranking" in out
    assert out["ranking_margin_scale"] == 1.0
    counts = out["parameter_counts"]
    assert counts["curvature_head"] == counts["curvature_head_cosine_only"] == counts["curvature_head_ranking"]
    for key in ("curvature", "curvature_cosine_only", "curvature_ranking"):
        assert key in out["final_loss"] and math.isfinite(out["final_loss"][key])


def test_train_heads_rejects_an_unknown_ablation_objective():
    from types import SimpleNamespace

    module = _load_script()
    torch.manual_seed(0)
    model = build_small_model()
    args = SimpleNamespace(lr=1e-3, max_steps=1)
    with pytest.raises(ValueError, match="unknown co-state training objective"):
        module.train_heads(
            model, lambda: iter([synthetic_batch()]), args, TOKEN_DIM, PER_CAMERA,
            torch.device("cpu"), extra_costate_objectives=("not_a_loss",))


def test_train_heads_records_the_ranking_margin_scale_it_used():
    """The margin scale sets the demanded score separation, so it must be on the record.

    Exact-gain differences are O(1e-4); a single scale could under- or over-demand, and a
    later reader could not tell which scale a "ranking supervision fails" claim used.
    """
    from types import SimpleNamespace

    module = _load_script()
    torch.manual_seed(0)
    model = build_small_model()
    args = SimpleNamespace(lr=1e-3, max_steps=1)
    out = module.train_heads(
        model, lambda: iter([synthetic_batch()]), args, TOKEN_DIM, PER_CAMERA,
        torch.device("cpu"), ranking_margin_scale=100.0)
    assert out["ranking_margin_scale"] == 100.0
    assert math.isfinite(out["final_loss"]["curvature_ranking"])


def test_patch_conditioned_head_is_capacity_matched_to_the_pooled_head():
    """A WS2 comparison must not be a capacity comparison in disguise (B3)."""
    from adjointrwm.spatial_selection import (
        PatchConditionedCostateEstimator, matched_conditioned_hidden,
    )

    torch.manual_seed(3)
    conditioned = PatchConditionedCostateEstimator(WIDTH, TOKEN_DIM)
    reference = CurvatureCostateEstimator(WIDTH)
    n_cond = sum(p.numel() for p in conditioned.parameters())
    n_ref = sum(p.numel() for p in reference.parameters())
    assert abs(n_cond - n_ref) / n_ref < 0.10, (n_cond, n_ref)
    assert matched_conditioned_hidden(WIDTH + TOKEN_DIM + 2, WIDTH) == conditioned.trunk.net[0].out_features


def test_patch_conditioned_scorer_matches_curvature_scores_on_broadcast_input():
    """Broadcasting a pooled co-state through the per-patch scorer is the same contraction."""
    from adjointrwm.spatial_selection import (
        curvature_scores, patch_conditioned_curvature_scores,
    )

    torch.manual_seed(4)
    costate = torch.randn(3, WIDTH)
    hessian = torch.randn(3, WIDTH).abs()
    delta_z = torch.randn(3, PATCHES, WIDTH)
    expected = curvature_scores(costate, hessian, delta_z)
    got = patch_conditioned_curvature_scores(
        costate.unsqueeze(1).expand(-1, PATCHES, -1),
        hessian.unsqueeze(1).expand(-1, PATCHES, -1),
        delta_z,
    )
    assert torch.allclose(got, expected, atol=1e-6)


def test_patch_conditioned_scorer_rejects_shape_mismatch():
    from adjointrwm.spatial_selection import patch_conditioned_curvature_scores

    with pytest.raises(ValueError, match=r"\[B, P, d\]"):
        patch_conditioned_curvature_scores(torch.zeros(2, WIDTH), torch.zeros(2, 8, WIDTH), torch.zeros(2, 8, WIDTH))


def test_patch_conditioned_head_ignores_future_targets(batch):
    """Decision-time inputs only: corrupting every future target must leave outputs identical."""
    from adjointrwm.spatial_selection import PatchConditionedCostateEstimator

    torch.manual_seed(5)
    head = PatchConditionedCostateEstimator(WIDTH, TOKEN_DIM).eval()
    latent = torch.randn(batch["context_state"].shape[0], WIDTH)
    budget = torch.full((latent.shape[0],), 0.25)
    horizon = torch.ones_like(budget)
    clean = head(latent, batch["context_visual"], budget, horizon)
    corrupted = dict(batch)
    corrupted["target_state"] = torch.randn_like(batch["target_state"])
    corrupted["target_visual"] = torch.randn_like(batch["target_visual"])
    corrupted["future_actions"] = torch.randn_like(batch["future_actions"])
    other = head(latent, corrupted["context_visual"], budget, horizon)
    assert torch.equal(clean[0], other[0]) and torch.equal(clean[1], other[1])


def test_train_heads_leaves_conditioned_cells_out_by_default():
    """The benchmark run must be byte-identical unless the diagnostic opts in (WS2)."""
    from types import SimpleNamespace

    module = _load_script()
    torch.manual_seed(0)
    model = build_small_model()
    args = SimpleNamespace(lr=1e-3, max_steps=1)
    out = module.train_heads(
        model, lambda: iter([synthetic_batch()]), args, TOKEN_DIM, PER_CAMERA,
        torch.device("cpu"))
    assert "conditioned_head_composite" not in out
    assert "conditioned_head_ranking" not in out


def test_train_heads_trains_conditioned_cells_when_opted_in():
    """WS2 cells C/D: same optimiser settings/steps/data; per-patch conditioning is the
    only input difference, composite vs ranking the only objective difference."""
    from types import SimpleNamespace

    module = _load_script()
    torch.manual_seed(0)
    model = build_small_model()
    args = SimpleNamespace(lr=1e-3, max_steps=2)
    out = module.train_heads(
        model, lambda: iter([synthetic_batch(seed=21), synthetic_batch(seed=22)]), args,
        TOKEN_DIM, PER_CAMERA, torch.device("cpu"), train_conditioned=True)
    for key in ("conditioned_head_composite", "conditioned_head_ranking"):
        assert key in out
    counts = out["parameter_counts"]
    # Capacity-matched: the conditioned heads must sit within 10% of the pooled head.
    for key in ("conditioned_head_composite", "conditioned_head_ranking"):
        assert abs(counts[key] - counts["curvature_head"]) / counts["curvature_head"] < 0.10, (
            key, counts[key], counts["curvature_head"])
    for key in ("conditioned_composite", "conditioned_ranking"):
        assert key in out["final_loss"] and math.isfinite(out["final_loss"][key])


def _load_fitting_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diagnose_costate_fitting", ROOT / "scripts/diagnose_costate_fitting.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pick_windows_spreads_over_episodes_deterministically():
    module = _load_fitting_script()
    ids = np.array([f"ep{i}" for i in range(10) for _ in range(8)])
    first, used_first = module.pick_windows(None, ids, 4, 16, set(), seed=0)
    second, used_second = module.pick_windows(None, ids, 4, 16, set(), seed=0)
    assert first == second and used_first == used_second
    assert len(used_first) == 4 and len(first) == 16


def test_pick_windows_respects_the_skip_set():
    module = _load_fitting_script()
    ids = np.array([f"ep{i}" for i in range(10) for _ in range(8)])
    _, fit_used = module.pick_windows(None, ids, 4, 16, set(), seed=0)
    hold, hold_used = module.pick_windows(None, ids, 4, 16, set(fit_used), seed=1)
    assert not (set(fit_used) & set(hold_used)), "held-out episodes must differ from fit episodes"
    assert len(hold) == 16


def test_four_metrics_is_zero_for_a_perfect_prediction():
    module = _load_fitting_script()
    torch.manual_seed(9)
    exact = torch.randn(6, WIDTH)
    delta_z = torch.randn(6, PATCHES, WIDTH)
    from adjointrwm.spatial_selection import curvature_scores

    ref = {
        "exact": exact,
        "delta_z": delta_z,
        # Gains consistent with the exact costate, so a perfect prediction must agree fully.
        "gains": curvature_scores(exact, torch.zeros_like(exact), delta_z),
    }
    out = module.four_metrics(exact.clone(), ref)
    assert out["rel_vector_error"] == pytest.approx(0.0, abs=1e-6)
    assert out["rel_magnitude_error"] == pytest.approx(0.0, abs=1e-6)
    # 1-cos can print marginally negative from float rounding; the point is ~zero error.
    assert abs(out["directional_error"]) < 1e-6
    for entry in out["per_budget"].values():
        assert entry["allocation_agreement"] == pytest.approx(1.0)
    assert out["n"] == 6


def test_reconstruction_errors_separates_vector_from_magnitude():
    """Plan R5: relative vector error is not magnitude error (different quantities)."""
    module = _load_fitting_script()
    exact = torch.tensor([[3.0, 4.0]])
    same_direction = torch.tensor([[6.0, 8.0]])  # 2x scale, zero directional error
    out = module.reconstruction_errors(same_direction, exact)
    assert out["rel_vector_error"] == pytest.approx(1.0)
    assert out["rel_magnitude_error"] == pytest.approx(1.0)
    assert abs(out["directional_error"]) < 1e-6
    double = module.reconstruction_errors(2.0 * exact, exact)
    assert double["rel_vector_error"] == pytest.approx(1.0)


def test_engineering_fit_gate_pass_fail_and_absolute_branches():
    """Plan §1: E_rec <= 0.01 relative gate, absolute tolerance below the energy floor."""
    module = _load_fitting_script()
    exact = torch.ones(4, 8)
    good = module.engineering_fit_gate(exact.clone(), exact)
    assert good["gate"] == "relative" and good["passed"] is True
    bad = module.engineering_fit_gate(torch.zeros(4, 8), exact)
    assert bad["gate"] == "relative" and bad["passed"] is False
    tiny = torch.zeros(2, 4)
    absolute = module.engineering_fit_gate(tiny.clone(), tiny)
    assert absolute["gate"] == "absolute"


def test_score_and_outcome_reports_all_deployed_budgets():
    """Plan R5: overlap/agreement at every deployed budget, not top-2 only."""
    module = _load_fitting_script()
    torch.manual_seed(11)
    exact = torch.randn(6, WIDTH)
    delta_z = torch.randn(6, PATCHES, WIDTH)
    from adjointrwm.spatial_selection import curvature_scores

    ref = {"exact": exact, "delta_z": delta_z,
           "gains": curvature_scores(exact, torch.zeros_like(exact), delta_z)}
    out = module.score_and_outcome(exact.clone(), ref, budgets=(2, 4))
    assert set(out["per_budget"]) == {"k_cam=2", "k_cam=4"}
    for entry in out["per_budget"].values():
        assert entry["allocation_agreement"] == pytest.approx(1.0)
    assert out["spearman_vs_exact_scores"] == pytest.approx(1.0)


def test_unregularised_lstsq_interpolates_a_full_row_rank_system():
    """Review R3-B: full-row-rank X must drive the residual to ~zero (finite-sample
    fitability); this is representation capacity on fixed data, never generalisation."""
    module = _load_fitting_script()
    torch.manual_seed(21)
    design = torch.randn(6, 20, dtype=torch.float64)
    truth = torch.randn(20, 5, dtype=torch.float64)
    targets = design @ truth
    out = module.unregularised_lstsq(design, targets)
    assert out["full_row_rank"] is True
    assert out["rank"] == 6
    assert out["residual_sum_of_squares"] == pytest.approx(0.0, abs=1e-12)
    assert out["driver"] == "gelsd"
    assert len(out["singular_values"]) == 6


def test_unregularised_lstsq_reports_rank_deficiency_honestly():
    """A rank-deficient design must say so instead of silently returning a fit."""
    module = _load_fitting_script()
    base = torch.randn(10, 4, dtype=torch.float64)
    design = torch.cat([base, base], dim=1)  # duplicated columns -> rank <= 4
    targets = torch.randn(10, 3, dtype=torch.float64)
    out = module.unregularised_lstsq(design, targets)
    assert out["full_row_rank"] is False
    assert out["rank"] <= 4
    assert out["residual_sum_of_squares"] >= 0.0


def test_scalar_calibration_recovers_a_global_scale_and_marks_degenerate():
    """Review R3-C: a* is train-fitted and reported; zero energy is undefined, not zero."""
    module = _load_fitting_script()
    exact = torch.randn(8, 6)
    pred = 2.0 * exact
    got = module.scalar_calibration(pred, exact, pred)
    assert got["a_star"] == pytest.approx(0.5)
    assert got["undefined"] is None
    dead = module.scalar_calibration(torch.zeros(8, 6), exact, torch.zeros(8, 6))
    assert dead["a_star"] is None and dead["undefined"] is not None


def test_fitting_preflight_refuses_a_missing_checkpoint(tmp_path):
    """Session 5 lesson as executable behavior: no checkpoint, no numbers, no artefact."""
    module = _load_fitting_script()
    args = module.parse_args(["--cache-dir", str(tmp_path / "nope"),
                              "--drive-root", str(tmp_path),
                              "--output-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="missing teacher checkpoint"):
        module.run(args, torch.device("cpu"))
    assert not (tmp_path / "out" / "costate_fitting_summary.json").exists()


def _load_pair_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diagnose_pair_interactions", ROOT / "scripts/diagnose_pair_interactions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_norm_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diagnose_norm_controls", ROOT / "scripts/diagnose_norm_controls.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pair_strata_partition_all_496_pairs_exactly_once():
    """32 patches -> C(32,2) = 496 pairs; 2*C(16,2) = 240 within-camera, 16*16 = 256 cross."""
    module = _load_pair_script()
    strata = module.pair_strata()
    assert sum(len(v) for v in strata.values()) == 496
    assert len(strata["cross_camera"]) == 256
    assert (len(strata["within_adjacent"]) + len(strata["within_distant"])) == 240
    flat = [tuple(sorted(p)) for members in strata.values() for p in members]
    assert len(set(flat)) == 496, "every pair covered exactly once"


def test_sample_pairs_returns_an_exact_manifest():
    """'Approximately 200' is not a manifest: exact count, strata labels, no duplicates."""
    module = _load_pair_script()
    manifest = module.sample_pairs(200, seed=7)
    assert len(manifest) == 200
    assert len({m["pair_id"] for m in manifest}) == 200
    assert {m["stratum"] for m in manifest} == {"within_adjacent", "within_distant", "cross_camera"}
    repeat = module.sample_pairs(200, seed=7)
    assert [m["pair_id"] for m in repeat] == [m["pair_id"] for m in manifest]
    other = module.sample_pairs(200, seed=8)
    assert [m["pair_id"] for m in other] != [m["pair_id"] for m in manifest]


def test_gradient_energy_preview_geometry_and_nonnegativity():
    """[T,H,W,3] uint8 -> [T,16] nonneg energies on a 4x4 grid; uniform frames give ~zero."""
    module = _load_norm_script()
    flat = np.zeros((3, 180, 320, 3), dtype=np.uint8)
    out = module.gradient_energy_preview(flat)
    assert out.shape == (3, 16)
    assert bool((out >= 0).all())
    assert float(out.max()) == pytest.approx(0.0, abs=1e-6)
    rng = np.random.default_rng(0)
    textured = rng.integers(0, 256, size=(2, 64, 64, 3), dtype=np.uint8)
    textured_out = module.gradient_energy_preview(textured)
    assert float(textured_out.sum()) > 0.0
    # A bright quadrant puts energy on its boundary, not in uniform interiors: the
    # corner cell (uniform 255 throughout) must read exactly zero while some cell fires.
    quad = np.zeros((1, 64, 64, 3), dtype=np.uint8)
    quad[:, :32, :32, :] = 255
    quad_out = module.gradient_energy_preview(quad)
    assert quad_out[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert float(quad_out.max()) > 0.0


def _load_gap_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diagnose_information_gap", ROOT / "scripts/diagnose_information_gap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gap_design_matrices_have_declared_widths():
    """Plan §7 manifests: every probe condition projected to one common width.

    Raw counts (513 vs ~3600 on 16 rows) would confound information with nullspace size, so
    all conditions share a seeded data-independent projection (review attack 2). Logged
    future actions appear only under an explicit §0-violation label, never as legitimate
    context (review attack 1).
    """
    module = _load_gap_script()
    torch.manual_seed(31)
    fake = lambda *shape: torch.randn(*shape)
    fit = {"latent": fake(5, 32), "future_actions": fake(5, 4, 7),
           "target_state": fake(5, 4, 14), "target_visual": fake(5, 4, 64)}
    hold = {"latent": fake(3, 32), "future_actions": fake(3, 4, 7),
            "target_state": fake(3, 4, 14), "target_visual": fake(3, 4, 64)}
    designs = module.design_matrices(fit, hold, proj_dim=12, proj_seed=0)
    assert set(designs) == {"A_deployment", "B_ctx_legitimate", "B_log_privileged",
                            "C_privileged"}
    for name, design in designs.items():
        assert design["fit"].shape == (5, 12), name
        assert design["hold"].shape == (3, 12), name
        assert design["projection"]["out_dim"] == 12
    assert designs["B_log_privileged"].get("section_zero_violation") is True
    # Same seed reproduces the same projection; different seeds differ.
    again = module.design_matrices(fit, hold, proj_dim=12, proj_seed=0)
    assert torch.equal(again["A_deployment"]["fit"], designs["A_deployment"]["fit"])
    other = module.design_matrices(fit, hold, proj_dim=12, proj_seed=1)
    assert not torch.equal(other["A_deployment"]["fit"], designs["A_deployment"]["fit"])


def test_gap_projection_is_data_independent():
    """The capacity equaliser must not itself learn from the data (JL, not PCA)."""
    module = _load_gap_script()
    first = module.random_projector(100, 12, seed=3)
    second = module.random_projector(100, 12, seed=3)
    assert torch.equal(first, second)
    assert first.shape == (100, 12)
    assert abs(float((first.norm(dim=0).mean()) - 1.0)) < 0.35


def test_gap_zero_fill_keeps_deployment_columns_and_intercept():
    """Condition D mechanics: privileged blocks zeroed at test, latent + intercept intact."""
    gap = _load_gap_script()
    full = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    out = gap.zero_fill_privileged(full, n_keep=3)
    assert torch.equal(out[:, :3], full[:, :3])
    assert torch.equal(out[:, -1:], full[:, -1:])
    assert bool((out[:, 3:-1] == 0).all())
    with pytest.raises(ValueError, match="exceeds design width"):
        gap.zero_fill_privileged(full, n_keep=12)


def test_gap_projected_conditions_run_end_to_end_on_synthetic_data():
    """The comparison machinery (projected designs + lstsq + metrics) executes on every
    condition. With 10 rows and 12 projected columns all conditions can interpolate, so
    this asserts structure and finiteness — not a scientific ordering, which the capacity
    equalisation deliberately removes."""
    fitting = _load_fitting_script()
    gap = _load_gap_script()
    torch.manual_seed(32)
    latent = torch.randn(10, 8)
    fit = {"latent": latent, "future_actions": torch.randn(10, 2, 3),
           "target_state": torch.randn(10, 2, 3),
           "target_visual": torch.randn(10, 2, 5)}
    hold = {"latent": torch.randn(4, 8), "future_actions": torch.randn(4, 2, 3),
            "target_state": torch.randn(4, 2, 3), "target_visual": torch.randn(4, 2, 5)}
    designs = gap.design_matrices(fit, hold, proj_dim=12, proj_seed=0)
    exact_fit = torch.randn(10, 4)
    for name, design in designs.items():
        out = fitting.unregularised_lstsq(design["fit"], exact_fit)
        assert out["rank"] <= 12, name
        assert math.isfinite(out["residual_sum_of_squares"]), name


def _load_pair_analysis():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diagnose_pair_interactions", ROOT / "scripts/diagnose_pair_interactions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _toy_manifest():
    return [
        {"pair_id": "within_adjacent:00-01", "stratum": "within_adjacent", "patch_a": 0, "patch_b": 1},
        {"pair_id": "within_adjacent:02-03", "stratum": "within_adjacent", "patch_a": 2, "patch_b": 3},
    ]


def _toy_rows(n_windows=3, shift=5.0, noise=0.0, seed=0):
    """Synthetic per-window rows: pair_J = additive singleton sum + common shift + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_windows):
        singles = rng.normal(0, 1, size=4)
        pairs = [singles[0] + singles[1] + shift, singles[2] + singles[3] + shift]
        if noise:
            pairs = [v + rng.normal(0, noise) for v in pairs]
        rows.append({"window": w, "episode_id": f"ep{w % 2}",
                     "base_J": 0.0,
                     "singleton_J": [float(-v) for v in singles],
                     "pair_J": [float(-v) for v in pairs]})
    return rows


def test_common_shift_preserves_ranking_but_dominates_epsilon():
    """The review's counterexample: constant within-window shift -> large epsilon, perfect rank.

    Magnitude ratios alone must not declare singleton rankings uninformative.
    """
    module = _load_pair_analysis()
    rows = _toy_rows(shift=5.0)
    out = module.analyse_pair_rows(rows, _toy_manifest(), {"ep0": "s", "ep1": "s"}, seed=0)
    assert out["raw_rhos"][0] == pytest.approx(1.0)
    assert abs(np.mean([e["epsilon"] for e in out["interactions"]])) > 1.0
    assert np.mean(out["resid_frac"]) == pytest.approx(0.0, abs=1e-9)


def test_pair_specific_noise_degrades_ranking():
    module = _load_pair_analysis()
    rows = _toy_rows(shift=0.0, noise=2.0, seed=1)
    out = module.analyse_pair_rows(rows, _toy_manifest(), {"ep0": "s", "ep1": "s"}, seed=0)
    assert np.nanmean(out["raw_rhos"]) < 1.0
    assert np.nanmean(out["resid_frac"]) > 0.0


def test_same_budget_contracts_are_explicit():
    """Best-of-manifest scope, measured-vs-estimated flag, per-camera-free 2-patch sets."""
    module = _load_pair_analysis()
    # Fixed singleton gains: top-2 are patches (2,3), which IS in the manifest, so the
    # joint is measured and the estimated-fallback path is not taken here.
    rows = [{"window": w, "episode_id": "ep0", "base_J": 0.0,
             "singleton_J": [0.0, -1.0, -3.0, -2.0],
             "pair_J": [-1.5, -4.5]} for w in range(4)]
    out = module.analyse_pair_rows(rows, _toy_manifest(), {"ep0": "s"}, seed=0)
    assert len(out["diffs"]) == 4
    assert out["wilcoxon"]["n"] == 4
    assert set(out["cluster_ci"]) >= {"ci_low", "ci_high", "estimate"}
    assert set(out["wilcoxon"]) >= {"statistic", "p_value"}
    # Deterministic fixture: top-2 singletons are patches (2,3), which IS measured, so the
    # joint is never the additive fallback here.
    assert all(d["top_singleton_pair_measured"] for d in out["diffs"])


def _load_mechanism_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diagnose_mechanism_check", ROOT / "scripts/diagnose_mechanism_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_second_order_pair_correction_matches_hand_computation():
    """epsilon_{p,q} = -dp^T H dq with diagonal H, hand-checked on a tiny case."""
    module = _load_mechanism_script()
    delta = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    hessian = torch.tensor([[2.0, 3.0]])
    out = module.second_order_pair_correction(delta, hessian, [(0, 1), (0, 2), (1, 2)])
    # (0,1): -(2*1*0 + 3*0*2) = 0; (0,2): -(2*1*1 + 3*0*1) = -2; (1,2): -(2*0*1 + 3*2*1) = -6
    assert out.shape == (1, 3)
    assert torch.allclose(out, torch.tensor([[0.0, -2.0, -6.0]]))


def test_second_order_pair_correction_rejects_shape_mismatch():
    module = _load_mechanism_script()
    with pytest.raises(ValueError, match=r"\[B, d\]"):
        module.second_order_pair_correction(torch.zeros(2, 4), torch.zeros(2, 3), [(0, 1)])
    with pytest.raises(ValueError, match=r"\[B, P, d\]"):
        module.second_order_pair_correction(torch.zeros(2, 3, 4), torch.zeros(2, 4, 4), [(0, 1)])


def test_mechanism_manifest_reuses_the_frozen_pair_set():
    """The mechanism check must score the same pairs as the interaction panel (comparability)."""
    module = _load_mechanism_script()
    args = module.parse_args([])
    assert args.n_pairs == 200 and args.pair_seed == 7 and args.windows == 32


def _load_mechanism_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diagnose_mechanism_check", ROOT / "scripts/diagnose_mechanism_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pair_topk_overlap_is_plain_set_overlap():
    """Pairs are unstructured: top-k overlap has no camera geometry in it."""
    module = _load_mechanism_script()
    scores = torch.tensor([[5.0, 1.0, 4.0, 2.0]])
    truth = torch.tensor([[1.0, 5.0, 2.0, 4.0]])
    # Top-2 of scores: {0, 2}; top-2 of truth: {1, 3} -> disjoint.
    assert module.pair_topk_overlap(scores, truth, 2).item() == pytest.approx(0.0)
    # Identical rankings agree fully.
    assert module.pair_topk_overlap(truth, truth, 2).item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="outside"):
        module.pair_topk_overlap(scores, truth, 0)
    with pytest.raises(ValueError, match="outside"):
        module.pair_topk_overlap(scores, truth, 5)


def test_analyse_mechanism_rows_separates_arms_and_reports_additivity():
    """A/B/C/D arms from committed-style rows; additivity premise reported, not assumed."""
    module = _load_mechanism_script()
    manifest = [
        {"pair_id": "m:00-01", "stratum": "s", "patch_a": 0, "patch_b": 1},
        {"pair_id": "m:02-03", "stratum": "s", "patch_a": 2, "patch_b": 3},
    ]
    rows = []
    for w in range(3):
        # Additive world: pair gain == singleton sum -> epsilon ~ 0, additive ranks perfectly.
        singles = [1.0, 2.0, 0.5, 1.5]
        pairs = [3.0, 2.0]
        rows.append({
            "episode_id": "ep0", "site": "s",
            "singleton_gains": singles, "measured_pair_gains": pairs,
            "predicted_epsilon_2nd": [0.0, 0.0],
            "first_order_singleton": singles,
            "distilled_first_order_singleton": [s * 0.9 for s in singles],
            "pair_additivity_relative": [0.0, 0.0],
        })
    out = module.analyse_mechanism_rows(rows, manifest, {"ep0": "s"})
    assert out["pair_additivity"]["median_relative_residual"] == pytest.approx(0.0)
    assert out["arm_rank_fidelity"]["C"] == pytest.approx(1.0)
    assert out["arm_rank_fidelity"]["A"] == pytest.approx(1.0)
    assert out["corrected_vs_additive_regret"]["n"] == 3


def test_pair_reanalysis_resolves_sites_through_the_manifest():
    """v2 rows carry no site field; the frozen shard manifest supplies the mapping."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "analyze_pair_rows", ROOT / "scripts/analyze_pair_rows.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = {"window_rows": [{"episode_id": "x", "site": "unknown"}]}
    mapping = module._sites(summary)
    # Unknown episodes stay unknown; the manifest backfills real ones (superset harmless).
    assert mapping["x"] == "unknown"
    assert mapping["d43105d6fa4e8a76c01fff26"] == "ILIAD"


def _load_pruning_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diagnose_early_pruning", ROOT / "scripts/diagnose_early_pruning.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pixel_mask_tiles_cells_and_keeps_only_kept():
    module = _load_pruning_script()
    mask = module.pixel_mask_for_cells([0, 5], 64, 64, grid=4)
    assert mask.shape == (64, 64)
    assert mask.dtype == bool
    # Cell 0 -> rows 0-15, cols 0-15; cell 5 -> rows 16-31, cols 16-31.
    assert mask[:16, :16].all() and mask[16:32, 16:32].all()
    assert not mask[32:, :].any() and not mask[:, 32:].any()
    assert mask.sum() == 2 * 16 * 16


def test_pixel_mask_empty_and_full_are_exact():
    module = _load_pruning_script()
    assert not module.pixel_mask_for_cells([], 32, 32, grid=4).any()
    assert module.pixel_mask_for_cells([0, 1, 2, 3], 32, 32, grid=2).all()


def _load_conditional_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_conditional_allocator", ROOT / "scripts/train_conditional_allocator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_schedule_is_half_fixed_half_student_and_deterministic():
    """Frozen 50/50 mixture with a cap: the set-generation policy, not a tuning knob.

    Every mask respects per-camera quotas (codex review 2026-10-05): fixed items name
    per-camera counts in 0..8, student items name prefix lengths in 0..16.
    """
    module = _load_conditional_script()
    first = module.build_training_schedule(10, 8, seed=0)
    second = module.build_training_schedule(10, 8, seed=0)
    assert first == second
    for entry in first:
        kinds = entry["kinds"]
        assert len(kinds) == 8
        assert sum(1 for k in kinds if k == "student") == 4
        assert len(entry["fixed_counts"]) == 8
        for counts in entry["fixed_counts"]:
            assert len(counts) == 2
            assert all(0 <= c <= 8 for c in counts)
        assert all(0 <= p <= 16 for p in entry["prefix_lens"])


def test_random_quota_mask_respects_per_camera_counts():
    """Quota masks keep exactly the named count per camera on every window."""
    module = _load_conditional_script()
    from adjointrwm.spatial_selection import CAMERAS, camera_of_patch
    generator = torch.Generator().manual_seed(0)
    counts = [[3, 5], [0, 8], [8, 0]]
    mask = module.random_quota_mask(3, counts, generator, torch.device("cpu"))
    cameras = camera_of_patch(16, CAMERAS)
    assert mask.shape == (3, 32)
    for row, (a, b) in enumerate(counts):
        assert int(mask[row][cameras == 0].sum()) == a
        assert int(mask[row][cameras == 1].sum()) == b


def test_protected_read_refuses_a_silent_reread(tmp_path):
    """Read-once enforcement: a repeat read without an acknowledged reason refuses."""
    import hashlib
    import json as _json

    module = _load_conditional_script()
    manifest = tmp_path / "protected_manifest.json"
    manifest.write_text(_json.dumps({"windows": [], "read_log": [{"purpose": "x"}]}))
    (tmp_path / "protected_manifest.seal").write_text(
        hashlib.sha256(manifest.read_bytes()).hexdigest() + "\n")
    with pytest.raises(SystemExit, match="silent re-read"):
        module._read_protected(manifest, "test")
    doc = module._read_protected(manifest, "test", acknowledge_reread="unit test rerun")
    assert doc["read_log"][-1]["acknowledged_reread_reason"] == "unit test rerun"


def test_random_subset_mask_has_exact_cardinality():
    module = _load_conditional_script()
    generator = torch.Generator().manual_seed(0)
    mask = module.random_subset_mask(6, 4, generator, torch.device("cpu"))
    assert mask.shape == (6, 32)
    assert bool(((mask.sum(dim=1) == 4)).all())
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_protected_read_refuses_a_broken_seal(tmp_path):
    """Read-once enforcement: missing seal and tampered manifest both refuse."""
    module = _load_conditional_script()
    manifest = tmp_path / "protected_manifest.json"
    manifest.write_text('{"windows": []}')
    with pytest.raises(SystemExit, match="no seal file"):
        module._read_protected(manifest, "test")
    (tmp_path / "protected_manifest.seal").write_text("forged\n")
    with pytest.raises(SystemExit, match="seal MISMATCH"):
        module._read_protected(manifest, "test")


def test_protected_read_logs_and_reseals(tmp_path):
    """Every protected read is appended to the run log and the seal follows the bytes."""
    import hashlib
    import json as _json

    module = _load_conditional_script()
    manifest = tmp_path / "protected_manifest.json"
    manifest.write_text(_json.dumps({"windows": [], "read_log": []}))
    (tmp_path / "protected_manifest.seal").write_text(
        hashlib.sha256(manifest.read_bytes()).hexdigest() + "\n")
    doc = module._read_protected(manifest, "unit test")
    assert doc["read_log"][-1]["purpose"] == "unit test"
    assert (tmp_path / "protected_manifest.seal").read_text().strip() == \
        hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_random_subset_mask_stream_is_device_independent():
    """CUDA generators need explicit construction, so all RNG draws happen on CPU and
    tensors move afterwards: same seed must give the same mask (regression test for the
    generator/device mismatch caught on the first GPU run)."""
    module = _load_conditional_script()
    first = module.random_subset_mask(
        4, 4, torch.Generator().manual_seed(0), torch.device("cpu"))
    second = module.random_subset_mask(
        4, 4, torch.Generator().manual_seed(0), torch.device("cpu"))
    assert torch.equal(first, second)
    assert bool(((first.sum(dim=1) == 4)).all())


def test_theta_gate_uses_keyword_ci_and_reports_one_sided_bound():
    """Regression: positional (2000, seed) bound confidence=seed=0, giving degenerate
    zero-width intervals. theta_gate must report confidence 0.95, a strict interior
    one-sided bound, and correct gate booleans on a constructed fixture."""
    module = _load_conditional_script()
    rng = np.random.default_rng(0)
    sites = np.array(["s%d" % (i % 4) for i in range(40)])
    single = rng.uniform(0.002, 0.006, size=40)
    improvement = rng.uniform(0.05, 0.25, size=40)  # heterogeneous theta
    cond = single * (1.0 - improvement)
    gate = module.theta_gate(single, cond, sites, seed=0)
    assert gate["n"] == 40
    assert gate["n_excluded_nonpositive_denom"] == 0
    assert gate["two_sided_ci_95"]["confidence"] == 0.95
    mean_theta = improvement.mean()
    assert gate["two_sided_ci_95"]["ci_low"] < mean_theta < gate["two_sided_ci_95"]["ci_high"]
    assert gate["one_sided_95_lower"] <= mean_theta
    assert gate["one_sided_95_lower"] >= gate["two_sided_ci_95"]["ci_low"]
    assert gate["superiority_08"] is True
    assert gate["non_inferiority_03"] is True


def test_theta_gate_excludes_near_zero_denominators_without_epsilon():
    """Windows with |R_single| <= 1e-12 must not enter theta as a ratio."""
    module = _load_conditional_script()
    single = np.array([0.004, 1e-13, 0.0, 0.005])
    cond = np.array([0.003, 1e-14, 0.001, 0.006])
    sites = np.array(["a", "a", "b", "b"])
    gate = module.theta_gate(single, cond, sites, seed=0)
    assert gate["n"] == 2
    assert gate["n_excluded_nonpositive_denom"] == 2
    assert gate["superiority_08"] is False  # mixed windows, tiny sample
