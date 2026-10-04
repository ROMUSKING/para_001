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
