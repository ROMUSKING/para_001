"""Tests for the Session 6A patch-dropout backbone prerequisite (spec B6).

Covers the patch-budget sampler, the paired validation masks and the checkpoint contract the
benchmark relies on. The trainer's real-data path needs the 12-site DROID cache and is
exercised on Colab, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_DIR = Path(__file__).resolve().parent.parent
for candidate in (REPO_DIR / "src", REPO_DIR / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np

from adjointrwm.data import WindowSpec  # noqa: E402
from adjointrwm.data.windows import slice_window  # noqa: E402
from adjointrwm.spatial_selection import (  # noqa: E402
    CAMERAS,
    PATCHES_PER_CAMERA,
    TRAIN_BUDGETS_PER_CAMERA,
    apply_patch_mask,
    camera_of_patch,
    sample_budget_masks,
)
from train_spatial_patch_dropout_backbone import (  # noqa: E402
    VALIDATION_BUDGETS,
    fixed_budget_masks,
)


# ---------------------------------------------------------------------------
# sample_budget_masks
# ---------------------------------------------------------------------------


def test_mask_shape_and_binary_values():
    mask = sample_budget_masks(7, generator=torch.Generator().manual_seed(0))
    assert mask.shape == (7, PATCHES_PER_CAMERA * CAMERAS)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_mask_is_symmetric_within_each_camera():
    """No camera may be emptied: that would confound spatial with modality selection (B5)."""
    mask = sample_budget_masks(64, generator=torch.Generator().manual_seed(1))
    cameras = camera_of_patch(PATCHES_PER_CAMERA, CAMERAS)
    for camera in range(CAMERAS):
        kept = mask[:, cameras == camera].sum(dim=1)
        assert int(kept.min()) >= 1, f"camera {camera} was emptied"


def test_mask_keeps_distinct_positions_within_a_camera():
    """randperm must not repeat an index; a duplicated patch would inflate the kept count."""
    mask = sample_budget_masks(32, generator=torch.Generator().manual_seed(2))
    cameras = camera_of_patch(PATCHES_PER_CAMERA, CAMERAS)
    for camera in range(CAMERAS):
        kept = mask[:, cameras == camera].sum(dim=1)
        assert int(kept.max()) <= PATCHES_PER_CAMERA


def test_mask_budget_is_drawn_only_from_the_allowed_set():
    budgets = (2, 4, 8)
    mask = sample_budget_masks(
        200, k_choices=budgets, generator=torch.Generator().manual_seed(3)
    )
    cameras = camera_of_patch(PATCHES_PER_CAMERA, CAMERAS)
    per_camera = torch.stack([mask[:, cameras == c].sum(dim=1) for c in range(CAMERAS)], dim=1)
    assert set(per_camera.unique().tolist()) <= set(float(b) for b in budgets)


def test_mask_draws_vary_across_windows_and_steps():
    """A sampler that always returns the same mask would train the k<P case on one subset."""
    generator = torch.Generator().manual_seed(4)
    first = sample_budget_masks(16, generator=generator)
    second = sample_budget_masks(16, generator=generator)
    assert not torch.equal(first, second)
    assert first.sum(dim=1).unique().numel() > 1, "every window drew the same budget"


def test_full_budget_choice_keeps_every_patch():
    mask = sample_budget_masks(
        4, k_choices=(PATCHES_PER_CAMERA,), generator=torch.Generator().manual_seed(5)
    )
    assert torch.equal(mask, torch.ones_like(mask))


@pytest.mark.parametrize("bad", [0, -1, PATCHES_PER_CAMERA + 1])
def test_rejects_out_of_range_budget(bad):
    with pytest.raises(ValueError, match="budget"):
        sample_budget_masks(2, k_choices=(bad,), generator=torch.Generator().manual_seed(6))


def test_rejects_empty_choices_and_bad_geometry():
    with pytest.raises(ValueError, match="must not be empty"):
        sample_budget_masks(2, k_choices=(), generator=torch.Generator().manual_seed(7))
    with pytest.raises(ValueError, match="is not"):
        sample_budget_masks(2, patches=PATCHES_PER_CAMERA * CAMERAS + 1,
                            generator=torch.Generator().manual_seed(8))


def test_rejects_non_positive_windows():
    with pytest.raises(ValueError, match="windows must be positive"):
        sample_budget_masks(0, generator=torch.Generator().manual_seed(9))


def test_is_deterministic_given_a_generator_seed():
    a = sample_budget_masks(8, generator=torch.Generator().manual_seed(11))
    b = sample_budget_masks(8, generator=torch.Generator().manual_seed(11))
    assert torch.equal(a, b)


def test_train_budgets_cover_evaluated_budgets_and_no_dropout_endpoint():
    """Training must include every evaluated budget and the full-patch endpoint (spec B6)."""
    assert {2, 4, 8} <= set(TRAIN_BUDGETS_PER_CAMERA)
    assert PATCHES_PER_CAMERA in TRAIN_BUDGETS_PER_CAMERA


# ---------------------------------------------------------------------------
# Fixed validation masks (paired checkpoint selection)
# ---------------------------------------------------------------------------


def test_fixed_budget_masks_are_identical_across_calls():
    """Selection is a paired comparison: the same masks every time, or it is not."""
    a = fixed_budget_masks(16, 4, PATCHES_PER_CAMERA, CAMERAS, 99, torch.device("cpu"))
    b = fixed_budget_masks(16, 4, PATCHES_PER_CAMERA, CAMERAS, 99, torch.device("cpu"))
    assert torch.equal(a, b)


def test_fixed_budget_masks_respect_the_requested_budget():
    cameras = camera_of_patch(PATCHES_PER_CAMERA, CAMERAS)
    mask = fixed_budget_masks(16, 8, PATCHES_PER_CAMERA, CAMERAS, 3, torch.device("cpu"))
    for camera in range(CAMERAS):
        assert torch.equal(mask[:, cameras == camera].sum(dim=1),
                           torch.full((16,), 8.0))


def test_fixed_budget_masks_differ_across_budgets_and_draws():
    base = fixed_budget_masks(16, 2, PATCHES_PER_CAMERA, CAMERAS, 5, torch.device("cpu"))
    assert not torch.equal(base, fixed_budget_masks(16, 8, PATCHES_PER_CAMERA, CAMERAS, 5, torch.device("cpu")))
    assert not torch.equal(base, fixed_budget_masks(16, 2, PATCHES_PER_CAMERA, CAMERAS, 1005, torch.device("cpu")))


def test_validation_budgets_include_evaluated_and_endpoint():
    assert {2, 4, 8} <= set(VALIDATION_BUDGETS)
    assert PATCHES_PER_CAMERA in VALIDATION_BUDGETS


# ---------------------------------------------------------------------------
# The sampler must be usable by the deployed-selector mask (train/eval identity)
# ---------------------------------------------------------------------------


def test_sampled_mask_composes_with_apply_patch_mask():
    """The training mask must go through the same op the benchmark selects with."""
    visual = torch.randn(3, 4, PATCHES_PER_CAMERA * CAMERAS, 8)
    mask = sample_budget_masks(3, generator=torch.Generator().manual_seed(12))
    masked = apply_patch_mask(visual, mask)
    assert masked.shape == visual.shape
    for window in range(3):
        for patch in range(PATCHES_PER_CAMERA * CAMERAS):
            if mask[window, patch] == 0:
                assert torch.equal(masked[window, :, patch], torch.zeros(4, 8))
            else:
                assert torch.equal(masked[window, :, patch], visual[window, :, patch])


def test_masking_preserves_patch_positions_so_spatial_pos_stays_aligned():
    """Masking must not renumber the patch axis; that is what keeps the grid meaningful."""
    visual = torch.randn(2, 3, PATCHES_PER_CAMERA * CAMERAS, 5)
    mask = sample_budget_masks(2, k_choices=(2,), generator=torch.Generator().manual_seed(13))
    masked = apply_patch_mask(visual, mask)
    assert masked.shape[2] == visual.shape[2]


def test_apply_patch_mask_rejects_shape_mismatch():
    visual = torch.randn(2, 3, PATCHES_PER_CAMERA * CAMERAS, 5)
    with pytest.raises(ValueError, match="does not match"):
        apply_patch_mask(visual, torch.ones(2, PATCHES_PER_CAMERA))

# ---------------------------------------------------------------------------
# Auxiliary per-frame passthrough (needed for the early_cls_attention comparator)
# ---------------------------------------------------------------------------


def _toy_cache(tmp_path, with_cls_attention=True):
    """A two-episode cache in the layout build_spatial_patch_cache.py writes."""
    records = []
    for index in range(2):
        length = 24
        arrays = {
            "states": np.arange(length * 14, dtype=np.float32).reshape(length, 14),
            "actions": np.arange(length * 7, dtype=np.float32).reshape(length, 7),
            "exterior_patches": np.ones((length, 16, 8), dtype=np.float32),
            "wrist_patches": np.ones((length, 16, 8), dtype=np.float32),
            "exterior_embeddings": np.ones((length, 8), dtype=np.float32),
            "wrist_embeddings": np.ones((length, 8), dtype=np.float32),
        }
        if with_cls_attention:
            arrays["cls_attention"] = np.arange(length * 32, dtype=np.float32).reshape(length, 32)
        path = tmp_path / f"ep{index}.npz"
        np.savez(path, **arrays)
        records.append({"episode_id": f"ep{index}", "cached_path": str(path), "length": length})
    return records


def _dataset(tmp_path, with_cls_attention=True, auxiliary_keys=("cls_attention",)):
    from adjointrwm.data import WindowDataset

    records = _toy_cache(tmp_path, with_cls_attention)
    normaliser = {
        "state_mean": np.zeros(14, np.float32), "state_std": np.ones(14, np.float32),
        "action_mean": np.zeros(7, np.float32), "action_std": np.ones(7, np.float32),
    }
    return WindowDataset(
        records, spec=WindowSpec(context_len=8, horizon=4, stride=2), normaliser=normaliser,
        input_visual_keys=("exterior_patches", "wrist_patches"),
        target_visual_keys=("exterior_embeddings", "wrist_embeddings"),
        visual_layout="tokens", auxiliary_keys=auxiliary_keys,
    )


def test_window_carries_auxiliary_array_with_context_length(tmp_path):
    dataset = _dataset(tmp_path)
    window = dataset[0]
    assert window["cls_attention"].shape == (8, 32)


def test_auxiliary_is_sliced_with_the_same_start_as_context_visual(tmp_path):
    """A misaligned auxiliary array would score the wrong frames and silently corrupt a comparator."""
    dataset = _dataset(tmp_path)
    for item in (0, 1, 2):
        window = dataset[item]
        start = int(window["window_start"])
        expected = np.arange(24 * 32, dtype=np.float32).reshape(24, 32)[start:start + 8]
        assert np.allclose(window["cls_attention"], expected)
        assert window["context_visual"].shape[0] == window["cls_attention"].shape[0]


def test_auxiliary_defaults_to_absent_when_not_requested(tmp_path):
    dataset = _dataset(tmp_path, auxiliary_keys=())
    assert "cls_attention" not in dataset[0]


def test_missing_auxiliary_key_fails_loudly(tmp_path):
    """Silently dropping the array would make early_cls_attention a duplicate of another comparator."""
    dataset = _dataset(tmp_path, with_cls_attention=False)
    with pytest.raises(KeyError, match="cls_attention"):
        _ = dataset[0]


def test_auxiliary_frame_count_mismatch_is_rejected(tmp_path):

    states = np.zeros((20, 3), np.float32)
    actions = np.zeros((20, 2), np.float32)
    with pytest.raises(ValueError, match="frames"):
        slice_window(states, actions, 0, WindowSpec(context_len=4, horizon=2),
                     auxiliary={"cls_attention": np.zeros((19, 5), np.float32)})


# ---------------------------------------------------------------------------
# Stratified evaluation sampling (audit finding B)
# ---------------------------------------------------------------------------


def test_stratified_sampling_spreads_over_episodes_not_a_prefix():
    """A prefix of an ordered loader is not a sample of episodes.

    Session 6A's first real run drew 384 windows from 3 episodes out of 50, which left the
    site-clustered bootstrap with 3 clusters. This pins the opposite property.
    """
    from adjointrwm.data.windows import stratified_window_indices

    episodes = [f"ep{i:03d}" for i in range(50)]
    sites = {f"ep{i:03d}": f"site{i % 14:02d}" for i in range(50)}
    ids = np.array([e for e in episodes for _ in range(40)])
    chosen = stratified_window_indices(ids, sites, total=384, seed=0)
    covered = {str(e) for e in ids[chosen]}
    assert len(chosen) == 384
    assert len(covered) == 50, f"only {len(covered)} episodes reached"


def test_stratified_sampling_covers_many_sites():
    from adjointrwm.data.windows import stratified_window_indices

    episodes = [f"ep{i:03d}" for i in range(50)]
    sites = {f"ep{i:03d}": f"site{i % 14:02d}" for i in range(50)}
    ids = np.array([e for e in episodes for _ in range(40)])
    chosen = stratified_window_indices(ids, sites, total=200, seed=0)
    covered = {sites[str(e)] for e in ids[chosen]}
    assert len(covered) == 14, f"only {len(covered)} sites reached"


def test_stratified_sampling_returns_sorted_unique_indices():
    from adjointrwm.data.windows import stratified_window_indices

    ids = np.array(["a"] * 10 + ["b"] * 10)
    sites = {"a": "x", "b": "y"}
    chosen = stratified_window_indices(ids, sites, total=7, seed=1)
    assert list(chosen) == sorted(set(chosen.tolist()))
    assert chosen.min() >= 0 and chosen.max() < len(ids)


def test_stratified_sampling_is_deterministic_for_a_seed():
    from adjointrwm.data.windows import stratified_window_indices

    rng = np.random.default_rng(0)
    ids = np.array([f"ep{i}" for i in range(20) for _ in range(5)])
    sites = {f"ep{i}": f"s{i % 4}" for i in range(20)}
    a = stratified_window_indices(ids, sites, 30, seed=7)
    b = stratified_window_indices(ids, sites, 30, seed=7)
    assert np.array_equal(a, b)


def test_stratified_sampling_never_returns_more_than_available():
    from adjointrwm.data.windows import stratified_window_indices

    ids = np.array(["a"] * 3 + ["b"] * 2)
    sites = {"a": "x", "b": "y"}
    chosen = stratified_window_indices(ids, sites, total=99, seed=0)
    assert len(chosen) == 5


def test_stratified_sampling_handles_zero_and_empty():
    from adjointrwm.data.windows import stratified_window_indices

    assert len(stratified_window_indices(np.array(["a"]), {"a": "x"}, 0)) == 0
    with pytest.raises(ValueError, match="empty"):
        stratified_window_indices(np.array([], dtype=object), {}, 5)


def test_stratified_sampling_distributes_roughly_evenly_across_episodes():
    from adjointrwm.data.windows import stratified_window_indices

    episodes = [f"ep{i}" for i in range(10)]
    sites = {e: "s" for e in episodes}
    ids = np.array([e for e in episodes for _ in range(50)])
    chosen = stratified_window_indices(ids, sites, total=100, seed=0)
    counts = {}
    for e in ids[chosen]:
        counts[str(e)] = counts.get(str(e), 0) + 1
    assert min(counts.values()) == max(counts.values()) == 10, counts


# ---------------------------------------------------------------------------
# Bottleneck diagnostic: rank correlation and capacity knob
# ---------------------------------------------------------------------------


def _spearman():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diag", REPO_DIR / "scripts/diagnose_spatial_selection_bottleneck.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spearman_is_one_for_a_monotone_relation():
    module = _spearman()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.allclose(module.spearman(x, x), torch.tensor([1.0], dtype=torch.float64))


def test_spearman_is_minus_one_for_a_reversed_relation():
    module = _spearman()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.allclose(module.spearman(x, -x), torch.tensor([-1.0], dtype=torch.float64))


def test_spearman_is_invariant_to_affine_rescaling():
    """Ranking only: a positive scale or shift must not change rho."""
    module = _spearman()
    x = torch.tensor([[1.0, 5.0, 2.0, 9.0]])
    y = torch.tensor([[3.0, 1.0, 7.0, 2.0]])
    a = module.spearman(x, y)
    b = module.spearman(x * 17.0 + 4.0, y * 0.3 - 2.0)
    assert torch.allclose(a, b, atol=1e-9)


def test_spearman_handles_ties_by_averaging_ranks():
    module = _spearman()
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    y = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # A constant signal has no rank information; it must be NaN, not a spurious number.
    assert torch.isnan(module.spearman(x, y)).all()


def test_spearman_computes_per_row():
    module = _spearman()
    # Row 0: ranks (0,1,2) against (1,0,2) -> rho = 0.5. Row 1: (2,0,1) against (0,2,1)
    # -> rho = -1. Hand-checked, so this pins the sign convention too.
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    y = torch.tensor([[2.0, 1.0, 3.0], [1.0, 3.0, 2.0]])
    out = module.spearman(x, y)
    assert out.shape == (2,)
    assert out[0].item() == pytest.approx(0.5)
    assert out[1].item() == pytest.approx(-1.0)


def test_spearman_rejects_shape_mismatch():
    module = _spearman()
    with pytest.raises(ValueError, match="shape mismatch"):
        module.spearman(torch.zeros(2, 4), torch.zeros(2, 5))


def test_curvature_estimator_hidden_knob_changes_capacity_only():
    from adjointrwm.allocators import CurvatureCostateEstimator

    default = CurvatureCostateEstimator(64)
    wide = CurvatureCostateEstimator(64, hidden=512)
    assert sum(p.numel() for p in default.parameters()) < sum(p.numel() for p in wide.parameters())
    # The heads themselves are unchanged, so capacity comparisons stay interpretable.
    assert default.costate_head.weight.shape == wide.costate_head.weight.shape
    assert default.hessian_head.weight.shape == wide.hessian_head.weight.shape


def test_curvature_estimator_default_is_unchanged():
    """The default must stay exactly 2*d so capacity matching elsewhere is unaffected."""
    from adjointrwm.allocators import MLP, CurvatureCostateEstimator

    d = 32
    expected = MLP(d + 2, 2 * d, d)
    actual = CurvatureCostateEstimator(d).objective_condition
    assert sum(p.numel() for p in expected.parameters()) == sum(p.numel() for p in actual.parameters())


def test_topk_overlap_is_one_for_the_truth():
    module = _spearman()
    truth = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.4, 0.3]])
    assert module.topk_overlap(truth.clone(), truth, 2).item() == pytest.approx(1.0)


def test_topk_overlap_is_zero_for_the_worst_possible_ranking():
    module = _spearman()
    truth = torch.tensor([[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    worst = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    assert module.topk_overlap(worst, truth, 2).item() == pytest.approx(0.0)


def test_topk_overlap_respects_the_per_camera_partition():
    """With P_cam=16 and k_cam=2 the chance level is 2/16 = 0.125, not 4/32 = 0.125 by accident.

    The two coincide numerically here, so this test pins the *value* under the symmetric
    per-camera budget rather than the arithmetic difference.
    """
    module = _spearman()
    torch.manual_seed(0)
    truth = torch.randn(8, 32)
    noise = torch.randn(8, 32)
    observed = module.topk_overlap(noise, truth, 2).mean().item()
    assert 0.03 < observed < 0.35, observed


def test_topk_overlap_is_bounded():
    module = _spearman()
    torch.manual_seed(1)
    truth = torch.randn(16, 32)
    out = module.topk_overlap(torch.randn(16, 32), truth, 4)
    assert out.shape == (16,)
    assert bool(((out >= 0) & (out <= 1)).all())
