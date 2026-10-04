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
