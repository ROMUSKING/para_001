"""Fixed-budget spatial patch selection (Session 6A).

Implements the policy suite and evaluation contract of
``docs/plans/2026-10-03-session-6a-spatial-selection-spec.md``: at a fixed patch budget
``k`` out of ``P`` spatial tokens, does belief-space sensitivity select more useful
visual regions than a competent direct gain predictor, curvature, and inexpensive
token-selection heuristics?

Design decisions adopted from the peer-critic review (spec section 2):

* **B1 (rank-one attention perturbation).** ``SpatialPatchAdapter`` pools ``P`` patch
  tokens through a *single* learned query, so ``pooled = out_proj(sum_h alpha_h v_h)``.
  Duplicating patch ``p`` in the key/value set therefore has the exact closed form
  :func:`rank_one_perturbation`,

      d(pooled)_p = out_proj( sum_h  alpha_{h,p} / (1 + alpha_{h,p}) * (v_{h,p} - ctx_h) ),

  a rank-one direction ``(v_p - pooled)`` scaled by the attention weight. This scores all
  ``P`` patches in ``O(1)`` forward passes with no ``P`` individual rollouts, so it stays
  inside the information boundary of a deployed selector. The spec's simplified form
  ``alpha_p (v_p - pooled)`` is the head-averaged, first-order-in-``alpha`` form; it is
  available via ``exact_heads=False`` and the tests pin both against explicit numerical
  differences.

* **B1 (sign convention).** The repository convention is
  ``Gain = -lambda^T dz - 0.5 dz^T H dz``. Every scorer here returns *gains* (larger is
  better) and never ``+lambda^T dz``.

* **B2 (grounded variance reduction).** The belief-space term is a variance reduction read
  from the model's own predictive variance head,
  ``delta_Sigma_p = diag(sigma^2(z_S)) - diag(sigma^2(z_{S u p}))``, not the
  ``delta_cov = effects.pow(2)`` proxy of Sessions 4/5, which collapsed VOI and curvature
  into the same functional family up to a sign flip. It is a *predictive*-variance
  reduction: a single trained ``state_logvar_head`` does not separate epistemic from
  aleatoric uncertainty, and the code and reports do not claim that split.

* **B3 (privilege).** Deployable selectors read only decision-time information: the
  distilled ``lambda_hat``/``H_hat`` from :class:`CurvatureCostateEstimator`, early visual
  features, and the model's own predictive variance. The autograd co-state and the
  rollout-search oracle are labelled diagnostic and never presented as deployable.

* **B4 (raw early features).** Heuristics score *raw*, pre-``LayerNorm`` patch tokens;
  scoring normalised embeddings would measure the LayerNorm, not the image.

* **B5 (symmetric budget).** The budget is partitioned symmetrically across the two
  cameras (``k_cam`` of 16 patches each) so spatial selection is not confounded with
  camera-modality selection; :func:`select_topk_per_camera` enforces it.

* **B6 (patch dropout).** :func:`apply_patch_mask` zeroes the *content* of dropped
  patches while keeping all ``P`` positions, so a mask is exactly the operation a
  deployed selector can perform, and no positional embedding is ever misaligned.

Conventions: ``visual`` is ``[B, T, P, D]``; ``mask`` is ``[B, P]`` in ``{0, 1}`` with
``1`` meaning retained; every score tensor is ``[B, P]`` *gains relative to the null
selection* (all patches dropped).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .allocators import CurvatureCostateEstimator, MLP

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

PATCHES_PER_CAMERA = 16
CAMERAS = 2
TOTAL_PATCHES = PATCHES_PER_CAMERA * CAMERAS
GRID_SIDE = 4  # 4x4 DINOv2 ViT-S/14 patch grid per camera

#: ``(total budget k, per-camera budget k_cam)`` pairs evaluated in Session 6A.
BUDGETS: tuple[tuple[int, int], ...] = ((4, 2), (8, 4), (16, 8))

#: Tier 0: inexpensive heuristics, ``O(P)``, no learned model.
HEURISTIC_POLICIES = ("uniform_grid", "stratified_random", "early_feature_norm", "early_cls_attention")
#: Tier 1: deployable learned allocators, ``O(P)``, no future targets.
DEPLOYABLE_POLICIES = (
    "direct_ranking_critic",
    "direct_critic_privileged",
    "second_order_curvature",
    "belief_space_voi",
)
#: The eight comparators of spec section 3.
COMPARATORS = HEURISTIC_POLICIES + DEPLOYABLE_POLICIES
#: Tier 2: non-deployable references (spec section 3, items 9-10).
DIAGNOSTIC_POLICIES = ("exact_costate_reference", "calibrated_greedy_oracle")

#: Belief-space weight grid, including the sign-flip controls required by B2.
BETAS: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)

#: Batch keys a deployable selector may read (``adjointrwm.models.INPUT_KEYS``).
DEPLOYABLE_INPUT_KEYS = ("context_state", "context_action", "context_visual", "future_actions")


def camera_of_patch(patches_per_camera: int = PATCHES_PER_CAMERA, cameras: int = CAMERAS) -> torch.Tensor:
    """``[P]`` long tensor mapping each patch index to its camera id."""
    return torch.arange(cameras).repeat_interleave(patches_per_camera)


#: Per-camera budgets a patch-dropout backbone may be trained at. The evaluated Session 6A
#: budgets are ``(2, 4, 8)`` per camera (spec section 3); ``16`` is the no-dropout endpoint
#: ``P_cam``. Keeping the endpoint in the training distribution is what protects the
#: full-patch case the Session 2 result rests on.
TRAIN_BUDGETS_PER_CAMERA: tuple[int, ...] = (2, 4, 8, 16)


def sample_budget_masks(
    windows: int,
    patches: int = TOTAL_PATCHES,
    patches_per_camera: int = PATCHES_PER_CAMERA,
    k_choices: Sequence[int] = TRAIN_BUDGETS_PER_CAMERA,
    cameras: int = CAMERAS,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Draw patch-dropout masks ``[B, P]`` with a symmetric per-camera budget (spec B6).

    Each window independently draws ``k_cam ~ Uniform(k_choices)`` *per camera*, so the kept
    budget is symmetric within a camera and never empties one -- the same partition
    Session 6A evaluates, which is what keeps spatial selection separated from camera
    modality selection (spec B5).

    The mask is binary over patch positions and is meant to be consumed by
    :func:`apply_patch_mask`, which zeroes dropped content and keeps every position, so a
    model trained through this sampler is evaluated through exactly the same operation.
    """
    if windows <= 0:
        raise ValueError(f"windows must be positive, got {windows}")
    choices = [int(k) for k in k_choices]
    if not choices:
        raise ValueError("k_choices must not be empty")
    if any(k <= 0 or k > patches_per_camera for k in choices):
        raise ValueError(
            f"every budget must lie in [1, {patches_per_camera}], got {choices}"
        )
    if patches != patches_per_camera * cameras:
        raise ValueError(
            f"patches {patches} is not {patches_per_camera} x {cameras}"
        )

    mask = torch.zeros(windows, patches)
    for window in range(windows):
        for camera in range(cameras):
            k = int(choices[int(torch.randint(len(choices), (1,), generator=generator))])
            start = camera * patches_per_camera
            # randperm over the camera's own patch range: k distinct positions, no duplicates.
            order = torch.randperm(patches_per_camera, generator=generator)[:k]
            mask[window, start + order] = 1.0
    if device is not None:
        mask = mask.to(device)
    return mask


def grid_coordinates(patches_per_camera: int = PATCHES_PER_CAMERA, grid_side: int = GRID_SIDE) -> torch.Tensor:
    """``[P, 2]`` ``(row, col)`` coordinates of each patch inside its camera's grid."""
    if patches_per_camera != grid_side * grid_side:
        raise ValueError(f"{patches_per_camera} patches do not fill a {grid_side}x{grid_side} grid")
    rows = torch.arange(grid_side).repeat_interleave(grid_side)
    cols = torch.arange(grid_side).repeat(grid_side)
    return torch.stack([rows, cols], dim=1)


# ---------------------------------------------------------------------------
# Attention terms and the rank-one perturbation identity (B1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttentionTerms:
    """Per-head attention terms of the single-query pooling in :class:`SpatialPatchAdapter`.

    Shapes with ``N = B * T``: ``alpha`` ``[N, H, P]``, ``values`` ``[N, P, H, dh]``,
    ``context`` ``[N, H, dh]``, ``pooled`` ``[N, E]``.
    """

    alpha: torch.Tensor
    values: torch.Tensor
    context: torch.Tensor
    pooled: torch.Tensor
    out_weight: torch.Tensor

    @property
    def num_heads(self) -> int:
        return int(self.alpha.shape[1])

    @property
    def head_dim(self) -> int:
        return int(self.values.shape[-1])


def attention_terms(adapter: nn.Module, visual: torch.Tensor) -> AttentionTerms:
    """Recompute the adapter's attention weights and value vectors exactly.

    ``SpatialPatchAdapter`` wraps ``nn.MultiheadAttention`` and discards the weights, so
    they are recomputed here from that module's own parameters: one source of truth, and
    ``average_attn_weights=False`` semantics (per-head weights) as B1 requires.
    """
    patches = visual.shape[2]
    projected = adapter.patch_proj(visual) + adapter.spatial_pos[:, :, :patches, :]
    batch, frames, _, width = projected.shape
    flat = projected.reshape(batch * frames, patches, width)
    attn = adapter.pool_attn
    if not getattr(attn, "_qkv_same_embed_dim", True):
        raise ValueError("attention_terms expects a self-attention block (kdim == vdim == embed_dim)")
    if not hasattr(attn, "in_proj_weight"):
        raise ValueError("attention_terms expects a batch_first nn.MultiheadAttention with a fused in-projection")

    query = adapter.pool_query.reshape(1, 1, width).expand(batch * frames, 1, width)
    embed_dim, num_heads = attn.embed_dim, attn.num_heads
    head_dim = embed_dim // num_heads
    w_q, w_k, w_v = attn.in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = attn.in_proj_bias.chunk(3, dim=0)

    q = F.linear(query, w_q, b_q).reshape(batch * frames, num_heads, head_dim)
    k = F.linear(flat, w_k, b_k).reshape(batch * frames, patches, num_heads, head_dim)
    v = F.linear(flat, w_v, b_v).reshape(batch * frames, patches, num_heads, head_dim)

    alpha = torch.softmax(torch.einsum("nhd,nphd->nhp", q, k) / math.sqrt(head_dim), dim=-1)
    context = torch.einsum("nhp,nphd->nhd", alpha, v)
    pooled = attn.out_proj(context.reshape(batch * frames, 1, embed_dim)).reshape(batch * frames, embed_dim)
    return AttentionTerms(alpha=alpha, values=v, context=context, pooled=pooled, out_weight=attn.out_proj.weight)


def rank_one_perturbation(
    terms: AttentionTerms,
    exact_heads: bool = True,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Exact change of the pooled vector from duplicating each patch in the key/value set.

    Appending a second copy of patch ``p`` renormalises the softmax to total mass
    ``1 + alpha_{h,p}``, so the head context moves by
    ``alpha_{h,p} / (1 + alpha_{h,p}) * (v_{h,p} - ctx_h)`` and the pooled output follows
    through the linear output projection. Because this is a *difference*, the projection
    bias drops out.

    Args:
        terms: per-head attention terms from :func:`attention_terms`.
        exact_heads: use the exact per-head coefficient ``alpha_{h,p} / (1 + alpha_{h,p})``.
            With ``False``, use the spec section 2 shorthand ``alpha_p (v_p - pooled)``,
            which head-averages ``alpha`` and drops the renormalisation factor.
        batch_size: leading dimension ``B`` to split the flattened ``B * T`` back apart.
            ``None`` keeps the flat ``[B * T, P, d_model]`` layout.

    Returns:
        ``[B, T, P, d_model]`` (or ``[B * T, P, d_model]``) perturbations of the
        pre-``out_norm`` pooled vector.

    Note:
        The spec's shorthand ``alpha_p (v_p - pooled)`` makes two simplifications, and the
        tests pin both magnitudes:

        * dropping ``1 / (1 + alpha)`` is first order in ``alpha`` -- about 12% at
          ``P = 8`` but only about 3% at the spec's ``P = 16`` per camera;
        * head-averaging ``alpha`` is likewise first order (about 3% at ``P = 16``), and
          only becomes large when ``alpha`` itself is large, i.e. at small patch counts
          (about 90% at ``P = 4``).

        Both are therefore faithful approximations *at the spec's operating point*, which
        is why the exact per-head form is still the default: it costs nothing extra and it
        is exactly right at every ``P``.
    """
    frames, patches, num_heads, _ = terms.values.shape
    embed_dim = terms.pooled.shape[-1]
    residual = terms.values - terms.context.unsqueeze(1)                     # [N, P, H, dh]
    if exact_heads:
        coefficient = (terms.alpha / (1.0 + terms.alpha)).permute(0, 2, 1).unsqueeze(-1)
    else:
        coefficient = terms.alpha.mean(dim=1).view(frames, 1, patches, 1).permute(0, 2, 1, 3)
    delta = F.linear((coefficient * residual).reshape(frames, patches, embed_dim), terms.out_weight)
    if batch_size is None:
        return delta
    return delta.reshape(batch_size, frames // batch_size, patches, embed_dim)


def explicit_duplicate_pooling(adapter: nn.Module, visual: torch.Tensor, patch: int) -> torch.Tensor:
    """Numerical reference: pooled output when ``patch`` appears twice in the key/value set.

    This is the O(P)-expensive way to obtain what :func:`rank_one_perturbation` derives in
    closed form; the tests assert the two agree, which is the numerical proof of the
    identity.
    """
    patches = visual.shape[2]
    projected = adapter.patch_proj(visual) + adapter.spatial_pos[:, :, :patches, :]
    batch, frames, _, width = projected.shape
    flat = projected.reshape(batch * frames, patches, width)
    attn = adapter.pool_attn
    embed_dim, num_heads = attn.embed_dim, attn.num_heads
    head_dim = embed_dim // num_heads
    w_q, w_k, w_v = attn.in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = attn.in_proj_bias.chunk(3, dim=0)
    query = adapter.pool_query.reshape(1, 1, width).expand(batch * frames, 1, width)

    duplicated_k = torch.cat([flat, flat[:, patch : patch + 1]], dim=1)
    duplicated_v = torch.cat([flat, flat[:, patch : patch + 1]], dim=1)
    q = F.linear(query, w_q, b_q).reshape(batch * frames, num_heads, head_dim)
    k = F.linear(duplicated_k, w_k, b_k).reshape(batch * frames, patches + 1, num_heads, head_dim)
    v = F.linear(duplicated_v, w_v, b_v).reshape(batch * frames, patches + 1, num_heads, head_dim)
    alpha = torch.softmax(torch.einsum("nhd,nphd->nhp", q, k) / math.sqrt(head_dim), dim=-1)
    context = torch.einsum("nhp,nphd->nhd", alpha, v)
    return attn.out_proj(context.reshape(batch * frames, 1, embed_dim)).reshape(batch, frames, embed_dim)


def rank_one_first_order(terms: AttentionTerms, batch_size: int | None = None) -> torch.Tensor:
    """The exact per-head direction with only the ``1/(1+alpha)`` factor dropped.

    Isolates the two simplifications in the spec's shorthand: this one is genuinely
    first-order in ``alpha`` (~6% at ``P = 16``), whereas head-averaging ``alpha`` is not
    small. Kept so the two errors can be attributed separately rather than conflated.
    """
    frames, patches, _, _ = terms.values.shape
    embed_dim = terms.pooled.shape[-1]
    residual = terms.values - terms.context.unsqueeze(1)
    coefficient = terms.alpha.permute(0, 2, 1).unsqueeze(-1)
    delta = F.linear((coefficient * residual).reshape(frames, patches, embed_dim), terms.out_weight)
    return delta if batch_size is None else delta.reshape(batch_size, frames // batch_size, patches, embed_dim)


def patch_identity_for_tests(model: nn.Module, batch: Mapping[str, torch.Tensor], patch: int) -> torch.Tensor:
    """Ground truth for :func:`rank_one_perturbation`: pooled change from duplicating ``patch``.

    This is the O(P) numerical difference that the closed form replaces, exposed so the
    identity can be *tested* against an independent computation rather than against itself.
    It is a test/reference helper, not part of any deployable selector.
    """
    visual = batch["context_visual"]
    duplicated = torch.cat([visual, visual[:, :, patch : patch + 1]], dim=2)
    positions = torch.cat(
        [torch.arange(visual.shape[2], device=visual.device), torch.tensor([patch], device=visual.device)]
    )
    with torch.no_grad():
        base = _pooled_pre_ln(model, visual, torch.arange(visual.shape[2], device=visual.device))
        perturbed = _pooled_pre_ln(model, duplicated, positions)
    return perturbed - base


def _pooled_pre_ln(model: nn.Module, visual: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """``[B, T, d]`` pooled vector *before* the adapter's ``out_norm``.

    ``positions`` selects which ``spatial_pos`` entry each token gets, so a duplicated
    patch keeps its own grid position instead of borrowing the next one.
    """
    adapter = model.visual_adapter
    patches = visual.shape[2]
    projected = adapter.patch_proj(visual) + adapter.spatial_pos[:, :, positions, :]
    batch, frames, _, width = projected.shape
    flat = projected.reshape(batch * frames, patches, width)
    attn = adapter.pool_attn
    embed_dim, num_heads = attn.embed_dim, attn.num_heads
    head_dim = embed_dim // num_heads
    w_q, w_k, w_v = attn.in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = attn.in_proj_bias.chunk(3, dim=0)
    query = adapter.pool_query.reshape(1, 1, width).expand(batch * frames, 1, width)
    q = F.linear(query, w_q, b_q).reshape(batch * frames, num_heads, head_dim)
    k = F.linear(flat, w_k, b_k).reshape(batch * frames, patches, num_heads, head_dim)
    v = F.linear(flat, w_v, b_v).reshape(batch * frames, patches, num_heads, head_dim)
    alpha = torch.softmax(torch.einsum("nhd,nphd->nhp", q, k) / math.sqrt(head_dim), dim=-1)
    context = torch.einsum("nhp,nphd->nhd", alpha, v)
    return attn.out_proj(context.reshape(batch * frames, 1, embed_dim)).reshape(batch, frames, embed_dim)


def layernorm_jvp(x: torch.Tensor, delta: torch.Tensor, eps: float) -> torch.Tensor:
    """Directional derivative of ``LayerNorm(x)`` along ``delta`` (identity affine params).

    ``d xhat = (delta - mean(delta) - xhat * mean(xhat * delta)) / sqrt(var + eps)``.
    Used to push the rank-one attention perturbation through the adapter's ``out_norm``
    without a second network evaluation.
    """
    centred = x - x.mean(dim=-1, keepdim=True)
    inverse = torch.rsqrt(centred.pow(2).mean(dim=-1, keepdim=True) + eps)
    normalised = centred * inverse
    mean_delta = delta.mean(dim=-1, keepdim=True)
    projection = (normalised * delta).mean(dim=-1, keepdim=True)
    return inverse * (delta - mean_delta - normalised * projection)


# ---------------------------------------------------------------------------
# Masking and the evaluation objective
# ---------------------------------------------------------------------------


def apply_patch_mask(visual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero the content of dropped patches, keeping all ``P`` positions.

    ``visual`` is ``[B, T, P, D]`` and ``mask`` is ``[B, P]``. Positions are preserved so
    that ``spatial_pos`` stays aligned with the true grid location, which is what makes the
    operation a mask a deployed selector can actually perform (B6).
    """
    if visual.dim() != 4:
        raise ValueError(f"expected visual [B, T, P, D], got {tuple(visual.shape)}")
    if mask.dim() != 2:
        raise ValueError(f"expected mask [B, P], got {tuple(mask.shape)}")
    if mask.shape != (visual.shape[0], visual.shape[2]):
        raise ValueError(
            f"mask {tuple(mask.shape)} does not match the patch axes of visual {tuple(visual.shape)}"
        )
    return visual * mask.to(visual.dtype).view(-1, 1, mask.shape[1], 1)


def select_topk_per_camera(
    scores: torch.Tensor,
    k_per_camera: int,
    cameras: torch.Tensor | None = None,
    num_cameras: int = CAMERAS,
) -> torch.Tensor:
    """Top-``k_per_camera`` patches *within each camera* -> binary mask ``[B, P]``.

    Enforcing the symmetric per-camera budget is what keeps spatial selection from being
    confounded with camera-modality selection (B5).
    """
    if scores.dim() != 2:
        raise ValueError(f"scores must be [B, P], got {tuple(scores.shape)}")
    patches = scores.shape[1]
    camera_index = camera_of_patch(patches // num_cameras, num_cameras) if cameras is None else cameras.to(scores.device)
    if camera_index.numel() != patches:
        raise ValueError("camera_of_patch must cover every patch")
    mask = torch.zeros_like(scores)
    rows = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
    for camera in range(num_cameras):
        columns = torch.nonzero(camera_index == camera, as_tuple=False).flatten()
        if k_per_camera > columns.numel():
            raise ValueError(f"k_per_camera={k_per_camera} exceeds the {columns.numel()} patches of camera {camera}")
        chosen = columns[scores[:, columns].topk(k_per_camera, dim=1).indices]  # [B, k]
        mask[rows, chosen] = 1.0
    return mask


def encode_masked(model: nn.Module, batch: Mapping[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    """Latent ``z_S`` for a binary patch mask ``[B, P]``."""
    visual = batch["context_visual"]
    if visual.dim() == 3:  # flat [B, T, P*D]: reshape so the mask lands on the patch axis
        windows, frames, flat = visual.shape
        patches = mask.shape[1]
        if flat % patches:
            raise ValueError(f"flat visual width {flat} is not divisible by {patches} patches")
        visual = visual.reshape(windows, frames, patches, flat // patches)
    return model.encode_context(apply_patch_mask(visual, mask), batch["context_state"], batch["context_action"])


def objective_at_masks(model: nn.Module, batch: Mapping[str, torch.Tensor], masks: torch.Tensor) -> torch.Tensor:
    """Downstream objective ``J(z_S)`` for every mask in ``masks`` ``[B, M, P]``.

    Returns ``[B, M]``. All ``B * M`` masked contexts are encoded in one batched forward,
    so evaluating a policy costs one rollout-equivalent regardless of the budget.
    """
    from .models.common import prediction_objective

    if masks.dim() != 3:
        raise ValueError(f"masks must be [B, M, P], got {tuple(masks.shape)}")
    windows, candidates, _ = masks.shape
    flat = masks.reshape(windows * candidates, -1)
    visual = batch["context_visual"].unsqueeze(1).expand(-1, candidates, *batch["context_visual"].shape[1:])
    state = batch["context_state"].unsqueeze(1).expand(-1, candidates, *batch["context_state"].shape[1:])
    action = batch["context_action"].unsqueeze(1).expand(-1, candidates, *batch["context_action"].shape[1:])
    latent = model.encode_context(
        apply_patch_mask(visual.reshape(windows * candidates, *visual.shape[2:]), flat),
        state.reshape(windows * candidates, *state.shape[2:]),
        action.reshape(windows * candidates, *action.shape[2:]),
    )
    actions = batch["future_actions"].unsqueeze(1).expand(-1, candidates, *batch["future_actions"].shape[1:])
    prediction = model.rollout(
        latent, actions.reshape(windows * candidates, *batch["future_actions"].shape[1:])
    )
    target_state = batch["target_state"].unsqueeze(1).expand(-1, candidates, *batch["target_state"].shape[1:])
    target_visual = batch["target_visual"].unsqueeze(1).expand(-1, candidates, *batch["target_visual"].shape[1:])
    objective = prediction_objective(
        prediction,
        target_state.reshape(windows * candidates, *batch["target_state"].shape[1:]),
        target_visual.reshape(windows * candidates, *batch["target_visual"].shape[1:]),
    )
    return objective.reshape(windows, candidates)


# ---------------------------------------------------------------------------
# Co-state and grounded-epistemic machinery
# ---------------------------------------------------------------------------


@dataclass
class CotangentBundle:
    """Everything one backward pass yields about patch utility.

    Attributes:
        terms: per-head attention terms at the full-patch operating point.
        pooled_cotangent: ``[B, T, d]`` ``dJ/d(pooled_pre_ln)``, i.e. the co-state pushed
            back through the temporal encoder, the rollout and ``out_norm``.
        first_order_gains: ``[B, P]`` cotangent sensitivity ``-<g, d(pooled)_p>``, the
            diagnostic ceiling that needs true future targets.
        delta_pooled: ``[B, P, d]`` frame-averaged perturbation in the adapter's **pooled**
            basis. This is *not* a latent-space perturbation and must not be contracted with
            a latent co-state; it is retained only to validate the closed form against
            :func:`patch_identity_for_tests` and to expose the O(1) cotangent path. The
            deployable scorers use :func:`latent_patch_perturbations` instead.
        latent: ``[B, d]`` the encoded latent of the unmasked context.
        exact_costate: ``[B, d]`` ``dJ/dz``, the diagnostic autograd co-state.
    """

    terms: AttentionTerms
    pooled_cotangent: torch.Tensor
    first_order_gains: torch.Tensor
    delta_pooled: torch.Tensor
    latent: torch.Tensor
    exact_costate: torch.Tensor


def spatial_encode_with_pooled(
    model: nn.Module,
    context_visual: torch.Tensor,
    context_state: torch.Tensor,
    context_action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(z, pooled_pre_ln)`` for a spatial arm, mirroring ``encode_context``.

    Re-derives the adapter pooling so that ``pooled`` stays a node in the autograd graph;
    the returned latent is numerically identical to ``model.encode_context(...)``, which
    the test suite asserts.
    """
    adapter = model.visual_adapter
    patches = context_visual.shape[2]
    projected = adapter.patch_proj(context_visual) + adapter.spatial_pos[:, :, :patches, :]
    batch, frames, _, width = projected.shape
    flat = projected.reshape(batch * frames, patches, width)
    attn = adapter.pool_attn
    embed_dim, num_heads = attn.embed_dim, attn.num_heads
    head_dim = embed_dim // num_heads
    w_q, w_k, w_v = attn.in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = attn.in_proj_bias.chunk(3, dim=0)
    query = adapter.pool_query.reshape(1, 1, width).expand(batch * frames, 1, width)
    q = F.linear(query, w_q, b_q).reshape(batch * frames, num_heads, head_dim)
    k = F.linear(flat, w_k, b_k).reshape(batch * frames, patches, num_heads, head_dim)
    v = F.linear(flat, w_v, b_v).reshape(batch * frames, patches, num_heads, head_dim)
    alpha = torch.softmax(torch.einsum("nhd,nphd->nhp", q, k) / math.sqrt(head_dim), dim=-1)
    context = torch.einsum("nhp,nphd->nhd", alpha, v)
    pooled = attn.out_proj(context.reshape(batch * frames, 1, embed_dim)).reshape(batch, frames, embed_dim)

    visual = adapter.out_norm(pooled)
    tokens = visual + model.state_adapter(context_state) + model.context_action_adapter(context_action)
    tokens = tokens + model.position[:, : tokens.shape[1]]
    latent = model.temporal_norm(model.temporal_encoder(tokens)[:, -1])
    return latent, pooled


def cotangent_bundle(model: nn.Module, batch: Mapping[str, torch.Tensor], exact_heads: bool = True) -> CotangentBundle:
    """One forward + one backward pass that scores all ``P`` patches (B1).

    The single backward yields ``dJ/d(pooled_pre_ln)``; contracting it with the closed-form
    rank-one attention perturbation gives every patch's first-order gain without ever
    running ``P`` rollouts. Also returns the exact autograd co-state ``dJ/dz``.

    The returned ``delta_pooled`` lives in the adapter's pooled basis and is **not** a
    latent-space perturbation; it exists to validate the closed form and to expose this O(1)
    path. Use :func:`latent_patch_perturbations` for anything that contracts with a latent
    co-state.
    """
    from .models.common import prediction_objective

    visual = batch["context_visual"].detach().requires_grad_(True)
    latent, pooled = spatial_encode_with_pooled(model, visual, batch["context_state"], batch["context_action"])
    prediction = model.rollout(latent, batch["future_actions"])
    objective = prediction_objective(prediction, batch["target_state"], batch["target_visual"])

    cotangent, exact_costate = torch.autograd.grad(objective.sum(), [pooled, latent], retain_graph=False)
    del visual

    windows = batch["context_state"].shape[0]
    with torch.no_grad():
        terms = attention_terms(model.visual_adapter, batch["context_visual"])
        perturbation = rank_one_perturbation(terms, exact_heads=exact_heads, batch_size=windows)
        gains = -(cotangent.unsqueeze(2) * perturbation).sum(dim=(1, 3))       # [B, P]
        delta_pooled = perturbation.mean(dim=1)                                # [B, P, d]
        model.zero_grad(set_to_none=True)
    return CotangentBundle(
        terms=terms,
        pooled_cotangent=cotangent.detach(),
        first_order_gains=gains.detach(),
        delta_pooled=delta_pooled.detach(),
        latent=latent.detach(),
        exact_costate=exact_costate.detach(),
    )


def curvature_scores(costate: torch.Tensor, hessian: torch.Tensor, delta_z: torch.Tensor) -> torch.Tensor:
    """``-lambda^T dz - 0.5 * dz^T H dz`` per patch (repo sign convention).

    ``costate``/``hessian``/``delta_z`` must all live in the **same** space: the repo's
    latent space of dimension ``d_model``. Contracting a latent-space co-state with a
    perturbation measured elsewhere (e.g. the adapter's pooled space) is dimensionally
    valid but scientifically wrong -- the two bases differ by the temporal encoder, so the
    ranking would be meaningless. :func:`latent_patch_perturbations` supplies the genuine
    latent-space quantity.
    """
    if costate.shape[-1] != delta_z.shape[-1] or hessian.shape[-1] != delta_z.shape[-1]:
        raise ValueError(
            "costate, hessian and delta_z must share a dimension; got "
            f"{costate.shape[-1]}, {hessian.shape[-1]}, {delta_z.shape[-1]}"
        )
    first = -(costate.unsqueeze(1) * delta_z).sum(-1)
    second = 0.5 * (hessian.unsqueeze(1) * delta_z.pow(2)).sum(-1)
    return first - second


def latent_patch_perturbations(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    base_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact latent-space perturbation ``dz_p = z(S u {p}) - z(S)`` -> ``[B, P, d]``.

    This is the quantity the repo's ``first_order_scores`` and
    ``second_order_curvature_scores`` contract with a co-state, so it is the only honest
    input to the curvature, VOI and privileged-critic scorers. It costs one *encode* per
    patch (no rollout), so it is far cheaper than the grounded variance term but is not
    O(1).

    It replaces an earlier proxy that frame-averaged the adapter's pre-``out_norm``
    perturbation. That proxy lives in the pooled basis, not the latent basis, and was
    measured to correlate only ~0.2 (mean cosine) with this ground truth -- too weak to
    rank patches. Keeping it would have silently changed what the benchmark measures.
    """
    patches = batch["context_visual"].shape[2]
    windows = batch["context_state"].shape[0]
    device = batch["context_state"].device
    base = torch.zeros(windows, patches, device=device) if base_mask is None else base_mask.to(device)

    candidates = base.unsqueeze(1).expand(-1, patches, -1).clone()
    candidates[:, torch.arange(patches), torch.arange(patches)] = 1.0

    with torch.no_grad():
        base_latent = encode_masked(model, batch, base)
        visual = batch["context_visual"].unsqueeze(1).expand(-1, patches, *batch["context_visual"].shape[1:])
        state = batch["context_state"].unsqueeze(1).expand(-1, patches, *batch["context_state"].shape[1:])
        action = batch["context_action"].unsqueeze(1).expand(-1, patches, *batch["context_action"].shape[1:])
        shape = (windows * patches, *batch["context_visual"].shape[1:])
        latent = model.encode_context(
            apply_patch_mask(visual.reshape(shape), candidates.reshape(-1, patches)),
            state.reshape(windows * patches, *state.shape[2:]),
            action.reshape(windows * patches, *action.shape[2:]),
        )
        # ``latent`` is [B * P, d]; the base latent must be tiled to match before the
        # subtraction, otherwise broadcasting silently produces a [B, B*P, d] tensor.
        tiled = base_latent.unsqueeze(1).expand(-1, patches, -1).reshape(windows * patches, -1)
    return (latent - tiled).reshape(windows, patches, -1)


def belief_space_voi_scores(
    costate: torch.Tensor,
    hessian: torch.Tensor,
    delta_z: torch.Tensor,
    delta_sigma: torch.Tensor,
    sigma_precision: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """Curvature gains plus ``(beta / 2) * sum_d precision_d * (delta_Sigma_{p,d})^2``.

    ``delta_sigma`` is the *grounded* predictive variance reduction
    ``diag(sigma^2(z_S)) - diag(sigma^2(z_{S u p}))`` read from the model's own
    ``state_logvar_head``, and ``sigma_precision = E_t exp(-logvar)`` is the matching
    precision weight, so the term has the units of a squared state displacement. ``beta = 0``
    reduces exactly to :func:`curvature_scores`, which the tests assert.

    Scope of the claim: this is a *predictive*-variance reduction, which is what the model
    actually parameterises. It is **not** separated into epistemic and aleatoric parts -- a
    single trained ``state_logvar_head`` cannot make that split. The term is therefore
    reported as predictive-variance reduction, and the docs and reports must not call it a
    pure epistemic VOI. It is still a different functional family from curvature (it uses
    the variance head rather than ``delta_z^2``), which is the point B2 was making.
    """
    curvature = curvature_scores(costate, hessian, delta_z)
    epistemic = 0.5 * beta * (sigma_precision.unsqueeze(1) * delta_sigma.pow(2)).sum(-1)
    return curvature + epistemic


def predictive_precision(prediction: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """``E_t exp(-state_logvar)`` -> ``[B, state_dim]``: the model's own predicted precision.

    This is the Gaussian NLL's curvature with respect to the predicted state mean, so it is
    the natural quadratic weight for a variance-space perturbation and needs no future target.
    """
    return (-prediction["state_logvar"]).exp().mean(dim=1)


def epistemic_variance_reduction(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    mask: torch.Tensor | None = None,
    chunk: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(delta_Sigma [B, P, S], sigma_precision [B, S])`` from the model's variance head.

    ``mask`` is the base selection ``S`` (``None`` = the null selection, all patches
    dropped). ``delta_Sigma_p = diag(sigma^2(z_S)) - diag(sigma^2(z_{S u p}))`` is computed
    by rolling out the latent for ``S`` and for ``S u {p}`` for every patch, so unlike the
    cotangent term this genuinely costs one rollout per patch; that cost is reported, not
    hidden, because it is exactly what distinguishes grounded VOI from curvature.
    """
    patches = batch["context_visual"].shape[2]
    windows = batch["context_state"].shape[0]
    device = batch["context_state"].device
    base_mask = torch.zeros(windows, patches, device=device) if mask is None else mask.to(device)
    if base_mask.shape != (windows, patches):
        raise ValueError(f"mask must be [{windows}, {patches}], got {tuple(base_mask.shape)}")

    with torch.no_grad():
        base_latent = encode_masked(model, batch, base_mask)
        base_prediction = model.rollout(base_latent, batch["future_actions"])
        base_variance = base_prediction["state_logvar"].exp().mean(dim=1)
        precision = predictive_precision(base_prediction)

        # delta_Sigma_p compares S against S u {p}; already-selected patches stay on.
        per_patch = base_mask.unsqueeze(1).expand(-1, patches, -1).clone()
        per_patch[:, torch.arange(patches), torch.arange(patches)] = 1.0

        deltas = []
        for start in range(0, patches, max(1, chunk)):
            block = per_patch[:, start : start + max(1, chunk)]
            variance = objective_free_variance(model, batch, block)
            deltas.append(base_variance.unsqueeze(1) - variance)
        delta_sigma = torch.cat(deltas, dim=1)
    return delta_sigma, precision


def objective_free_variance(model: nn.Module, batch: Mapping[str, torch.Tensor], masks: torch.Tensor) -> torch.Tensor:
    """``E_t sigma^2`` of the state prediction for each mask in ``masks`` ``[B, M, P]``."""
    windows, candidates, _ = masks.shape
    visual = batch["context_visual"].unsqueeze(1).expand(-1, candidates, *batch["context_visual"].shape[1:])
    state = batch["context_state"].unsqueeze(1).expand(-1, candidates, *batch["context_state"].shape[1:])
    action = batch["context_action"].unsqueeze(1).expand(-1, candidates, *batch["context_action"].shape[1:])
    flat_visual = visual.reshape(windows * candidates, *visual.shape[2:])
    latent = model.encode_context(
        apply_patch_mask(flat_visual, masks.reshape(windows * candidates, -1)),
        state.reshape(windows * candidates, *state.shape[2:]),
        action.reshape(windows * candidates, *action.shape[2:]),
    )
    prediction = model.rollout(
        latent,
        batch["future_actions"].unsqueeze(1).expand(-1, candidates, *batch["future_actions"].shape[1:]).reshape(
            windows * candidates, *batch["future_actions"].shape[1:]
        ),
    )
    return prediction["state_logvar"].exp().mean(dim=1).reshape(windows, candidates, -1)


# ---------------------------------------------------------------------------
# Tier 0: inexpensive heuristics
# ---------------------------------------------------------------------------


def grid_side_for(patches_per_camera: int) -> int:
    """Square grid side implied by a per-camera patch count."""
    side = int(round(math.sqrt(patches_per_camera)))
    if side * side != patches_per_camera:
        raise ValueError(f"{patches_per_camera} patches per camera do not fill a square grid")
    return side


def uniform_grid_scores(
    windows: int,
    patches_per_camera: int = PATCHES_PER_CAMERA,
    cameras: int = CAMERAS,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Deterministic scores that select maximally spread cells (corners first).

    Cells are ordered by descending distance from the grid centre, ties broken by
    ``(row, col)``, so the top-4 of a 4x4 grid is exactly its four corners (the spec's
    worked example) and the ordering is a fixed property of the grid, not of the data.
    """
    grid_side = grid_side_for(patches_per_camera)
    coordinates = grid_coordinates(patches_per_camera, grid_side).to(torch.float64)
    centre = (grid_side - 1) / 2.0
    distance = ((coordinates - centre) ** 2).sum(-1)
    order = torch.argsort(distance, descending=True, stable=True)
    scores = torch.empty(patches_per_camera, dtype=torch.float32)
    scores[order] = torch.arange(order.numel(), 0, -1, dtype=torch.float32)
    per_camera = scores.to(device).unsqueeze(0).expand(windows, -1)
    return per_camera.unsqueeze(1).expand(-1, cameras, -1).reshape(windows, -1).contiguous()


def stratified_random_scores(
    windows: int,
    k_per_camera: int,
    seed: int = 0,
    patches_per_camera: int = PATCHES_PER_CAMERA,
    cameras: int = CAMERAS,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Random selection stratified across the four quadrants of each camera's grid.

    ``k_cam`` picks are spread as evenly as possible over the quadrants (largest-remainder
    allocation, so e.g. ``k_cam = 6`` gives 2/2/1/1), and within a quadrant the chosen
    patches are a uniform random subset. Scoring *every* patch independently -- rather than
    giving a whole stratum one score -- matters: equal scores inside a stratum would make
    top-k break ties by patch index, which is deterministic and biased toward low indices
    rather than random.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    grid_side = grid_side_for(patches_per_camera)
    coordinates = grid_coordinates(patches_per_camera, grid_side)
    half = max(1, grid_side // 2)
    quadrant = coordinates[:, 0] // half * 2 + coordinates[:, 1] // half
    strata = sorted({int(q) for q in quadrant})

    # Largest-remainder allocation of k_cam picks over the strata.
    share, remainder = divmod(k_per_camera, len(strata))
    sizes = torch.full((len(strata),), share, dtype=torch.long)
    if remainder:
        fractional = torch.tensor([len(torch.nonzero(quadrant == s, as_tuple=False)) for s in strata],
                                  dtype=torch.float64)
        order = torch.argsort(fractional, descending=True, stable=True)[:remainder]
        sizes[order] += 1

    scores = torch.zeros(windows, cameras, patches_per_camera)
    for index, stratum in enumerate(strata):
        members = torch.nonzero(quadrant == stratum, as_tuple=False).flatten()
        needed = min(int(sizes[index]), members.numel())
        noise = torch.rand(windows, cameras, members.numel(), generator=generator)
        # Keep only the `needed` best of each stratum, with distinct values so top-k never
        # has to break a tie inside the stratum.
        keep = noise.topk(needed, dim=-1).indices
        selected = torch.zeros(windows, cameras, members.numel())
        selected.scatter_(-1, keep, 1.0)
        scores[:, :, members] = selected * noise
    return scores.reshape(windows, -1).to(device)


def early_feature_norm_scores(context_visual: torch.Tensor) -> torch.Tensor:
    """``mean_t ||x_p||_2`` on the *raw* pre-``LayerNorm`` patch tokens (B4)."""
    return context_visual.float().norm(dim=-1).mean(dim=1)


def early_cls_attention_scores(
    context_visual: torch.Tensor,
    cls_attention: torch.Tensor | None = None,
    cls_token: torch.Tensor | None = None,
) -> torch.Tensor:
    """CLS-to-patch attention energy per patch.

    ``cls_attention`` is the DINOv2 ViT last-layer attention map from the CLS query to each
    patch, expected as ``[B, T, P]``. ``cls_token`` (``[B, T, D]``) is the fallback source,
    giving the cosine energy between the CLS vector and each patch token.

    With neither supplied the caller gets ``None`` rather than a silent substitute: falling
    back to the patch-norm score would make ``early_cls_attention`` an exact duplicate of
    ``early_feature_norm`` while appearing to be a distinct comparator, which would
    overstate how many independent baselines were tried.
    """
    if cls_attention is not None:
        return cls_attention.float().mean(dim=1)
    if cls_token is not None:
        return F.cosine_similarity(
            cls_token.float().unsqueeze(-2), context_visual.float(), dim=-1
        ).mean(dim=1)
    return None


# ---------------------------------------------------------------------------
# Tier 1: learned allocator heads
# ---------------------------------------------------------------------------


def matched_patch_critic_hidden(fan_in: int, d_model: int) -> int:
    """Hidden width whose parameter count is closest to :class:`CurvatureCostateEstimator`'s.

    Keeps the direct critics capacity-matched to the distilled co-state path, so a
    comparison is not a capacity comparison in disguise (B3). ``fan_in`` is the critic's
    own input width, since the two critics see different feature sets.
    """
    target = sum(p.numel() for p in CurvatureCostateEstimator(d_model).parameters())
    return max(1, round((target - 1) / (fan_in + 1)))


class PatchRankingCritic(nn.Module):
    """Capacity-matched direct critic over early patch features -> one score per patch.

    Sees only decision-time information (frame-mean raw patch tokens, the context latent,
    budget and horizon) and *no* co-state, which is exactly the handicap B3 warns about;
    it is kept as the cheap learned baseline the privileged critic must beat.
    """

    def __init__(self, token_dim: int, d_model: int, hidden: int):
        super().__init__()
        self.token_dim = token_dim
        self.d_model = d_model
        self.net = MLP(token_dim + d_model + 2, hidden, 1)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        latent: torch.Tensor,
        budget: torch.Tensor,
        horizon: torch.Tensor,
    ) -> torch.Tensor:
        pooled = patch_tokens.mean(dim=1)                                       # [B, P, D]
        patches = pooled.shape[1]
        context = latent.unsqueeze(1).expand(-1, patches, -1)
        extra = torch.stack([budget, horizon], dim=-1).unsqueeze(1).expand(-1, patches, -1)
        return self.net(torch.cat([pooled, context, extra], dim=-1)).squeeze(-1)


class PrivilegedPatchCritic(nn.Module):
    """Direct critic fed ``[lambda_hat, H_hat, dz_p]``; the decisive comparator (B3).

    Same capacity class as :class:`PatchRankingCritic`, but it sees the distilled
    co-state features, so any advantage of belief-space VOI over it is attributable to the
    epistemic term rather than to having seen the co-state at all.
    """

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.net = MLP(3 * d_model + 2, hidden, 1)
        self.d_model = d_model

    def forward(self, costate: torch.Tensor, hessian: torch.Tensor, delta_z: torch.Tensor, budget: torch.Tensor, horizon: torch.Tensor):
        patches = delta_z.shape[1]
        costate_b = costate.unsqueeze(1).expand(-1, patches, -1)
        hessian_b = hessian.unsqueeze(1).expand(-1, patches, -1)
        extra = torch.stack([budget, horizon], dim=-1).unsqueeze(1).expand(-1, patches, -1)
        return self.net(torch.cat([costate_b, hessian_b, delta_z, extra], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Tier 2: oracles
# ---------------------------------------------------------------------------


def greedy_oracle_masks(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    k_per_camera: int,
    cameras: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedy forward selection under the symmetric per-camera budget -> mask ``[B, P]``.

    At each of the ``k_cam * num_cameras`` steps every admissible candidate is scored by a
    *full* masked rollout and the best is taken, so this is a genuine upper-bound reference
    rather than a linearised estimate. Candidates that would push a camera past its quota
    are inadmissible. It costs ``k_cam * P_cam`` rollout-evaluations per window, which is
    why it is Tier 2 only.

    Returns:
        ``(mask [B, P], marginal_gains)`` where ``marginal_gains[i][t] = J(S_t) - J(S_{t+1})``
        is the **positive** objective reduction achieved by the ``t``-th greedy pick, so the
        submodularity diagnostic can test that these are non-increasing.
    """
    windows = batch["context_state"].shape[0]
    patches = batch["context_visual"].shape[2]
    device = batch["context_state"].device
    num_cameras = CAMERAS
    if cameras is None:
        if patches % num_cameras:
            raise ValueError(f"{patches} patches do not divide into {num_cameras} cameras")
        cameras = camera_of_patch(patches // num_cameras, num_cameras)
    else:
        num_cameras = int(cameras.max().item()) + 1
    camera_index = cameras.to(device)

    selected = torch.zeros(windows, patches, device=device)
    used = torch.zeros(windows, num_cameras, device=device)
    chain: List[List[float]] = [[] for _ in range(windows)]

    one_hot = torch.nn.functional.one_hot
    patch_ids = torch.arange(patches, device=device)
    with torch.no_grad():
        for _ in range(k_per_camera * num_cameras):
            candidates = selected.unsqueeze(1) + one_hot(patch_ids, patches).unsqueeze(0)   # [B, P, P]
            # A candidate is admissible when it is genuinely new (its own camera is not yet
            # full) -- adding an already-selected patch must never count as progress.
            new_patch = 1.0 - selected
            headroom = (k_per_camera - used).gather(1, camera_index.view(1, -1).expand(windows, patches))
            admissible = (new_patch > 0) & (headroom > 0)

            objectives = objective_at_masks(model, batch, candidates).masked_fill(~admissible, float("inf"))
            choice = objectives.argmin(dim=1)                                               # [B]
            best = objectives.gather(1, choice.unsqueeze(1)).squeeze(1)                     # [B]
            previous = objective_at_masks(model, batch, selected.unsqueeze(1))[:, 0]
            # Positive means "this pick reduced the objective". Greedy minimises J, so the
            # reduction is previous - best; storing best - previous would invert the sign and
            # make the submodularity diagnostic count diminishing returns as violations.
            chain = [row + [float(previous[i] - best[i])] for i, row in enumerate(chain)]

            selected.scatter_(1, choice.unsqueeze(1), 1.0)
            picked_camera = camera_index.view(1, -1).expand(windows, patches).gather(1, choice.unsqueeze(1))
            used.scatter_add_(1, picked_camera, torch.ones(windows, 1, device=device))
    return selected, chain


def exhaustive_oracle_masks(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    k_per_camera: int,
    patches_per_camera: int = PATCHES_PER_CAMERA,
    cameras_count: int = CAMERAS,
    max_candidates: int = 20_000,
    max_candidates_per_chunk: int = 64,
) -> torch.Tensor:
    """Exhaustive optimum over per-camera combinations -> mask ``[B, P]``.

    Only tractable for small ``k_per_camera``: the joint space is a *product* over cameras,
    ``C(P_cam, k_cam) ** C``, because the joint attention makes the objective non-separable
    across cameras. At the spec's 16 patches and ``k_cam = 2`` that is 120^2 = 14,400
    subsets, so calibration is confined to a few windows.

    ``max_candidates`` bounds the search space; ``max_candidates_per_chunk`` bounds the number of
    (window, candidate) masked contexts scored in one batched rollout, which is what keeps peak
    memory finite. It is divided across windows, so the product is what matters.
    """
    per_camera = list(itertools.combinations(range(patches_per_camera), k_per_camera))
    if len(per_camera) ** cameras_count > max_candidates:
        raise ValueError(
            f"exhaustive search over {len(per_camera)}**{cameras_count} subsets exceeds "
            f"max_candidates={max_candidates}; lower k_per_camera"
        )
    windows = batch["context_state"].shape[0]
    patches = patches_per_camera * cameras_count
    device = batch["context_state"].device
    masks = torch.zeros(len(per_camera) ** cameras_count, patches, device=device)
    for index, combination in enumerate(itertools.product(per_camera, repeat=cameras_count)):
        for camera, chosen in enumerate(combination):
            masks[index, camera * patches_per_camera + torch.tensor(chosen, dtype=torch.long)] = 1.0

    # Score in candidate chunks. Masking materialises one context_visual per
    # (window, candidate) pair, so evaluating all 14,400 subsets for every window at once
    # asks for tens of GiB and OOMs a 22 GiB L4. The argmin is taken per window afterwards,
    # so chunking changes nothing about which subset wins.
    best = torch.full((windows,), float("inf"), device=device)
    best_index = torch.zeros(windows, dtype=torch.long, device=device)
    # Chunk over *both* axes. Masking builds one context_visual per (window, candidate), and
    # the adapter's attention pooling then runs over windows x candidates x frames at once, so
    # a window-major chunk still needs windows x chunk activations. Bounding the product keeps
    # peak memory flat regardless of how many subsets are searched.
    per_window = max(1, int(max_candidates_per_chunk) // max(1, windows))
    for start in range(0, masks.shape[0], per_window):
        piece = masks[start:start + per_window]
        expanded = piece.unsqueeze(0).expand(windows, -1, -1).reshape(windows, -1, patches)
        objectives = objective_at_masks(model, batch, expanded)
        del expanded
        local = objectives.argmin(dim=1)
        values = objectives.gather(1, local.unsqueeze(1)).squeeze(1)
        improved = values < best
        best = torch.where(improved, values, best)
        best_index = torch.where(improved, local + start, best_index)
        del objectives
    return masks[best_index]


# ---------------------------------------------------------------------------
# Metrics and statistics
# ---------------------------------------------------------------------------


def trimmed_mean(values: np.ndarray, fraction: float = 0.1) -> float:
    """Symmetric trimmed mean; the primary headline statistic (spec section 4)."""
    ordered = np.sort(np.asarray(values, dtype=float))
    n = ordered.size
    if n == 0:
        return float("nan")
    cut = int(math.floor(fraction * n))
    trimmed = ordered[cut : n - cut] if n - 2 * cut > 0 else ordered
    return float(trimmed.mean())


def wilcoxon_signed_rank(differences: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Normal approximation with tie correction and continuity correction (no SciPy in the
    dependency set). Returns ``{'statistic', 'p_value', 'n', 'n_ties'}``.
    """
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n == 0:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": 0, "n_ties": 0}
    nonzero = d[d != 0.0]
    n_nonzero = int(nonzero.size)
    if n_nonzero == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n": n, "n_ties": n}
    ranks = _average_ranks(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    statistic = min(positive, float(ranks.sum()) - positive)

    counts = np.bincount(_rank_codes(np.abs(nonzero)))
    n_ties = int((counts[counts > 1]).sum())
    mean = float(ranks.sum()) / 2.0
    tie_term = float((counts[counts > 1] ** 3 - counts[counts > 1]).sum())
    # Under H0, W+ is the sum of a uniformly random subset of the ranks 1..n, so
    # Var = (1/4) * sum_i i^2 = n(n+1)(2n+1)/24, reduced by the tie correction
    # sum(t^3 - t)/24.
    variance = (n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) - tie_term) / 24.0
    if variance <= 0:
        return {"statistic": statistic, "p_value": 1.0, "n": n, "n_ties": n_ties}
    z = (abs(positive - mean) - 0.5) / math.sqrt(variance)
    return {
        "statistic": statistic,
        "p_value": float(2.0 * _standard_normal_sf(max(z, 0.0))),
        "n": n,
        "n_ties": n_ties,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or sorted_values[index] != sorted_values[start]:
            ranks[order[start:index]] = 0.5 * (start + index + 1)
            start = index
    return ranks


def _rank_codes(values: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(values, return_inverse=True)
    return inverse


def _standard_normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def site_clustered_bootstrap_ci(
    values: np.ndarray,
    sites: np.ndarray,
    num_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Cluster bootstrap over sites, accounting for intra-site correlation (spec section 4).

    Resamples whole sites with replacement rather than windows, because windows from one
    episode are strongly correlated and a window-level bootstrap understates uncertainty.
    """
    values = np.asarray(values, dtype=float)
    sites = np.asarray(sites)
    unique, inverse = np.unique(sites, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=unique.size)
    counts = np.bincount(inverse, minlength=unique.size).astype(float)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, unique.size, size=(num_resamples, unique.size))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "estimate": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": confidence,
        "num_sites": int(unique.size),
        "num_windows": int(values.size),
    }


def additivity_r2(pairs: Sequence[tuple[float, float]]) -> dict:
    """``R^2`` of the true objective on the additive prediction ``sum_{p in S} s_p``.

    Separates non-additivity error (the linear surrogate simply cannot represent the
    interaction between selected patches) from ranking error.
    """
    if len(pairs) < 2:
        return {"r2": float("nan"), "n": int(len(pairs)), "slope": float("nan"), "intercept": float("nan")}
    predicted = np.array([p for p, _ in pairs], dtype=float)
    actual = np.array([y for _, y in pairs], dtype=float)
    design = np.stack([predicted, np.ones_like(predicted)], axis=1)
    slope, intercept = np.linalg.lstsq(design, actual, rcond=None)[0]
    residual = actual - (slope * predicted + intercept)
    total = actual - actual.mean()
    explained = float((total ** 2).sum())
    r2 = float(1.0 - float((residual ** 2).sum()) / explained) if explained > 0 else float("nan")
    return {"r2": r2, "n": int(len(pairs)), "slope": float(slope), "intercept": float(intercept)}


def submodularity_violation_rate(gains: Sequence[Sequence[float]]) -> dict:
    """Fraction of steps that violate diminishing marginal returns.

    ``gains`` holds each greedy chain's per-step *objective reductions*
    (``J(S_t) - J(S_{t+1})``, positive when a pick helps). Submodularity means those
    reductions must be non-increasing along the chain; a step whose reduction exceeds the
    previous one is a violation.
    """
    total = 0
    violations = 0
    for chain in gains:
        for index in range(1, len(chain)):
            total += 1
            if chain[index] > chain[index - 1]:
                violations += 1
    return {
        "rate": float(violations / total) if total else float("nan"),
        "violations": int(violations),
        "comparisons": int(total),
    }


def regret_against_oracle(objectives: torch.Tensor, oracle_objectives: torch.Tensor) -> np.ndarray:
    """``r(S) = J(z_S) - J(z_greedy)``, per window.

    Because the greedy rollout search is a *reference* rather than a proven optimum, ``r``
    is a lower bound on the regret against the true best selection and can be slightly
    negative when a policy beats greedy. Negative values are kept, never clipped: clipping
    them would hide the very fact that the reference is beatable.
    """
    return (objectives - oracle_objectives).detach().cpu().numpy().astype(float)


# ---------------------------------------------------------------------------
# Information boundary (B3)
# ---------------------------------------------------------------------------


def deployable_view(batch: Mapping[str, torch.Tensor]) -> dict:
    """Strip a batch down to the keys a deployed selector may read.

    Every deployable scorer is fed this view, so "does not see future targets" is enforced
    by construction rather than by convention, and the test suite can assert that
    perturbing the target keys leaves deployable selections bit-identical.
    """
    return {key: value for key, value in batch.items() if key in DEPLOYABLE_INPUT_KEYS}


def assert_deployable(policy: str) -> None:
    """Raise if ``policy`` is claimed as deployable but is a Tier 2 reference."""
    if policy in DIAGNOSTIC_POLICIES:
        raise ValueError(
            f"{policy!r} is a non-deployable diagnostic reference (it needs ground-truth future "
            "targets or exhaustive rollouts); it must not be reported as a deployable selector"
        )
    if policy not in COMPARATORS:
        raise ValueError(f"unknown policy {policy!r}; choose from {COMPARATORS + DIAGNOSTIC_POLICIES}")


# ---------------------------------------------------------------------------
# The policy suite
# ---------------------------------------------------------------------------


@dataclass
class SelectionContext:
    """Per-batch quantities every scorer in the suite needs, computed once.

    ``inputs`` is the *deployable view* of the batch, so Tier 1 scorers cannot reach a
    future target even by accident. ``bundle`` and ``delta_sigma`` come from the single
    cotangent backward and the grounded variance head respectively.
    """

    inputs: Mapping[str, torch.Tensor]
    bundle: CotangentBundle
    costate_hat: torch.Tensor
    hessian_hat: torch.Tensor
    delta_sigma: torch.Tensor
    sigma_precision: torch.Tensor
    delta_z: torch.Tensor
    budget_fraction: torch.Tensor
    horizon_fraction: torch.Tensor
    patches_per_camera: int = PATCHES_PER_CAMERA
    cameras: int = CAMERAS
    token_dim: int | None = None
    seed: int = 0
    #: Optional ``[B, T, P]`` DINOv2 CLS-to-patch attention map from the feature cache.
    cls_attention: torch.Tensor | None = None
    #: Optional ``[B, T, D]`` CLS token, used when no attention map was cached.
    cls_token: torch.Tensor | None = None


def selection_context(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    curvature_head: CurvatureCostateEstimator,
    patches_per_camera: int = PATCHES_PER_CAMERA,
    cameras: int = CAMERAS,
    beta: float = 0.5,
    seed: int = 0,
    token_dim: int | None = None,
) -> SelectionContext:
    """Build the shared :class:`SelectionContext` for one batch.

    Everything the deployable scorers read is decision-time information. The distilled
    ``lambda_hat``/``H_hat`` come from ``curvature_head(z)``; the autograd co-state and the
    cotangent gains are confined to the diagnostic reference (B3).

    Cost note: ``cotangent_bundle`` is one forward + one backward, but the *full* context is
    not O(1). ``latent_patch_perturbations`` costs one encode per patch and
    ``epistemic_variance_reduction`` one rollout per patch. Only the cotangent term, which
    the diagnostic reference uses, has the O(P) -> O(1) property the spec describes, and
    the deployable Tier 1 scorers inherit the cost of their grounded inputs. This is
    reported rather than glossed over, because it is the difference between a scoring rule
    and a search.
    """
    inputs = deployable_view(batch)
    bundle = cotangent_bundle(model, batch)
    cls_attention = batch.get("cls_attention")
    windows = bundle.latent.shape[0]
    budget_fraction = torch.full((windows,), 1.0 / max(1, patches_per_camera), device=bundle.latent.device)
    horizon_fraction = torch.ones_like(budget_fraction)
    with torch.no_grad():
        costate_hat, hessian_hat = curvature_head(bundle.latent, budget_fraction, horizon_fraction)
        delta_sigma, precision = epistemic_variance_reduction(model, batch)
        delta_z = latent_patch_perturbations(model, batch)
    return SelectionContext(
        inputs=inputs,
        bundle=bundle,
        costate_hat=costate_hat.detach(),
        hessian_hat=hessian_hat.detach(),
        delta_sigma=delta_sigma.detach(),
        sigma_precision=precision.detach(),
        delta_z=delta_z.detach(),
        budget_fraction=budget_fraction,
        horizon_fraction=horizon_fraction,
        patches_per_camera=patches_per_camera,
        cameras=cameras,
        token_dim=token_dim if token_dim is not None else int(batch["context_visual"].shape[-1]),
        seed=seed,
        cls_attention=None if cls_attention is None else cls_attention.detach(),
    )


def score_policy(
    policy: str,
    context: SelectionContext,
    heads: Mapping[str, nn.Module] | None = None,
    beta: float = 0.5,
    k_per_camera: int = PATCHES_PER_CAMERA // 2,
) -> torch.Tensor:
    """Gains ``[B, P]`` for one of the eight comparators (spec section 3).

    Only ``exact_costate_reference`` reads anything derived from future targets; every
    other policy is fed :meth:`SelectionContext.inputs` only.
    """
    heads = heads or {}
    visual = context.inputs["context_visual"]
    windows, patches = visual.shape[0], visual.shape[2]
    device = visual.device

    if policy == "uniform_grid":
        return uniform_grid_scores(windows, context.patches_per_camera, context.cameras, device=device)
    if policy == "stratified_random":
        return stratified_random_scores(
            windows, k_per_camera, seed=context.seed, patches_per_camera=context.patches_per_camera,
            cameras=context.cameras, device=device,
        )
    if policy == "early_feature_norm":
        return early_feature_norm_scores(visual)
    if policy == "early_cls_attention":
        scores = early_cls_attention_scores(visual, context.cls_attention, context.cls_token)
        if scores is None:
            raise ValueError(
                "early_cls_attention needs the cached DINOv2 CLS-to-patch attention map or a "
                "CLS token; refusing to substitute the patch-norm score, which would make this "
                "comparator an exact duplicate of early_feature_norm. Set "
                "SelectionContext.cls_attention / .cls_token from the feature cache."
            )
        return scores
    if policy == "direct_ranking_critic":
        critic = heads.get(policy)
        if critic is None:
            raise KeyError(f"{policy} needs a trained head; pass heads={{'{policy}': module}}")
        return critic(visual, context.bundle.latent, context.budget_fraction, context.horizon_fraction)
    if policy == "direct_critic_privileged":
        critic = heads.get(policy)
        if critic is None:
            raise KeyError(f"{policy} needs a trained head; pass heads={{'{policy}': module}}")
        return critic(
            context.costate_hat, context.hessian_hat, context.delta_z,
            context.budget_fraction, context.horizon_fraction,
        )
    if policy == "second_order_curvature":
        return curvature_scores(context.costate_hat, context.hessian_hat, context.delta_z)
    if policy == "belief_space_voi":
        return belief_space_voi_scores(
            context.costate_hat, context.hessian_hat, context.delta_z,
            context.delta_sigma, context.sigma_precision, beta=beta,
        )
    if policy == "exact_costate_reference":
        return context.bundle.first_order_gains
    if policy in DIAGNOSTIC_POLICIES:
        raise ValueError(
            f"{policy!r} selects by rollout search and has no scoring function; "
            "use greedy_oracle_masks / exhaustive_oracle_masks directly"
        )
    raise ValueError(f"unknown policy {policy!r}; choose from {COMPARATORS + DIAGNOSTIC_POLICIES}")