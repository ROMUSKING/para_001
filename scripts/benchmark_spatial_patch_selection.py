#!/usr/bin/env python3
"""scripts/benchmark_spatial_patch_selection.py

Session 6A: Fixed-Budget Spatial Patch Selection Benchmark.

At a fixed patch budget ``k`` out of ``P`` spatial tokens, does belief-space sensitivity
select more useful visual regions than a competent direct gain predictor, curvature, and
inexpensive token-selection heuristics? Implements
``docs/plans/2026-10-03-session-6a-spatial-selection-spec.md``.

The selection maths lives in :mod:`adjointrwm.spatial_selection`; this file is the CLI,
the DROID multi-site data path, the head-training loop and the report writer.

Usage::

    # CPU dry run on synthetic fixtures (fast; not evidence)
    python scripts/benchmark_spatial_patch_selection.py --synthetic

    # L4 run over the 12-site DROID shard
    python scripts/benchmark_spatial_patch_selection.py \\
        --cache-dir /content/cache_e3_1 --run-id spatial_patches_20261003
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parent.parent
for candidate in (REPO_DIR / "src", Path("/content/para_001/src"), Path("/content/src")):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from adjointrwm.allocators import CurvatureCostateEstimator, pairwise_margin_ranking_loss  # noqa: E402
from adjointrwm.data import WindowDataset, WindowSpec, fit_normaliser  # noqa: E402
from adjointrwm.io import atomic_write_json  # noqa: E402
from adjointrwm.models.common import ArmDims  # noqa: E402
from adjointrwm.models.registry import build_arm  # noqa: E402
from adjointrwm.spatial_selection import (  # noqa: E402
    BUDGETS,
    CAMERAS,
    COMPARATORS,
    DEPLOYABLE_POLICIES,
    DIAGNOSTIC_POLICIES,
    PATCHES_PER_CAMERA,
    HEURISTIC_POLICIES,
    PatchRankingCritic,
    PrivilegedPatchCritic,
    additivity_r2,
    assert_deployable,
    camera_of_patch,
    curvature_scores,
    deployable_view,
    exhaustive_oracle_masks,
    greedy_oracle_masks,
    latent_patch_perturbations,
    matched_patch_critic_hidden,
    objective_at_masks,
    score_policy,
    select_topk_per_camera,
    selection_context,
    site_clustered_bootstrap_ci,
    submodularity_violation_rate,
    trimmed_mean,
    wilcoxon_signed_rank,
)

#: Everything that produces a mask and a regret row: the eight comparators plus the
#: Tier 2 autograd co-state reference. ``calibrated_greedy_oracle`` defines the regret
#: baseline and is reported separately (its regret is zero by construction).
SCORED_POLICIES = COMPARATORS + ("exact_costate_reference",)

#: Greedy rollout search is a *reference*, not a proven optimum. Under a symmetric
#: per-camera budget the joint selection space is a product over cameras and the objective
#: is non-separable, so greedy can be strictly beaten. Regret against it is therefore a
#: lower bound on true regret, and can go slightly negative. The calibration below measures
#: the gap where exhaustive search is tractable.
GREEDY_REFERENCE_NAME = "calibrated_greedy_oracle"
GREEDY_IS_OPTIMAL = False

#: Spec section 5, pre-registered.
EXIT_GATE_MARGIN = 0.08          # >= 8.0% trimmed-mean-regret advantage over uniform_grid / early_feature_norm
NON_INFERIORITY_MARGIN = 0.03   # within 3.0% of direct_critic_privileged
WILCOXON_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Session 6A: Fixed-Budget Spatial Patch Selection")
    parser.add_argument("--synthetic", "--synthetic-test", action="store_true", dest="synthetic",
                        help="CPU dry run on synthetic fixtures (not evidence)")
    parser.add_argument("--cache-dir", type=str, default="/content/cache_e3_1")
    parser.add_argument("--drive-root", type=str, default="/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production")
    parser.add_argument("--run-id", type=str, default="spatial_patches_20261003")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks/spatial_selection")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers; the shard is .npz-IO-bound (Session 0)")
    parser.add_argument("--head-batches", type=int, default=16,
                        help="train batches used to fit the heads; the same prefix for every seed")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--patches-per-camera", type=int, default=PATCHES_PER_CAMERA)
    parser.add_argument("--beta", type=float, default=0.5, help="belief-space VOI weight")
    parser.add_argument("--eval-windows", type=int, default=0, help="cap on scored test windows (0 = all)")
    parser.add_argument("--oracle-windows", type=int, default=64, help="windows given the rollout-search oracle")
    parser.add_argument("--exhaustive-k", type=int, default=2, help="k_cam for the exhaustive calibration (0 disables)")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_spatial_splits(cache_dir: Path, patches_per_camera: int):
    """Load the multi-site DROID shard with token-layout patch features.

    Returns ``(train, val, test, site_by_episode)``. Split assignment stays with the cached
    manifest (split before windowing); the normaliser is fitted on train episodes only.

    All three splits come from this one definition so that a trainer cannot drift from the
    evaluator's data path. The validation split is what a patch-dropout backbone selects its
    checkpoint on; the test split is for the benchmark alone and no training code may read it.
    """
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.exists():
        manifest_path = cache_dir / "e3_1_droid_500_manifest.json"
    if not manifest_path.exists():
        fallback = REPO_DIR / "results/data/droid_e3_1/e3_1_droid_500_manifest.json"
        if fallback.exists():
            manifest_path = fallback
        else:
            raise FileNotFoundError(f"Missing cache manifest at {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    episodes_dir = cache_dir / "episodes"
    input_keys = ("exterior_patches", "wrist_patches")
    target_keys = ("exterior_embeddings", "wrist_embeddings")

    by_split: Dict[str, list] = {"train": [], "val": [], "test": []}
    site_by_episode: Dict[str, str] = {}
    for episode in manifest.get("episodes", []):
        split = {"val": "val", "validation": "val"}.get(episode.get("split", "train"), episode.get("split", "train"))
        episode_id = episode["episode_id"]
        site_by_episode[episode_id] = episode.get("site", "unknown")
        path = episodes_dir / f"{episode_id}.npz"
        if not path.exists():
            continue
        by_split[split].append(
            {"episode_id": episode_id, "cached_path": str(path), "length": episode.get("length", 100),
             "split": split, "site": episode.get("site", "unknown")}
        )

    normalisation_path = cache_dir / "normalisation.npz"
    if normalisation_path.exists():
        with np.load(normalisation_path) as archive:
            normalisation = {key: archive[key] for key in archive.files}
    else:
        arrays = []
        for record in by_split["train"]:
            with np.load(record["cached_path"]) as episode:
                arrays.append({"states": episode["states"], "actions": episode["actions"]})
        normalisation = fit_normaliser([a["states"] for a in arrays], [a["actions"] for a in arrays])

    spec = WindowSpec(context_len=8, horizon=4, stride=2)
    kwargs = dict(spec=spec, normaliser=normalisation, input_visual_keys=input_keys,
                  target_visual_keys=target_keys, visual_layout="tokens",
                  # The early_cls_attention comparator reads the cached DINOv2 CLS-to-patch map.
                  # Requesting it here means a cache without it fails loudly instead of the
                  # scorer silently substituting the patch-norm score, which would make that
                  # comparator an exact duplicate of early_feature_norm.
                  auxiliary_keys=("cls_attention",))
    if patches_per_camera != PATCHES_PER_CAMERA:
        raise ValueError(
            f"the cached shard has {PATCHES_PER_CAMERA} patches per camera; "
            f"--patches-per-camera={patches_per_camera} needs a re-extracted feature cache"
        )
    return (
        WindowDataset(by_split["train"], **kwargs),
        WindowDataset(by_split["val"], **kwargs),
        WindowDataset(by_split["test"], **kwargs),
        site_by_episode,
    )


def load_spatial_dataset(cache_dir: Path, patches_per_camera: int):
    """``(train, test, site_by_episode)``: the evaluator's view of the shard."""
    train, _val, test, site_by_episode = load_spatial_splits(cache_dir, patches_per_camera)
    return train, test, site_by_episode


def make_synthetic_batch(windows: int, frames: int, patches: int, token_dim: int,
                         state_dim: int, action_dim: int, horizon: int, target_visual_dim: int,
                         seed: int, device: torch.device) -> dict:
    """Synthetic fixtures for CPU dry runs and unit tests. Random, never evidence."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draw = lambda *shape: torch.randn(*shape, generator=generator).to(device)  # noqa: E731
    return {
        "context_visual": draw(windows, frames, patches, token_dim),
        "context_state": draw(windows, frames, state_dim),
        "context_action": draw(windows, frames, action_dim),
        "future_actions": draw(windows, horizon, action_dim),
        "target_state": draw(windows, horizon, state_dim),
        "target_visual": draw(windows, horizon, target_visual_dim),
        # Synthetic stand-in for the cached DINOv2 CLS-to-patch attention map, so the
        # comparator is exercised end to end. Random, hence "not evidence".
        "cls_attention": draw(windows, frames, patches).abs(),
    }


def move_to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def build_spatial_model(args, patches_per_camera: int, device: torch.device, width: int | None = None,
                        token_dim: int = 384):
    dims = ArmDims(
        state_dim=14, action_dim=7, visual_tokens=patches_per_camera * CAMERAS,
        visual_token_dim=token_dim, target_visual_dim=2 * token_dim, context_len=8, horizon=4,
    )
    model = build_arm("spatial_adjoint_rwm", dims, width=width or args.width,
                      transformer_heads=8, transformer_layers=2)
    return model.to(device).eval()


def load_teacher_weights(model: nn.Module, path: Path) -> dict:
    """Load a trained spatial backbone if present; report honestly when absent (B6).

    A model trained at ``P = 32`` and evaluated at ``k = 4`` without patch dropout is
    off-distribution, so whether the checkpoint was patch-dropout trained is recorded in
    the summary rather than assumed.
    """
    if not path.exists():
        return {"loaded": False, "path": str(path), "reason": "checkpoint not found; using a randomly initialised backbone"}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "loaded": True, "path": str(path),
        "missing_keys": sorted(missing)[:8], "unexpected_keys": sorted(unexpected)[:8],
        "patch_dropout_trained": bool(checkpoint.get("patch_dropout_trained", False)),
    }


# ---------------------------------------------------------------------------
# Head training
# ---------------------------------------------------------------------------


def exact_marginal_gains(model: nn.Module, batch: Mapping[str, torch.Tensor],
                         base_mask: torch.Tensor | None = None) -> torch.Tensor:
    """True objective reduction ``J(z_S) - J(z_{S u p})`` per patch ``[B, P]``.

    This is the supervision the ranking critics are trained on. It needs future targets,
    so it is a *training label* only: no deployable selector may call it.
    """
    patches = batch["context_visual"].shape[2]
    windows = batch["context_state"].shape[0]
    device = batch["context_state"].device
    base = torch.zeros(windows, patches, device=device) if base_mask is None else base_mask.to(device)
    candidates = base.unsqueeze(1).expand(-1, patches, -1).clone()
    candidates[:, torch.arange(patches), torch.arange(patches)] = 1.0

    with torch.no_grad():
        base_objective = objective_at_masks(model, batch, base.unsqueeze(1))[:, 0]
        per_patch = objective_at_masks(model, batch, candidates)
    return base_objective.unsqueeze(1) - per_patch


def train_heads(model: nn.Module, batch_factory, args,
                token_dim: int, patches_per_camera: int, device: torch.device) -> dict:
    """Train the distilled curvature head and both ranking critics on exact marginal gains.

    Each head is an independent module with its own optimiser, so a critic cannot benefit
    from the co-state head's training signal and the three losses stay separable.

    ``batch_factory`` is a callable returning a fresh iterable of device-resident batches,
    re-invoked on every pass: the caller keeps its windows on the host to bound GPU memory, so
    a one-shot iterable could not survive the many passes ``max_steps`` requires.
    """
    from adjointrwm.spatial_selection import cotangent_bundle

    d_model = model.d_model
    curvature = CurvatureCostateEstimator(d_model).to(device)
    ranking = PatchRankingCritic(token_dim, d_model, matched_patch_critic_hidden(token_dim + d_model + 2, d_model)).to(device)
    privileged = PrivilegedPatchCritic(d_model, matched_patch_critic_hidden(3 * d_model + 2, d_model)).to(device)

    optimisers = {
        "curvature": torch.optim.AdamW(curvature.parameters(), lr=args.lr),
        "direct_ranking_critic": torch.optim.AdamW(ranking.parameters(), lr=args.lr),
        "direct_critic_privileged": torch.optim.AdamW(privileged.parameters(), lr=args.lr),
    }
    # `batches` is a callable returning a fresh iterable of device-resident batches, cycled for
    # `max_steps`. The batch size is read from the batch in hand, so nothing has to be resident
    # all at once.
    history = {name: [] for name in optimisers}
    modules = {"curvature": curvature, "direct_ranking_critic": ranking, "direct_critic_privileged": privileged}

    # `batches` is a *factory* returning a fresh iterable, because the caller keeps its batches
    # on the host to bound GPU memory: a one-shot generator could only be traversed once, and
    # the training loop needs many passes over the same windows.
    def cycle():
        while True:
            produced = 0
            for item in batch_factory():
                produced += 1
                yield item
            if produced == 0:
                raise SystemExit("no batches available for head training")

    stream = cycle()
    for step in range(args.max_steps):
        batch = next(stream)
        windows = batch["context_state"].shape[0]
        budget_fraction = torch.full((windows,), 1.0 / max(1, patches_per_camera), device=device)
        horizon_fraction = torch.ones_like(budget_fraction)

        # Labels are exact and need no gradient; every head input below is detached, so no
        # training signal can reach the frozen world model.
        gains = exact_marginal_gains(model, batch)
        bundle = cotangent_bundle(model, batch)
        latent = bundle.latent.detach()
        # Genuine latent-space perturbation: the curvature, VOI and privileged-critic
        # scorers all contract with a latent co-state, so they need dz in the latent basis.
        delta_z = latent_patch_perturbations(model, batch)

        # 1. Distilled co-state: direction to the autograd co-state plus curvature alignment,
        #    so lambda_hat / H_hat are decision-time quantities that need no future target.
        costate_hat, hessian_hat = curvature(latent, budget_fraction, horizon_fraction)
        direction = (1.0 - F.cosine_similarity(costate_hat, bundle.exact_costate, dim=-1)).mean()
        predicted_curvature = (hessian_hat.unsqueeze(1) * delta_z.pow(2)).sum(-1)
        curvature_alignment = F.smooth_l1_loss(predicted_curvature, gains)
        curvature_loss = direction + 0.5 * curvature_alignment

        # 2/3. Ranking critics against the same exact marginal gains (pairwise hinge).
        ranking_loss = pairwise_margin_ranking_loss(
            ranking(batch["context_visual"], latent, budget_fraction, horizon_fraction), gains
        )
        privileged_loss = pairwise_margin_ranking_loss(
            privileged(costate_hat.detach(), hessian_hat.detach(), delta_z, budget_fraction, horizon_fraction), gains
        )

        for name, loss in (
            ("curvature", curvature_loss),
            ("direct_ranking_critic", ranking_loss),
            ("direct_critic_privileged", privileged_loss),
        ):
            optimisers[name].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modules[name].parameters(), 1.0)
            optimisers[name].step()
            history[name].append(float(loss.detach()))

    return {
        "heads": {"direct_ranking_critic": ranking.eval(), "direct_critic_privileged": privileged.eval()},
        "curvature_head": curvature.eval(),
        "final_loss": {name: values[-1] for name, values in history.items() if values},
        "parameter_counts": {
            "curvature_head": sum(p.numel() for p in curvature.parameters()),
            "direct_ranking_critic": sum(p.numel() for p in ranking.parameters()),
            "direct_critic_privileged": sum(p.numel() for p in privileged.parameters()),
        },
        "steps": args.max_steps,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def calibrate_greedy_optimality(model: nn.Module, batch: Mapping, k_cam: int,
                                patches_per_camera: int) -> dict:
    """Measure the greedy oracle's gap to the exhaustive optimum (spec B5).

    The greedy search is only a trustworthy ceiling if it is close to the true optimum, so
    this is computed rather than assumed. Only tractable for small ``k_cam``.
    """
    # Calibration runs on a couple of windows: the exhaustive space is 14,400 subsets at
    # k_cam=2, and its purpose is to bound the greedy gap, not to re-estimate every window.
    small = batch["context_state"].shape[0] > 2
    if small:
        batch = {k: (v[:2] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] > 2 else v)
                 for k, v in batch.items()}
    # No autograd: calibration searches ~14,400 subsets and every retained graph node across
    # those chunks accumulates. This is a measurement of an already-trained model, so no
    # gradient is wanted, and without this the search OOMs a 22 GiB L4.
    with torch.no_grad():
        greedy_mask, _ = greedy_oracle_masks(model, batch, k_cam, camera_of_patch(patches_per_camera, CAMERAS))
        best_mask = exhaustive_oracle_masks(model, batch, k_cam, patches_per_camera=patches_per_camera)
        greedy_value = objective_at_masks(model, batch, greedy_mask.unsqueeze(1))[:, 0]
        best_value = objective_at_masks(model, batch, best_mask.unsqueeze(1))[:, 0]
    gap = (greedy_value - best_value).detach().cpu().numpy()
    return {
        "k_per_camera": k_cam,
        "n_windows": int(gap.size),
        "mean_excess_objective": float(gap.mean()),
        "max_excess_objective": float(gap.max()),
        "fraction_optimal": float((gap <= 1e-6).mean()),
    }


def beta_sensitivity(model, curvature_head, heads: dict, batches: Sequence[Mapping], args,
                     k_cam: int, patches_per_camera: int, token_dim: int) -> dict:
    """Sweep the belief-space weight over ``{0, 0.5, 1}`` and its sign flip (spec B2).

    ``beta = 0`` reproduces ``second_order_curvature`` exactly and is the control that shows
    the epistemic term is doing the work; negative ``beta`` shows whether the sign of the
    variance reduction is even the right one.
    """
    index = camera_of_patch(patches_per_camera, CAMERAS)
    rows = {}
    for beta in (0.0, 0.5, 1.0, -0.5, -1.0):
        regrets = []
        for batch in batches:
            context = selection_context(
                model, batch, curvature_head, patches_per_camera=patches_per_camera,
                cameras=CAMERAS, beta=beta, seed=args.seed, token_dim=token_dim,
            )
            gains = score_policy("belief_space_voi", context, heads=heads, beta=beta, k_per_camera=k_cam)
            mask = select_topk_per_camera(gains, k_cam, index, CAMERAS)
            with torch.no_grad():
                selected = objective_at_masks(model, batch, mask.unsqueeze(1))[:, 0]
                oracle = objective_at_masks(model, batch, greedy_oracle_masks(
                    model, batch, k_cam, index)[0].unsqueeze(1))[:, 0]
            regrets.extend((selected - oracle).cpu().tolist())
        values = np.asarray(regrets, dtype=float)
        if values.size == 0:
            # An exhausted stream would otherwise report NaN, which reads as a measurement.
            raise RuntimeError(
                f"beta={beta} produced no windows at k_cam={k_cam}; the evaluation stream was "
                "exhausted, so the sweep cannot be summarised"
            )
        rows[f"beta={beta:+.1f}"] = {
            "mean_regret": float(values.mean()),
            "trimmed_mean_regret_10pct": trimmed_mean(values),
            "n_windows": int(values.size),
        }
    return rows


def assert_beta_zero_matches_curvature(beta_rows: Mapping[str, Mapping], results: Sequence[Mapping]) -> None:
    """``beta = 0`` must reproduce ``second_order_curvature`` regret exactly (spec B2 control).

    The two are computed by different code paths, so this is a genuine cross-check that the
    epistemic term is the *only* difference between them; a mismatch means one path is
    scoring something other than what its name says.
    """
    for budget, rows in beta_rows.items():
        reference = next((r for r in results if f"k={budget.split('=')[1]}" in r), None)
        if reference is None:
            continue
        curvature = reference["per_policy"]["second_order_curvature"]["trimmed_mean_regret_10pct"]
        control = rows["beta=+0.0"]["trimmed_mean_regret_10pct"]
        if not np.isclose(curvature, control, rtol=1e-6, atol=1e-9):
            raise AssertionError(
                f"{budget}: beta=0 VOI ({control:.6g}) must equal second_order_curvature "
                f"({curvature:.6g}); the two code paths disagree"
            )


def unavailable_policies(context, heads: Mapping[str, nn.Module] | None = None) -> list:
    """Policies that cannot be scored with the data this batch actually carries.

    A comparator that silently substitutes another one's score is worse than a missing
    one: it would inflate the apparent number of independent baselines. Such policies are
    reported as unavailable with the reason, never filled in.
    """
    missing = []
    for policy in SCORED_POLICIES:
        try:
            score_policy(policy, context, heads=heads or {}, k_per_camera=1)
        except (ValueError, KeyError) as error:
            missing.append({"policy": policy, "reason": str(error)})
    return missing


def evaluate(model: nn.Module, curvature_head, heads: dict, batches: Sequence[Mapping], args,
             k_cam: int, patches_per_camera: int, token_dim: int, sites: Sequence[str] | None = None,
             device: torch.device | None = None) -> dict:
    """Score every policy at one budget and return per-window regrets and diagnostics.

    Per-window regrets are also returned in ``rows`` and written to parquet by the caller.
    Session 5's audit found that committing only aggregates makes the paired comparison
    permanently untestable, because the per-window values needed for a paired bootstrap or a
    Wilcoxon test are then gone; the aggregates here reproduce exactly either way, but only the
    rows support inference.
    """
    # `batches` is any iterable of already-resident batches; a one-shot iterator cannot be
    # indexed, so the batch size is read from the first item as the stream is consumed.
    regrets: Dict[str, List[float]] = {name: [] for name in SCORED_POLICIES + ("calibrated_greedy_oracle",)}
    objectives: Dict[str, List[float]] = {name: [] for name in SCORED_POLICIES}
    oracle_objectives: List[float] = []
    additive_pairs: List[tuple[float, float]] = []
    greedy_chains: List[List[float]] = []
    # Free the cached allocator between batches: every scored policy holds a cotangent
    # bundle and a perturbation tensor, and on a 22 GiB L4 the retained segments from one
    # batch are enough to starve the next.
    free_cuda = device is not None and device.type == "cuda"
    # Provenance for each scored window, so the parquet rows can be clustered by site and by
    # episode rather than treated as i.i.d. (spec 4.3, and remediation item 3 of the Session 5
    # audit: stride-2 windows overlap, so resampling them independently understates variance).
    row_windows: List[int] = []
    row_episodes: List[str] = []
    row_sites: List[str] = []

    for batch in batches:
        context = selection_context(
            model, batch, curvature_head,
            patches_per_camera=patches_per_camera, cameras=CAMERAS,
            beta=args.beta, seed=args.seed, token_dim=token_dim,
        )
        masks, all_gains = {}, {}
        unavailable = {entry["policy"]: entry["reason"] for entry in unavailable_policies(context, heads)}
        for policy in SCORED_POLICIES:
            if policy in unavailable:
                continue
            gains = score_policy(policy, context, heads=heads, beta=args.beta, k_per_camera=k_cam).detach()
            all_gains[policy] = gains
            masks[policy] = select_topk_per_camera(gains, k_cam, camera_of_patch(patches_per_camera, CAMERAS), CAMERAS)
        scored = [p for p in SCORED_POLICIES if p in masks]
        with torch.no_grad():
            selected_objectives = objective_at_masks(
                model, batch, torch.stack([masks[p] for p in scored], dim=1)
            )
            oracle_mask, chain = greedy_oracle_masks(
                model, batch, k_cam, camera_of_patch(patches_per_camera, CAMERAS)
            )
            oracle_value = objective_at_masks(model, batch, oracle_mask.unsqueeze(1))[:, 0]
        greedy_chains.extend(chain)
        oracle_objectives.extend(oracle_value.cpu().tolist())
        # Sites come from the caller because the batch carries episode ids, not sites.
        ids = batch.get("episode_id")
        count = batch["context_state"].shape[0]
        for offset in range(count):
            row_windows.append(len(row_episodes))
            episode = str(ids[offset]) if ids is not None else "unknown"
            row_episodes.append(episode)
            row_sites.append(sites[len(row_episodes) - 1] if sites is not None else "unknown")

        for index, policy in enumerate(scored):
            value = selected_objectives[:, index]
            regrets[policy].extend((value - oracle_value).cpu().tolist())
            objectives[policy].extend(value.cpu().tolist())
            # Additivity diagnostic: does the additive surrogate sum_{p in S} s_p predict
            # the measured objective J(z_S) for the same selection?
            gains = all_gains[policy]
            additive_pairs.extend(
                (float(gains[i][masks[policy][i].bool()].sum()), float(value[i]))
                for i in range(value.shape[0])
            )
        # The greedy reference defines regret, so its regret is zero by construction; sized
        # from the batch just scored rather than from a pre-read of the stream.
        regrets["calibrated_greedy_oracle"].extend([0.0] * value.shape[0])
        if free_cuda:
            torch.cuda.empty_cache()

    scored = len(regrets[SCORED_POLICIES[0]])
    policy_availability = unavailable or None
    site_labels = list(sites) if sites is not None and len(sites) == scored else None
    summary = {}
    for policy in regrets:
        values = np.asarray(regrets[policy], dtype=float)
        if values.size == 0:
            continue
        entry = {
            "mean_regret": float(values.mean()),
            "median_regret": float(np.median(values)),
            "trimmed_mean_regret_10pct": trimmed_mean(values),
            "std_regret": float(values.std()),
            "n_windows": int(values.size),
            # Negative regret means the policy beat the greedy reference; reporting the
            # count keeps the reference's weakness visible instead of hiding it.
            "n_negative_regret": int((values < -1e-9).sum()),
        }
        if site_labels is not None and policy in SCORED_POLICIES:
            entry["site_clustered_ci_95"] = site_clustered_bootstrap_ci(
                values, np.asarray(site_labels), num_resamples=args.bootstrap_resamples, seed=args.seed
            )
        summary[policy] = entry

    comparisons = {}
    if "belief_space_voi" in summary and "direct_ranking_critic" in summary:
        voi = np.asarray(regrets["belief_space_voi"], dtype=float)
        critic = np.asarray(regrets["direct_ranking_critic"], dtype=float)
        comparisons["voi_minus_direct_ranking"] = wilcoxon_signed_rank(voi - critic)
    for reference in ("direct_critic_privileged", "second_order_curvature"):
        if "belief_space_voi" in summary and reference in summary:
            voi = np.asarray(regrets["belief_space_voi"], dtype=float)
            other = np.asarray(regrets[reference], dtype=float)
            comparisons[f"voi_minus_{reference}"] = wilcoxon_signed_rank(voi - other)

    diagnostics = {
        "additivity_r2": additivity_r2(additive_pairs),
        "submodularity": submodularity_violation_rate(greedy_chains),
        "distillation_gap": (
            summary.get("exact_costate_reference", {}).get("trimmed_mean_regret_10pct", float("nan"))
            - summary.get("belief_space_voi", {}).get("trimmed_mean_regret_10pct", float("nan"))
        ),
        "greedy_oracle_mean_objective": float(np.mean(oracle_objectives)) if oracle_objectives else float("nan"),
    }
    # Long-form per-window rows, committed as parquet so the paired and clustered inference
    # the exit gate rests on stays re-runnable from the artefact alone.
    rows = []
    for policy, values in regrets.items():
        if len(values) != len(row_episodes):
            continue  # a policy that was unavailable for some batches has no aligned rows
        for index, value in enumerate(values):
            rows.append({
                "window": index,
                "k_per_camera": k_cam,
                "k_total": k_cam * CAMERAS,
                "policy": policy,
                "regret": float(value),
                "episode_id": row_episodes[index],
                "site": row_sites[index],
            })

    return {
        "k_total": k_cam * CAMERAS, "k_per_camera": k_cam, "per_policy": summary,
        "comparisons": comparisons, "diagnostics": diagnostics,
        "unavailable_policies": policy_availability,
        "rows": rows,
    }


def detect_duplicate_seed_metrics(seed_provenance: Sequence[Mapping]) -> dict:
    """Flag seeds whose trained heads are bit-identical, i.e. duplicate replicates.

    Session 5 reported "3 random seeds" while two of the three were the same evaluation:
    a silent checkpoint fallback made seed 2 load seed 0's teacher, and every performance
    scalar matched bit-for-bit (audit ``2026-10-03_session5_seed_duplication_and_voi_identity_audit.md``,
    remediation item 5). Head parameter hashes are recorded per seed precisely so this is
    checkable from the artefact. Wall-clock fields are not part of the comparison, because
    identical work legitimately differs in timing.
    """
    hashes = {}
    for entry in seed_provenance:
        combined = "|".join(
            f"{name}={digest['sha256']}" for name, digest in sorted(entry.get("head_sha256", {}).items())
        )
        hashes.setdefault(combined, []).append(entry["seed"])
    duplicated = {key: seeds for key, seeds in hashes.items() if len(seeds) > 1}
    return {
        "n_seeds": len(seed_provenance),
        "n_distinct_head_parameter_sets": len(hashes),
        "duplicate_seed_groups": sorted(seeds for seeds in duplicated.values()),
        "seeds_are_distinct_replicates": not duplicated,
        "note": (
            "The backbone is a single shared checkpoint across seeds; what varies per seed is "
            "the head initialisation and its training. Identical head hashes mean the seeds are "
            "the same evaluation, not independent replicates."
        ),
    }


def run_oracle_calibration(model: nn.Module, batch: Mapping, args, patches_per_camera: int) -> list:
    """Exhaustive calibration of the greedy oracle at the requested small budgets."""
    if args.exhaustive_k <= 0 or args.exhaustive_k > patches_per_camera:
        return []
    combinations = math.comb(patches_per_camera, args.exhaustive_k) ** CAMERAS
    if combinations > 20_000:
        print(f"  skipping exhaustive calibration at k_cam={args.exhaustive_k}: "
              f"{combinations} subsets is intractable")
        return []
    print(f"  calibrating the greedy oracle against {combinations} exhaustive subsets "
          f"(k_cam={args.exhaustive_k})...")
    return [calibrate_greedy_optimality(model, batch, args.exhaustive_k, patches_per_camera)]


def evaluate_exit_gate(budget_results: Sequence[Mapping], synthetic: bool = False) -> dict:
    """Apply the pre-registered exit gate (spec section 5).

    Criteria are evaluated **per budget** and must hold at every budget: averaging
    advantages across budgets would let a strong small-budget result mask a failure at a
    large one, which the pre-registration does not authorise.

    On synthetic fixtures the gate is refused outright. Random tensors can satisfy a
    numerical threshold by chance, and a "pass" on invented data would be a fabricated
    result, so no criterion is reported as met and the branch says so.
    """
    def trimmed(policy: str, budget: Mapping | None = None) -> float:
        source = [budget] if budget is not None else list(budget_results)
        values = [r["per_policy"].get(policy, {}).get("trimmed_mean_regret_10pct", float("nan")) for r in source]
        finite = [v for v in values if not np.isnan(v)]
        return float(np.mean(finite)) if finite else float("nan")

    per_budget = {}
    for budget in budget_results:
        voi = trimmed("belief_space_voi", budget)
        advantages = {}
        for baseline in ("uniform_grid", "early_feature_norm"):
            base = trimmed(baseline, budget)
            # A relative advantage is meaningless when the baseline regret is near zero.
            advantages[baseline] = float((base - voi) / base) if base > 1e-9 else float("nan")
        privileged = trimmed("direct_critic_privileged", budget)
        p_value = budget.get("comparisons", {}).get("voi_minus_second_order_curvature", {}).get(
            "p_value", float("nan"))
        per_budget[f"k={budget['k_total']}"] = {
            "belief_space_voi_trimmed_mean_regret": voi,
            "advantage_over_baseline": advantages,
            "criterion_1_primary_advantage_met": bool(
                all(not np.isnan(v) and v >= EXIT_GATE_MARGIN for v in advantages.values())
            ),
            "criterion_2_non_inferiority_met": bool(
                not np.isnan(privileged) and voi <= privileged * (1.0 + NON_INFERIORITY_MARGIN)
            ),
            "wilcoxon_p_vs_curvature": p_value,
        }

    if synthetic:
        return {
            "evaluated": False,
            "reason": "synthetic fixtures are not evidence; the exit gate is not evaluated",
            "per_budget": per_budget,
            "criterion_1_primary_advantage_met": False,
            "criterion_2_non_inferiority_met": False,
            "branch": "not_evaluated: synthetic run (not evidence)",
            "thresholds": {"advantage": EXIT_GATE_MARGIN, "non_inferiority": NON_INFERIORITY_MARGIN,
                           "alpha": WILCOXON_ALPHA},
        }

    criterion_1 = all(entry["criterion_1_primary_advantage_met"] for entry in per_budget.values())
    criterion_2 = all(entry["criterion_2_non_inferiority_met"] for entry in per_budget.values())
    p_values = [entry["wilcoxon_p_vs_curvature"] for entry in per_budget.values()]
    tie = bool(p_values) and all(not np.isnan(p) and p > WILCOXON_ALPHA for p in p_values)

    def trimmed_overall(policy: str) -> float:
        return trimmed(policy)

    voi_overall = trimmed_overall("belief_space_voi")
    curvature = trimmed_overall("second_order_curvature")
    learned_best = min(
        (trimmed_overall(p) for p in ("belief_space_voi", "second_order_curvature", "direct_ranking_critic",
                                      "direct_critic_privileged") if not np.isnan(trimmed_overall(p))),
        default=float("nan"),
    )
    negative = bool(learned_best >= min(trimmed_overall("uniform_grid"),
                                        trimmed_overall("early_feature_norm")) - 1e-12)

    if not criterion_1 and negative:
        branch = "negative_result: geometric coverage dominates downstream sensitivity"
    elif tie and not criterion_1:
        branch = "tie: second-order curvature captures the dominant spatial signal"
    elif criterion_1 and criterion_2 and not tie:
        branch = "pass: belief-space VOI leads on trimmed-mean regret at every budget"
    else:
        branch = "inconclusive"
    return {
        "evaluated": True,
        "belief_space_voi_trimmed_mean_regret": voi_overall,
        "second_order_curvature_trimmed_mean_regret": curvature,
        "per_budget": per_budget,
        "criterion_1_primary_advantage_met": bool(criterion_1),
        "criterion_2_non_inferiority_met": bool(criterion_2),
        "tie_with_curvature": tie,
        "branch": branch,
        "thresholds": {"advantage": EXIT_GATE_MARGIN, "non_inferiority": NON_INFERIORITY_MARGIN,
                       "alpha": WILCOXON_ALPHA},
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(args, device: torch.device) -> dict:
    """Run the full Session 6A suite and return the summary dictionary."""
    patches_per_camera = args.patches_per_camera
    token_dim = 384
    if args.synthetic:
        # Small fixtures keep the CPU dry run quick; the layout contracts are identical.
        # 16 patches per camera keeps the real 4x4 grid geometry, so the heuristics and
        # the symmetric-budget code paths are exercised exactly as in a real run.
        patches_per_camera, token_dim, width = 16, 16, 64
        frames, state_dim, action_dim, horizon = 4, 14, 7, 4
        head_batches_host = None  # set per branch below
        synthetic_train = [move_to_device(make_synthetic_batch(args.batch_size, frames, patches_per_camera * CAMERAS,
                                                              token_dim, state_dim, action_dim, horizon,
                                                              2 * token_dim, 1000 + s, device), device)
                           for s in range(4)]

        def stream_head_batches():
            for batch in synthetic_train:
                yield batch

        synthetic_eval = [move_to_device(make_synthetic_batch(args.batch_size, frames, patches_per_camera * CAMERAS,
                                                             token_dim, state_dim, action_dim, horizon,
                                                             2 * token_dim, 2000 + s, device), device)
                           for s in range(2)]
        eval_host = None
        # Already resident on the device, so the accessor just replays the same batches; each
        # call yields a fresh iterator so every pass scores the identical windows in order.
        eval_batches = lambda: iter(synthetic_eval)  # noqa: E731
        site_by_episode = None
        checkpoint = {"loaded": False, "reason": "synthetic run"}
        args.max_steps = min(args.max_steps, 40)
    else:
        frames, state_dim, action_dim, horizon, width = 8, 14, 7, 4, args.width
        checkpoint = load_teacher_weights(
            build_spatial_model(args, patches_per_camera, device, width, token_dim),
            Path(args.drive_root) / "runs" / args.run_id / "best_spatial.pt",
        )
        if not checkpoint.get("loaded"):
            # Checked before any data is read: a randomly initialised backbone cannot show
            # whether spatial selection helps a trained predictor, and every downstream number
            # would be an artefact of the initialisation. Refuse rather than emit a gate on
            # invented evidence.
            raise SystemExit(
                f"Refusing to run in 'real' mode without a trained spatial backbone: no "
                f"checkpoint at {checkpoint['path']}. Pass --synthetic for a dry run, or supply "
                f"a patch-dropout-trained checkpoint (spec B6)."
            )
        if not checkpoint.get("patch_dropout_trained"):
            # Recorded, and surfaced in the summary: evaluating a P=32 model at k=4 without
            # patch dropout is a distribution shift (spec B6). Not fatal -- the benchmark can
            # still measure relative regret -- but it must never be hidden.
            checkpoint["patch_dropout_warning"] = (
                "checkpoint does not record patch-dropout training; results at k < P are "
                "measured off-distribution (spec B6)"
            )

        train, test, site_by_episode = load_spatial_dataset(Path(args.cache_dir), patches_per_camera)
        from torch.utils.data import DataLoader

        # Batches stay on the host and are moved per use. Materialising every window of the
        # shard on the GPU is tens of GiB of patch tokens and leaves the model no headroom;
        # the .npz shard is IO-bound, so workers are enabled for the reason Session 0 measured.
        loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=args.num_workers > 0,
        )
        train_loader = DataLoader(train, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test, shuffle=False, **loader_kwargs)

        # Head training reuses a bounded prefix: it needs the same windows every seed for a
        # paired comparison, and the full train split is not needed to fit three heads.
        def stream_head_batches():
            # Rebuilt per seed: the previous seed's host windows were released to free memory,
            # so the factory must re-read them rather than close over a dead list.
            for index, batch in enumerate(train_loader):
                if index >= args.head_batches:
                    break
                yield move_to_device(batch, device)

        # Evaluation batches are held on the host, not the GPU: the whole test split as patch
        # tokens is tens of GiB. Each pass re-uploads one batch at a time, so peak memory is a
        # single batch and every pass sees the identical windows in the identical order, which
        # is what makes the paired tests and the per-seed comparison valid.
        def build_eval_batches():
            batches, total = [], 0
            for batch in test_loader:
                if args.eval_windows and total >= args.eval_windows:
                    break
                batches.append(batch)
                total += batch["context_state"].shape[0]
            return batches

        eval_host = build_eval_batches()

        def eval_batches():
            """A fresh device-side view of the evaluation windows, one batch resident at a time.

            Re-read from the loader on every call, so a seed that released its host-side copy
            does not leave later passes iterating a dead list.
            """
            for batch in build_eval_batches():
                yield move_to_device(batch, device)

    results = []
    all_rows: List[dict] = []
    # Seed provenance, recorded so a reader can tell distinct replicates from duplicates.
    # Remediation item 1 of the Session 5 audit: an artefact that does not record what each
    # seed actually loaded cannot be checked for the duplicate-seed failure mode after the
    # fact, and that failure invalidated Session 5's primary endpoint.
    seed_provenance: List[dict] = []
    backbone_sha = ""
    if checkpoint.get("loaded"):
        digest = hashlib.sha256()
        with open(checkpoint["path"], "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        backbone_sha = digest.hexdigest()

    # Synthetic runs take a single seed: random fixtures have no replicate structure to
    # measure, and pretending otherwise would exercise the duplicate-seed check on noise.
    for seed in range(1 if args.synthetic else args.num_seeds):
        torch.manual_seed(args.seed + seed)
        model = build_spatial_model(args, patches_per_camera, device, width, token_dim)
        loaded_from = "synthetic (random init, not evidence)"
        if not args.synthetic and checkpoint.get("loaded"):
            state = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(state.get("model_state_dict", state), strict=False)
            # One shared, immutable backbone across seeds: the heads are what is retrained
            # per seed. Recording that explicitly prevents a later reader from assuming the
            # backbone was re-trained per seed, or that seeds differ only by RNG.
            loaded_from = f"{checkpoint['path']} (shared across seeds; heads retrained per seed)"
            print(f"  seed {seed}: loaded backbone {loaded_from}")
            print(f"           missing={len(missing)} unexpected={len(unexpected)} keys")

        trained = train_heads(model, stream_head_batches, args, token_dim, patches_per_camera, device)
        curvature_head, heads = trained["curvature_head"], trained["heads"]

        sites = None
        if site_by_episode is not None:
            # Map scored windows back to sites from the host-side batches, so the site label
            # order matches the evaluation order exactly; falls back to per-window unknown when
            # the dataset has no episode ids (the bootstrap then degrades to window-level).
            sites = []
            for batch in eval_host:
                ids = batch.get("episode_id")
                sites.extend([site_by_episode.get(e, "unknown") for e in ids] if ids else ["unknown"] * batch["context_state"].shape[0])

        head_state = {
            name: {
                "sha256": hashlib.sha256(
                    b"".join(t.detach().cpu().numpy().tobytes() for t in module.state_dict().values())
                ).hexdigest()
            }
            for name, module in (
                ("curvature", curvature_head),
                *[(k, v) for k, v in heads.items() if isinstance(v, torch.nn.Module)],
            )
        }
        seed_provenance.append({
            "seed": seed,
            "torch_seed": args.seed + seed,
            "backbone_path": loaded_from,
            "backbone_sha256": backbone_sha,
            "head_sha256": head_state,
            "head_steps": trained["steps"],
            "head_final_loss": trained["final_loss"],
        })

        for k_cam in (k for total, k in BUDGETS if k <= patches_per_camera):
            budget_result = evaluate(model, curvature_head, heads, eval_batches(), args, k_cam,
                                      patches_per_camera, token_dim, sites, device=device)
            budget_result["seed"] = seed
            for row in budget_result.pop("rows", []):
                row["seed"] = seed
                all_rows.append(row)
            results.append(budget_result)

        # The diagnostics run after head training and scoring have built and released many
        # graphs. Free the cached blocks first: the exhaustive search is the most
        # memory-hungry step and should not compete with retained-but-unused segments.
        if device.type == "cuda":
            torch.cuda.empty_cache()
        calibration = run_oracle_calibration(model, next(iter(eval_batches())), args, patches_per_camera)
        beta_rows = {
            f"k_total={k_cam * CAMERAS}": beta_sensitivity(
                model, curvature_head, heads, eval_batches(), args, k_cam, patches_per_camera, token_dim)
            for k_cam in (k for total, k in BUDGETS if k <= patches_per_camera)
        }
        # Free the cached allocator between seeds so the next seed's expansion does not
        # fragment against this one's. Host-side windows are re-read per seed by the
        # factories above rather than cached across the whole sweep.
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert_beta_zero_matches_curvature(beta_rows, results)

    duplicate_seeds = detect_duplicate_seed_metrics(seed_provenance)
    summary = {
        "mode": "synthetic" if args.synthetic else "real",
        "device": str(device),
        "run_id": args.run_id,
        "spec": "docs/plans/2026-10-03-session-6a-spatial-selection-spec.md",
        "patches_per_camera": patches_per_camera,
        "total_patches": patches_per_camera * CAMERAS,
        "budgets": {f"k={r['k_total']}": {"k_per_camera": r["k_per_camera"], "seed": r["seed"]} for r in results},
        "beta": args.beta,
        "teacher_checkpoint": checkpoint,
        "backbone_sha256": backbone_sha,
        "seed_provenance": seed_provenance,
        "duplicate_seed_check": duplicate_seeds,
        "head_training": {"steps": trained["steps"], "final_loss": trained["final_loss"],
                          "parameter_counts": trained["parameter_counts"]},
        "policy_tiers": {"heuristics": list(HEURISTIC_POLICIES), "deployable": list(DEPLOYABLE_POLICIES),
                         "diagnostic": list(DIAGNOSTIC_POLICIES)},
        "by_budget": results,
        "beta_sensitivity": beta_rows,
        "greedy_oracle_calibration": calibration,
        "exit_gate": evaluate_exit_gate(results, synthetic=args.synthetic),
        # Popped by main() and written as parquet; kept out of the JSON so the summary stays
        # readable and small.
        "per_window_rows": all_rows,
        "caveats": [
            "Latency is deliberately not claimed here; wall-clock token selection is Session 6C (spec Q4).",
            "grounded epistemic VOI costs one rollout per patch, unlike the O(1) cotangent scorers.",
            "A checkpoint not trained with patch dropout is off-distribution at k < P (spec B6).",
            "Regret is measured against a greedy rollout *reference*, not a proven optimum; "
            "it is a lower bound and may be negative where a policy beats greedy.",
            "Deployable scorers need one encode per patch (latent perturbation) and, for "
            "belief-space VOI, one rollout per patch; only the cotangent term is O(1).",
        ],
    }
    return summary


def write_report(summary: Mapping, path: Path) -> None:
    """Render the Markdown report from the same numbers written to the summary JSON."""
    lines = [
        "# Session 6A: Fixed-Budget Spatial Patch Selection",
        "",
        f"**Mode:** `{summary['mode']}` · **Device:** `{summary['device']}` · "
        f"**Run:** `{summary['run_id']}` · **Patches:** `{summary['total_patches']}` "
        f"(`{summary['patches_per_camera']}` per camera) · **beta:** `{summary['beta']}`",
        "",
    ]
    if summary["mode"] == "synthetic":
        lines += ["> **Synthetic fixtures — not evidence.** Random tensors, no held-out data.", ""]

    for result in summary["by_budget"]:
        lines += [
            f"## Budget k = {result['k_total']} ({result['k_per_camera']} per camera)",
            "",
            "| Policy | Mean regret | Median | 10% trimmed mean | Std |",
            "|---|---:|---:|---:|---:|",
        ]
        for policy, stats in result["per_policy"].items():
            lines.append(
                f"| `{policy}` | {stats['mean_regret']:.5f} | {stats['median_regret']:.5f} | "
                f"{stats['trimmed_mean_regret_10pct']:.5f} | {stats['std_regret']:.5f} |"
            )
        diagnostics = result["diagnostics"]
        lines += [
            "",
            f"- Additivity R^2: `{diagnostics['additivity_r2']['r2']:.4f}` over "
            f"`{diagnostics['additivity_r2']['n']}` selections",
            f"- Submodularity violation rate: `{diagnostics['submodularity']['rate']:.4f}` "
            f"(`{diagnostics['submodularity']['violations']}` / `{diagnostics['submodularity']['comparisons']}`)",
            f"- Distillation gap (exact reference − VOI trimmed mean): `{diagnostics['distillation_gap']:.5f}`",
        ]
        for name, test in result["comparisons"].items():
            lines.append(f"- Paired Wilcoxon `{name}`: statistic `{test['statistic']:.1f}`, "
                         f"p = `{test['p_value']:.4g}` (n = `{test['n']}`)")
        lines.append("")

    if summary.get("beta_sensitivity"):
        lines += ["## Belief-Space Weight Sensitivity (spec B2)", "",
                  "| beta | Mean regret | 10% trimmed mean |", "|---:|---:|---:|"]
        for budget, rows in summary["beta_sensitivity"].items():
            lines.append(f"| **{budget}** | | |")
            for label, stats in rows.items():
                lines.append(f"| `{label}` | {stats['mean_regret']:.5f} | {stats['trimmed_mean_regret_10pct']:.5f} |")
        lines += ["", "`beta = 0` must equal `second_order_curvature`; negative `beta` is the sign-flip control.", ""]

    if summary.get("greedy_oracle_calibration"):
        lines += ["## Greedy Oracle Calibration (spec B5)", ""]
        for entry in summary["greedy_oracle_calibration"]:
            lines.append(
                f"- k_cam = {entry['k_per_camera']} over {entry['n_windows']} windows: mean excess "
                f"objective vs exhaustive `{entry['mean_excess_objective']:.3e}`, max "
                f"`{entry['max_excess_objective']:.3e}`, optimal on "
                f"`{entry['fraction_optimal'] * 100:.1f}%` of windows"
            )
        lines.append("")

    gate = summary["exit_gate"]
    lines += ["## Exit Gate (pre-registered, spec section 5)", ""]
    if not gate.get("evaluated", True):
        lines += [f"> **{gate['reason']}.**", ""]
    for budget, entry in gate.get("per_budget", {}).items():
        lines.append(f"### {budget}")
        lines.append("")
        lines.append(
            f"- Belief-space VOI trimmed mean regret: `{entry['belief_space_voi_trimmed_mean_regret']:.5f}`"
        )
        for baseline, value in entry["advantage_over_baseline"].items():
            rendered = "n/a" if np.isnan(value) else f"{value * 100:.2f}%"
            lines.append(f"- Advantage over `{baseline}`: `{rendered}` "
                         f"(threshold `{gate['thresholds']['advantage'] * 100:.1f}%`)")
        lines += [
            f"- Criterion 1 (primary advantage): "
            f"**{'MET' if entry['criterion_1_primary_advantage_met'] else 'NOT MET'}**",
            f"- Criterion 2 (non-inferiority vs privileged critic): "
            f"**{'MET' if entry['criterion_2_non_inferiority_met'] else 'NOT MET'}**",
            f"- Wilcoxon p (VOI vs curvature): `{entry['wilcoxon_p_vs_curvature']:.4g}`",
            "",
        ]
    lines += [
        f"- Criterion 1 across all budgets: "
        f"**{'MET' if gate['criterion_1_primary_advantage_met'] else 'NOT MET'}**",
        f"- Criterion 2 across all budgets: "
        f"**{'MET' if gate['criterion_2_non_inferiority_met'] else 'NOT MET'}**",
        "",
        f"**Branch: `{gate['branch']}`**",
        "",
        "## Caveats",
        "",
    ]
    lines += [f"- {caveat}" for caveat in summary["caveats"]]
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 70)
    print("Session 6A: Fixed-Budget Spatial Patch Selection")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    summary = run(args, device)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = summary.pop("per_window_rows", None)
    atomic_write_json(output_dir / "spatial_selection_summary.json", summary)
    write_report(summary, output_dir / "spatial_selection_report.md")

    # Per-window regrets, committed so the paired and clustered inference the exit gate rests
    # on stays re-runnable. Session 5 could not test its own primary endpoint because only
    # aggregates were committed.
    if rows:
        try:
            import pandas as pd

            frame = pd.DataFrame(rows)
            frame.to_parquet(output_dir / "spatial_selection_per_window.parquet", index=False)
            print(f"Saved {len(frame)} per-window rows to "
                  f"{output_dir / 'spatial_selection_per_window.parquet'}")
        except ImportError:
            path = output_dir / "spatial_selection_per_window.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"pandas unavailable; saved {len(rows)} per-window rows to {path}")

    check = summary["duplicate_seed_check"]
    print(f"Seeds: {check['n_seeds']} run, {check['n_distinct_head_parameter_sets']} distinct "
          f"head parameter sets -> distinct replicates: {check['seeds_are_distinct_replicates']}")
    print(f"Saved summary to {output_dir / 'spatial_selection_summary.json'}")
    print(f"Saved report to {output_dir / 'spatial_selection_report.md'}")
    print(f"Exit gate: {summary['exit_gate']['branch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())