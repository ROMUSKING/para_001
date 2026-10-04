#!/usr/bin/env python3
"""scripts/diagnose_spatial_selection_bottleneck.py

Session 6A diagnostic: **what actually limits fixed-budget spatial selection?**

Session 6A reached its pre-registered negative branch: belief-space VOI lost to
``early_feature_norm``, a raw DINOv2 patch-token L2 norm, at every budget. Two candidate
explanations were left open, and they imply completely different next steps:

1. **The distilled co-state is bad.** ``exact_costate_reference`` reaches trimmed-mean regret
   0.00007 against VOI's 0.00598, so roughly 85x of the loss sits somewhere between the autograd
   co-state and the distilled ``lambda_hat``.
2. **Additive per-patch scoring is the wrong functional shape.** Additivity ``R^2`` was ~4e-05
   over 3,456 selections and submodularity violations rose with budget, so the surrogate
   ``sum_{p in S} s_p`` may simply not be able to predict ``J(z_S)`` for *any* scorer.

These are distinguishable, and this script distinguishes them:

* **Ranking fidelity.** Spearman correlation between each scorer's per-patch gains and the
  **exact** marginal gains ``J(z_S) - J(z_{S u p})``, per window. A scorer that ranks well but
  selects badly is being hurt by non-additivity; a scorer that ranks badly is the problem.
* **Ceiling under the same surrogate.** Regret of selecting by the exact gains themselves. If
  exact-gain top-k is near the greedy reference, the surrogate family can express a good policy and
  the scorers are at fault. If even exact-gain top-k loses badly to ``early_feature_norm``, the
  additive form is the ceiling and no scorer can be rescued.
* **Distillation sweep.** ``CurvatureCostateEstimator`` capacity and training length, reporting
  both rank correlation and realised regret, so the ``lambda_hat`` -> regret curve is visible.

Every number here is a *diagnostic*, not a gate. Nothing in this file decides an exit criterion.

Usage::

    python scripts/diagnose_spatial_selection_bottleneck.py \\
        --cache-dir /content/spatial_cache_e3_1 --drive-root /content/local_runs \\
        --run-id spatial_patches_20261003 --windows 256
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_DIR = Path(__file__).resolve().parent.parent
for candidate in (REPO_DIR / "src", Path("/content/para_001/src"), Path("/content/src")):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO_DIR / "scripts"))

from adjointrwm.allocators import CurvatureCostateEstimator  # noqa: E402
from adjointrwm.data.windows import stratified_window_indices  # noqa: E402
from adjointrwm.io import atomic_write_json  # noqa: E402
from adjointrwm.spatial_selection import (  # noqa: E402
    BUDGETS,
    CAMERAS,
    camera_of_patch,
    cotangent_bundle,
    curvature_scores,
    early_feature_norm_scores,
    epistemic_variance_reduction,
    greedy_oracle_masks,
    latent_patch_perturbations,
    objective_at_masks,
    patch_conditioned_curvature_scores,
    select_topk_per_camera,
    trimmed_mean,
)

from benchmark_spatial_patch_selection import (  # noqa: E402
    build_spatial_model,
    exact_marginal_gains,
    load_spatial_splits,
    move_to_device,
    train_heads,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Diagnose the Session 6A selection bottleneck")
    parser.add_argument("--cache-dir", type=str, default="/content/spatial_cache_e3_1")
    parser.add_argument("--drive-root", type=str, default="/content/local_runs")
    parser.add_argument("--run-id", type=str, default="spatial_patches_20261003")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks/spatial_bottleneck")
    parser.add_argument("--windows", type=int, default=256, help="held-out windows, stratified")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head-steps", type=int, default=300)
    parser.add_argument("--head-batches", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    # Distillation sweep: (hidden multiplier, steps). CurvatureCostateEstimator's width is set
    # directly, so the multiplier is relative to the production width.
    parser.add_argument("--sweep-hidden", type=str, default="256,512,1024")
    parser.add_argument("--sweep-steps", type=str, default="100,300")
    parser.add_argument("--train-conditioned", action="store_true",
                        help="train the WS2 candidate-conditioned heads (cells C/D) alongside; "
                             "off by default so the diagnostic reproduces the earlier sweep exactly")
    parser.add_argument("--ranking-margin-scales", type=str, default="1.0",
                        help="margin_scale values for the WS1b ranking head, comma-separated. "
                             "Recorded per row, not tuned: exact-gain differences are O(1e-4), so "
                             "scale sets the demanded score separation and a single scale could "
                             "under- or over-demand. Same default 1.0 the critics use.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Rank correlation without scipy
# ---------------------------------------------------------------------------


def topk_overlap(scores: torch.Tensor, truth: torch.Tensor, k_cam: int) -> torch.Tensor:
    """Fraction of the true top-``k`` patches that the scorer also selects, per row.

    Mean Spearman over 32 patches measures a *global* ordering and can look poor while the
    handful of patches that actually get selected are right. This measures the selection, which
    is what regret depends on.
    """
    index = camera_of_patch(scores.shape[1] // CAMERAS, CAMERAS)
    chosen, ideal = [], []
    for camera in range(CAMERAS):
        columns = torch.nonzero(index == camera, as_tuple=False).flatten()
        chosen.append(scores[:, columns].topk(k_cam, dim=1).indices)
        ideal.append(truth[:, columns].topk(k_cam, dim=1).indices)
    chosen = torch.cat(chosen, dim=1)
    ideal = torch.cat(ideal, dim=1)
    hits = (chosen.unsqueeze(-1) == ideal.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
    return hits.float() / ideal.shape[-1]


def _rank(values: torch.Tensor) -> torch.Tensor:
    """Tie-averaged ranks along the last axis, ascending from 0.

    Ties must share their average rank. A plain ``argsort`` assigns distinct ranks by position,
    which makes a *constant* row look perfectly ordered and so yields a spurious ``rho = 1`` --
    the opposite of the "no rank information" it should report.

    """
    n = values.shape[-1]
    order = values.argsort(dim=-1)
    ordered = values.gather(-1, order)
    # Group id per position, incrementing whenever the sorted value changes.
    starts_group = torch.ones_like(ordered, dtype=torch.long)
    starts_group[..., 1:] = (ordered[..., 1:] != ordered[..., :-1]).long()
    group = torch.cumsum(starts_group, dim=-1) - 1

    # Group sizes and offsets are indexed **by group id**, not by element position. Cumulating
    # per-position sizes double-counts, because every element of a group carries that group's
    # size, which would spread one tied group across several apparent offsets and leave a
    # constant row looking ordered.
    sizes = torch.zeros(values.shape[0], n, dtype=torch.long, device=values.device)
    sizes.scatter_add_(1, group, torch.ones_like(group))
    offsets = torch.cumsum(sizes, dim=1) - sizes
    average = (offsets.gather(1, group) + (sizes.gather(1, group) - 1) / 2.0).to(torch.float64)

    ranks = torch.empty_like(average)
    ranks.scatter_(-1, order, average)
    return ranks


def spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise Spearman correlation; returns ``[B]``. Ranking only, so scale-free."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    a, b = _rank(a.double()), _rank(b.double())
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    numerator = (a * b).sum(dim=-1)
    denominator = a.norm(dim=-1) * b.norm(dim=-1)
    # A window where one signal is constant has no rank correlation to report.
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-12),
                       torch.full_like(numerator, float("nan")))


# ---------------------------------------------------------------------------
# Regret of a fixed scoring rule
# ---------------------------------------------------------------------------


def regret_of_scores(
    model,
    batch: Mapping[str, torch.Tensor],
    gains: torch.Tensor,
    k_cam: int,
    patches_per_camera: int,
    oracle_value: torch.Tensor,
) -> torch.Tensor:
    """Regret of selecting ``topk`` patches by ``gains``, per window."""
    mask = select_topk_per_camera(gains, k_cam, camera_of_patch(patches_per_camera, CAMERAS), CAMERAS)
    with torch.no_grad():
        value = objective_at_masks(model, batch, mask.unsqueeze(1))[:, 0]
    return value - oracle_value


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def load_backbone(args, patches_per_camera: int, device: torch.device, width: int):
    path = Path(args.drive_root) / "runs" / args.run_id / "best_spatial.pt"
    if not path.exists():
        raise SystemExit(
            f"no trained backbone at {path}. Run scripts/train_spatial_patch_dropout_backbone.py "
            "first; a randomly initialised world model would make every number below meaningless."
        )
    model = build_spatial_model(args, patches_per_camera, device, width, 384)
    state = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state.get("model_state_dict", state), strict=False)
    return model, path, {
        "path": str(path),
        "missing_keys": sorted(missing)[:8],
        "unexpected_keys": sorted(unexpected)[:8],
        "patch_dropout_trained": bool(state.get("patch_dropout_trained", False)),
    }


def run(args, device: torch.device) -> dict:
    patches_per_camera = 16
    patches = patches_per_camera * CAMERAS
    token_dim = 384

    model, backbone_path, backbone_meta = load_backbone(args, patches_per_camera, device, args.width)
    train_set, _val, test_set, site_by_episode = load_spatial_splits(Path(args.cache_dir), patches_per_camera)

    indices = stratified_window_indices(
        test_set.episode_ids(), site_by_episode, args.windows, seed=args.seed
    )
    test_subset = Subset(test_set, indices.tolist())
    # Subset does not forward dataset methods, so the episode ids of the selected windows are
    # read from the parent by index rather than from the subset.
    parent_ids = np.asarray(test_set.episode_ids())
    episodes_seen = {str(e) for e in parent_ids[indices].tolist()}
    sites_seen = {site_by_episode.get(e, "unknown") for e in episodes_seen}
    print(f"diagnostic windows: {len(indices)} over {len(episodes_seen)} episodes, "
          f"{len(sites_seen)} sites", flush=True)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"), persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_subset, shuffle=False, **loader_kwargs)

    def stream_head_batches():
        for index, batch in enumerate(train_loader):
            if index >= args.head_batches:
                break
            yield move_to_device(batch, device)

    # ``train_heads`` is the Session 6A training loop and reads ``args.max_steps``; this
    # diagnostic names its own budget ``--head-steps``, so the namespace is adapted rather than
    # duplicating the loop.
    production_args = argparse.Namespace(**vars(args))
    production_args.max_steps = args.head_steps

    # The production heads, so the fidelity numbers are comparable with Session 6A's own run.
    trained = train_heads(model, stream_head_batches, production_args, token_dim,
                          patches_per_camera, device)
    curvature_head = trained["curvature_head"]

    hidden_choices = [int(h) for h in args.sweep_hidden.split(",") if h.strip()]
    step_choices = [int(s) for s in args.sweep_steps.split(",") if s.strip()]
    margin_choices = [float(m) for m in args.ranking_margin_scales.split(",") if m.strip()]

    sweep: List[dict] = []
    fidelity: Dict[str, Dict[str, list]] = {}
    started = time.time()

    for hidden in hidden_choices:
        for steps in step_choices:
            for margin_scale in margin_choices:
                torch.manual_seed(args.seed)
                head = CurvatureCostateEstimator(model.d_model, hidden=hidden).to(device)
                sweep_args = argparse.Namespace(**vars(args))
                sweep_args.max_steps = steps
                trained_sweep = train_heads(
                    model, stream_head_batches, sweep_args, token_dim, patches_per_camera, device,
                    curvature_head=head, ranking_margin_scale=margin_scale,
                    train_conditioned=args.train_conditioned,
                )
                head = trained_sweep["curvature_head"]
                # WS1a/WS1b ablations ride along in the same training loop (identical inputs,
                # steps, data; only the objective differs). They are evaluated, not just trained.
                head_cosine_only = trained_sweep.get("curvature_head_cosine_only")
                head_ranking = trained_sweep.get("curvature_head_ranking")
                # WS2 cells C/D. Absent unless --train-conditioned (default off).
                head_cond_composite = trained_sweep.get("conditioned_head_composite")
                head_cond_ranking = trained_sweep.get("conditioned_head_ranking")
                print(f"  sweep hidden={hidden} steps={steps} "
                      f"params={sum(p.numel() for p in head.parameters())} "
                      f"margin_scale={trained_sweep.get('ranking_margin_scale')}", flush=True)

                correlations: Dict[str, List[float]] = {}
                # Top-k overlap against the true gains, per budget: the *selection* statistic, which
                # mean rank correlation over 32 patches can misrepresent.
                overlaps: Dict[str, Dict[str, List[float]]] = {f"k_cam={k}": {} for _t, k in BUDGETS}
                # Keyed by per-camera budget, which is what the loop below selects with; BUDGETS holds
                # (total, per-camera) pairs, so the key is the second element.
                regrets: Dict[int, Dict[str, List[float]]] = {k_cam: {} for _total, k_cam in BUDGETS}

                for batch_index, batch in enumerate(test_loader):
                    batch = move_to_device(batch, device)
                    windows = batch["context_state"].shape[0]
                    with torch.no_grad():
                        exact = exact_marginal_gains(model, batch)
                        budget_fraction = torch.full((windows,), 1.0 / patches_per_camera, device=device)
                        horizon_fraction = torch.ones_like(budget_fraction)
                        latent = model.encode_context(
                            batch["context_visual"], batch["context_state"], batch["context_action"]
                        ).detach()
                        costate_hat, hessian_hat = head(latent, budget_fraction, horizon_fraction)
                        # WS1 ablation heads see the same decision-time inputs; their outputs are
                        # scored through the identical curvature path, so only training differs.
                        costate_co, hessian_co = (
                            head_cosine_only(latent, budget_fraction, horizon_fraction)
                            if head_cosine_only is not None else (None, None))
                        costate_r, hessian_r = (
                            head_ranking(latent, budget_fraction, horizon_fraction)
                            if head_ranking is not None else (None, None))
                        # WS2 cells C/D read the candidate's own embedding; same decision-time
                        # inputs as PatchRankingCritic, so Tier 1 status is preserved.
                        cond_composite, cond_ranking = (None, None), (None, None)
                        if head_cond_composite is not None:
                            cond_composite = head_cond_composite(
                                latent, batch["context_visual"], budget_fraction, horizon_fraction)
                        if head_cond_ranking is not None:
                            cond_ranking = head_cond_ranking(
                                latent, batch["context_visual"], budget_fraction, horizon_fraction)
                        delta_z = latent_patch_perturbations(model, batch)
                        delta_sigma, precision = epistemic_variance_reduction(model, batch)

                    # The autograd reference needs a backward pass, so it is outside no_grad.
                    bundle = cotangent_bundle(model, batch)

                    zero_hessian = torch.zeros_like(hessian_hat)
                    scorers = {
                        # Distilled: what the deployable policies actually use.
                        "curvature_distilled": curvature_scores(costate_hat, hessian_hat, delta_z),
                        # Autograd cotangent sensitivity: the Tier 2 reference.
                        "first_order_exact": bundle.first_order_gains,
                        # The epistemic term ALONE, which Session 6A could not measure because it was
                        # 9.1e-06 of the curvature term and so could not reorder a combined score.
                        "delta_sigma_only": (precision.unsqueeze(1) * delta_sigma.pow(2)).sum(-1),
                        "patch_norm": early_feature_norm_scores(batch["context_visual"]),
                        # The ceiling of the additive surrogate: rank by the true marginal gains.
                        "exact_gains": exact,
                        # --- Privilege ablation (rotating critic `cline`, P0) -------------------
                        # The "~110x distillation gap" compares distilled-vs-exact AND
                        # second-order-vs-first-order at the same time, so it cannot say whether
                        # lambda_hat, H_hat, or the functional form is at fault. These three isolate
                        # them by changing exactly one factor each.
                        # distilled lambda, distilled H, first-order term only -> isolates lambda_hat
                        "first_order_distilled": curvature_scores(costate_hat, zero_hessian, delta_z),
                        # exact lambda, distilled H -> isolates lambda_hat with the curvature kept
                        "curvature_exact_costate": curvature_scores(bundle.exact_costate, hessian_hat, delta_z),
                        # exact lambda, no H -> the pure autograd reference in scorer form
                        "first_order_exact_costate": curvature_scores(bundle.exact_costate, zero_hessian, delta_z),
                        # --- WS1a/WS1b (codex review): same class, same inputs, same data --------
                        # as the production head; only the training objective differs. Reported
                        # alongside, never gated: a failure here does NOT license an input-
                        # sufficiency conclusion (capacity, optimisation, singleton labels remain).
                        **({"curvature_cosine_only": curvature_scores(costate_co, hessian_co, delta_z)}
                           if costate_co is not None else {}),
                        **({"curvature_ranking": curvature_scores(costate_r, hessian_r, delta_z)}
                           if costate_r is not None else {}),
                        # --- WS2 cells C/D: per-patch co-states, per-patch scorer ---------------
                        **({"conditioned_composite": patch_conditioned_curvature_scores(
                            cond_composite[0], cond_composite[1], delta_z)}
                           if cond_composite[0] is not None else {}),
                        **({"conditioned_ranking": patch_conditioned_curvature_scores(
                            cond_ranking[0], cond_ranking[1], delta_z)}
                           if cond_ranking[0] is not None else {}),
                    }

                    for name, gains in scorers.items():
                        values = spearman(gains.float(), exact.float())
                        correlations.setdefault(name, []).extend(values.detach().cpu().tolist())

                    # Top-k overlap, not just mean rank correlation. Mean rho over 32 patches is a
                    # global ordering statistic and can look poor while the *selected* set is right,
                    # which is exactly the regime the earlier `patch_norm` result sits in.
                    for _total, k_cam in BUDGETS:
                        truth = exact
                        for name, gains in scorers.items():
                            overlap = topk_overlap(gains.detach(), truth, k_cam)
                            overlaps.setdefault(f"k_cam={k_cam}", {}).setdefault(name, []).extend(
                                overlap.detach().cpu().tolist()
                            )

                    for _total, k_cam in BUDGETS:
                        with torch.no_grad():
                            oracle = objective_at_masks(
                                model, batch, greedy_oracle_masks(
                                    model, batch, k_cam, camera_of_patch(patches_per_camera, CAMERAS)
                                )[0].unsqueeze(1))[:, 0]
                        for name, gains in scorers.items():
                            entry = regrets[k_cam].setdefault(name, [])
                            entry.extend(
                                regret_of_scores(model, batch, gains.detach(), k_cam,
                                                  patches_per_camera, oracle).detach().cpu().tolist()
                            )
                    del batch
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                row = {
                    "hidden": hidden,
                    "steps": steps,
                    "ranking_margin_scale": margin_scale,
                    "parameters": int(sum(p.numel() for p in head.parameters())),
                    "rank_correlation_vs_exact_gains": {
                        name: float(np.nanmean(values)) for name, values in correlations.items()
                    },
                    "rank_correlation_n": {name: int(np.isfinite(values).sum())
                                           for name, values in correlations.items()},
                    "topk_overlap_vs_exact_gains": {
                        budget: {
                            # Chance overlap under a symmetric per-camera budget of k_cam of P_cam.
                            name: {
                                "mean": float(np.mean(values)),
                                # k_cam / patches_per_camera, the expected value under random choice.
                                "chance": k_cam / patches_per_camera,
                                "n": int(len(values)),
                            }
                            for name, values in per_scorer.items()
                        }
                        for budget, per_scorer in overlaps.items()
                        for k_cam in [int(budget.split("=")[1])]
                    },
                    "regret": {},
                    "elapsed_s": round(time.time() - started, 1),
                }
                for k_cam, per_policy in regrets.items():
                    row["regret"][f"k_cam={k_cam}"] = {
                        name: {
                            "mean": float(np.mean(values)),
                            "trimmed_mean": trimmed_mean(np.asarray(values, dtype=float)),
                            "n": int(len(values)),
                        }
                        for name, values in per_policy.items()
                    }
                sweep.append(row)
                print(f"    rho(exact) distilled="
                      f"{row['rank_correlation_vs_exact_gains']['curvature_distilled']:.4f} "
                      f"autograd={row['rank_correlation_vs_exact_gains']['first_order_exact']:.4f} "
                      f"delta_sigma={row['rank_correlation_vs_exact_gains']['delta_sigma_only']:.4f} "
                      f"norm={row['rank_correlation_vs_exact_gains']['patch_norm']:.4f}", flush=True)

    production = next((r for r in sweep if r["hidden"] == model.d_model and r["steps"] == args.head_steps),
                      sweep[-1] if sweep else None)

    return {
        "benchmark": "session_6a_selection_bottleneck",
        "mode": "real",
        "device": str(device),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backbone": backbone_meta,
        "backbone_path": str(backbone_path),
        "windows": int(len(indices)),
        "episodes": sorted(episodes_seen),
        "num_episodes": len(episodes_seen),
        "sites": sorted(sites_seen),
        "num_sites": len(sites_seen),
        "seed": args.seed,
        "sweep": sweep,
        "production_point": production,
        "reading_guide": {
            "rank_correlation_vs_exact_gains":
                "Spearman rho between a scorer's per-patch gains and the exact marginal gains "
                "J(z_S) - J(z_{S u p}). High rho means the scorer orders patches correctly.",
            "regret":
                "Regret of top-k selection by that scorer, against the greedy rollout reference. "
                "Compare 'exact_gains' (the additive surrogate's ceiling) with 'patch_norm': if "
                "exact_gains also loses badly, the additive form is the ceiling and no scorer can "
                "be rescued by better training.",
            "delta_sigma_only":
                "The epistemic term alone. Session 6A could not test it because it was 9.1e-06 of "
                "the curvature term; this measures whether it carries any ranking signal at all.",
        },
        "caveats": [
            "Diagnostic only: nothing here evaluates a pre-registered exit gate.",
            "Spearman is computed per window across the 32 patches, then averaged; windows where "
            "either signal is constant yield NaN and are dropped from the mean.",
            "The greedy reference is not a proven optimum, so every regret here is a lower bound.",
        ],
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 70)
    print("Session 6A diagnostic: what limits fixed-budget spatial selection?")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    summary = run(args, device)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "spatial_bottleneck_summary.json", summary)
    print(f"Saved summary to {output_dir / 'spatial_bottleneck_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())