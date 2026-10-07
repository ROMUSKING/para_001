#!/usr/bin/env python3
"""scripts/diagnose_mechanism_check.py

Session 6C mechanism check (review work-package item 3): does exact derivative structure
predict measured joint-mask outcomes? Distill it more accurately ONLY on success.

Design (peer-reviewed, codex, milestone ``Session-6C-mechanism-design``):

- The distilled head trains on the TRAIN split (production protocol). Evaluation windows
  come from the test split and never overlap training batches — a test-fitted head would
  make any win an in-sample diagnostic, not held-out evidence.
- Pair-latent additivity is MEASURED, not assumed: the Taylor premise
  ε ≈ −δ_pᵀHδ_q requires δ_{pq} ≈ δ_p + δ_q, checked per window as a relative residual.
- Four ranking arms isolate the mechanism claim:
  A additive-distilled (first-order λ̂ scores summed — pure deployable baseline);
  B corrected-distilled (A plus the distilled-Hessian pair correction);
  C oracle-ceiling (measured singleton gains plus the same Hessian correction);
  D measured best pair (hindsight reference, never a policy).
  A B-over-A win supports Hessian-correction value *within the distilled world*; C bounds
  it; a tie under the near-ceiling additive baseline (ρ 0.99) shows nothing either way.
- Per-window correlations averaged (pairs repeat across windows — pooled correlation would
  let between-window effects dominate), top-k overlap emitted, episode-clustered
  uncertainty on the decision comparison.

For each window, with all quantities at the common null-selection expansion point and the
same mask-effect definitions as the labels (zero-content masking, positions kept):

- measured: G({p}), G({p,q}) and ε = G({p,q}) − G({p}) − G({q}) from actual no-grad rollouts
  over the frozen 200-pair manifest;
- predicted: second-order pair correction −δ_pᵀHδ_q (diagonal H) with H = distilled Ĥ.

Usage::

    python scripts/diagnose_mechanism_check.py \\
        --cache-dir /content/spatial_cache_e3_1 --drive-root /content/local_runs \\
        --run-id spatial_patches_20261003 --output-dir results/benchmarks/mechanism_check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_DIR = Path(__file__).resolve().parent.parent
for candidate in (REPO_DIR / "src", Path("/content/para_001/src")):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO_DIR / "scripts"))

from adjointrwm.io import atomic_write_json  # noqa: E402
from adjointrwm.spatial_selection import (  # noqa: E402
    CAMERAS,
    PATCHES_PER_CAMERA,
    TOTAL_PATCHES,
    cotangent_bundle,
    curvature_scores,
    latent_patch_perturbations,
    objective_at_masks,
    site_clustered_bootstrap_ci,
    wilcoxon_signed_rank,
)  # noqa: E402

from benchmark_spatial_patch_selection import (  # noqa: E402
    build_spatial_model,
    load_spatial_splits,
    move_to_device,
    train_heads,
)
from diagnose_pair_interactions import sample_pairs  # noqa: E402
from diagnose_spatial_selection_bottleneck import spearman  # noqa: E402


def pair_topk_overlap(scores: torch.Tensor, truth: torch.Tensor, k: int) -> torch.Tensor:
    """Fraction of the true top-``k`` pairs also ranked top-``k`` by the scorer, per row.

    Unlike :func:`topk_overlap` (per-camera patch geometry), pairs are an unstructured
    candidate list, so this is a plain set overlap. Pure function — unit-tested.
    """
    if k <= 0 or k > scores.shape[1]:
        raise ValueError(f"k={k} outside [1, {scores.shape[1]}]")
    chosen = scores.topk(k, dim=1).indices
    ideal = truth.topk(k, dim=1).indices
    hits = (chosen.unsqueeze(-1) == ideal.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
    return hits.float() / k


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Session 6C mechanism check")
    parser.add_argument("--cache-dir", type=str, default="/content/spatial_cache_e3_1")
    parser.add_argument("--drive-root", type=str, default="/content/local_runs")
    parser.add_argument("--run-id", type=str, default="spatial_patches_20261003")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks/mechanism_check")
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--pair-seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--head-steps", type=int, default=300)
    parser.add_argument("--head-batches", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-pairs", type=int, default=64)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def second_order_pair_correction(delta: torch.Tensor, hessian: torch.Tensor,
                                 pairs: List[tuple]) -> torch.Tensor:
    """Predicted interaction −δ_pᵀHδ_q per pair, diagonal H.

    ``delta`` is ``[B, P, d]`` exact latent perturbations from the common null-selection
    expansion point; ``hessian`` is ``[B, d]`` (distilled diagonal). Returns ``[B, M]``.
    Pure function — unit-tested against hand computation.
    """
    if hessian.dim() != 2 or delta.dim() != 3:
        raise ValueError(f"expected hessian [B, d] and delta [B, P, d], got "
                         f"{tuple(hessian.shape)} and {tuple(delta.shape)}")
    out = []
    for a, b in pairs:
        out.append(-(hessian * delta[:, a, :] * delta[:, b, :]).sum(dim=-1))
    return torch.stack(out, dim=1)


def run(args, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    path = Path(args.drive_root) / "runs" / args.run_id / "best_spatial.pt"
    if not path.exists():
        raise SystemExit(f"missing teacher checkpoint at {path}: hard failure")
    model = build_spatial_model(args, PATCHES_PER_CAMERA, device, args.width, 384)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model.eval()

    manifest = sample_pairs(args.n_pairs, args.pair_seed)
    pairs = [(m["patch_a"], m["patch_b"]) for m in manifest]
    train_set, _val, test_set, site_by_episode = load_spatial_splits(
        Path(args.cache_dir), PATCHES_PER_CAMERA)
    from adjointrwm.data.windows import stratified_window_indices
    parent_ids = np.asarray(test_set.episode_ids())
    indices = stratified_window_indices(parent_ids, site_by_episode, args.windows, seed=args.seed)
    episodes_seen = sorted({str(e) for e in parent_ids[indices].tolist()})
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"),
                         persistent_workers=args.num_workers > 0)

    # Head training uses the TRAIN split (production protocol). The evaluation windows below
    # come from the test split and never overlap these batches: a test-fitted head would make
    # any mechanism win an in-sample diagnostic, not held-out evidence.
    train_loader = DataLoader(train_set, shuffle=False, **loader_kwargs)

    def stream_head_batches():
        for index, batch in enumerate(train_loader):
            if index >= args.head_batches:
                break
            yield move_to_device(batch, device)

    import argparse as _argparse
    head_args = _argparse.Namespace(**vars(args))
    head_args.max_steps = args.head_steps
    head_args.lr = args.lr
    trained = train_heads(model, stream_head_batches, head_args, 384,
                          PATCHES_PER_CAMERA, device)
    from adjointrwm.allocators import CurvatureCostateEstimator
    head = trained["curvature_head"]
    print(f"distilled head trained on TRAIN split ({trained['steps']} steps)", flush=True)

    test_loader = DataLoader(Subset(test_set, indices.tolist()), shuffle=False, **loader_kwargs)
    single_masks = torch.eye(TOTAL_PATCHES, device=device)
    pair_masks = torch.zeros(len(manifest), TOTAL_PATCHES, device=device)
    for row, entry in enumerate(manifest):
        pair_masks[row, entry["patch_a"]] = 1.0
        pair_masks[row, entry["patch_b"]] = 1.0

    rows: List[dict] = []
    for batch in test_loader:
        batch = move_to_device(batch, device)
        windows = batch["context_state"].shape[0]
        with torch.enable_grad():
            bundle = cotangent_bundle(model, batch)
            delta = latent_patch_perturbations(model, batch)
        budget = torch.full((windows,), 1.0 / PATCHES_PER_CAMERA, device=device)
        horizon = torch.ones_like(budget)
        with torch.no_grad():
            costate_hat, hessian_hat = head(bundle.latent.detach(), budget, horizon)
            # Pair-latent additivity: encode the pair masks and compare against the summed
            # singleton displacements. The Taylor premise stands or falls on this residual.
            base = objective_at_masks(
                model, batch, torch.zeros(windows, 1, TOTAL_PATCHES, device=device))[:, 0]
            singles = objective_at_masks(
                model, batch, single_masks.unsqueeze(0).expand(windows, -1, -1))
            pair_J, pair_latents = [], []
            for start in range(0, len(manifest), max(1, args.chunk_pairs)):
                piece = pair_masks[start:start + max(1, args.chunk_pairs)]
                expanded = piece.unsqueeze(0).expand(windows, -1, -1)
                pair_J.append(objective_at_masks(model, batch, expanded))
                visual = batch["context_visual"].unsqueeze(1).expand(
                    -1, piece.shape[0], *batch["context_visual"].shape[1:])
                masked = visual.reshape(windows * piece.shape[0], *visual.shape[2:])
                flat_mask = piece.unsqueeze(0).expand(windows, -1, -1).reshape(
                    windows * piece.shape[0], -1)
                from adjointrwm.spatial_selection import apply_patch_mask
                masked_visual = apply_patch_mask(
                    masked, flat_mask)
                candidates = piece.shape[0]
                lat = model.encode_context(
                    masked_visual,
                    batch["context_state"].unsqueeze(1).expand(
                        -1, candidates, *batch["context_state"].shape[1:]).reshape(
                        windows * candidates, *batch["context_state"].shape[1:]),
                    batch["context_action"].unsqueeze(1).expand(
                        -1, candidates, *batch["context_action"].shape[1:]).reshape(
                        windows * candidates, *batch["context_action"].shape[1:]))
                pair_latents.append(lat.reshape(windows, piece.shape[0], -1))
            pair_J = torch.cat(pair_J, dim=1)
            pair_latents = torch.cat(pair_latents, dim=1)
            latent_null = bundle.latent.detach()
            delta_pq = pair_latents - latent_null.unsqueeze(1)
            delta_sum = torch.stack([delta[:, a, :] + delta[:, b, :] for a, b in pairs],
                                    dim=1)
            additivity = (delta_pq - delta_sum).norm(dim=-1) / delta_sum.norm(
                dim=-1).clamp_min(1e-12)
            first_order = curvature_scores(bundle.exact_costate,
                                           torch.zeros_like(bundle.exact_costate), delta)
            distilled_first = curvature_scores(costate_hat, torch.zeros_like(costate_hat),
                                               delta)
            pair_pred_2nd = second_order_pair_correction(
                delta.detach(), hessian_hat.detach(), pairs)
            base_cpu = base.detach().cpu()
            distilled_cpu = distilled_first.detach().cpu()
            for i in range(windows):
                single_gains = (base_cpu[i] - singles.detach().cpu()[i]).tolist()
                measured = (base_cpu[i] - pair_J.detach().cpu()[i]).tolist()
                rows.append({
                    "episode_id": str(batch["episode_id"][i]),
                    "site": site_by_episode.get(str(batch["episode_id"][i]), "unknown"),
                    "singleton_gains": single_gains,
                    "measured_pair_gains": measured,
                    "predicted_epsilon_2nd": pair_pred_2nd.detach().cpu()[i].tolist(),
                    "first_order_singleton": first_order.detach().cpu()[i].tolist(),
                    "distilled_first_order_singleton": [float(v) for v in distilled_cpu[i]],
                    "pair_additivity_relative": additivity.detach().cpu()[i].tolist(),
                })
        del batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # Analysis on committed rows (same definitions the summary reports).
    analysis = analyse_mechanism_rows(rows, manifest, site_by_episode)
    return {
        "benchmark": "session_6c_mechanism_check",
        "mode": "real",
        "device": str(device),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backbone": {"path": str(path), "sha256": sha256_file(path)},
        "head": {"steps": trained["steps"], "lr": args.lr,
                 "trained_on": "TRAIN split (production protocol; test windows disjoint)"},
        "pair_manifest": manifest,
        "n_pairs": len(manifest),
        "pair_seed": args.pair_seed,
        "episodes": episodes_seen,
        "n_windows": len(rows),
        "seed": args.seed,
        "window_rows": rows,
        **analysis,
        "caveats": [
            "Hindsight best pair is a diagnostic reference, never a deployable policy.",
            "Distilled diagonal Hessian only; no exact full Hessian exists here.",
        ],
    }


def analyse_mechanism_rows(rows: Sequence[dict], manifest: Sequence[dict],
                          site_by_episode: Mapping[str, str]) -> dict:
    """Pure analysis from committed rows: additivity premise, epsilon prediction, arms.

    Arms — A additive-distilled (first-order λ̂ sums), B corrected-distilled (A plus the
    distilled-Hessian pair correction), C oracle-ceiling (measured singleton sums plus the
    same correction), D measured best pair (hindsight). Per-window correlations averaged;
    top-k overlap vs the measured best; regret differences with Wilcoxon + site-clustered
    uncertainty. Unit-testable without a GPU.
    """
    from adjointrwm.spatial_selection import (  # noqa: E402
        site_clustered_bootstrap_ci,
        wilcoxon_signed_rank,
    )
    from diagnose_spatial_selection_bottleneck import spearman, topk_overlap  # noqa: E402

    eps_pred_all, eps_meas_all, additivity_all = [], [], []
    arm_rhos: Dict[str, list] = {"A": [], "B": [], "C": []}
    arm_overlaps: Dict[str, list] = {"A": [], "B": [], "C": []}
    regrets: Dict[str, list] = {"A": [], "B": [], "C": []}
    sites: List[str] = []
    for row in rows:
        single = np.array(row["singleton_gains"])
        measured = np.array(row["measured_pair_gains"])
        pred_eps = np.array(row["predicted_epsilon_2nd"])
        dist_first = np.array(row["distilled_first_order_singleton"])
        a = np.array([m["patch_a"] for m in manifest])
        b = np.array([m["patch_b"] for m in manifest])
        meas_eps = measured - single[a] - single[b]
        eps_pred_all.extend(pred_eps.tolist())
        eps_meas_all.extend(meas_eps.tolist())
        additivity = np.array(row["pair_additivity_relative"])
        additivity_all.extend(additivity.tolist())
        arms = {
            "A": dist_first[a] + dist_first[b],
            "B": dist_first[a] + dist_first[b] + pred_eps,
            "C": single[a] + single[b] + pred_eps,
        }
        with torch.no_grad():
            for name, scores in arms.items():
                rho = spearman(torch.as_tensor(scores[None, :]),
                               torch.as_tensor(measured[None, :]))
                arm_rhos[name].append(float(rho[0]) if torch.isfinite(rho[0])
                                      else float("nan"))
                overlap = pair_topk_overlap(torch.as_tensor(scores[None, :]),
                                            torch.as_tensor(measured[None, :]), 2)
                arm_overlaps[name].append(float(overlap[0]))
        best = float(measured.max())
        for name, scores in arms.items():
            chosen = int(np.argmax(scores))
            regrets[name].append(best - float(measured[chosen]))
        sites.append(row.get("site", "unknown"))
    differences = np.array(regrets["A"]) - np.array(regrets["B"])
    wilcoxon = wilcoxon_signed_rank(differences)
    cluster_ci = site_clustered_bootstrap_ci(
        differences, np.asarray(sites), num_resamples=2000, seed=0)

    def trimmed(values: np.ndarray, fraction: float = 0.1) -> float:
        ordered = np.sort(np.asarray(values, dtype=float))
        cut = int(len(ordered) * fraction)
        core = ordered[cut: len(ordered) - cut] if cut else ordered
        return float(core.mean()) if len(core) else float("nan")

    eps_pred_all = np.array(eps_pred_all)
    eps_meas_all = np.array(eps_meas_all)
    return {
        "pair_additivity": {
            "median_relative_residual": float(np.median(additivity_all)),
            "mean_relative_residual": float(np.mean(additivity_all)),
            "fraction_below_10pct": float(np.mean(np.array(additivity_all) < 0.10)),
        },
        "epsilon_prediction": {
            "mean_abs_predicted": float(np.abs(eps_pred_all).mean()),
            "mean_abs_measured": float(np.abs(eps_meas_all).mean()),
            "correlation": float(np.corrcoef(eps_pred_all, eps_meas_all)[0, 1])
            if len(eps_pred_all) > 2 else float("nan"),
        },
        "arm_rank_fidelity": {name: float(np.nanmean(values))
                              for name, values in arm_rhos.items()},
        "arm_overlap_with_measured_best": {name: float(np.nanmean(values))
                                           for name, values in arm_overlaps.items()},
        "arm_regret_vs_measured_best": {
            name: {"mean": float(np.mean(values)), "trimmed_mean": trimmed(np.asarray(values)),
                   "n": len(values)} for name, values in regrets.items()},
        "corrected_vs_additive_regret": {
            "definition": ("per-window regret(A) − regret(B): positive means the Hessian "
                           "correction helped within the distilled world"),
            "mean": float(differences.mean()),
            "trimmed_mean": trimmed(differences),
            "wilcoxon": wilcoxon,
            "site_clustered_ci_95": cluster_ci,
            "n": int(len(differences)),
        },
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 70)
    print("Session 6C mechanism check: second-order structure vs measured joints")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    summary = run(args, device)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "mechanism_check_summary.json", summary)
    print(f"Saved summary to {output_dir / 'mechanism_check_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
