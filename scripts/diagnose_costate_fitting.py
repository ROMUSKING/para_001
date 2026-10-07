#!/usr/bin/env python3
"""scripts/diagnose_costate_fitting.py

Session 6C Experiment 1: can the training problem be fitted at all?

Frozen design (``docs/plans/2026-10-04-session-6c-learnability-audit-plan.md`` §1): one frozen
teacher checkpoint, declared horizon/objective, 16 fixed train windows from 4 train-split
episodes + 16 held-out windows from 4 *different* train-split episodes, no augmentation, no
intentional randomness beyond 3 init seeds. Two students: closed-form ridge (the linear-signal
floor) and the production composite head at hidden 256.

Four metrics, not just loss: relative magnitude error, directional error, candidate-score
error (Spearman + top-2 overlap of ``curvature_scores`` vs exact gains), and allocation
agreement (fraction of windows with identical top-k sets). The decision table in the plan
routes on these jointly.

Provenance is a prerequisite, not an extra (Session 5 lesson): missing backbone checkpoint
is a hard failure; the summary records requested seed, actual checkpoint identity (SHA-256),
teacher identity and label-generation config.

Usage::

    python scripts/diagnose_costate_fitting.py \\
        --cache-dir /content/spatial_cache_e3_1 --drive-root /content/local_runs \\
        --run-id spatial_patches_20261003 --output-dir results/benchmarks/costate_fitting
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_DIR = Path(__file__).resolve().parent.parent
for candidate in (REPO_DIR / "src", Path("/content/para_001/src")):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO_DIR / "scripts"))

from adjointrwm.allocators import CurvatureCostateEstimator  # noqa: E402
from adjointrwm.io import atomic_write_json  # noqa: E402
from adjointrwm.spatial_selection import (  # noqa: E402
    CAMERAS,
    camera_of_patch,
    cotangent_bundle,
    curvature_scores,
    latent_patch_perturbations,
    select_topk_per_camera,
    trimmed_mean,
)

from benchmark_spatial_patch_selection import (  # noqa: E402
    build_spatial_model,
    exact_marginal_gains,
    load_spatial_splits,
    move_to_device,
)
from diagnose_spatial_selection_bottleneck import (  # noqa: E402
    spearman,
    topk_overlap,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Session 6C Exp 1: tiny-set costate fitting check")
    parser.add_argument("--cache-dir", type=str, default="/content/spatial_cache_e3_1")
    parser.add_argument("--drive-root", type=str, default="/content/local_runs")
    parser.add_argument("--run-id", type=str, default="spatial_patches_20261003")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks/costate_fitting")
    parser.add_argument("--train-windows", type=int, default=16)
    parser.add_argument("--train-episodes", type=int, default=4)
    parser.add_argument("--heldout-windows", type=int, default=16)
    parser.add_argument("--heldout-episodes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-seeds", type=str, default="0,1,2")
    return parser.parse_args(argv)


def _repo_commit() -> str:
    """Best-effort repo identity for provenance; never fails the run if git is absent."""
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10, cwd=REPO_DIR).stdout.strip()
    except Exception:
        return "unknown (git unavailable)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_backbone(args, device: torch.device):
    """Load or hard-fail: a missing checkpoint must never silently degrade to random init."""
    path = Path(args.drive_root) / "runs" / args.run_id / "best_spatial.pt"
    if not path.exists():
        raise SystemExit(
            f"missing teacher checkpoint at {path}: refusing to run (Session 5 lesson — "
            "a randomly initialised backbone would make every number below meaningless)"
        )
    model = build_spatial_model(args, 16, device, args.width, 384)
    state = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state.get("model_state_dict", state), strict=False)
    return model, {
        "path": str(path),
        "sha256": sha256_file(path),
        "missing_keys": sorted(missing)[:8],
        "unexpected_keys": sorted(unexpected)[:8],
        "patch_dropout_trained": bool(state.get("patch_dropout_trained", False)),
    }


def pick_windows(test_like, episode_ids: np.ndarray, n_episodes: int, n_windows: int,
                 skip: set, seed: int):
    """Deterministic episode-stratified pick: round-robin over seeded episode order."""
    by_episode: Dict[str, List[int]] = {}
    for index, episode in enumerate(episode_ids):
        by_episode.setdefault(str(episode), []).append(index)
    episodes = sorted(e for e in by_episode if e not in skip)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    chosen, used = [], set()
    per_episode = max(1, n_windows // max(1, n_episodes))
    for episode in episodes:
        if len(used) >= n_episodes:
            break
        take = by_episode[episode][:per_episode]
        if take:
            chosen.extend(take)
            used.add(episode)
    return sorted(chosen)[:n_windows], sorted(used)


def collect(model, batches: List[Mapping], device: torch.device) -> dict:
    """Exact labels + decision-time inputs for a fixed window set (no training here).

    The autograd reference needs its graph: ``cotangent_bundle`` and
    ``latent_patch_perturbations`` run with grad enabled (as in ``train_heads``) and their
    outputs are detached for storage. Only ``exact_marginal_gains`` is gradient-free by
    construction (it carries its own ``no_grad``).
    """
    latents, exacts, deltas, gain_list = [], [], [], []
    for batch in batches:
        with torch.enable_grad():
            bundle = cotangent_bundle(model, batch)
            delta = latent_patch_perturbations(model, batch)
        gains = exact_marginal_gains(model, batch)
        with torch.no_grad():
            latents.append(bundle.latent.detach().cpu())
            exacts.append(bundle.exact_costate.detach().cpu())
            deltas.append(delta.detach().cpu())
            gain_list.append(gains.detach().cpu())
    return {
        "latent": torch.cat(latents),
        "exact": torch.cat(exacts),
        "delta_z": torch.cat(deltas),
        "gains": torch.cat(gain_list),
    }


def reconstruction_errors(pred: torch.Tensor, exact: torch.Tensor) -> dict:
    """Layer 1 (derivative reconstruction): vector error and magnitude error separately.

    ``rel_vector_error`` (‖λ̂−λ‖/‖λ‖) is reconstruction fidelity; ``rel_magnitude_error``
    (|‖λ̂‖−‖λ‖|/‖λ‖) is scale fidelity. They are different quantities — a vector can point
    the right way at the wrong scale — so the plan forbids reporting one as the other.
    """
    denom = exact.norm(dim=-1).clamp_min(1e-12)
    vector = (pred - exact).norm(dim=-1) / denom
    magnitude = (pred.norm(dim=-1) - exact.norm(dim=-1)).abs() / denom
    direction = 1.0 - F.cosine_similarity(pred, exact, dim=-1)
    return {
        "rel_vector_error": float(vector.mean()),
        "rel_magnitude_error": float(magnitude.mean()),
        "directional_error": float(direction.mean()),
    }


def score_and_outcome(pred: torch.Tensor, ref: dict, budgets: tuple = (2, 4, 8)) -> dict:
    """Layers 2+3: score fidelity and outcome quality at every deployed budget.

    Layer 2 compares the *identical* scorer under λ̂ vs λ (zero Hessian, true Δz held fixed),
    so only the co-state differs. Layer 3 reports realised selection quality. Spearman is
    NA-aware: undefined correlations are NaN, never zero. Budgets are per-camera k_cam over
    the 16-patch camera grid (total k = 2 × k_cam); patches are kept, not removed.
    """
    exact = ref["exact"]
    scores = curvature_scores(pred, torch.zeros_like(pred), ref["delta_z"])
    scores_exact = curvature_scores(exact, torch.zeros_like(exact), ref["delta_z"])
    rho_fidelity = spearman(scores.float(), scores_exact.float())
    rho_gains = spearman(scores.float(), ref["gains"].float())
    finite_fidelity = torch.isfinite(rho_fidelity)
    finite_gains = torch.isfinite(rho_gains)
    out = {
        # Layer 2: same scorer, λ̂ vs λ. Mixes nothing else in.
        "spearman_vs_exact_scores": (float(rho_fidelity[finite_fidelity].mean())
                                     if int(finite_fidelity.sum()) else float("nan")),
        "spearman_vs_exact_scores_n": int(finite_fidelity.sum()),
        # Layers 2+3 combined (score approximation AND singleton-vs-gains mismatch).
        "spearman_vs_exact_gains": (float(rho_gains[finite_gains].mean())
                                    if int(finite_gains.sum()) else float("nan")),
        "spearman_vs_exact_gains_n": int(finite_gains.sum()),
        "per_budget": {},
    }
    with torch.no_grad():
        for k_cam in budgets:
            overlap = topk_overlap(scores, ref["gains"], k_cam)
            truth_top = ref["gains"].topk(k_cam, dim=1).indices.sort(dim=1).values
            pred_top = scores.topk(k_cam, dim=1).indices.sort(dim=1).values
            agreement = (truth_top == pred_top).all(dim=1).float()
            out["per_budget"][f"k_cam={k_cam}"] = {
                "topk_overlap": float(overlap.mean()),
                "allocation_agreement": float(agreement.mean()),
            }
    out["n"] = int(pred.shape[0])
    return out


def four_metrics(pred: torch.Tensor, ref: dict, budgets: tuple = (2, 4, 8)) -> dict:
    """All three layers in one call (kept for the CLI summary path).

    Budgets exceeding the per-camera patch count are dropped: the synthetic test fixtures
    use 4 patches per camera, where k_cam=8 is not a valid top-k. Production always passes
    the full (2, 4, 8) grid against 16 patches per camera.
    """
    patches_per_camera = ref["delta_z"].shape[1] // CAMERAS
    valid = tuple(k for k in budgets if k <= patches_per_camera)
    if not valid:
        raise ValueError(f"no valid budget in {budgets} for {patches_per_camera} patches per camera")
    out = reconstruction_errors(pred, ref["exact"])
    out.update(score_and_outcome(pred, ref, budgets=valid))
    return out


def unregularised_lstsq(design: torch.Tensor, targets: torch.Tensor,
                        rcond: float = 1e-12) -> dict:
    """Closed-form least squares in float64 with an SVD-based driver (review R3-B).

    No normal equations (which square the condition number): ``torch.linalg.lstsq`` with
    ``driver="gelsd"``. Reports effective rank, singular values and residual, so a
    full-row-rank interpolation success reads as finite-sample fitability — never as
    generalisation or information sufficiency — and a rank-deficient residual points at
    input rank, duplicate inputs or target inconsistency. Must not retroactively pass any
    production gate.
    """
    double_design = design.to(torch.float64)
    double_targets = targets.to(torch.float64)
    solution, residuals, rank, singular = torch.linalg.lstsq(
        double_design, double_targets, rcond=rcond, driver="gelsd")
    residual = float((targets.to(torch.float64) - double_design @ solution).pow(2).sum())
    return {
        "weights": solution.to(design.dtype),
        "rank": int(rank),
        "n_rows": int(design.shape[0]),
        "n_cols": int(design.shape[1]),
        "full_row_rank": bool(rank == min(design.shape)),
        "singular_values": [float(v) for v in singular],
        "residual_sum_of_squares": residual,
        "rcond": rcond,
        "driver": "gelsd",
    }


def scalar_calibration(pred_fit: torch.Tensor, exact_fit: torch.Tensor,
                       pred_hold: torch.Tensor) -> dict:
    """Single train-fitted global scale a* (review R3-C).

    a* = max(0, Σ λ̂ᵀλ / Σ ‖λ̂‖²) on the fit set only; applied unchanged to held-out.
    A large train improvement from one scalar means global miscalibration dominates; no
    held-out gain preserves the generalisation question with a better-localised training
    problem. Undefined (zero denominator) is reported, never defaulted silently.
    """
    numerator = float((pred_fit * exact_fit).sum())
    denominator = float(pred_fit.pow(2).sum())
    if denominator <= 0:
        return {"a_star": None, "undefined": "zero prediction energy on the fit set"}
    return {"a_star": max(0.0, numerator / denominator), "undefined": None}


def engineering_fit_gate(pred: torch.Tensor, exact: torch.Tensor, energy_floor: float = 1e-6,
                         absolute_tol: float = 1e-4) -> dict:
    """Numerical fit gate E_rec ≤ 0.01 (plan §1): engineering sanity, not a research threshold.

    E_rec = sqrt(Σ‖λ̂−λ‖²/Σ‖λ‖²) over the tiny train set. Below ``energy_floor`` total target
    energy the relative gate is meaningless, so the absolute tolerance rules instead and the
    condition is reported (not silently passed).
    """
    energy = float((exact.pow(2).sum()))
    if energy < energy_floor:
        absolute = float(((pred - exact).pow(2).sum()).sqrt())
        return {"gate": "absolute", "energy": energy, "absolute_error": absolute,
                "passed": absolute <= absolute_tol, "absolute_tol": absolute_tol}
    value = float((((pred - exact).pow(2).sum()) / exact.pow(2).sum()).sqrt())
    return {"gate": "relative", "energy": energy, "E_rec": value,
            "passed": value <= 0.01, "threshold": 0.01}


def run(args, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    model, teacher = load_backbone(args, device)
    _train, _val, _test, site_by_episode = load_spatial_splits(Path(args.cache_dir), 16)
    # Fit-set and held-out set both come from the TRAIN split (different episodes): this
    # separates optimisation failure from generalisation failure without adding shift.
    train_set = _train
    all_ids = np.asarray(train_set.episode_ids())
    fit_idx, fit_episodes = pick_windows(train_set, all_ids, args.train_episodes,
                                         args.train_windows, set(), args.seed)
    hold_idx, hold_episodes = pick_windows(train_set, all_ids, args.heldout_episodes,
                                           args.heldout_windows, set(fit_episodes), args.seed + 1)
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"),
                         persistent_workers=args.num_workers > 0)

    def batches_for(indices):
        loader = DataLoader(Subset(train_set, indices), shuffle=False, **loader_kwargs)
        return [move_to_device(b, device) for b in loader]

    fit_batches, hold_batches = batches_for(fit_idx), batches_for(hold_idx)
    fit = collect(model, fit_batches, device)
    hold = collect(model, hold_batches, device)
    print(f"fit: {fit['latent'].shape[0]} windows over {len(fit_episodes)} episodes; "
          f"held-out: {hold['latent'].shape[0]} over {len(hold_episodes)}", flush=True)

    # (0) Stored-label injection (preflight control): exact λ straight through scorer and
    # evaluator. Must reproduce the exact-λ result; anything else is an indexing,
    # normalisation, conversion or mask-identity bug upstream of all learning.
    injection = {
        "fit": four_metrics(fit["exact"], fit),
        "held_out": four_metrics(hold["exact"], hold),
    }

    # Constant predictor (train-mean λ): the no-signal floor every student must beat to claim
    # anything beyond memorising the mean. Train-only statistics, applied to both sets.
    train_mean = fit["exact"].mean(dim=0, keepdim=True)
    constant = {
        "fit": four_metrics(train_mean.expand_as(fit["exact"]), fit),
        "held_out": four_metrics(train_mean.expand_as(hold["exact"]), hold),
    }

    # (a0) Unregularised linear control (review R3-B): float64 SVD solve on the actual
    # student inputs (latent + intercept). One deterministic fit, no seeds.
    ones_all = torch.ones(fit["latent"].shape[0], 1)
    lstsq = unregularised_lstsq(torch.cat([fit["latent"], ones_all], dim=1), fit["exact"])
    hold_ones = torch.ones(hold["latent"].shape[0], 1)
    lstsq_fit = torch.cat([fit["latent"], ones_all], dim=1) @ lstsq["weights"]
    lstsq_hold = torch.cat([hold["latent"], hold_ones], dim=1) @ lstsq["weights"]
    lstsq_calibration = scalar_calibration(lstsq_fit, fit["exact"], lstsq_hold)
    lstsq_row = {
        "init_seed": "n/a (deterministic)", "student": "lstsq_unregularised",
        "parameters": int(lstsq["weights"].numel()), "lstsq": {k: v for k, v in lstsq.items()
                                                               if k != "weights"},
        "fit": four_metrics(lstsq_fit, fit),
        "held_out": four_metrics(lstsq_hold, hold),
        "fit_gate": engineering_fit_gate(lstsq_fit, fit["exact"]),
        "scalar_calibration": lstsq_calibration,
    }
    if lstsq_calibration["a_star"] is not None:
        a = lstsq_calibration["a_star"]
        lstsq_row["calibrated"] = {
            "fit": four_metrics(a * lstsq_fit, fit),
            "held_out": four_metrics(a * lstsq_hold, hold),
        }
    results: List[dict] = [lstsq_row]

    for init_seed in [int(s) for s in args.init_seeds.split(",") if s.strip()]:
        # (a) Ridge floor: closed form on the fit set, evaluated on both sets. A linear
        # diagnostic only — fitting 16 high-dimensional windows proves no generalisable
        # linear signal (see the constant predictor and held-out columns).
        torch.manual_seed(init_seed)
        ones = torch.ones(fit["latent"].shape[0], 1)
        design = torch.cat([fit["latent"], ones], dim=1)
        gram = design.T @ design + args.ridge_alpha * torch.eye(design.shape[1])
        weights = torch.linalg.solve(gram, design.T @ fit["exact"])
        hold_design = torch.cat([hold["latent"], torch.ones(hold["latent"].shape[0], 1)], dim=1)
        ridge_fit = design @ weights
        ridge_hold = hold_design @ weights
        results.append({
            "init_seed": init_seed, "student": "ridge",
            "parameters": int(weights.numel()),
            "fit": four_metrics(ridge_fit, fit),
            "held_out": four_metrics(ridge_hold, hold),
            "fit_gate": engineering_fit_gate(ridge_fit, fit["exact"]),
        })

        # (b) Production head, composite loss, fixed step budget on the fit set only.
        torch.manual_seed(init_seed)
        head = CurvatureCostateEstimator(model.d_model, hidden=args.hidden).to(device)
        optimiser = torch.optim.AdamW(head.parameters(), lr=args.lr)
        budget = torch.full((fit["latent"].shape[0],), 1.0 / 16)
        horizon = torch.ones_like(budget)
        stream = ([fit_batches[i % len(fit_batches)] for i in range(args.steps)])
        for batch in stream:
            windows = batch["context_state"].shape[0]
            bf = torch.full((windows,), 1.0 / 16, device=device)
            hf = torch.ones_like(bf)
            gains = exact_marginal_gains(model, batch)
            bundle = cotangent_bundle(model, batch)
            latent = bundle.latent.detach()
            delta_z = latent_patch_perturbations(model, batch)
            costate_hat, hessian_hat = head(latent, bf, hf)
            direction = (1.0 - F.cosine_similarity(costate_hat, bundle.exact_costate, dim=-1)).mean()
            predicted = (hessian_hat.unsqueeze(1) * delta_z.pow(2)).sum(-1)
            loss = direction + 0.5 * F.smooth_l1_loss(predicted, gains)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimiser.step()
        # (c) Pure reconstruction control: same small network, unregularised λ-MSE only.
        # Diagnostic control, not a benchmark candidate: separates "cannot represent or
        # optimise the labels" from "the composite objective constrains the wrong thing".
        torch.manual_seed(init_seed)
        pure = CurvatureCostateEstimator(model.d_model, hidden=args.hidden).to(device)
        pure_opt = torch.optim.AdamW(pure.parameters(), lr=args.lr)
        for batch in ([fit_batches[i % len(fit_batches)] for i in range(args.steps)]):
            windows = batch["context_state"].shape[0]
            bf = torch.full((windows,), 1.0 / 16, device=device)
            hf = torch.ones_like(bf)
            bundle = cotangent_bundle(model, batch)
            latent = bundle.latent.detach()
            pred, _ = pure(latent, bf, hf)
            loss = F.mse_loss(pred, bundle.exact_costate.detach())
            pure_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pure.parameters(), 1.0)
            pure_opt.step()
        pure = pure.eval()
        head = head.eval()
        with torch.no_grad():
            bf = torch.full((fit["latent"].shape[0],), 1.0 / 16)
            ones_f = torch.ones_like(bf)
            pred_fit, _ = head(fit["latent"].to(device), bf.to(device), ones_f.to(device))
            pure_fit, _ = pure(fit["latent"].to(device), bf.to(device), ones_f.to(device))
            bh = torch.full((hold["latent"].shape[0],), 1.0 / 16)
            ones_h = torch.ones_like(bh)
            pred_hold, _ = head(hold["latent"].to(device), bh.to(device), ones_h.to(device))
            pure_hold, _ = pure(hold["latent"].to(device), bh.to(device), ones_h.to(device))
        calibration = scalar_calibration(pred_fit.cpu(), fit["exact"], pred_hold.cpu())
        composite_row = {
            "init_seed": init_seed, "student": f"composite_h{args.hidden}",
            "parameters": int(sum(p.numel() for p in head.parameters())),
            "fit": four_metrics(pred_fit.cpu(), fit),
            "held_out": four_metrics(pred_hold.cpu(), hold),
            "fit_gate": engineering_fit_gate(pred_fit.cpu(), fit["exact"]),
            "scalar_calibration": calibration,
        }
        if calibration["a_star"] is not None:
            a = calibration["a_star"]
            composite_row["calibrated"] = {
                "fit": four_metrics(a * pred_fit.cpu(), fit),
                "held_out": four_metrics(a * pred_hold.cpu(), hold),
            }
        results.append(composite_row)
        pure_calibration = scalar_calibration(pure_fit.cpu(), fit["exact"], pure_hold.cpu())
        pure_row = {
            "init_seed": init_seed, "student": f"pure_mse_h{args.hidden}",
            "parameters": int(sum(p.numel() for p in pure.parameters())),
            "fit": four_metrics(pure_fit.cpu(), fit),
            "held_out": four_metrics(pure_hold.cpu(), hold),
            "fit_gate": engineering_fit_gate(pure_fit.cpu(), fit["exact"]),
            "scalar_calibration": pure_calibration,
        }
        if pure_calibration["a_star"] is not None:
            a = pure_calibration["a_star"]
            pure_row["calibrated"] = {
                "fit": four_metrics(a * pure_fit.cpu(), fit),
                "held_out": four_metrics(a * pure_hold.cpu(), hold),
            }
        results.append(pure_row)
        print(f"  seed {init_seed}: ridge fit-vec "
              f"{results[-3]['fit']['rel_vector_error']:.4f} | composite fit-vec "
              f"{results[-2]['fit']['rel_vector_error']:.4f} gate "
              f"{results[-2]['fit_gate']['passed']} | pure fit-vec "
              f"{results[-1]['fit']['rel_vector_error']:.4f} gate "
              f"{results[-1]['fit_gate']['passed']}", flush=True)

    return {
        "benchmark": "session_6c_costate_fitting",
        "mode": "real",
        "device": str(device),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "teacher": teacher,
        "label_generation": {
            "objective": "prediction_objective (Gaussian NLL state + 0.25*(1-cos) visual)",
            "horizon": 4,
            "gains": "exact_marginal_gains J(z_S)-J(z_{S u p}) from null selection",
            "costate": "cotangent_bundle exact dJ/dz (needs future targets)",
        },
        "fit_episodes": fit_episodes,
        "heldout_episodes": hold_episodes,
        "n_fit_windows": int(fit["latent"].shape[0]),
        "n_heldout_windows": int(hold["latent"].shape[0]),
        "steps": args.steps,
        "lr": args.lr,
        "ridge_alpha": args.ridge_alpha,
        "precision": str(torch.get_default_dtype()),
        "torch_version": torch.__version__,
        "repo_commit": _repo_commit(),
        "normaliser": ("fitted on the manifest train split by load_spatial_splits "
                       "(DEV-20261004-02); committed reference results/benchmarks/"
                       "spatial_backbone/spatial_normaliser.json"),
        "seed": args.seed,
        "init_seeds": args.init_seeds,
        "note_on_replication": ("one frozen teacher with three student initialisations, not "
                                "three independent teacher replications"),
        "stored_label_injection": injection,
        "constant_predictor": constant,
        "results": results,
        "reading_guide": {
            "rel_vector_error": "Can the student reconstruct the target vector at all?",
            "rel_magnitude_error": "Can it match scale specifically?",
            "directional_error": "Can it match direction (what cosine supervision trains)?",
            "spearman_vs_exact_scores": "Layer 2: same scorer under predicted vs exact costate.",
            "spearman_vs_exact_gains": "Layers 2+3 combined; cannot isolate distillation alone.",
            "per_budget": "Layer 3 outcome quality at each deployed budget.",
            "fit_gate": "Engineering sanity E_rec <= 0.01 on the tiny train set (plan §1).",
        },
        "caveats": [
            "Fitting a tiny set does not prove sufficiency: a flexible model can memorise noisy targets.",
            "Single future trajectory per state: no irreducible-uncertainty claim may rest on this data.",
            "Ridge sees the pooled latent only, matching the production head's input.",
            "The four held-out episodes support a small diagnostic, not a broad generalisation claim.",
        ],
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 70)
    print("Session 6C Experiment 1: can the costate training problem be fitted?")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    summary = run(args, device)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "costate_fitting_summary.json", summary)
    print(f"Saved summary to {output_dir / 'costate_fitting_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
