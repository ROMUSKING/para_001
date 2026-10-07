#!/usr/bin/env python3
"""scripts/diagnose_information_gap.py

Session 6C Experiment 2 (bounded linear-probe form, under the prospective amendment in
``docs/plans/2026-10-04-session-6c-learnability-audit-plan.md`` §7 — do not run until that
amendment is peer-reviewed and recorded).

Question: is the useful exact-gradient information predictable from permitted inputs?
Three closed-form linear conditions on the SAME 16+16 windows, identical numerics/labels;
the only variation is the input columns (no iterative training, no hyperparameters):

- A (deployment): X = [latent z, 1]. The actual deployable problem.
- B (+ legitimate context): X = [z, future_actions flat, budget, horizon, 1]. Future
  actions are rollout inputs the selector serves, not outcomes — recorded as an explicit,
  attackable premise that they are legitimate query context.
- C (privileged, diagnostic only): X = [z, target_state flat, target_visual flat, 1].
  Never λ or exact scores (those are the label-injection control, not information).

 readouts: Exp-1 four-metric set per condition per split + rank/singular/residual per fit.
 Directional evidence only: no gates pass here, no superiority claims, no "only privileged
 generalises" beyond this diagnostic split.

Usage::

    python scripts/diagnose_information_gap.py \\
        --cache-dir /content/spatial_cache_e3_1 --drive-root /content/local_runs \\
        --run-id spatial_patches_20261003 --output-dir results/benchmarks/information_gap
"""

from __future__ import annotations

import argparse
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
    cotangent_bundle,
    latent_patch_perturbations,
)

from benchmark_spatial_patch_selection import (  # noqa: E402
    exact_marginal_gains,
    load_spatial_splits,
    move_to_device,
)
from diagnose_costate_fitting import (  # noqa: E402
    engineering_fit_gate,
    four_metrics,
    load_backbone,
    pick_windows,
    scalar_calibration,
    unregularised_lstsq,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Session 6C Exp 2: information-gap linear probes")
    parser.add_argument("--cache-dir", type=str, default="/content/spatial_cache_e3_1")
    parser.add_argument("--drive-root", type=str, default="/content/local_runs")
    parser.add_argument("--run-id", type=str, default="spatial_patches_20261003")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks/information_gap")
    parser.add_argument("--train-windows", type=int, default=16)
    parser.add_argument("--train-episodes", type=int, default=4)
    parser.add_argument("--heldout-windows", type=int, default=16)
    parser.add_argument("--heldout-episodes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rcond", type=float, default=1e-12)
    return parser.parse_args(argv)


def collect_gap(model, batches: List[Mapping], device: torch.device) -> dict:
    """Fixed-window data with deployment AND privileged columns (labels identical to Exp 1)."""
    latent_l, exact_l, delta_l, gain_l, act_l, tstate_l, tvis_l = ([] for _ in range(7))
    for batch in batches:
        with torch.enable_grad():
            bundle = cotangent_bundle(model, batch)
            delta = latent_patch_perturbations(model, batch)
        gains = exact_marginal_gains(model, batch)
        with torch.no_grad():
            latent_l.append(bundle.latent.detach().cpu())
            exact_l.append(bundle.exact_costate.detach().cpu())
            delta_l.append(delta.detach().cpu())
            gain_l.append(gains.detach().cpu())
            act_l.append(batch["future_actions"].detach().cpu())
            tstate_l.append(batch["target_state"].detach().cpu())
            tvis_l.append(batch["target_visual"].detach().cpu())
    cat = torch.cat
    return {
        "latent": cat(latent_l), "exact": cat(exact_l), "delta_z": cat(delta_l),
        "gains": cat(gain_l), "future_actions": cat(act_l),
        "target_state": cat(tstate_l), "target_visual": cat(tvis_l),
    }


def random_projector(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
    """Data-independent Gaussian projection (plan §7 capacity equaliser).

    Seeded, fixed before seeing any data, identical distribution for every condition:
    Johnson–Lindenstrauss capacity control, not feature learning. Without it, raw column
    counts (513 vs ~3600 on 16 rows) confound information content with nullspace size —
    every condition is rank-deficient and C's nullspace contains the targets.
    """
    generator = torch.Generator().manual_seed(seed)
    projector = torch.randn(in_dim, out_dim, dtype=torch.float64, generator=generator)
    return projector / max(1, in_dim) ** 0.5


def zero_fill_privileged(full_test: torch.Tensor, n_keep: int) -> torch.Tensor:
    """Condition-D test features: keep the first ``n_keep`` (deployment) columns, zero the
    privileged remainder, preserve the intercept column. Unit-tested; the only untested part
    of D is the GPU fit itself."""
    if n_keep + 1 > full_test.shape[1]:
        raise ValueError(f"n_keep={n_keep} exceeds design width {full_test.shape[1]}")
    out = full_test.clone()
    out[:, n_keep:-1] = 0.0
    return out


def design_matrices(fit: dict, hold: dict, proj_dim: int = 12,
                    proj_seed: int = 0) -> Dict[str, dict]:
    """Exact per-condition input manifests (plan §7, revised per kilo review).

    A/B_ctx/B_log/C are projected to a common width; D fits full C columns and zero-fills
    privileged blocks at test. Shapes recorded, never assumed. Logged future actions appear
    only under an explicit §0-violation label (B_log), never as legitimate context.
    """
    def flat(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(t.shape[0], -1)

    ones_fit = torch.ones(fit["latent"].shape[0], 1)
    ones_hold = torch.ones(hold["latent"].shape[0], 1)
    budget_fit = torch.full((fit["latent"].shape[0], 1), 1.0 / 16)
    budget_hold = torch.full((hold["latent"].shape[0], 1), 1.0 / 16)
    horizon_fit = torch.ones(fit["latent"].shape[0], 1)
    horizon_hold = torch.ones(hold["latent"].shape[0], 1)
    raw = {
        "A_deployment": {
            "fit": torch.cat([fit["latent"], ones_fit], dim=1),
            "hold": torch.cat([hold["latent"], ones_hold], dim=1),
            "columns": ["latent[d]", "intercept"],
        },
        "B_ctx_legitimate": {
            "fit": torch.cat([fit["latent"], budget_fit, horizon_fit, ones_fit], dim=1),
            "hold": torch.cat([hold["latent"], budget_hold, horizon_hold, ones_hold], dim=1),
            "columns": ["latent[d]", "budget", "horizon", "intercept"],
        },
        "B_log_privileged": {
            "fit": torch.cat([fit["latent"], flat(fit["future_actions"]), ones_fit], dim=1),
            "hold": torch.cat([hold["latent"], flat(hold["future_actions"]), ones_hold], dim=1),
            "columns": ["latent[d]", "future_actions[H*7]", "intercept"],
            "section_zero_violation": True,
        },
        "C_privileged": {
            "fit": torch.cat([fit["latent"], flat(fit["target_state"]),
                              flat(fit["target_visual"]), ones_fit], dim=1),
            "hold": torch.cat([hold["latent"], flat(hold["target_state"]),
                               flat(hold["target_visual"]), ones_hold], dim=1),
            "columns": ["latent[d]", "target_state[H*14]", "target_visual[H*Dv]", "intercept"],
        },
    }
    out = {}
    for offset, (name, spec) in enumerate(sorted(raw.items())):
        projector = random_projector(spec["fit"].shape[1], proj_dim, proj_seed + offset)
        entry = dict(spec)
        entry["fit"] = spec["fit"].to(torch.float64) @ projector
        entry["hold"] = spec["hold"].to(torch.float64) @ projector
        entry["projection"] = {"out_dim": proj_dim, "seed": proj_seed + offset,
                               "kind": "seeded-gaussian-JL"}
        out[name] = entry
    return out


def run(args, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    model, teacher = load_backbone(args, device)
    train_set, _val, _test, _sites = load_spatial_splits(Path(args.cache_dir), 16)
    all_ids = np.asarray(train_set.episode_ids())
    fit_idx, fit_episodes = pick_windows(train_set, all_ids, args.train_episodes,
                                         args.train_windows, set(), args.seed)
    hold_idx, hold_episodes = pick_windows(train_set, all_ids, args.heldout_episodes,
                                           args.heldout_windows, set(fit_episodes),
                                           args.seed + 1)
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"),
                         persistent_workers=args.num_workers > 0)

    def batches_for(indices):
        loader = DataLoader(Subset(train_set, indices), shuffle=False, **loader_kwargs)
        return [move_to_device(b, device) for b in loader]

    fit = collect_gap(model, batches_for(fit_idx), device)
    hold = collect_gap(model, batches_for(hold_idx), device)
    print(f"fit: {fit['latent'].shape[0]} windows over {len(fit_episodes)} episodes; "
          f"held-out: {hold['latent'].shape[0]} over {len(hold_episodes)}", flush=True)
    print(f"target dims: state {tuple(hold['target_state'].shape)} "
          f"visual {tuple(hold['target_visual'].shape)}", flush=True)

    # Condition D (true extrapolation, review attack 4): a SEPARATE full-space
    # privileged fit (its own lstsq object — not C's weights, which live in a different,
    # projected space and cannot be transferred), evaluated with privileged blocks
    # zero-filled at test, i.e. deployment features only. The training row below belongs
    # to this object, not to C; weight hashes prove which object produced which row.
    # D ≈ A-held-out means the full-privilege fit depended on privileged columns (gap
    # confirmed directionally); D ≈ C-held-out would deny the gap.
    def flat_rows(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(t.shape[0], -1)

    def weight_sha(weights: torch.Tensor) -> str:
        import hashlib

        return hashlib.sha256(weights.detach().cpu().numpy().tobytes()).hexdigest()

    ones_fit = torch.ones(fit["latent"].shape[0], 1)
    ones_hold = torch.ones(hold["latent"].shape[0], 1)
    full_c_fit = torch.cat([fit["latent"], flat_rows(fit["target_state"]),
                            flat_rows(fit["target_visual"]), ones_fit], dim=1)
    hold_zeroed = zero_fill_privileged(
        torch.cat([hold["latent"], flat_rows(hold["target_state"]),
                   flat_rows(hold["target_visual"]), ones_hold], dim=1),
        n_keep=hold["latent"].shape[1])
    full_c_hold = torch.cat([hold["latent"], flat_rows(hold["target_state"]),
                             flat_rows(hold["target_visual"]), ones_hold], dim=1)
    d_fit = unregularised_lstsq(full_c_fit, fit["exact"], rcond=args.rcond)
    d_weights = d_fit["weights"]
    d_pred_hold_zeroed = hold_zeroed @ d_weights
    d_pred_hold_full = full_c_hold @ d_weights
    # Near-zero output is confirmed via prediction norms, not inferred from rel error 1.0
    # (which λ̂=2λ also satisfies): report mean prediction norm against mean target norm.
    d_pred_norm = float(d_pred_hold_zeroed.norm(dim=-1).mean())
    d_target_norm = float(hold["exact"].norm(dim=-1).mean())
    conditions = [{
        "condition": "D_extrapolation",
        "fitted_object": "separate full-space privileged lstsq (own weights, NOT C's weights)",
        "weight_sha256": weight_sha(d_weights),
        "columns": ["latent[d]", "target_state[H*14]", "target_visual[H*Dv]", "intercept"],
        "n_columns": int(full_c_fit.shape[1]),
        "lstsq": {k: v for k, v in d_fit.items() if k != "weights"},
        "fit": four_metrics((full_c_fit @ d_weights).to(fit["latent"].dtype),
                            {k: fit[k] for k in ("exact", "delta_z", "gains")}),
        "held_out": four_metrics(d_pred_hold_zeroed.to(hold["latent"].dtype),
                                 {k: hold[k] for k in ("exact", "delta_z", "gains")}),
        "held_out_prediction_norm": d_pred_norm,
        "held_out_target_norm": d_target_norm,
        "held_out_full_columns_reference": four_metrics(
            d_pred_hold_full.to(hold["latent"].dtype),
            {k: hold[k] for k in ("exact", "delta_z", "gains")}),
        "fit_gate": engineering_fit_gate((full_c_fit @ d_weights).to(fit["latent"].dtype),
                                         fit["exact"]),
        "scalar_calibration": {"a_star": None,
                               "undefined": "not applicable to the extrapolation arm"},
    }]
    print(f"  D_extrapolation: cols={full_c_fit.shape[1]} rank={d_fit['rank']} "
          f"hold_vec={conditions[0]['held_out']['rel_vector_error']:.4f} "
          f"hold_pred_norm={d_pred_norm:.2e} hold_target_norm={d_target_norm:.2e}", flush=True)
    for name, design in design_matrices(fit, hold).items():
        fit_result = unregularised_lstsq(design["fit"], fit["exact"], rcond=args.rcond)
        weights = fit_result["weights"]
        pred_fit = design["fit"] @ weights
        pred_hold = design["hold"] @ weights
        calibration = scalar_calibration(pred_fit, fit["exact"], pred_hold)
        row = {
            "condition": name,
            "columns": design["columns"],
            "n_columns": int(design["fit"].shape[1]),
            "projection": design.get("projection"),
            "weight_sha256": weight_sha(weights),
            "lstsq": {k: v for k, v in fit_result.items() if k != "weights"},
            "fit": four_metrics(pred_fit, {k: fit[k] for k in
                                           ("exact", "delta_z", "gains")}),
            "held_out": four_metrics(pred_hold, {k: hold[k] for k in
                                                 ("exact", "delta_z", "gains")}),
            "fit_gate": engineering_fit_gate(pred_fit, fit["exact"]),
            "scalar_calibration": calibration,
        }
        if calibration["a_star"] is not None:
            a = calibration["a_star"]
            row["calibrated"] = {
                "fit": four_metrics(a * pred_fit, {k: fit[k] for k in
                                                   ("exact", "delta_z", "gains")}),
                "held_out": four_metrics(a * pred_hold, {k: hold[k] for k in
                                                         ("exact", "delta_z", "gains")}),
            }
        conditions.append(row)
        print(f"  {name}: cols={row['n_columns']} rank={row['lstsq']['rank']} "
              f"fit_vec={row['fit']['rel_vector_error']:.4f} "
              f"hold_vec={row['held_out']['rel_vector_error']:.4f} "
              f"hold_rho={row['held_out']['spearman_vs_exact_scores']:.4f}", flush=True)

    return {
        "benchmark": "session_6c_information_gap",
        "mode": "real",
        "device": str(device),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "teacher": teacher,
        "amendment": ("prospective Exp-2 amendment (plan §7): directional evidence only; "
                      "no gates pass here; no superiority claims"),
        "fit_episodes": fit_episodes,
        "heldout_episodes": hold_episodes,
        "fit_window_indices": fit_idx,
        "heldout_window_indices": hold_idx,
        "n_fit_windows": int(fit["latent"].shape[0]),
        "n_heldout_windows": int(hold["latent"].shape[0]),
        "seed": args.seed,
        "rcond": args.rcond,
        "conditions": conditions,
        "caveats": [
            "Condition C sees train targets in aggregate: a fit advantage over A/B is "
            "expected mechanically; only the held-out contrast is informative, and only "
            "directionally on this diagnostic split.",
            "Single future trajectory per state: no irreducible-uncertainty claim.",
            "Linear probes only: a nonlinear information advantage would not appear here.",
        ],
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 70)
    print("Session 6C Experiment 2 (bounded linear probes): information gap")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    summary = run(args, device)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "information_gap_summary.json", summary)
    print(f"Saved summary to {output_dir / 'information_gap_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
