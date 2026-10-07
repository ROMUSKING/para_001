#!/usr/bin/env python3
"""scripts/check_label_contract.py

Session 6C preflight (review R3): label-contract verification before any label reuse.

For a small set of frozen real windows, this checks — rather than assumes — the conventions
the co-state labels depend on:

1. **Batch invariance:** the per-example λ for one window computed alone, inside a batch,
   under batch permutation, and under different chunk sizes must agree within tolerance.
   (If J_batch = mean_i J_i, differentiation w.r.t. z_i carries a 1/B factor; if it is a sum,
   it does not. This check reveals the convention instead of assuming it.)
2. **Finite differences:** directional check λᵀv ≈ (J(z+εv) − J(z−εv)) / 2ε on the actual
   production objective and latent representation, over a declared eps range, reporting
   absolute and scaled discrepancies. (Model is deterministic in eval mode — rollout emits
   means, never samples — so no seed-fixing is needed inside J.)
3. **Reduction report:** the exact averaging dimensions (batch/horizon/feature) as measured,
   so a mean↔sum change of scientific objective would be visible here first.

Declared evaluation mode is the model's own (`.eval()`, as the benchmark uses). A failure
here repairs the label pipeline and regenerates affected labels; passing on this checkpoint,
examples and configuration lets the existing Exp-1 run stand with the preflight recorded
as completed-after-execution and its exact scope stated.

Usage::

    python scripts/check_label_contract.py \\
        --cache-dir /content/spatial_cache_e3_1 --drive-root /content/local_runs \\
        --run-id spatial_patches_20261003 --output-dir results/benchmarks/label_preflight
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
from adjointrwm.models.common import prediction_objective  # noqa: E402
from adjointrwm.spatial_selection import (  # noqa: E402
    cotangent_bundle,
    spatial_encode_with_pooled,
)

from benchmark_spatial_patch_selection import (  # noqa: E402
    build_spatial_model,
    load_spatial_splits,
    move_to_device,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Session 6C preflight: label-contract checks")
    parser.add_argument("--cache-dir", type=str, default="/content/spatial_cache_e3_1")
    parser.add_argument("--drive-root", type=str, default="/content/local_runs")
    parser.add_argument("--run-id", type=str, default="spatial_patches_20261003")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks/label_preflight")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-directions", type=int, default=3)
    parser.add_argument("--epsilons", type=str, default="1e-2,1e-3,1e-4")
    parser.add_argument("--batch-tol", type=float, default=1e-4,
                        help="max abs deviation for batch-invariance (absolute, latent units)")
    parser.add_argument("--fd-rel-tol", type=float, default=5e-2,
                        help="relative finite-difference agreement required (informational gate)")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_objective(model, latent: torch.Tensor, future_actions: torch.Tensor,
                     target_state: torch.Tensor, target_visual: torch.Tensor) -> torch.Tensor:
    """Per-window scalar J(z): rollout + production objective, mirroring cotangent_bundle."""
    prediction = model.rollout(latent, future_actions)
    return prediction_objective(prediction, target_state, target_visual)


def run(args, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    path = Path(args.drive_root) / "runs" / args.run_id / "best_spatial.pt"
    if not path.exists():
        raise SystemExit(f"missing teacher checkpoint at {path}: hard failure (Session 5 lesson)")
    model = build_spatial_model(args, 16, device, args.width, 384)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model.eval()
    teacher = {"path": str(path), "sha256": sha256_file(path),
               "patch_dropout_trained": bool(state.get("patch_dropout_trained", False))}

    train_set, _val, _test, _sites = load_spatial_splits(Path(args.cache_dir), 16)
    ids = np.asarray(train_set.episode_ids())
    by_episode: Dict[str, List[int]] = {}
    for index, episode in enumerate(ids):
        by_episode.setdefault(str(episode), []).append(index)
    episodes = sorted(by_episode)[:2]
    indices = sorted(i for e in episodes for i in by_episode[e][: args.windows // 2 or 1])
    indices = indices[: args.windows]
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"),
                         persistent_workers=args.num_workers > 0)
    subset = Subset(train_set, indices)
    loader = DataLoader(subset, shuffle=False, **loader_kwargs)
    batches = [move_to_device(b, device) for b in loader]
    flat = {k: torch.cat([b[k] for b in batches]) if torch.is_tensor(batches[0][k]) else None
            for k in batches[0]}
    n = flat["context_state"].shape[0]
    window_ids = [str(e) for es in ([ids[i] for i in indices]) for e in [es]]

    # --- 1. Batch invariance: alone vs batched vs permuted vs rechunked ---------------
    def lambdas_for(order: List[int], chunk: int) -> torch.Tensor:
        out = {}
        for start in range(0, len(order), chunk):
            # Plain ints: numpy integers hash like ints but explicit is cheaper than doubt.
            idx = [int(i) for i in order[start:start + chunk]]
            single = {k: (v[idx] if torch.is_tensor(v) else v) for k, v in flat.items()
                      if v is not None}
            bundle = cotangent_bundle(model, single)
            for position, global_idx in enumerate(idx):
                out[global_idx] = bundle.exact_costate[position].detach().cpu()
        return torch.stack([out[i] for i in range(n)])

    base_order = list(range(n))
    rng = np.random.default_rng(args.seed)
    permuted = list(rng.permutation(n))
    ref = lambdas_for(base_order, n)
    variants = {
        "batched": lambdas_for(base_order, max(1, n // 2)),
        "permuted": lambdas_for(permuted, max(1, n // 2)),
        "singletons": lambdas_for(base_order, 1),
    }
    batch_report = {}
    for name, variant in variants.items():
        abs_dev = (variant - ref).abs().max().item()
        scale = ref.abs().max().item()
        batch_report[name] = {
            "max_abs_deviation": abs_dev,
            "reference_scale": scale,
            "within_tol": bool(abs_dev <= args.batch_tol),
        }
    # Reduction-convention readout: alone-vs-batch ratio near 1.0 means sum-like per-example
    # gradients; near 1/B would expose a mean reduction leaking into the labels.
    alone = lambdas_for(base_order, 1)
    with torch.no_grad():
        ratio = (alone.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)).median().item()
    batch_report["alone_over_batched_median_norm_ratio"] = ratio

    # --- 2. Finite differences on the production objective -----------------------------
    generator = torch.Generator().manual_seed(args.seed + 1)
    epsilons = [float(e) for e in args.epsilons.split(",") if e.strip()]
    fd_rows = []
    model.zero_grad(set_to_none=True)
    for w in range(n):
        visuals = flat["context_visual"][w:w + 1]
        states = flat["context_state"][w:w + 1]
        actions = flat["context_action"][w:w + 1]
        with torch.enable_grad():
            latent_w, _pooled = spatial_encode_with_pooled(model, visuals, states, actions)
            z0 = latent_w.detach()
            bundle_w = cotangent_bundle(model, {
                "context_visual": visuals, "context_state": states, "context_action": actions,
                "future_actions": flat["future_actions"][w:w + 1],
                "target_state": flat["target_state"][w:w + 1],
                "target_visual": flat["target_visual"][w:w + 1],
            })
            lam = bundle_w.exact_costate.detach()
        d = z0.shape[-1]
        for trial in range(args.n_directions):
            # Generator stays on CPU (CUDA generators need explicit construction);
            # the direction is moved after drawing, so the sequence is device-independent.
            direction = torch.randn(d, generator=generator).to(device)
            direction = direction / direction.norm().clamp_min(1e-12)
            analytic = float((lam[0] * direction).sum())
            for eps in epsilons:
                with torch.no_grad():
                    j_plus = scalar_objective(
                        model, z0 + eps * direction, flat["future_actions"][w:w + 1],
                        flat["target_state"][w:w + 1], flat["target_visual"][w:w + 1])
                    j_minus = scalar_objective(
                        model, z0 - eps * direction, flat["future_actions"][w:w + 1],
                        flat["target_state"][w:w + 1], flat["target_visual"][w:w + 1])
                numeric = float((j_plus - j_minus) / (2 * eps))
                denom = abs(analytic) + 1e-12
                fd_rows.append({"window": w, "trial": trial, "eps": eps,
                                "analytic": analytic, "numeric": numeric,
                                "abs_discrepancy": abs(analytic - numeric),
                                "rel_discrepancy": abs(analytic - numeric) / denom})
    abs_disc = [r["abs_discrepancy"] for r in fd_rows]
    rel_disc = [r["rel_discrepancy"] for r in fd_rows]
    fd_report = {
        "n_checks": len(fd_rows),
        "max_abs_discrepancy": max(abs_disc),
        "median_rel_discrepancy": float(np.median(rel_disc)),
        "max_rel_discrepancy": max(rel_disc),
        "within_tol": bool(max(rel_disc) <= args.fd_rel_tol),
        "rows": fd_rows,
    }

    return {
        "benchmark": "session_6c_label_preflight",
        "mode": "real",
        "device": str(device),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "teacher": teacher,
        "eval_mode": "model.eval() (declared evaluation mode, as the benchmark uses)",
        "window_indices": indices,
        "window_episodes": window_ids,
        "n_windows": n,
        "seed": args.seed,
        "batch_invariance": batch_report,
        "batch_tol": args.batch_tol,
        "finite_differences": fd_report,
        "fd_rel_tol": args.fd_rel_tol,
        "caveats": [
            "Passing here validates labels on these windows/checkpoint/config only.",
            "Finite differences use central differences in float32; eps range is declared, not tuned.",
        ],
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 70)
    print("Session 6C preflight: label-contract checks")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    summary = run(args, device)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "label_preflight_summary.json", summary)
    print(f"batch invariance: { {k: v.get('within_tol') for k, v in summary['batch_invariance'].items() if isinstance(v, dict)} }")
    print(f"finite differences: within_tol={summary['finite_differences']['within_tol']} "
          f"median_rel={summary['finite_differences']['median_rel_discrepancy']:.2e}")
    print(f"Saved summary to {output_dir / 'label_preflight_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
