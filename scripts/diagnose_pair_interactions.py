#!/usr/bin/env python3
"""scripts/diagnose_pair_interactions.py

Session 6C interaction panel (review R8): does ranking patches individually identify a good
set of k? Runs after the label preflight passes — never gated on distillation success, since
a singleton-to-set mismatch would make all label-fitting work uninformative.

Frozen manifest rules (declared before seeing outcomes):
- All 496 unordered pairs among the 32 patches exist; the panel measures an exact,
  predeclared subset with within-camera, cross-camera and spatial-distance strata.
- Masks RETAIN patches (gains measured from the null/empty selection, as in
  ``exact_marginal_gains``); the derivative reference state is the empty selection and
  effect signs follow it. Mixing references without testing is forbidden by this script's
  own manifest check.
- The empty mask must be a legal model input (finite objective); otherwise the run fails
  loudly instead of silently redefining the anchor. A declared valid anchor S0 with
  G_S0(S) = L(S0) − L(S0∪S) is the only permitted fallback, and it changes the estimand.
- Greedy chains built on realised future losses are hindsight diagnostic references, never
  deployable policies; a learned conditional selector is compared against the same
  fixed/direct baselines with its extra scoring calls charged.

Usage::

    python scripts/diagnose_pair_interactions.py \\
        --cache-dir /content/spatial_cache_e3_1 --drive-root /content/local_runs \\
        --run-id spatial_patches_20261003 --output-dir results/benchmarks/pair_interactions
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

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
    camera_of_patch,
    grid_coordinates,
    objective_at_masks,
)  # noqa: E402

from benchmark_spatial_patch_selection import (  # noqa: E402
    build_spatial_model,
    load_spatial_splits,
    move_to_device,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Session 6C interaction panel")
    parser.add_argument("--cache-dir", type=str, default="/content/spatial_cache_e3_1")
    parser.add_argument("--drive-root", type=str, default="/content/local_runs")
    parser.add_argument("--run-id", type=str, default="spatial_patches_20261003")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks/pair_interactions")
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-seed", type=int, default=7)
    return parser.parse_args(argv)


def pair_strata(patches_per_camera: int = PATCHES_PER_CAMERA,
                cameras: int = CAMERAS) -> Dict[str, List[tuple]]:
    """Every unordered patch pair, labelled by stratum.

    Strata: within-camera pairs, cross-camera pairs; each further split by grid distance
    (adjacent vs distant, split at Manhattan distance 2 within a camera; cross-camera pairs
    form their own distance class since cameras have no shared grid).
    """
    cameras_of = camera_of_patch(patches_per_camera, cameras)
    coords = grid_coordinates(patches_per_camera).tolist()
    strata: Dict[str, List[tuple]] = {"within_adjacent": [], "within_distant": [],
                                      "cross_camera": []}
    for first, second in itertools.combinations(range(patches_per_camera * cameras), 2):
        if int(cameras_of[first]) == int(cameras_of[second]):
            manhattan = (abs(coords[first % patches_per_camera][0]
                             - coords[second % patches_per_camera][0])
                         + abs(coords[first % patches_per_camera][1]
                               - coords[second % patches_per_camera][1]))
            key = "within_adjacent" if manhattan < 2 else "within_distant"
        else:
            key = "cross_camera"
        strata[key].append((first, second))
    return strata


def sample_pairs(n_pairs: int, seed: int,
                 patches_per_camera: int = PATCHES_PER_CAMERA,
                 cameras: int = CAMERAS) -> List[dict]:
    """Exact, predeclared pair manifest: stratified sample with a fixed seed.

    Allocation is proportional to stratum size (largest remainder), so the manifest mirrors
    the pair population instead of oversampling a convenient stratum. The seed and the full
    id list are committed, making "approximately 200" an exact, re-checkable manifest.
    """
    strata = pair_strata(patches_per_camera, cameras)
    total = sum(len(v) for v in strata.values())
    rng = np.random.default_rng(seed)
    manifest: List[dict] = []
    for name in sorted(strata):
        members = sorted(strata[name])
        quota = n_pairs * len(members) / total
        take = min(len(members), int(quota))
        chosen = sorted(rng.choice(len(members), size=take, replace=False).tolist())
        manifest.extend({"pair_id": f"{name}:{members[i][0]:02d}-{members[i][1]:02d}",
                         "stratum": name, "patch_a": members[i][0],
                         "patch_b": members[i][1]} for i in chosen)
    # Largest-remainder top-up to reach exactly n_pairs without breaking strata balance.
    remainder = [(n_pairs * len(strata[name]) / total) % 1 for name in sorted(strata)]
    order = sorted(range(len(remainder)), key=remainder.__getitem__, reverse=True)
    names = sorted(strata)
    taken = {m["pair_id"] for m in manifest}
    extra = 0
    while len(manifest) < n_pairs:
        name = names[order[extra % len(order)]]
        members = sorted(strata[name])
        candidates = [f"{name}:{a:02d}-{b:02d}" for a, b in members if f"{name}:{a:02d}-{b:02d}" not in taken]
        if not candidates:
            extra += 1
            if extra > 2 * len(order):
                break
            continue
        pick = candidates[int(rng.integers(len(candidates)))]
        a, b = pick.split(":")[1].split("-")
        manifest.append({"pair_id": pick, "stratum": name, "patch_a": int(a), "patch_b": int(b)})
        taken.add(pick)
        extra += 1
    manifest = sorted(manifest, key=lambda m: m["pair_id"])[:n_pairs]
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()






def analyse_pair_rows(window_rows: Sequence[dict], manifest: Sequence[dict],
                        site_by_episode: Mapping[str, str], seed: int) -> dict:
    """Per-window interaction + same-budget analysis from committed rows (pure).

    Extracted so the analysis is unit-testable without a GPU: identical definitions
    feed the summary. Imports are local so module import stays light on CPU.
    """
    from adjointrwm.spatial_selection import (  # noqa: E402
        site_clustered_bootstrap_ci,
        wilcoxon_signed_rank,
    )
    from diagnose_spatial_selection_bottleneck import spearman  # noqa: E402

    interactions, resid_frac, raw_rhos = [], [], []
    for row in window_rows:
        base_j = row["base_J"]
        single_gains = np.array([base_j - v for v in row["singleton_J"]])
        pair_gains = np.array([base_j - v for v in row["pair_J"]])
        for r, entry in enumerate(manifest):
            a, b = entry["patch_a"], entry["patch_b"]
            interactions.append({
                "window": row["window"],
                "episode_id": row["episode_id"],
                "pair_id": entry["pair_id"],
                "stratum": entry["stratum"],
                "epsilon": float(pair_gains[r] - single_gains[a] - single_gains[b]),
                "gain_pair": float(pair_gains[r]),
                "gain_a": float(single_gains[a]),
                "gain_b": float(single_gains[b]),
            })
        # Ranking fidelity: how well summed singleton gains order the measured pair
        # gains within this window (Spearman is shift-invariant, so a common shift that
        # preserves order scores highly here by construction).
        additive = np.array([single_gains[manifest[r]["patch_a"]]
                             + single_gains[manifest[r]["patch_b"]]
                             for r in range(len(manifest))])
        with torch.no_grad():
            rho_raw = spearman(torch.as_tensor(additive[None, :]),
                               torch.as_tensor(pair_gains[None, :]))
        raw_rhos.append(float(rho_raw[0]) if torch.isfinite(rho_raw[0]) else float("nan"))
        # Residual interaction variation: after removing the per-window means, what fraction
        # of pair-gain spread is pair-specific (ordering-relevant) rather than a common
        # shift (ordering-irrelevant)? Near 1.0 means the action is all in pair specifics.
        eps_centered = (pair_gains - additive) - np.mean(pair_gains - additive)
        gains_centered = pair_gains - np.mean(pair_gains)
        denom = float(np.std(gains_centered))
        resid_frac.append(float(np.std(eps_centered)) / denom if denom > 0 else float("nan"))

    # Same-budget decision consequence, fully contracted: the JOINT mask of the global top-2
    # singletons (measured J, 2 patches) vs the best MEASURED pair (2 patches). Both are
    # 2-patch sets under identical (unconstrained) selection — unlike the previous version,
    # which summed top-4 singleton gains against a 2-patch joint (a budget mismatch).
    # "Best" is the minimum over the 200 evaluated pairs: a restricted hindsight reference,
    # not the best of all 496 legal pairs. Quantity: gain difference (joint units).
    diffs, sites = [], []
    for row in window_rows:
        base_j = row["base_J"]
        single_gains = np.array([base_j - v for v in row["singleton_J"]])
        pair_gains = np.array([base_j - v for v in row["pair_J"]])
        top2 = np.sort(single_gains)[-2:]
        best_pair_gain = float(pair_gains.max())
        # Joint mask of the top-2 singletons: look up the measured pair if present.
        order = np.argsort(single_gains)[-2:]
        key = f"{min(order):02d}-{max(order):02d}"
        joint_gain = None
        for r, entry in enumerate(manifest):
            if (entry["patch_a"], entry["patch_b"]) == (int(min(order)), int(max(order))):
                joint_gain = float(pair_gains[r])
                break
        if joint_gain is None:
            # The top-2 singleton set was not among the evaluated pairs: fall back to the
            # additive estimate AND flag it, so the comparison never silently mixes measured
            # with estimated joints.
            joint_gain = float(top2.sum())
            estimated = True
        else:
            estimated = False
        diffs.append({"window": row["window"], "episode_id": row["episode_id"],
                      "difference": best_pair_gain - joint_gain,
                      "top_singleton_pair_measured": (not estimated)})
        sites.append(site_by_episode.get(row["episode_id"], "unknown"))
    differences = np.array([d["difference"] for d in diffs])
    wilcoxon = wilcoxon_signed_rank(differences)
    cluster_ci = site_clustered_bootstrap_ci(
        differences, np.asarray(sites), num_resamples=2000, seed=seed)

    def trimmed(values: np.ndarray, fraction: float = 0.1) -> float:
        ordered = np.sort(np.asarray(values, dtype=float))
        cut = int(len(ordered) * fraction)
        core = ordered[cut: len(ordered) - cut] if cut else ordered
        return float(core.mean()) if len(core) else float("nan")

    epsilons = np.array([e["epsilon"] for e in interactions])
    pair_gains_all = np.array([e["gain_pair"] for e in interactions])
    return {
        "interactions": interactions,
        "raw_rhos": raw_rhos,
        "resid_frac": resid_frac,
        "diffs": diffs,
        "sites": sites,
        "differences": differences,
        "wilcoxon": wilcoxon,
        "cluster_ci": cluster_ci,
        "trimmed": trimmed,
    }


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
    _train, _val, test_set, site_by_episode = load_spatial_splits(
        Path(args.cache_dir), PATCHES_PER_CAMERA)
    parent_ids = np.asarray(test_set.episode_ids())
    from adjointrwm.data.windows import stratified_window_indices
    indices = stratified_window_indices(parent_ids, site_by_episode, args.windows, seed=args.seed)
    episodes_seen = sorted({str(e) for e in parent_ids[indices].tolist()})
    loader = DataLoader(Subset(test_set, indices.tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=args.num_workers > 0)

    # Empty-mask legality gate: the null selection must be a legal model input.
    legality: dict = {}
    probe = move_to_device(next(iter(loader)), device)
    with torch.no_grad():
        null_objective = objective_at_masks(
            model, probe, torch.zeros(probe["context_state"].shape[0], 1,
                                      TOTAL_PATCHES, device=device))
    legality["empty_mask_finite"] = bool(torch.isfinite(null_objective).all())
    legality["empty_mask_mean_objective"] = float(null_objective.mean())
    if not legality["empty_mask_finite"]:
        raise SystemExit("empty mask is not a legal model input (non-finite objective): "
                         "declare a valid anchor S0 instead of silently redefining the estimand")
    print(f"empty-mask legality: finite, mean J = {legality['empty_mask_mean_objective']:.4f}",
          flush=True)

    # Masks live on the device: objective_at_masks expands them against device-resident
    # visuals, and a CPU mask faults at the multiply (caught on the first run).
    pair_masks = torch.zeros(len(manifest), TOTAL_PATCHES, device=device)
    for row, entry in enumerate(manifest):
        pair_masks[row, entry["patch_a"]] = 1.0
        pair_masks[row, entry["patch_b"]] = 1.0
    single_masks = torch.eye(TOTAL_PATCHES, device=device)

    # Per-window rows (review: batch means cannot support per-window demeaning, so the
    # raw per-(window, mask) losses are committed, not just aggregates).
    window_rows: List[dict] = []
    window_counter = 0
    started = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            windows = batch["context_state"].shape[0]
            episode_ids = [str(e) for e in batch.get("episode_id", ["unknown"] * windows)]
            base = objective_at_masks(
                model, batch, torch.zeros(windows, 1, TOTAL_PATCHES, device=device))[:, 0]
            singles = objective_at_masks(
                model, batch, single_masks.unsqueeze(0).expand(windows, -1, -1))
            pairs = objective_at_masks(
                model, batch, pair_masks.unsqueeze(0).expand(windows, -1, -1).to(device))
            base_cpu = base.detach().cpu()
            singles_cpu = singles.detach().cpu()
            pairs_cpu = pairs.detach().cpu()
            for i in range(windows):
                window_rows.append({
                    "window": window_counter,
                    "episode_id": episode_ids[i],
                    "base_J": float(base_cpu[i]),
                    "singleton_J": [float(v) for v in singles_cpu[i]],
                    "pair_J": [float(v) for v in pairs_cpu[i]],
                })
                window_counter += 1
            del batch
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Analysis from the committed rows (same definitions the summary reports).
    analysis = analyse_pair_rows(window_rows, manifest, site_by_episode, args.seed)
    interactions = analysis["interactions"]
    raw_rhos = analysis["raw_rhos"]
    resid_frac = analysis["resid_frac"]
    diffs = analysis["diffs"]
    sites = analysis["sites"]
    differences = analysis["differences"]
    wilcoxon = analysis["wilcoxon"]
    cluster_ci = analysis["cluster_ci"]

    def trimmed(values: np.ndarray, fraction: float = 0.1) -> float:
        ordered = np.sort(np.asarray(values, dtype=float))
        cut = int(len(ordered) * fraction)
        core = ordered[cut: len(ordered) - cut] if cut else ordered
        return float(core.mean()) if len(core) else float("nan")

    epsilons = np.array([e["epsilon"] for e in interactions])
    pair_gains_all = np.array([e["gain_pair"] for e in interactions])
    return {
        "benchmark": "session_6c_pair_interactions",
        "mode": "real",
        "device": str(device),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backbone": {"path": str(path), "sha256": sha256_file(path)},
        "pair_manifest": manifest,
        "n_pairs": len(manifest),
        "pair_seed": args.pair_seed,
        "mask_semantics": "masks RETAIN patches; gains measured from the null selection",
        "empty_mask_legality": legality,
        "episodes": episodes_seen,
        "n_windows": int(len(indices)),
        "seed": args.seed,
        "window_rows": window_rows,
        "interactions": {
            "n": int(len(epsilons)),
            "mean_epsilon": float(epsilons.mean()),
            "mean_abs_epsilon": float(np.abs(epsilons).mean()),
            "mean_pair_gain": float(pair_gains_all.mean()),
            "mean_abs_pair_gain": float(np.abs(pair_gains_all).mean()),
            "ranking_fidelity_additive_vs_measured": float(np.nanmean(raw_rhos)),
            "residual_interaction_fraction": float(np.nanmean(resid_frac)),
            "n_windows_ranked": int(np.isfinite(raw_rhos).sum()),
        },
        "same_budget_topk_vs_bestpair": {
            "definition": ("per-window gain difference: best measured pair gain minus joint "
                           "gain of the global top-2 singleton set; both are measured 2-patch "
                           "joints except where flagged estimated (additive fallback when the "
                           "top-2 set was not among the evaluated pairs)"),
            "scope": "best of the 200 evaluated pairs (restricted hindsight, not best of 496)",
            "quantity": "gain difference in J units (positive = best pair wins)",
            "mean": float(differences.mean()),
            "trimmed_mean": trimmed(differences),
            "fraction_estimated_singleton_sets": float(np.mean(
                [not d["top_singleton_pair_measured"] for d in diffs])),
            "wilcoxon": wilcoxon,
            "site_clustered_ci_95": cluster_ci,
            "n": int(len(differences)),
            "per_window": diffs,
        },
        "elapsed_s": round(time.time() - started, 1),
        "caveats": [
            "Hindsight pairs are diagnostic, never deployable policies.",
            "The better pair's predictability from deployment inputs is NOT established here.",
        ],
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 70)
    print("Session 6C interaction panel: singleton vs joint labels")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    summary = run(args, device)
    from adjointrwm.io import atomic_write_json
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "pair_interactions_summary.json", summary)
    print(f"Saved summary to {output_dir / 'pair_interactions_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
