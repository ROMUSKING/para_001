#!/usr/bin/env python3
"""scripts/train_conditional_allocator.py

Session 6C conditional-gain experiment (peer-reviewed spec
``docs/plans/2026-10-05-session-6c-conditional-spec.md`` — implement only what it freezes).

Two MSE heads, one input family, matched capacity, one frozen pipeline:

- singleton-benefit head: PatchRankingCritic-shaped MLP over
  ``[patch_emb; latent; budget; horizon]`` trained by MSE on measured singleton gains
  ``b_p = J(S0) - J(S0 u {p})`` (S0 = null selection, verified legal). Only S=∅ items
  train it.
- conditional-benefit head: ConditionalBenefitHead over the same inputs plus the
  selected set (mean selected embedding + relative size), trained by MSE on measured
  conditional gains ``b_p(S) = J(S) - J(S u {p})`` over unselected patches only.

Set generation (frozen 50/50, capped): per training batch, half the items use fixed
outcome-independent masks (seeded schedule over total sizes {0, 2, 4}), half use
student-generated greedy sets from the current singleton head (sizes cycle {0, 2, 4}).
Any batch exceeding the 50% student cap is rejected before the optimiser step; realised
fractions are reported. Futures never enter deployed inputs — labels only. Label
rollouts and sequential scoring calls are counted in a cost ledger.

Protected evaluation (read-once): ``make-protected`` writes a frozen manifest of VAL-split
windows (episode-disjoint from every diagnostic set used so far) plus a seal file with its
SHA-256. ``evaluate`` refuses a manifest whose seal mismatches and appends every protected
read to a run log with timestamp and purpose. No diagnostic-window information may
influence model selection — including thresholds, budgets, stopping rules, or cost
trade-offs.

Subcommands::

    python scripts/train_conditional_allocator.py make-protected --cache-dir ... --output-dir ...
    python scripts/train_conditional_allocator.py train --cache-dir ... --drive-root ... --run-id ...
    python scripts/train_conditional_allocator.py evaluate --cache-dir ... --protected-manifest ... --checkpoint-dir ...
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
    ConditionalBenefitHead,
    PatchRankingCritic,
    camera_of_patch,
    early_feature_norm_scores,
    greedy_oracle_masks,
    matched_patch_critic_hidden,
    objective_at_masks,
    select_topk_per_camera,
    site_clustered_bootstrap_ci,
    trimmed_mean,
    uniform_grid_scores,
    wilcoxon_signed_rank,
)  # noqa: E402

from benchmark_spatial_patch_selection import (  # noqa: E402
    build_spatial_model,
    exact_marginal_gains,
    load_spatial_splits,
    move_to_device,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def theta_gate(single: np.ndarray, cond: np.ndarray, sites: np.ndarray,
               seed: int = 0, num_resamples: int = 2000) -> dict:
    """Frozen θ rule (conditional spec §4) as a pure function, so it can be tested.

    θ_w = 1 − R_cond(w)/R_single(w) per window (paired — never a ratio of
    aggregates). Kept windows satisfy ``single > 1e-12``; every other window
    (near-zero denominator or negative reference regret — the exact top-k
    singleton ordering is hindsight, not a guaranteed joint optimum) is excluded
    from θ and reported under the absolute-margin fallback instead (no silent
    epsilon, no ratio on a non-positive denominator). Superiority needs the
    one-sided 95% lower bound ≥ 0.08; non-inferiority needs it above −0.03. The
    one-sided bound is the 5th percentile of the site-clustered bootstrap
    distribution of the mean (whole sites with replacement, same resampling as
    :func:`site_clustered_bootstrap_ci`).
    """
    single = np.asarray(single, dtype=float)
    cond = np.asarray(cond, dtype=float)
    sites = np.asarray(sites)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_all = np.where(single > 1e-12, 1.0 - cond / single, np.nan)
    finite_mask = np.isfinite(theta_all)
    theta_windows = theta_all[finite_mask]
    theta_sites = sites[finite_mask]
    if len(theta_windows):
        theta_ci = site_clustered_bootstrap_ci(
            theta_windows, theta_sites, num_resamples=num_resamples, seed=seed)
        unique, inverse = np.unique(theta_sites, return_inverse=True)
        sums = np.bincount(inverse, weights=theta_windows, minlength=unique.size)
        counts = np.bincount(inverse, minlength=unique.size)
        rng = np.random.default_rng(seed + 1)
        draws = rng.integers(0, unique.size, size=(num_resamples, unique.size))
        means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        onesided_low = float(np.quantile(means, 0.05))
    else:
        theta_ci = {"estimate": float("nan"), "ci_low": float("nan"),
                    "ci_high": float("nan"), "num_sites": 0, "num_windows": 0}
        onesided_low = float("nan")
    return {
        "definition": "per-window 1 - R_cond/R_single over windows with R_single > 1e-12; "
                      "all other windows use the absolute-margin fallback "
                      "(absolute trimmed-mean delta on the excluded subset, "
                      "reported alongside — never a ratio on a non-positive denominator)",
        "n": int(len(theta_windows)),
        "n_excluded_nonpositive_denom": int((~finite_mask).sum()),
        "two_sided_ci_95": theta_ci,
        "one_sided_95_lower": onesided_low,
        "superiority_08": bool(onesided_low >= 0.08),
        "non_inferiority_03": bool(onesided_low > -0.03),
        "bootstrap": {"num_resamples": int(num_resamples), "seed": int(seed)},
    }


def build_training_schedule(n_steps: int, batch_size: int, seed: int) -> List[dict]:
    """Frozen per-step set-generation schedule: 50% fixed masks, 50% student sets.

    Every mask respects per-camera quotas by construction (codex review 2026-10-05:
    global totals neither satisfied quotas nor covered the evaluation range).
    Fixed items name per-camera counts ~ Uniform{0..8} (outcome-independent; spans
    every deployment prefix size at each evaluation budget k_cam ∈ {2,4,8}).
    Student items name a prefix length ~ Uniform{0..16} applied at runtime to the
    current singleton head's per-camera greedy top-8 ordered by score, i.e. the
    contexts a greedy sequential deployment policy actually encounters (with
    singleton rather than conditional scores — declared, not hidden). The schedule
    carries no data — only the pattern — so it cannot leak outcomes. Returned list
    length is n_steps.
    """
    rng = np.random.default_rng(seed)
    schedule = []
    for step in range(n_steps):
        kinds = (["fixed"] * (batch_size // 2)
                 + ["student"] * (batch_size - batch_size // 2))
        rng.shuffle(kinds)
        fixed_counts = rng.integers(0, 9, size=(batch_size, CAMERAS)).tolist()
        prefix_lens = rng.integers(0, 2 * 8 + 1, size=batch_size).tolist()
        schedule.append({"step": step, "kinds": kinds, "fixed_counts": fixed_counts,
                         "prefix_lens": prefix_lens})
    return schedule


def random_quota_mask(windows: int, per_cam_counts, generator: torch.Generator,
                      device: torch.device) -> torch.Tensor:
    """Outcome-independent random mask with per-camera quotas.

    ``per_cam_counts`` is ``[windows, CAMERAS]`` (or ``[CAMERAS]`` broadcast over
    windows); each camera keeps exactly its count, sampled uniformly. Every
    returned mask is quota-legal by construction.
    """
    cameras = camera_of_patch(PATCHES_PER_CAMERA, CAMERAS)
    counts = torch.as_tensor(per_cam_counts, dtype=torch.long)
    if counts.dim() == 1:
        counts = counts.unsqueeze(0).expand(windows, -1)
    # Generators stay on CPU (CUDA generators need explicit construction); tensors
    # move afterwards, so the random stream is device-independent.
    key = torch.rand(windows, TOTAL_PATCHES, generator=generator)
    mask = torch.zeros(windows, TOTAL_PATCHES)
    for cam in range(CAMERAS):
        cols = (cameras == cam).nonzero(as_tuple=True)[0].tolist()
        order = key[:, cols].argsort(dim=1)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(len(cols)).expand_as(order))
        keep = (rank < counts[:, cam].unsqueeze(1)).float()
        mask[:, cols] = keep
    return mask.to(device).float()


def random_subset_mask(windows: int, total: int, generator: torch.Generator,
                       device: torch.device) -> torch.Tensor:
    """Uniform random subset of exactly ``total`` patches (outcome-independent)."""
    # Generators stay on CPU (CUDA generators need explicit construction); tensors move
    # after drawing, so the random stream is device-independent.
    order = torch.argsort(torch.rand(windows, TOTAL_PATCHES, generator=generator), dim=1
                          ).to(device)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(TOTAL_PATCHES, device=device).expand_as(order))
    return (rank < total).float()


def make_protected(args) -> int:
    manifest_path = Path(args.cache_dir) / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("episodes", [])
    val_recs = [r for r in records if r.get("split") == "val"]
    if not val_recs:
        raise SystemExit("no val-split episodes: protected set has nowhere disjoint to live")
    # Diagnostic episodes consumed so far (test split, committed artefacts).
    used = set()
    for artefact in ("results/benchmarks/spatial_bottleneck/spatial_bottleneck_summary.json",):
        artefact_path = REPO_DIR / artefact
        if artefact_path.exists():
            used.update(json.loads(artefact_path.read_text()).get("episodes", []))
    val_ids = [r["episode_id"] for r in val_recs]
    overlap = set(val_ids) & used
    if overlap:
        raise SystemExit(f"protected pool overlaps diagnostic episodes: {sorted(overlap)[:3]}")
    rng = np.random.default_rng(args.seed)
    order = list(val_ids)
    rng.shuffle(order)
    from adjointrwm.data.windows import WindowSpec, window_starts
    spec = WindowSpec(context_len=8, horizon=4, stride=2)
    length_of = {r["episode_id"]: int(r.get("length", 0)) for r in val_recs}
    picked = []
    per_episode = max(1, args.protected_windows // max(1, len(order)))
    for episode in order:
        for start in window_starts(length_of[episode], spec)[:per_episode]:
            if len(picked) >= args.protected_windows:
                break
            picked.append({"episode_id": episode, "window_start": start})
        if len(picked) >= args.protected_windows:
            break
    doc = {
        "protected_manifest_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "split": "val",
        "disjoint_from_diagnostic_episodes": True,
        "n_windows": len(picked),
        "windows": picked,
        "read_log": [],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = output_dir / "protected_manifest.json"
    manifest_file.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    seal = sha256_file(manifest_file)
    (output_dir / "protected_manifest.seal").write_text(seal + "\n")
    print(f"protected manifest: {len(picked)} windows; seal {seal[:16]}...")
    return 0


def _read_protected(manifest_path: Path, purpose: str, acknowledge_reread: str = "") -> dict:
    """Read-once enforced: seal must match, and every read is appended to the run log.

    A repeat read is refused unless ``acknowledge_reread`` names the reason, which is
    then recorded as a deviation entry (codex review 2026-10-05: logging reads is not
    enforcement). Re-running a frozen-head evaluation with fixed analysis code is a
    legitimate acknowledgeable reason; tuning on protected outcomes is not.
    """
    doc = json.loads(manifest_path.read_text())
    seal_path = manifest_path.parent / "protected_manifest.seal"
    if not seal_path.exists():
        raise SystemExit("protected manifest has no seal file: refusing to read")
    if sha256_file(manifest_path) != seal_path.read_text().strip():
        raise SystemExit("protected manifest seal MISMATCH: the manifest changed after freezing; "
                         "record a deviation before proceeding")
    if doc.get("read_log") and not acknowledge_reread:
        raise SystemExit(
            f"protected manifest already read {len(doc['read_log'])} time(s) "
            f"(last: {doc['read_log'][-1]}): refusing a silent re-read. Pass "
            "--acknowledge-reread with the reason to record it as a deviation.")
    entry = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "purpose": purpose}
    if acknowledge_reread:
        entry["acknowledged_reread_reason"] = acknowledge_reread
    doc.setdefault("read_log", []).append(entry)
    manifest_path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (manifest_path.parent / "protected_manifest.seal").write_text(
        sha256_file(manifest_path) + "\n")
    return doc


def load_backbone_for_train(args, device: torch.device):
    path = Path(args.drive_root) / "runs" / args.run_id / "best_spatial.pt"
    if not path.exists():
        raise SystemExit(f"missing teacher checkpoint at {path}: hard failure")
    model = build_spatial_model(args, PATCHES_PER_CAMERA, device, args.width, 384)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model.eval()
    return model, {"path": str(path), "sha256": sha256_file(path)}


def train_heads_command(args) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model, teacher = load_backbone_for_train(args, device)
    train_set, _val, _test, _sites = load_spatial_splits(Path(args.cache_dir),
                                                         PATCHES_PER_CAMERA)
    from adjointrwm.data.windows import stratified_window_indices
    parent_ids = np.asarray(train_set.episode_ids())
    site_by_episode = {r["episode_id"]: r.get("site", "unknown")
                       for r in json.loads((Path(args.cache_dir) / "cache_manifest.json")
                                           .read_text()).get("episodes", [])}
    indices = stratified_window_indices(parent_ids, site_by_episode, args.train_windows,
                                        seed=args.seed)
    loader = DataLoader(Subset(train_set, indices.tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=args.num_workers > 0)
    batches = [move_to_device(b, device) for b in loader]
    token_dim = batches[0]["context_visual"].shape[-1]
    d_model = model.d_model
    singleton = PatchRankingCritic(
        token_dim, d_model, matched_patch_critic_hidden(token_dim + d_model + 2, d_model)
    ).to(device)
    conditional = ConditionalBenefitHead(token_dim, d_model).to(device)
    opt_single = torch.optim.AdamW(singleton.parameters(), lr=args.lr)
    opt_cond = torch.optim.AdamW(conditional.parameters(), lr=args.lr)
    schedule = build_training_schedule(args.steps, args.batch_size, args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    stats = {"student_fraction": [], "label_rollouts": 0, "scoring_calls": 0}
    step_cursor = 0
    for entry in schedule:
        batch = batches[step_cursor % len(batches)]
        step_cursor += 1
        windows = batch["context_state"].shape[0]
        latent = model.encode_context(batch["context_visual"], batch["context_state"],
                                      batch["context_action"]).detach()
        with torch.no_grad():
            greedy_scores = singleton(
                batch["context_visual"], latent,
                torch.full((windows,), 1.0 / 16, device=device),
                torch.ones(windows, device=device))
        # Student contexts: per-camera greedy top-8 from the CURRENT singleton head
        # (quota-legal), truncated at runtime to the schedule's prefix length — the
        # contexts a greedy sequential deployment policy encounters (declared).
        cameras = camera_of_patch(PATCHES_PER_CAMERA, CAMERAS).to(device)
        student_top8 = select_topk_per_camera(greedy_scores, 8, cameras, CAMERAS)
        student_fraction = sum(1 for k in entry["kinds"] if k == "student") / len(entry["kinds"])
        stats["student_fraction"].append(student_fraction)
        if student_fraction > 0.5 + 1e-9:
            raise SystemExit(f"student cap violated at step {entry['step']}: "
                             f"fraction {student_fraction}")
        single_losses, cond_losses = [], []
        for position, kind in enumerate(entry["kinds"]):
            row = position % windows
            single = {k: (v[row:row + 1] if torch.is_tensor(v) else v)
                      for k, v in batch.items()}
            if kind == "fixed":
                base = random_quota_mask(
                    1, [entry["fixed_counts"][position]], generator, device)[0]
            else:
                ranked = student_top8[row].nonzero(as_tuple=True)[0][
                    greedy_scores[row][student_top8[row] > 0.5].argsort(descending=True)]
                prefix = ranked[:entry["prefix_lens"][position] % (2 * 8 + 1)]
                base = torch.zeros(TOTAL_PATCHES, device=device)
                base[prefix] = 1.0
            unselected = base < 0.5
            if int(unselected.sum()) == 0:
                continue
            with torch.no_grad():
                j_s = objective_at_masks(model, single, base.unsqueeze(0).unsqueeze(0))[:, 0]
                grown = base.unsqueeze(0).expand(TOTAL_PATCHES, -1).clone()
                grown[torch.arange(TOTAL_PATCHES, device=device),
                      torch.arange(TOTAL_PATCHES, device=device)] = 1.0
                # [1, P]: one J per grown mask; no candidate indexing ([:, 0] here would
                # collapse all 32 grown masks to a scalar — caught on the first GPU run).
                j_grown = objective_at_masks(model, single, grown.unsqueeze(0))
                stats["label_rollouts"] += 1 + TOTAL_PATCHES
            cond_gains = (j_s - j_grown).squeeze(0)
            stats["scoring_calls"] += 1
            latent_row = latent[row:row + 1]
            visual_row = batch["context_visual"][row:row + 1]
            bf = torch.full((1,), 1.0 / 16, device=device)
            hf = torch.ones_like(bf)
            keep = unselected.unsqueeze(0)
            if int(base.sum()) == 0:
                pred_single = singleton(visual_row, latent_row, bf, hf)
                single_losses.append(torch.nn.functional.mse_loss(
                    pred_single[keep].squeeze(0), cond_gains[unselected].detach()))
            pred_cond = conditional(visual_row, latent_row, base.unsqueeze(0), bf, hf)
            cond_losses.append(torch.nn.functional.mse_loss(
                pred_cond[keep].squeeze(0), cond_gains[unselected].detach()))
        opt_single.zero_grad(set_to_none=True)
        opt_cond.zero_grad(set_to_none=True)
        if single_losses:
            torch.stack(single_losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(singleton.parameters(), 1.0)
            opt_single.step()
        if cond_losses:
            torch.stack(cond_losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(conditional.parameters(), 1.0)
            opt_cond.step()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"singleton": singleton.state_dict(), "conditional": conditional.state_dict()},
               output_dir / "benefit_heads.pt")
    summary = {
        "teacher": teacher,
        "steps": args.steps,
        "student_fraction_mean": float(np.mean(stats["student_fraction"])),
        "student_fraction_max": float(np.max(stats["student_fraction"])),
        "label_rollouts": stats["label_rollouts"],
        "scoring_calls": stats["scoring_calls"],
        "parameter_counts": {
            "singleton": sum(p.numel() for p in singleton.parameters()),
            "conditional": sum(p.numel() for p in conditional.parameters()),
        },
        "checkpoint": str(output_dir / "benefit_heads.pt"),
        "checkpoint_sha256": sha256_file(output_dir / "benefit_heads.pt"),
    }
    atomic_write_json(output_dir / "benefit_heads_summary.json", summary)
    print(f"trained both heads ({args.steps} steps); student fraction mean "
          f"{summary['student_fraction_mean']:.3f} max {summary['student_fraction_max']:.3f}; "
          f"label rollouts {summary['label_rollouts']}")
    return 0


def evaluate_command(args) -> int:
    from adjointrwm.spatial_selection import early_feature_norm_scores, uniform_grid_scores
    from benchmark_spatial_patch_selection import exact_marginal_gains

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    manifest_path = Path(args.protected_manifest)
    manifest = _read_protected(manifest_path, "protected evaluation",
                               args.acknowledge_reread or "")
    checkpoint_dir = Path(args.checkpoint_dir)
    heads = torch.load(checkpoint_dir / "benefit_heads.pt", map_location="cpu",
                       weights_only=False)
    backbone_path = Path(args.drive_root) / "runs" / args.run_id / "best_spatial.pt"
    if not backbone_path.exists():
        raise SystemExit(f"missing backbone at {backbone_path}: hard failure")
    expected_teacher = (json.loads((checkpoint_dir / "benefit_heads_summary.json").read_text())
                        .get("teacher", {}).get("sha256"))
    if expected_teacher and sha256_file(backbone_path) != expected_teacher:
        raise SystemExit("backbone checkpoint changed since training: re-train or record "
                         "a deviation before evaluating")
    model_args = argparse.Namespace(width=args.width, patches_per_camera=PATCHES_PER_CAMERA)
    model = build_spatial_model(model_args, PATCHES_PER_CAMERA, device, args.width, 384)
    backbone_state = torch.load(backbone_path, map_location="cpu", weights_only=False)
    model.load_state_dict(backbone_state.get("model_state_dict", backbone_state), strict=False)
    model.eval()
    token_dim = 384
    d_model = model.d_model
    singleton = PatchRankingCritic(
        token_dim, d_model, matched_patch_critic_hidden(token_dim + d_model + 2, d_model))
    singleton.load_state_dict(heads["singleton"])
    singleton = singleton.to(device).eval()
    conditional = ConditionalBenefitHead(token_dim, d_model)
    conditional.load_state_dict(heads["conditional"])
    conditional = conditional.to(device).eval()

    # NOTE: protected windows live on the VAL split; rebuild the identical dataset here.
    manifest_by_episode = manifest["windows"]
    _t, val_set, _te, _si = load_spatial_splits(Path(args.cache_dir), PATCHES_PER_CAMERA)
    position_of = {}
    for position in range(len(val_set)):
        record_index, start = val_set.index[position]
        key = (val_set.records[record_index]["episode_id"], start)
        position_of.setdefault(key, position)
    missing = [w for w in manifest_by_episode
               if (w["episode_id"], w["window_start"]) not in position_of]
    if missing:
        raise SystemExit(f"{len(missing)} protected windows not found in the val dataset "
                         f"(e.g. {missing[0]}): the cache changed under a frozen manifest")
    order = [position_of[(w["episode_id"], w["window_start"])] for w in manifest_by_episode]
    loader = DataLoader(Subset(val_set, order), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=args.num_workers > 0)
    budgets = (2, 4, 8)
    # Device-resident once: every gather/index below pairs it with CUDA tensors, and a CPU
    # index faults the gather (caught on the first GPU run).
    cameras = camera_of_patch(PATCHES_PER_CAMERA, CAMERAS).to(device)
    eval_site_by_episode = {r["episode_id"]: r.get("site", "unknown")
                            for r in json.loads(
                                (Path(args.cache_dir) / "cache_manifest.json").read_text()
                            ).get("episodes", [])}
    # Δ_k per budget: L(singleton set) − L(conditional set), plus norm/uniform/exact refs.
    per_budget: Dict[str, Dict[str, list]] = {
        f"k_cam={k}": {"delta": [], "vs_norm": [], "vs_uniform": [],
                       "single_regret": [], "cond_regret": [],
                       "sites": []} for k in budgets}
    scoring_calls = {"singleton": 0, "conditional": 0}
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            windows = batch["context_state"].shape[0]
            latent = model.encode_context(batch["context_visual"], batch["context_state"],
                                          batch["context_action"])
            budget = torch.full((windows,), 1.0 / 16, device=device)
            horizon = torch.ones_like(budget)
            norm_scores = early_feature_norm_scores(batch["context_visual"])
            uniform_scores = uniform_grid_scores(windows, PATCHES_PER_CAMERA, CAMERAS,
                                                 device=device)
            single_scores = singleton(batch["context_visual"], latent, budget, horizon)
            exact = exact_marginal_gains(model, batch)
            scoring_calls["singleton"] += 1
            scoring_calls["oracle_reference_rollouts"] = scoring_calls.get(
                "oracle_reference_rollouts", 0) + 1
            for k_cam in budgets:
                total = k_cam * CAMERAS
                masks, names = [], []
                for name, gains in (("norm", norm_scores), ("uniform", uniform_scores),
                                    ("single", single_scores), ("exact", exact)):
                    masks.append(select_topk_per_camera(gains, k_cam, cameras, CAMERAS))
                    names.append(name)
                # NOTE: "exact" top-k is the hindsight ordering reference on protected
                # windows (measured jointly below like every other arm).
                # Conditional: greedy sequential selection with per-camera quotas.
                cond_selected = torch.zeros(windows, TOTAL_PATCHES, device=device)
                counts = torch.zeros(windows, CAMERAS, device=device, dtype=torch.long)
                for _step in range(total):
                    allowed = (counts.gather(
                        1, cameras.unsqueeze(0).expand(windows, -1)) < k_cam)
                    allowed = allowed & (cond_selected < 0.5)
                    scores = conditional(batch["context_visual"], latent, cond_selected,
                                         budget, horizon)
                    scoring_calls["conditional"] += 1
                    scores = torch.where(allowed, scores,
                                         torch.full_like(scores, float("-inf")))
                    pick = scores.argmax(dim=1)
                    cond_selected[torch.arange(windows, device=device), pick] = 1.0
                    counts.scatter_add_(
                        1, cameras[pick].unsqueeze(1),
                        torch.ones(windows, 1, dtype=torch.long, device=device))
                masks.append(cond_selected)
                names.append("conditional")
                stacked = torch.stack(masks, dim=1)
                values = objective_at_masks(model, batch, stacked)
                by_name = dict(zip(names, values.unbind(dim=1)))
                key = f"k_cam={k_cam}"
                per_budget[key]["sites"].extend(
                    [eval_site_by_episode.get(str(e), "unknown")
                     for e in batch["episode_id"]])
                per_budget[key]["delta"].extend(
                    (by_name["single"] - by_name["conditional"]).cpu().tolist())
                per_budget[key]["vs_norm"].extend(
                    (by_name["norm"] - by_name["conditional"]).cpu().tolist())
                per_budget[key]["vs_uniform"].extend(
                    (by_name["uniform"] - by_name["conditional"]).cpu().tolist())
                per_budget[key]["single_regret"].extend(
                    (by_name["single"] - by_name["exact"]).cpu().tolist())
                per_budget[key]["cond_regret"].extend(
                    (by_name["conditional"] - by_name["exact"]).cpu().tolist())
            del batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
    report: Dict[str, dict] = {}
    for key, series in per_budget.items():
        delta = np.asarray(series["delta"])
        single = np.asarray(series["single_regret"])
        cond = np.asarray(series["cond_regret"])
        sites_arr = np.asarray(series["sites"])
        gate = theta_gate(single, cond, sites_arr, seed=args.seed)
        # Absolute-margin fallback on the θ-excluded subset (codex review 2026-10-05):
        # the gate must not ride on windows where no ratio exists. Reported, never
        # folded into θ.
        excluded = single <= 1e-12
        fallback_delta = delta[excluded]
        # Upper-tail guardrail: conditional must not buy a better mean with a worse
        # bad tail. Report the loss-side quantiles of per-window Δ (positive Δ favours
        # conditional) and the harmed fraction.
        report[key] = {
            "theta": gate,
            "theta_fallback_absolute_on_excluded": {
                "n_excluded": int(excluded.sum()),
                "trimmed_mean": trimmed_mean(fallback_delta) if excluded.any() else 0.0,
                "mean": float(fallback_delta.mean()) if excluded.any() else 0.0,
            },
            "tail_guardrail": {
                "p10_delta": float(np.quantile(delta, 0.10)),
                "p05_delta": float(np.quantile(delta, 0.05)),
                "fraction_harmed": float(np.mean(delta < 0.0)),
            },
            "delta_conditional_minus_singleton": {
                "mean": float(delta.mean()), "trimmed_mean": trimmed_mean(delta),
                "n": int(delta.size), "wilcoxon": wilcoxon_signed_rank(delta),
                "site_clustered_ci_95": site_clustered_bootstrap_ci(
                    delta, sites_arr, num_resamples=2000, seed=args.seed),
                "per_window_delta": [float(v) for v in delta],
                "per_window_single_regret": [float(v) for v in single],
                "per_window_cond_regret": [float(v) for v in cond],
                "sites": [str(s) for s in sites_arr],
            },
            "vs_norm_mean": float(np.mean(series["vs_norm"])),
            "vs_uniform_mean": float(np.mean(series["vs_uniform"])),
            "single_regret_trimmed": trimmed_mean(np.asarray(series["single_regret"])),
            "cond_regret_trimmed": trimmed_mean(np.asarray(series["cond_regret"])),
        }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "benchmark": "session_6c_conditional_eval",
        "mode": "real",
        "device": str(device),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protected_manifest": str(manifest_path),
        "manifest_windows": len(manifest["windows"]),
        "checkpoint_dir": str(checkpoint_dir),
        "scoring_calls": scoring_calls,
        "bootstrap": {"num_resamples": 2000, "seed": int(args.seed)},
        "primary_budget": "k_cam=4 (diagnostic re-run: all budgets reported without "
                          "multiplicity control; confirmatory run gates on k_cam=4 only)",
        "cost_gate": {
            "rule": "superiority requires no-higher total selection-path cost; this "
                    "re-run REPORTS cost only — conditional greedy selection charges one "
                    "head forward per pick and cannot pass the strict gate by construction. "
                    "No quality-cost trade-off is claimed.",
            "conditional_scoring_calls": scoring_calls.get("conditional", 0),
            "singleton_scoring_calls": scoring_calls.get("singleton", 0),
            "passes_strict_no_higher_cost": (
                scoring_calls.get("conditional", 0) <= scoring_calls.get("singleton", 0)),
        },
        "per_budget": report,
        "caveats": [
            "Conditional greedy selection charges one head forward per pick "
            f"({sum(scoring_calls.values())} total here); cost gate applies.",
            "Exact-gain top-k is a hindsight reference, never a deployable policy.",
        ],
    }
    atomic_write_json(output_dir / "conditional_eval_summary.json", summary)
    for key, entry in report.items():
        print(f"  {key}: Δ(conditional−singleton) trimmed "
              f"{entry['delta_conditional_minus_singleton']['trimmed_mean']:.6f} "
              f"p={entry['delta_conditional_minus_singleton']['wilcoxon']['p_value']:.4g}",
              flush=True)
    print(f"Saved summary to {output_dir / 'conditional_eval_summary.json'}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Session 6C conditional allocator")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cache-dir", type=str, default="/content/spatial_cache_e3_1")
    common.add_argument("--seed", type=int, default=0)

    make_p = sub.add_parser("make-protected", parents=[common],
                            help="freeze the protected eval manifest + seal")
    make_p.add_argument("--output-dir", type=str,
                        default="results/benchmarks/conditional_protected")
    make_p.add_argument("--protected-windows", type=int, default=192)

    train_p = sub.add_parser("train", parents=[common], help="train both benefit heads")
    train_p.add_argument("--drive-root", type=str, default="/content/local_runs")
    train_p.add_argument("--run-id", type=str, default="spatial_patches_20261003")
    train_p.add_argument("--output-dir", type=str, default="results/benchmarks/conditional_heads")
    train_p.add_argument("--steps", type=int, default=300)
    train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--num-workers", type=int, default=4)
    train_p.add_argument("--width", type=int, default=512)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--train-windows", type=int, default=512)

    eval_p = sub.add_parser("evaluate", parents=[common], help="protected evaluation (read-once)")
    eval_p.add_argument("--protected-manifest", type=str, required=True)
    eval_p.add_argument("--checkpoint-dir", type=str, required=True)
    eval_p.add_argument("--drive-root", type=str, default="/content/local_runs")
    eval_p.add_argument("--run-id", type=str, default="spatial_patches_20261003")
    eval_p.add_argument("--width", type=int, default=512)
    eval_p.add_argument("--output-dir", type=str, default="results/benchmarks/conditional_eval")
    eval_p.add_argument("--acknowledge-reread", type=str, default="",
                        help="required when the manifest was already read: records the reason "
                             "as a deviation instead of refusing")
    eval_p.add_argument("--batch-size", type=int, default=8)
    eval_p.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args(argv)
    if args.command == "make-protected":
        return make_protected(args)
    if args.command == "train":
        return train_heads_command(args)
    if args.command == "evaluate":
        return evaluate_command(args)
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
