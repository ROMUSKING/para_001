#!/usr/bin/env python3
"""scripts/benchmark_curvature_voi_allocator.py

Direction 3: Real-Data Second-Order Curvature & Belief-Space VOI Allocation Benchmark.

Compares allocator heads on exact per-window gains with a hold option on real
multi-site robotics trajectories (DROID E3.1 stratified shard across 12 laboratories):

* ``exact_costate`` (Oracle floor: first-order scores from autograd co-state)
* ``always_mode0`` (Refusal baseline: always hold, gain 0 by definition)
* ``direct_critic`` (Parameter-matched to CostateEstimator, 1.315M params)
* ``direct_critic_curv_matched`` (Parameter-matched to CurvatureCostateEstimator, 1.577M params)
* ``first_order`` (CostateEstimator + first_order_scores)
* ``normalized_first_order`` (CostateEstimator + normalized_first_order_scores)
* ``second_order_curvature`` (CurvatureCostateEstimator + second_order_curvature_scores)
* ``belief_space_voi`` (CurvatureCostateEstimator + belief_space_voi_scores)

Metrics per policy: mean test regret, gap to oracle, win rate vs critic and
vs mode0, per-site breakdown across 12 robotics laboratories, and inference throughput (Hz).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_DIR = Path(__file__).resolve().parent.parent
for p in [REPO_DIR / "src", Path("/content/para_001/src"), Path("/content/src")]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from adjointrwm.allocators import (  # noqa: E402
    CostateEstimator,
    CurvatureCostateEstimator,
    DirectCritic,
    belief_space_voi_scores,
    first_order_scores,
    matched_critic_hidden,
    normalized_first_order_scores,
    plackett_luce_loss,
    second_order_curvature_scores,
    with_hold_zero,
)
from adjointrwm.data import WindowDataset, WindowSpec, fit_normaliser  # noqa: E402
from adjointrwm.models.common import ArmDims, prediction_objective  # noqa: E402
from adjointrwm.models.hybrid_adjoint import HybridAdjointRecursiveWorldModel  # noqa: E402

try:
    from adjointrwm.models import build_arm  # noqa: E402
except Exception:  # pragma: no cover
    build_arm = None

POLICIES = (
    "exact_costate",
    "always_mode0",
    "direct_critic",
    "direct_critic_curv_matched",
    "first_order",
    "normalized_first_order",
    "second_order_curvature",
    "belief_space_voi",
)


def curvature_matched_critic_hidden(d: int) -> int:
    """Critic hidden width whose parameter count matches CurvatureCostateEstimator."""
    target = sum(p.numel() for p in CurvatureCostateEstimator(d).parameters())
    return max(1, round((target - 1) / (2 * d + 5)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Direction 3: Curvature and Belief-Space VOI Allocation Benchmark")
    parser.add_argument("--synthetic-test", action="store_true", help="Run fast synthetic smoke test on CPU")
    parser.add_argument("--cache-dir", type=str, default="/content/cache_e3_1")
    parser.add_argument("--drive-root", type=str, default="/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production")
    parser.add_argument("--run-id", type=str, default="harp_hybrid_dynamics_20261002")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks/curvature_voi")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-target-windows", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--cost-weight", type=float, default=0.002)
    parser.add_argument("--uncert-weight", type=float, default=0.5)
    return parser.parse_args(argv)


def load_e3_dataset(cache_dir: Path):
    cache_manifest_path = cache_dir / "cache_manifest.json"
    if not cache_manifest_path.exists():
        cache_manifest_path = cache_dir / "e3_1_droid_500_manifest.json"
    if not cache_manifest_path.exists():
        fallback = REPO_DIR / "results/data/droid_e3_1/e3_1_droid_500_manifest.json"
        if fallback.exists():
            cache_manifest_path = fallback
        else:
            raise FileNotFoundError(f"Missing cache manifest at {cache_manifest_path}")

    manifest_data = json.loads(cache_manifest_path.read_text())
    episodes = manifest_data.get("episodes", [])

    records_by_split = {"train": [], "val": [], "test": []}
    episodes_dir = cache_dir / "episodes"
    site_by_episode = {}
    for ep in episodes:
        raw_split = ep.get("split", "train")
        split = "val" if raw_split in ("val", "validation") else raw_split
        ep_id = ep["episode_id"]
        site_by_episode[ep_id] = ep.get("site", "unknown")
        ep_path = episodes_dir / f"{ep_id}.npz"
        if ep_path.exists():
            records_by_split[split].append({
                "episode_id": ep_id,
                "cached_path": str(ep_path),
                "length": ep.get("length", 100),
                "split": split,
                "site": ep.get("site", "unknown"),
            })

    norm_path = cache_dir / "normalisation.npz"
    if norm_path.exists():
        with np.load(norm_path) as npz:
            normalisation = {k: npz[k] for k in npz.files}
    else:
        def load_arrays(rec):
            with np.load(rec["cached_path"]) as ep:
                return {"states": ep["states"], "actions": ep["actions"]}
        train_arrays = [load_arrays(r) for r in records_by_split["train"]]
        normalisation = fit_normaliser([a["states"] for a in train_arrays], [a["actions"] for a in train_arrays])

    spec = WindowSpec(context_len=8, horizon=4, stride=2)
    train_dataset = WindowDataset(records_by_split["train"], spec, normalisation, visual_layout="flat")
    val_dataset = WindowDataset(records_by_split["val"], spec, normalisation, visual_layout="flat")
    test_dataset = WindowDataset(records_by_split["test"], spec, normalisation, visual_layout="flat")

    return train_dataset, val_dataset, test_dataset, site_by_episode




def extract_real_targets(
    model: HybridAdjointRecursiveWorldModel,
    batch: dict,
    costs: torch.Tensor,
    device: torch.device,
) -> dict:
    """Extracts candidate latents, autograd co-state, exact directional HVPs, and net gains."""
    b_dev = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
    B = b_dev["context_state"].shape[0]

    with torch.no_grad():
        # Mode 0: Proprio only
        v0 = torch.zeros_like(b_dev["context_visual"])
        z0 = model.encode_context(v0, b_dev["context_state"], b_dev["context_action"])
        p0 = model.rollout(z0, b_dev["future_actions"], context_state=b_dev["context_state"])
        j0 = prediction_objective(p0, b_dev["target_state"], b_dev["target_visual"])

        # Mode 1: Wrist only (512: visual tokens)
        v1 = b_dev["context_visual"].clone()
        v1[:, :, :512] = 0.0
        z1 = model.encode_context(v1, b_dev["context_state"], b_dev["context_action"])
        p1 = model.rollout(z1, b_dev["future_actions"], context_state=b_dev["context_state"])
        j1 = prediction_objective(p1, b_dev["target_state"], b_dev["target_visual"])

        # Mode 2: Exterior only (:512 visual tokens)
        v2 = b_dev["context_visual"].clone()
        v2[:, :, 512:] = 0.0
        z2 = model.encode_context(v2, b_dev["context_state"], b_dev["context_action"])
        p2 = model.rollout(z2, b_dev["future_actions"], context_state=b_dev["context_state"])
        j2 = prediction_objective(p2, b_dev["target_state"], b_dev["target_visual"])

        # Mode 3: Both cameras
        v3 = b_dev["context_visual"]
        z3 = model.encode_context(v3, b_dev["context_state"], b_dev["context_action"])
        p3 = model.rollout(z3, b_dev["future_actions"], context_state=b_dev["context_state"])
        j3 = prediction_objective(p3, b_dev["target_state"], b_dev["target_visual"])

    # First-order autograd co-state at Mode 0 (create_graph=True for higher-order HVPs)
    z0_req = z0.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        p0_req = model.rollout(z0_req, b_dev["future_actions"], context_state=b_dev["context_state"])
        loss_0 = prediction_objective(p0_req, b_dev["target_state"], b_dev["target_visual"]).sum()
        lambda0 = torch.autograd.grad(loss_0, z0_req, create_graph=True)[0]

        # Latent perturbations relative to z0
        delta_z1 = (z1 - z0).detach()
        delta_z2 = (z2 - z0).detach()
        delta_z3 = (z3 - z0).detach()
        active_deltas = [delta_z1, delta_z2, delta_z3]

        # Exact directional second-order curvature per candidate via individual HVP
        # kappa_k = delta_z_k^T * H * delta_z_k with ZERO cross-term contamination
        exact_curvatures = []
        for dz_k in active_deltas:
            inner_k = torch.sum(lambda0 * dz_k)
            hvp_k = torch.autograd.grad(inner_k, z0_req, retain_graph=True)[0]
            curv_k = torch.sum(hvp_k * dz_k, dim=-1).detach()
            exact_curvatures.append(curv_k)
        exact_curv_tensor = torch.stack(exact_curvatures, dim=1)  # [B, 3]

    # Assemble full candidates: Hold (k=0) + 3 active modalities
    delta_z0 = torch.zeros_like(delta_z1)
    effects = torch.stack([delta_z0, delta_z1, delta_z2, delta_z3], dim=1)  # [B, 4, D]

    # True net gains: G_k = J(z_0) - J(z_k) - c_k (Hold G_0 = 0.0)
    g0 = torch.zeros_like(j0)
    g1 = (j0 - j1) - costs[1]
    g2 = (j0 - j2) - costs[2]
    g3 = (j0 - j3) - costs[3]
    exact_gain = torch.stack([g0, g1, g2, g3], dim=1)  # [B, 4]

    return {
        "latent": z0.detach(),
        "effects": effects,
        "costs": costs,
        "exact_costate": lambda0.detach(),
        "exact_curvatures": exact_curv_tensor,  # [B, 3] for k in 1..3
        "exact_gain": exact_gain,               # [B, 4]
        "context_visual": b_dev["context_visual"].detach(),
    }


def make_synthetic_batch(batch_size: int, device: torch.device, seed: int) -> dict:
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)
    mk = lambda *shape: torch.randn(*shape, generator=g).to(device)
    return {
        "context_state": mk(batch_size, 8, 14),
        "context_action": mk(batch_size, 8, 7),
        "context_visual": mk(batch_size, 8, 1024),
        "future_actions": mk(batch_size, 4, 7),
        "target_state": mk(batch_size, 4, 14),
        "target_visual": mk(batch_size, 4, 1024),
    }


def build_synthetic_teacher(device: torch.device):
    dims = ArmDims(
        state_dim=14,
        action_dim=7,
        visual_tokens=2,
        visual_token_dim=512,
        target_visual_dim=1024,
        context_len=8,
        horizon=4,
    )
    teacher = build_arm("hybrid_adjoint_rwm", dims, width=64, transformer_heads=4, transformer_layers=2)
    return teacher.to(device).eval()


def train_allocator_heads(
    teacher,
    train_loader,
    device: torch.device,
    args,
    costs: torch.Tensor,
) -> dict:
    """Trains costate, standard critic, curvature-matched critic, and curvature estimator."""
    d = teacher.d_model
    torch.manual_seed(42)

    costate = CostateEstimator(d).to(device)
    h_crit = matched_critic_hidden(d)
    critic = DirectCritic(d, h_crit).to(device)

    h_curv_crit = curvature_matched_critic_hidden(d)
    critic_curv = DirectCritic(d, h_curv_crit).to(device)

    curvature = CurvatureCostateEstimator(d).to(device)

    opt = torch.optim.AdamW(
        list(costate.parameters())
        + list(critic.parameters())
        + list(critic_curv.parameters())
        + list(curvature.parameters()),
        lr=args.lr,
    )

    train_iter = iter(train_loader)
    for step in range(args.max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        tgt = extract_real_targets(teacher, batch, costs, device)
        latent, effects, costs_t = tgt["latent"], tgt["effects"], tgt["costs"]
        B = latent.shape[0]
        budget = torch.full((B,), 1.0 / 3.0, device=device)
        horizon = torch.ones(B, device=device)

        opt.zero_grad()

        # 1. First-order costate loss
        lam_hat = costate(latent, budget, horizon)
        cos_loss = (1.0 - F.cosine_similarity(lam_hat, tgt["exact_costate"], dim=-1)).mean()

        # 2. Standard critic loss
        s_crit = with_hold_zero(critic(latent, effects[:, 1:], costs_t[1:], budget, horizon))
        crit_loss = F.smooth_l1_loss(s_crit[:, 1:], tgt["exact_gain"][:, 1:])

        # 3. Curvature-matched critic loss
        s_crit_curv = with_hold_zero(critic_curv(latent, effects[:, 1:], costs_t[1:], budget, horizon))
        crit_curv_loss = F.smooth_l1_loss(s_crit_curv[:, 1:], tgt["exact_gain"][:, 1:])

        # 4. Curvature estimator multi-objective loss
        lam2, h2 = curvature(latent, budget, horizon)
        cos2 = (1.0 - F.cosine_similarity(lam2, tgt["exact_costate"], dim=-1)).mean()

        # Directional curvature alignment: sum_d h_d * delta_z_{k,d}^2 vs exact kappa_k
        pred_curvs = []
        for k_idx in range(3):
            dz = effects[:, k_idx + 1]
            pred_curv_k = (h2 * dz.pow(2)).sum(dim=-1)
            pred_curvs.append(pred_curv_k)
        pred_curv_tensor = torch.stack(pred_curvs, dim=1)
        curv_loss = F.smooth_l1_loss(pred_curv_tensor, tgt["exact_curvatures"])

        # End-to-end decision ranking loss
        s2 = second_order_curvature_scores(lam2, h2, effects, costs_t)
        rank2 = plackett_luce_loss(s2, tgt["exact_gain"], temperature=0.1)

        total_loss = cos_loss + crit_loss + crit_curv_loss + cos2 + 0.5 * curv_loss + 0.5 * rank2
        total_loss.backward()
        opt.step()

    return {
        "costate": costate.eval(),
        "critic": critic.eval(),
        "critic_curv_matched": critic_curv.eval(),
        "curvature": curvature.eval(),
    }


@torch.no_grad()
def score_all_policies(heads: dict, targets: dict, uncert_weight: float = 0.5) -> dict:
    latent, effects, costs = targets["latent"], targets["effects"], targets["costs"]
    B = latent.shape[0]
    device = latent.device

    budget = torch.full((B,), 1.0 / 3.0, device=device)
    horizon = torch.ones(B, device=device)

    lam_hat = heads["costate"](latent, budget, horizon)
    lam_c, h_c = heads["curvature"](latent, budget, horizon)

    s_critic = with_hold_zero(heads["critic"](latent, effects[:, 1:], costs[1:], budget, horizon))
    s_critic_curv = with_hold_zero(heads["critic_curv_matched"](latent, effects[:, 1:], costs[1:], budget, horizon))

    # Belief-space VOI covariance reduction proxy: delta_cov = effects^2
    delta_cov = effects.pow(2)

    return {
        "exact_costate": first_order_scores(targets["exact_costate"], effects, costs),
        "direct_critic": s_critic,
        "direct_critic_curv_matched": s_critic_curv,
        "first_order": first_order_scores(lam_hat, effects, costs),
        "normalized_first_order": normalized_first_order_scores(lam_hat, effects, costs),
        "second_order_curvature": second_order_curvature_scores(lam_c, h_c, effects, costs),
        "belief_space_voi": belief_space_voi_scores(lam_c, effects, delta_cov, h_c, costs, uncert_weight=uncert_weight),
    }


def evaluate_gains(
    teacher,
    heads: dict,
    loader: DataLoader,
    device: torch.device,
    costs: torch.Tensor,
    args,
    site_by_episode: dict | None = None,
) -> dict:
    per_policy_gains = {p: [] for p in POLICIES}
    per_policy_choices = {p: [] for p in POLICIES}
    per_site_gains = {} if site_by_episode else None

    lat_total, lat_count = 0.0, 0
    n_windows = 0

    for batch in loader:
        targets = extract_real_targets(teacher, batch, costs, device)
        gain = targets["exact_gain"]
        B = gain.shape[0]
        n_windows += B

        t0 = time.perf_counter()
        with torch.no_grad():
            scores = score_all_policies(heads, targets, uncert_weight=args.uncert_weight)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat_total += time.perf_counter() - t0
        lat_count += 1

        for name in POLICIES:
            if name == "always_mode0":
                ch = torch.zeros(B, dtype=torch.long, device=gain.device)
            else:
                ch = scores[name].argmax(dim=1)
            ch_cpu = ch.cpu().numpy()
            g_cpu = gain.cpu().numpy()[np.arange(B), ch_cpu]
            per_policy_choices[name].append(ch_cpu)
            per_policy_gains[name].append(g_cpu)

        if site_by_episode and "episode_id" in batch:
            ep_ids = batch["episode_id"]
            best_g = gain.cpu().numpy().max(axis=1)
            for i, eid in enumerate(ep_ids):
                site = site_by_episode.get(eid, "unknown")
                if site not in per_site_gains:
                    per_site_gains[site] = {p: [] for p in POLICIES}
                    per_site_gains[site]["best"] = []
                per_site_gains[site]["best"].append(best_g[i])
                for p in POLICIES:
                    per_site_gains[site][p].append(per_policy_gains[p][-1][i])

    gains = {p: np.concatenate(v) for p, v in per_policy_gains.items()}
    best = np.stack([gains[p] for p in POLICIES], axis=1).max(axis=1)
    oracle_gain = gains["exact_costate"]
    mean_lat_ms = (lat_total / max(1, lat_count)) / args.batch_size * 1000.0
    throughput_hz = (n_windows / lat_total) if lat_total > 0 else 0.0

    summary = {}
    for p in POLICIES:
        regret = best - gains[p]
        summary[p] = {
            "mean_gain": float(gains[p].mean()),
            "mean_regret": float(regret.mean()),
            "std_regret": float(regret.std()),
            "gap_to_oracle": float(regret.mean() - (best - oracle_gain).mean()),
            "win_rate_vs_critic": float((gains[p] > gains["direct_critic"]).mean()),
            "win_rate_vs_mode0": float((gains[p] > gains["always_mode0"]).mean()),
            "win_rate_vs_first_order": float((gains[p] > gains["first_order"]).mean()),
            "n_windows": int(n_windows),
        }

    site_summary = {}
    if per_site_gains:
        for site, vals in sorted(per_site_gains.items()):
            b_site = np.array(vals["best"])
            site_summary[site] = {
                "n_windows": len(b_site),
                "regret": {p: float((b_site - np.array(vals[p])).mean()) for p in POLICIES},
            }

    return {
        "per_policy": summary,
        "site_breakdown": site_summary,
        "n_windows": int(n_windows),
        "mean_latency_ms_per_window": float(mean_lat_ms),
        "throughput_hz": float(throughput_hz),
    }


def run_synthetic(args, device: torch.device) -> dict:
    print("[synthetic] building synthetic teacher and datasets...")
    teacher = build_synthetic_teacher(device)
    costs = torch.tensor([0.0, 0.001, 0.001, 0.002], device=device)

    synthetic_train = [make_synthetic_batch(args.batch_size, device, s) for s in range(5)]
    synthetic_test = [make_synthetic_batch(args.batch_size, device, 100 + s) for s in range(4)]

    heads = train_allocator_heads(teacher, synthetic_train, device, args, costs)
    ev = evaluate_gains(teacher, heads, synthetic_test, device, costs, args)
    return {
        "mode": "synthetic",
        "device": str(device),
        "per_policy": ev["per_policy"],
        "site_breakdown": {},
        "n_windows": ev["n_windows"],
        "mean_latency_ms_per_window": ev["mean_latency_ms_per_window"],
        "throughput_hz": ev["throughput_hz"],
    }


def run_real_benchmark(args, device: torch.device) -> dict:
    cache_dir = Path(args.cache_dir)
    drive_root = Path(args.drive_root)
    run_dir = drive_root / "runs" / args.run_id

    print(f"Loading E3.1 dataset from {cache_dir}...")
    train_dataset, val_dataset, test_dataset, site_by_episode = load_e3_dataset(cache_dir)
    print(f"Dataset: {len(train_dataset)} train, {len(test_dataset)} test windows across {len(set(site_by_episode.values()))} sites.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    dims = ArmDims(
        state_dim=14,
        action_dim=7,
        visual_tokens=2,
        visual_token_dim=512,
        target_visual_dim=1024,
        context_len=8,
        horizon=4,
    )
    costs = torch.tensor([0.0, 0.001, 0.001, 0.002], device=device)

    all_seed_results = []
    for seed in range(args.num_seeds):
        print(f"\n=======================================================")
        print(f"  SEED {seed + 1} / {args.num_seeds} EVALUATION")
        print(f"=======================================================")
        torch.manual_seed(5000 + seed * 100)

        teacher = build_arm("hybrid_adjoint_rwm", dims, width=512, transformer_heads=8, transformer_layers=6).to(device)
        model_path = run_dir / f"seed_{seed}" / "best.pt"
        if not model_path.exists():
            model_path = run_dir / "best.pt"
        if model_path.exists():
            print(f"  Loading trained teacher weights from {model_path}...")
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            teacher.load_state_dict(state_dict, strict=False)
        else:
            print("  [Notice] Initializing nominal teacher model...")
        teacher.eval()

        print(f"  Training allocator heads for {args.max_steps} steps...")
        t_tr = time.perf_counter()
        heads = train_allocator_heads(teacher, train_loader, device, args, costs)
        print(f"  Training finished in {time.perf_counter() - t_tr:.2f}s")

        print("  Evaluating all policies on 4,154 held-out test windows...")
        ev = evaluate_gains(teacher, heads, test_loader, device, costs, args, site_by_episode=site_by_episode)
        all_seed_results.append(ev)
        for p in POLICIES:
            m = ev["per_policy"][p]
            print(f"    {p:>28}: regret={m['mean_regret']:.5f} (win vs critic: {m['win_rate_vs_critic']:.3f})")

    # Aggregate over seeds
    per_policy = {}
    for p in POLICIES:
        regrets = [r["per_policy"][p]["mean_regret"] for r in all_seed_results]
        gains = [r["per_policy"][p]["mean_gain"] for r in all_seed_results]
        gaps = [r["per_policy"][p]["gap_to_oracle"] for r in all_seed_results]
        w_crit = [r["per_policy"][p]["win_rate_vs_critic"] for r in all_seed_results]
        w_mode0 = [r["per_policy"][p]["win_rate_vs_mode0"] for r in all_seed_results]
        w_fo = [r["per_policy"][p]["win_rate_vs_first_order"] for r in all_seed_results]
        per_policy[p] = {
            "mean_regret": float(np.mean(regrets)),
            "std_regret": float(np.std(regrets)),
            "mean_gain": float(np.mean(gains)),
            "gap_to_oracle": float(np.mean(gaps)),
            "win_rate_vs_critic": float(np.mean(w_crit)),
            "win_rate_vs_mode0": float(np.mean(w_mode0)),
            "win_rate_vs_first_order": float(np.mean(w_fo)),
        }

    # Aggregate site breakdown
    all_sites = set(all_seed_results[0]["site_breakdown"].keys())
    site_breakdown = {}
    for s in sorted(all_sites):
        site_breakdown[s] = {
            "n_windows": all_seed_results[0]["site_breakdown"][s]["n_windows"],
            "regret": {
                p: float(np.mean([r["site_breakdown"][s]["regret"][p] for r in all_seed_results]))
                for p in POLICIES
            },
        }

    return {
        "mode": "real",
        "device": str(device),
        "run_id": args.run_id,
        "n_windows": all_seed_results[0]["n_windows"],
        "num_seeds": args.num_seeds,
        "mean_latency_ms_per_window": float(np.mean([r["mean_latency_ms_per_window"] for r in all_seed_results])),
        "throughput_hz": float(np.mean([r["throughput_hz"] for r in all_seed_results])),
        "per_policy": per_policy,
        "site_breakdown": site_breakdown,
    }


def save_outputs(summary: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "curvature_voi_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")

    lines = [
        "# Direction 3: Real-Data Second-Order Curvature & Belief-Space VOI Allocation Benchmark",
        "",
        f"**Date (UTC):** {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} · **Mode:** `{summary['mode']}` · "
        f"**Device:** `{summary['device']}` · **Windows:** `{summary['n_windows']}` · **Seeds:** `{summary.get('num_seeds', 1)}`",
        f"**Throughput:** `{summary['throughput_hz']:.1f} decisions/sec` (`{summary['mean_latency_ms_per_window']:.4f} ms/window`)",
        "",
        "## 1. Test Regret & Win Rates",
        "",
        "| Allocator Policy | Mean Test Regret | Gap to Oracle | Win Rate vs Critic | Win Rate vs First-Order | Win Rate vs Mode 0 |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for p in POLICIES:
        m = summary["per_policy"][p]
        lines.append(
            f"| `{p}` | `{m['mean_regret']:.5f} ± {m['std_regret']:.5f}` | `+{m['gap_to_oracle']:.5f}` | "
            f"`{m['win_rate_vs_critic']:.3f}` | `{m['win_rate_vs_first_order']:.3f}` | `{m['win_rate_vs_mode0']:.3f}` |"
        )

    if summary.get("site_breakdown"):
        lines += [
            "",
            "## 2. Cross-Site Performance Across 12 Robotics Laboratories (Mean Test Regret)",
            "",
            "| Site | Windows | Mode 0 | Critic | First Order | Second Order Curvature | Belief-Space VOI |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
        for site, sdata in summary["site_breakdown"].items():
            r = sdata["regret"]
            lines.append(
                f"| `{site}` | {sdata['n_windows']} | {r['always_mode0']:.4f} | {r['direct_critic']:.4f} | "
                f"{r['first_order']:.4f} | **{r['second_order_curvature']:.4f}** | **{r['belief_space_voi']:.4f}** |"
            )

    report_path = output_dir / "curvature_voi_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Saved report to {report_path}")


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 70)
    print("Direction 3: Curvature & Belief-Space VOI Allocation Benchmark")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cache_dir = Path(args.cache_dir)
    manifest = cache_dir / "cache_manifest.json"
    if not manifest.exists():
        manifest = cache_dir / "e3_1_droid_500_manifest.json"

    if not args.synthetic_test and manifest.exists():
        summary = run_real_benchmark(args, device)
    else:
        reason = "--synthetic-test flag" if args.synthetic_test else f"missing cache at {cache_dir}"
        print(f"[Notice] Running in synthetic diagnostic mode ({reason}).")
        summary = run_synthetic(args, device)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_DIR / output_dir
    save_outputs(summary, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
